import asyncio
import json
import time
import signal
from typing import Dict, Any, List
import redis.asyncio as redis
from .config import settings
from .storage.postgres import postgres
from .storage.qdrant import qdrant_storage
from .minio_client import minio_storage
from .embedding import embedding_generator
from .monitoring import logger, metrics
from .agents.core import LanguageIdentifier, SlangDecoder
from .agents.analysis import (
    TopicClassifier,
    EntityExtractor,
    SentimentAnalyzer,
    EmotionDetector,
)
from .agents.high_level import BiasDetector, Summarizer, QualityScorer

# Shutdown flag
shutdown_event = asyncio.Event()


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Shutdown signal received, initiating graceful shutdown...")
    shutdown_event.set()


class ProcessingPipeline:
    """
    Orchestrates the document processing flow through various AI agents.
    Implements 2026 best practices for modularity and observability.
    """

    def __init__(self):
        logger.info("Initializing NiruSense agents...")
        self.lang_id = LanguageIdentifier()
        self.slang_decoder = SlangDecoder()
        self.topic_classifier = TopicClassifier()
        self.entity_extractor = EntityExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.emotion_detector = EmotionDetector()
        self.bias_detector = BiasDetector()
        self.summarizer = Summarizer()
        self.quality_scorer = QualityScorer()

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document"""
        start_time = time.time()
        source = data.get("source", "unknown")
        doc_id = None

        logger.info(
            f"Processing document from {source}",
            extra={"source": source, "url": data.get("url", "N/A")},
        )

        try:
            # 1. Pre-processing
            text = (
                data.get("text", "")
                or data.get("raw_content", "")
                or data.get("summary", "")
            )

            if not text or len(text) < 10:
                raise ValueError("Document text too short or empty")

            # Truncate if too long (optimization)
            if len(text) > settings.MAX_TEXT_LENGTH:
                text = text[: settings.MAX_TEXT_LENGTH]

            # Pre-processing steps
            lang_res = self.lang_id.execute(text)
            slang_res = self.slang_decoder.execute(text)
            normalized_text = slang_res.get("normalized_text", text)

            # 2. Parallel Core Analysis
            # Run independent analysis tasks concurrently for speed
            results = await asyncio.gather(
                asyncio.to_thread(self.topic_classifier.execute, normalized_text),
                asyncio.to_thread(self.entity_extractor.execute, normalized_text),
                asyncio.to_thread(self.sentiment_analyzer.execute, normalized_text),
                asyncio.to_thread(self.emotion_detector.execute, normalized_text),
                return_exceptions=True,
            )

            # Unpack results, handling exceptions
            topic_res = results[0] if not isinstance(results[0], Exception) else {}
            entity_res = results[1] if not isinstance(results[1], Exception) else {}
            sentiment_res = results[2] if not isinstance(results[2], Exception) else {}
            emotion_res = results[3] if not isinstance(results[3], Exception) else {}

            # Log any agent failures
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    logger.error(f"Core analysis agent {i} failed: {res}")

            # 3. High-Level Analysis (run in thread pool to avoid blocking event loop)
            bias_res, summary_res, quality_res = await asyncio.gather(
                asyncio.to_thread(self.bias_detector.execute, normalized_text),
                asyncio.to_thread(self.summarizer.execute, normalized_text),
                asyncio.to_thread(self.quality_scorer.execute, normalized_text),
                return_exceptions=True,
            )
            if isinstance(bias_res, Exception):
                bias_res = {}
                logger.error(f"Bias detector failed: {bias_res}")
            if isinstance(summary_res, Exception):
                summary_res = {}
                logger.error(f"Summarizer failed: {summary_res}")
            if isinstance(quality_res, Exception):
                quality_res = {}
                logger.error(f"Quality scorer failed: {quality_res}")

            # 4. Generate Embedding
            embed_text = (
                f"{normalized_text}\n\nSummary: {summary_res.get('summary', '')}"
            )
            vector = embedding_generator.embed(embed_text)

            # 5. Storage
            # Save to PostgreSQL
            doc_id = await postgres.save_document(
                {
                    "url": data.get("url", f"generated-{int(time.time())}"),
                    "raw_content": text,
                    "normalized_content": normalized_text,
                    "source": source,
                    "source_domain": data.get("source_domain", source),
                    "published_at": data.get("published_at", data.get("timestamp")),
                }
            )

            # Save analysis results
            analysis_results = {
                "language": lang_res,
                "slang": slang_res,
                "topics": topic_res,
                "entities": entity_res,
                "sentiment": sentiment_res,
                "emotion": emotion_res,
                "bias": bias_res,
                "summary": summary_res,
                "quality": quality_res,
            }

            for agent_name, res in analysis_results.items():
                if isinstance(res, dict) and not res.get("error"):
                    await postgres.save_analysis(doc_id, agent_name, res)

            # Save to Qdrant
            payload = {
                "document_id": str(doc_id),
                "text": normalized_text[:1000],
                "summary": summary_res.get("summary", "")[:500],
                "topics": topic_res.get("topics", []),
                "sentiment": sentiment_res.get("sentiment", "unknown"),
                "entities": [e.get("text", "") for e in entity_res.get("entities", [])][
                    :10
                ],
                "quality_score": quality_res.get("quality_score", 0),
                "language": lang_res.get("lang", "unknown"),
                "source": source,
                "timestamp": data.get("timestamp", int(time.time())),
            }

            qdrant_storage.upsert(str(doc_id), vector, payload)

            processing_time = time.time() - start_time
            metrics.record_document_processed(processing_time)

            logger.info(
                f"Document {doc_id} processed successfully",
                extra={"doc_id": str(doc_id), "processing_time": processing_time},
            )

            return {
                "status": "success",
                "doc_id": str(doc_id),
                "processing_time": processing_time,
            }

        except Exception as e:
            metrics.record_document_failed()
            logger.error(f"Document processing failed: {e}")
            return {"status": "error", "error": str(e)}


pipeline = ProcessingPipeline()


async def _process_single_message(
    redis_client, message_id: str, message_data: dict, s3_key: str
):
    """Process a single message from the Redis stream"""
    try:
        # Fetch from MinIO using robust client
        file_data = minio_storage.get_object(s3_key)

        # Parse JSON payload
        full_payload = json.loads(file_data)
        full_payload.update(message_data)

        # Process document
        result = await pipeline.process(full_payload)

        # Acknowledge message only if successful or explicitly failed (not transient)
        await redis_client.xack(
            settings.REDIS_STREAM_KEY, settings.REDIS_CONSUMER_GROUP, message_id
        )

        logger.info(f"Message {message_id} processed and acknowledged")
        return result

    except Exception as e:
        logger.error(f"Error processing message {message_id}: {e}")
        # Logic to handle failed messages (e.g. DLQ) could go here
        # For now, we rely on Redis PEL (Pending Entries List) for retries
        raise


async def recover_pending_messages(redis_client, consumer_name: str):
    """
    Recover pending messages that were claimed but not acknowledged.
    This handles crash recovery.
    """
    try:
        # Check pending entries list
        pending = await redis_client.xpending(
            settings.REDIS_STREAM_KEY, settings.REDIS_CONSUMER_GROUP
        )

        if pending["pending"] > 0:
            logger.info(f"Found {pending['pending']} pending messages to recover")

            # Claim messages older than 10 minutes (assuming crash)
            min_idle_time = 600000  # 10 minutes in ms

            # Read pending messages
            messages = await redis_client.xautoclaim(
                settings.REDIS_STREAM_KEY,
                settings.REDIS_CONSUMER_GROUP,
                consumer_name,
                min_idle_time,
                start_id="0-0",
                count=10,
            )

            # messages format: (start_id, messages_list)
            if messages and len(messages) > 1 and messages[1]:
                msg_list = messages[1]
                logger.info(f"Recovered {len(msg_list)} messages")

                # Process recovered messages immediately
                tasks = []
                for message_id, message_data in msg_list:
                    s3_key = message_data.get("s3_key")
                    if s3_key:
                        tasks.append(
                            _process_single_message(
                                redis_client, message_id, message_data, s3_key
                            )
                        )

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

    except Exception as e:
        logger.error(f"Error recovering pending messages: {e}")


async def main():
    """Main orchestrator loop"""
    # Setup signal handlers
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        logger.debug("Signal handlers not registered (running in background thread)")

    # Initialize connections
    logger.info("Connecting to PostgreSQL...")
    await postgres.connect()

    logger.info(
        f"Connecting to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}..."
    )
    r = redis.from_url(
        f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
        decode_responses=True,
        socket_timeout=10,
    )

    # Create consumer group
    try:
        await r.xgroup_create(
            settings.REDIS_STREAM_KEY,
            settings.REDIS_CONSUMER_GROUP,
            id="0",
            mkstream=True,
        )
        logger.info(f"Created consumer group: {settings.REDIS_CONSUMER_GROUP}")
    except Exception:
        pass  # Group exists

    consumer_name = f"orchestrator-{int(time.time())}"
    logger.info(f"NiruSense Orchestrator ({consumer_name}) started")

    # Initial recovery pass
    await recover_pending_messages(r, consumer_name)

    while not shutdown_event.is_set():
        try:
            # Read new messages
            streams = await r.xreadgroup(
                settings.REDIS_CONSUMER_GROUP,
                consumer_name,
                {settings.REDIS_STREAM_KEY: ">"},
                count=settings.BATCH_SIZE,
                block=2000,
            )

            if not streams:
                # Periodic recovery check while idle
                if int(time.time()) % 60 == 0:
                    await recover_pending_messages(r, consumer_name)
                continue

            tasks = []

            for stream, messages in streams:
                for message_id, message_data in messages:
                    s3_key = message_data.get("s3_key")
                    if s3_key:
                        tasks.append(
                            _process_single_message(r, message_id, message_data, s3_key)
                        )
                    else:
                        logger.warning(f"Missing s3_key in message: {message_id}")
                        await r.xack(
                            settings.REDIS_STREAM_KEY,
                            settings.REDIS_CONSUMER_GROUP,
                            message_id,
                        )

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(1)

    # Cleanup
    logger.info("Shutting down orchestrator...")
    await postgres.close()
    await r.close()
    logger.info("Orchestrator shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
