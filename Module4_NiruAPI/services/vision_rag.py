"""
Vision RAG Service - Multimodal RAG using Cohere Embed-4 and Gemini 2.5 Flash

Extended to support:
- Image/PDF processing (existing)
- Video frame extraction and analysis
- Audio transcription for RAG integration
"""

import os
import time
from typing import List, Dict, Optional, Union, Generator
from pathlib import Path
import numpy as np
from loguru import logger
from PIL import Image
import tempfile
import base64
import io

from Module2_NiruParser.embedders.vision_embedder import VisionEmbedder
from Module4_NiruAPI.services.pdf_page_extractor import PDFPageExtractor

# Video processing (optional)
try:
    from Module4_NiruAPI.services.video_processor import (
        VideoProcessor,
        ExtractionResult,
        ExtractedFrame,
        VideoMetadata,
    )

    VIDEO_PROCESSING_AVAILABLE = True
except ImportError:
    VIDEO_PROCESSING_AVAILABLE = False
    logger.debug("Video processing not available")

# Audio transcription (optional)
try:
    from Module6_NiruVoice.providers.whisper_provider import (
        WhisperProvider,
        WhisperConfig,
        TranscriptionResult,
    )

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.debug("Whisper provider not available")

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning(
        "google-generativeai not available. Install with: pip install google-generativeai"
    )


class VisionRAGService:
    """Vision RAG service for multimodal search and visual question answering

    Supports:
    - Image embedding and visual QA (Cohere Embed-4 + Gemini 2.5 Flash)
    - Video frame extraction and analysis
    - Audio transcription integration for RAG
    """

    def __init__(
        self,
        cohere_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        vision_embedder: Optional[VisionEmbedder] = None,
        enable_video: bool = True,
        enable_audio: bool = True,
    ):
        """
        Initialize Vision RAG service

        Args:
            cohere_api_key: Cohere API key (if None, reads from COHERE_API_KEY env var)
            gemini_api_key: Gemini API key (if None, reads from GEMINI_API_KEY env var)
            vision_embedder: Optional pre-initialized VisionEmbedder
            enable_video: Enable video processing capabilities
            enable_audio: Enable audio transcription capabilities
        """
        # Initialize vision embedder
        if vision_embedder:
            self.vision_embedder = vision_embedder
        else:
            self.vision_embedder = VisionEmbedder(api_key=cohere_api_key)

        # Initialize Gemini for VQA
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai not available. Install with: pip install google-generativeai"
            )

        gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not set in environment or provided")

        genai.configure(api_key=gemini_key)
        self.gemini_model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )  # Use Gemini 2.5 Flash

        # Initialize video processor
        self.video_processor = None
        if enable_video and VIDEO_PROCESSING_AVAILABLE:
            self.video_processor = VideoProcessor()
            logger.info("Video processing enabled")
        elif enable_video:
            logger.warning("Video processing requested but not available")

        # Initialize audio transcriber
        self.whisper_provider = None
        if enable_audio and WHISPER_AVAILABLE:
            self.whisper_provider = WhisperProvider()
            logger.info("Audio transcription enabled")
        elif enable_audio:
            logger.warning("Audio transcription requested but not available")

        logger.info("Vision RAG service initialized")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def search_images(
        self,
        query_text: str,
        image_embeddings: List[np.ndarray],
        image_metadata: List[Dict],
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Search for most relevant images using text query

        Args:
            query_text: Text query/question
            image_embeddings: List of image embeddings
            image_metadata: List of metadata dicts for each image
            top_k: Number of top results to return

        Returns:
            List of dicts with image info and similarity scores, sorted by relevance
        """
        try:
            # Embed the text query
            query_embedding = self.vision_embedder.embed_text(query_text)

            # Calculate similarities
            results = []
            for i, (img_emb, metadata) in enumerate(
                zip(image_embeddings, image_metadata)
            ):
                similarity = self._cosine_similarity(query_embedding, img_emb)
                results.append(
                    {
                        "index": i,
                        "metadata": metadata,
                        "similarity": similarity,
                    }
                )

            # Sort by similarity (descending)
            results.sort(key=lambda x: x["similarity"], reverse=True)

            # Return top_k results
            top_results = results[:top_k]

            logger.info(
                f"Found {len(top_results)} relevant images for query: {query_text[:50]}..."
            )
            return top_results

        except Exception as e:
            logger.error(f"Error searching images: {e}")
            return []

    def answer_visual_question(
        self,
        question: str,
        image_paths: List[Union[str, Path]],
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> str:
        """
        Answer a question about one or more images using Gemini

        Args:
            question: Question to ask about the images
            image_paths: List of paths to images to analyze
            temperature: Generation temperature
            max_tokens: Maximum tokens in response

        Returns:
            Answer text
        """
        try:
            # Prepare images for Gemini
            image_parts = []
            for img_path in image_paths:
                img_path = Path(img_path)
                if not img_path.exists():
                    logger.warning(f"Image not found: {img_path}")
                    continue

                # Read and encode image
                with open(img_path, "rb") as f:
                    image_data = f.read()

                image_part = {
                    "mime_type": (
                        "image/jpeg"
                        if img_path.suffix.lower() in [".jpg", ".jpeg"]
                        else "image/png"
                    ),
                    "data": image_data,
                }
                image_parts.append(image_part)

            if not image_parts:
                return "No valid images provided for analysis."

            # Prepare prompt
            prompt = f"""Analyze the provided image(s) and answer the following question:

Question: {question}

Provide a detailed, accurate answer based on what you see in the image(s). If the question cannot be answered from the images, please state that clearly."""

            # Generate response with Gemini
            response = self.gemini_model.generate_content(
                [prompt] + image_parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            answer = response.text
            logger.info(f"Generated visual answer (length: {len(answer)})")
            return answer

        except Exception as e:
            logger.error(f"Error generating visual answer: {e}")
            return f"I encountered an error while analyzing the images: {str(e)}"

    def answer_visual_question_stream(
        self,
        question: str,
        image_paths: List[Union[str, Path]],
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ):
        """
        Answer a question about one or more images using Gemini with streaming

        Args:
            question: Question to ask about the images
            image_paths: List of paths to images to analyze
            temperature: Generation temperature
            max_tokens: Maximum tokens in response

        Yields:
            Text chunks as they are generated
        """
        try:
            # Prepare images for Gemini
            image_parts = []
            for img_path in image_paths:
                img_path = Path(img_path)
                if not img_path.exists():
                    logger.warning(f"Image not found: {img_path}")
                    continue

                # Read and encode image
                with open(img_path, "rb") as f:
                    image_data = f.read()

                image_part = {
                    "mime_type": (
                        "image/jpeg"
                        if img_path.suffix.lower() in [".jpg", ".jpeg"]
                        else "image/png"
                    ),
                    "data": image_data,
                }
                image_parts.append(image_part)

            if not image_parts:
                yield "No valid images provided for analysis."
                return

            # Prepare prompt
            prompt = f"""Analyze the provided image(s) and answer the following question:

Question: {question}

Provide a detailed, accurate answer based on what you see in the image(s). If the question cannot be answered from the images, please state that clearly."""

            # Generate streaming response with Gemini
            response = self.gemini_model.generate_content(
                [prompt] + image_parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
                stream=True,  # Enable streaming
            )

            # Yield chunks as they arrive
            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Error generating streaming visual answer: {e}")
            yield f"I encountered an error while analyzing the images: {str(e)}"

    def query(
        self,
        question: str,
        session_images: List[Dict],
        top_k: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        stream: bool = False,
    ) -> Dict:
        """
        Complete Vision RAG query: search for relevant images and answer question

        Args:
            question: User's question
            session_images: List of dicts with keys: {id, file_path, embedding, metadata}
            top_k: Number of images to retrieve
            temperature: Generation temperature
            max_tokens: Maximum tokens in response

        Returns:
            Dict with answer, sources (images), and metadata
        """
        if not session_images:
            return {
                "answer": "No images available in this session. Please upload images or PDFs first.",
                "sources": [],
                "query_time": 0.0,
                "retrieved_images": 0,
            }

        import time

        start_time = time.time()

        try:
            # Extract embeddings and metadata
            image_embeddings = []
            image_metadata = []

            for img_data in session_images:
                # Convert embedding to numpy array if it's a list
                embedding = img_data.get("embedding")
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                elif not isinstance(embedding, np.ndarray):
                    logger.warning(
                        f"Invalid embedding type for image {img_data.get('id')}"
                    )
                    continue

                image_embeddings.append(embedding)
                image_metadata.append(img_data.get("metadata", {}))

            if not image_embeddings:
                return {
                    "answer": "No valid image embeddings found. Please re-upload your images.",
                    "sources": [],
                    "query_time": time.time() - start_time,
                    "retrieved_images": 0,
                }

            # Search for relevant images
            search_results = self.search_images(
                query_text=question,
                image_embeddings=image_embeddings,
                image_metadata=image_metadata,
                top_k=top_k,
            )

            if not search_results:
                return {
                    "answer": "I couldn't find any relevant images for your question.",
                    "sources": [],
                    "query_time": time.time() - start_time,
                    "retrieved_images": 0,
                }

            # Get paths to top images
            top_image_paths = []
            top_image_metadata = []

            for result in search_results:
                idx = result["index"]
                img_data = session_images[idx]
                file_path = img_data.get("file_path")

                if file_path and Path(file_path).exists():
                    top_image_paths.append(file_path)
                    top_image_metadata.append(
                        {
                            **img_data.get("metadata", {}),
                            "similarity": result["similarity"],
                        }
                    )

            # Generate answer using Gemini (streaming or non-streaming)
            if stream:
                # Return streaming generator
                answer_stream = self.answer_visual_question_stream(
                    question=question,
                    image_paths=top_image_paths,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Format sources
                sources = []
                for meta in top_image_metadata:
                    sources.append(
                        {
                            "file_path": meta.get("file_path", ""),
                            "filename": meta.get("filename", ""),
                            "page_number": meta.get("page_number"),
                            "source_file": meta.get("source_file", ""),
                            "similarity": meta.get("similarity", 0.0),
                        }
                    )

                return {
                    "answer_stream": answer_stream,
                    "sources": sources,
                    "query_time": 0.0,  # Will be calculated after streaming
                    "retrieved_images": len(top_image_paths),
                    "model_used": "gemini-2.5-flash",
                    "stream": True,
                }
            else:
                # Non-streaming
                answer = self.answer_visual_question(
                    question=question,
                    image_paths=top_image_paths,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                query_time = time.time() - start_time

                # Format sources
                sources = []
                for meta in top_image_metadata:
                    sources.append(
                        {
                            "file_path": meta.get("file_path", ""),
                            "filename": meta.get("filename", ""),
                            "page_number": meta.get("page_number"),
                            "source_file": meta.get("source_file", ""),
                            "similarity": meta.get("similarity", 0.0),
                        }
                    )

                return {
                    "answer": answer,
                    "sources": sources,
                    "query_time": query_time,
                    "retrieved_images": len(top_image_paths),
                    "model_used": "gemini-2.5-flash",
                }

        except Exception as e:
            logger.error(f"Error in Vision RAG query: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "query_time": time.time() - start_time,
                "retrieved_images": 0,
            }

    # ==================== VIDEO PROCESSING ====================

    def process_video(
        self,
        video_path: Union[str, Path],
        num_frames: int = 10,
        extract_audio: bool = True,
        transcribe_audio: bool = True,
    ) -> Dict:
        """
        Process video for RAG: extract frames, embed them, and transcribe audio

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            extract_audio: Whether to extract audio track
            transcribe_audio: Whether to transcribe audio (requires Whisper)

        Returns:
            Dict with embedded frames, transcription, and metadata
        """
        if not self.video_processor:
            return {
                "success": False,
                "error": "Video processing not available. Install opencv-python and ensure ffmpeg is installed.",
                "frames": [],
                "transcription": None,
            }

        video_path = Path(video_path)
        start_time = time.time()

        try:
            logger.info(f"Processing video: {video_path.name}")

            # Extract frames and audio
            extraction_result = self.video_processor.process_video(
                video_path=video_path,
                num_frames=num_frames,
                extract_audio=extract_audio,
            )

            # Embed extracted frames
            embedded_frames = []
            for frame in extraction_result.frames:
                try:
                    embedding = self.vision_embedder.embed_image(frame.frame_path)
                    embedded_frames.append(
                        {
                            "file_path": frame.frame_path,
                            "timestamp": frame.timestamp,
                            "frame_number": frame.frame_number,
                            "embedding": embedding.tolist(),
                            "metadata": {
                                **frame.metadata,
                                "source_video": str(video_path),
                                "source_type": "video_frame",
                            },
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to embed frame at {frame.timestamp}s: {e}")

            # Transcribe audio if available
            transcription = None
            if (
                transcribe_audio
                and extraction_result.audio_path
                and self.whisper_provider
            ):
                try:
                    transcription_result = self.whisper_provider.transcribe(
                        extraction_result.audio_path
                    )
                    transcription = {
                        "text": transcription_result.text,
                        "language": transcription_result.language,
                        "duration": transcription_result.duration,
                        "confidence": transcription_result.confidence,
                        "segments": transcription_result.segments,
                    }
                    logger.info(
                        f"Transcribed {transcription_result.duration:.1f}s of audio"
                    )
                except Exception as e:
                    logger.warning(f"Failed to transcribe audio: {e}")

            processing_time = time.time() - start_time

            result = {
                "success": True,
                "frames": embedded_frames,
                "transcription": transcription,
                "audio_path": extraction_result.audio_path,
                "video_metadata": extraction_result.video_metadata.to_dict(),
                "output_dir": extraction_result.output_dir,
                "processing_time": processing_time,
            }

            logger.info(
                f"Video processing complete: {len(embedded_frames)} frames embedded, "
                f"transcription: {'yes' if transcription else 'no'}, "
                f"time: {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {
                "success": False,
                "error": str(e),
                "frames": [],
                "transcription": None,
            }

    async def process_video_async(
        self,
        video_path: Union[str, Path],
        num_frames: int = 10,
        extract_audio: bool = True,
        transcribe_audio: bool = True,
    ) -> Dict:
        """
        Process video asynchronously

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            extract_audio: Whether to extract audio track
            transcribe_audio: Whether to transcribe audio

        Returns:
            Dict with embedded frames, transcription, and metadata
        """
        import asyncio

        # Run video processing in executor (CPU-bound)
        loop = asyncio.get_event_loop()

        # First, extract frames and audio synchronously
        if not self.video_processor:
            return {
                "success": False,
                "error": "Video processing not available",
                "frames": [],
                "transcription": None,
            }

        video_path = Path(video_path)
        start_time = time.time()

        try:
            # Extract in executor
            extraction_result = await loop.run_in_executor(
                None,
                lambda: self.video_processor.process_video(
                    video_path=video_path,
                    num_frames=num_frames,
                    extract_audio=extract_audio,
                ),
            )

            # Embed frames in executor
            embedded_frames = []
            for frame in extraction_result.frames:
                try:
                    embedding = await loop.run_in_executor(
                        None,
                        lambda fp=frame.frame_path: self.vision_embedder.embed_image(
                            fp
                        ),
                    )
                    embedded_frames.append(
                        {
                            "file_path": frame.frame_path,
                            "timestamp": frame.timestamp,
                            "frame_number": frame.frame_number,
                            "embedding": embedding.tolist(),
                            "metadata": {
                                **frame.metadata,
                                "source_video": str(video_path),
                                "source_type": "video_frame",
                            },
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to embed frame: {e}")

            # Transcribe audio asynchronously
            transcription = None
            if (
                transcribe_audio
                and extraction_result.audio_path
                and self.whisper_provider
            ):
                try:
                    transcription_result = await self.whisper_provider.transcribe_async(
                        extraction_result.audio_path
                    )
                    transcription = {
                        "text": transcription_result.text,
                        "language": transcription_result.language,
                        "duration": transcription_result.duration,
                        "confidence": transcription_result.confidence,
                        "segments": transcription_result.segments,
                    }
                except Exception as e:
                    logger.warning(f"Failed to transcribe audio: {e}")

            processing_time = time.time() - start_time

            return {
                "success": True,
                "frames": embedded_frames,
                "transcription": transcription,
                "audio_path": extraction_result.audio_path,
                "video_metadata": extraction_result.video_metadata.to_dict(),
                "output_dir": extraction_result.output_dir,
                "processing_time": processing_time,
            }

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {
                "success": False,
                "error": str(e),
                "frames": [],
                "transcription": None,
            }

    # ==================== AUDIO PROCESSING ====================

    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        context_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Transcribe audio file for RAG integration

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            context_prompt: Optional context to improve transcription

        Returns:
            Dict with transcription text, segments, and metadata
        """
        if not self.whisper_provider:
            return {
                "success": False,
                "error": "Audio transcription not available. Check Whisper configuration.",
                "text": "",
                "segments": [],
            }

        audio_path = Path(audio_path)

        if not audio_path.exists():
            return {
                "success": False,
                "error": f"Audio file not found: {audio_path}",
                "text": "",
                "segments": [],
            }

        try:
            logger.info(f"Transcribing audio: {audio_path.name}")
            start_time = time.time()

            result = self.whisper_provider.transcribe(
                audio_path=audio_path,
                language=language,
                prompt=context_prompt,
            )

            processing_time = time.time() - start_time

            return {
                "success": True,
                "text": result.text,
                "language": result.language,
                "duration": result.duration,
                "confidence": result.confidence,
                "segments": result.segments,
                "model_used": result.model_used,
                "processing_time": processing_time,
            }

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "segments": [],
            }

    async def transcribe_audio_async(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        context_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Transcribe audio file asynchronously

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            context_prompt: Optional context to improve transcription

        Returns:
            Dict with transcription text, segments, and metadata
        """
        if not self.whisper_provider:
            return {
                "success": False,
                "error": "Audio transcription not available",
                "text": "",
                "segments": [],
            }

        audio_path = Path(audio_path)

        if not audio_path.exists():
            return {
                "success": False,
                "error": f"Audio file not found: {audio_path}",
                "text": "",
                "segments": [],
            }

        try:
            logger.info(f"Transcribing audio (async): {audio_path.name}")
            start_time = time.time()

            result = await self.whisper_provider.transcribe_async(
                audio_path=audio_path,
                language=language,
                prompt=context_prompt,
            )

            processing_time = time.time() - start_time

            return {
                "success": True,
                "text": result.text,
                "language": result.language,
                "duration": result.duration,
                "confidence": result.confidence,
                "segments": result.segments,
                "model_used": result.model_used,
                "processing_time": processing_time,
            }

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "segments": [],
            }

    def embed_audio_transcript(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> Dict:
        """
        Transcribe audio and generate text embeddings for RAG retrieval

        Args:
            audio_path: Path to audio file
            language: Language code

        Returns:
            Dict with text, embedding, segments, and metadata
        """
        # First transcribe
        transcription = self.transcribe_audio(audio_path, language=language)

        if not transcription["success"]:
            return transcription

        try:
            # Generate text embedding for the full transcript
            text_embedding = self.vision_embedder.embed_text(transcription["text"])

            # Also embed individual segments for fine-grained retrieval
            segment_embeddings = []
            for segment in transcription.get("segments", []):
                if segment.get("text", "").strip():
                    try:
                        seg_embedding = self.vision_embedder.embed_text(segment["text"])
                        segment_embeddings.append(
                            {
                                "start": segment.get("start", 0),
                                "end": segment.get("end", 0),
                                "text": segment["text"],
                                "embedding": seg_embedding.tolist(),
                            }
                        )
                    except Exception:
                        pass  # Skip failed segments

            return {
                "success": True,
                "text": transcription["text"],
                "embedding": text_embedding.tolist(),
                "segment_embeddings": segment_embeddings,
                "language": transcription.get("language"),
                "duration": transcription.get("duration"),
                "confidence": transcription.get("confidence"),
                "source_path": str(audio_path),
                "source_type": "audio",
            }

        except Exception as e:
            logger.error(f"Error embedding audio transcript: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": transcription["text"],
            }

    # ==================== UNIFIED MULTIMODAL QUERY ====================

    def query_multimodal(
        self,
        question: str,
        session_images: List[Dict] = None,
        session_transcripts: List[Dict] = None,
        top_k_images: int = 3,
        top_k_transcripts: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Dict:
        """
        Query across multiple modalities (images + audio transcripts)

        Args:
            question: User's question
            session_images: List of embedded images from session
            session_transcripts: List of embedded transcripts from session
            top_k_images: Number of images to retrieve
            top_k_transcripts: Number of transcript segments to retrieve
            temperature: Generation temperature
            max_tokens: Max response tokens

        Returns:
            Dict with answer, sources, and metadata
        """
        session_images = session_images or []
        session_transcripts = session_transcripts or []

        start_time = time.time()

        if not session_images and not session_transcripts:
            return {
                "answer": "No media available in this session. Please upload images, videos, or audio files first.",
                "sources": [],
                "query_time": 0.0,
            }

        try:
            retrieved_context = []
            sources = []

            # Retrieve relevant images
            if session_images:
                image_results = self.query(
                    question=question,
                    session_images=session_images,
                    top_k=top_k_images,
                    stream=False,
                )

                if image_results.get("sources"):
                    sources.extend(
                        [{**src, "type": "image"} for src in image_results["sources"]]
                    )

            # Retrieve relevant transcript segments
            if session_transcripts:
                query_embedding = self.vision_embedder.embed_text(question)

                # Search transcript embeddings
                transcript_scores = []
                for i, transcript in enumerate(session_transcripts):
                    embedding = transcript.get("embedding")
                    if embedding:
                        if isinstance(embedding, list):
                            embedding = np.array(embedding)
                        similarity = self._cosine_similarity(query_embedding, embedding)
                        transcript_scores.append(
                            {
                                "index": i,
                                "similarity": similarity,
                                "transcript": transcript,
                            }
                        )

                # Sort and take top_k
                transcript_scores.sort(key=lambda x: x["similarity"], reverse=True)

                for item in transcript_scores[:top_k_transcripts]:
                    transcript = item["transcript"]
                    retrieved_context.append(
                        {
                            "type": "transcript",
                            "text": transcript.get("text", ""),
                            "source": transcript.get("source_path", ""),
                            "similarity": item["similarity"],
                        }
                    )
                    sources.append(
                        {
                            "type": "transcript",
                            "source_path": transcript.get("source_path", ""),
                            "duration": transcript.get("duration"),
                            "similarity": item["similarity"],
                        }
                    )

            # Build context for generation
            context_text = ""
            if retrieved_context:
                context_text = "\n\n".join(
                    [
                        f"[{ctx['type'].upper()}] {ctx['text']}"
                        for ctx in retrieved_context
                    ]
                )

            # If we have images, use visual QA
            if session_images and sources:
                # Use the existing visual query for images
                image_result = self.query(
                    question=question,
                    session_images=session_images,
                    top_k=top_k_images,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Augment with transcript context
                if context_text:
                    augmented_prompt = f"""Based on the images and the following additional context from audio/video transcripts:

{context_text}

Answer the question: {question}"""

                    # Re-query with augmented context
                    answer = image_result.get("answer", "")
                    if "transcript" in [s.get("type") for s in sources]:
                        answer = f"{answer}\n\nAdditional context from transcripts:\n{context_text[:500]}..."
                else:
                    answer = image_result.get("answer", "")
            else:
                # Text-only response based on transcripts
                answer = f"Based on the audio/video transcripts:\n\n{context_text}"

            query_time = time.time() - start_time

            return {
                "answer": answer,
                "sources": sources,
                "query_time": query_time,
                "modalities_used": [
                    "image" if session_images else None,
                    "transcript" if session_transcripts else None,
                ],
            }

        except Exception as e:
            logger.error(f"Error in multimodal query: {e}")
            return {
                "answer": f"Error processing multimodal query: {str(e)}",
                "sources": [],
                "query_time": time.time() - start_time,
            }

    def cleanup_video_extraction(self, output_dir: str) -> None:
        """
        Clean up extracted video files

        Args:
            output_dir: Directory to clean up
        """
        if self.video_processor:
            import shutil

            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
                logger.debug(f"Cleaned up: {output_dir}")
