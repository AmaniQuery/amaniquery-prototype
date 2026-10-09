"""
Main Processing Pipeline - Orchestrates ETL and embedding
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from .config import Config
from .extractors import HTMLExtractor, PDFExtractor
from .cleaners import TextCleaner
from .chunkers import TextChunker
from .enrichers import MetadataEnricher
from .embedders import TextEmbedder


class ProcessingPipeline:
    """Main ETL and embedding pipeline"""

    def __init__(self, config: Optional[Config] = None):
        """Initialize pipeline with all components"""
        self.config = config or Config()

        # Initialize components
        logger.info("Initializing processing pipeline")
        self.html_extractor = HTMLExtractor()
        self.pdf_extractor = PDFExtractor()
        self.cleaner = TextCleaner()
        self.chunker = TextChunker(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=self.config.CHUNK_SEPARATORS,
        )
        self.enricher = MetadataEnricher()
        self.embedder = TextEmbedder(
            model_name=self.config.EMBEDDING_MODEL,
            batch_size=self.config.EMBEDDING_BATCH_SIZE,
            normalize=self.config.NORMALIZE_EMBEDDINGS,
        )

        # Initialize database storage
        try:
            from Module3_NiruDB.database_storage import DatabaseStorage

            self.db_storage = DatabaseStorage()
            logger.info("Database storage initialized")
        except Exception as e:
            logger.warning(f"Database storage not available: {e}")
            self.db_storage = None

        # Metrics
        self.documents_processed = 0
        self.documents_succeeded = 0
        self.documents_failed = 0
        self.total_chunks_created = 0

        logger.info("Pipeline initialized successfully")

    def process_document(self, raw_doc: Dict) -> List[Dict]:
        """
        Process a single document through the full pipeline

        Args:
            raw_doc: Raw document dictionary from crawler

        Returns:
            List of processed chunks with embeddings
        """
        try:
            # 1. Extract text based on content type
            logger.info(f"Processing: {raw_doc.get('title', 'Untitled')[:50]}...")

            content_type = raw_doc.get("content_type", "html")
            text = ""
            extracted_meta = {}

            if content_type == "pdf":
                pdf_path = raw_doc.get("pdf_path")
                if pdf_path:
                    result = self.pdf_extractor.extract(pdf_path)
                    text = result["text"]
                    extracted_meta = {
                        "author": result.get("author"),
                        "creation_date": result.get("creation_date"),
                    }
            elif content_type == "html":
                html = raw_doc.get("raw_html", "")
                result = self.html_extractor.extract(html, url=raw_doc.get("url"))
                text = result["text"]
                extracted_meta = {
                    "author": result.get("author"),
                    "date": result.get("date"),
                }
            else:
                # Plain text
                text = raw_doc.get("content", "")

            # Validate text length
            if len(text) < self.config.MIN_TEXT_LENGTH:
                logger.warning(f"Text too short ({len(text)} chars), skipping")
                return []

            if len(text) > self.config.MAX_TEXT_LENGTH:
                logger.warning(f"Text too long ({len(text)} chars), truncating")
                text = text[: self.config.MAX_TEXT_LENGTH]

            # 2. Clean text
            text = self.cleaner.clean(text, aggressive=False)
            text = self.cleaner.fix_encoding(text)
            text = self.cleaner.remove_boilerplate(text)

            # 3. Prepare metadata
            metadata = {
                "url": raw_doc.get("url", ""),
                "source_url": raw_doc.get("url", ""),
                "title": raw_doc.get("title", "Untitled"),
                "category": raw_doc.get("category", "Unknown"),
                "source_name": raw_doc.get("source_name", "Unknown"),
                "author": raw_doc.get("author") or extracted_meta.get("author"),
                "publication_date": (
                    raw_doc.get("publication_date")
                    or extracted_meta.get("date")
                    or extracted_meta.get("creation_date")
                ),
                "crawl_date": raw_doc.get("crawl_date"),
            }

            # 4. Chunk text
            chunks = self.chunker.chunk(text, metadata)

            if not chunks:
                logger.warning("No chunks created")
                return []

            # Limit chunks per document
            if len(chunks) > self.config.MAX_CHUNKS_PER_DOC:
                logger.info(f"Limiting to {self.config.MAX_CHUNKS_PER_DOC} chunks")
                chunks = chunks[: self.config.MAX_CHUNKS_PER_DOC]

            # 5. Enrich chunks with metadata
            chunks = self.enricher.enrich_batch(chunks)

            # 6. Generate embeddings
            chunks = self.embedder.embed_chunks(chunks)

            self.documents_processed += 1
            if chunks:
                self.documents_succeeded += 1
                self.total_chunks_created += len(chunks)
            logger.info(f"Pipeline completed: {len(chunks)} chunks created")
            return chunks
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            self.documents_processed += 1
            self.documents_failed += 1
            return []

    def process_batch(self, raw_docs: List[Dict]) -> List[Dict]:
        """Process multiple documents in parallel using ThreadPoolExecutor"""
        all_chunks = []
        max_workers = min(self.config.MAX_WORKERS, len(raw_docs) or 1)

        if max_workers <= 1 or len(raw_docs) <= 1:
            for doc in raw_docs:
                chunks = self.process_document(doc)
                all_chunks.extend(chunks)
            return all_chunks

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self.process_document, doc): doc for doc in raw_docs
            }
            for future in as_completed(future_map):
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error in parallel document processing: {e}")

        return all_chunks

    def save_chunks(self, chunks: List[Dict], output_file: Path):
        """Save processed chunks to JSONL file (streaming compatible)"""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    # Convert to JSON (embeddings are already lists)
                    line = json.dumps(chunk, ensure_ascii=False, default=str)
                    f.write(line + "\n")

            logger.info(f"Saved {len(chunks)} chunks to {output_file}")

        except Exception as e:
            logger.error(f"Error saving chunks: {e}")

    def iter_raw_documents(self, jsonl_file: Path) -> Iterator[Dict]:
        """Stream raw documents from JSONL file as a generator"""
        try:
            count = 0
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
                        count += 1
            logger.info(f"Streamed {count} documents from {jsonl_file}")
        except Exception as e:
            logger.error(f"Error streaming documents: {e}")

    def load_raw_documents(self, jsonl_file: Path) -> List[Dict]:
        """Load raw documents from JSONL file into memory"""
        return list(self.iter_raw_documents(jsonl_file))
