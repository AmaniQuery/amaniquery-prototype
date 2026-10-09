"""
Populate vector database from processed data

Enhanced with:
- Batch processing to prevent write failures
- Retry logic with exponential backoff
- Progress tracking and resumption
- Memory-efficient streaming for large datasets
"""

import sys
import json
import time
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Generator
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from Module3_NiruDB.vector_store import VectorStore
from Module4_NiruAPI.config_manager import ConfigManager

# Batch configuration
DEFAULT_BATCH_SIZE = 50  # Smaller batches for reliability
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # Exponential backoff base (seconds)
PROGRESS_FILE = Path(__file__).parent.parent / "data" / ".populate_progress.json"


def load_progress() -> Dict:
    """Load progress from previous run for resumption"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_files": [], "last_run": None}


def save_progress(progress: Dict):
    """Save progress for resumption"""
    try:
        progress["last_run"] = datetime.utcnow().isoformat()
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save progress: {e}")


def clear_progress():
    """Clear progress file (for fresh start)"""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


def stream_chunks_from_file(
    jsonl_file: Path, batch_size: int = DEFAULT_BATCH_SIZE
) -> Generator[List[Dict], None, None]:
    """
    Stream chunks from JSONL file in batches (memory efficient)

    Args:
        jsonl_file: Path to JSONL file
        batch_size: Number of chunks per batch

    Yields:
        List of chunks (batch)
    """
    batch = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    chunk = json.loads(line)
                    batch.append(chunk)

                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
                    continue

    # Yield remaining chunks
    if batch:
        yield batch


def add_batch_with_retry(
    vector_store,
    store_name: str,
    chunks: List[Dict],
    namespace: str,
    max_retries: int = MAX_RETRIES,
) -> bool:
    """
    Add a batch of chunks with retry logic and exponential backoff

    Args:
        vector_store: VectorStore instance
        store_name: Name of the store (for logging)
        chunks: List of chunks to add
        namespace: Namespace for the chunks
        max_retries: Maximum number of retry attempts

    Returns:
        True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            vector_store.add_documents(
                chunks, batch_size=len(chunks), namespace=namespace
            )
            return True
        except Exception as e:
            delay = RETRY_DELAY_BASE ** (attempt + 1)

            if attempt < max_retries - 1:
                logger.warning(
                    f"Batch failed for {store_name} (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Batch failed for {store_name} after {max_retries} attempts: {e}"
                )
                return False

    return False


def determine_namespace(category: str, publication_date: str) -> str:
    """Determine the appropriate namespace based on category and publication date"""
    # Check for historical data (pre-2010)
    if publication_date:
        try:
            # Extract year from publication date
            if isinstance(publication_date, str):
                # Handle various date formats
                import re

                year_match = re.search(r"(\d{4})", publication_date)
                if year_match:
                    year = int(year_match.group(1))
                    if year < 2010:
                        return "historical"
        except (ValueError, AttributeError):
            pass

    # Map categories to namespaces
    category_lower = category.lower()

    # Kenya Law namespace
    if any(
        keyword in category_lower
        for keyword in [
            "kenyan law",
            "case law",
            "kenya gazette",
            "kenya law blog",
            "cause lists",
            "constitution",
            "act",
            "legislation",
            "judgment",
        ]
    ):
        return "kenya_law"

    # Kenya News namespace
    elif any(keyword in category_lower for keyword in ["kenyan news", "news"]):
        return "kenya_news"

    # Kenya Parliament namespace
    elif any(
        keyword in category_lower
        for keyword in [
            "parliament",
            "parliamentary record",
            "bill",
            "hansard",
            "budget",
        ]
    ):
        return "kenya_parliament"

    # Global Trends namespace (default for global content)
    elif any(keyword in category_lower for keyword in ["global trend"]):
        return "global_trends"

    # Default fallback
    else:
        return "kenya_law"  # Default to kenya_law for legal content


def main(
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = True,
    fresh_start: bool = False,
    backends: Optional[List[str]] = None,
):
    """
    Load processed data into vector databases and Elasticsearch

    Args:
        batch_size: Number of chunks per batch (default: 50)
        resume: Resume from last progress (default: True)
        fresh_start: Clear progress and start fresh (default: False)
        backends: List of backends to populate (default: all available)
    """
    print("=" * 60)
    print("[DB] Populating Databases (Batch Mode)")
    print("=" * 60)
    print(f"   Batch size: {batch_size}")
    print(f"   Max retries: {MAX_RETRIES}")
    print(f"   Resume mode: {resume}")

    # Handle progress
    if fresh_start:
        clear_progress()
        print("   [INIT] Starting fresh (progress cleared)")

    progress = load_progress() if resume else {"completed_files": [], "last_run": None}

    if progress.get("last_run"):
        print(f"   [INFO] Resuming from: {progress['last_run']}")
        print(
            f"   [INFO] Previously completed: {len(progress['completed_files'])} files"
        )

    # Initialize config manager (optional - fallback to env vars)
    config_manager = None
    try:
        config_manager = ConfigManager()
        print("[OK] ConfigManager initialized")
    except Exception as e:
        print(f"[WARN] ConfigManager not available ({e}), using environment variables")
        config_manager = None

    # Initialize vector stores for different backends
    vector_stores = []
    available_backends = backends or ["upstash", "qdrant", "chromadb"]

    # Try to initialize Upstash Vector Store
    if "upstash" in available_backends:
        try:
            print("[INIT] Initializing Upstash Vector Store...")
            upstash_store = VectorStore(
                backend="upstash",
                collection_name="amaniquery_docs",
                config_manager=config_manager,
            )
            vector_stores.append(("Upstash", upstash_store))
            print("[OK] Upstash Vector Store initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Upstash: {e}")

    # Try to initialize QDrant Vector Store
    if "qdrant" in available_backends:
        try:
            print("[INIT] Initializing QDrant Vector Store...")
            qdrant_store = VectorStore(
                backend="qdrant",
                collection_name="amaniquery_docs",
                config_manager=config_manager,
            )
            vector_stores.append(("QDrant", qdrant_store))
            print("[OK] QDrant Vector Store initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize QDrant: {e}")

    # Initialize ChromaDB as fallback
    if "chromadb" in available_backends:
        try:
            print("[INIT] Initializing ChromaDB Vector Store...")
            chromadb_store = VectorStore(
                backend="chromadb",
                collection_name="amaniquery_docs",
                config_manager=config_manager,
            )
            vector_stores.append(("ChromaDB", chromadb_store))
            print("[OK] ChromaDB Vector Store initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize ChromaDB: {e}")

    if not vector_stores:
        print("[ERROR] Error: No vector stores could be initialized")
        print("   Please check your configuration and API keys")
        return {"status": "failed", "error": "No vector stores available"}

    print(f"\n[INFO] Initialized {len(vector_stores)} vector stores:")
    for name, store in vector_stores:
        print(f"   - {name}")

    # Find processed data
    processed_path = Path(__file__).parent.parent / "data" / "processed"

    if not processed_path.exists():
        print(f"[ERROR] Error: Processed data directory not found")
        print(f"   Please run Module 2 (NiruParser) first")
        return {"status": "failed", "error": "No processed data"}

    # Find all processed JSONL files
    jsonl_files = list(processed_path.rglob("*_processed.jsonl"))

    if not jsonl_files:
        print(f"[WARN] No processed data files found")
        print(f"   Please run Module 2 (NiruParser) first")
        return {"status": "failed", "error": "No JSONL files found"}

    # Filter out already completed files if resuming
    if resume and progress.get("completed_files"):
        completed_set = set(progress["completed_files"])
        jsonl_files = [f for f in jsonl_files if str(f) not in completed_set]
        print(f"\n[INFO] Found {len(jsonl_files)} remaining files to process")
    else:
        print(f"\n[INFO] Found {len(jsonl_files)} processed files\n")

    # Statistics tracking
    stats = {
        "total_chunks": 0,
        "total_batches": 0,
        "failed_batches": 0,
        "files_processed": 0,
        "stores_updated": {name: 0 for name, _ in vector_stores},
        "namespaces": {},
    }

    # Process each file
    for jsonl_file in tqdm(jsonl_files, desc="Processing files"):
        file_path_str = str(jsonl_file)
        print(f"\n[FILE] Processing: {jsonl_file.name}")

        file_chunks = 0
        file_batches = 0
        file_failed = 0

        # Stream chunks in batches (memory efficient)
        for batch_chunks in stream_chunks_from_file(jsonl_file, batch_size):
            if not batch_chunks:
                continue

            # Group batch by namespace
            namespace_batches = {}
            for chunk in batch_chunks:
                category = chunk.get("category", "Unknown")
                publication_date = chunk.get("publication_date", "")
                ns = determine_namespace(category, publication_date)

                if ns not in namespace_batches:
                    namespace_batches[ns] = []
                namespace_batches[ns].append(chunk)

                # Track namespace stats
                if ns not in stats["namespaces"]:
                    stats["namespaces"][ns] = 0
                stats["namespaces"][ns] += 1

            # Process each namespace batch for each store
            for namespace, ns_chunks in namespace_batches.items():
                for store_name, vector_store in vector_stores:
                    success = add_batch_with_retry(
                        vector_store=vector_store,
                        store_name=store_name,
                        chunks=ns_chunks,
                        namespace=namespace,
                        max_retries=MAX_RETRIES,
                    )

                    if success:
                        stats["stores_updated"][store_name] += len(ns_chunks)

                        # Also index in elasticsearch if available
                        if vector_store.es_client is not None:
                            try:
                                for chunk in ns_chunks:
                                    doc_id = str(chunk.get("chunk_id"))
                                    vector_store.index_document(
                                        doc_id, chunk, namespace=namespace
                                    )
                            except Exception as e:
                                logger.warning(f"Elasticsearch indexing error: {e}")
                    else:
                        file_failed += 1
                        stats["failed_batches"] += 1

            file_chunks += len(batch_chunks)
            file_batches += 1
            stats["total_batches"] += 1

        # Update statistics
        stats["total_chunks"] += file_chunks
        stats["files_processed"] += 1

        # Mark file as completed
        progress["completed_files"].append(file_path_str)
        save_progress(progress)

        print(f"   [OK] Processed {file_chunks} chunks in {file_batches} batches")
        if file_failed > 0:
            print(f"   [WARN] {file_failed} batch failures")

    # Show final stats
    print("\n" + "=" * 60)
    print(f"[OK] Database Population Complete!")
    print(f"[INFO] Total files processed: {stats['files_processed']}")
    print(f"[INFO] Total chunks added: {stats['total_chunks']}")
    print(f"[INFO] Total batches: {stats['total_batches']}")
    if stats["failed_batches"] > 0:
        print(f"[WARN] Failed batches: {stats['failed_batches']}")
    print(f"[INFO] Vector stores updated: {len(vector_stores)}")

    print(f"\n[STATS] Chunks per store:")
    for store_name, count in stats["stores_updated"].items():
        print(f"   - {store_name}: {count} chunks")

    print(f"\n[STATS] Chunks per namespace:")
    for namespace, count in stats["namespaces"].items():
        print(f"   - {namespace}: {count} chunks")

    print(f"\n[STATS] Database Statistics:")
    for store_name, vector_store in vector_stores:
        try:
            store_stats = vector_store.get_stats()
            print(f"\n[STORE] {store_name}:")
            print(f"   Backend: {store_stats.get('backend', 'unknown')}")
            print(f"   Collection: {store_stats.get('collection_name', 'unknown')}")
            print(f"   Total chunks: {store_stats.get('total_chunks', 'unknown')}")

            if store_stats.get("elasticsearch_enabled"):
                es_docs = store_stats.get("elasticsearch_docs", 0)
                print(f"   Elasticsearch docs: {es_docs}")

            if store_stats.get("persist_directory"):
                print(f"   Storage location: {store_stats['persist_directory']}")

            if store_stats.get("sample_categories"):
                print(
                    f"   Categories: {', '.join(store_stats['sample_categories'].keys())}"
                )

        except Exception as e:
            print(f"   [ERROR] Could not get stats: {e}")

    # Clear progress on successful completion
    if stats["failed_batches"] == 0:
        clear_progress()
        print("\n[OK] Progress cleared (all files completed successfully)")

    print("\n" + "=" * 60)
    print("[DONE] All databases updated successfully!")
    print("   Data is now available in:")
    print("   - Vector databases (for semantic search)")
    print("   - Elasticsearch (for full-text search)")
    print("=" * 60)

    return {
        "status": "completed" if stats["failed_batches"] == 0 else "partial",
        "total_chunks": stats["total_chunks"],
        "total_batches": stats["total_batches"],
        "failed_batches": stats["failed_batches"],
        "files_processed": stats["files_processed"],
        "stores_updated": stats["stores_updated"],
        "namespaces": stats["namespaces"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Populate vector databases from processed data"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of chunks per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--fresh",
        "-f",
        action="store_true",
        help="Start fresh (clear progress from previous runs)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from previous progress"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["upstash", "qdrant", "chromadb"],
        help="Specific backends to populate (default: all available)",
    )

    args = parser.parse_args()

    result = main(
        batch_size=args.batch_size,
        resume=not args.no_resume,
        fresh_start=args.fresh,
        backends=args.backends,
    )

    # Exit with appropriate code
    if result and result.get("status") == "completed":
        sys.exit(0)
    else:
        sys.exit(1)
