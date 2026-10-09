"""
Process all raw data from Module1
"""

import sys
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Module2_NiruParser.pipeline import ProcessingPipeline
from Module2_NiruParser.config import Config


def main():
    """Process all raw data files"""
    print("=" * 60)
    print("[START] Starting Data Processing Pipeline")
    print("=" * 60)

    # Initialize pipeline
    config = Config()
    pipeline = ProcessingPipeline(config)

    # Configure logging
    logger.add(
        config.PROJECT_ROOT / "logs" / "processing.log",
        rotation="100 MB",
        level=config.LOG_LEVEL,
    )

    # Find all raw data files
    raw_data_path = config.RAW_DATA_PATH

    if not raw_data_path.exists():
        logger.error(f"Raw data path does not exist: {raw_data_path}")
        print(f"[ERROR] Error: Raw data directory not found")
        print(f"   Please run Module 1 (NiruSpider) first")
        return

    # Find all JSONL files from crawlers
    jsonl_files = list(raw_data_path.rglob("*.jsonl"))

    if not jsonl_files:
        logger.warning("No JSONL files found in raw data directory")
        print(f"[WARN] No data files found in {raw_data_path}")
        print(f"   Please run Module 1 (NiruSpider) first")
        return

    print(f"\n[INFO] Found {len(jsonl_files)} data files to process\n")

    # Process each file
    total_chunks = 0

    for jsonl_file in tqdm(jsonl_files, desc="Processing files"):
        logger.info(f"Processing file: {jsonl_file.name}")
        print(f"\n[FILE] Processing: {jsonl_file.name}")

        # Stream raw documents (generator - memory efficient)
        raw_docs = list(pipeline.load_raw_documents(jsonl_file))

        if not raw_docs:
            logger.warning(f"No documents found in {jsonl_file.name}")
            continue

        print(f"   Loaded {len(raw_docs)} documents")

        # Save raw documents to database
        if pipeline.db_storage:
            try:
                saved_raw = pipeline.db_storage.save_raw_documents(raw_docs)
                print(f"   [SAVE] Saved {saved_raw} raw documents to database")
            except Exception as e:
                logger.error(f"Failed to save raw documents to database: {e}")
                print(f"   [ERROR] Failed to save raw documents to database")

        # Process documents in parallel
        all_chunks = []
        max_workers = min(config.MAX_WORKERS, len(raw_docs) or 1)

        if max_workers <= 1:
            for doc in tqdm(raw_docs, desc="  Documents", leave=False):
                chunks = pipeline.process_document(doc)
                all_chunks.extend(chunks)
        else:
            progress = tqdm(total=len(raw_docs), desc="  Documents", leave=False)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(pipeline.process_document, doc): doc
                    for doc in raw_docs
                }
                for future in as_completed(future_map):
                    try:
                        chunks = future.result()
                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.error(f"Error in parallel processing: {e}")
                    progress.update(1)
            progress.close()

        if all_chunks:
            # Determine output filename
            category = all_chunks[0].get("category", "unknown")
            output_file = config.get_output_path(
                category, jsonl_file.stem + "_processed.jsonl"
            )

            # Save processed chunks to file
            pipeline.save_chunks(all_chunks, output_file)

            # Save processed chunks to database
            if pipeline.db_storage:
                try:
                    saved_chunks = pipeline.db_storage.save_processed_chunks(all_chunks)
                    print(
                        f"   [SAVE] Saved {saved_chunks} processed chunks to database"
                    )

                    # Mark raw documents as processed
                    urls = [doc.get("url") for doc in raw_docs if doc.get("url")]
                    if urls:
                        pipeline.db_storage.mark_raw_documents_processed(urls)
                        print(f"   [OK] Marked {len(urls)} raw documents as processed")

                except Exception as e:
                    logger.error(f"Failed to save processed chunks to database: {e}")
                    print(f"   [ERROR] Failed to save processed chunks to database")

            total_chunks += len(all_chunks)
            print(f"   [OK] Created {len(all_chunks)} chunks")
        else:
            print(f"   [WARN] No chunks created")

    print("\n" + "=" * 60)
    print(f"[OK] Processing Complete!")
    print(f"[INFO] Total chunks created: {total_chunks}")
    print(f"[INFO] Documents processed: {pipeline.documents_processed}")
    print(f"[INFO] Documents succeeded: {pipeline.documents_succeeded}")
    print(f"[INFO] Documents failed: {pipeline.documents_failed}")
    print(f"[INFO] Processed data saved to: {config.PROCESSED_DATA_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
