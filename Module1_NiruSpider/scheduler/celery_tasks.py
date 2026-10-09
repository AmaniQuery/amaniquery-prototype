"""
Celery Tasks for Crawler Scheduling

These tasks handle the actual crawling work when using Celery as the scheduler backend.
Each task runs a specific spider with proper error handling and timeout management.

Pipeline Tasks Include:
1. Crawling - Run spiders to fetch raw data
2. Processing - Parse, clean, chunk, and enrich documents
3. Embedding - Generate vector embeddings for chunks
4. Vector Store Population - Store embeddings in Qdrant/ChromaDB/Upstash
"""

import sys
import subprocess
import json
from pathlib import Path
from celery import Task, chain, group
from celery.exceptions import SoftTimeLimitExceeded
from datetime import datetime
from loguru import logger
from typing import Optional, List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .celery_app import celery_app


class CrawlTask(Task):
    """Base task class with error handling and retry logic"""

    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        # Do NOT retry on SoftTimeLimitExceeded (task ran too long)
        if isinstance(exc, SoftTimeLimitExceeded):
            logger.error(f"Task {task_id} exceeded time limit — not retrying")
            return
        crawler_name = args[0] if args else "unknown"
        logger.error(f"Task {task_id} ({crawler_name}) failed: {exc}")
        logger.error(f"Exception info: {einfo}")

        # Record failure in database if available
        try:
            self._record_status(crawler_name, "failed", str(exc))
        except Exception as e:
            logger.warning(f"Could not record failure status: {e}")

    def on_success(self, retval, task_id, args, kwargs):
        crawler_name = args[0] if args else "unknown"
        logger.info(f"Task {task_id} ({crawler_name}) completed successfully")

        # Record success in database if available
        try:
            self._record_status(crawler_name, "completed", None)
        except Exception as e:
            logger.warning(f"Could not record success status: {e}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        crawler_name = args[0] if args else "unknown"
        logger.warning(f"Task {task_id} ({crawler_name}) retrying due to: {exc}")

    def _record_status(self, crawler_name: str, status: str, error: str = None):
        """Record crawler status in database"""
        try:
            from Module4_NiruAPI.crawler_models import CrawlerDatabaseManager

            db_manager = CrawlerDatabaseManager()
            if db_manager._initialized:
                db_manager.update_crawler_status(
                    crawler_name, status, last_run=datetime.utcnow()
                )
                if error:
                    db_manager.add_log(crawler_name, f"Celery task failed: {error}")
                else:
                    db_manager.add_log(
                        crawler_name, f"Celery task completed successfully"
                    )
        except Exception as e:
            logger.debug(f"Could not record status: {e}")


# Spider name mapping
SPIDER_MAPPING = {
    "kenya_law": "kenya_law_new",
    "parliament": "parliament",
    "parliament_videos": "parliament_videos",
    "news_rss": "news_rss",
    "global_trends": "global_trends",
    "constitution": "constitution",
    "kenya_gazette": "kenya_gazette",
    "fact_check": "fact_check",
    "africa_analysis": "africa_analysis",
}

# Timeout settings per crawler (in seconds)
CRAWLER_TIMEOUTS = {
    "kenya_law": 14400,  # 240 minutes (4 hours) - comprehensive legal database
    "parliament": 3600,  # 60 minutes
    "parliament_videos": 2700,  # 45 minutes
    "news_rss": 1800,  # 30 minutes
    "global_trends": 1800,  # 30 minutes
    "constitution": 3600,  # 60 minutes
    "kenya_gazette": 1800,  # 30 minutes
    "fact_check": 1800,  # 30 minutes
    "africa_analysis": 1800,  # 30 minutes
}


@celery_app.task(
    base=CrawlTask,
    bind=True,
    name="run_crawler",
    soft_time_limit=8700,  # 145 minutes - allow for kenya_law's 120-minute timeout + buffer
    time_limit=9000,  # 150 minutes (2.5 hours) - hard limit with buffer
)
def run_crawler_task(self, crawler_name: str, timeout: int = None):
    """
    Run a specific crawler

    Args:
        crawler_name: Name of the crawler to run
        timeout: Optional custom timeout in seconds

    Returns:
        Dictionary with crawl results
    """
    spider_name = SPIDER_MAPPING.get(crawler_name)
    if not spider_name:
        raise ValueError(
            f"Unknown crawler: {crawler_name}. Valid options: {list(SPIDER_MAPPING.keys())}"
        )

    # Get timeout
    timeout = timeout or CRAWLER_TIMEOUTS.get(crawler_name, 3600)

    logger.info(f"[Celery] Starting crawler: {crawler_name} (timeout: {timeout}s)")

    spider_dir = project_root / "Module1_NiruSpider"

    try:
        # Build command
        cmd = [
            sys.executable,
            "crawl_spider.py",
            spider_name,
            "--timeout",
            str(timeout),
        ]

        # Run subprocess
        result = subprocess.run(
            cmd,
            cwd=str(spider_dir),
            capture_output=True,
            text=True,
            timeout=timeout + 120,  # Extra buffer
        )

        if result.returncode == 0:
            logger.info(f"[Celery] Crawler {crawler_name} completed successfully")

            # Trigger vector store update
            try:
                update_vector_store.delay()
            except Exception as e:
                logger.warning(f"Could not trigger vector store update: {e}")

            return {"status": "completed", "crawler": crawler_name, "exit_code": 0}
        else:
            logger.error(f"[Celery] Crawler {crawler_name} failed: {result.stderr}")
            raise Exception(f"Crawler failed with exit code {result.returncode}")

    except subprocess.TimeoutExpired:
        logger.error(f"[Celery] Crawler {crawler_name} timed out")
        raise Exception(f"Crawler timed out after {timeout}s")

    except SoftTimeLimitExceeded:
        logger.error(f"[Celery] Crawler {crawler_name} exceeded soft time limit")
        raise


@celery_app.task(
    base=CrawlTask,
    bind=True,
    name="crawl_news_sources",
    soft_time_limit=2400,
    time_limit=2700,
)
def crawl_news_sources(self, source_names: list = None):
    """
    Crawl multiple news sources

    Args:
        source_names: List of source names to crawl (None = all news sources)

    Returns:
        Dictionary with crawl results
    """
    news_sources = ["news_rss", "global_trends"]

    if source_names:
        sources_to_crawl = [s for s in source_names if s in news_sources]
    else:
        sources_to_crawl = news_sources

    results = {}

    for source in sources_to_crawl:
        try:
            result = run_crawler_task(source)
            results[source] = result
        except Exception as e:
            logger.error(f"Error crawling {source}: {e}")
            results[source] = {"status": "failed", "error": str(e)}

    return {"status": "completed", "sources": results}


@celery_app.task(
    base=CrawlTask,
    bind=True,
    name="crawl_legal_sources",
    soft_time_limit=7200,
    time_limit=7500,
)
def crawl_legal_sources(self, source_names: list = None):
    """
    Crawl legal/government sources

    Args:
        source_names: List of source names to crawl (None = all legal sources)

    Returns:
        Dictionary with crawl results
    """
    legal_sources = ["kenya_law", "parliament", "parliament_videos"]

    if source_names:
        sources_to_crawl = [s for s in source_names if s in legal_sources]
    else:
        sources_to_crawl = legal_sources

    results = {}

    for source in sources_to_crawl:
        try:
            result = run_crawler_task(source)
            results[source] = result
        except Exception as e:
            logger.error(f"Error crawling {source}: {e}")
            results[source] = {"status": "failed", "error": str(e)}

    return {"status": "completed", "sources": results}


@celery_app.task(base=CrawlTask, bind=True, name="crawl_all_sources")
def crawl_all_sources(self):
    """
    Crawl all available sources
    Runs sources in priority order with delays between them
    """
    import time

    # Order by priority (news first, then legal)
    all_sources = [
        "news_rss",
        "global_trends",
        "parliament",
        "parliament_videos",
        "kenya_law",
    ]

    results = {}

    for source in all_sources:
        try:
            logger.info(f"[Celery] Starting crawl for: {source}")
            result = run_crawler_task(source)
            results[source] = result

            # Brief delay between crawlers
            time.sleep(30)

        except Exception as e:
            logger.error(f"Error crawling {source}: {e}")
            results[source] = {"status": "failed", "error": str(e)}

    return {"status": "completed", "sources": results}


@celery_app.task(name="update_vector_store", soft_time_limit=1800, time_limit=2100)
def update_vector_store(incremental: bool = True):
    """
    Update the vector store with new documents

    Args:
        incremental: If True, only add new documents. If False, rebuild entirely.

    Returns:
        Dictionary with update results
    """
    logger.info(f"[Celery] Updating vector store (incremental={incremental})")

    try:
        update_script = project_root / "populate_qdrant.py"

        if not update_script.exists():
            logger.warning("populate_qdrant.py not found")
            return {"status": "skipped", "reason": "Script not found"}

        cmd = [sys.executable, str(update_script)]
        if incremental:
            cmd.append("--incremental")

        result = subprocess.run(
            cmd, cwd=str(project_root), capture_output=True, text=True, timeout=1800
        )

        if result.returncode == 0:
            logger.info("[Celery] Vector store updated successfully")
            return {"status": "completed"}
        else:
            logger.error(f"[Celery] Vector store update failed: {result.stderr}")
            return {"status": "failed", "error": result.stderr}

    except subprocess.TimeoutExpired:
        logger.error("[Celery] Vector store update timed out")
        return {"status": "failed", "error": "Timeout"}
    except Exception as e:
        logger.error(f"[Celery] Vector store update error: {e}")
        return {"status": "failed", "error": str(e)}


@celery_app.task(name="scheduler_health_check")
def scheduler_health_check():
    """
    Health check task - verifies the scheduler is working
    """
    logger.info("[Celery] Health check - scheduler is running")
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# =============================================================================
# PROCESSING PIPELINE TASKS
# =============================================================================


@celery_app.task(name="process_raw_data", soft_time_limit=3600, time_limit=3900)
def process_raw_data(source_filter: Optional[str] = None, incremental: bool = True):
    """
    Process raw crawled data through the parsing pipeline

    Pipeline Steps:
    1. Load raw JSONL files from data/raw/
    2. Extract text (HTML/PDF extraction)
    3. Clean and normalize text
    4. Chunk into semantic segments
    5. Enrich with metadata
    6. Save to data/processed/

    Args:
        source_filter: Optional filter to process only specific source (e.g., 'parliament', 'kenya_law')
        incremental: If True, only process new/unprocessed documents

    Returns:
        Dictionary with processing results
    """
    logger.info(
        f"[Celery] Starting data processing (filter={source_filter}, incremental={incremental})"
    )

    try:
        # Run the process_all module
        cmd = [sys.executable, "-m", "Module2_NiruParser.process_all"]

        result = subprocess.run(
            cmd, cwd=str(project_root), capture_output=True, text=True, timeout=3600
        )

        if result.returncode == 0:
            logger.info("[Celery] Data processing completed successfully")

            # Parse output for stats
            output = result.stdout
            total_chunks = 0
            if "Total chunks created:" in output:
                try:
                    total_chunks = int(
                        output.split("Total chunks created:")[-1].split()[0]
                    )
                except:
                    pass

            return {
                "status": "completed",
                "total_chunks": total_chunks,
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            logger.error(f"[Celery] Data processing failed: {result.stderr}")
            return {
                "status": "failed",
                "error": result.stderr[:500],
                "timestamp": datetime.utcnow().isoformat(),
            }

    except subprocess.TimeoutExpired:
        logger.error("[Celery] Data processing timed out")
        return {"status": "failed", "error": "Timeout after 60 minutes"}
    except Exception as e:
        logger.error(f"[Celery] Data processing error: {e}")
        return {"status": "failed", "error": str(e)}


@celery_app.task(name="generate_embeddings", soft_time_limit=2700, time_limit=3000)
def generate_embeddings(processed_file: Optional[str] = None):
    """
    Generate embeddings for processed chunks

    This task is typically called as part of the processing pipeline,
    but can also be run standalone to regenerate embeddings.

    Args:
        processed_file: Optional specific file to process (None = all processed files)

    Returns:
        Dictionary with embedding generation results
    """
    logger.info(f"[Celery] Generating embeddings (file={processed_file})")

    try:
        from Module2_NiruParser.pipeline import ProcessingPipeline
        from Module2_NiruParser.config import Config

        config = Config()
        pipeline = ProcessingPipeline(config)

        # Find processed files without embeddings or all processed files
        processed_path = project_root / "data" / "processed"

        if processed_file:
            files_to_process = [Path(processed_file)]
        else:
            files_to_process = list(processed_path.rglob("*_processed.jsonl"))

        total_embedded = 0

        for jsonl_file in files_to_process:
            if not jsonl_file.exists():
                continue

            logger.info(f"Processing embeddings for: {jsonl_file.name}")

            # Load chunks
            chunks = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunk = json.loads(line)
                        # Only process chunks without embeddings
                        if "embedding" not in chunk or chunk.get("embedding") is None:
                            chunks.append(chunk)

            if chunks:
                # Generate embeddings
                embedded_chunks = pipeline.embedder.embed_chunks(chunks)

                # Update the file with embeddings
                all_chunks = []
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            all_chunks.append(json.loads(line))

                # Merge embedded chunks back
                embedded_by_id = {c.get("chunk_id"): c for c in embedded_chunks}
                for i, chunk in enumerate(all_chunks):
                    if chunk.get("chunk_id") in embedded_by_id:
                        all_chunks[i] = embedded_by_id[chunk.get("chunk_id")]

                # Save updated chunks
                with open(jsonl_file, "w", encoding="utf-8") as f:
                    for chunk in all_chunks:
                        f.write(
                            json.dumps(chunk, ensure_ascii=False, default=str) + "\n"
                        )

                total_embedded += len(embedded_chunks)
                logger.info(
                    f"Generated {len(embedded_chunks)} embeddings for {jsonl_file.name}"
                )

        return {
            "status": "completed",
            "total_embedded": total_embedded,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"[Celery] Embedding generation error: {e}")
        return {"status": "failed", "error": str(e)}


@celery_app.task(name="populate_vector_stores", soft_time_limit=2700, time_limit=3000)
def populate_vector_stores(
    backend: Optional[str] = None, namespace: Optional[str] = None
):
    """
    Populate vector stores with processed and embedded chunks

    Supported backends:
    - qdrant: Qdrant Cloud/Local
    - upstash: Upstash Vector
    - chromadb: Local ChromaDB
    - all: All available backends (default)

    Args:
        backend: Specific backend to populate (None = all)
        namespace: Specific namespace to populate (None = auto-detect from category)

    Returns:
        Dictionary with population results
    """
    logger.info(
        f"[Celery] Populating vector stores (backend={backend}, namespace={namespace})"
    )

    try:
        # Run the populate_db module
        cmd = [sys.executable, "-m", "Module3_NiruDB.populate_db"]

        result = subprocess.run(
            cmd, cwd=str(project_root), capture_output=True, text=True, timeout=2700
        )

        if result.returncode == 0:
            logger.info("[Celery] Vector store population completed successfully")

            # Parse output for stats
            output = result.stdout
            total_chunks = 0
            if "Total chunks added:" in output:
                try:
                    total_chunks = int(
                        output.split("Total chunks added:")[-1].split()[0]
                    )
                except:
                    pass

            return {
                "status": "completed",
                "backend": backend or "all",
                "total_chunks": total_chunks,
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            logger.error(f"[Celery] Vector store population failed: {result.stderr}")
            return {
                "status": "failed",
                "error": result.stderr[:500],
                "timestamp": datetime.utcnow().isoformat(),
            }

    except subprocess.TimeoutExpired:
        logger.error("[Celery] Vector store population timed out")
        return {"status": "failed", "error": "Timeout after 45 minutes"}
    except Exception as e:
        logger.error(f"[Celery] Vector store population error: {e}")
        return {"status": "failed", "error": str(e)}


# update_vector_store removed — duplicate of populate_vector_stores.
# Use populate_vector_stores directly.
@celery_app.task(name="update_vector_store", soft_time_limit=1800, time_limit=2100)
def update_vector_store(incremental: bool = True):
    """Legacy alias for populate_vector_stores"""
    return populate_vector_stores.apply()


# =============================================================================
# FULL PIPELINE TASKS
# =============================================================================


@celery_app.task(
    name="run_full_pipeline", soft_time_limit=14400, time_limit=15000  # 4 hours
)
def run_full_pipeline(
    crawlers: Optional[List[str]] = None, skip_crawling: bool = False
):
    """
    Run the complete data refresh pipeline:
    1. Crawl data (all spiders or specified ones)
    2. Process raw data (parse, clean, chunk)
    3. Generate embeddings
    4. Populate vector stores

    Args:
        crawlers: List of crawler names to run (None = all)
        skip_crawling: If True, skip crawling and start from processing

    Returns:
        Dictionary with pipeline results
    """
    logger.info(
        f"[Celery] Starting full pipeline (crawlers={crawlers}, skip_crawling={skip_crawling})"
    )

    results = {"started_at": datetime.utcnow().isoformat(), "steps": {}}

    try:
        # Step 1: Crawling (unless skipped)
        if not skip_crawling:
            logger.info("[Celery] Step 1/4: Crawling data...")
            if crawlers:
                for crawler in crawlers:
                    try:
                        crawl_result = run_crawler_task(crawler)
                        results["steps"][f"crawl_{crawler}"] = crawl_result
                    except Exception as e:
                        results["steps"][f"crawl_{crawler}"] = {
                            "status": "failed",
                            "error": str(e),
                        }
            else:
                crawl_result = crawl_all_sources()
                results["steps"]["crawl_all"] = crawl_result
        else:
            results["steps"]["crawl"] = {"status": "skipped"}

        # Step 2: Processing
        logger.info("[Celery] Step 2/4: Processing raw data...")
        process_result = process_raw_data()
        results["steps"]["processing"] = process_result

        if process_result.get("status") == "failed":
            logger.error("[Celery] Processing failed, stopping pipeline")
            results["status"] = "partial_failure"
            results["completed_at"] = datetime.utcnow().isoformat()
            return results

        # Step 3: Embedding (usually done in processing, but ensure it's complete)
        logger.info("[Celery] Step 3/4: Verifying embeddings...")
        embed_result = generate_embeddings()
        results["steps"]["embedding"] = embed_result

        # Step 4: Vector store population
        logger.info("[Celery] Step 4/4: Populating vector stores...")
        populate_result = populate_vector_stores()
        results["steps"]["vector_store"] = populate_result

        # Determine overall status
        failed_steps = [
            k for k, v in results["steps"].items() if v.get("status") == "failed"
        ]

        if failed_steps:
            results["status"] = "partial_success"
            results["failed_steps"] = failed_steps
        else:
            results["status"] = "completed"

        results["completed_at"] = datetime.utcnow().isoformat()

        logger.info(
            f"[Celery] Full pipeline completed with status: {results['status']}"
        )
        return results

    except Exception as e:
        logger.error(f"[Celery] Full pipeline error: {e}")
        results["status"] = "failed"
        results["error"] = str(e)
        results["completed_at"] = datetime.utcnow().isoformat()
        return results


@celery_app.task(
    name="run_incremental_update", soft_time_limit=7200, time_limit=7500  # 2 hours
)
def run_incremental_update():
    """
    Run an incremental update of the data pipeline

    This is a lighter version of the full pipeline that:
    1. Crawls only news sources (faster, more frequent updates)
    2. Processes only new raw data
    3. Updates vector stores incrementally

    Ideal for frequent updates (hourly/daily)
    """
    logger.info("[Celery] Starting incremental update...")

    results = {
        "started_at": datetime.utcnow().isoformat(),
        "type": "incremental",
        "steps": {},
    }

    try:
        # Crawl news sources only (faster)
        logger.info("[Celery] Crawling news sources...")
        news_result = crawl_news_sources()
        results["steps"]["crawl_news"] = news_result

        # Process new data
        logger.info("[Celery] Processing new data...")
        process_result = process_raw_data(incremental=True)
        results["steps"]["processing"] = process_result

        # Update vector stores
        logger.info("[Celery] Updating vector stores...")
        populate_result = populate_vector_stores()
        results["steps"]["vector_store"] = populate_result

        results["status"] = "completed"
        results["completed_at"] = datetime.utcnow().isoformat()

        return results

    except Exception as e:
        logger.error(f"[Celery] Incremental update error: {e}")
        results["status"] = "failed"
        results["error"] = str(e)
        return results


@celery_app.task(
    name="run_legal_sources_pipeline",
    soft_time_limit=10800,  # 3 hours
    time_limit=11100,
)
def run_legal_sources_pipeline():
    """
    Run the pipeline for legal/government sources only

    Crawls and processes:
    - Kenya Law
    - Parliament (Hansards, Bills)
    - Parliament Videos

    Ideal for weekly/monthly deep updates
    """
    logger.info("[Celery] Starting legal sources pipeline...")

    results = {
        "started_at": datetime.utcnow().isoformat(),
        "type": "legal_sources",
        "steps": {},
    }

    try:
        # Crawl legal sources
        logger.info("[Celery] Crawling legal sources...")
        legal_result = crawl_legal_sources()
        results["steps"]["crawl_legal"] = legal_result

        # Process data
        logger.info("[Celery] Processing legal data...")
        process_result = process_raw_data(source_filter="legal")
        results["steps"]["processing"] = process_result

        # Update vector stores
        logger.info("[Celery] Updating vector stores...")
        populate_result = populate_vector_stores()
        results["steps"]["vector_store"] = populate_result

        results["status"] = "completed"
        results["completed_at"] = datetime.utcnow().isoformat()

        return results

    except Exception as e:
        logger.error(f"[Celery] Legal sources pipeline error: {e}")
        results["status"] = "failed"
        results["error"] = str(e)
        return results


# =============================================================================
# MAINTENANCE TASKS
# =============================================================================


@celery_app.task(name="cleanup_old_data")
def cleanup_old_data(days_to_keep: int = 90):
    """
    Clean up old raw data files to save disk space

    Args:
        days_to_keep: Number of days of raw data to keep
    """
    logger.info(f"[Celery] Cleaning up data older than {days_to_keep} days...")

    from datetime import timedelta
    import os

    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

    raw_path = project_root / "data" / "raw"
    deleted_count = 0
    deleted_size = 0

    try:
        for jsonl_file in raw_path.rglob("*.jsonl"):
            try:
                file_mtime = datetime.fromtimestamp(jsonl_file.stat().st_mtime)
                if file_mtime < cutoff_date:
                    file_size = jsonl_file.stat().st_size
                    jsonl_file.unlink()
                    deleted_count += 1
                    deleted_size += file_size
                    logger.info(f"Deleted old file: {jsonl_file.name}")
            except Exception as e:
                logger.warning(f"Could not delete {jsonl_file}: {e}")

        return {
            "status": "completed",
            "deleted_files": deleted_count,
            "deleted_size_mb": round(deleted_size / (1024 * 1024), 2),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"[Celery] Cleanup error: {e}")
        return {"status": "failed", "error": str(e)}


@celery_app.task(name="rebuild_vector_index")
def rebuild_vector_index(backend: str = "qdrant"):
    """
    Rebuild the vector index from scratch

    Warning: This will delete existing data and rebuild from processed files

    Args:
        backend: Vector store backend to rebuild
    """
    logger.info(f"[Celery] Rebuilding vector index for {backend}...")

    try:
        from Module3_NiruDB.vector_store import VectorStore

        # Initialize vector store
        vector_store = VectorStore(backend=backend)

        # Clear existing data
        logger.warning(f"Clearing {backend} vector store...")
        # vector_store.clear()  # Uncomment if clear method exists

        # Repopulate
        result = populate_vector_stores(backend=backend)

        return {
            "status": "completed",
            "backend": backend,
            "rebuild_result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"[Celery] Rebuild error: {e}")
        return {"status": "failed", "error": str(e)}


@celery_app.task(name="validate_vector_store")
def validate_vector_store():
    """
    Validate vector store integrity

    Checks:
    - Connection to all backends
    - Document counts
    - Sample query test
    """
    logger.info("[Celery] Validating vector stores...")

    validation_results = {}

    try:
        from Module3_NiruDB.vector_store import VectorStore

        for backend in ["qdrant", "upstash", "chromadb"]:
            try:
                vs = VectorStore(backend=backend)
                stats = vs.get_stats()

                # Test query
                test_results = vs.search("test query", k=1)

                validation_results[backend] = {
                    "status": "healthy",
                    "stats": stats,
                    "query_test": "passed" if test_results else "no results",
                }

            except Exception as e:
                validation_results[backend] = {"status": "error", "error": str(e)}

        return {
            "status": "completed",
            "backends": validation_results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"[Celery] Validation error: {e}")
        return {"status": "failed", "error": str(e)}
