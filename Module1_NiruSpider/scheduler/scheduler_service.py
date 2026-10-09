"""
Scheduler Service for AmaniQuery Crawlers

This service provides automatic scheduling for all crawlers to keep the vector store
up to date with new data. It supports two backends:
1. Celery + Redis (production, distributed)
2. APScheduler (standalone, single-process)

Usage:
    # Start with Celery (requires Redis)
    python scheduler_service.py --backend celery

    # Start standalone (no Redis required)
    python scheduler_service.py --backend apscheduler

    # Start with custom config
    python scheduler_service.py --config scheduler_config.yaml
"""

import os
import sys
import signal
import threading
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

# Configure logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(
    log_dir / "scheduler_{time}.log", rotation="1 day", retention="7 days", level="INFO"
)


class CrawlerType(str, Enum):
    """Types of crawlers available"""

    KENYA_LAW = "kenya_law"
    PARLIAMENT = "parliament"
    PARLIAMENT_VIDEOS = "parliament_videos"
    NEWS_RSS = "news_rss"
    GLOBAL_TRENDS = "global_trends"


class PipelineTask(str, Enum):
    """Types of pipeline tasks"""

    PROCESS_DATA = "process_data"
    GENERATE_EMBEDDINGS = "generate_embeddings"
    POPULATE_VECTORS = "populate_vectors"
    FULL_PIPELINE = "full_pipeline"
    INCREMENTAL_UPDATE = "incremental_update"
    VALIDATE_VECTORS = "validate_vectors"
    CLEANUP_DATA = "cleanup_data"


@dataclass
class CrawlerSchedule:
    """Configuration for a crawler's schedule"""

    crawler_type: CrawlerType
    enabled: bool = True
    interval_hours: int = 6  # Run every N hours
    cron_expression: Optional[str] = None  # Alternative: use cron
    timeout_minutes: int = 60
    max_retries: int = 3
    retry_delay_minutes: int = 15
    priority: int = 1  # Lower = higher priority
    description: str = ""

    # Runtime state
    last_run: Optional[datetime] = None
    last_status: str = "never_run"
    consecutive_failures: int = 0


@dataclass
class PipelineSchedule:
    """Configuration for a pipeline task's schedule"""

    task_type: PipelineTask
    enabled: bool = True
    interval_hours: int = 4
    cron_expression: Optional[str] = None
    timeout_minutes: int = 60
    description: str = ""

    # Runtime state
    last_run: Optional[datetime] = None
    last_status: str = "never_run"


@dataclass
class SchedulerConfig:
    """Overall scheduler configuration"""

    # Crawler schedules
    schedules: Dict[str, CrawlerSchedule] = field(default_factory=dict)

    # Pipeline schedules
    pipeline_schedules: Dict[str, PipelineSchedule] = field(default_factory=dict)

    # General settings
    max_concurrent_crawlers: int = 2
    health_check_interval_minutes: int = 5
    vector_store_update_after_crawl: bool = True
    notify_on_failure: bool = True

    # Retry settings
    global_max_retries: int = 3
    backoff_multiplier: float = 2.0

    @classmethod
    def default(cls) -> "SchedulerConfig":
        """Create default configuration"""
        config = cls()

        # Crawler schedules
        config.schedules = {
            CrawlerType.NEWS_RSS.value: CrawlerSchedule(
                crawler_type=CrawlerType.NEWS_RSS,
                interval_hours=4,
                timeout_minutes=30,
                priority=1,
                description="Kenyan news from RSS feeds - high frequency for breaking news",
            ),
            CrawlerType.GLOBAL_TRENDS.value: CrawlerSchedule(
                crawler_type=CrawlerType.GLOBAL_TRENDS,
                interval_hours=6,
                timeout_minutes=30,
                priority=2,
                description="Global news and trends from RSS feeds",
            ),
            CrawlerType.PARLIAMENT.value: CrawlerSchedule(
                crawler_type=CrawlerType.PARLIAMENT,
                interval_hours=12,
                timeout_minutes=60,
                priority=3,
                description="Parliament Hansards and Bills - less frequent, large dataset",
            ),
            CrawlerType.PARLIAMENT_VIDEOS.value: CrawlerSchedule(
                crawler_type=CrawlerType.PARLIAMENT_VIDEOS,
                interval_hours=24,
                timeout_minutes=45,
                priority=4,
                description="YouTube Parliament videos - daily updates",
            ),
            CrawlerType.KENYA_LAW.value: CrawlerSchedule(
                crawler_type=CrawlerType.KENYA_LAW,
                interval_hours=48,
                timeout_minutes=90,
                priority=5,
                description="Kenya Law database - infrequent updates, comprehensive crawl",
            ),
        }

        # Pipeline schedules
        config.pipeline_schedules = {
            PipelineTask.PROCESS_DATA.value: PipelineSchedule(
                task_type=PipelineTask.PROCESS_DATA,
                interval_hours=4,
                timeout_minutes=60,
                description="Process raw crawled data - extract, clean, chunk, enrich",
            ),
            PipelineTask.GENERATE_EMBEDDINGS.value: PipelineSchedule(
                task_type=PipelineTask.GENERATE_EMBEDDINGS,
                interval_hours=4,
                timeout_minutes=45,
                description="Generate vector embeddings for processed chunks",
            ),
            PipelineTask.POPULATE_VECTORS.value: PipelineSchedule(
                task_type=PipelineTask.POPULATE_VECTORS,
                interval_hours=4,
                timeout_minutes=45,
                description="Populate Qdrant/ChromaDB/Upstash with embeddings",
            ),
            PipelineTask.INCREMENTAL_UPDATE.value: PipelineSchedule(
                task_type=PipelineTask.INCREMENTAL_UPDATE,
                interval_hours=6,
                timeout_minutes=120,
                description="Incremental update - crawl news, process, embed, store",
            ),
            PipelineTask.VALIDATE_VECTORS.value: PipelineSchedule(
                task_type=PipelineTask.VALIDATE_VECTORS,
                interval_hours=24,
                timeout_minutes=15,
                description="Validate vector store integrity",
            ),
            PipelineTask.CLEANUP_DATA.value: PipelineSchedule(
                task_type=PipelineTask.CLEANUP_DATA,
                enabled=False,  # Disabled by default, run manually or enable
                interval_hours=168,  # Weekly
                timeout_minutes=30,
                description="Clean up old raw data files",
            ),
            PipelineTask.FULL_PIPELINE.value: PipelineSchedule(
                task_type=PipelineTask.FULL_PIPELINE,
                enabled=False,  # Disabled by default - run via cron instead
                interval_hours=168,  # Weekly
                timeout_minutes=240,  # 4 hours
                description="Full pipeline refresh - all crawlers + processing + vectors",
            ),
        }

        return config

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "schedules": {
                name: {
                    "crawler_type": sched.crawler_type.value,
                    "enabled": sched.enabled,
                    "interval_hours": sched.interval_hours,
                    "cron_expression": sched.cron_expression,
                    "timeout_minutes": sched.timeout_minutes,
                    "max_retries": sched.max_retries,
                    "retry_delay_minutes": sched.retry_delay_minutes,
                    "priority": sched.priority,
                    "description": sched.description,
                }
                for name, sched in self.schedules.items()
            },
            "max_concurrent_crawlers": self.max_concurrent_crawlers,
            "health_check_interval_minutes": self.health_check_interval_minutes,
            "vector_store_update_after_crawl": self.vector_store_update_after_crawl,
            "notify_on_failure": self.notify_on_failure,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SchedulerConfig":
        """Create from dictionary"""
        config = cls()
        config.max_concurrent_crawlers = data.get("max_concurrent_crawlers", 2)
        config.health_check_interval_minutes = data.get(
            "health_check_interval_minutes", 5
        )
        config.vector_store_update_after_crawl = data.get(
            "vector_store_update_after_crawl", True
        )
        config.notify_on_failure = data.get("notify_on_failure", True)

        for name, sched_data in data.get("schedules", {}).items():
            config.schedules[name] = CrawlerSchedule(
                crawler_type=CrawlerType(sched_data["crawler_type"]),
                enabled=sched_data.get("enabled", True),
                interval_hours=sched_data.get("interval_hours", 6),
                cron_expression=sched_data.get("cron_expression"),
                timeout_minutes=sched_data.get("timeout_minutes", 60),
                max_retries=sched_data.get("max_retries", 3),
                retry_delay_minutes=sched_data.get("retry_delay_minutes", 15),
                priority=sched_data.get("priority", 1),
                description=sched_data.get("description", ""),
            )

        return config

    def save(self, path: Path):
        """Save configuration to file"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SchedulerConfig":
        """Load configuration from file"""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


class CrawlerRunner:
    """Runs individual crawlers with timeout and error handling"""

    # Map crawler types to spider script arguments
    SPIDER_MAPPING = {
        CrawlerType.KENYA_LAW: "kenya_law",
        CrawlerType.PARLIAMENT: "parliament",
        CrawlerType.PARLIAMENT_VIDEOS: "parliament_videos",
        CrawlerType.NEWS_RSS: "news_rss",
        CrawlerType.GLOBAL_TRENDS: "global_trends",
    }

    def __init__(self, spider_dir: Path = None):
        self.spider_dir = spider_dir or (project_root / "Module1_NiruSpider")
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def run_crawler(
        self,
        crawler_type: CrawlerType,
        timeout_seconds: int = 3600,
        callback: Optional[Callable[[str, bool, str], None]] = None,
    ) -> bool:
        """
        Run a crawler as a subprocess

        Args:
            crawler_type: Type of crawler to run
            timeout_seconds: Maximum time for the crawl
            callback: Optional callback(crawler_name, success, message)

        Returns:
            True if crawl succeeded, False otherwise
        """
        spider_name = self.SPIDER_MAPPING.get(crawler_type)
        if not spider_name:
            logger.error(f"Unknown crawler type: {crawler_type}")
            return False

        logger.info(
            f"Starting crawler: {crawler_type.value} (timeout: {timeout_seconds}s)"
        )

        try:
            # Build command
            cmd = [
                sys.executable,
                "crawl_spider.py",
                spider_name,
                "--timeout",
                str(timeout_seconds),
            ]

            # Start process
            process = subprocess.Popen(
                cmd,
                cwd=str(self.spider_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            with self._lock:
                self.running_processes[crawler_type.value] = process

            # Wait for completion with timeout
            try:
                stdout, _ = process.communicate(
                    timeout=timeout_seconds + 60
                )  # Extra buffer
                exit_code = process.returncode

                if exit_code == 0:
                    # Log success with last few lines of output
                    logger.info(f"Crawler {crawler_type.value} completed successfully")
                    if stdout:
                        last_lines = "\n".join(stdout.strip().split("\n")[-10:])
                        logger.debug(
                            f"Crawler {crawler_type.value} output (last 10 lines):\n{last_lines}"
                        )
                    if callback:
                        callback(crawler_type.value, True, "Completed successfully")
                    return True
                else:
                    # Log failure with full output for debugging
                    logger.error(
                        f"Crawler {crawler_type.value} failed with exit code {exit_code}"
                    )
                    if stdout:
                        logger.error(f"Crawler {crawler_type.value} output:\n{stdout}")
                    else:
                        logger.error(f"Crawler {crawler_type.value} produced no output")
                    if callback:
                        callback(crawler_type.value, False, f"Exit code: {exit_code}")
                    return False

            except subprocess.TimeoutExpired:
                logger.warning(
                    f"Crawler {crawler_type.value} timed out, killing process"
                )
                process.kill()
                process.wait()
                if callback:
                    callback(crawler_type.value, False, "Timeout")
                return False

        except Exception as e:
            logger.error(f"Error running crawler {crawler_type.value}: {e}")
            if callback:
                callback(crawler_type.value, False, str(e))
            return False

        finally:
            with self._lock:
                self.running_processes.pop(crawler_type.value, None)

    def stop_crawler(self, crawler_type: CrawlerType) -> bool:
        """Stop a running crawler"""
        with self._lock:
            process = self.running_processes.get(crawler_type.value)
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                self.running_processes.pop(crawler_type.value, None)
                logger.info(f"Stopped crawler: {crawler_type.value}")
                return True
        return False

    def stop_all(self):
        """Stop all running crawlers"""
        with self._lock:
            for name, process in list(self.running_processes.items()):
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")
            self.running_processes.clear()

    def get_running_crawlers(self) -> list:
        """Get list of currently running crawlers"""
        with self._lock:
            return list(self.running_processes.keys())


class APSchedulerBackend:
    """
    APScheduler-based scheduler backend
    Runs in a single process, no Redis required
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.runner = CrawlerRunner()
        self.scheduler = None
        self._shutdown_event = threading.Event()
        self._executor = None

    def _setup_scheduler(self):
        """Setup APScheduler"""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.executors.pool import ThreadPoolExecutor
            from apscheduler.jobstores.memory import MemoryJobStore

            jobstores = {"default": MemoryJobStore()}

            executors = {
                "default": ThreadPoolExecutor(
                    max_workers=self.config.max_concurrent_crawlers
                )
            }

            job_defaults = {
                "coalesce": True,  # Combine missed runs
                "max_instances": 1,  # Only one instance per job
                "misfire_grace_time": 3600,  # 1 hour grace period
            }

            self.scheduler = BackgroundScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone="UTC",
            )

            return True

        except ImportError:
            logger.error(
                "APScheduler not installed. Install with: pip install apscheduler"
            )
            return False

    def _run_crawler_job(self, crawler_type: CrawlerType, schedule: CrawlerSchedule):
        """Job function that runs a crawler"""
        if self._shutdown_event.is_set():
            return

        logger.info(f"[Scheduled] Running crawler: {crawler_type.value}")

        # Check if we're at max concurrent crawlers
        running = self.runner.get_running_crawlers()
        if len(running) >= self.config.max_concurrent_crawlers:
            logger.warning(
                f"Max concurrent crawlers reached ({len(running)}), skipping {crawler_type.value}"
            )
            return

        # Run the crawler
        timeout_seconds = schedule.timeout_minutes * 60

        def on_complete(name: str, success: bool, message: str):
            schedule.last_run = datetime.utcnow()
            if success:
                schedule.last_status = "success"
                schedule.consecutive_failures = 0

                # Trigger vector store update if configured
                if self.config.vector_store_update_after_crawl:
                    self._trigger_vector_store_update()
            else:
                schedule.last_status = f"failed: {message}"
                schedule.consecutive_failures += 1

                if self.config.notify_on_failure:
                    logger.error(f"Crawler {name} failed: {message}")

        success = self.runner.run_crawler(
            crawler_type, timeout_seconds=timeout_seconds, callback=on_complete
        )

        return success

    def _trigger_vector_store_update(self):
        """Trigger vector store update after successful crawl"""
        logger.info("Triggering vector store update...")
        try:
            # Import and run the population script
            update_script = project_root / "populate_qdrant.py"
            if update_script.exists():
                subprocess.Popen(
                    [sys.executable, str(update_script), "--incremental"],
                    cwd=str(project_root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("Vector store update triggered")
            else:
                logger.warning(
                    "populate_qdrant.py not found, skipping vector store update"
                )
        except Exception as e:
            logger.error(f"Error triggering vector store update: {e}")

    def _run_pipeline_task(self, task_type: PipelineTask, schedule: PipelineSchedule):
        """Job function that runs a pipeline task"""
        if self._shutdown_event.is_set():
            return

        logger.info(f"[Scheduled] Running pipeline task: {task_type.value}")

        timeout_seconds = schedule.timeout_minutes * 60

        try:
            if task_type == PipelineTask.PROCESS_DATA:
                success = self._run_process_data(timeout_seconds)
            elif task_type == PipelineTask.GENERATE_EMBEDDINGS:
                success = self._run_generate_embeddings(timeout_seconds)
            elif task_type == PipelineTask.POPULATE_VECTORS:
                success = self._run_populate_vectors(timeout_seconds)
            elif task_type == PipelineTask.INCREMENTAL_UPDATE:
                success = self._run_incremental_update(timeout_seconds)
            elif task_type == PipelineTask.FULL_PIPELINE:
                success = self._run_full_pipeline(timeout_seconds)
            elif task_type == PipelineTask.VALIDATE_VECTORS:
                success = self._run_validate_vectors(timeout_seconds)
            elif task_type == PipelineTask.CLEANUP_DATA:
                success = self._run_cleanup_data(timeout_seconds)
            else:
                logger.error(f"Unknown pipeline task: {task_type}")
                success = False

            schedule.last_run = datetime.utcnow()
            schedule.last_status = "success" if success else "failed"

            return success

        except Exception as e:
            logger.error(f"Error running pipeline task {task_type.value}: {e}")
            schedule.last_run = datetime.utcnow()
            schedule.last_status = f"error: {str(e)}"
            return False

    def _run_process_data(self, timeout_seconds: int) -> bool:
        """Run data processing pipeline"""
        logger.info("Running data processing pipeline...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "Module2_NiruParser.process_all"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

            if result.returncode == 0:
                logger.info("Data processing completed successfully")
                return True
            else:
                logger.error(f"Data processing failed: {result.stderr[:500]}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Data processing timed out")
            return False

    def _run_generate_embeddings(self, timeout_seconds: int) -> bool:
        """Run embedding generation"""
        logger.info("Running embedding generation...")
        try:
            # Use the processing pipeline's embedder
            from Module2_NiruParser.pipeline import ProcessingPipeline
            from Module2_NiruParser.config import Config

            config = Config()
            pipeline = ProcessingPipeline(config)

            processed_path = project_root / "data" / "processed"
            total_embedded = 0

            for jsonl_file in processed_path.rglob("*_processed.jsonl"):
                chunks = []
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            chunk = json.loads(line)
                            if (
                                "embedding" not in chunk
                                or chunk.get("embedding") is None
                            ):
                                chunks.append(chunk)

                if chunks:
                    embedded_chunks = pipeline.embedder.embed_chunks(chunks)
                    total_embedded += len(embedded_chunks)

                    # Update file with embeddings
                    all_chunks = []
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                all_chunks.append(json.loads(line))

                    embedded_by_id = {c.get("chunk_id"): c for c in embedded_chunks}
                    for i, chunk in enumerate(all_chunks):
                        if chunk.get("chunk_id") in embedded_by_id:
                            all_chunks[i] = embedded_by_id[chunk.get("chunk_id")]

                    with open(jsonl_file, "w", encoding="utf-8") as f:
                        for chunk in all_chunks:
                            f.write(
                                json.dumps(chunk, ensure_ascii=False, default=str)
                                + "\n"
                            )

            logger.info(
                f"Embedding generation completed: {total_embedded} embeddings generated"
            )
            return True

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return False

    def _run_populate_vectors(self, timeout_seconds: int) -> bool:
        """Run vector store population"""
        logger.info("Running vector store population...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "Module3_NiruDB.populate_db"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

            if result.returncode == 0:
                logger.info("Vector store population completed successfully")
                return True
            else:
                logger.error(f"Vector store population failed: {result.stderr[:500]}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Vector store population timed out")
            return False

    def _run_incremental_update(self, timeout_seconds: int) -> bool:
        """Run incremental update (news crawl + process + vectors)"""
        logger.info("Running incremental update...")

        success = True

        # 1. Crawl news sources
        for crawler_type in [CrawlerType.NEWS_RSS, CrawlerType.GLOBAL_TRENDS]:
            schedule = self.config.schedules.get(crawler_type.value)
            if schedule and schedule.enabled:
                if not self.runner.run_crawler(
                    crawler_type, timeout_seconds=schedule.timeout_minutes * 60
                ):
                    success = False

        # 2. Process data
        if not self._run_process_data(timeout_seconds // 3):
            success = False

        # 3. Populate vectors
        if not self._run_populate_vectors(timeout_seconds // 3):
            success = False

        return success

    def _run_full_pipeline(self, timeout_seconds: int) -> bool:
        """Run full pipeline (all crawlers + process + vectors)"""
        logger.info("Running full pipeline...")

        success = True
        time_per_stage = timeout_seconds // 4

        # 1. Run all crawlers
        for crawler_type in CrawlerType:
            schedule = self.config.schedules.get(crawler_type.value)
            if schedule and schedule.enabled:
                if not self.runner.run_crawler(
                    crawler_type, timeout_seconds=schedule.timeout_minutes * 60
                ):
                    logger.warning(
                        f"Crawler {crawler_type.value} failed, continuing..."
                    )

        # 2. Process data
        if not self._run_process_data(time_per_stage):
            success = False

        # 3. Generate embeddings
        if not self._run_generate_embeddings(time_per_stage):
            success = False

        # 4. Populate vectors
        if not self._run_populate_vectors(time_per_stage):
            success = False

        return success

    def _run_validate_vectors(self, timeout_seconds: int) -> bool:
        """Validate vector store integrity"""
        logger.info("Validating vector stores...")
        try:
            from Module3_NiruDB.vector_store import VectorStore

            backends = ["qdrant", "chromadb"]  # Skip upstash if not configured

            for backend in backends:
                try:
                    vs = VectorStore(backend=backend)
                    stats = vs.get_stats()

                    # Test query
                    results = vs.search("test query", k=1)

                    logger.info(
                        f"{backend}: {stats.get('total_chunks', 'unknown')} chunks, query test {'passed' if results else 'no results'}"
                    )

                except Exception as e:
                    logger.warning(f"{backend}: validation failed - {e}")

            return True

        except Exception as e:
            logger.error(f"Vector store validation failed: {e}")
            return False

    def _run_cleanup_data(self, timeout_seconds: int) -> bool:
        """Clean up old raw data files"""
        logger.info("Cleaning up old data files...")
        try:
            from datetime import timedelta

            days_to_keep = 90
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            raw_path = project_root / "data" / "raw"
            deleted_count = 0

            for jsonl_file in raw_path.rglob("*.jsonl"):
                try:
                    file_mtime = datetime.fromtimestamp(jsonl_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        jsonl_file.unlink()
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete {jsonl_file}: {e}")

            logger.info(f"Cleanup completed: deleted {deleted_count} old files")
            return True

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False

    def start(self):
        """Start the scheduler"""
        if not self._setup_scheduler():
            return False

        logger.info("Starting APScheduler backend...")

        # Add jobs for each enabled crawler
        for name, schedule in self.config.schedules.items():
            if not schedule.enabled:
                logger.info(f"Skipping disabled crawler: {name}")
                continue

            crawler_type = schedule.crawler_type

            if schedule.cron_expression:
                # Use cron expression
                from apscheduler.triggers.cron import CronTrigger

                trigger = CronTrigger.from_crontab(schedule.cron_expression)
            else:
                # Use interval
                from apscheduler.triggers.interval import IntervalTrigger

                trigger = IntervalTrigger(hours=schedule.interval_hours)

            self.scheduler.add_job(
                self._run_crawler_job,
                trigger=trigger,
                args=[crawler_type, schedule],
                id=f"crawler_{name}",
                name=f"Crawl {name}",
                replace_existing=True,
            )

            logger.info(f"Scheduled crawler: {name} (every {schedule.interval_hours}h)")

        # Add jobs for each enabled pipeline task
        for name, schedule in self.config.pipeline_schedules.items():
            if not schedule.enabled:
                logger.info(f"Skipping disabled pipeline task: {name}")
                continue

            task_type = schedule.task_type

            if schedule.cron_expression:
                from apscheduler.triggers.cron import CronTrigger

                trigger = CronTrigger.from_crontab(schedule.cron_expression)
            else:
                from apscheduler.triggers.interval import IntervalTrigger

                trigger = IntervalTrigger(hours=schedule.interval_hours)

            self.scheduler.add_job(
                self._run_pipeline_task,
                trigger=trigger,
                args=[task_type, schedule],
                id=f"pipeline_{name}",
                name=f"Pipeline: {name}",
                replace_existing=True,
            )

            logger.info(
                f"Scheduled pipeline task: {name} (every {schedule.interval_hours}h)"
            )

        # Add health check job
        from apscheduler.triggers.interval import IntervalTrigger

        self.scheduler.add_job(
            self._health_check,
            trigger=IntervalTrigger(minutes=self.config.health_check_interval_minutes),
            id="health_check",
            name="Health Check",
            replace_existing=True,
        )

        # Start scheduler
        self.scheduler.start()
        logger.info("APScheduler started successfully")

        # Run initial crawl for high-priority items
        self._run_initial_crawls()

        return True

    def _run_initial_crawls(self):
        """Run initial crawls for items that haven't run recently"""
        logger.info("Checking for initial crawls needed...")

        # Sort by priority
        sorted_schedules = sorted(
            [
                (name, sched)
                for name, sched in self.config.schedules.items()
                if sched.enabled
            ],
            key=lambda x: x[1].priority,
        )

        for name, schedule in sorted_schedules[
            :2
        ]:  # Run top 2 priority crawlers initially
            if schedule.last_run is None:
                logger.info(f"Triggering initial crawl for: {name}")
                threading.Thread(
                    target=self._run_crawler_job,
                    args=[schedule.crawler_type, schedule],
                    daemon=True,
                ).start()
                time.sleep(5)  # Stagger starts

    def _health_check(self):
        """Periodic health check"""
        running = self.runner.get_running_crawlers()
        jobs = self.scheduler.get_jobs()

        logger.debug(
            f"Health check: {len(running)} crawlers running, {len(jobs)} jobs scheduled"
        )

        # Log status of each crawler
        for name, schedule in self.config.schedules.items():
            if schedule.enabled:
                status = "running" if name in running else schedule.last_status
                last_run = (
                    schedule.last_run.isoformat() if schedule.last_run else "never"
                )
                logger.debug(f"  {name}: {status} (last: {last_run})")

    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping APScheduler backend...")
        self._shutdown_event.set()

        if self.scheduler:
            self.scheduler.shutdown(wait=False)

        self.runner.stop_all()
        logger.info("APScheduler stopped")

    def get_status(self) -> dict:
        """Get current scheduler status"""
        running = self.runner.get_running_crawlers()

        schedules_status = {}
        for name, schedule in self.config.schedules.items():
            schedules_status[name] = {
                "enabled": schedule.enabled,
                "running": name in running,
                "last_run": (
                    schedule.last_run.isoformat() if schedule.last_run else None
                ),
                "last_status": schedule.last_status,
                "consecutive_failures": schedule.consecutive_failures,
                "interval_hours": schedule.interval_hours,
            }

        return {
            "backend": "apscheduler",
            "running": self.scheduler.running if self.scheduler else False,
            "running_crawlers": running,
            "schedules": schedules_status,
        }

    def trigger_crawler(self, crawler_type: CrawlerType) -> bool:
        """Manually trigger a crawler"""
        schedule = self.config.schedules.get(crawler_type.value)
        if not schedule:
            logger.error(f"Unknown crawler: {crawler_type}")
            return False

        logger.info(f"Manually triggering crawler: {crawler_type.value}")
        threading.Thread(
            target=self._run_crawler_job, args=[crawler_type, schedule], daemon=True
        ).start()
        return True


class CeleryBackend:
    """
    Celery-based scheduler backend
    Requires Redis, supports distributed crawling
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._celery_beat_process = None
        self._celery_worker_process = None

    def start(self):
        """Start Celery Beat and Worker"""
        logger.info("Starting Celery backend...")

        try:
            # Start Celery Beat (scheduler)
            self._celery_beat_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "celery",
                    "-A",
                    "Module1_NiruSpider.scheduler.celery_app",
                    "beat",
                    "--loglevel=info",
                ],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            # Start Celery Worker
            self._celery_worker_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "celery",
                    "-A",
                    "Module1_NiruSpider.scheduler.celery_app",
                    "worker",
                    "--loglevel=info",
                    "-Q",
                    "crawling",
                    "--concurrency",
                    str(self.config.max_concurrent_crawlers),
                ],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            logger.info("Celery Beat and Worker started")
            return True

        except Exception as e:
            logger.error(f"Error starting Celery: {e}")
            self.stop()
            return False

    def stop(self):
        """Stop Celery processes"""
        logger.info("Stopping Celery backend...")

        for process in [self._celery_beat_process, self._celery_worker_process]:
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception as e:
                    logger.error(f"Error stopping Celery process: {e}")

        self._celery_beat_process = None
        self._celery_worker_process = None
        logger.info("Celery stopped")

    def get_status(self) -> dict:
        """Get Celery status"""
        beat_running = (
            self._celery_beat_process and self._celery_beat_process.poll() is None
        )
        worker_running = (
            self._celery_worker_process and self._celery_worker_process.poll() is None
        )

        return {
            "backend": "celery",
            "beat_running": beat_running,
            "worker_running": worker_running,
            "running": beat_running and worker_running,
        }

    def trigger_crawler(self, crawler_type: CrawlerType) -> bool:
        """Trigger a crawler via Celery"""
        try:
            from .celery_tasks import run_crawler_task

            run_crawler_task.delay(crawler_type.value)
            return True
        except Exception as e:
            logger.error(f"Error triggering crawler via Celery: {e}")
            return False


class SchedulerService:
    """
    Main scheduler service that manages crawler scheduling
    """

    def __init__(self, config: SchedulerConfig = None, backend: str = "apscheduler"):
        self.config = config or SchedulerConfig.default()
        self.backend_type = backend
        self.backend = None
        self._shutdown_event = threading.Event()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self._shutdown_event.set()
        self.stop()

    def start(self):
        """Start the scheduler service"""
        logger.info(f"Starting Scheduler Service with {self.backend_type} backend")

        if self.backend_type == "celery":
            self.backend = CeleryBackend(self.config)
        else:
            self.backend = APSchedulerBackend(self.config)

        if not self.backend.start():
            logger.error("Failed to start scheduler backend")
            return False

        logger.info("Scheduler Service started successfully")
        return True

    def stop(self):
        """Stop the scheduler service"""
        logger.info("Stopping Scheduler Service...")
        if self.backend:
            self.backend.stop()
        logger.info("Scheduler Service stopped")

    def run_forever(self):
        """Run the scheduler until shutdown"""
        if not self.start():
            return

        logger.info("Scheduler running. Press Ctrl+C to stop.")

        try:
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=60)

                # Periodic status log
                if not self._shutdown_event.is_set():
                    status = self.get_status()
                    logger.info(
                        f"Scheduler status: {status.get('running_crawlers', [])} running"
                    )

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def get_status(self) -> dict:
        """Get scheduler status"""
        if self.backend:
            return self.backend.get_status()
        return {"running": False}

    def trigger_crawler(self, crawler_type: CrawlerType) -> bool:
        """Manually trigger a crawler"""
        if self.backend:
            return self.backend.trigger_crawler(crawler_type)
        return False

    def update_schedule(self, crawler_type: CrawlerType, **kwargs):
        """Update a crawler's schedule"""
        name = crawler_type.value
        if name in self.config.schedules:
            schedule = self.config.schedules[name]
            for key, value in kwargs.items():
                if hasattr(schedule, key):
                    setattr(schedule, key, value)
            logger.info(f"Updated schedule for {name}: {kwargs}")
            return True
        return False


def main():
    """Main entry point for the scheduler service"""
    import argparse

    parser = argparse.ArgumentParser(description="AmaniQuery Crawler Scheduler Service")
    parser.add_argument(
        "--backend",
        choices=["apscheduler", "celery"],
        default="apscheduler",
        help="Scheduler backend to use (default: apscheduler)",
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--save-default-config",
        type=str,
        help="Save default configuration to file and exit",
    )
    parser.add_argument(
        "--status", action="store_true", help="Print scheduler status and exit"
    )

    args = parser.parse_args()

    # Save default config if requested
    if args.save_default_config:
        config = SchedulerConfig.default()
        config.save(Path(args.save_default_config))
        print(f"Default configuration saved to: {args.save_default_config}")
        return

    # Load configuration
    if args.config:
        config = SchedulerConfig.load(Path(args.config))
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        config = SchedulerConfig.default()
        logger.info("Using default configuration")

    # Create and run service
    service = SchedulerService(config=config, backend=args.backend)

    if args.status:
        if service.start():
            time.sleep(2)
            print(json.dumps(service.get_status(), indent=2))
            service.stop()
    else:
        service.run_forever()


if __name__ == "__main__":
    main()
