"""
Celery Application Configuration

This configures the Celery app for distributed crawler scheduling.
Requires Redis as the message broker.

Usage:
    # Start worker
    celery -A Module1_NiruSpider.scheduler.celery_app worker --loglevel=info -Q crawling

    # Start beat scheduler
    celery -A Module1_NiruSpider.scheduler.celery_app beat --loglevel=info

    # Start both (development)
    celery -A Module1_NiruSpider.scheduler.celery_app worker --beat --loglevel=info -Q crawling
"""

import os
from celery import Celery
from loguru import logger

# Get Redis URL from environment
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "niruspider",
    broker=redis_url,
    backend=redis_url,
    include=["Module1_NiruSpider.scheduler.celery_tasks"],
)

# Import beat schedule
from .celery_beat_schedule import beat_schedule

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task tracking
    task_track_started=True,
    result_extended=True,
    # Time limits (can be overridden per task)
    task_time_limit=7200,  # 2 hour max per task
    task_soft_time_limit=6900,  # 115 minutes soft limit
    # Worker settings
    worker_concurrency=4,  # Run 4 tasks concurrently
    worker_prefetch_multiplier=2,  # Prefetch 2 tasks per worker
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks (prevent memory leaks)
    worker_max_memory_per_child=512000,  # 512MB max per worker
    # Task acknowledgment
    task_acks_late=True,  # Acknowledge tasks after completion
    task_reject_on_worker_lost=True,  # Reject task if worker dies
    # Beat scheduler
    beat_schedule=beat_schedule,
    beat_scheduler="celery.beat:PersistentScheduler",
    beat_schedule_filename="celerybeat-schedule",
    # Result backend settings
    result_expires=86400,  # Results expire after 24 hours
    # Task routing
    task_routes={
        "run_crawler": {"queue": "crawling"},
        "crawl_news_sources": {"queue": "crawling"},
        "crawl_legal_sources": {"queue": "crawling"},
        "crawl_all_sources": {"queue": "crawling"},
        "update_vector_store": {"queue": "crawling"},
        "scheduler_health_check": {"queue": "celery"},
    },
    # Default queue
    task_default_queue="celery",
    # Logging
    worker_hijack_root_logger=False,
)

logger.info(
    f"Celery app configured with broker: {redis_url.split('@')[-1] if '@' in redis_url else redis_url}"
)
logger.info(f"Beat schedule loaded with {len(beat_schedule)} tasks")
