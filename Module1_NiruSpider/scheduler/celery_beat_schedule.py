"""
Celery Beat Schedule Configuration

This defines the automatic scheduling for all crawlers and data pipelines
to keep the vector store up to date with new legal documents and news.

Schedule Overview:
- News RSS: Every 4 hours (breaking news)
- Global Trends: Every 6 hours (international news)
- Parliament: Every 12 hours (Hansards, Bills)
- Parliament Videos: Daily at 3 AM
- Kenya Law: Every 2 days at 2 AM (comprehensive legal database)

Data Pipeline:
- Incremental Update: Every 4 hours (process new data, update vectors)
- Full Pipeline: Weekly on Sunday (complete refresh)
- Processing: After each crawl batch
- Vector Store Validation: Daily at midnight

Maintenance:
- Cleanup old data: Weekly
- Health Check: Every 30 minutes
"""

from celery.schedules import crontab

# Schedule for periodic crawling and data processing
beat_schedule = {
    # ============== HIGH FREQUENCY - NEWS ==============
    # Crawl Kenyan news every 4 hours
    # Breaking news needs frequent updates
    "crawl-news-rss-4h": {
        "task": "run_crawler",
        "schedule": crontab(minute=0, hour="*/4"),  # Every 4 hours
        "args": ("news_rss",),
        "options": {"queue": "crawling"},
    },
    # Crawl global trends every 6 hours
    "crawl-global-trends-6h": {
        "task": "run_crawler",
        "schedule": crontab(minute=30, hour="*/6"),  # Every 6 hours, offset by 30 min
        "args": ("global_trends",),
        "options": {"queue": "crawling"},
    },
    # ============== MEDIUM FREQUENCY - PARLIAMENT ==============
    # Crawl Parliament (Hansards, Bills) every 12 hours
    "crawl-parliament-12h": {
        "task": "run_crawler",
        "schedule": crontab(minute=0, hour="6,18"),  # 6 AM and 6 PM
        "args": ("parliament",),
        "options": {"queue": "crawling"},
    },
    # ============== LOW FREQUENCY - LEGAL ==============
    # Crawl Parliament videos daily at 3 AM
    "crawl-parliament-videos-daily": {
        "task": "run_crawler",
        "schedule": crontab(minute=0, hour=3),  # Daily at 3 AM
        "args": ("parliament_videos",),
        "options": {"queue": "crawling"},
    },
    # Crawl Kenya Law every 2 days at 2 AM
    # This is a comprehensive crawl of the legal database
    "crawl-kenya-law-48h": {
        "task": "run_crawler",
        "schedule": crontab(minute=0, hour=2, day_of_week="0,2,4,6"),  # Every other day
        "args": ("kenya_law",),
        "options": {"queue": "crawling"},
    },
    # Crawl Constitution sources weekly at 4 AM on Monday
    "crawl-constitution-weekly": {
        "task": "run_crawler",
        "schedule": crontab(minute=0, hour=4, day_of_week="1"),
        "args": ("constitution",),
        "options": {"queue": "crawling"},
    },
    # ============== NEW SPIDERS ==============
    # Crawl Kenya Gazette daily at 5 AM
    "crawl-kenya-gazette-daily": {
        "task": "run_crawler",
        "schedule": crontab(minute=0, hour=5),
        "args": ("kenya_gazette",),
        "options": {"queue": "crawling"},
    },
    # Crawl Africa Check Kenya fact-checks daily at 6 AM
    "crawl-fact-check-daily": {
        "task": "run_crawler",
        "schedule": crontab(minute=0, hour=6),
        "args": ("fact_check",),
        "options": {"queue": "crawling"},
    },
    # Crawl The Conversation Africa analysis daily at 7 AM
    "crawl-africa-analysis-daily": {
        "task": "run_crawler",
        "schedule": crontab(minute=0, hour=7),
        "args": ("africa_analysis",),
        "options": {"queue": "crawling"},
    },
    # ============== DATA PROCESSING PIPELINE ==============
    # Process raw data every 4 hours (after news crawls complete)
    # This runs: text extraction -> cleaning -> chunking -> enrichment
    "process-raw-data-4h": {
        "task": "process_raw_data",
        "schedule": crontab(minute=30, hour="*/4"),  # 30 min after news crawl
        "kwargs": {"incremental": True},
        "options": {"queue": "crawling"},
    },
    # Generate/verify embeddings every 4 hours
    "generate-embeddings-4h": {
        "task": "generate_embeddings",
        "schedule": crontab(minute=45, hour="*/4"),  # 45 min after news crawl
        "options": {"queue": "crawling"},
    },
    # ============== VECTOR STORE UPDATES ==============
    # Populate vector stores every 4 hours (after processing completes)
    "populate-vector-stores-4h": {
        "task": "populate_vector_stores",
        "schedule": crontab(minute=55, hour="*/4"),  # 55 min after news crawl
        "options": {"queue": "crawling"},
    },
    # Validate vector store integrity daily at midnight
    "validate-vector-store-daily": {
        "task": "validate_vector_store",
        "schedule": crontab(minute=0, hour=0),  # Midnight
        "options": {"queue": "crawling"},
    },
    # ============== FULL PIPELINE RUNS ==============
    # Run incremental update every 6 hours
    # This is a coordinated crawl+process+embed+store operation for news
    "incremental-update-6h": {
        "task": "run_incremental_update",
        "schedule": crontab(minute=0, hour="1,7,13,19"),  # 1 AM, 7 AM, 1 PM, 7 PM
        "options": {"queue": "crawling"},
    },
    # Run legal sources pipeline weekly on Wednesday at 1 AM
    "legal-pipeline-weekly": {
        "task": "run_legal_sources_pipeline",
        "schedule": crontab(minute=0, hour=1, day_of_week="3"),  # Wednesday 1 AM
        "options": {"queue": "crawling"},
    },
    # Full pipeline refresh weekly on Sunday at 1 AM
    # This rebuilds everything from scratch
    "full-pipeline-weekly": {
        "task": "run_full_pipeline",
        "schedule": crontab(minute=0, hour=1, day_of_week="0"),  # Sunday 1 AM
        "kwargs": {"skip_crawling": False},
        "options": {"queue": "crawling"},
    },
    # ============== MAINTENANCE ==============
    # Clean up old raw data files weekly on Saturday at 4 AM
    "cleanup-old-data-weekly": {
        "task": "cleanup_old_data",
        "schedule": crontab(minute=0, hour=4, day_of_week="6"),  # Saturday 4 AM
        "kwargs": {"days_to_keep": 90},
        "options": {"queue": "celery"},
    },
    # Rebuild vector index monthly on the 1st at 2 AM
    "rebuild-vector-index-monthly": {
        "task": "rebuild_vector_index",
        "schedule": crontab(minute=0, hour=2, day_of_month="1"),  # 1st of month, 2 AM
        "kwargs": {"backend": "qdrant"},
        "options": {"queue": "crawling"},
    },
    # Health check every 30 minutes
    "scheduler-health-check": {
        "task": "scheduler_health_check",
        "schedule": crontab(minute="*/30"),  # Every 30 minutes
    },
}


# Alternative schedules for different environments

# Development schedule - more frequent for testing
beat_schedule_dev = {
    "crawl-news-rss-dev": {
        "task": "run_crawler",
        "schedule": crontab(minute="*/30"),  # Every 30 minutes
        "args": ("news_rss",),
    },
    "process-data-dev": {
        "task": "process_raw_data",
        "schedule": crontab(minute="*/45"),  # Every 45 minutes
        "kwargs": {"incremental": True},
    },
    "populate-vectors-dev": {
        "task": "populate_vector_stores",
        "schedule": crontab(minute="*/60"),  # Every hour
    },
    "health-check-dev": {
        "task": "scheduler_health_check",
        "schedule": crontab(minute="*/5"),  # Every 5 minutes
    },
}

# Minimal schedule - for low-resource environments
beat_schedule_minimal = {
    "crawl-news-daily": {
        "task": "crawl_news_sources",
        "schedule": crontab(minute=0, hour=8),  # Daily at 8 AM
    },
    "crawl-legal-weekly": {
        "task": "crawl_legal_sources",
        "schedule": crontab(minute=0, hour=2, day_of_week="0"),  # Sunday 2 AM
    },
    "process-data-daily": {
        "task": "process_raw_data",
        "schedule": crontab(minute=30, hour=8),  # Daily at 8:30 AM
    },
    "populate-vectors-daily": {
        "task": "populate_vector_stores",
        "schedule": crontab(minute=0, hour=9),  # Daily at 9 AM
    },
    "health-check-hourly": {
        "task": "scheduler_health_check",
        "schedule": crontab(minute=0),  # Every hour
    },
}

# High-frequency schedule - for production with resources
beat_schedule_production = {
    # News every 2 hours
    "crawl-news-2h": {
        "task": "run_crawler",
        "schedule": crontab(minute=0, hour="*/2"),
        "args": ("news_rss",),
    },
    # Global trends every 4 hours
    "crawl-global-4h": {
        "task": "run_crawler",
        "schedule": crontab(minute=15, hour="*/4"),
        "args": ("global_trends",),
    },
    # Parliament every 8 hours
    "crawl-parliament-8h": {
        "task": "run_crawler",
        "schedule": crontab(minute=0, hour="4,12,20"),
        "args": ("parliament",),
    },
    # Process data every 2 hours
    "process-data-2h": {
        "task": "process_raw_data",
        "schedule": crontab(minute=30, hour="*/2"),
        "kwargs": {"incremental": True},
    },
    # Update vectors every 2 hours
    "populate-vectors-2h": {
        "task": "populate_vector_stores",
        "schedule": crontab(minute=45, hour="*/2"),
    },
    # Kenya Law daily
    "crawl-kenya-law-daily": {
        "task": "run_crawler",
        "schedule": crontab(minute=0, hour=3),
        "args": ("kenya_law",),
    },
    # Full pipeline weekly
    "full-pipeline-weekly": {
        "task": "run_full_pipeline",
        "schedule": crontab(minute=0, hour=0, day_of_week="0"),
    },
    # Validation twice daily
    "validate-12h": {
        "task": "validate_vector_store",
        "schedule": crontab(minute=0, hour="0,12"),
    },
    # Health check every 15 minutes
    "health-check-15m": {
        "task": "scheduler_health_check",
        "schedule": crontab(minute="*/15"),
    },
}
