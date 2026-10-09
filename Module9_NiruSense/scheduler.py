"""
NiruSense Scheduler
Schedules periodic processing and semantic analysis tasks
"""

import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
from loguru import logger


class NiruSenseScheduler:
    """Scheduler for NiruSense processing tasks"""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.enabled = (
            os.getenv("ENABLE_NIRUSENSE_SCHEDULER", "false").lower() == "true"
        )

        # Configuration from environment
        self.batch_interval_minutes = int(os.getenv("NIRUSENSE_BATCH_INTERVAL", "30"))
        self.cleanup_hour = int(os.getenv("NIRUSENSE_CLEANUP_HOUR", "3"))
        self.metrics_interval_minutes = int(
            os.getenv("NIRUSENSE_METRICS_INTERVAL", "15")
        )

    def start(self):
        """Start the scheduler with configured jobs"""
        if not self.enabled:
            logger.info(
                "NiruSense scheduler disabled (set ENABLE_NIRUSENSE_SCHEDULER=true to enable)"
            )
            return False

        try:
            # Helper to run async functions in scheduler
            def run_async(coro):
                """Helper to run async coroutine in sync context"""
                import asyncio

                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()

            # Job 1: Process pending documents every N minutes
            self.scheduler.add_job(
                func=self._process_pending_batch,
                trigger=IntervalTrigger(minutes=self.batch_interval_minutes),
                id="nirusense_batch_processing",
                name="Process Pending Documents",
                replace_existing=True,
            )
            logger.info(
                f"[DATE] Scheduled: Batch processing every {self.batch_interval_minutes} minutes"
            )

            # Job 2: Cleanup old processed documents (daily at configured hour)
            self.scheduler.add_job(
                func=lambda: run_async(self._cleanup_old_documents()),
                trigger=CronTrigger(hour=self.cleanup_hour, minute=0),
                id="nirusense_cleanup",
                name="Cleanup Old Documents",
                replace_existing=True,
            )
            logger.info(
                f"[DATE] Scheduled: Cleanup daily at {self.cleanup_hour}:00 UTC"
            )

            # Job 3: Update metrics and health status
            self.scheduler.add_job(
                func=self._update_metrics,
                trigger=IntervalTrigger(minutes=self.metrics_interval_minutes),
                id="nirusense_metrics",
                name="Update Metrics",
                replace_existing=True,
            )
            logger.info(
                f"[DATE] Scheduled: Metrics update every {self.metrics_interval_minutes} minutes"
            )

            # Job 4: Re-index failed documents (weekly on Sunday at 2 AM)
            self.scheduler.add_job(
                func=lambda: run_async(self._reprocess_failed()),
                trigger=CronTrigger(day_of_week="sun", hour=2, minute=0),
                id="nirusense_reprocess_failed",
                name="Reprocess Failed Documents",
                replace_existing=True,
            )
            logger.info(
                "[DATE] Scheduled: Reprocess failed documents weekly (Sunday 2:00 AM UTC)"
            )

            self.scheduler.start()
            logger.info("[OK] NiruSense scheduler started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start NiruSense scheduler: {e}")
            return False

    def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("NiruSense scheduler stopped")

    def get_status(self):
        """Get scheduler status"""
        if not self.enabled:
            return {"running": False, "message": "Scheduler disabled"}

        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": (
                        job.next_run_time.isoformat() if job.next_run_time else None
                    ),
                }
            )

        return {
            "running": self.scheduler.running,
            "jobs": jobs,
            "config": {
                "batch_interval_minutes": self.batch_interval_minutes,
                "cleanup_hour": self.cleanup_hour,
                "metrics_interval_minutes": self.metrics_interval_minutes,
            },
        }

    def _process_pending_batch(self):
        """Process a batch of pending documents from the queue"""
        try:
            logger.info("[STATS] Running scheduled batch processing...")
            from Module9_NiruSense.processing.monitoring import metrics

            # Get current stats
            stats = metrics.get_metrics()
            pending = stats.get("documents_pending", 0)

            if pending > 0:
                logger.info(f"Found {pending} pending documents to process")
                # The orchestrator will automatically process these from Redis stream
            else:
                logger.debug("No pending documents in queue")

        except Exception as e:
            logger.error(f"Batch processing job failed: {e}")

    async def _cleanup_old_documents(self):
        """Cleanup old processed documents and logs"""
        try:
            logger.info("[CLEAN] Running scheduled cleanup...")
            import asyncio
            from Module9_NiruSense.processing.storage.postgres import postgres

            # Delete analysis results older than configured days
            cleanup_days = int(os.getenv("NIRUSENSE_CLEANUP_DAYS", "90"))

            # Run the async cleanup
            result = await postgres.cleanup_old_data(days=cleanup_days)

            if "error" in result:
                logger.error(f"Cleanup failed: {result['error']}")
            else:
                logger.info(
                    f"Cleanup complete: {result.get('documents_deleted', 0)} documents, "
                    f"{result.get('analysis_deleted', 0)} analysis results deleted "
                    f"(older than {cleanup_days} days)"
                )

        except Exception as e:
            logger.error(f"Cleanup job failed: {e}")

    def _update_metrics(self):
        """Update and log metrics"""
        try:
            from Module9_NiruSense.processing.monitoring import metrics

            stats = metrics.get_metrics()
            logger.info(
                f"📊 Metrics: {stats['documents_processed']} processed, "
                f"{stats['documents_failed']} failed, "
                f"{stats['success_rate']:.1f}% success rate"
            )

        except Exception as e:
            logger.error(f"Metrics update job failed: {e}")

    async def _reprocess_failed(self):
        """Reprocess failed documents from the database"""
        try:
            logger.info("[SYNC] Running scheduled reprocessing of failed documents...")
            import asyncio
            from Module9_NiruSense.processing.storage.postgres import postgres

            # Get failed documents (documents without analysis results)
            failed_docs = await postgres.get_failed_documents(limit=50)

            if not failed_docs:
                logger.info("No failed documents to reprocess")
                return

            logger.info(f"Found {len(failed_docs)} failed documents to reprocess")

            # Reprocess each document
            success_count = 0
            for doc_data in failed_docs:
                try:
                    # Re-process via pipeline directly
                    from Module9_NiruSense.processing.orchestrator import pipeline

                    result = await pipeline.process(doc_data)
                    if result.get("status") == "success":
                        success_count += 1
                except Exception as e:
                    logger.error(
                        f"Failed to reprocess document {doc_data.get('id')}: {e}"
                    )

            logger.info(
                f"Reprocessing complete: {success_count}/{len(failed_docs)} successful"
            )

        except Exception as e:
            logger.error(f"Reprocess job failed: {e}")


# Global scheduler instance
_nirusense_scheduler = None


def get_nirusense_scheduler():
    """Get or create the global scheduler instance"""
    global _nirusense_scheduler
    if _nirusense_scheduler is None:
        _nirusense_scheduler = NiruSenseScheduler()
    return _nirusense_scheduler
