"""
CrawlerManager - Manages crawler processes and status
Extracted from api.py for modularity
Enhanced with robust zombie detection and process management
"""

import os
import sys
import json
import subprocess
import threading
import time
import signal
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
from fastapi import HTTPException

from Module4_NiruAPI.crawler_models import CrawlerDatabaseManager


class CrawlerManager:
    """Manager for crawler processes with database-backed status tracking

    Features:
    - Database-backed status persistence
    - Zombie process detection and cleanup
    - Process timeout monitoring
    - Graceful shutdown with SIGTERM/SIGINT
    - Automatic restart on failure (optional)
    """

    # Default timeout for spider processes (60 minutes)
    DEFAULT_PROCESS_TIMEOUT = 60 * 60  # seconds

    # Maximum consecutive failures before marking as problematic
    MAX_CONSECUTIVE_FAILURES = 3

    # Zombie check interval
    ZOMBIE_CHECK_INTERVAL = 30  # seconds

    def __init__(self, database_storage=None):
        self.crawlers = {}
        self.processes = {}
        self.logs = {}
        self.failure_counts = {}  # Track consecutive failures per crawler
        self.status_file = Path(__file__).parent / "crawler_status.json"
        self.database_storage = database_storage
        self._shutdown_requested = False

        # Process timeouts (can be customized per crawler)
        self.process_timeouts = {
            "kenya_law": 90 * 60,  # 90 minutes (large dataset)
            "parliament": 60 * 60,  # 60 minutes
            "news_rss": 30 * 60,  # 30 minutes (smaller dataset)
            "global_trends": 30 * 60,  # 30 minutes
            "parliament_videos": 45 * 60,  # 45 minutes
        }

        # Initialize database manager
        try:
            self.db_manager = CrawlerDatabaseManager()
            if self.db_manager._initialized:
                logger.info("Using PostgreSQL for crawler status storage")
            else:
                logger.warning("PostgreSQL not available, using in-memory storage")
        except Exception as e:
            logger.warning(f"Failed to initialize crawler database manager: {e}")
            self.db_manager = None

        # Load status from database or migrate from file
        self.load_status()

        # Clean up any zombie processes from previous runs
        self._cleanup_zombies_on_startup()

        # Start background status checker
        self.status_thread = threading.Thread(target=self._status_checker, daemon=True)
        self.status_thread.start()

        # Start zombie monitor thread
        self.zombie_thread = threading.Thread(target=self._zombie_monitor, daemon=True)
        self.zombie_thread.start()

    def set_database_storage(self, database_storage):
        """Set database storage for last run times lookup"""
        self.database_storage = database_storage

    def shutdown(self):
        """Graceful shutdown - stop all running processes"""
        logger.info("CrawlerManager shutdown requested")
        self._shutdown_requested = True

        # Stop all running crawlers
        for crawler_name in list(self.processes.keys()):
            try:
                self.stop_crawler(crawler_name)
            except Exception as e:
                logger.error(f"Error stopping {crawler_name} during shutdown: {e}")

    def _cleanup_zombies_on_startup(self):
        """Clean up zombie processes from previous runs on startup"""
        logger.info("Checking for zombie processes from previous runs...")

        for name, status in self.crawlers.items():
            if status.get("status") == "running":
                pid = status.get("pid")
                is_alive = False

                if pid:
                    try:
                        if psutil.pid_exists(pid):
                            proc = psutil.Process(pid)
                            # Check if it's actually a Python process running our spider
                            if "python" in proc.name().lower():
                                cmdline = " ".join(proc.cmdline())
                                if "crawl_spider" in cmdline or name in cmdline:
                                    is_alive = True
                                    logger.info(
                                        f"Found active crawler process {name} (PID: {pid})"
                                    )
                    except (
                        psutil.NoSuchProcess,
                        psutil.AccessDenied,
                        psutil.ZombieProcess,
                    ):
                        pass
                    except Exception as e:
                        logger.warning(f"Error checking PID {pid} for {name}: {e}")

                if not is_alive:
                    logger.warning(
                        f"Detected zombie crawler {name} (PID: {pid}). Marking as failed."
                    )
                    self.crawlers[name]["status"] = "failed"
                    self.crawlers[name]["pid"] = None
                    self.crawlers[name]["start_time"] = None
                    self._add_log(
                        name,
                        f"System startup: Marked stale process (PID: {pid}) as failed",
                    )

                    if self.db_manager and self.db_manager._initialized:
                        try:
                            self.db_manager.update_crawler_status(
                                name, "failed", pid=None
                            )
                        except Exception:
                            pass

        self.save_status()

    def _zombie_monitor(self):
        """Background thread to detect and clean up zombie processes"""
        while not self._shutdown_requested:
            try:
                self._check_for_zombies()
                self._check_process_timeouts()
            except Exception as e:
                logger.error(f"Error in zombie monitor: {e}")

            time.sleep(self.ZOMBIE_CHECK_INTERVAL)

    def _check_for_zombies(self):
        """Check for zombie processes that are marked as running but aren't"""
        for name, status in list(self.crawlers.items()):
            if status.get("status") != "running":
                continue

            pid = status.get("pid")

            # If we have this process in our tracking, skip (handled by status_checker)
            if name in self.processes:
                continue

            # Check if PID is actually alive
            is_alive = False
            if pid:
                try:
                    if psutil.pid_exists(pid):
                        proc = psutil.Process(pid)
                        if proc.status() != psutil.STATUS_ZOMBIE:
                            is_alive = True
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    pass
                except Exception:
                    pass

            if not is_alive:
                logger.warning(
                    f"Zombie detected: {name} (PID: {pid}) - Process not found"
                )
                self.crawlers[name]["status"] = "failed"
                self.crawlers[name]["pid"] = None
                self.crawlers[name]["start_time"] = None
                self._add_log(
                    name, f"Zombie cleanup: Process (PID: {pid}) no longer exists"
                )
                self.save_status()

    def _check_process_timeouts(self):
        """Check if any running processes have exceeded their timeout"""
        current_time = datetime.utcnow()

        for name, process_info in list(self.processes.items()):
            start_time_str = process_info.get("start_time")
            if not start_time_str:
                continue

            try:
                # Parse start time
                start_time_str = start_time_str.rstrip("Z")
                start_time = datetime.fromisoformat(start_time_str)

                # Get timeout for this crawler
                timeout = self.process_timeouts.get(name, self.DEFAULT_PROCESS_TIMEOUT)

                # Check if timeout exceeded
                elapsed = (current_time - start_time).total_seconds()
                if elapsed > timeout:
                    logger.warning(
                        f"Process timeout: {name} has been running for {elapsed/60:.1f} minutes (limit: {timeout/60:.1f})"
                    )
                    self._add_log(
                        name,
                        f"Process timeout exceeded ({elapsed/60:.1f} min > {timeout/60:.1f} min). Force stopping...",
                    )

                    # Force stop the process
                    try:
                        self._force_stop_process(name)
                    except Exception as e:
                        logger.error(f"Error force stopping {name}: {e}")

            except Exception as e:
                logger.warning(f"Error checking timeout for {name}: {e}")

    def _force_stop_process(self, crawler_name: str):
        """Force stop a process and all its children"""
        if crawler_name not in self.processes:
            return

        process_info = self.processes[crawler_name]
        pid = process_info.get("pid")
        process = process_info.get("process")

        try:
            # Try to kill the process tree (parent and children)
            if pid and psutil.pid_exists(pid):
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)

                # Send SIGTERM to children first
                for child in children:
                    try:
                        child.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Send SIGTERM to parent
                try:
                    parent.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

                # Wait a bit
                time.sleep(2)

                # Force kill any survivors
                for child in children:
                    try:
                        if child.is_running():
                            child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                try:
                    if parent.is_running():
                        parent.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            elif process:
                # Fallback to subprocess methods
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error in _force_stop_process for {crawler_name}: {e}")

        finally:
            # Clean up tracking
            if crawler_name in self.processes:
                del self.processes[crawler_name]

            # Update failure count
            self.failure_counts[crawler_name] = (
                self.failure_counts.get(crawler_name, 0) + 1
            )

            self.crawlers[crawler_name]["status"] = "failed"
            self.crawlers[crawler_name]["pid"] = None
            self.crawlers[crawler_name]["start_time"] = None
            self.crawlers[crawler_name]["last_run"] = (
                datetime.utcnow().isoformat() + "Z"
            )
            self._add_log(crawler_name, f"Process force stopped (PID: {pid})")
            self.save_status()

    def load_status(self):
        """Load crawler status from database or migrate from file"""
        # Initialize default crawlers
        default_crawlers = {
            "kenya_law": {"status": "idle", "last_run": None},
            "parliament": {"status": "idle", "last_run": None},
            "news_rss": {"status": "idle", "last_run": None},
            "global_trends": {"status": "idle", "last_run": None},
            "parliament_videos": {"status": "idle", "last_run": None},
        }

        if self.db_manager and self.db_manager._initialized:
            # Load from database
            try:
                db_statuses = self.db_manager.get_crawler_status()
                for name, default_status in default_crawlers.items():
                    if name in db_statuses:
                        self.crawlers[name] = db_statuses[name]
                    else:
                        # Initialize new crawler in database
                        self.crawlers[name] = default_status
                        self.db_manager.update_crawler_status(
                            name,
                            default_status["status"],
                        )

                    # Initialize failure counts
                    self.failure_counts[name] = 0

                # Load logs from database
                for name in self.crawlers.keys():
                    self.logs[name] = self.db_manager.get_logs(name, limit=100)

                logger.info("Loaded crawler status from PostgreSQL database")

                # Migrate from file if it exists (one-time migration)
                if self.status_file.exists():
                    self._migrate_from_file()

            except Exception as e:
                logger.error(f"Error loading crawler status from database: {e}")
                self.crawlers = default_crawlers.copy()
                self.logs = {name: [] for name in default_crawlers.keys()}
                self.failure_counts = {name: 0 for name in default_crawlers.keys()}
        else:
            # Fallback to file-based storage
            try:
                if self.status_file.exists():
                    with open(self.status_file, "r") as f:
                        data = json.load(f)
                        self.crawlers = data.get("crawlers", default_crawlers)
                        self.logs = data.get(
                            "logs", {name: [] for name in default_crawlers.keys()}
                        )
                else:
                    self.crawlers = default_crawlers.copy()
                    self.logs = {name: [] for name in default_crawlers.keys()}

                self.failure_counts = {name: 0 for name in default_crawlers.keys()}
            except Exception as e:
                logger.error(f"Error loading crawler status from file: {e}")
                self.crawlers = default_crawlers.copy()
                self.logs = {name: [] for name in default_crawlers.keys()}
                self.failure_counts = {name: 0 for name in default_crawlers.keys()}

    def _migrate_from_file(self):
        """Migrate data from JSON file to database (one-time operation)"""
        try:
            if not self.status_file.exists():
                return

            logger.info("Migrating crawler status from JSON file to database...")
            with open(self.status_file, "r") as f:
                data = json.load(f)
                file_crawlers = data.get("crawlers", {})
                file_logs = data.get("logs", {})

            # Migrate crawler statuses
            for name, status in file_crawlers.items():
                last_run = None
                if status.get("last_run"):
                    try:
                        # Try parsing ISO format datetime string
                        last_run_str = status["last_run"]
                        if isinstance(last_run_str, str):
                            # Remove 'Z' suffix if present and parse
                            if last_run_str.endswith("Z"):
                                last_run_str = last_run_str[:-1] + "+00:00"
                            last_run = datetime.fromisoformat(
                                last_run_str.replace("Z", "")
                            )
                    except Exception:
                        pass

                self.db_manager.update_crawler_status(
                    name,
                    status.get("status", "idle"),
                    last_run=last_run,
                    pid=status.get("pid"),
                    start_time=None,
                )

            # Migrate logs
            for name, logs in file_logs.items():
                for log_entry in logs:
                    # Parse timestamp from log entry
                    try:
                        if log_entry.startswith("["):
                            timestamp_str = log_entry.split("]")[0][1:]
                            message = log_entry.split("]", 1)[1].strip()
                            try:
                                # Try parsing ISO format
                                if timestamp_str.endswith("Z"):
                                    timestamp_str = timestamp_str[:-1] + "+00:00"
                                timestamp = datetime.fromisoformat(
                                    timestamp_str.replace("Z", "")
                                )
                            except Exception:
                                # Fallback to current time if parsing fails
                                timestamp = datetime.utcnow()
                            self.db_manager.add_log(name, message, timestamp)
                        else:
                            self.db_manager.add_log(name, log_entry)
                    except Exception as e:
                        logger.warning(f"Error migrating log entry: {e}")
                        self.db_manager.add_log(name, log_entry)

            # Backup and remove old file
            backup_file = self.status_file.with_suffix(".json.backup")
            if not backup_file.exists():
                import shutil

                shutil.copy2(self.status_file, backup_file)
                logger.info(f"Backed up old status file to {backup_file}")

            logger.info("Migration completed successfully")
        except Exception as e:
            logger.error(f"Error migrating from file to database: {e}")

    def save_status(self):
        """Save crawler status to database"""
        if self.db_manager and self.db_manager._initialized:
            # Save to database
            try:
                for name, status in self.crawlers.items():
                    last_run = None
                    if status.get("last_run"):
                        try:
                            last_run_str = status["last_run"]
                            if isinstance(last_run_str, str):
                                if last_run_str.endswith("Z"):
                                    last_run_str = last_run_str[:-1] + "+00:00"
                                last_run = datetime.fromisoformat(
                                    last_run_str.replace("Z", "")
                                )
                        except Exception:
                            pass

                    start_time = None
                    if status.get("start_time"):
                        try:
                            start_time_str = status["start_time"]
                            if isinstance(start_time_str, str):
                                if start_time_str.endswith("Z"):
                                    start_time_str = start_time_str[:-1] + "+00:00"
                                start_time = datetime.fromisoformat(
                                    start_time_str.replace("Z", "")
                                )
                        except Exception:
                            pass

                    self.db_manager.update_crawler_status(
                        name,
                        status.get("status", "idle"),
                        last_run=last_run,
                        pid=status.get("pid"),
                        start_time=start_time,
                    )
            except Exception as e:
                logger.error(f"Error saving crawler status to database: {e}")
        else:
            # Fallback to file
            try:
                data = {"crawlers": self.crawlers, "logs": self.logs}
                with open(self.status_file, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving crawler status to file: {e}")

    def _status_checker(self):
        """Background thread to check process status"""
        while not self._shutdown_requested:
            try:
                for crawler_name, process_info in list(self.processes.items()):
                    pid = process_info["pid"]
                    try:
                        # Check if process is still running
                        process = process_info["process"]
                        if process.poll() is not None:
                            # Process finished
                            exit_code = process.returncode
                            last_run_time = datetime.utcnow()

                            if exit_code == 0:
                                self.crawlers[crawler_name]["status"] = "idle"
                                self._add_log(
                                    crawler_name,
                                    f"Process completed successfully (PID: {pid}, exit code: 0)",
                                )
                                # Reset failure count on success
                                self.failure_counts[crawler_name] = 0
                            else:
                                self.crawlers[crawler_name]["status"] = "failed"
                                self._add_log(
                                    crawler_name,
                                    f"Process failed with exit code {exit_code} (PID: {pid})",
                                )
                                # Increment failure count
                                self.failure_counts[crawler_name] = (
                                    self.failure_counts.get(crawler_name, 0) + 1
                                )

                                if (
                                    self.failure_counts[crawler_name]
                                    >= self.MAX_CONSECUTIVE_FAILURES
                                ):
                                    self._add_log(
                                        crawler_name,
                                        f"Warning: {self.failure_counts[crawler_name]} consecutive failures",
                                    )

                            # Clean up
                            del self.processes[crawler_name]
                            self.crawlers[crawler_name]["last_run"] = (
                                last_run_time.isoformat() + "Z"
                            )
                            self.crawlers[crawler_name]["pid"] = None
                            self.crawlers[crawler_name]["start_time"] = None
                            self.save_status()
                        else:
                            # Process still running - verify it's not a zombie
                            try:
                                if pid and psutil.pid_exists(pid):
                                    proc = psutil.Process(pid)
                                    if proc.status() == psutil.STATUS_ZOMBIE:
                                        # Process became zombie
                                        logger.warning(
                                            f"Process {crawler_name} (PID: {pid}) became zombie"
                                        )
                                        self._add_log(
                                            crawler_name,
                                            f"Process became zombie (PID: {pid})",
                                        )

                                        # Try to reap it
                                        try:
                                            process.wait(timeout=1)
                                        except subprocess.TimeoutExpired:
                                            process.kill()
                                            try:
                                                process.wait(timeout=5)
                                            except Exception:
                                                pass

                                        del self.processes[crawler_name]
                                        self.crawlers[crawler_name]["status"] = "failed"
                                        self.crawlers[crawler_name]["pid"] = None
                                        self.crawlers[crawler_name]["start_time"] = None
                                        self.crawlers[crawler_name]["last_run"] = (
                                            datetime.utcnow().isoformat() + "Z"
                                        )
                                        self.failure_counts[crawler_name] = (
                                            self.failure_counts.get(crawler_name, 0) + 1
                                        )
                                        self.save_status()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                            except Exception as e:
                                logger.warning(
                                    f"Error checking zombie status for {crawler_name}: {e}"
                                )

                    except Exception as e:
                        logger.error(f"Error checking process {pid}: {e}")
                        self.crawlers[crawler_name]["status"] = "failed"
                        self.crawlers[crawler_name]["pid"] = None
                        self.crawlers[crawler_name]["start_time"] = None
                        self._add_log(crawler_name, f"Error monitoring process: {e}")
                        self.failure_counts[crawler_name] = (
                            self.failure_counts.get(crawler_name, 0) + 1
                        )
                        if crawler_name in self.processes:
                            del self.processes[crawler_name]
                        self.save_status()

                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in status checker: {e}")
                time.sleep(10)

    def _add_log(self, crawler_name: str, message: str):
        """Add a log entry for a crawler"""
        timestamp = datetime.utcnow()
        timestamp_str = timestamp.isoformat() + "Z"

        if self.db_manager and self.db_manager._initialized:
            # Save to database
            try:
                self.db_manager.add_log(crawler_name, message, timestamp)
                # Update in-memory cache (last 100)
                if crawler_name not in self.logs:
                    self.logs[crawler_name] = []
                self.logs[crawler_name].append(f"[{timestamp_str}] {message}")
                if len(self.logs[crawler_name]) > 100:
                    self.logs[crawler_name] = self.logs[crawler_name][-100:]
            except Exception as e:
                logger.error(f"Error adding log to database: {e}")
                # Fallback to in-memory
                if crawler_name not in self.logs:
                    self.logs[crawler_name] = []
                self.logs[crawler_name].append(f"[{timestamp_str}] {message}")
                if len(self.logs[crawler_name]) > 100:
                    self.logs[crawler_name] = self.logs[crawler_name][-100:]
        else:
            # Fallback to in-memory
            if crawler_name not in self.logs:
                self.logs[crawler_name] = []
            self.logs[crawler_name].append(f"[{timestamp_str}] {message}")
            if len(self.logs[crawler_name]) > 100:
                self.logs[crawler_name] = self.logs[crawler_name][-100:]

    def get_crawler_status(self):
        """Get status of all crawlers"""
        # Initialize default crawlers if not exists
        default_crawlers = {
            "kenya_law": {"status": "idle", "last_run": None},
            "parliament": {"status": "idle", "last_run": None},
            "news_rss": {"status": "idle", "last_run": None},
            "global_trends": {"status": "idle", "last_run": None},
            "parliament_videos": {"status": "idle", "last_run": None},
        }

        # Load from database if available
        if self.db_manager and self.db_manager._initialized:
            try:
                db_statuses = self.db_manager.get_crawler_status()
                for name, default_status in default_crawlers.items():
                    if name in db_statuses:
                        self.crawlers[name] = db_statuses[name]
                    else:
                        self.crawlers[name] = default_status

                # Load logs from database
                for name in self.crawlers.keys():
                    self.logs[name] = self.db_manager.get_logs(name, limit=100)
            except Exception as e:
                logger.error(f"Error loading crawler status from database: {e}")
                # Fallback to in-memory
                for name, default_status in default_crawlers.items():
                    if name not in self.crawlers:
                        self.crawlers[name] = default_status
                        self.logs[name] = []
        else:
            # Merge with saved status (in-memory)
            for name, default_status in default_crawlers.items():
                if name not in self.crawlers:
                    self.crawlers[name] = default_status
                    self.logs[name] = []

        # Try to get actual last run times from database
        self._update_last_run_times()

        # Return current status with logs
        result = {}
        for name, status in self.crawlers.items():
            result[name] = {**status, "logs": self.logs.get(name, [])}

        return result

    def _update_last_run_times(self):
        """Update last run times from database"""
        try:
            if self.database_storage is None:
                logger.warning(
                    "Database storage not available, skipping last run times update"
                )
                return

            # Map crawler names to database categories/sources
            crawler_mapping = {
                "kenya_law": {"category": "Kenyan Law"},
                "parliament": {"category": "Parliament"},
                "news_rss": {"source_name": "News RSS"},
                "global_trends": {"category": "Global Trend"},
                "parliament_videos": {"category": "Parliamentary Record"},
            }

            with self.database_storage.get_db_session() as db:
                for crawler_name, filters in crawler_mapping.items():
                    try:
                        # Query the most recent crawl_date for this crawler type
                        from sqlalchemy import func
                        from Module3_NiruDB.database_storage import RawDocument

                        query = db.query(func.max(RawDocument.crawl_date))

                        if "category" in filters:
                            query = query.filter(
                                RawDocument.category == filters["category"]
                            )
                        if "source_name" in filters:
                            query = query.filter(
                                RawDocument.source_name == filters["source_name"]
                            )

                        last_run = query.scalar()

                        if last_run:
                            self.crawlers[crawler_name]["last_run"] = (
                                last_run.isoformat() + "Z"
                            )
                        else:
                            # No data found, keep as None or set to never
                            self.crawlers[crawler_name]["last_run"] = None

                    except Exception as e:
                        logger.warning(
                            f"Error getting last run time for {crawler_name}: {e}"
                        )
                        self.crawlers[crawler_name]["last_run"] = None

        except Exception as e:
            logger.warning(f"Error updating last run times from database: {e}")

    def start_crawler(self, crawler_name: str, force: bool = False):
        """Start a specific crawler

        Args:
            crawler_name: Name of the crawler to start
            force: If True, will start even if crawler has high failure count
        """
        if crawler_name not in self.crawlers:
            raise HTTPException(
                status_code=404, detail=f"Crawler {crawler_name} not found"
            )

        # Check if already running
        if crawler_name in self.processes:
            return {
                "status": "already_running",
                "message": f"Crawler {crawler_name} is already running",
            }

        # Check if there's a stale process we need to clean up first
        if self.crawlers[crawler_name].get("status") == "running":
            pid = self.crawlers[crawler_name].get("pid")
            if pid:
                # Try to verify if it's actually running
                try:
                    if psutil.pid_exists(pid):
                        proc = psutil.Process(pid)
                        if proc.is_running() and "python" in proc.name().lower():
                            return {
                                "status": "already_running",
                                "message": f"Crawler {crawler_name} appears to be running (PID: {pid})",
                            }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

                # It's a zombie, clean it up
                logger.info(
                    f"Cleaning up stale process entry for {crawler_name} (PID: {pid})"
                )
                self.crawlers[crawler_name]["status"] = "idle"
                self.crawlers[crawler_name]["pid"] = None
                self.crawlers[crawler_name]["start_time"] = None

        # Check failure count (unless forced)
        if (
            not force
            and self.failure_counts.get(crawler_name, 0)
            >= self.MAX_CONSECUTIVE_FAILURES
        ):
            return {
                "status": "blocked",
                "message": f"Crawler {crawler_name} has failed {self.failure_counts[crawler_name]} consecutive times. Use force=true to override or reset_failure_count to clear.",
                "failure_count": self.failure_counts[crawler_name],
            }

        try:
            # Get spider directory
            spider_dir = Path(__file__).parent.parent / "Module1_NiruSpider"

            # Map crawler names to spider names
            spider_mapping = {
                "kenya_law": "kenya_law",
                "parliament": "parliament",
                "news_rss": "news_rss",
                "global_trends": "global_trends",
                "parliament_videos": "parliament_videos",
            }

            if crawler_name not in spider_mapping:
                raise HTTPException(
                    status_code=404, detail=f"Unknown crawler: {crawler_name}"
                )

            spider_name = spider_mapping[crawler_name]

            # Start subprocess with log capture
            # Pass timeout as argument to crawl_spider.py (in seconds)
            timeout_seconds = self.process_timeouts.get(
                crawler_name, self.DEFAULT_PROCESS_TIMEOUT
            )
            cmd = [
                sys.executable,
                "crawl_spider.py",
                spider_name,
                "--timeout",
                str(timeout_seconds),
            ]

            self._add_log(
                crawler_name,
                f"Starting crawler with {timeout_seconds // 60} minute timeout...",
            )

            # Create subprocess with pipes for log capture
            process = subprocess.Popen(
                cmd,
                cwd=str(spider_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr with stdout
                bufsize=1,
                universal_newlines=True,
                # Create new process group for proper signal handling
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if sys.platform == "win32"
                    else 0
                ),
            )

            # Store process info
            start_time = datetime.utcnow()
            self.processes[crawler_name] = {
                "process": process,
                "pid": process.pid,
                "start_time": start_time.isoformat() + "Z",
            }

            # Update status
            self.crawlers[crawler_name]["status"] = "running"
            self.crawlers[crawler_name]["pid"] = process.pid
            self.crawlers[crawler_name]["start_time"] = start_time.isoformat() + "Z"
            self._add_log(crawler_name, f"Started crawler process (PID: {process.pid})")
            self.save_status()

            # Start log reader thread
            log_thread = threading.Thread(
                target=self._read_process_logs,
                args=(crawler_name, process),
                daemon=True,
            )
            log_thread.start()

            logger.info(
                f"Started {crawler_name} crawler subprocess (PID: {process.pid})"
            )
            return {
                "status": "started",
                "message": f"Crawler {crawler_name} started successfully",
                "pid": process.pid,
                "timeout_minutes": timeout_seconds // 60,
            }

        except Exception as e:
            logger.error(f"Error starting crawler {crawler_name}: {e}")
            self.crawlers[crawler_name]["status"] = "failed"
            self.failure_counts[crawler_name] = (
                self.failure_counts.get(crawler_name, 0) + 1
            )
            self._add_log(crawler_name, f"Failed to start: {str(e)}")
            self.save_status()
            raise HTTPException(
                status_code=500, detail=f"Failed to start crawler: {str(e)}"
            )

    def _read_process_logs(self, crawler_name: str, process: subprocess.Popen):
        """Read logs from subprocess and store them"""
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                # Clean and store log line
                clean_line = line.strip()
                if clean_line:
                    self._add_log(crawler_name, clean_line)
        except Exception as e:
            self._add_log(crawler_name, f"Error reading logs: {str(e)}")

    def stop_crawler(self, crawler_name: str):
        """Stop a specific crawler with graceful shutdown"""
        if crawler_name not in self.crawlers:
            raise HTTPException(
                status_code=404, detail=f"Crawler {crawler_name} not found"
            )

        if crawler_name not in self.processes:
            # Check if there's a stale PID we need to clean up
            pid = self.crawlers[crawler_name].get("pid")
            if pid and self.crawlers[crawler_name].get("status") == "running":
                # Try to kill the stale process
                try:
                    if psutil.pid_exists(pid):
                        proc = psutil.Process(pid)
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            proc.kill()
                        self._add_log(
                            crawler_name, f"Stopped stale process (PID: {pid})"
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                except Exception as e:
                    logger.warning(f"Error stopping stale process {pid}: {e}")

                # Clean up status
                self.crawlers[crawler_name]["status"] = "idle"
                self.crawlers[crawler_name]["pid"] = None
                self.crawlers[crawler_name]["start_time"] = None
                self.crawlers[crawler_name]["last_run"] = (
                    datetime.utcnow().isoformat() + "Z"
                )
                self.save_status()
                return {
                    "status": "stopped",
                    "message": f"Cleaned up stale crawler {crawler_name}",
                }

            return {
                "status": "not_running",
                "message": f"Crawler {crawler_name} is not running",
            }

        try:
            process_info = self.processes[crawler_name]
            process = process_info["process"]
            pid = process_info["pid"]

            self._add_log(crawler_name, f"Stopping crawler (PID: {pid})...")

            # Try graceful termination first (sends SIGTERM)
            try:
                # Also terminate child processes
                if pid and psutil.pid_exists(pid):
                    parent = psutil.Process(pid)
                    children = parent.children(recursive=True)

                    # Send SIGTERM to children
                    for child in children:
                        try:
                            child.terminate()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                process.terminate()
            except Exception as e:
                logger.warning(f"Error sending SIGTERM to {crawler_name}: {e}")

            # Wait for graceful shutdown (15 seconds)
            try:
                process.wait(timeout=15)
                self._add_log(crawler_name, f"Process stopped gracefully (PID: {pid})")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't respond
                logger.warning(
                    f"Process {crawler_name} did not respond to SIGTERM, force killing..."
                )
                self._add_log(
                    crawler_name,
                    f"Process did not respond to SIGTERM, force killing (PID: {pid})",
                )

                # Kill entire process tree
                try:
                    if pid and psutil.pid_exists(pid):
                        parent = psutil.Process(pid)
                        children = parent.children(recursive=True)

                        for child in children:
                            try:
                                child.kill()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass

                        parent.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    pass

            # Clean up
            del self.processes[crawler_name]
            self.crawlers[crawler_name]["status"] = "idle"
            self.crawlers[crawler_name]["pid"] = None
            self.crawlers[crawler_name]["start_time"] = None
            self.crawlers[crawler_name]["last_run"] = (
                datetime.utcnow().isoformat() + "Z"
            )
            self.save_status()

            return {
                "status": "stopped",
                "message": f"Crawler {crawler_name} stopped successfully",
            }

        except Exception as e:
            logger.error(f"Error stopping crawler {crawler_name}: {e}")
            # Try to clean up anyway
            if crawler_name in self.processes:
                del self.processes[crawler_name]
            self.crawlers[crawler_name]["status"] = "failed"
            self.crawlers[crawler_name]["pid"] = None
            self.crawlers[crawler_name]["start_time"] = None
            self._add_log(crawler_name, f"Error during stop: {e}")
            self.save_status()
            raise HTTPException(
                status_code=500, detail=f"Failed to stop crawler: {str(e)}"
            )

    def stop_all_crawlers(self):
        """Stop all running crawlers"""
        results = {}
        for crawler_name in list(self.processes.keys()):
            try:
                result = self.stop_crawler(crawler_name)
                results[crawler_name] = result
            except Exception as e:
                results[crawler_name] = {"status": "error", "message": str(e)}
        return results

    def get_health_status(self):
        """Get overall health status of the crawler system"""
        running_count = len(self.processes)
        failed_crawlers = [
            name
            for name, status in self.crawlers.items()
            if status.get("status") == "failed"
        ]

        # Check for crawlers with high failure counts
        problematic_crawlers = [
            name
            for name, count in self.failure_counts.items()
            if count >= self.MAX_CONSECUTIVE_FAILURES
        ]

        return {
            "healthy": len(failed_crawlers) == 0 and len(problematic_crawlers) == 0,
            "running_count": running_count,
            "failed_crawlers": failed_crawlers,
            "problematic_crawlers": problematic_crawlers,
            "failure_counts": self.failure_counts.copy(),
        }

    def reset_failure_count(self, crawler_name: str):
        """Reset the failure count for a crawler"""
        if crawler_name in self.failure_counts:
            self.failure_counts[crawler_name] = 0
            self._add_log(crawler_name, "Failure count reset")
            return {
                "status": "success",
                "message": f"Failure count for {crawler_name} reset",
            }
        return {"status": "not_found", "message": f"Crawler {crawler_name} not found"}
