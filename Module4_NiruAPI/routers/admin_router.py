"""
Admin Router - Administrative endpoints for AmaniQuery
Includes crawler management, document management, config, and system info
"""

import os
import sys
import json
import asyncio
import threading
import subprocess
import time
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, Depends, Form
from fastapi import Request as FastAPIRequest
from loguru import logger
from pydantic import BaseModel
import psutil

router = APIRouter(prefix="/api/admin", tags=["Admin"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class ConfigSetRequest(BaseModel):
    """Config set request"""

    key: str
    value: str
    description: Optional[str] = ""


# =============================================================================
# DEPENDENCIES - Module-level globals set by main app
# =============================================================================

crawler_manager = None
vector_store = None
config_manager = None
database_storage = None
cache_manager = None  # Optional async cache manager


def get_crawler_manager():
    """Get crawler manager with lazy initialization fallback"""
    global crawler_manager
    if crawler_manager is None:
        logger.warning(
            "Crawler manager not initialized via dependency injection, attempting lazy initialization"
        )
        try:
            from Module4_NiruAPI.crawler_manager import CrawlerManager

            crawler_manager = CrawlerManager()
            logger.info("Crawler manager lazily initialized successfully")
        except Exception as e:
            logger.error(f"Failed to lazily initialize crawler manager: {e}")
            # Don't raise here, let the endpoint handle the None check or raise 503
    return crawler_manager


def get_vector_store():
    """Get vector store with lazy initialization fallback"""
    global vector_store
    if vector_store is None:
        logger.warning(
            "Vector store not initialized via dependency injection, attempting lazy initialization"
        )
        try:
            from Module3_NiruDB.vector_store import VectorStore

            vector_store = VectorStore()
            logger.info("Vector store lazily initialized successfully")
        except Exception as e:
            logger.error(f"Failed to lazily initialize vector store: {e}")
    return vector_store


def get_config_manager():
    """Get config manager with lazy initialization fallback"""
    global config_manager
    if config_manager is None:
        logger.warning(
            "Config manager not initialized via dependency injection, attempting lazy initialization"
        )
        try:
            from Module4_NiruAPI.config_manager import ConfigManager

            config_manager = ConfigManager()
            logger.info("Config manager lazily initialized successfully")
        except Exception as e:
            logger.error(f"Failed to lazily initialize config manager: {e}")
    return config_manager


def get_database_storage():
    """Get database storage with lazy initialization fallback"""
    global database_storage
    if database_storage is None:
        logger.warning(
            "Database storage not initialized via dependency injection, attempting lazy initialization"
        )
        try:
            from Module3_NiruDB.database_storage import DatabaseStorage

            database_storage = DatabaseStorage()
            logger.info("Database storage lazily initialized successfully")
        except Exception as e:
            logger.error(f"Failed to lazily initialize database storage: {e}")
    return database_storage


def get_admin_dependency():
    """Get admin dependency - conditional based on ENABLE_AUTH"""
    if os.getenv("ENABLE_AUTH", "false").lower() == "true":
        try:
            from Module8_NiruAuth.dependencies import require_admin

            return require_admin
        except ImportError:
            pass

    # No-op dependency when auth is disabled
    def no_auth_required(request: Request):
        return None

    return no_auth_required


_admin_dependency = get_admin_dependency()


# =============================================================================
# CRAWLER ENDPOINTS
# =============================================================================


@router.get("/crawlers")
async def get_crawler_status(request: Request, admin=Depends(_admin_dependency)):
    """Get status of all crawlers - cached for 5 seconds"""
    crawler_manager = get_crawler_manager()
    if crawler_manager is None:
        raise HTTPException(status_code=503, detail="Crawler manager not initialized")

    cache_key = "cache:admin:crawlers"
    if cache_manager:
        cached_result = await cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result

    try:
        crawlers = crawler_manager.get_crawler_status()
        result = {"crawlers": crawlers}

        if cache_manager:
            await cache_manager.set(cache_key, result, ttl=5)

        return result
    except Exception as e:
        logger.error(f"Error getting crawler status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crawlers/{crawler_name}/start")
async def start_crawler(
    crawler_name: str, request: Request, admin=Depends(_admin_dependency)
):
    """Start a specific crawler"""
    if crawler_manager is None:
        raise HTTPException(status_code=503, detail="Crawler manager not initialized")

    try:
        result = crawler_manager.start_crawler(crawler_name)

        if cache_manager:
            cache_manager.invalidate_crawler_cache()

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting crawler {crawler_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crawlers/{crawler_name}/stop")
async def stop_crawler(
    crawler_name: str, request: Request, admin=Depends(_admin_dependency)
):
    """Stop a specific crawler"""
    if crawler_manager is None:
        raise HTTPException(status_code=503, detail="Crawler manager not initialized")

    try:
        result = crawler_manager.stop_crawler(crawler_name)

        if cache_manager:
            cache_manager.invalidate_crawler_cache()

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping crawler {crawler_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/crawlers/{name}/logs")
async def get_crawler_logs(
    name: str, request: Request, admin=Depends(_admin_dependency), limit: int = 100
):
    """Get logs for a specific crawler"""
    if crawler_manager is None:
        raise HTTPException(status_code=503, detail="Crawler manager not initialized")

    logs = crawler_manager.get_logs(name, limit)
    return {"crawler": name, "logs": logs}


# =============================================================================
# DOCUMENT ENDPOINTS
# =============================================================================


@router.get("/documents")
async def search_documents(
    request: Request,
    query: str = "",
    category: Optional[str] = None,
    source: Optional[str] = None,
    namespace: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    admin=Depends(_admin_dependency),
):
    """Search and retrieve documents from the database"""
    vector_store = get_vector_store()
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        filter_dict = {}
        if category:
            filter_dict["category"] = category
        if source:
            filter_dict["source"] = source

        search_namespaces = None
        if namespace:
            search_namespaces = [ns.strip() for ns in namespace.split(",")]

        if query:
            results = vector_store.query(
                query_text=query,
                n_results=limit,
                filter=filter_dict if filter_dict else None,
                namespace=search_namespaces,
            )
        else:
            results = vector_store.query(
                query_text="",
                n_results=limit,
                filter=filter_dict if filter_dict else None,
                namespace=search_namespaces,
            )

        documents = []
        for chunk in results:
            metadata = chunk.get("metadata", {})
            documents.append(
                {
                    "id": chunk.get("id", ""),
                    "content": chunk.get("text", ""),
                    "metadata": {
                        "title": metadata.get("title", ""),
                        "url": metadata.get("source_url", ""),
                        "source": metadata.get("source_name", ""),
                        "category": metadata.get("category", ""),
                        "date": metadata.get("publication_date", ""),
                        "author": metadata.get("author", ""),
                        "sentiment_polarity": metadata.get("sentiment_polarity"),
                        "sentiment_label": metadata.get("sentiment_label"),
                    },
                    "score": (
                        1.0 - chunk.get("distance", 0.0)
                        if chunk.get("distance") is not None
                        else 0
                    ),
                }
            )

        return {
            "documents": documents,
            "total": len(documents),
            "query": query,
            "filters": filter_dict,
        }

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str, request: Request, admin=Depends(_admin_dependency)):
    """Get a specific document by ID"""
    vector_store = get_vector_store()
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        document = vector_store.get_document(doc_id)

        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CONFIG ENDPOINTS
# =============================================================================


@router.get("/config")
async def get_config_list(request: Request, admin=Depends(_admin_dependency)):
    """Get list of all configuration keys - cached for 300 seconds"""
    config_manager = get_config_manager()
    config_manager = get_config_manager()
    if config_manager is None:
        return {
            "error": "Config manager not initialized",
            "message": "PostgreSQL database connection required",
            "status": "unavailable",
        }

    cache_key = "cache:admin:config"
    if cache_manager:
        cached_result = await cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result

    try:
        result = config_manager.list_configs()

        if cache_manager:
            await cache_manager.set(cache_key, result, ttl=300)

        return result
    except Exception as e:
        logger.error(f"Error getting config list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config")
async def set_config(
    config: ConfigSetRequest, request: Request, admin=Depends(_admin_dependency)
):
    """Set a configuration value"""
    config_manager = get_config_manager()
    if config_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Config manager not initialized",
        )

    try:
        config_manager.set_config(config.key, config.value, config.description or "")

        if cache_manager:
            await cache_manager.delete("cache:admin:config")

        return {"message": f"Config {config.key} set successfully"}
    except Exception as e:
        logger.error(f"Error setting config {config.key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config/{key}")
async def update_config_entry(
    key: str, request: FastAPIRequest, admin=Depends(_admin_dependency)
):
    """Update a configuration value"""
    config_manager = get_config_manager()
    if config_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Config manager not initialized",
        )

    try:
        body = await request.json()
        value = body.get("value", "")
        description = body.get("description", "")
        config_manager.set_config(key, value, description)

        if cache_manager:
            await cache_manager.delete("cache:admin:config")

        return {"message": f"Config {key} updated successfully"}
    except Exception as e:
        logger.error(f"Error updating config {key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/config/{key}")
async def delete_config_entry(
    key: str, request: Request, admin=Depends(_admin_dependency)
):
    """Delete a configuration value"""
    config_manager = get_config_manager()
    if config_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Config manager not initialized",
        )

    try:
        config_manager.delete_config(key)

        if cache_manager:
            await cache_manager.delete("cache:admin:config")

        return {"message": f"Config {key} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting config {key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DATABASE ENDPOINTS
# =============================================================================


@router.get("/databases")
async def get_database_stats(request: Request, admin=Depends(_admin_dependency)):
    """Get statistics for all vector database backends - cached for 60 seconds"""
    vector_store = get_vector_store()
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    cache_key = "cache:admin:databases"
    if cache_manager:
        cached_result = await cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result

    try:
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, vector_store.get_stats)

        databases = []
        used_names = set()

        # Primary backend
        primary_name = stats.get("backend", "unknown")
        if primary_name not in used_names:
            primary_db = {
                "name": primary_name,
                "type": "primary",
                "status": "active" if stats.get("total_chunks", 0) > 0 else "inactive",
                "total_chunks": stats.get("total_chunks", 0),
                "categories": stats.get("sample_categories", {}),
                "persist_directory": stats.get("persist_directory", ""),
                "elasticsearch_docs": stats.get("elasticsearch_docs", 0),
                "elasticsearch_enabled": stats.get("elasticsearch_enabled", False),
            }
            databases.append(primary_db)
            used_names.add(primary_name)

        # Cloud backends
        cloud_backends = stats.get("cloud_backends", [])
        for backend_name in cloud_backends:
            if backend_name not in used_names:
                db_info = {
                    "name": backend_name,
                    "type": "cloud",
                    "status": "active",
                    "total_chunks": 0,
                    "categories": {},
                    "persist_directory": "",
                }
                databases.append(db_info)
                used_names.add(backend_name)

        result = {
            "databases": databases,
            "total_databases": len(databases),
            "active_databases": len(
                [db for db in databases if db["status"] == "active"]
            ),
        }

        if cache_manager:
            await cache_manager.set(cache_key, result, ttl=60)

        return result

    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/database-storage")
async def getdatabase_storage_stats(request: Request, admin=Depends(_admin_dependency)):
    """Get database storage statistics - cached for 60 seconds"""
    database_storage = get_database_storage()
    if database_storage is None:
        raise HTTPException(status_code=503, detail="Database storage not initialized")

    cache_key = "cache:admin:database-storage"
    if cache_manager:
        cached_result = await cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result

    try:
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, database_storage.get_stats)

        if cache_manager:
            await cache_manager.set(cache_key, stats, ttl=60)

        return stats
    except Exception as e:
        logger.error(f"Error getting database storage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SYSTEM INFO ENDPOINTS
# =============================================================================


@router.get("/system-info")
async def get_system_info(request: Request, admin=Depends(_admin_dependency)):
    """Get system information"""
    try:
        import platform

        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
            },
            "disk": {
                "total": psutil.disk_usage("/").total,
                "free": psutil.disk_usage("/").free,
                "percent": psutil.disk_usage("/").percent,
            },
            "uptime": psutil.boot_time(),
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        import platform

        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "note": "Install psutil for detailed system info",
        }


@router.post("/execute")
async def execute_command(
    command: str,
    request: Request,
    cwd: Optional[str] = None,
    admin=Depends(_admin_dependency),
):
    """Execute a shell command (admin only)"""
    try:
        # Security check - only allow safe commands
        allowed_commands = [
            "ls",
            "pwd",
            "ps",
            "top",
            "df",
            "du",
            "free",
            "uptime",
            "python",
            "pip",
            "npm",
            "node",
            "git",
            "docker",
        ]

        cmd_parts = command.split()
        if not cmd_parts:
            raise HTTPException(status_code=400, detail="Empty command")

        base_cmd = cmd_parts[0]
        if base_cmd not in allowed_commands:
            raise HTTPException(
                status_code=403, detail=f"Command '{base_cmd}' not allowed"
            )

        working_dir = cwd or str(Path(__file__).parent.parent.parent)

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
            shell=True,
        )

        stdout, stderr = await process.communicate()

        return {
            "command": command,
            "exit_code": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "cwd": working_dir,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SCHEDULER ENDPOINTS
# =============================================================================

# Module-level scheduler service instance
_scheduler_service = None


def get_scheduler_service():
    """Get or create scheduler service instance"""
    global _scheduler_service
    if _scheduler_service is None:
        try:
            from Module1_NiruSpider.scheduler import SchedulerService, SchedulerConfig

            config = SchedulerConfig.default()
            _scheduler_service = SchedulerService(config=config, backend="apscheduler")
        except ImportError as e:
            logger.warning(f"Could not import scheduler service: {e}")
            return None
    return _scheduler_service


@router.get("/scheduler")
async def get_scheduler_status(request: Request, admin=Depends(_admin_dependency)):
    """Get scheduler status and configuration"""
    scheduler = get_scheduler_service()
    if scheduler is None:
        return {
            "status": "unavailable",
            "message": "Scheduler service not available",
            "running": False,
        }

    try:
        status = scheduler.get_status()

        # Add schedule information
        schedules_info = {}
        for name, schedule in scheduler.config.schedules.items():
            schedules_info[name] = {
                "enabled": schedule.enabled,
                "interval_hours": schedule.interval_hours,
                "timeout_minutes": schedule.timeout_minutes,
                "priority": schedule.priority,
                "description": schedule.description,
                "last_run": (
                    schedule.last_run.isoformat() if schedule.last_run else None
                ),
                "last_status": schedule.last_status,
                "consecutive_failures": schedule.consecutive_failures,
            }

        return {**status, "schedules": schedules_info}
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        return {"status": "error", "message": str(e), "running": False}


@router.post("/scheduler/start")
async def start_scheduler(request: Request, admin=Depends(_admin_dependency)):
    """Start the scheduler service"""
    scheduler = get_scheduler_service()
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler service not available")

    try:
        # Start in background thread
        def start_scheduler_thread():
            scheduler.start()

        thread = threading.Thread(target=start_scheduler_thread, daemon=True)
        thread.start()

        # Wait a moment for startup
        time.sleep(2)

        return {"status": "started", "message": "Scheduler service started"}
    except Exception as e:
        logger.error(f"Error starting scheduler: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scheduler/stop")
async def stop_scheduler(request: Request, admin=Depends(_admin_dependency)):
    """Stop the scheduler service"""
    scheduler = get_scheduler_service()
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler service not available")

    try:
        scheduler.stop()
        return {"status": "stopped", "message": "Scheduler service stopped"}
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scheduler/trigger/{crawler_name}")
async def trigger_scheduled_crawler(
    crawler_name: str, request: Request, admin=Depends(_admin_dependency)
):
    """Manually trigger a scheduled crawler"""
    scheduler = get_scheduler_service()
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler service not available")

    try:
        from Module1_NiruSpider.scheduler import CrawlerType

        # Validate crawler name
        valid_crawlers = [ct.value for ct in CrawlerType]
        if crawler_name not in valid_crawlers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid crawler name. Valid options: {valid_crawlers}",
            )

        crawler_type = CrawlerType(crawler_name)
        success = scheduler.trigger_crawler(crawler_type)

        if success:
            return {
                "status": "triggered",
                "crawler": crawler_name,
                "message": f"Crawler {crawler_name} triggered successfully",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to trigger crawler")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering crawler {crawler_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/scheduler/schedule/{crawler_name}")
async def update_crawler_schedule(
    crawler_name: str, request: Request, admin=Depends(_admin_dependency)
):
    """Update a crawler's schedule configuration"""
    scheduler = get_scheduler_service()
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler service not available")

    try:
        from Module1_NiruSpider.scheduler import CrawlerType

        # Validate crawler name
        valid_crawlers = [ct.value for ct in CrawlerType]
        if crawler_name not in valid_crawlers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid crawler name. Valid options: {valid_crawlers}",
            )

        body = await request.json()

        # Extract updateable fields
        updates = {}
        if "enabled" in body:
            updates["enabled"] = body["enabled"]
        if "interval_hours" in body:
            updates["interval_hours"] = body["interval_hours"]
        if "timeout_minutes" in body:
            updates["timeout_minutes"] = body["timeout_minutes"]

        if not updates:
            raise HTTPException(status_code=400, detail="No valid updates provided")

        crawler_type = CrawlerType(crawler_name)
        success = scheduler.update_schedule(crawler_type, **updates)

        if success:
            return {"status": "updated", "crawler": crawler_name, "updates": updates}
        else:
            raise HTTPException(
                status_code=404, detail=f"Crawler {crawler_name} not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating schedule for {crawler_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scheduler/health")
async def get_scheduler_health(request: Request, admin=Depends(_admin_dependency)):
    """Get scheduler health status"""
    # Also check crawler manager health
    health = {
        "scheduler": {"status": "unknown"},
        "crawler_manager": {"status": "unknown"},
        "overall": "unknown",
    }

    # Check scheduler
    scheduler = get_scheduler_service()
    if scheduler:
        try:
            status = scheduler.get_status()
            health["scheduler"] = {
                "status": "running" if status.get("running") else "stopped",
                "running_crawlers": status.get("running_crawlers", []),
            }
        except Exception as e:
            health["scheduler"] = {"status": "error", "error": str(e)}
    else:
        health["scheduler"] = {"status": "unavailable"}

    # Check crawler manager
    if crawler_manager:
        try:
            cm_health = crawler_manager.get_health_status()
            health["crawler_manager"] = cm_health
        except Exception as e:
            health["crawler_manager"] = {"status": "error", "error": str(e)}
    else:
        health["crawler_manager"] = {"status": "unavailable"}

    # Determine overall health
    if health["scheduler"]["status"] == "running" and health["crawler_manager"].get(
        "healthy", False
    ):
        health["overall"] = "healthy"
    elif (
        health["scheduler"]["status"] == "error"
        or health["crawler_manager"].get("status") == "error"
    ):
        health["overall"] = "degraded"
    else:
        health["overall"] = "operational"

    return health
