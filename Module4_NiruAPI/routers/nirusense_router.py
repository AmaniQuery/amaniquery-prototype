"""
NiruSense Router
Health, analysis, and document management for the NiruSense processing pipeline
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from loguru import logger

router = APIRouter(prefix="/nirusense", tags=["NiruSense"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=10, ge=1, le=100)
    topics: Optional[List[str]] = Field(default=None, description="Filter by topics")
    sentiment: Optional[str] = Field(default=None, description="Filter by sentiment")
    source: Optional[str] = Field(default=None, description="Filter by source")
    min_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class IngestRequest(BaseModel):
    url: str = Field(..., description="Document URL to ingest")
    source: str = Field(default="api", description="Source identifier")
    text: Optional[str] = Field(
        default=None, description="Raw text content (skips scraping)"
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class SchedulerTriggerRequest(BaseModel):
    job_id: str = Field(
        ...,
        description="Job ID to trigger: batch_processing, cleanup, metrics, reprocess_failed",
    )


# =============================================================================
# STATE CONTAINER
# =============================================================================


class NiruSenseRouterState:
    health_checker = None
    pipeline = None
    metrics = None
    scheduler = None
    settings = None
    orchestrator_running = False


_state = NiruSenseRouterState()


# =============================================================================
# HELPERS
# =============================================================================


def _get_health_checker():
    if _state.health_checker is None:
        raise HTTPException(status_code=503, detail="NiruSense not initialized")
    return _state.health_checker


def _get_pipeline():
    if _state.pipeline is None:
        raise HTTPException(
            status_code=503, detail="NiruSense pipeline not initialized"
        )
    return _state.pipeline


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================


@router.get("/health")
async def nirusense_health():
    try:
        hc = _get_health_checker()
        health_status = await hc.get_overall_health()
        if health_status["status"] == "unhealthy":
            return JSONResponse(status_code=503, content=health_status)
        health_status["orchestrator_running"] = _state.orchestrator_running
        return health_status
    except HTTPException:
        raise
    except ImportError as e:
        return {
            "status": "unavailable",
            "error": "NiruSense module not properly configured",
            "details": str(e),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "error", "error": str(e)}
        )


@router.get("/health/redis")
async def nirusense_health_redis():
    try:
        hc = _get_health_checker()
        return await hc.check_redis()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/postgres")
async def nirusense_health_postgres():
    try:
        hc = _get_health_checker()
        return await hc.check_postgres()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/qdrant")
async def nirusense_health_qdrant():
    try:
        hc = _get_health_checker()
        return await hc.check_qdrant()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/agents")
async def nirusense_health_agents():
    try:
        hc = _get_health_checker()
        return await hc.check_agents()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# METRICS & STATUS
# =============================================================================


@router.get("/metrics")
async def nirusense_metrics():
    if _state.metrics is None:
        raise HTTPException(status_code=503, detail="NiruSense metrics not available")
    try:
        return _state.metrics.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def nirusense_status():
    if _state.settings is None:
        raise HTTPException(status_code=503, detail="NiruSense not initialized")
    try:
        import os

        return {
            "service": "NiruSense",
            "enabled": os.getenv("ENABLE_NIRUSENSE", "false").lower() == "true",
            "orchestrator_running": _state.orchestrator_running,
            "agents": {
                "total": 9,
                "enabled": [
                    "language_identifier",
                    "slang_decoder",
                    "topic_classifier",
                    "entity_extractor",
                    "sentiment_analyzer",
                    "emotion_detector",
                    "bias_detector",
                    "summarizer",
                    "quality_scorer",
                ],
            },
            "models": {
                "embedding": _state.settings.MODEL_EMBEDDING,
                "language": _state.settings.MODEL_LANGUAGE,
                "slang": _state.settings.MODEL_SLANG,
                "topic": _state.settings.MODEL_TOPIC,
                "ner": _state.settings.MODEL_NER,
                "sentiment": _state.settings.MODEL_SENTIMENT,
                "emotion": _state.settings.MODEL_EMOTION,
                "summarizer": _state.settings.MODEL_SUMMARIZER,
            },
            "configuration": {
                "redis_stream": _state.settings.REDIS_STREAM_KEY,
                "qdrant_collection": _state.settings.QDRANT_COLLECTION,
                "parallel_agents": _state.settings.ENABLE_PARALLEL_AGENTS,
                "max_text_length": _state.settings.MAX_TEXT_LENGTH,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DOCUMENT SEARCH & RETRIEVAL
# =============================================================================


@router.post("/documents/search")
async def search_documents(request: DocumentSearchRequest):
    """Search NiruSense-analyzed documents with optional filters"""
    try:
        from Module9_NiruSense.processing.qdrant_search import search_analyzed_documents

        results = await search_analyzed_documents(
            query=request.query,
            top_k=request.top_k,
            topics=request.topics,
            sentiment=request.sentiment,
            source=request.source,
            min_quality_score=request.min_quality_score,
        )
        return {"results": results, "total": len(results)}
    except ImportError:
        raise HTTPException(
            status_code=501, detail="NiruSense document search not available"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get a document with full NiruSense analysis by ID"""
    try:
        from Module9_NiruSense.processing.storage.postgres import postgres
        from Module9_NiruSense.processing.storage.qdrant import qdrant_storage

        doc = await postgres.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        analysis = await postgres.get_analysis(doc_id)
        # Look up vector point for embedding
        vector_data = None
        try:
            vector_data = qdrant_storage.client.retrieve(
                collection_name=qdrant_storage.collection_name,
                ids=[doc_id],
            )
        except Exception:
            pass
        return {
            "document": doc,
            "analysis": analysis,
            "vector_metadata": vector_data[0].payload if vector_data else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/by-url")
async def get_document_by_url(url: str = Query(..., description="Document URL")):
    """Look up a document by its source URL"""
    try:
        from Module9_NiruSense.processing.storage.postgres import postgres

        doc = await postgres.get_document_by_url(url)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        analysis = await postgres.get_analysis(doc["id"])
        return {"document": doc, "analysis": analysis}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# INGESTION
# =============================================================================


@router.post("/ingest")
async def ingest_document(request: IngestRequest):
    """Trigger document ingestion into NiruSense processing pipeline"""
    try:
        from Module9_NiruSense.processing.minio_client import minio_storage
        from Module9_NiruSense.processing.config import settings
        import redis.asyncio as redis
        import json
        import time

        payload = {
            "url": request.url,
            "source": request.source,
            "text": request.text,
            "metadata": request.metadata or {},
            "timestamp": int(time.time()),
        }

        # Store raw data in MinIO
        s3_key = f"api_ingestion/{request.source}/{int(time.time())}.json"
        minio_storage.put_object(s3_key, json.dumps(payload))

        # Push to Redis stream for orchestrator
        r = redis.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
            decode_responses=True,
        )
        await r.xadd(
            settings.REDIS_STREAM_KEY,
            {"s3_key": s3_key, "source": request.source, "url": request.url},
        )
        await r.close()

        return {"status": "queued", "s3_key": s3_key, "url": request.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SCHEDULER
# =============================================================================


@router.get("/scheduler")
async def scheduler_status():
    """Get NiruSense scheduler status"""
    if _state.scheduler is None:
        return {"running": False, "message": "Scheduler not initialized"}
    try:
        return _state.scheduler.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scheduler/trigger")
async def trigger_scheduler_job(request: SchedulerTriggerRequest):
    """Manually trigger a scheduler job"""
    if _state.scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    try:
        job_map = {
            "batch_processing": _state.scheduler._process_pending_batch,
            "cleanup": lambda: _state.scheduler._cleanup_old_documents(),
            "metrics": _state.scheduler._update_metrics,
            "reprocess_failed": lambda: _state.scheduler._reprocess_failed(),
        }
        job_fn = job_map.get(request.job_id)
        if not job_fn:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown job: {request.job_id}. Valid: {list(job_map.keys())}",
            )
        result = job_fn()
        return {"status": "triggered", "job_id": request.job_id, "result": str(result)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
