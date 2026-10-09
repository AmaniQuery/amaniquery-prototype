"""
Monitoring Router - Agent monitoring and metrics endpoints for AmaniQuery
"""

import os
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

router = APIRouter(prefix="/api/admin", tags=["Admin", "Agent Monitoring"])


# =============================================================================
# DEPENDENCIES - Module-level globals set by main app
# =============================================================================

db_session_factory = None


def get_db():
    """Get database session"""
    if db_session_factory is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    db = db_session_factory()
    try:
        yield db
    finally:
        db.close()


def get_admin_dependency():
    """Get admin dependency"""
    if os.getenv("ENABLE_AUTH", "false").lower() == "true":
        try:
            from Module8_NiruAuth.dependencies import require_admin

            return require_admin
        except ImportError:
            pass

    def no_auth_required(request: Request):
        return None

    return no_auth_required


_admin_dependency = get_admin_dependency()


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("/agent-metrics")
async def get_agent_metrics(
    request: Request, admin=Depends(_admin_dependency), days: int = 30
):
    """Get agent performance metrics from database"""
    try:
        from Module3_NiruDB.agent_monitoring import get_agent_metrics as get_metrics

        db = next(get_db())

        try:
            metrics = get_metrics(db, days=days)
            return metrics
        finally:
            db.close()

    except ImportError:
        logger.warning("Agent monitoring module not available")
        return {
            "total_queries": 0,
            "avg_confidence": 0.0,
            "avg_response_time_ms": 0,
            "human_review_rate": 0.0,
            "persona_distribution": {"wanjiku": 0, "wakili": 0, "mwanahabari": 0},
            "intent_distribution": {"news": 0, "law": 0, "hybrid": 0, "general": 0},
            "confidence_buckets": {"low": 0, "medium": 0, "high": 0},
        }
    except Exception as e:
        logger.error(f"Error getting agent metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query-logs")
async def get_query_logs(
    request: Request,
    admin=Depends(_admin_dependency),
    limit: int = 100,
    offset: int = 0,
):
    """Get agent query logs from database"""
    try:
        from Module3_NiruDB.agent_models import AgentQueryLog
        from sqlalchemy import desc

        db = next(get_db())

        try:
            total = db.query(AgentQueryLog).count()

            logs_query = (
                db.query(AgentQueryLog)
                .order_by(desc(AgentQueryLog.timestamp))
                .limit(limit)
                .offset(offset)
            )

            logs = [
                {
                    "id": log.id,
                    "timestamp": log.timestamp.isoformat(),
                    "query": log.query,
                    "persona": log.persona,
                    "intent": log.intent,
                    "confidence": log.confidence,
                    "response_time_ms": log.response_time_ms,
                    "evidence_count": log.evidence_count,
                    "reasoning_steps": log.reasoning_steps,
                    "human_review_required": log.human_review_required,
                    "agent_path": log.agent_path or [],
                    "quality_issues": log.quality_issues or [],
                    "reasoning_path": log.reasoning_path,
                    "user_feedback": log.user_feedback,
                }
                for log in logs_query.all()
            ]

            return {
                "logs": logs,
                "total": total,
                "page": offset // limit + 1,
                "page_size": limit,
            }
        finally:
            db.close()

    except ImportError:
        return {"logs": [], "total": 0, "page": 1, "page_size": limit}
    except Exception as e:
        logger.error(f"Error getting query logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/review-queue")
async def get_review_queue(request: Request, admin=Depends(_admin_dependency)):
    """Get queries pending human review from database"""
    try:
        from Module3_NiruDB.agent_monitoring import get_review_queue as get_queue

        db = next(get_db())

        try:
            queue = get_queue(db)
            return {"queue": queue, "total": len(queue)}
        finally:
            db.close()

    except ImportError:
        return {"queue": [], "total": 0}
    except Exception as e:
        logger.error(f"Error getting review queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/review-queue/{log_id}/approve")
async def approve_review(
    log_id: str, request: Request, admin=Depends(_admin_dependency)
):
    """Approve a query in the review queue"""
    try:
        from Module3_NiruDB.agent_models import AgentQueryLog

        db = next(get_db())

        try:
            log = db.query(AgentQueryLog).filter(AgentQueryLog.id == log_id).first()
            if not log:
                raise HTTPException(status_code=404, detail="Query log not found")

            # Get user_id from auth context if available
            user_id = None
            try:
                from Module8_NiruAuth.dependencies import get_auth_context

                auth_context = get_auth_context(request)
                user_id = auth_context.user_id if auth_context else None
            except ImportError:
                pass

            log.review_status = "approved"
            log.reviewed_at = datetime.utcnow()
            log.reviewed_by = user_id

            db.commit()
            logger.info(f"Review approved for log_id: {log_id}")
            return {"message": "Query approved successfully", "log_id": log_id}
        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/review-queue/{log_id}/reject")
async def reject_review(
    log_id: str, request: Request, admin=Depends(_admin_dependency)
):
    """Reject a query in the review queue with feedback"""
    try:
        from Module3_NiruDB.agent_models import AgentQueryLog

        body = await request.json()
        feedback = body.get("feedback", "")

        db = next(get_db())

        try:
            log = db.query(AgentQueryLog).filter(AgentQueryLog.id == log_id).first()
            if not log:
                raise HTTPException(status_code=404, detail="Query log not found")

            # Get user_id from auth context if available
            user_id = None
            try:
                from Module8_NiruAuth.dependencies import get_auth_context

                auth_context = get_auth_context(request)
                user_id = auth_context.user_id if auth_context else None
            except ImportError:
                pass

            log.review_status = "rejected"
            log.reviewed_at = datetime.utcnow()
            log.reviewed_by = user_id
            log.review_feedback = feedback

            db.commit()
            logger.info(f"Review rejected for log_id: {log_id}, feedback: {feedback}")
            return {
                "message": "Query rejected with feedback",
                "log_id": log_id,
                "feedback": feedback,
            }
        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain")
async def initiate_retrain(request: Request, admin=Depends(_admin_dependency)):
    """Initiate model retraining (stub - in development)"""
    try:
        logger.info("Model retraining requested")
        return {
            "message": "Retraining initiated. This feature is currently in development.",
            "status": "queued",
            "estimated_time_hours": 2,
        }
    except Exception as e:
        logger.error(f"Error initiating retrain: {e}")
        raise HTTPException(status_code=500, detail=str(e))
