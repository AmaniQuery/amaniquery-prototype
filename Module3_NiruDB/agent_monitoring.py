"""
Agent Monitoring Helper Functions
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import func, case
import logging

from .agent_models import AgentQueryLog, generate_query_log_id

logger = logging.getLogger(__name__)


def log_agent_query(
    db: Session,
    session_id: str,
    query: str,
    result: Dict[str, Any],
    response_time_ms: int,
    user_id: Optional[str] = None,
) -> Optional[str]:
    """
    Log an agent query execution to the database

    Args:
        db: Database session
        session_id: Chat session ID
        query: User query text
        result: Agent execution result with metadata
        response_time_ms: Response time in milliseconds
        user_id: Optional user ID

    Returns:
        Log entry ID or None if logging failed
    """
    try:
        metadata = result.get("metadata", {})

        log_entry = AgentQueryLog(
            id=generate_query_log_id(),
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            query=query,
            persona=metadata.get("persona", "wanjiku"),
            intent=metadata.get("intent_classification", "general"),
            confidence=metadata.get("confidence", 0.0),
            response_time_ms=response_time_ms,
            evidence_count=metadata.get("evidence_count", 0),
            reasoning_steps=metadata.get("reasoning_steps", 0),
            human_review_required=metadata.get("human_review_required", False),
            agent_path=result.get("agent_path", []),
            quality_issues=result.get("quality_issues", []),
            reasoning_path=result.get("reasoning_path", {}),
        )

        db.add(log_entry)
        db.commit()

        logger.info(f"Logged agent query: {log_entry.id}")
        return log_entry.id

    except Exception as e:
        logger.error(f"Failed to log agent query: {e}")
        db.rollback()
        return None


def get_agent_metrics(db: Session, days: int = 30) -> Dict[str, Any]:
    """
    Calculate agent performance metrics

    Args:
        db: Database session
        days: Number of days to include in analysis

    Returns:
        Dictionary with aggregated metrics
    """
    try:
        start_date = datetime.utcnow() - timedelta(days=days)

        # Get all logs in date range
        logs = (
            db.query(AgentQueryLog).filter(AgentQueryLog.timestamp >= start_date).all()
        )

        if not logs:
            return {
                "total_queries": 0,
                "avg_confidence": 0.0,
                "avg_response_time_ms": 0,
                "human_review_rate": 0.0,
                "persona_distribution": {"wanjiku": 0, "wakili": 0, "mwanahabari": 0},
                "intent_distribution": {"news": 0, "law": 0, "hybrid": 0, "general": 0},
                "confidence_buckets": {"low": 0, "medium": 0, "high": 0},
            }

        total_queries = len(logs)
        avg_confidence = sum(log.confidence for log in logs) / total_queries
        avg_response_time = sum(log.response_time_ms for log in logs) / total_queries
        human_review_count = sum(1 for log in logs if log.human_review_required)

        # Persona distribution
        persona_dist = {"wanjiku": 0, "wakili": 0, "mwanahabari": 0}
        for log in logs:
            if log.persona in persona_dist:
                persona_dist[log.persona] += 1

        # Intent distribution
        intent_dist = {"news": 0, "law": 0, "hybrid": 0, "general": 0}
        for log in logs:
            if log.intent in intent_dist:
                intent_dist[log.intent] += 1

        # Confidence buckets
        conf_buckets = {"low": 0, "medium": 0, "high": 0}
        for log in logs:
            if log.confidence < 0.6:
                conf_buckets["low"] += 1
            elif log.confidence < 0.8:
                conf_buckets["medium"] += 1
            else:
                conf_buckets["high"] += 1

        return {
            "total_queries": total_queries,
            "avg_confidence": round(avg_confidence, 3),
            "avg_response_time_ms": round(avg_response_time),
            "human_review_rate": round(human_review_count / total_queries, 3),
            "persona_distribution": persona_dist,
            "intent_distribution": intent_dist,
            "confidence_buckets": conf_buckets,
        }

    except Exception as e:
        logger.error(f"Failed to calculate agent metrics: {e}")
        return {
            "total_queries": 0,
            "avg_confidence": 0.0,
            "avg_response_time_ms": 0,
            "human_review_rate": 0.0,
            "persona_distribution": {"wanjiku": 0, "wakili": 0, "mwanahabari": 0},
            "intent_distribution": {"news": 0, "law": 0, "hybrid": 0, "general": 0},
            "confidence_buckets": {"low": 0, "medium": 0, "high": 0},
        }


def get_review_queue(db: Session) -> List[Dict[str, Any]]:
    """
    Get queries pending human review

    Args:
        db: Database session

    Returns:
        List of queries needing review with priority
    """
    try:
        logs = (
            db.query(AgentQueryLog)
            .filter(
                AgentQueryLog.human_review_required == True,
                AgentQueryLog.review_status.is_(None),
            )
            .order_by(
                AgentQueryLog.confidence.asc(),  # Low confidence first
                AgentQueryLog.timestamp.desc(),
            )
            .limit(100)
            .all()
        )

        queue = []
        for log in logs:
            # Determine priority
            if log.confidence < 0.4:
                priority = "high"
            elif log.confidence < 0.6:
                priority = "medium"
            else:
                priority = "low"

            # Determine review reason
            reasons = []
            if log.confidence < 0.6:
                reasons.append("Low confidence score")
            if log.quality_issues:
                reasons.extend(log.quality_issues)
            if log.intent == "law":
                reasons.append("Legal query - requires expert review")

            queue.append(
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
                    "review_reason": "; ".join(reasons),
                    "priority": priority,
                }
            )

        return queue

    except Exception as e:
        logger.error(f"Failed to get review queue: {e}")
        return []
