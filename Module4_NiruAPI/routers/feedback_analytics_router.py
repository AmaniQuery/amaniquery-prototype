"""
Feedback Analytics Router
Advanced feedback analytics and training integration
"""

from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from Module3_NiruDB.chat_manager_v2 import get_chat_manager

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback_analytics"])


@router.get("/analytics")
async def get_feedback_analytics(
    days: int = Query(30, ge=1, le=365, description="Time window in days")
):
    """
    Get comprehensive feedback analytics

    Returns:
        - Feedback distribution (positive/negative/neutral)
        - Quality score correlations
        - Training dataset candidates
        - Improvement trends
    """
    try:
        chat_manager = get_chat_manager()

        # Get feedback stats from chat manager
        feedback_stats = chat_manager.get_feedback_stats()

        # Get training dataset stats
        from Module4_NiruAPI.agents.dataset_generator import DatasetGenerator

        generator = DatasetGenerator()
        training_stats = generator.get_dataset_stats()

        # Calculate correlations
        total_feedback = feedback_stats.get("total_feedback", 0)
        positive_feedback = feedback_stats.get("positive_count", 0)
        training_candidates = training_stats.get("kept_for_training", 0)

        # Calculate conversion rate (positive feedback → training)
        conversion_rate = (
            (training_candidates / positive_feedback * 100)
            if positive_feedback > 0
            else 0
        )

        return {
            "time_window_days": days,
            "feedback_distribution": {
                "total": total_feedback,
                "positive": positive_feedback,
                "negative": feedback_stats.get("negative_count", 0),
                "positive_rate": (
                    (positive_feedback / total_feedback * 100)
                    if total_feedback > 0
                    else 0
                ),
            },
            "training_impact": {
                "total_scored": training_stats.get("total_scored", 0),
                "training_candidates": training_candidates,
                "awaiting_export": training_stats.get("awaiting_export", 0),
                "conversion_rate": f"{conversion_rate:.1f}%",
                "average_quality_score": training_stats.get("average_score", 0),
            },
            "score_distribution": training_stats.get("score_distribution", {}),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get feedback analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-impact")
async def get_training_impact():
    """
    Show how feedback contributes to training dataset

    Returns:
        - Feedback-approved training examples
        - Quality score improvements
        - Export readiness
    """
    try:
        from Module4_NiruAPI.agents.dataset_generator import DatasetGenerator

        generator = DatasetGenerator()

        # Get high-quality interactions
        high_quality = generator.get_high_quality_interactions(
            min_score=4.0, limit=10, exported=False
        )

        stats = generator.get_dataset_stats()

        return {
            "total_training_candidates": stats.get("kept_for_training", 0),
            "high_quality_preview": [
                {
                    "query": (
                        item.user_query[:100] + "..."
                        if len(item.user_query) > 100
                        else item.user_query
                    ),
                    "quality_score": item.quality_score,
                    "has_positive_feedback": "positive"
                    in (item.scoring_reason or "").lower(),
                }
                for item in high_quality[:5]
            ],
            "export_ready": stats.get("awaiting_export", 0),
            "average_quality": stats.get("average_score", 0),
            "distribution": stats.get("score_distribution", {}),
        }
    except Exception as e:
        logger.error(f"Failed to get training impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reviewed")
async def get_reviewed_interactions(
    min_score: float = Query(4.0, ge=1.0, le=5.0), limit: int = Query(50, ge=1, le=200)
):
    """
    Get interactions with positive feedback for manual review

    Args:
        min_score: Minimum quality score
        limit: Maximum interactions to return

    Returns:
        List of reviewed interactions ready for training
    """
    try:
        from Module4_NiruAPI.agents.dataset_generator import DatasetGenerator

        generator = DatasetGenerator()

        interactions = generator.get_high_quality_interactions(
            min_score=min_score, limit=limit, exported=False
        )

        return {
            "count": len(interactions),
            "min_score": min_score,
            "interactions": [
                {
                    "id": item.id,
                    "query": item.user_query,
                    "quality_score": item.quality_score,
                    "intent": item.intent,
                    "cluster_tags": item.cluster_tags,
                    "created_at": (
                        item.created_at.isoformat() if item.created_at else None
                    ),
                }
                for item in interactions
            ],
        }
    except Exception as e:
        logger.error(f"Failed to get reviewed interactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
