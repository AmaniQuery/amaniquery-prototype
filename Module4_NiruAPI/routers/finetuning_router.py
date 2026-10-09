"""
Fine-tuning API Router
Endpoints for quality scoring and dataset export
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from loguru import logger

from Module3_NiruDB.chat_models import TrainingDataResponse, QualityScoreResult
from Module4_NiruAPI.agents.quality_scorer import (
    QualityScorer,
    score_and_save_interaction,
)
from Module4_NiruAPI.agents.dataset_generator import (
    DatasetGenerator,
    export_training_dataset,
)
from Module3_NiruDB.chat_manager_v2 import get_chat_manager

router = APIRouter(prefix="/api/v1/finetuning", tags=["finetuning"])


# =============================================================================
# STATE CONTAINER
# =============================================================================


class RouterState:
    """State container for router dependencies"""

    quality_scorer: Optional[QualityScorer] = None
    dataset_generator: Optional[DatasetGenerator] = None


_state = RouterState()


def get_scorer() -> QualityScorer:
    """Get or create quality scorer instance"""
    if _state.quality_scorer is None:
        _state.quality_scorer = QualityScorer()
    return _state.quality_scorer


def get_generator() -> DatasetGenerator:
    """Get or create dataset generator instance"""
    if _state.dataset_generator is None:
        _state.dataset_generator = DatasetGenerator()
    return _state.dataset_generator


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/score/{message_id}")
async def score_message(message_id: str):
    """
    Score a specific message for training quality.
    Automatically saves to training dataset if score >= threshold.
    """
    try:
        chat_manager = get_chat_manager()
        dataset_id = await score_and_save_interaction(message_id, chat_manager)

        if dataset_id:
            return {
                "message_id": message_id,
                "dataset_id": dataset_id,
                "status": "scored_and_saved",
            }
        else:
            return {
                "message_id": message_id,
                "status": "scored_but_not_saved",
                "reason": "Score below threshold or error occurred",
            }
    except Exception as e:
        logger.error(f"Failed to score message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-score")
async def batch_score_interactions(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    limit: int = Query(100, ge=1, le=500, description="Maximum interactions to score"),
):
    """
    Score recent interactions in batch.
    Useful for catching up on un-scored interactions.
    """
    try:
        chat_manager = get_chat_manager()
        scorer = get_scorer()

        # Get recent interactions
        recent_queries = chat_manager.get_recent_queries(limit=limit, hours=hours)

        logger.info(f"Batch scoring {len(recent_queries)} interactions...")

        scored_count = 0
        saved_count = 0

        for query in recent_queries:
            try:
                # Score and save
                dataset_id = await score_and_save_interaction(query["id"], chat_manager)
                scored_count += 1
                if dataset_id:
                    saved_count += 1
            except Exception as e:
                logger.warning(f"Failed to score query {query['id']}: {e}")
                continue

        return {
            "status": "completed",
            "scored_count": scored_count,
            "saved_count": saved_count,
            "total_candidates": len(recent_queries),
        }
    except Exception as e:
        logger.error(f"Batch scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset", response_model=List[TrainingDataResponse])
async def get_training_dataset(
    min_score: float = Query(4.0, ge=1.0, le=5.0, description="Minimum quality score"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum interactions"),
    exported: bool = Query(False, description="Include already exported data"),
):
    """Get high-quality interactions for review"""
    try:
        generator = get_generator()
        interactions = generator.get_high_quality_interactions(
            min_score=min_score, limit=limit, exported=exported
        )
        return interactions
    except Exception as e:
        logger.error(f"Failed to get dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_dataset(
    format: str = Query(
        "alpaca", regex="^(alpaca|sharegpt)$", description="Export format"
    ),
    min_score: float = Query(4.0, ge=1.0, le=5.0, description="Minimum quality score"),
    limit: int = Query(1000, ge=1, le=5000, description="Maximum interactions"),
):
    """
    Export training dataset to file.
    Returns statistics about the export.
    """
    try:
        stats = export_training_dataset(
            output_dir="./training_data", format=format, min_score=min_score
        )
        return stats
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_dataset_stats():
    """Get statistics about the training dataset"""
    try:
        generator = get_generator()
        stats = generator.get_dataset_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-score-all")
async def auto_score_all_recent(
    days: int = Query(7, ge=1, le=30, description="Score interactions from last N days")
):
    """
    Automatically score ALL recent interactions.
    WARNING: This can be expensive with many interactions.
    """
    try:
        chat_manager = get_chat_manager()
        hours = days * 24

        # Get all recent queries
        recent_queries = chat_manager.get_recent_queries(
            limit=1000, hours=hours  # Process max 1000 at a time
        )

        logger.info(
            f"Auto-scoring {len(recent_queries)} interactions from last {days} days..."
        )

        scored = 0
        saved = 0
        errors = 0

        for query in recent_queries:
            try:
                dataset_id = await score_and_save_interaction(query["id"], chat_manager)
                scored += 1
                if dataset_id:
                    saved += 1
            except Exception as e:
                errors += 1
                logger.warning(f"Error scoring {query['id']}: {e}")

        return {
            "status": "completed",
            "days_processed": days,
            "total_candidates": len(recent_queries),
            "scored": scored,
            "saved_for_training": saved,
            "errors": errors,
            "success_rate": (
                f"{(scored / len(recent_queries) * 100):.1f}%"
                if recent_queries
                else "0%"
            ),
        }
    except Exception as e:
        logger.error(f"Auto-score failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
