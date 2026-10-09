"""
Clustering API Router
Endpoints for task cluster analysis and management
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from Module3_NiruDB.chat_models import (
    TaskClusterCreate,
    TaskClusterResponse,
    ClusterSuggestion,
)
from Module4_NiruAPI.agents.task_clustering import (
    ClusterAnalyzer,
    run_clustering_analysis,
)

router = APIRouter(prefix="/api/v1/clusters", tags=["clustering"])


# =============================================================================
# STATE CONTAINER
# =============================================================================


class RouterState:
    """State container for router dependencies"""

    cluster_analyzer: Optional[ClusterAnalyzer] = None


_state = RouterState()


def get_analyzer() -> ClusterAnalyzer:
    """Get or create cluster analyzer instance"""
    if _state.cluster_analyzer is None:
        _state.cluster_analyzer = ClusterAnalyzer()
    return _state.cluster_analyzer


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/analyze")
async def analyze_clusters(
    query_limit: int = Query(
        50, ge=10, le=200, description="Number of recent queries to analyze"
    ),
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
):
    """
    Trigger cluster analysis on recent queries.

    Returns suggested new clusters based on query patterns.
    """
    try:
        logger.info(
            f"Starting cluster analysis: {query_limit} queries, {hours}h window"
        )
        result = run_clustering_analysis(query_limit=query_limit, hours=hours)
        return result
    except Exception as e:
        logger.error(f"Cluster analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[TaskClusterResponse])
async def list_clusters(
    active_only: bool = Query(True, description="Only return active clusters")
):
    """List all task clusters"""
    try:
        analyzer = get_analyzer()
        clusters = analyzer.get_all_clusters()

        if active_only:
            clusters = [c for c in clusters if c.is_active]

        return clusters
    except Exception as e:
        logger.error(f"Failed to list clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=dict)
async def create_cluster(cluster: TaskClusterCreate):
    """Create a new task cluster"""
    try:
        analyzer = get_analyzer()
        cluster_id = analyzer.create_cluster(cluster)

        if cluster_id is None:
            raise HTTPException(status_code=400, detail="Failed to create cluster")

        return {
            "id": cluster_id,
            "cluster_name": cluster.cluster_name,
            "status": "created",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/merge")
async def merge_suggestion(suggestion: ClusterSuggestion):
    """
    Merge a suggested cluster into the active taxonomy.

    This converts a suggestion into an actual cluster.
    """
    try:
        analyzer = get_analyzer()

        # Check confidence threshold
        if suggestion.confidence < 0.7:
            raise HTTPException(
                status_code=400,
                detail=f"Confidence too low ({suggestion.confidence:.2f}). Minimum is 0.70",
            )

        # Create cluster from suggestion
        cluster_data = TaskClusterCreate(
            cluster_name=suggestion.group_name,
            description=suggestion.description,
            representative_queries=suggestion.representative_queries,
            metadata_tags=suggestion.suggested_metadata_tags,
        )

        cluster_id = analyzer.create_cluster(cluster_data)

        if cluster_id is None:
            raise HTTPException(status_code=400, detail="Failed to merge cluster")

        logger.info(
            f"Merged cluster suggestion: {suggestion.group_name} (ID: {cluster_id})"
        )

        return {
            "id": cluster_id,
            "cluster_name": suggestion.group_name,
            "status": "merged",
            "confidence": suggestion.confidence,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to merge suggestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_cluster_stats():
    """Get clustering statistics"""
    try:
        analyzer = get_analyzer()
        clusters = analyzer.get_all_clusters()

        total_clusters = len(clusters)
        active_clusters = sum(1 for c in clusters if c.is_active)
        total_queries = sum(c.query_count for c in clusters)

        # Get top 5 clusters by query count
        top_clusters = sorted(clusters, key=lambda c: c.query_count, reverse=True)[:5]

        return {
            "total_clusters": total_clusters,
            "active_clusters": active_clusters,
            "total_classified_queries": total_queries,
            "average_queries_per_cluster": (
                total_queries / total_clusters if total_clusters > 0 else 0
            ),
            "top_clusters": [
                {
                    "name": c.cluster_name,
                    "query_count": c.query_count,
                    "description": c.description,
                }
                for c in top_clusters
            ],
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
