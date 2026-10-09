"""
Task Clustering Module for AmaniQuery
Analyzes user queries to discover emerging task patterns and suggest new clusters.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field

from Module3_NiruDB.chat_manager_v2 import get_chat_manager
from Module3_NiruDB.chat_models import (
    TaskCluster,
    ClusterSuggestion,
    TaskClusterCreate,
    TaskClusterResponse,
)
from Module4_NiruAPI.agents.amaniq_v2 import MoonshotClient, AmaniQConfig

# =============================================================================
# CLUSTERING PROMPT
# =============================================================================

CLUSTERING_PROMPT_TEMPLATE = """You are maintaining an evolving taxonomy of tasks for AmaniQuery, a Kenyan legal research assistant.

Here are the current top {cluster_count} task clusters (with 3 example queries each):
{current_clusters}

Now analyze these {query_count} recent queries (with their extracted intents):
{recent_queries}

Your task:
1. Identify up to 3 new meaningful task groups if patterns emerge
2. For each new group, provide:
   - group_name: Short descriptive name (e.g., "Constitutional Law Research")
   - description: What characterizes this task group (1-2 sentences)
   - representative_queries: 3 example queries from the data
   - suggested_metadata_tags: 3-5 relevant tags
   - confidence: Your confidence score (0.0 to 1.0)

IMPORTANT RULES:
- Only suggest NEW clusters that don't overlap with existing ones
- A cluster must have at least 5 queries in the recent data
- Confidence < 0.7 = weak pattern, don't suggest
- Focus on task TYPE, not topic (e.g., "Statute Comparison" not "Land Law")

Return JSON with this schema:
{
  "suggestions": [
    {
      "group_name": "...",
      "description": "...",
      "representative_queries": ["...", "...", "..."],
      "suggested_metadata_tags": ["...", "..."],
      "confidence": 0.85
    }
  ],
  "analysis_summary": "Brief summary of patterns found"
}

Return ONLY valid JSON, no markdown."""


# =============================================================================
# CLUSTER ANALYZER
# =============================================================================


class ClusterAnalyzer:
    """Analyzes queries and suggests new task clusters"""

    def __init__(self, chat_manager=None, moonshot_client=None):
        self.chat_manager = chat_manager or get_chat_manager()
        self.moonshot_client = moonshot_client
        if not self.moonshot_client:
            config = AmaniQConfig()
            self.moonshot_client = MoonshotClient.get_client(config)

    def get_current_clusters(self, limit: int = 12) -> List[Dict[str, Any]]:
        """Get current active task clusters from database"""
        try:
            from Module3_NiruDB.chat_models import Base, create_database_engine
            from sqlalchemy.orm import sessionmaker

            # Get database session
            with self.chat_manager._get_db_session() as db:
                clusters = (
                    db.query(TaskCluster)
                    .filter(TaskCluster.is_active == True)
                    .order_by(TaskCluster.query_count.desc())
                    .limit(limit)
                    .all()
                )

                return [
                    {
                        "cluster_name": c.cluster_name,
                        "description": c.description,
                        "representative_queries": c.representative_queries,
                        "query_count": c.query_count,
                    }
                    for c in clusters
                ]
        except Exception as e:
            logger.warning(f"Failed to fetch current clusters: {e}")
            return []

    def analyze_recent_queries(
        self, limit: int = 50, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Fetch and analyze recent queries"""
        try:
            # Get recent queries
            recent_queries = self.chat_manager.get_recent_queries(
                limit=limit, hours=hours
            )

            # Get intent information
            query_intents = self.chat_manager.get_query_intents(limit=limit)

            # Merge intent data with queries
            intent_map = {qi["query"]: qi.get("intent") for qi in query_intents}

            for query in recent_queries:
                query["intent"] = intent_map.get(query["content"])

            return recent_queries
        except Exception as e:
            logger.error(f"Failed to analyze recent queries: {e}")
            return []

    def suggest_new_clusters(
        self,
        recent_queries: List[Dict[str, Any]],
        current_clusters: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Use Moonshot AI to suggest new clusters"""

        # Format current clusters
        clusters_str = "\n".join(
            [
                f"- **{c['cluster_name']}**: {c['description']}\n  Examples: {c['representative_queries'][:3]}"
                for c in current_clusters
            ]
        )

        # Format recent queries
        queries_str = "\n".join(
            [
                f"{i+1}. \"{q['content']}\" (intent: {q.get('intent', 'unknown')})"
                for i, q in enumerate(recent_queries[:50])
            ]
        )

        # Build prompt
        prompt = CLUSTERING_PROMPT_TEMPLATE.format(
            cluster_count=len(current_clusters),
            current_clusters=clusters_str or "No clusters yet",
            query_count=len(recent_queries),
            recent_queries=queries_str,
        )

        logger.info(f"Requesting cluster suggestions from Moonshot AI...")

        try:
            response = self.moonshot_client.chat.completions.create(
                model="moonshot-v1-32k",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,  # Lower temp for more consistent analysis
                max_tokens=2000,
            )

            result = json.loads(response.choices[0].message.content)
            logger.info(
                f"Received {len(result.get('suggestions', []))} cluster suggestions"
            )

            return result
        except Exception as e:
            logger.error(f"Failed to get cluster suggestions: {e}")
            return {"suggestions": [], "analysis_summary": f"Error: {str(e)}"}

    def create_cluster(self, cluster_data: TaskClusterCreate) -> Optional[int]:
        """Create a new task cluster in the database"""
        try:
            with self.chat_manager._get_db_session() as db:
                cluster = TaskCluster(
                    cluster_name=cluster_data.cluster_name,
                    description=cluster_data.description,
                    representative_queries=cluster_data.representative_queries,
                    metadata_tags=cluster_data.metadata_tags or [],
                    query_count=0,
                    is_active=True,
                )
                db.add(cluster)
                db.commit()
                db.refresh(cluster)
                logger.info(
                    f"Created new cluster: {cluster.cluster_name} (ID: {cluster.id})"
                )
                return cluster.id
        except Exception as e:
            logger.error(f"Failed to create cluster: {e}")
            return None

    def get_all_clusters(self) -> List[TaskClusterResponse]:
        """Get all active clusters"""
        try:
            with self.chat_manager._get_db_session() as db:
                clusters = (
                    db.query(TaskCluster)
                    .filter(TaskCluster.is_active == True)
                    .order_by(TaskCluster.query_count.desc())
                    .all()
                )

                return [
                    TaskClusterResponse(
                        id=c.id,
                        cluster_name=c.cluster_name,
                        description=c.description,
                        representative_queries=c.representative_queries,
                        metadata_tags=c.metadata_tags,
                        query_count=c.query_count,
                        is_active=c.is_active,
                        created_at=c.created_at,
                        last_updated=c.last_updated,
                    )
                    for c in clusters
                ]
        except Exception as e:
            logger.error(f"Failed to fetch clusters: {e}")
            return []

    def run_full_analysis(
        self, query_limit: int = 50, hours: int = 24
    ) -> Dict[str, Any]:
        """Run complete clustering analysis pipeline"""
        logger.info("=" * 80)
        logger.info("Starting Task Clustering Analysis")
        logger.info("=" * 80)

        # Step 1: Get current clusters
        current_clusters = self.get_current_clusters(limit=12)
        logger.info(f"Found {len(current_clusters)} existing clusters")

        # Step 2: Analyze recent queries
        recent_queries = self.analyze_recent_queries(limit=query_limit, hours=hours)
        logger.info(f"Analyzed {len(recent_queries)} recent queries from last {hours}h")

        if len(recent_queries) < 10:
            logger.warning("Not enough queries for meaningful analysis")
            return {
                "status": "skipped",
                "reason": "Insufficient query volume",
                "suggestions": [],
            }

        # Step 3: Get suggestions
        result = self.suggest_new_clusters(recent_queries, current_clusters)

        logger.info("=" * 80)
        logger.info(
            f"Analysis complete: {len(result.get('suggestions', []))} suggestions"
        )
        logger.info("=" * 80)

        return {
            "status": "completed",
            "current_clusters": current_clusters,
            "recent_query_count": len(recent_queries),
            "suggestions": result.get("suggestions", []),
            "analysis_summary": result.get("analysis_summary", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def run_clustering_analysis(query_limit: int = 50, hours: int = 24) -> Dict[str, Any]:
    """Convenience function to run clustering analysis"""
    analyzer = ClusterAnalyzer()
    return analyzer.run_full_analysis(query_limit=query_limit, hours=hours)
