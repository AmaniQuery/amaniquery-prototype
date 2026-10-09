"""
Dataset Generator for Fine-tuning
Converts scored interactions to training formats (Alpaca, ShareGPT)
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from loguru import logger

from Module3_NiruDB.chat_models import TrainingDataset, TrainingDataResponse
from Module3_NiruDB.chat_manager_v2 import get_chat_manager

# =============================================================================
# DATASET GENERATOR
# =============================================================================


class DatasetGenerator:
    """Converts scored interactions to fine-tuning formats"""

    def __init__(self, chat_manager=None):
        self.chat_manager = chat_manager or get_chat_manager()

    def to_alpaca_format(self, interaction: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert to Alpaca instruction format.

        Format:
        {
          "instruction": "user query",
          "input": "",
          "output": "formatted ideal answer with citations"
        }
        """
        return {
            "instruction": interaction["user_query"],
            "input": "",
            "output": self._format_ideal_answer(
                interaction["assistant_response"], interaction.get("sources", [])
            ),
        }

    def to_sharegpt_format(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert to ShareGPT conversation format.

        Format:
        {
          "conversations": [
            {"from": "human", "value": "query"},
            {"from": "gpt", "value": "answer"}
          ],
          "source": "amaniquery_production",
          "metadata": {...}
        }
        """
        return {
            "conversations": [
                {"from": "human", "value": interaction["user_query"]},
                {"from": "gpt", "value": interaction["assistant_response"]},
            ],
            "source": "amaniquery_production",
            "quality_score": interaction.get("quality_score"),
            "metadata": {
                "intent": interaction.get("intent"),
                "cluster_tags": interaction.get("cluster_tags"),
                "expertise_level": interaction.get("expertise_level"),
                "has_sources": len(interaction.get("sources", [])) > 0,
            },
        }

    def _format_ideal_answer(self, answer: str, sources: List[Dict]) -> str:
        """
        Format answer with proper citations for training.
        Ensures consistent citation format.
        """
        formatted_answer = answer

        # If sources exist, ensure they're properly formatted
        if sources:
            citations_section = "\n\n**Relevant Citations:**\n"
            for i, source in enumerate(sources[:5], 1):
                title = source.get("title", "Untitled")
                url = source.get("url", "")
                if url:
                    citations_section += f"{i}. [{title}]({url})\n"
                else:
                    citations_section += f"{i}. {title}\n"

            # Add citations if not already present
            if (
                "**Relevant Citations:**" not in formatted_answer
                and "**Sources:**" not in formatted_answer
            ):
                formatted_answer += citations_section

        return formatted_answer

    def get_high_quality_interactions(
        self, min_score: float = 4.0, limit: int = 1000, exported: bool = False
    ) -> List[TrainingDataResponse]:
        """
        Get high-quality interactions from database.

        Args:
            min_score: Minimum quality score
            limit: Maximum number of interactions
            exported: If False, only get unexported data

        Returns:
            List of TrainingDataResponse objects
        """
        try:
            with self.chat_manager._get_db_session() as db:
                query = db.query(TrainingDataset).filter(
                    TrainingDataset.keep_for_finetune == True,
                    TrainingDataset.quality_score >= min_score,
                )

                if not exported:
                    query = query.filter(TrainingDataset.exported_at == None)

                results = (
                    query.order_by(TrainingDataset.quality_score.desc())
                    .limit(limit)
                    .all()
                )

                return [
                    TrainingDataResponse(
                        id=r.id,
                        user_query=r.user_query,
                        assistant_response=r.assistant_response,
                        sources=r.sources,
                        quality_score=r.quality_score,
                        score_criteria=r.score_criteria,
                        keep_for_finetune=r.keep_for_finetune,
                        scoring_reason=r.scoring_reason,
                        intent=r.intent,
                        expertise_level=r.expertise_level,
                        cluster_tags=r.cluster_tags,
                        exported_at=r.exported_at,
                        export_format=r.export_format,
                        created_at=r.created_at,
                    )
                    for r in results
                ]
        except Exception as e:
            logger.error(f"Failed to get high-quality interactions: {e}")
            return []

    def export_dataset(
        self,
        output_path: str,
        format: str = "alpaca",
        min_score: float = 4.0,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """
        Export high-quality interactions to JSONL file.

        Args:
            output_path: Path to output file
            format: Export format ('alpaca' or 'sharegpt')
            min_score: Minimum quality score
            limit: Maximum number of interactions

        Returns:
            Export statistics
        """
        logger.info(f"Exporting dataset to {output_path} in {format} format...")

        # Get data
        interactions = self.get_high_quality_interactions(
            min_score=min_score, limit=limit, exported=False
        )

        if not interactions:
            logger.warning("No interactions to export")
            return {"exported_count": 0, "output_path": output_path, "format": format}

        # Convert to format
        exported_data = []
        for interaction in interactions:
            interaction_dict = {
                "user_query": interaction.user_query,
                "assistant_response": interaction.assistant_response,
                "sources": interaction.sources,
                "quality_score": interaction.quality_score,
                "intent": interaction.intent,
                "cluster_tags": interaction.cluster_tags,
                "expertise_level": interaction.expertise_level,
            }

            if format == "alpaca":
                exported_data.append(self.to_alpaca_format(interaction_dict))
            elif format == "sharegpt":
                exported_data.append(self.to_sharegpt_format(interaction_dict))
            else:
                raise ValueError(f"Unsupported format: {format}")

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for item in exported_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Mark as exported
        self._mark_as_exported([i.id for i in interactions], format)

        stats = {
            "exported_count": len(exported_data),
            "output_path": str(output_file),
            "format": format,
            "min_score": min_score,
            "avg_score": sum(i.quality_score for i in interactions) / len(interactions),
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(f"Exported {len(exported_data)} interactions to {output_file}")
        return stats

    def _mark_as_exported(self, interaction_ids: List[int], format: str):
        """Mark interactions as exported"""
        try:
            with self.chat_manager._get_db_session() as db:
                from sqlalchemy import update

                stmt = (
                    update(TrainingDataset)
                    .where(TrainingDataset.id.in_(interaction_ids))
                    .values(exported_at=datetime.utcnow(), export_format=format)
                )
                db.execute(stmt)
                db.commit()

                logger.info(f"Marked {len(interaction_ids)} interactions as exported")
        except Exception as e:
            logger.error(f"Failed to mark as exported: {e}")

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the training dataset"""
        try:
            with self.chat_manager._get_db_session() as db:
                from sqlalchemy import func

                total = db.query(func.count(TrainingDataset.id)).scalar()
                kept = (
                    db.query(func.count(TrainingDataset.id))
                    .filter(TrainingDataset.keep_for_finetune == True)
                    .scalar()
                )
                exported = (
                    db.query(func.count(TrainingDataset.id))
                    .filter(TrainingDataset.exported_at != None)
                    .scalar()
                )

                avg_score = db.query(func.avg(TrainingDataset.quality_score)).scalar()

                # Score distribution
                score_buckets = {
                    "excellent_4.5+": db.query(func.count(TrainingDataset.id))
                    .filter(TrainingDataset.quality_score >= 4.5)
                    .scalar(),
                    "good_3.5-4.5": db.query(func.count(TrainingDataset.id))
                    .filter(
                        TrainingDataset.quality_score >= 3.5,
                        TrainingDataset.quality_score < 4.5,
                    )
                    .scalar(),
                    "fair_2.5-3.5": db.query(func.count(TrainingDataset.id))
                    .filter(
                        TrainingDataset.quality_score >= 2.5,
                        TrainingDataset.quality_score < 3.5,
                    )
                    .scalar(),
                    "poor_below_2.5": db.query(func.count(TrainingDataset.id))
                    .filter(TrainingDataset.quality_score < 2.5)
                    .scalar(),
                }

                return {
                    "total_scored": total or 0,
                    "kept_for_training": kept or 0,
                    "exported": exported or 0,
                    "awaiting_export": (kept or 0) - (exported or 0),
                    "average_score": float(avg_score) if avg_score else 0.0,
                    "score_distribution": score_buckets,
                }
        except Exception as e:
            logger.error(f"Failed to get dataset stats: {e}")
            return {}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def export_training_dataset(
    output_dir: str = "./training_data", format: str = "alpaca", min_score: float = 4.0
) -> Dict[str, Any]:
    """
    Convenience function to export training dataset.

    Args:
        output_dir: Output directory
        format: Export format
        min_score: Minimum quality score

    Returns:
        Export statistics
    """
    generator = DatasetGenerator()

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"amaniquery_training_{format}_{timestamp}.jsonl"
    output_path = Path(output_dir) / filename

    return generator.export_dataset(
        output_path=str(output_path), format=format, min_score=min_score
    )
