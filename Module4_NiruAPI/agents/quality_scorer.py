"""
Quality Scorer for Fine-tuning Dataset
Evaluates query-response pairs for training data quality
"""

import json
from typing import List, Dict, Any, Optional
from loguru import logger
from pydantic import BaseModel

from Module3_NiruDB.chat_models import (
    QualityScoreResult,
    TrainingDataCreate,
    TrainingDataset,
)
from Module4_NiruAPI.agents.amaniq_v2 import MoonshotClient, AmaniQConfig

# =============================================================================
# SCORING PROMPT
# =============================================================================

QUALITY_SCORING_PROMPT = """You are evaluating an AmaniQuery interaction for fine-tuning quality.

Rate this interaction on a scale of 1-5 based on these criteria:

**Criteria (each rated 1-5):**
1. **Factual Correctness**: Claims are verifiable from sources, no errors
2. **Citation Quality**: Proper format, URLs provided, relevant excerpts
3. **No Hallucinations**: No invented cases, statutes, or facts
4. **Kenyan Context**: Uses proper Kenyan legal terminology and nuances
5. **User Satisfaction**: Quality of answer (infer from feedback if provided)

**Interaction:**
User Query: {query}

Assistant Response: {answer}

Sources Cited: {sources}

User Feedback: {feedback}

**Return JSON with this exact schema:**
{{
  "score": 4.2,
  "keep_for_finetune": true,
  "criteria": {{
    "factual_correctness": 5,
    "citation_quality": 4,
    "no_hallucination": 5,
    "kenyan_context": 4,
    "user_satisfaction": 4
  }},
  "reason": "Excellent response with proper citations to Constitution and case law. Uses correct Kenyan legal terminology. Minor: could provide more context on BBI case history."
}}

**Scoring Guidelines:**
- Score >= 4.5: Exceptional, definitely keep
- Score 3.5-4.4: Good, likely keep
- Score 2.5-3.4: Fair, manual review needed
- Score < 2.5: Poor, discard

Return ONLY valid JSON, no markdown."""


# =============================================================================
# QUALITY SCORER
# =============================================================================


class QualityScorer:
    """Scores query-response pairs for fine-tuning dataset quality"""

    def __init__(self, moonshot_client=None):
        self.moonshot_client = moonshot_client
        if not self.moonshot_client:
            config = AmaniQConfig()
            self.moonshot_client = MoonshotClient.get_client(config)

    def score_interaction(
        self,
        query: str,
        answer: str,
        sources: Optional[List[Dict]] = None,
        feedback: Optional[str] = None,
    ) -> QualityScoreResult:
        """
        Score a single interaction for training quality.

        Args:
            query: User's question
            answer: Assistant's response
            sources: List of sources cited
            feedback: User feedback (like/dislike/comment)

        Returns:
            QualityScoreResult with score and detailed criteria
        """
        # Format sources for prompt
        sources_str = self._format_sources(sources) if sources else "No sources cited"

        # Format feedback
        feedback_str = feedback if feedback else "No explicit feedback"

        # Build prompt
        prompt = QUALITY_SCORING_PROMPT.format(
            query=query,
            answer=answer[:2000],  # Truncate long answers
            sources=sources_str,
            feedback=feedback_str,
        )

        logger.info(f"Scoring interaction: {query[:50]}...")

        try:
            response = self.moonshot_client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,  # Low temp for consistent scoring
                max_tokens=500,
            )

            result = json.loads(response.choices[0].message.content)

            return QualityScoreResult(
                score=result["score"],
                keep_for_finetune=result["keep_for_finetune"],
                criteria=result["criteria"],
                reason=result["reason"],
            )
        except Exception as e:
            logger.error(f"Failed to score interaction: {e}")
            # Return conservative default
            return QualityScoreResult(
                score=2.0,
                keep_for_finetune=False,
                criteria={},
                reason=f"Scoring failed: {str(e)}",
            )

    def batch_score(
        self, interactions: List[Dict[str, Any]], batch_size: int = 10
    ) -> List[QualityScoreResult]:
        """
        Score multiple interactions efficiently.

        Args:
            interactions: List of interaction dicts with query, answer, sources, feedback
            batch_size: Process in batches to avoid rate limits

        Returns:
            List of QualityScoreResult objects
        """
        results = []

        for i in range(0, len(interactions), batch_size):
            batch = interactions[i : i + batch_size]
            logger.info(
                f"Scoring batch {i//batch_size + 1}/{(len(interactions) + batch_size - 1)//batch_size}"
            )

            for interaction in batch:
                result = self.score_interaction(
                    query=interaction.get("query", ""),
                    answer=interaction.get("answer", ""),
                    sources=interaction.get("sources"),
                    feedback=interaction.get("feedback"),
                )
                results.append(result)

        logger.info(
            f"Scored {len(results)} interactions. Kept: {sum(1 for r in results if r.keep_for_finetune)}"
        )
        return results

    def _format_sources(self, sources: List[Dict]) -> str:
        """Format sources list for readability"""
        if not sources:
            return "None"

        formatted = []
        for i, src in enumerate(sources[:5], 1):  # Limit to first 5
            title = src.get("title", "Untitled")
            url = src.get("url", "")
            formatted.append(f"{i}. {title} - {url}")

        return "\n".join(formatted)

    def save_to_dataset(
        self,
        interaction: Dict[str, Any],
        score_result: QualityScoreResult,
        chat_manager,
    ) -> Optional[int]:
        """
        Save scored interaction to training dataset.

        Args:
            interaction: Original interaction data
            score_result: Quality scoring result
            chat_manager: Database manager

        Returns:
            Dataset entry ID or None if failed
        """
        try:
            with chat_manager._get_db_session() as db:
                entry = TrainingDataset(
                    message_id=interaction.get("message_id"),
                    user_query=interaction["query"],
                    assistant_response=interaction["answer"],
                    sources=interaction.get("sources"),
                    quality_score=score_result.score,
                    score_criteria=score_result.criteria,
                    keep_for_finetune=score_result.keep_for_finetune,
                    scoring_reason=score_result.reason,
                    intent=interaction.get("intent"),
                    expertise_level=interaction.get("expertise_level"),
                    cluster_tags=interaction.get("cluster_tags"),
                )
                db.add(entry)
                db.commit()
                db.refresh(entry)

                logger.info(
                    f"Saved to training dataset: ID={entry.id}, Score={score_result.score:.1f}"
                )
                return entry.id
        except Exception as e:
            logger.error(f"Failed to save to dataset: {e}")
            return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def score_and_save_interaction(message_id: str, chat_manager) -> Optional[int]:
    """
    Score a message and save to training dataset if high quality.

    Args:
        message_id: ID of the chat message
        chat_manager: Database manager

    Returns:
        Training dataset entry ID or None
    """
    try:
        # Get message and its pair
        with chat_manager._get_db_session() as db:
            from Module3_NiruDB.chat_models import ChatMessage

            user_msg = (
                db.query(ChatMessage)
                .filter(ChatMessage.id == message_id, ChatMessage.role == "user")
                .first()
            )

            if not user_msg:
                logger.warning(f"Message {message_id} not found")
                return None

            # Find assistant response
            assistant_msg = (
                db.query(ChatMessage)
                .filter(
                    ChatMessage.session_id == user_msg.session_id,
                    ChatMessage.role == "assistant",
                    ChatMessage.created_at > user_msg.created_at,
                )
                .order_by(ChatMessage.created_at)
                .first()
            )

            if not assistant_msg:
                logger.warning(f"No assistant response found for {message_id}")
                return None

            # Prepare interaction data
            interaction = {
                "message_id": message_id,
                "query": user_msg.content,
                "answer": assistant_msg.content,
                "sources": assistant_msg.sources,
                "feedback": user_msg.feedback_type,
            }

        # Score it
        scorer = QualityScorer()
        score_result = scorer.score_interaction(
            query=interaction["query"],
            answer=interaction["answer"],
            sources=interaction["sources"],
            feedback=interaction["feedback"],
        )

        # Save if quality threshold met
        if score_result.keep_for_finetune:
            return scorer.save_to_dataset(interaction, score_result, chat_manager)
        else:
            logger.info(
                f"Interaction scored {score_result.score:.1f}, not keeping for training"
            )
            return None

    except Exception as e:
        logger.error(f"Error in score_and_save_interaction: {e}")
        return None
