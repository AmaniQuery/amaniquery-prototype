"""
Feedback Handler - Manages feedback loops and self-critique
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger


class FeedbackHandler:
    """
    Handles feedback loops and self-critique mechanisms
    """

    def __init__(self):
        """Initialize feedback handler"""
        self.feedback_history: List[Dict[str, Any]] = []

    def process_feedback(
        self,
        query: str,
        answer: str,
        user_feedback: Optional[str] = None,
        rating: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process user feedback

        Args:
            query: Original query
            answer: Generated answer
            user_feedback: Optional user feedback text
            rating: Optional rating (0-1)

        Returns:
            Processed feedback data
        """
        feedback = {
            "query": query,
            "answer": answer[:500],  # Truncate for storage
            "user_feedback": user_feedback,
            "rating": rating,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.feedback_history.append(feedback)

        # Keep only last 1000 feedback items
        if len(self.feedback_history) > 1000:
            self.feedback_history = self.feedback_history[-1000:]

        logger.info(
            f"Processed feedback: rating={rating}, has_text={bool(user_feedback)}"
        )

        return feedback

    def self_critique(
        self,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        reflection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform self-critique of the answer

        Args:
            query: Original query
            answer: Generated answer
            sources: Sources used
            reflection: Optional reflection text

        Returns:
            Critique results
        """
        critique = {
            "query": query,
            "answer_length": len(answer),
            "sources_count": len(sources),
            "issues": [],
            "suggestions": [],
            "score": 0.0,
        }

        # Check answer length
        if len(answer) < 100:
            critique["issues"].append("Answer is too short")
            critique["suggestions"].append("Provide more detailed information")
        elif len(answer) > 5000:
            critique["issues"].append("Answer is very long")
            critique["suggestions"].append("Consider summarizing key points")

        # Check sources
        if len(sources) == 0:
            critique["issues"].append("No sources provided")
            critique["suggestions"].append("Add source citations")
        elif len(sources) < 3:
            critique["issues"].append("Few sources")
            critique["suggestions"].append("Consider adding more sources")

        # Calculate score
        score = 0.5  # Base score

        if 100 <= len(answer) <= 5000:
            score += 0.2

        if len(sources) >= 3:
            score += 0.2

        if len(sources) >= 5:
            score += 0.1

        critique["score"] = min(score, 1.0)

        return critique

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about feedback"""
        if not self.feedback_history:
            return {"total_feedback": 0, "average_rating": 0.0, "feedback_with_text": 0}

        ratings = [
            f["rating"] for f in self.feedback_history if f.get("rating") is not None
        ]
        feedback_with_text = sum(
            1 for f in self.feedback_history if f.get("user_feedback")
        )

        return {
            "total_feedback": len(self.feedback_history),
            "average_rating": sum(ratings) / len(ratings) if ratings else 0.0,
            "feedback_with_text": feedback_with_text,
        }
