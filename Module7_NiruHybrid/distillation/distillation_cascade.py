from typing import List, Dict, Any, Optional, Callable, Union
import time
import logging
import numpy as np
from .teacher_student import TeacherStudentPair

logger = logging.getLogger(__name__)


class DistillationCascade:
    """
    Implements a retrieval cascade using a Teacher-Student pair.

    Pattern:
    1. Student Model (Fast): Retrieves a large set of candidates (Recall focus).
       - Typically a Vector Store Retriever (Bi-Encoder).
    2. Teacher Model (Accurate): Re-ranks a smaller subset of candidates (Precision focus).
       - Typically a Cross-Encoder or Large Bi-Encoder.
    """

    def __init__(self, model_pair: TeacherStudentPair):
        self.pair = model_pair

    def retrieve_and_rerank(
        self,
        query: str,
        retriever_func: Callable[[str, int], List[Dict[str, Any]]],
        initial_k: int = 50,
        final_k: int = 5,
        adaptive: bool = False,
        confidence_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Execute the cascade.

        Args:
            query: The user query.
            retriever_func: Function(query, k) -> List[Dict]. The "Student" retrieval step.
            initial_k: Number of documents the Student should retrieve.
            final_k: Number of documents the Teacher should return after re-ranking.
            adaptive: If True, skip re-ranking if Student is confident enough.
            confidence_threshold: Threshold for adaptive skipping.

        Returns:
            Dict containing results and metadata.
        """
        start_time = time.time()

        # Step 1: Student Retrieval (Fast)
        try:
            candidates = retriever_func(query, initial_k)
        except Exception as e:
            logger.error(f"Student retrieval failed: {e}")
            return {"results": [], "error": str(e)}

        if not candidates:
            return {"results": [], "meta": {"msg": "No candidates found"}}

        # Adaptive Check: If Student is confident, return immediately
        # We assume candidates have a 'score' field from the vector store
        if adaptive and len(candidates) >= 2:
            top_score = candidates[0].get("score", 0.0)
            second_score = candidates[1].get("score", 0.0)

            # Simple confidence metric: Margin between top 1 and 2
            # Or absolute score threshold (depends on metric)
            # Here we use a placeholder logic. In production, calibrate this.
            if top_score > confidence_threshold:
                logger.info(
                    f"Adaptive skip: Student score {top_score} > {confidence_threshold}"
                )
                return {
                    "results": candidates[:final_k],
                    "meta": {
                        "initial_k": initial_k,
                        "final_k": final_k,
                        "execution_time": time.time() - start_time,
                        "strategy": "student-only (adaptive)",
                    },
                }

        # Step 2: Teacher Re-ranking (Accurate)
        scored_candidates = []

        # Prepare batch for efficiency
        pairs_to_score = []
        valid_candidates = []

        for doc in candidates:
            text = doc.get("text") or doc.get("content") or ""
            if not text:
                continue
            pairs_to_score.append((query, text))
            valid_candidates.append(doc)

        if pairs_to_score:
            try:
                # Teacher scores the batch of pairs
                # CrossEncoderWrapper.predict returns a list/array of scores
                scores = self.pair.teacher.predict(pairs_to_score)

                # Ensure scores is iterable (handle single float return)
                if isinstance(scores, (float, int)):
                    scores = [scores]

                for doc, score in zip(valid_candidates, scores):
                    new_doc = doc.copy()
                    new_doc["teacher_score"] = float(score)
                    new_doc["student_score"] = doc.get("score", 0.0)
                    new_doc["score"] = float(
                        score
                    )  # Update main score for downstream compatibility
                    scored_candidates.append(new_doc)

            except Exception as e:
                logger.warning(
                    f"Batch scoring failed: {e}. Falling back to sequential scoring."
                )
                # Fallback to sequential scoring
                for doc in valid_candidates:
                    try:
                        text = doc.get("text") or doc.get("content") or ""
                        score = self.pair.teacher.get_score(query, text)
                        new_doc = doc.copy()
                        new_doc["teacher_score"] = score
                        new_doc["student_score"] = doc.get("score", 0.0)
                        new_doc["score"] = score
                        scored_candidates.append(new_doc)
                    except Exception as inner_e:
                        logger.warning(f"Sequential scoring failed for doc: {inner_e}")
                        continue

        # Sort by teacher score
        scored_candidates.sort(key=lambda x: x["teacher_score"], reverse=True)

        # Return top K
        results = scored_candidates[:final_k]

        execution_time = time.time() - start_time

        return {
            "results": results,
            "meta": {
                "initial_k": initial_k,
                "final_k": final_k,
                "execution_time": execution_time,
                "strategy": "student-retrieve-teacher-rerank",
            },
        }
