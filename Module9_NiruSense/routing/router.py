from .config import SCORE_THRESHOLD, ENTROPY_THRESHOLD, COMPILED_SARCASM_PATTERNS
from .llm_client import LLMClient
import math


class UncertaintyRouter:
    def __init__(self):
        self.llm_client = LLMClient()

    def route(self, text: str, ensemble_score: float, entropy: float = 0.0) -> dict:
        """
        Decides whether to use the ensemble score or route to an LLM.

        Args:
            text: The input text.
            ensemble_score: The sentiment score from the local model (-1 to 1).
            entropy: The entropy of the local model's prediction (0 to 1+).

        Returns:
            dict: Final sentiment result.
        """
        triggers = []

        # 1. Check Sarcasm Patterns
        for pattern in COMPILED_SARCASM_PATTERNS:
            if pattern.search(text):
                triggers.append("sarcasm_detected")
                break

        # 2. Check Score Ambiguity
        if abs(ensemble_score) < SCORE_THRESHOLD:
            triggers.append("ambiguous_score")

        # 3. Check Entropy (Uncertainty)
        if entropy > ENTROPY_THRESHOLD:
            triggers.append("high_entropy")

        # Decision
        if triggers:
            print(f"Routing to LLM due to: {triggers}")
            llm_result = self.llm_client.analyze_text(text)
            llm_result["source"] = "LLM"
            llm_result["triggers"] = triggers
            return llm_result
        else:
            # Return local model result formatted similarly
            label = "Neutral"
            if ensemble_score > 0.38:
                label = "Positive"
            if ensemble_score < -0.38:
                label = "Negative"

            return {
                "sentiment_score": ensemble_score,
                "sentiment_label": label,
                "is_sarcastic": False,
                "explanation": "Local model confidence high.",
                "source": "Local",
                "triggers": [],
            }
