from .base import BaseAgent
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class BiasDetector(BaseAgent):
    def __init__(self):
        super().__init__("bias_detector")
        # Reuse the zero-shot model for bias detection with specific labels
        self.model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        self.labels = ["neutral", "biased", "hate speech", "tribalism"]
        try:
            self.pipe = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
            )
        except Exception as e:
            print(f"Warning: Could not load BiasDetector: {e}")
            self.pipe = None

    def process(self, text: str, metadata: dict = None) -> dict:
        if not self.pipe:
            return {"bias_level": "unknown", "flags": [], "error": "Model not loaded"}

        try:
            result = self.pipe(text, self.labels, multi_label=True)
            # Logic: if 'neutral' is top, low bias. If others are high, flag them.
            scores = {
                label: score for label, score in zip(result["labels"], result["scores"])
            }

            flags = []
            bias_level = "low"

            if scores.get("hate speech", 0) > 0.5:
                flags.append("hate_speech")
                bias_level = "high"
            if scores.get("tribalism", 0) > 0.5:
                flags.append("tribalism")
                bias_level = "high"
            if scores.get("biased", 0) > 0.6 and bias_level == "low":
                bias_level = "medium"

            return {"bias_level": bias_level, "flags": flags, "scores": scores}
        except Exception as e:
            print(f"Error in BiasDetector: {e}")
            return {"bias_level": "error", "flags": []}


class Summarizer(BaseAgent):
    def __init__(self):
        super().__init__("summarizer")
        self.model_name = "google/mt5-small"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        except Exception as e:
            print(f"Warning: Could not load Summarizer: {e}")
            self.model = None

    def process(self, text: str, metadata: dict = None) -> dict:
        if not self.model:
            return {"summary": text[:200] + "...", "error": "Model not loaded"}

        try:
            prefix = "summarize: "
            inputs = self.tokenizer(
                prefix + text, return_tensors="pt", max_length=512, truncation=True
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )

            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"summary": summary}
        except Exception as e:
            print(f"Error in Summarizer: {e}")
            return {"summary": text[:200] + "...", "error": str(e)}


class QualityScorer(BaseAgent):
    def __init__(self):
        super().__init__("quality_scorer")

    def process(self, text: str, metadata: dict = None) -> dict:
        # Heuristic-based quality scoring
        # 1. Length check
        length_score = (
            min(len(text) / 500, 1.0) * 10
        )  # Normalize to 0-10 based on 500 chars

        # 2. Information Density (mocked by word length avg)
        words = text.split()
        if not words:
            return {"quality_score": 0.0, "reasoning": "Empty text"}

        avg_word_len = sum(len(w) for w in words) / len(words)
        density_score = min(avg_word_len / 6, 1.0) * 10

        # 3. Penalize if too short
        if len(words) < 5:
            return {"quality_score": 2.0, "reasoning": "Too short"}

        final_score = (length_score * 0.4) + (density_score * 0.6)

        # Cap at 10
        final_score = min(final_score, 10.0)

        return {
            "quality_score": round(final_score, 2),
            "reasoning": f"Length: {len(words)} words, Avg Word Len: {round(avg_word_len, 1)}",
        }
