from .base import BaseAgent
from transformers import pipeline
import torch


class TopicClassifier(BaseAgent):
    def __init__(self):
        super().__init__("topic_classifier")
        # Zero-shot classification for flexible topic detection
        self.model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        self.candidate_labels = [
            "Politics",
            "Economy",
            "Sports",
            "Social Issues",
            "Technology",
            "Entertainment",
            "Security",
        ]
        try:
            self.pipe = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
            )
        except Exception as e:
            print(f"Warning: Could not load TopicClassifier: {e}")
            self.pipe = None

    def process(self, text: str, metadata: dict = None) -> dict:
        if not self.pipe:
            return {"topics": [], "scores": [], "error": "Model not loaded"}

        try:
            result = self.pipe(text, self.candidate_labels, multi_label=True)
            # Filter for relevant topics (e.g., score > 0.4)
            topics = [
                label
                for label, score in zip(result["labels"], result["scores"])
                if score > 0.4
            ]
            scores = [score for score in result["scores"] if score > 0.4]
            return {"topics": topics, "scores": scores}
        except Exception as e:
            print(f"Error in TopicClassifier: {e}")
            return {"topics": [], "error": str(e)}


class EntityExtractor(BaseAgent):
    def __init__(self):
        super().__init__("entity_extractor")
        self.model_name = "Davlan/xlm-roberta-base-ner-hrl"
        try:
            self.pipe = pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1,
            )
        except Exception as e:
            print(f"Warning: Could not load EntityExtractor: {e}")
            self.pipe = None

    def process(self, text: str, metadata: dict = None) -> dict:
        if not self.pipe:
            return {"entities": [], "error": "Model not loaded"}

        try:
            results = self.pipe(text)
            # Convert numpy floats to python floats for JSON serialization
            entities = []
            for res in results:
                entities.append(
                    {
                        "text": res["word"],
                        "label": res["entity_group"],
                        "score": float(res["score"]),
                    }
                )
            return {"entities": entities}
        except Exception as e:
            print(f"Error in EntityExtractor: {e}")
            return {"entities": [], "error": str(e)}


class SentimentAnalyzer(BaseAgent):
    def __init__(self):
        super().__init__("sentiment_analyzer")
        self.model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        try:
            self.pipe = pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1,
            )
        except Exception as e:
            print(f"Warning: Could not load SentimentAnalyzer: {e}")
            self.pipe = None

    def process(self, text: str, metadata: dict = None) -> dict:
        if not self.pipe:
            return {"sentiment": "unknown", "score": 0.0, "error": "Model not loaded"}

        try:
            # Truncate
            result = self.pipe(text[:512])[0]
            # Get highest score
            top_sentiment = max(result, key=lambda x: x["score"])
            return {
                "sentiment": top_sentiment["label"],
                "score": float(top_sentiment["score"]),
            }
        except Exception as e:
            print(f"Error in SentimentAnalyzer: {e}")
            return {"sentiment": "error", "score": 0.0}


class EmotionDetector(BaseAgent):
    def __init__(self):
        super().__init__("emotion_detector")
        self.model_name = "j-hartmann/emotion-english-distilroberta-base"
        try:
            self.pipe = pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1,
            )
        except Exception as e:
            print(f"Warning: Could not load EmotionDetector: {e}")
            self.pipe = None

    def process(self, text: str, metadata: dict = None) -> dict:
        if not self.pipe:
            return {"emotion": "unknown", "score": 0.0, "error": "Model not loaded"}

        try:
            result = self.pipe(text[:512])[0]
            top_emotion = max(result, key=lambda x: x["score"])
            return {
                "emotion": top_emotion["label"],
                "score": float(top_emotion["score"]),
            }
        except Exception as e:
            print(f"Error in EmotionDetector: {e}")
            return {"emotion": "error", "score": 0.0}
