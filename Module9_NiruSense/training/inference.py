import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os


class SentimentInference:
    def __init__(self, model_path, tokenizer_path="Davlan/afriberta-large"):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.labels = ["Negative", "Neutral", "Positive", "Mixed"]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="numpy",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        logits, intensity = self.session.run(None, ort_inputs)

        probs = self.softmax(logits[0])
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        label = self.labels[pred_idx]

        # We can map the classification to a score or use the regression head.
        # Let's use the regression head output (0-1) and map it?
        # 0 -> -1, 0.5 -> 0, 1 -> 1
        raw_intensity = float(intensity)
        mapped_score = (raw_intensity * 2) - 1

        # Clamp
        mapped_score = max(-1.0, min(1.0, mapped_score))

        return {
            "sentiment": label,
            "score": mapped_score,
            "confidence": confidence,
            "raw_intensity": raw_intensity,
        }


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text", type=str, default="Kenya ni nchi nzuri sana")
    args = parser.parse_args()

    inference = SentimentInference(args.model_path)
    result = inference.predict(args.text)
    print(result)
