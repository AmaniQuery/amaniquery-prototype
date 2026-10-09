from .base import BaseAgent

from .base import BaseAgent
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


class LanguageIdentifier(BaseAgent):
    def __init__(self):
        super().__init__("language_identifier")
        # Using a lightweight model for Language ID as fallback or the configured one
        # "papluca/xlm-roberta-base-language-detection" is a good alternative if Davlan is not purely for LID
        self.model_name = "papluca/xlm-roberta-base-language-detection"
        try:
            self.pipe = pipeline(
                "text-classification",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
            )
        except Exception as e:
            print(f"Warning: Could not load Language ID model: {e}")
            self.pipe = None

    def process(self, text: str, metadata: dict = None) -> dict:
        if not self.pipe:
            return {"lang": "unknown", "conf": 0.0, "error": "Model not loaded"}

        try:
            # Truncate text to avoid token limit errors
            result = self.pipe(text[:512], top_k=1)
            # Result format: [{'label': 'sw', 'score': 0.99}]
            lang = result[0]["label"]
            score = result[0]["score"]

            # Custom logic for Sheng detection (heuristic + model)
            # If model says Swahili/English but contains specific Sheng markers
            sheng_markers = ["wasee", "form ni", "bazenga", "mbogi", "rieng"]
            if any(marker in text.lower() for marker in sheng_markers):
                return {"lang": "sheng", "conf": 0.85, "base_lang": lang}

            return {"lang": lang, "conf": score}
        except Exception as e:
            print(f"Error in LanguageID: {e}")
            return {"lang": "error", "conf": 0.0}


class SlangDecoder(BaseAgent):
    def __init__(self):
        super().__init__("slang_decoder")
        # Using flan-t5-base as a robust default for instruction following on CPU
        # In production with GPU, swap this for "meta-llama/Llama-3.2-3B-Instruct"
        self.model_name = "google/flan-t5-base"
        try:
            from transformers import AutoModelForSeq2SeqLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        except Exception as e:
            print(f"Warning: Could not load SlangDecoder model: {e}")
            self.model = None

    def process(self, text: str, metadata: dict = None) -> dict:
        if not self.model:
            return {
                "normalized_text": text,
                "mappings": {},
                "error": "Model not loaded",
            }

        try:
            # Construct prompt for slang normalization
            prompt = (
                f"Translate this Kenyan Sheng/Slang text to standard English: {text}"
            )

            inputs = self.tokenizer(
                prompt, return_tensors="pt", max_length=512, truncation=True
            ).to(self.device)

            # Generate translation
            outputs = self.model.generate(
                **inputs, max_new_tokens=128, temperature=0.3, do_sample=True
            )

            normalized_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            return {
                "normalized_text": normalized_text,
                "mappings": {},  # Mappings would require token-level alignment, skipping for seq2seq
                "original_text": text,
            }
        except Exception as e:
            print(f"Error in SlangDecoder: {e}")
            return {"normalized_text": text, "error": str(e)}
