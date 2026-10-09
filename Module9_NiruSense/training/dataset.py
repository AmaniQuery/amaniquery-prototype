import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import re


class KenyanSentimentDataset(Dataset):
    def __init__(self, data_path, tokenizer_name="Davlan/afriberta-large", max_len=128):
        self.data = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        # Map labels to integers if they are strings
        # Assuming columns: 'text', 'label' (0-3), 'intensity' (0.0-1.0)
        # Label mapping: 0: Negative, 1: Neutral, 2: Positive, 3: Mixed
        self.label_map = {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2,
            "Mixed": 3,
            "negative": 0,
            "neutral": 1,
            "positive": 2,
            "mixed": 3,
        }

    def __len__(self):
        return len(self.data)

    def preprocess(self, text):
        # Remove handles
        text = re.sub(r"@\w+", "", text)
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Normalize spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def __getitem__(self, index):
        text = str(self.data.loc[index, "text"])
        text = self.preprocess(text)

        label = self.data.loc[index, "label"]
        if isinstance(label, str):
            label = self.label_map.get(label, 1)  # Default to Neutral

        intensity = float(self.data.loc[index, "intensity"])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
            "intensity": torch.tensor(intensity, dtype=torch.float),
        }
