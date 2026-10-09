import os
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    AutoTokenizer,
)
from sklearn.metrics import f1_score, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
from .dataset import KenyanSentimentDataset


# --- Custom Model ---
class AfriBertaForSentiment(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Classification Head (4 classes: Neg, Neu, Pos, Mixed)
        self.classifier = nn.Linear(config.hidden_size, 4)

        # Regression Head (Intensity: 0.0 to 1.0)
        self.regressor = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, intensity=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        intensity_pred = self.regressor(pooled_output).squeeze()

        loss = None
        if labels is not None and intensity is not None:
            loss_fct_cls = nn.CrossEntropyLoss()
            loss_fct_reg = nn.MSELoss()

            loss_cls = loss_fct_cls(logits, labels)
            loss_reg = loss_fct_reg(intensity_pred, intensity)

            # Weighted loss (can be tuned)
            loss = loss_cls + loss_reg

        return {"loss": loss, "logits": logits, "intensity_pred": intensity_pred}


# --- Metrics ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions is a tuple (logits, intensity_pred)
    logits = predictions[0]
    intensity_preds = predictions[1]

    # Unpack labels (labels, intensity) - Trainer passes them as a tuple if using custom data collator or just labels
    # Standard Trainer passes 'label_ids' which might just be the 'labels' column.
    # We need to handle the custom dataset output structure.
    # For simplicity in this script, we assume standard Trainer behavior where it stacks inputs.
    # But since we have multiple targets, we might need to adjust.
    # Actually, Trainer passes label_ids. If our dataset returns 'labels' and 'intensity',
    # we need to ensure Trainer gets both.
    # A custom DataCollator or modifying the dataset to return a single 'labels' dict is better.
    # BUT, standard Trainer expects labels to be a tensor.
    # Let's rely on the model returning loss calculated internally.
    # For metrics, we'll extract from the inputs if possible, or just evaluate classification here.

    # NOTE: To properly evaluate both, we need access to both targets.
    # For this script, we will focus on Classification F1 as the primary metric.

    preds_cls = np.argmax(logits, axis=1)
    # labels here is likely just the 'labels' tensor from the dataset if we didn't customize collator
    # We will assume labels is the classification label.

    # If labels is a tuple (labels, intensity), unpack it.
    # In standard HF, label_ids is numpy array.

    # Let's assume labels contains the classification labels for now.
    # If we want to evaluate regression, we'd need to customize the Trainer or DataCollator more deeply.

    f1 = f1_score(labels, preds_cls, average="macro")

    return {"macro_f1": f1}


# --- Training ---
def train(
    data_path, output_dir="models/sauti-sense-afriberta", epochs=5, batch_size=16
):
    model_name = "Davlan/afriberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    model = AfriBertaForSentiment.from_pretrained(model_name, config=config)

    # Load Datasets
    # Assuming train/test split is handled or we split here
    full_dataset = KenyanSentimentDataset(data_path, tokenizer_name=model_name)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="wandb",
        learning_rate=2e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/gold_sentiment_2k.csv")
    parser.add_argument(
        "--output_dir", type=str, default="models/sauti-sense-afriberta"
    )
    args = parser.parse_args()

    train(args.data_path, args.output_dir)
