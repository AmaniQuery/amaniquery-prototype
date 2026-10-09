"""
Continual Learning System

Fine-tunes hybrid encoder on generated data with gradient accumulation
and checkpoint management for streaming updates.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import json
from loguru import logger

from ..config import RetentionConfig, default_config
from ..hybrid_encoder import HybridEncoder
from ..diffusion.text_diffusion import TextDiffusionModel
from ..diffusion.embedding_diffusion import EmbeddingDiffusionModel


class ContinualLearningDataset(Dataset):
    """Dataset for continual learning from generated data"""

    def __init__(
        self,
        texts: Optional[List[str]] = None,
        embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        self.texts = texts
        self.embeddings = embeddings
        self.labels = labels

        if texts is None and embeddings is None:
            raise ValueError("Either texts or embeddings must be provided")

    def __len__(self):
        if self.texts is not None:
            return len(self.texts)
        elif self.embeddings is not None:
            return self.embeddings.shape[0]
        else:
            return 0

    def __getitem__(self, idx):
        item = {}

        if self.texts is not None:
            item["text"] = self.texts[idx]

        if self.embeddings is not None:
            item["embeddings"] = self.embeddings[idx]

        if self.labels is not None:
            item["labels"] = self.labels[idx]

        return item


class ContinualLearner:
    """Continual learning system for model updates"""

    def __init__(
        self,
        hybrid_encoder: HybridEncoder,
        text_diffusion: Optional[TextDiffusionModel] = None,
        embedding_diffusion: Optional[EmbeddingDiffusionModel] = None,
        learning_rate: float = 1e-5,
        gradient_accumulation_steps: int = 8,
        max_grad_norm: float = 1.0,
        update_frequency: int = 100,
        checkpoint_dir: Optional[Path] = None,
        config: Optional[RetentionConfig] = None,
    ):
        if config is not None:
            learning_rate = config.learning_rate
            gradient_accumulation_steps = config.gradient_accumulation_steps
            max_grad_norm = config.max_grad_norm
            update_frequency = config.update_frequency

        self.hybrid_encoder = hybrid_encoder
        self.text_diffusion = text_diffusion
        self.embedding_diffusion = embedding_diffusion

        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.update_frequency = update_frequency

        self.checkpoint_dir = checkpoint_dir or Path("models/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.hybrid_encoder.parameters(), lr=learning_rate
        )

        # Training state
        self.generated_samples_buffer = []
        self.update_counter = 0
        self.total_updates = 0
        self.training_history = []

    def add_generated_sample(
        self,
        text: Optional[str] = None,
        embeddings: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Add generated sample to buffer for training

        Args:
            text: Generated text
            embeddings: Generated embeddings
            metadata: Optional metadata
        """
        sample = {
            "text": text,
            "embeddings": embeddings,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.generated_samples_buffer.append(sample)
        self.update_counter += 1

        # Trigger update if frequency reached
        if self.update_counter >= self.update_frequency:
            self.update_model()
            self.update_counter = 0

    def update_model(self):
        """Update model weights using buffered generated samples"""
        if len(self.generated_samples_buffer) == 0:
            logger.warning("No generated samples in buffer for training")
            return

        logger.info(
            f"Updating model with {len(self.generated_samples_buffer)} generated samples"
        )

        # Prepare dataset
        texts = [
            s["text"] for s in self.generated_samples_buffer if s["text"] is not None
        ]
        embeddings = [
            s["embeddings"]
            for s in self.generated_samples_buffer
            if s["embeddings"] is not None
        ]

        # Create dataset
        if texts:
            dataset = ContinualLearningDataset(texts=texts)
        elif embeddings:
            embeddings_tensor = torch.stack(embeddings) if len(embeddings) > 0 else None
            if embeddings_tensor is not None:
                dataset = ContinualLearningDataset(embeddings=embeddings_tensor)
            else:
                logger.warning("No valid data for training")
                return
        else:
            logger.warning("No valid data for training")
            return

        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=4,  # Small batch for continual learning
            shuffle=True,
            num_workers=0,
        )

        # Training step
        self.hybrid_encoder.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Forward pass
            if "text" in batch:
                # Encode text
                # Note: This is simplified - would need proper tokenization
                # For now, assume embeddings are provided
                if "embeddings" in batch:
                    input_embeddings = batch["embeddings"]
                else:
                    # Would tokenize and encode here
                    continue
            elif "embeddings" in batch:
                input_embeddings = batch["embeddings"]
            else:
                continue

            # Forward through encoder
            output = self.hybrid_encoder(embeddings=input_embeddings)

            # Loss: reconstruction loss (MSE between input and output)
            # This encourages the model to preserve important patterns
            loss = nn.MSELoss()(output, input_embeddings)

            # Normalize loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.hybrid_encoder.parameters(), self.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Final gradient step if needed
        if num_batches % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.hybrid_encoder.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Record training history
        self.training_history.append(
            {
                "update": self.total_updates,
                "loss": avg_loss,
                "num_samples": len(self.generated_samples_buffer),
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(
            f"Model updated: loss={avg_loss:.4f}, "
            f"samples={len(self.generated_samples_buffer)}, "
            f"total_updates={self.total_updates}"
        )

        # Clear buffer
        self.generated_samples_buffer.clear()
        self.total_updates += 1

        # Save checkpoint periodically
        if self.total_updates % 10 == 0:
            self.save_checkpoint()

    def generate_training_data(
        self, query_context: Optional[str] = None, num_samples: int = 10
    ) -> List[Dict]:
        """
        Generate training data using diffusion models

        Args:
            query_context: Optional query context for generation
            num_samples: Number of samples to generate

        Returns:
            samples: List of generated samples
        """
        samples = []

        # Generate text samples
        if self.text_diffusion is not None:
            for _ in range(num_samples // 2):
                try:
                    generated_text = self.text_diffusion.generate(
                        condition=query_context, num_steps=50
                    )
                    samples.append(
                        {
                            "text": generated_text,
                            "embeddings": None,
                            "source": "text_diffusion",
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate text sample: {e}")

        # Generate embedding samples
        if self.embedding_diffusion is not None:
            for _ in range(num_samples // 2):
                try:
                    generated_embeddings = self.embedding_diffusion.generate(
                        seq_len=128, num_steps=50
                    )
                    samples.append(
                        {
                            "text": None,
                            "embeddings": generated_embeddings,
                            "source": "embedding_diffusion",
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate embedding sample: {e}")

        return samples

    def train_on_generated_data(
        self, query_context: Optional[str] = None, num_samples: int = 10
    ):
        """
        Generate data and train on it

        Args:
            query_context: Optional query context
            num_samples: Number of samples to generate
        """
        # Generate training data
        samples = self.generate_training_data(query_context, num_samples)

        # Add to buffer
        for sample in samples:
            self.add_generated_sample(
                text=sample.get("text"),
                embeddings=sample.get("embeddings"),
                metadata={"source": sample.get("source")},
            )

    def save_checkpoint(self, suffix: Optional[str] = None):
        """Save model checkpoint"""
        if suffix is None:
            suffix = f"update_{self.total_updates}"

        checkpoint_path = self.checkpoint_dir / f"hybrid_encoder_{suffix}.pt"

        checkpoint = {
            "model_state_dict": self.hybrid_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
            "training_history": self.training_history,
            "config": {
                "learning_rate": self.learning_rate,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_grad_norm": self.max_grad_norm,
                "update_frequency": self.update_frequency,
            },
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=(
                self.hybrid_encoder.device
                if hasattr(self.hybrid_encoder, "device")
                else "cpu"
            ),
        )

        self.hybrid_encoder.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_updates = checkpoint.get("total_updates", 0)
        self.training_history = checkpoint.get("training_history", [])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get continual learning statistics"""
        return {
            "total_updates": self.total_updates,
            "buffer_size": len(self.generated_samples_buffer),
            "update_frequency": self.update_frequency,
            "update_counter": self.update_counter,
            "recent_losses": (
                [h["loss"] for h in self.training_history[-10:]]
                if self.training_history
                else []
            ),
            "avg_recent_loss": (
                np.mean([h["loss"] for h in self.training_history[-10:]])
                if self.training_history
                else 0.0
            ),
        }
