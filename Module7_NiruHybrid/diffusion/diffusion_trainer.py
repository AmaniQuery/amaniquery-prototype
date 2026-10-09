"""
Training utilities for diffusion models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Callable
from pathlib import Path
import json
from loguru import logger

from .text_diffusion import TextDiffusionModel
from .embedding_diffusion import EmbeddingDiffusionModel
from ..config import DiffusionConfig, default_config


class DiffusionDataset(Dataset):
    """Dataset for diffusion model training"""

    def __init__(
        self,
        texts: Optional[List[str]] = None,
        embeddings: Optional[torch.Tensor] = None,
        tokenizer=None,
        max_length: int = 512,
    ):
        self.texts = texts
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.max_length = max_length

        if texts is None and embeddings is None:
            raise ValueError("Either texts or embeddings must be provided")

    def __len__(self):
        if self.texts is not None:
            return len(self.texts)
        else:
            return self.embeddings.shape[0]

    def __getitem__(self, idx):
        if self.embeddings is not None:
            return {"embeddings": self.embeddings[idx]}
        elif self.texts is not None:
            text = self.texts[idx]
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).squeeze(0)
                return {"text_ids": tokens}
            else:
                return {"text": text}
        else:
            raise ValueError("No data available")


class DiffusionTrainer:
    """Trainer for diffusion models"""

    def __init__(
        self,
        model: Optional[TextDiffusionModel] = None,
        embedding_model: Optional[EmbeddingDiffusionModel] = None,
        config: Optional[DiffusionConfig] = None,
        device: str = "cpu",
    ):
        self.config = config or default_config.diffusion
        self.device = torch.device(device)

        self.text_model = model
        self.embedding_model = embedding_model

        if self.text_model is not None:
            self.text_model.device = self.device
        if self.embedding_model is not None:
            self.embedding_model.device = self.device

    def train_text_diffusion(
        self,
        dataset: DiffusionDataset,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        save_dir: Optional[Path] = None,
        log_interval: int = 100,
    ):
        """Train text-to-text diffusion model"""
        if self.text_model is None:
            raise ValueError("Text diffusion model not initialized")

        num_epochs = num_epochs or self.config.num_epochs
        batch_size = batch_size or self.config.batch_size
        learning_rate = learning_rate or self.config.learning_rate
        gradient_accumulation_steps = (
            gradient_accumulation_steps or self.config.gradient_accumulation_steps
        )

        # Data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.text_model.denoiser.parameters(), lr=learning_rate
        )

        # Training loop
        self.text_model.denoiser.train()
        global_step = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                # Get text IDs
                if "text_ids" in batch:
                    text_ids = batch["text_ids"].to(self.device)
                elif "text" in batch:
                    # Tokenize if needed
                    if self.text_model.tokenizer is not None:
                        text_ids = self.text_model.tokenizer(
                            batch["text"],
                            padding=True,
                            truncation=True,
                            max_length=self.text_model.max_seq_length,
                            return_tensors="pt",
                        )["input_ids"].to(self.device)
                    else:
                        continue
                else:
                    continue

                # Training step
                loss_dict = self.text_model.train_step(text_ids)
                loss = loss_dict["loss"] / gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % log_interval == 0:
                        logger.info(
                            f"Epoch {epoch+1}/{num_epochs}, "
                            f"Step {global_step}, "
                            f"Loss: {loss.item() * gradient_accumulation_steps:.4f}"
                        )

                epoch_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}"
            )

            # Save checkpoint
            if save_dir is not None:
                self.save_checkpoint(save_dir, epoch, is_text_model=True)

    def train_embedding_diffusion(
        self,
        dataset: DiffusionDataset,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        save_dir: Optional[Path] = None,
        log_interval: int = 100,
    ):
        """Train text-to-embedding diffusion model"""
        if self.embedding_model is None:
            raise ValueError("Embedding diffusion model not initialized")

        num_epochs = num_epochs or self.config.num_epochs
        batch_size = batch_size or self.config.batch_size
        learning_rate = learning_rate or self.config.learning_rate
        gradient_accumulation_steps = (
            gradient_accumulation_steps or self.config.gradient_accumulation_steps
        )

        # Data loader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.embedding_model.denoiser.parameters(), lr=learning_rate
        )

        # Training loop
        self.embedding_model.denoiser.train()
        global_step = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                if "embeddings" not in batch:
                    continue

                embeddings = batch["embeddings"].to(self.device)
                if embeddings.dim() == 2:
                    embeddings = embeddings.unsqueeze(1)  # Add sequence dimension

                # Training step
                loss_dict = self.embedding_model.train_step(embeddings)
                loss = loss_dict["loss"] / gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % log_interval == 0:
                        logger.info(
                            f"Epoch {epoch+1}/{num_epochs}, "
                            f"Step {global_step}, "
                            f"Loss: {loss.item() * gradient_accumulation_steps:.4f}"
                        )

                epoch_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}"
            )

            # Save checkpoint
            if save_dir is not None:
                self.save_checkpoint(save_dir, epoch, is_text_model=False)

    def save_checkpoint(self, save_dir: Path, epoch: int, is_text_model: bool = True):
        """Save model checkpoint"""
        save_dir.mkdir(parents=True, exist_ok=True)

        if is_text_model and self.text_model is not None:
            checkpoint = {
                "epoch": epoch,
                "denoiser_state_dict": self.text_model.denoiser.state_dict(),
                "scheduler_betas": self.text_model.scheduler.betas.cpu(),
                "scheduler_alphas": self.text_model.scheduler.alphas.cpu(),
                "scheduler_alphas_cumprod": self.text_model.scheduler.alphas_cumprod.cpu(),
            }
            torch.save(checkpoint, save_dir / f"text_diffusion_epoch_{epoch}.pt")
            logger.info(f"Saved text diffusion checkpoint: epoch {epoch}")

        elif not is_text_model and self.embedding_model is not None:
            checkpoint = {
                "epoch": epoch,
                "denoiser_state_dict": self.embedding_model.denoiser.state_dict(),
                "scheduler_betas": self.embedding_model.scheduler.betas.cpu(),
                "scheduler_alphas": self.embedding_model.scheduler.alphas.cpu(),
                "scheduler_alphas_cumprod": self.embedding_model.scheduler.alphas_cumprod.cpu(),
            }
            torch.save(checkpoint, save_dir / f"embedding_diffusion_epoch_{epoch}.pt")
            logger.info(f"Saved embedding diffusion checkpoint: epoch {epoch}")

    def load_checkpoint(self, checkpoint_path: Path, is_text_model: bool = True):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if is_text_model and self.text_model is not None:
            self.text_model.denoiser.load_state_dict(checkpoint["denoiser_state_dict"])
            self.text_model.scheduler.betas = checkpoint["scheduler_betas"].to(
                self.device
            )
            self.text_model.scheduler.alphas = checkpoint["scheduler_alphas"].to(
                self.device
            )
            self.text_model.scheduler.alphas_cumprod = checkpoint[
                "scheduler_alphas_cumprod"
            ].to(self.device)
            logger.info(f"Loaded text diffusion checkpoint from {checkpoint_path}")

        elif not is_text_model and self.embedding_model is not None:
            self.embedding_model.denoiser.load_state_dict(
                checkpoint["denoiser_state_dict"]
            )
            self.embedding_model.scheduler.betas = checkpoint["scheduler_betas"].to(
                self.device
            )
            self.embedding_model.scheduler.alphas = checkpoint["scheduler_alphas"].to(
                self.device
            )
            self.embedding_model.scheduler.alphas_cumprod = checkpoint[
                "scheduler_alphas_cumprod"
            ].to(self.device)
            logger.info(f"Loaded embedding diffusion checkpoint from {checkpoint_path}")
