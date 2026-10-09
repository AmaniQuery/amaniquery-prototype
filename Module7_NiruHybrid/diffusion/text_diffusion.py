"""
Text-to-Text Diffusion Model for Synthetic Document Generation

Implements denoising diffusion for generating synthetic text documents
conditioned on query context for data augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math
import numpy as np
from dataclasses import dataclass

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available, using basic implementation")

from ..config import DiffusionConfig, default_config


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for timesteps

        Args:
            timesteps: [batch_size]

        Returns:
            embeddings: [batch_size, dim]
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class DiffusionNoiseScheduler:
    """Noise scheduler for diffusion process"""

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule

        # Compute betas
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule == "cosine":
            # Cosine schedule
            s = 0.008
            steps = torch.arange(num_steps + 1, dtype=torch.float32)
            alphas_cumprod = (
                torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(
        self, x: torch.Tensor, timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to input at given timesteps

        Args:
            x: [batch_size, seq_len, embed_dim] or [batch_size, seq_len]
            timesteps: [batch_size]

        Returns:
            noisy_x: Noisy input
            noise: Added noise
        """
        # Sample noise
        noise = torch.randn_like(x)

        # Get sqrt(alpha_cumprod) for each timestep
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(
                -1
            )

        # Add noise
        noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

        return noisy_x, noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps"""
        return torch.randint(0, self.num_steps, (batch_size,), device=device)


class TextDiffusionDenoiser(nn.Module):
    """Denoising network for text diffusion"""

    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 768,
        hidden_dim: int = 2048,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_length: int = 512,
        use_pretrained: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length

        # Time embedding
        self.time_embed_dim = embed_dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(embed_dim),
            nn.Linear(embed_dim, self.time_embed_dim),
            nn.GELU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # Use pretrained GPT-2 as backbone if available
        if use_pretrained and TRANSFORMERS_AVAILABLE:
            self.backbone = GPT2LMHeadModel.from_pretrained("gpt2")
            # Modify to accept time embeddings
            self.time_projection = nn.Linear(self.time_embed_dim, embed_dim)
        else:
            # Custom transformer
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, embed_dim))

            # Transformer blocks
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )

            # Output projection
            self.output_proj = nn.Linear(embed_dim, vocab_size)
            self.time_projection = nn.Linear(self.time_embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict noise at given timestep

        Args:
            x: [batch_size, seq_len] - Noisy token IDs or embeddings
            timesteps: [batch_size] - Diffusion timesteps
            condition: Optional [batch_size, cond_len] - Conditioning (e.g., query)

        Returns:
            predicted_noise: [batch_size, seq_len, vocab_size] or [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = x.shape[:2]

        # Time embedding
        time_emb = self.time_embed(timesteps)  # [batch_size, time_embed_dim]
        time_emb = self.time_projection(time_emb)  # [batch_size, embed_dim]

        if hasattr(self, "backbone"):
            # Use GPT-2 backbone
            # Convert x to embeddings if needed
            if x.dtype == torch.long:
                x_emb = self.backbone.transformer.wte(x)
            else:
                x_emb = x

            # Add time embedding (broadcast and add)
            time_emb_expanded = time_emb.unsqueeze(1).expand(-1, seq_len, -1)
            x_emb = x_emb + time_emb_expanded

            # Add condition if provided
            if condition is not None:
                # Simple concatenation or cross-attention could be added here
                pass

            # Forward through GPT-2
            outputs = self.backbone.transformer(inputs_embeds=x_emb)
            hidden_states = outputs.last_hidden_state

            # Predict noise (project to vocab or embedding space)
            if hasattr(self.backbone, "lm_head"):
                # For token prediction
                logits = self.backbone.lm_head(hidden_states)
                return logits
            else:
                # For embedding prediction
                return hidden_states
        else:
            # Custom transformer
            if x.dtype == torch.long:
                x_emb = self.token_embedding(x)
            else:
                x_emb = x

            # Add positional encoding
            if seq_len <= self.max_seq_length:
                x_emb = x_emb + self.pos_embedding[:, :seq_len, :]

            # Add time embedding
            time_emb_expanded = time_emb.unsqueeze(1).expand(-1, seq_len, -1)
            x_emb = x_emb + time_emb_expanded

            # Forward through transformer
            hidden_states = self.transformer(x_emb)

            # Output projection
            output = self.output_proj(hidden_states)
            return output


class TextDiffusionModel:
    """Text-to-text diffusion model for synthetic document generation"""

    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 768,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
        max_seq_length: int = 512,
        device: str = "cpu",
        config: Optional[DiffusionConfig] = None,
    ):
        if config is not None:
            num_steps = config.text_diffusion_steps
            beta_start = config.text_diffusion_beta_start
            beta_end = config.text_diffusion_beta_end
            schedule = config.text_diffusion_schedule
            vocab_size = config.text_vocab_size
            max_seq_length = config.text_max_length

        self.device = torch.device(device)
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        # Noise scheduler
        self.scheduler = DiffusionNoiseScheduler(
            num_steps=num_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule=schedule,
        )
        self.scheduler.betas = self.scheduler.betas.to(self.device)
        self.scheduler.alphas = self.scheduler.alphas.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.scheduler.sqrt_alphas_cumprod = self.scheduler.sqrt_alphas_cumprod.to(
            self.device
        )
        self.scheduler.sqrt_one_minus_alphas_cumprod = (
            self.scheduler.sqrt_one_minus_alphas_cumprod.to(self.device)
        )

        # Denoiser
        self.denoiser = TextDiffusionDenoiser(
            vocab_size=vocab_size, embed_dim=embed_dim, max_seq_length=max_seq_length
        ).to(self.device)

        # Tokenizer (if available)
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            except:
                self.tokenizer = None
        else:
            self.tokenizer = None

    def train_step(
        self, text_ids: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training step

        Args:
            text_ids: [batch_size, seq_len] - Token IDs
            condition: Optional [batch_size, cond_len] - Conditioning

        Returns:
            loss_dict: Dictionary with loss and metrics
        """
        self.denoiser.train()
        batch_size = text_ids.shape[0]

        # Sample timesteps
        timesteps = self.scheduler.sample_timesteps(batch_size, self.device)

        # Add noise
        noisy_text, noise = self.scheduler.add_noise(text_ids.float(), timesteps)
        noisy_text = noisy_text.long()  # Convert back to long for token IDs

        # Predict noise
        predicted_noise = self.denoiser(noisy_text, timesteps, condition)

        # Compute loss (MSE for embeddings or cross-entropy for tokens)
        if predicted_noise.shape[-1] == self.vocab_size:
            # Token prediction - use cross-entropy
            loss = F.cross_entropy(
                predicted_noise.view(-1, self.vocab_size),
                text_ids.view(-1),
                ignore_index=-100,
            )
        else:
            # Embedding prediction - use MSE
            # Convert text_ids to embeddings for comparison
            text_emb = (
                self.denoiser.token_embedding(text_ids)
                if hasattr(self.denoiser, "token_embedding")
                else None
            )
            if text_emb is not None:
                loss = F.mse_loss(predicted_noise, text_emb)
            else:
                # Fallback: use token prediction loss
                loss = F.cross_entropy(
                    predicted_noise.view(-1, self.vocab_size),
                    text_ids.view(-1),
                    ignore_index=-100,
                )

        return {"loss": loss}

    def generate(
        self,
        condition: Optional[str] = None,
        condition_ids: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        temperature: float = 1.0,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Generate synthetic text

        Args:
            condition: Conditioning text (query context)
            condition_ids: Pre-tokenized condition
            num_steps: Number of diffusion steps
            guidance_scale: Guidance scale for conditioning
            temperature: Sampling temperature
            max_length: Maximum generation length

        Returns:
            generated_text: Generated text string
        """
        self.denoiser.eval()
        max_length = max_length or self.max_seq_length

        with torch.no_grad():
            # Start from random noise
            batch_size = 1
            x = torch.randint(
                0, self.vocab_size, (batch_size, max_length), device=self.device
            )

            # Prepare condition
            if condition is not None and self.tokenizer is not None:
                cond_ids = self.tokenizer.encode(condition, return_tensors="pt").to(
                    self.device
                )
            elif condition_ids is not None:
                cond_ids = condition_ids.to(self.device)
            else:
                cond_ids = None

            # Denoising loop
            timesteps = torch.linspace(
                self.scheduler.num_steps - 1, 0, num_steps, device=self.device
            ).long()

            for i, t in enumerate(timesteps):
                # Predict noise
                t_batch = t.unsqueeze(0).repeat(batch_size)
                predicted_noise = self.denoiser(x, t_batch, cond_ids)

                # Convert to logits if needed
                if predicted_noise.shape[-1] == self.vocab_size:
                    logits = predicted_noise
                else:
                    # Project to vocab
                    if hasattr(self.denoiser, "output_proj"):
                        logits = self.denoiser.output_proj(predicted_noise)
                    else:
                        # Use embedding similarity
                        if hasattr(self.denoiser, "token_embedding"):
                            emb = self.denoiser.token_embedding.weight
                            logits = torch.matmul(predicted_noise, emb.t())
                        else:
                            logits = predicted_noise

                # Sample next tokens
                if i < len(timesteps) - 1:
                    # Use DDPM sampling
                    alpha_t = self.scheduler.alphas_cumprod[t]
                    alpha_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i + 1 < len(timesteps)
                        else torch.tensor(1.0)
                    )

                    # Predict x_0
                    pred_x0 = (
                        x.float() - torch.sqrt(1 - alpha_t) * predicted_noise
                    ) / torch.sqrt(alpha_t)

                    # Sample x_{t-1}
                    posterior_variance = self.scheduler.posterior_variance[t]
                    noise = torch.randn_like(x.float())
                    x = (
                        torch.sqrt(alpha_prev) * pred_x0
                        + torch.sqrt(posterior_variance) * noise
                    )
                    x = x.long().clamp(0, self.vocab_size - 1)
                else:
                    # Final step: sample from logits
                    if temperature > 0:
                        probs = F.softmax(logits / temperature, dim=-1)
                        x = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(
                            batch_size, -1
                        )
                    else:
                        x = torch.argmax(logits, dim=-1)

            # Decode to text
            if self.tokenizer is not None:
                generated_text = self.tokenizer.decode(x[0], skip_special_tokens=True)
            else:
                generated_text = f"Generated tokens: {x[0].tolist()}"

            return generated_text
