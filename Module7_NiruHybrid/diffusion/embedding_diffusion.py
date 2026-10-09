"""
Text-to-Embedding Diffusion Model

Direct embedding generation via diffusion for faster augmentation
compared to text-to-text + embedding pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
from ..config import DiffusionConfig, default_config
from .text_diffusion import DiffusionNoiseScheduler, SinusoidalPositionalEmbedding


class EmbeddingDiffusionDenoiser(nn.Module):
    """Denoising network for embedding diffusion"""

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 8,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Time embedding
        self.time_embed_dim = embed_dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(embed_dim),
            nn.Linear(embed_dim, self.time_embed_dim),
            nn.GELU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # Condition projection (if provided)
        self.condition_proj = None
        if condition_dim is not None:
            self.condition_proj = nn.Linear(condition_dim, embed_dim)

        # Transformer encoder for denoising
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection (predicts noise in embedding space)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Time embedding projection (created once, reused in forward)
        self.time_emb_proj = nn.Linear(self.time_embed_dim, embed_dim)

    def forward(
        self,
        noisy_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict noise in embedding space

        Args:
            noisy_embeddings: [batch_size, seq_len, embed_dim] - Noisy embeddings
            timesteps: [batch_size] - Diffusion timesteps
            condition: Optional [batch_size, cond_len, embed_dim] - Conditioning embeddings

        Returns:
            predicted_noise: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = noisy_embeddings.shape

        # Time embedding
        time_emb = self.time_embed(timesteps)  # [batch_size, time_embed_dim]
        time_emb_proj = self.time_emb_proj(time_emb)
        time_emb_expanded = time_emb_proj.unsqueeze(1).expand(-1, seq_len, -1)

        # Add time embedding
        x = noisy_embeddings + time_emb_expanded

        # Add condition if provided
        if condition is not None and self.condition_proj is not None:
            # Project condition to embedding space
            cond_proj = self.condition_proj(condition)
            # Simple addition (could use cross-attention for better fusion)
            if cond_proj.dim() == 2:
                cond_proj = cond_proj.unsqueeze(1).expand(-1, seq_len, -1)
            x = x + cond_proj

        # Forward through transformer
        hidden_states = self.transformer(x)

        # Output projection
        output = self.output_norm(hidden_states)
        output = self.output_proj(output)

        return output


class EmbeddingDiffusionModel:
    """Text-to-embedding diffusion model"""

    def __init__(
        self,
        embed_dim: int = 384,
        num_steps: int = 500,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
        device: str = "cpu",
        config: Optional[DiffusionConfig] = None,
    ):
        if config is not None:
            embed_dim = config.embedding_dim
            num_steps = config.embedding_diffusion_steps
            beta_start = config.embedding_diffusion_beta_start
            beta_end = config.embedding_diffusion_beta_end
            schedule = config.embedding_diffusion_schedule

        self.device = torch.device(device)
        self.embed_dim = embed_dim

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
        self.scheduler.alphas_cumprod_prev = self.scheduler.alphas_cumprod_prev.to(
            self.device
        )
        self.scheduler.posterior_variance = self.scheduler.posterior_variance.to(
            self.device
        )

        # Denoiser
        self.denoiser = EmbeddingDiffusionDenoiser(embed_dim=embed_dim).to(self.device)

    def train_step(
        self, embeddings: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training step

        Args:
            embeddings: [batch_size, seq_len, embed_dim] - Target embeddings
            condition: Optional [batch_size, cond_len, embed_dim] - Conditioning

        Returns:
            loss_dict: Dictionary with loss and metrics
        """
        self.denoiser.train()
        batch_size = embeddings.shape[0]

        # Sample timesteps
        timesteps = self.scheduler.sample_timesteps(batch_size, self.device)

        # Add noise
        noise = torch.randn_like(embeddings)
        sqrt_alphas_cumprod_t = self.scheduler.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas_cumprod_t = self.scheduler.sqrt_one_minus_alphas_cumprod[
            timesteps
        ]

        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(embeddings.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(
                -1
            )

        noisy_embeddings = (
            sqrt_alphas_cumprod_t * embeddings + sqrt_one_minus_alphas_cumprod_t * noise
        )

        # Predict noise
        predicted_noise = self.denoiser(noisy_embeddings, timesteps, condition)

        # Compute MSE loss
        loss = F.mse_loss(predicted_noise, noise)

        return {"loss": loss}

    def generate(
        self,
        condition: Optional[torch.Tensor] = None,
        seq_len: int = 128,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Generate synthetic embeddings

        Args:
            condition: Optional [batch_size, cond_len, embed_dim] - Conditioning embeddings
            seq_len: Length of sequence to generate
            num_steps: Number of diffusion steps
            guidance_scale: Guidance scale for conditioning
            batch_size: Batch size

        Returns:
            generated_embeddings: [batch_size, seq_len, embed_dim]
        """
        self.denoiser.eval()

        with torch.no_grad():
            # Start from random noise
            x = torch.randn(batch_size, seq_len, self.embed_dim, device=self.device)

            # Denoising loop
            timesteps = torch.linspace(
                self.scheduler.num_steps - 1, 0, num_steps, device=self.device
            ).long()

            for i, t in enumerate(timesteps):
                # Predict noise
                t_batch = t.unsqueeze(0).repeat(batch_size)
                predicted_noise = self.denoiser(x, t_batch, condition)

                # Apply guidance if condition provided
                if condition is not None and guidance_scale > 1.0:
                    # Unconditional prediction
                    uncond_noise = self.denoiser(x, t_batch, None)
                    # Guided prediction
                    predicted_noise = uncond_noise + guidance_scale * (
                        predicted_noise - uncond_noise
                    )

                # DDPM sampling
                if i < len(timesteps) - 1:
                    alpha_t = self.scheduler.alphas_cumprod[t]
                    alpha_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i + 1 < len(timesteps)
                        else torch.tensor(1.0, device=self.device)
                    )

                    # Predict x_0
                    pred_x0 = (
                        x - torch.sqrt(1.0 - alpha_t) * predicted_noise
                    ) / torch.sqrt(alpha_t)

                    # Sample x_{t-1}
                    posterior_variance = self.scheduler.posterior_variance[t]
                    noise = torch.randn_like(x)
                    x = (
                        torch.sqrt(alpha_prev) * pred_x0
                        + torch.sqrt(posterior_variance) * noise
                    )
                else:
                    # Final step: use predicted x_0
                    alpha_t = self.scheduler.alphas_cumprod[t]
                    x = (x - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(
                        alpha_t
                    )

            return x

    def generate_from_text(
        self,
        text_encoder,
        text: str,
        seq_len: int = 128,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """
        Generate embeddings conditioned on text

        Args:
            text_encoder: Encoder to convert text to embeddings
            text: Input text
            seq_len: Length of sequence to generate
            num_steps: Number of diffusion steps
            guidance_scale: Guidance scale

        Returns:
            generated_embeddings: [1, seq_len, embed_dim]
        """
        # Encode text to embeddings
        if hasattr(text_encoder, "encode"):
            condition_emb = text_encoder.encode(text)
        else:
            # Fallback: use text encoder directly
            condition_emb = text_encoder(text)

        # Ensure correct shape
        if condition_emb.dim() == 2:
            condition_emb = condition_emb.unsqueeze(0)  # [1, cond_len, embed_dim]

        # Generate
        return self.generate(
            condition=condition_emb,
            seq_len=seq_len,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            batch_size=1,
        )
