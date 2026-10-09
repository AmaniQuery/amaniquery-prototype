"""
Hybrid Convolutional-Transformer Encoder

Combines 1D convolutions for local pattern extraction with transformer blocks
for global context, optimized for streaming with quantized attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Any

from Module7_NiruHybrid.quantization.quantized_attention import (
    QuantizedMultiHeadAttention,
    QuantizedFeedForward,
)
from Module7_NiruHybrid.quantization.attention_streaming import StreamingAttention
from Module7_NiruHybrid.config import HybridEncoderConfig, default_config


class ConvolutionalBlock(nn.Module):
    """1D Convolutional block for local pattern extraction"""

    def __init__(
        self,
        embed_dim: int,
        kernel_sizes: List[int] = [3, 5, 7],
        num_filters: int = 256,
        activation: str = "gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters

        # Multi-scale convolutions
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2,  # Same padding
                )
                for k in kernel_sizes
            ]
        )

        # Batch normalization for each conv
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(num_filters) for _ in kernel_sizes]
        )

        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        # Projection to combine multi-scale features
        self.projection = nn.Linear(num_filters * len(kernel_sizes), embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [batch_size, seq_len, embed_dim]

        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # Convert to [batch_size, embed_dim, seq_len] for conv1d
        x_conv = x.transpose(1, 2)  # [batch_size, embed_dim, seq_len]

        # Apply multi-scale convolutions
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            out = conv(x_conv)  # [batch_size, num_filters, seq_len]
            out = bn(out)
            out = self.activation(out)
            # Transpose back: [batch_size, seq_len, num_filters]
            out = out.transpose(1, 2)
            conv_outputs.append(out)

        # Concatenate multi-scale features
        combined = torch.cat(
            conv_outputs, dim=-1
        )  # [batch_size, seq_len, num_filters * len(kernel_sizes)]

        # Project back to embed_dim
        output = self.projection(combined)
        output = self.dropout(output)

        # Residual connection
        output = output + x

        return output


class TransformerBlock(nn.Module):
    """Transformer block with quantized attention"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        quantize: bool = True,
        use_streaming: bool = False,
        chunk_size: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Attention
        if use_streaming:
            from quantization.attention_streaming import StreamingAttention

            self.attention = StreamingAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                chunk_size=chunk_size,
                dropout=dropout,
                quantize=quantize,
            )
        else:
            self.attention = QuantizedMultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                quantize_weights=quantize,
            )

        # Feed-forward
        self.feed_forward = QuantizedFeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            quantize_weights=quantize,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [batch_size, seq_len, embed_dim]
            key_padding_mask: Optional [batch_size, seq_len]

        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        if isinstance(self.attention, StreamingAttention):
            attn_out = self.attention(
                x, x, x, use_streaming=True, key_padding_mask=key_padding_mask
            )
        else:
            attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(attn_out)

        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        ff_out = self.feed_forward(x)
        x = residual + self.dropout(ff_out)

        return x


class HybridFusion(nn.Module):
    """Fusion mechanism for combining conv and transformer features"""

    def __init__(
        self,
        embed_dim: int,
        fusion_method: str = "concat_projection",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion_method = fusion_method
        self.embed_dim = embed_dim

        if fusion_method == "concat_projection":
            self.projection = nn.Linear(embed_dim * 2, embed_dim)
        elif fusion_method == "weighted_sum":
            self.weight_conv = nn.Parameter(torch.tensor(0.5))
            self.weight_transformer = nn.Parameter(torch.tensor(0.5))
        elif fusion_method == "attention":
            # Attention-based fusion
            self.query = nn.Linear(embed_dim, embed_dim)
            self.key = nn.Linear(embed_dim, embed_dim)
            self.value = nn.Linear(embed_dim, embed_dim)
            self.projection = nn.Linear(embed_dim, embed_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, conv_features: torch.Tensor, transformer_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse convolutional and transformer features

        Args:
            conv_features: [batch_size, seq_len, embed_dim]
            transformer_features: [batch_size, seq_len, embed_dim]

        Returns:
            fused: [batch_size, seq_len, embed_dim]
        """
        if self.fusion_method == "concat_projection":
            # Concatenate and project
            combined = torch.cat([conv_features, transformer_features], dim=-1)
            fused = self.projection(combined)

        elif self.fusion_method == "weighted_sum":
            # Weighted combination
            weight_conv_norm = torch.sigmoid(self.weight_conv)
            weight_trans_norm = torch.sigmoid(self.weight_transformer)
            total = weight_conv_norm + weight_trans_norm
            weight_conv_norm = weight_conv_norm / total
            weight_trans_norm = weight_trans_norm / total

            fused = (
                weight_conv_norm * conv_features
                + weight_trans_norm * transformer_features
            )

        elif self.fusion_method == "attention":
            # Attention-based fusion
            # Use transformer features as query, conv as key/value
            Q = self.query(transformer_features)
            K = self.key(conv_features)
            V = self.value(conv_features)

            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim**0.5)
            attn_weights = F.softmax(scores, dim=-1)

            # Apply attention
            attn_output = torch.matmul(attn_weights, V)
            fused = self.projection(attn_output)

        fused = self.norm(fused)
        fused = self.dropout(fused)

        return fused


class HybridEncoder(nn.Module):
    """Hybrid Convolutional-Transformer Encoder"""

    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 384,
        hidden_dim: int = 768,
        output_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 12,
        conv_kernel_sizes: List[int] = [3, 5, 7],
        conv_num_filters: int = 256,
        fusion_method: str = "concat_projection",
        max_seq_length: int = 512,
        dropout: float = 0.1,
        use_streaming: bool = True,
        chunk_size: int = 128,
        quantize: bool = True,
        config: Optional[HybridEncoderConfig] = None,
    ):
        super().__init__()

        if config is not None:
            embed_dim = config.embedding_dim
            hidden_dim = config.hidden_dim
            output_dim = config.output_dim
            num_layers = config.num_layers
            num_heads = config.num_heads
            conv_kernel_sizes = config.conv_kernel_sizes
            conv_num_filters = config.conv_num_filters
            fusion_method = config.fusion_method
            max_seq_length = config.max_seq_length
            dropout = config.attention_dropout
            use_streaming = config.use_sliding_window
            chunk_size = config.chunk_size

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length
        self.use_streaming = use_streaming

        # Token embedding (if needed, otherwise expects pre-embedded input)
        self.token_embedding = (
            nn.Embedding(vocab_size, embed_dim) if vocab_size > 0 else None
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_length, embed_dim))

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList(
            [
                ConvolutionalBlock(
                    embed_dim=embed_dim,
                    kernel_sizes=conv_kernel_sizes,
                    num_filters=conv_num_filters,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=hidden_dim,
                    dropout=dropout,
                    quantize=quantize,
                    use_streaming=use_streaming,
                    chunk_size=chunk_size,
                )
                for _ in range(num_layers)
            ]
        )

        # Fusion layers
        self.fusion_layers = nn.ModuleList(
            [
                HybridFusion(
                    embed_dim=embed_dim, fusion_method=fusion_method, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_projection = nn.Linear(embed_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_streaming: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [batch_size, seq_len] - Token IDs (if token_embedding is used)
            embeddings: [batch_size, seq_len, embed_dim] - Pre-computed embeddings
            attention_mask: [batch_size, seq_len] - Attention mask
            use_streaming: Override streaming setting

        Returns:
            output: [batch_size, seq_len, output_dim]
        """
        if use_streaming is None:
            use_streaming = self.use_streaming

        # Get embeddings
        if embeddings is not None:
            x = embeddings
        elif input_ids is not None and self.token_embedding is not None:
            x = self.token_embedding(input_ids)
        else:
            raise ValueError("Either input_ids or embeddings must be provided")

        batch_size, seq_len, _ = x.shape

        # Add positional encoding
        if seq_len <= self.max_seq_length:
            x = x + self.pos_encoding[:, :seq_len, :]
        else:
            # For longer sequences, interpolate positional encoding
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            x = x + pos_enc

        # Process through hybrid layers
        for conv_block, transformer_block, fusion_layer in zip(
            self.conv_blocks, self.transformer_blocks, self.fusion_layers
        ):
            # Convolutional path
            conv_out = conv_block(x)

            # Transformer path
            transformer_out = transformer_block(x, key_padding_mask=attention_mask)

            # Fuse features
            x = fusion_layer(conv_out, transformer_out)

        # Output projection
        output = self.output_projection(x)
        output = self.output_norm(output)

        return output

    def encode(
        self,
        text: Optional[str] = None,
        embeddings: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        tokenizer: Optional[Any] = None,
        return_pooled: bool = True,
    ) -> torch.Tensor:
        """
        Encode text to embeddings

        Args:
            text: Input text (requires tokenizer)
            embeddings: Pre-computed embeddings
            input_ids: Token IDs
            tokenizer: Tokenizer to convert text to input_ids
            return_pooled: Return pooled (mean) embedding or full sequence

        Returns:
            embeddings: [batch_size, output_dim] or [batch_size, seq_len, output_dim]
        """
        if text is not None and tokenizer is not None:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
                padding=True,
            )
            input_ids = encoded["input_ids"].to(next(self.parameters()).device)
            if "attention_mask" in encoded:
                pass

        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids=input_ids, embeddings=embeddings)

            if return_pooled:
                output = output.mean(dim=1)

            if was_training:
                self.train()
            return output
