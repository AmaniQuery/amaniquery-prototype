"""
Quantized Multi-Head Attention with FP16/INT8 Mixed Precision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

try:
    import bitsandbytes as bnb

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available, using standard quantization")


class QuantizedLinear(nn.Module):
    """Quantized linear layer with INT8 weights and FP16 activations"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        use_bitsandbytes: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.use_bitsandbytes = use_bitsandbytes and BITSANDBYTES_AVAILABLE

        if self.use_bitsandbytes and bits == 8:
            # Use bitsandbytes for efficient INT8 quantization
            self.weight = bnb.nn.Linear8bitLt(
                in_features, out_features, has_fp16_weights=False
            )
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            # Standard linear layer with manual quantization
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.register_buffer("scale", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization"""
        if self.use_bitsandbytes:
            return self.weight(x)
        else:
            # Manual quantization
            if self.bits == 8:
                # Quantize weights to INT8
                w_quantized, scale = self._quantize_int8(self.weight)
                # Dequantize for computation (FP16)
                w_dequantized = w_quantized.float() * scale
                return F.linear(x, w_dequantized, self.bias)
            else:
                return F.linear(x, self.weight, self.bias)

    def _quantize_int8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to INT8"""
        scale = tensor.abs().max() / 127.0
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale


class QuantizedMultiHeadAttention(nn.Module):
    """Multi-head attention with quantized weights (INT8) and FP16 activations"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        quantize_weights: bool = True,
        quantize_activations: bool = False,
        attention_bits: int = 8,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations

        # Query, Key, Value projections with quantization
        if quantize_weights:
            self.q_proj = QuantizedLinear(embed_dim, embed_dim, bits=attention_bits)
            self.k_proj = QuantizedLinear(embed_dim, embed_dim, bits=attention_bits)
            self.v_proj = QuantizedLinear(embed_dim, embed_dim, bits=attention_bits)
            self.out_proj = QuantizedLinear(embed_dim, embed_dim, bits=attention_bits)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with quantized attention

        Args:
            query: [batch_size, seq_len, embed_dim]
            key: [batch_size, seq_len, embed_dim]
            value: [batch_size, seq_len, embed_dim]
            key_padding_mask: [batch_size, seq_len]
            attn_mask: [seq_len, seq_len] or [batch_size, num_heads, seq_len, seq_len]
            need_weights: Whether to return attention weights

        Returns:
            output: [batch_size, seq_len, embed_dim]
            attn_weights: Optional attention weights
        """
        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V (with quantization if enabled)
        # Convert to FP16 for activations (only on CUDA)
        if query.device.type != "cpu":
            if query.dtype != torch.float16:
                query = query.half()
            if key.dtype != torch.float16:
                key = key.half()
            if value.dtype != torch.float16:
                value = value.half()

        Q = self.q_proj(query)  # [batch_size, seq_len, embed_dim]
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [batch_size, num_heads, seq_len, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [batch_size, num_heads, seq_len, seq_len]

        # Apply masks
        if key_padding_mask is not None:
            # Expand mask: [batch_size, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf")
            )

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(attn_mask.bool(), float("-inf"))

        # Softmax (in FP16)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # [batch_size, num_heads, seq_len, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        if need_weights:
            # Average attention weights across heads
            attn_weights = attn_weights.mean(dim=1)
            return output, attn_weights

        return output, None

    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activation to FP16 (if enabled)"""
        if self.quantize_activations:
            return x.half()
        return x


class QuantizedFeedForward(nn.Module):
    """Feed-forward network with quantized weights"""

    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        quantize_weights: bool = True,
        weight_bits: int = 8,
    ):
        super().__init__()
        self.quantize_weights = quantize_weights

        if quantize_weights:
            self.fc1 = QuantizedLinear(embed_dim, ff_dim, bits=weight_bits)
            self.fc2 = QuantizedLinear(ff_dim, embed_dim, bits=weight_bits)
        else:
            self.fc1 = nn.Linear(embed_dim, ff_dim)
            self.fc2 = nn.Linear(ff_dim, embed_dim)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Convert to FP16 for activations (only on CUDA)
        if x.device.type != "cpu" and x.dtype != torch.float16:
            x = x.half()

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
