"""
Configuration for Hybrid Convolutional-Transformer Pipeline
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HybridEncoderConfig:
    """Configuration for hybrid encoder"""

    # Embedding dimensions
    embedding_dim: int = 384  # Compatible with existing vector store
    hidden_dim: int = 768
    output_dim: int = 384

    # Convolutional layers
    conv_kernel_sizes: list = field(default_factory=lambda: [3, 5, 7])
    conv_num_filters: int = 256
    conv_activation: str = "gelu"

    # Transformer blocks
    num_layers: int = 6
    num_heads: int = 12
    attention_dropout: float = 0.1
    ff_dropout: float = 0.1
    max_seq_length: int = 512

    # Fusion mechanism
    fusion_method: str = (
        "concat_projection"  # concat_projection, weighted_sum, attention
    )
    fusion_dropout: float = 0.1

    # Streaming
    chunk_size: int = 128
    overlap_size: int = 32
    use_sliding_window: bool = True

    # Device
    device: str = (
        "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    )


@dataclass
class QuantizationConfig:
    """Configuration for quantization"""

    # Attention quantization
    attention_weight_bits: int = 8  # INT8
    attention_activation_bits: int = 16  # FP16
    use_bitsandbytes: bool = True

    # Feed-forward quantization
    ff_weight_bits: int = 8
    ff_activation_bits: int = 16

    # Calibration
    calibration_samples: int = 1000
    calibration_batch_size: int = 32

    # Streaming attention
    streaming_chunk_size: int = 128
    streaming_overlap: int = 32


@dataclass
class DiffusionConfig:
    """Configuration for diffusion models"""

    # Text-to-text diffusion
    text_diffusion_steps: int = 1000
    text_diffusion_beta_start: float = 0.0001
    text_diffusion_beta_end: float = 0.02
    text_diffusion_schedule: str = "linear"
    text_vocab_size: int = 50257  # GPT-2 vocab size
    text_max_length: int = 512

    # Embedding diffusion
    embedding_diffusion_steps: int = 500
    embedding_diffusion_beta_start: float = 0.0001
    embedding_diffusion_beta_end: float = 0.02
    embedding_dim: int = 384

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 10
    gradient_accumulation_steps: int = 4

    # Generation
    num_generation_steps: int = 50
    guidance_scale: float = 7.5
    temperature: float = 1.0

    # Quality filtering
    min_quality_score: float = 0.6
    use_quality_filter: bool = True


@dataclass
class RetentionConfig:
    """Configuration for dynamic retention"""

    # Continual learning
    continual_learning_enabled: bool = True
    update_frequency: int = 100  # Update every N generated samples
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0

    # Memory management
    memory_buffer_size: int = 10000
    importance_threshold: float = 0.7
    eviction_policy: str = "lru"  # lru, lfu, importance
    retention_ratio: float = 0.1  # Keep top 10% of patterns

    # Adaptive retrieval
    adaptive_retrieval_enabled: bool = True
    coarse_top_k: int = 50
    fine_top_k: int = 5
    similarity_threshold: float = 0.5
    context_window_size: int = 5  # Number of recent queries for context


@dataclass
class StreamingConfig:
    """Configuration for streaming pipeline"""

    # Buffer settings
    buffer_size: int = 1000
    buffer_timeout: float = 0.1  # seconds

    # Processing
    batch_size: int = 32
    max_concurrent_streams: int = 10
    processing_timeout: float = 30.0  # seconds

    # Chunking
    query_chunk_size: int = 128
    data_chunk_size: int = 256
    overlap_size: int = 32


@dataclass
class DistillationConfig:
    """Configuration for model distillation cascade"""

    # Cascade settings
    enabled: bool = True
    student_top_k: int = 50
    teacher_top_k: int = 5

    # Distillation parameters
    temperature: float = 2.0
    alpha: float = 0.5  # Weight for distillation loss

    # Model paths (if different from main encoder)
    teacher_model_path: Optional[str] = None
    student_model_path: Optional[str] = None

    # Adaptive retrieval
    use_adaptive: bool = True
    confidence_threshold: float = 0.85


@dataclass
class HybridPipelineConfig:
    """Main configuration class"""

    encoder: HybridEncoderConfig = field(default_factory=HybridEncoderConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)

    # Paths
    model_dir: Path = field(default_factory=lambda: Path("models/hybrid"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("models/checkpoints"))
    cache_dir: Path = field(default_factory=lambda: Path("models/cache"))

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    def __post_init__(self):
        """Create directories if they don't exist"""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HybridPipelineConfig":
        """Create config from dictionary"""
        encoder_config = HybridEncoderConfig(**config_dict.get("encoder", {}))
        quantization_config = QuantizationConfig(**config_dict.get("quantization", {}))
        diffusion_config = DiffusionConfig(**config_dict.get("diffusion", {}))
        retention_config = RetentionConfig(**config_dict.get("retention", {}))
        streaming_config = StreamingConfig(**config_dict.get("streaming", {}))
        distillation_config = DistillationConfig(**config_dict.get("distillation", {}))

        return cls(
            encoder=encoder_config,
            quantization=quantization_config,
            diffusion=diffusion_config,
            retention=retention_config,
            streaming=streaming_config,
            distillation=distillation_config,
            **{
                k: v
                for k, v in config_dict.items()
                if k
                not in [
                    "encoder",
                    "quantization",
                    "diffusion",
                    "retention",
                    "streaming",
                    "distillation",
                ]
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "encoder": self.encoder.__dict__,
            "quantization": self.quantization.__dict__,
            "diffusion": self.diffusion.__dict__,
            "retention": self.retention.__dict__,
            "streaming": self.streaming.__dict__,
            "distillation": self.distillation.__dict__,
            "model_dir": str(self.model_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "cache_dir": str(self.cache_dir),
            "log_level": self.log_level,
            "log_file": self.log_file,
        }


# Default global config instance
default_config = HybridPipelineConfig()
