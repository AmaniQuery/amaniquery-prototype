"""
Configuration management for LiveKit Voice Agent
"""

import os
from typing import Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv
from loguru import logger

try:
    from pydantic import BaseModel, Field, field_validator

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.warning("Pydantic not available, using basic validation")

load_dotenv()


if PYDANTIC_AVAILABLE:

    class VoiceAgentConfig(BaseModel):
        """Configuration for the voice agent with Pydantic validation"""

        # STT/TTS providers (can be list for multiple providers)
        stt_providers: List[str] = Field(
            default=["openai"], description="STT providers in priority order"
        )
        tts_providers: List[str] = Field(
            default=["openai"], description="TTS providers in priority order"
        )
        language: str = Field(default="en", description="Language code")

        # Agent behavior
        max_response_length: int = Field(
            default=500, ge=1, description="Maximum words in response"
        )
        enable_follow_ups: bool = Field(
            default=True, description="Enable conversation context"
        )
        conversation_timeout: int = Field(
            default=300, ge=1, description="Session timeout in seconds"
        )

        # RAG pipeline settings
        rag_top_k: int = Field(
            default=5, ge=1, description="Number of documents to retrieve"
        )
        rag_temperature: float = Field(
            default=0.7, ge=0.0, le=2.0, description="LLM temperature"
        )
        rag_max_tokens: int = Field(
            default=1500, ge=1, description="Maximum tokens in response"
        )

        # Resilience settings
        enable_retry: bool = Field(default=True, description="Enable retry logic")
        max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
        enable_circuit_breaker: bool = Field(
            default=True, description="Enable circuit breaker"
        )
        circuit_breaker_threshold: int = Field(
            default=5, ge=1, description="Circuit breaker failure threshold"
        )

        # Provider fallback
        enable_provider_fallback: bool = Field(
            default=True, description="Enable automatic provider failover"
        )
        fallback_mode: str = Field(
            default="health_based",
            description="Fallback mode: sequential, round_robin, health_based, random",
        )

        # Performance settings
        enable_caching: bool = Field(
            default=True, description="Enable response caching"
        )
        cache_ttl: int = Field(default=3600, ge=1, description="Cache TTL in seconds")
        enable_rate_limiting: bool = Field(
            default=True, description="Enable rate limiting"
        )
        rate_limit_per_minute: int = Field(
            default=60, ge=1, description="Requests per minute limit"
        )

        # Session persistence
        enable_redis_sessions: bool = Field(
            default=False, description="Use Redis for session storage"
        )
        redis_url: Optional[str] = Field(
            default=None, description="Redis connection URL"
        )

        # Monitoring
        enable_metrics: bool = Field(
            default=True, description="Enable metrics collection"
        )
        enable_prometheus: bool = Field(
            default=False, description="Enable Prometheus metrics export"
        )
        prometheus_port: int = Field(
            default=9090, ge=1, le=65535, description="Prometheus metrics port"
        )

        @field_validator("stt_providers", "tts_providers")
        @classmethod
        def validate_providers(cls, v):
            """Validate provider lists"""
            if not v:
                raise ValueError("At least one provider must be specified")
            valid_stt = ["openai", "assemblyai", "deepgram", "nvidia_nim"]
            valid_tts = ["openai", "silero", "elevenlabs", "nvidia_nim", "vibevoice"]
            # Check if all providers are valid (relaxed for future providers)
            return v

        @field_validator("language")
        @classmethod
        def validate_language(cls, v):
            """Validate language code"""
            if v not in ["en", "sw"]:
                logger.warning(f"Unknown language: {v}, defaulting to en")
                return "en"
            return v

        @field_validator("fallback_mode")
        @classmethod
        def validate_fallback_mode(cls, v):
            """Validate fallback mode"""
            valid_modes = ["sequential", "round_robin", "health_based", "random"]
            if v not in valid_modes:
                logger.warning(
                    f"Unknown fallback mode: {v}, defaulting to health_based"
                )
                return "health_based"
            return v

        @property
        def stt_provider(self) -> str:
            """Get primary STT provider (backward compatibility)"""
            return self.stt_providers[0] if self.stt_providers else "openai"

        @property
        def tts_provider(self) -> str:
            """Get primary TTS provider (backward compatibility)"""
            return self.tts_providers[0] if self.tts_providers else "openai"

        def get_stt_config(self) -> dict:
            """Get STT provider configuration"""
            config = {
                "provider": self.stt_provider,
                "language": self.language,
            }

            if self.stt_provider == "openai":
                config["model"] = "whisper-1"
            elif self.stt_provider == "assemblyai":
                config["language_code"] = "en" if self.language == "en" else "sw"

            return config

        def get_tts_config(self) -> dict:
            """Get TTS provider configuration"""
            config = {
                "provider": self.tts_provider,
                "language": self.language,
            }

            if self.tts_provider == "openai":
                config["voice"] = "alloy"
                config["model"] = "tts-1"
            elif self.tts_provider == "silero":
                config["speaker"] = "en_0" if self.language == "en" else "sw_0"

            return config

        @classmethod
        def from_env(cls) -> "VoiceAgentConfig":
            """Load configuration from environment variables"""
            return _load_config_from_env(cls)

else:

    @dataclass
    class VoiceAgentConfig:
        """Configuration for the voice agent (fallback without Pydantic)"""

        # STT/TTS providers
        stt_providers: List[str] = None
        tts_providers: List[str] = None
        language: str = "en"

        # Agent behavior
        max_response_length: int = 500
        enable_follow_ups: bool = True
        conversation_timeout: int = 300

        # RAG pipeline settings
        rag_top_k: int = 5
        rag_temperature: float = 0.7
        rag_max_tokens: int = 1500

        # Resilience settings
        enable_retry: bool = True
        max_retries: int = 3
        enable_circuit_breaker: bool = True
        circuit_breaker_threshold: int = 5

        # Provider fallback
        enable_provider_fallback: bool = True
        fallback_mode: str = "health_based"

        # Performance settings
        enable_caching: bool = True
        cache_ttl: int = 3600
        enable_rate_limiting: bool = True
        rate_limit_per_minute: int = 60

        # Session persistence
        enable_redis_sessions: bool = False
        redis_url: Optional[str] = None

        # Monitoring
        enable_metrics: bool = True
        enable_prometheus: bool = False
        prometheus_port: int = 9090

        def __post_init__(self):
            """Initialize defaults"""
            if self.stt_providers is None:
                self.stt_providers = ["openai"]
            if self.tts_providers is None:
                self.tts_providers = ["openai"]

        @property
        def stt_provider(self) -> str:
            """Get primary STT provider"""
            return self.stt_providers[0] if self.stt_providers else "openai"

        @property
        def tts_provider(self) -> str:
            """Get primary TTS provider"""
            return self.tts_providers[0] if self.tts_providers else "openai"

        def get_stt_config(self) -> dict:
            """Get STT provider configuration"""
            config = {
                "provider": self.stt_provider,
                "language": self.language,
            }

            if self.stt_provider == "openai":
                config["model"] = "whisper-1"
            elif self.stt_provider == "assemblyai":
                config["language_code"] = "en" if self.language == "en" else "sw"

            return config

        def get_tts_config(self) -> dict:
            """Get TTS provider configuration"""
            config = {
                "provider": self.tts_provider,
                "language": self.language,
            }

            if self.tts_provider == "openai":
                config["voice"] = "alloy"
                config["model"] = "tts-1"
            elif self.tts_provider == "silero":
                config["speaker"] = "en_0" if self.language == "en" else "sw_0"

            return config

        @classmethod
        def from_env(cls) -> "VoiceAgentConfig":
            """Load configuration from environment variables"""
            return _load_config_from_env(cls)


def _load_config_from_env(cls) -> VoiceAgentConfig:
    """Load configuration from environment variables"""

    # STT/TTS providers (support multiple)
    stt_provider_str = os.getenv("VOICE_STT_PROVIDER", "openai")
    tts_provider_str = os.getenv("VOICE_TTS_PROVIDER", "vibevoice")

    # Parse provider lists (comma-separated) or single provider
    stt_providers = [p.strip().lower() for p in stt_provider_str.split(",")]
    tts_providers = [p.strip().lower() for p in tts_provider_str.split(",")]

    language = os.getenv("VOICE_LANGUAGE", "en").lower()

    # Agent behavior settings
    max_response_length = int(os.getenv("VOICE_MAX_RESPONSE_LENGTH", "500"))
    enable_follow_ups = os.getenv("VOICE_ENABLE_FOLLOW_UPS", "true").lower() == "true"
    conversation_timeout = int(os.getenv("VOICE_CONVERSATION_TIMEOUT", "300"))

    # RAG settings
    rag_top_k = int(os.getenv("VOICE_RAG_TOP_K", "5"))
    rag_temperature = float(os.getenv("VOICE_RAG_TEMPERATURE", "0.7"))
    rag_max_tokens = int(os.getenv("VOICE_RAG_MAX_TOKENS", "1500"))

    # Resilience settings
    enable_retry = os.getenv("VOICE_ENABLE_RETRY", "true").lower() == "true"
    max_retries = int(os.getenv("VOICE_MAX_RETRIES", "3"))
    enable_circuit_breaker = (
        os.getenv("VOICE_ENABLE_CIRCUIT_BREAKER", "true").lower() == "true"
    )
    circuit_breaker_threshold = int(os.getenv("VOICE_CIRCUIT_BREAKER_THRESHOLD", "5"))

    # Provider fallback
    enable_provider_fallback = (
        os.getenv("VOICE_PROVIDER_FALLBACK_ENABLED", "true").lower() == "true"
    )
    fallback_mode = os.getenv("VOICE_FALLBACK_MODE", "health_based").lower()

    # Performance settings
    enable_caching = os.getenv("VOICE_ENABLE_CACHING", "true").lower() == "true"
    cache_ttl = int(os.getenv("VOICE_CACHE_TTL", "3600"))
    enable_rate_limiting = (
        os.getenv("VOICE_ENABLE_RATE_LIMITING", "true").lower() == "true"
    )
    rate_limit_per_minute = int(os.getenv("VOICE_RATE_LIMIT_PER_MINUTE", "60"))

    # Session persistence
    enable_redis_sessions = os.getenv("VOICE_REDIS_SESSIONS", "false").lower() == "true"
    redis_url = os.getenv("VOICE_REDIS_URL") or os.getenv("REDIS_URL")

    # Monitoring
    enable_metrics = os.getenv("VOICE_ENABLE_METRICS", "true").lower() == "true"
    enable_prometheus = os.getenv("VOICE_ENABLE_PROMETHEUS", "false").lower() == "true"
    prometheus_port = int(os.getenv("VOICE_METRICS_PORT", "9090"))

    logger.info(
        f"Voice agent config loaded: STT={stt_providers}, TTS={tts_providers}, "
        f"Lang={language}, Fallback={enable_provider_fallback}"
    )

    config_data = {
        "stt_providers": stt_providers,
        "tts_providers": tts_providers,
        "language": language,
        "max_response_length": max_response_length,
        "enable_follow_ups": enable_follow_ups,
        "conversation_timeout": conversation_timeout,
        "rag_top_k": rag_top_k,
        "rag_temperature": rag_temperature,
        "rag_max_tokens": rag_max_tokens,
        "enable_retry": enable_retry,
        "max_retries": max_retries,
        "enable_circuit_breaker": enable_circuit_breaker,
        "circuit_breaker_threshold": circuit_breaker_threshold,
        "enable_provider_fallback": enable_provider_fallback,
        "fallback_mode": fallback_mode,
        "enable_caching": enable_caching,
        "cache_ttl": cache_ttl,
        "enable_rate_limiting": enable_rate_limiting,
        "rate_limit_per_minute": rate_limit_per_minute,
        "enable_redis_sessions": enable_redis_sessions,
        "redis_url": redis_url,
        "enable_metrics": enable_metrics,
        "enable_prometheus": enable_prometheus,
        "prometheus_port": prometheus_port,
    }

    return cls(**config_data)
