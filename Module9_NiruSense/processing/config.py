import os
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class Settings(BaseSettings):
    """Production-ready configuration for NiruSense processing pipeline"""

    # Redis Configuration
    REDIS_HOST: str = Field(default="localhost", description="Redis host address")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_STREAM_KEY: str = Field(
        default="niru_ingestion_stream", description="Redis stream key for ingestion"
    )
    REDIS_CONSUMER_GROUP: str = Field(
        default="niru_processing_group", description="Consumer group name"
    )
    REDIS_DLQ_KEY: str = Field(
        default="niru_processing_dlq",
        description="Dead letter queue for failed messages",
    )
    REDIS_MAX_RETRIES: int = Field(
        default=3, description="Maximum retry attempts for failed messages"
    )

    # PostgreSQL Configuration (Neon.tech)
    DATABASE_URL: str = Field(
        default="postgresql://user:password@ep-xyz.aws.neon.tech/sautisense?sslmode=require",
        description="PostgreSQL connection string",
    )
    DB_POOL_MIN_SIZE: int = Field(
        default=2, description="Minimum database connection pool size"
    )
    DB_POOL_MAX_SIZE: int = Field(
        default=10, description="Maximum database connection pool size"
    )
    DB_TIMEOUT: int = Field(default=30, description="Database query timeout in seconds")

    # Qdrant Configuration (Cloud)
    QDRANT_URL: str = Field(
        default="https://xyz-example.eu-central.aws.cloud.qdrant.io:6333",
        description="Qdrant instance URL",
    )
    QDRANT_API_KEY: str = Field(default="your-api-key", description="Qdrant API key")
    QDRANT_COLLECTION: str = Field(
        default="amaniquery_sense", description="Qdrant collection name"
    )
    QDRANT_BATCH_SIZE: int = Field(
        default=100, description="Batch size for Qdrant upserts"
    )

    # MinIO/S3 Configuration (Required for Raw Data)
    MINIO_ENDPOINT: str = Field(default="localhost:9000", description="MinIO endpoint")
    MINIO_ACCESS_KEY: str = Field(default="admin", description="MinIO access key")
    MINIO_SECRET_KEY: str = Field(
        default="miniopassword123", description="MinIO secret key"
    )
    MINIO_SECURE: bool = Field(default=False, description="Use HTTPS for MinIO")
    MINIO_BUCKET: str = Field(default="bronze-raw", description="MinIO bucket name")

    # Elasticsearch Configuration (Optional - for full text search)
    ELASTICSEARCH_URL: Optional[str] = Field(
        default=None, description="Elasticsearch URL"
    )
    ELASTICSEARCH_API_KEY: Optional[str] = Field(
        default=None, description="Elasticsearch API key"
    )
    ELASTICSEARCH_USERNAME: Optional[str] = Field(
        default=None, description="Elasticsearch username"
    )
    ELASTICSEARCH_PASSWORD: Optional[str] = Field(
        default=None, description="Elasticsearch password"
    )
    ELASTICSEARCH_INDEX: str = Field(
        default="amani_query", description="Elasticsearch index name"
    )

    # Model IDs (HuggingFace)
    MODEL_EMBEDDING: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5",
        description="Embedding model for vector generation",
    )
    MODEL_LANGUAGE: str = Field(
        default="papluca/xlm-roberta-base-language-detection",
        description="Language identification model",
    )
    MODEL_SLANG: str = Field(
        default="google/flan-t5-base",
        description="Slang/Sheng decoder model (use meta-llama/Llama-3.2-3B-Instruct with GPU)",
    )
    MODEL_TOPIC: str = Field(
        default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        description="Topic classification model",
    )
    MODEL_NER: str = Field(
        default="Davlan/xlm-roberta-base-ner-hrl",
        description="Named Entity Recognition model",
    )
    MODEL_SENTIMENT: str = Field(
        default="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        description="Sentiment analysis model",
    )
    MODEL_EMOTION: str = Field(
        default="j-hartmann/emotion-english-distilroberta-base",
        description="Emotion detection model",
    )
    MODEL_BIAS: str = Field(
        default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        description="Bias detection model (reuses topic model)",
    )
    MODEL_SUMMARIZER: str = Field(
        default="google/mt5-small", description="Text summarization model"
    )

    # Agent Configuration
    AGENT_TIMEOUT: int = Field(
        default=30, description="Agent processing timeout in seconds"
    )
    AGENT_MAX_RETRIES: int = Field(
        default=2, description="Maximum retries for agent failures"
    )
    TOPIC_CONFIDENCE_THRESHOLD: float = Field(
        default=0.4, description="Minimum confidence for topic classification"
    )
    ENTITY_CONFIDENCE_THRESHOLD: float = Field(
        default=0.7, description="Minimum confidence for entity extraction"
    )
    BIAS_THRESHOLD: float = Field(
        default=0.5, description="Threshold for bias detection"
    )

    # Kenyan-Specific Topics
    KENYAN_TOPICS: List[str] = Field(
        default=[
            "Politics",
            "Economy",
            "Sports",
            "Social Issues",
            "Technology",
            "Entertainment",
            "Security",
            "Education",
            "Health",
            "Agriculture",
            "Infrastructure",
            "Corruption",
            "Tribalism",
            "Religion",
            "Culture",
        ],
        description="Kenyan-specific topic categories",
    )

    # Processing Configuration
    BATCH_SIZE: int = Field(
        default=5, description="Number of documents to process concurrently"
    )
    MAX_TEXT_LENGTH: int = Field(
        default=10000, description="Maximum text length to process"
    )
    ENABLE_PARALLEL_AGENTS: bool = Field(
        default=True, description="Enable parallel agent execution"
    )

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    # Health Check Configuration
    HEALTH_CHECK_PORT: int = Field(
        default=8000, description="Port for health check endpoint"
    )
    ENABLE_HEALTH_CHECKS: bool = Field(
        default=True, description="Enable health check endpoints"
    )

    # Environment
    ENVIRONMENT: str = Field(
        default="development", description="Environment (development/production)"
    )
    DEBUG: bool = Field(default=False, description="Debug mode")

    model_config = SettingsConfigDict(
        env_file="../.env",  # Load from root AmaniQuery .env file
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        """Validate database URL format"""
        if not v.startswith("postgresql://"):
            raise ValueError("DATABASE_URL must start with postgresql://")
        return v

    @validator("QDRANT_URL")
    def validate_qdrant_url(cls, v):
        """Validate Qdrant URL format"""
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("QDRANT_URL must start with http:// or https://")
        return v

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()


# Global settings instance
settings = Settings()
