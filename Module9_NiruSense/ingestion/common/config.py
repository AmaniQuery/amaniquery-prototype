import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "admin"
    MINIO_SECRET_KEY: str = "miniopassword123"
    MINIO_BUCKET_RAW: str = "bronze-raw"
    MINIO_SECURE: bool = False

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_STREAM_KEY: str = "niru_ingestion_stream"

    # Scraper Settings
    TWITTER_KEYWORDS: list = [
        "Kenya",
        "Ruto",
        "Raila",
        "Maandamano",
        "Finance Bill",
        "Githinji",
        "MustGo",
        "MustGoKenya",
    ]
    TIKTOK_HASHTAGS: list = [
        "kenya",
        "nairobi",
        "genzkenya",
        "maandamano",
        "financebill",
        "githinji",
        "mustgo",
    ]
    NEWS_RSS_FEEDS: list = [
        "https://www.standardmedia.co.ke/rss/headlines.php",
        "https://nation.africa/rss",
        "https://www.the-star.co.ke/rss",
        "https://www.capitalfm.co.ke/news/rss/",
        "https://www.pulselive.co.ke/rss",
    ]

    class Config:
        env_file = ".env"


settings = Settings()
