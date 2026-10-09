import json
import time
from datetime import datetime
from minio import Minio
import redis
from .config import settings
import io


class StorageManager:
    def __init__(self):
        # Initialize MinIO
        self.minio_client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self._ensure_bucket(settings.MINIO_BUCKET_RAW)

        # Initialize Redis
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True
        )

    def _ensure_bucket(self, bucket_name):
        if not self.minio_client.bucket_exists(bucket_name):
            self.minio_client.make_bucket(bucket_name)

    def save_raw_data(self, source: str, data: dict):
        """
        Saves raw data to MinIO and pushes event to Redis Stream.
        """
        timestamp = datetime.now().isoformat()
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Add metadata
        data["ingestion_timestamp"] = timestamp
        data["source"] = source

        # 1. Save to MinIO (Bronze Layer)
        # Path: source/date/timestamp_id.json
        object_name = (
            f"{source}/{date_str}/{int(time.time())}_{data.get('id', 'unknown')}.json"
        )
        content = json.dumps(data).encode("utf-8")

        self.minio_client.put_object(
            settings.MINIO_BUCKET_RAW,
            object_name,
            io.BytesIO(content),
            length=len(content),
            content_type="application/json",
        )

        # 2. Push to Redis Stream for processing
        stream_payload = {
            "s3_key": object_name,
            "source": source,
            "timestamp": timestamp,
        }
        self.redis_client.xadd(settings.REDIS_STREAM_KEY, stream_payload)

        return object_name


storage = StorageManager()
