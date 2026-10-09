"""
MinIO Object Storage Client
Robust wrapper for MinIO operations with retries and error handling.
"""

from minio import Minio
from minio.error import S3Error
import urllib3
import time
from loguru import logger
from .config import settings


class MinioStorage:
    def __init__(self):
        self.client = None
        self._initialize()

    def _initialize(self):
        """Initialize MinIO client with retries"""
        try:
            # Create custom HTTP client with connection pooling
            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout.DEFAULT_TIMEOUT,
                cert_reqs="CERT_REQUIRED",
                retries=urllib3.Retry(
                    total=3, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504]
                ),
            )

            self.client = Minio(
                endpoint=settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
                http_client=http_client,
            )

            # Ensure bucket exists
            if not self.client.bucket_exists(settings.MINIO_BUCKET):
                try:
                    self.client.make_bucket(settings.MINIO_BUCKET)
                    logger.info(f"Created MinIO bucket: {settings.MINIO_BUCKET}")
                except S3Error as e:
                    # Ignore if bucket already exists (race condition)
                    if e.code != "BucketAlreadyOwnedByYou":
                        raise e

            logger.info(
                f"MinIO client initialized connected to {settings.MINIO_ENDPOINT}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            # Don't raise here, allow lazy initialization retry

    def get_object(self, object_name: str) -> bytes:
        """Get object content as bytes"""
        if not self.client:
            self._initialize()
            if not self.client:
                raise RuntimeError("MinIO client not initialized")

        try:
            response = self.client.get_object(settings.MINIO_BUCKET, object_name)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception as e:
            logger.error(f"Error fetching object {object_name}: {e}")
            raise

    def put_object(
        self, object_name: str, data: bytes, content_type: str = "application/json"
    ):
        """Upload object"""
        if not self.client:
            self._initialize()
            if not self.client:
                raise RuntimeError("MinIO client not initialized")

        import io

        try:
            self.client.put_object(
                settings.MINIO_BUCKET,
                object_name,
                io.BytesIO(data),
                len(data),
                content_type=content_type,
            )
        except Exception as e:
            logger.error(f"Error putting object {object_name}: {e}")
            raise


# Global instance
minio_storage = MinioStorage()
