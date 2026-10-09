"""
Cloudinary Service for File Storage
Handles uploading chat attachments (PDFs, images, text files) to Cloudinary
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api

    CLOUDINARY_AVAILABLE = True
except ImportError:
    CLOUDINARY_AVAILABLE = False
    logger.warning("Cloudinary not available. Install with: pip install cloudinary")


class CloudinaryService:
    """Service for uploading files to Cloudinary"""

    def __init__(self):
        """Initialize Cloudinary with credentials from environment variables"""
        if not CLOUDINARY_AVAILABLE:
            raise ImportError(
                "Cloudinary package not installed. Install with: pip install cloudinary"
            )

        cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
        api_key = os.getenv("CLOUDINARY_API_KEY")
        api_secret = os.getenv("CLOUDINARY_API_SECRET")

        if not all([cloud_name, api_key, api_secret]):
            raise ValueError(
                "Cloudinary credentials not found. Set CLOUDINARY_CLOUD_NAME, "
                "CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET environment variables."
            )

        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True,  # Use HTTPS
        )

        logger.info("Cloudinary service initialized")

    def upload_file(
        self,
        file_path: Path,
        filename: str,
        session_id: str,
        resource_type: Optional[str] = None,
        folder: str = "chat_attachments",
    ) -> Dict[str, Any]:
        """
        Upload a file to Cloudinary

        Args:
            file_path: Path to the file to upload
            filename: Original filename
            session_id: Chat session ID for organizing files
            resource_type: Cloudinary resource type ('image', 'raw', 'auto')
                          If None, will be auto-detected
            folder: Cloudinary folder path

        Returns:
            Dictionary with:
            - url: Public URL of the uploaded file
            - secure_url: HTTPS URL
            - public_id: Cloudinary public ID
            - format: File format
            - resource_type: Resource type used
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine resource type if not provided
        if resource_type is None:
            resource_type = self._detect_resource_type(filename)

        # Create folder path with session ID
        cloudinary_folder = f"{folder}/{session_id}"

        # Prepare upload options
        upload_options = {
            "folder": cloudinary_folder,
            "use_filename": True,
            "unique_filename": True,
            "resource_type": resource_type,
        }

        # For images, add optimization options
        if resource_type == "image":
            upload_options.update(
                {
                    "quality": "auto",
                    "fetch_format": "auto",
                }
            )

        try:
            # Upload file
            result = cloudinary.uploader.upload(str(file_path), **upload_options)

            logger.info(f"Uploaded {filename} to Cloudinary: {result.get('public_id')}")

            return {
                "url": result.get("url"),
                "secure_url": result.get("secure_url"),
                "public_id": result.get("public_id"),
                "format": result.get("format"),
                "resource_type": result.get("resource_type"),
                "bytes": result.get("bytes"),
                "width": result.get("width"),
                "height": result.get("height"),
            }

        except Exception as e:
            logger.error(f"Failed to upload {filename} to Cloudinary: {e}")
            raise

    def upload_bytes(
        self,
        file_content: bytes,
        filename: str,
        session_id: str,
        resource_type: Optional[str] = None,
        folder: str = "chat_attachments",
    ) -> Dict[str, Any]:
        """
        Upload file content from bytes to Cloudinary

        Args:
            file_content: File content as bytes
            filename: Original filename
            session_id: Chat session ID for organizing files
            resource_type: Cloudinary resource type ('image', 'raw', 'auto')
            folder: Cloudinary folder path

        Returns:
            Dictionary with upload result
        """
        import tempfile

        # Determine resource type if not provided
        if resource_type is None:
            resource_type = self._detect_resource_type(filename)

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(filename).suffix
        ) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = Path(tmp_file.name)

        try:
            result = self.upload_file(
                file_path=tmp_path,
                filename=filename,
                session_id=session_id,
                resource_type=resource_type,
                folder=folder,
            )
            return result
        finally:
            # Clean up temporary file
            try:
                tmp_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")

    def _detect_resource_type(self, filename: str) -> str:
        """
        Detect Cloudinary resource type from filename

        Args:
            filename: File name

        Returns:
            Resource type: 'image', 'raw', or 'auto'
        """
        ext = Path(filename).suffix.lower()

        # Image formats
        image_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".webp",
            ".svg",
            ".ico",
        }
        if ext in image_extensions:
            return "image"

        # For PDFs and other files, use 'raw' to preserve original format
        return "raw"

    def delete_file(self, public_id: str, resource_type: Optional[str] = None) -> bool:
        """
        Delete a file from Cloudinary

        Args:
            public_id: Cloudinary public ID
            resource_type: Resource type ('image', 'raw', 'auto')

        Returns:
            True if successful
        """
        try:
            result = cloudinary.uploader.destroy(
                public_id, resource_type=resource_type or "auto"
            )
            return result.get("result") == "ok"
        except Exception as e:
            logger.error(f"Failed to delete file {public_id} from Cloudinary: {e}")
            return False

    def get_thumbnail_url(
        self, public_id: str, width: int = 300, height: int = 300
    ) -> str:
        """
        Get a thumbnail URL for an image

        Args:
            public_id: Cloudinary public ID
            width: Thumbnail width
            height: Thumbnail height

        Returns:
            Thumbnail URL
        """
        return cloudinary.CloudinaryImage(public_id).build_url(
            transformation=[
                {"width": width, "height": height, "crop": "limit", "quality": "auto"}
            ]
        )
