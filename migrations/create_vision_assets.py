"""
Database Migration Script for Vision/Media Assets
Creates the vision_assets table for persistent multimodal storage
"""

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    DateTime,
    Boolean,
    Float,
    Integer,
    JSON,
    MetaData,
    Table,
    Index,
)
from datetime import datetime
from dotenv import load_dotenv
import os
import sys
import logging

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration():
    """Create vision_assets table for multimodal storage"""
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        return False

    try:
        logger.info(f"Connecting to database...")
        engine = create_engine(database_url)
        metadata = MetaData()

        logger.info("Creating vision_assets table...")

        # Vision/Media Assets table
        vision_assets = Table(
            "vision_assets",
            metadata,
            # Primary key
            Column("id", String(100), primary_key=True),
            # Session and user linking
            Column("session_id", String(100), nullable=False, index=True),
            Column("user_id", String(100), nullable=True, index=True),
            # File information
            Column(
                "file_type", String(20), nullable=False, index=True
            ),  # image, pdf, audio, video, video_frame
            Column("file_name", String(500), nullable=False),
            Column("file_path", String(1000), nullable=True),  # Local file path
            Column("cloudinary_url", String(1000), nullable=True),  # CDN URL
            Column("cloudinary_public_id", String(500), nullable=True),
            Column("file_size", Integer, nullable=True),  # Size in bytes
            Column("mime_type", String(100), nullable=True),
            # Embedding data
            Column("embedding", JSON, nullable=True),  # Vector embedding as JSON array
            Column(
                "embedding_model", String(100), nullable=True
            ),  # e.g., "cohere-embed-4"
            # Extracted content (for OCR, transcription)
            Column("extracted_text", Text, nullable=True),
            Column("extraction_confidence", Float, nullable=True),
            # Video/audio specific fields
            Column("duration", Float, nullable=True),  # Duration in seconds
            Column(
                "timestamp", Float, nullable=True
            ),  # Timestamp within source (for frames)
            Column("frame_number", Integer, nullable=True),  # For video frames
            Column(
                "parent_asset_id", String(100), nullable=True, index=True
            ),  # Link to source video/audio
            # Transcription segments (for audio)
            Column("segments", JSON, nullable=True),  # Timestamped segments
            Column("language", String(20), nullable=True),  # Detected language
            # Metadata
            Column("metadata_json", JSON, nullable=True),  # Additional metadata
            Column(
                "processing_status", String(50), default="pending", index=True
            ),  # pending, processing, completed, failed
            Column("processing_error", Text, nullable=True),
            # Timestamps
            Column(
                "created_at",
                DateTime,
                default=datetime.utcnow,
                nullable=False,
                index=True,
            ),
            Column(
                "updated_at",
                DateTime,
                default=datetime.utcnow,
                onupdate=datetime.utcnow,
            ),
            Column("expires_at", DateTime, nullable=True, index=True),  # For cleanup
        )

        # Create table
        metadata.create_all(engine, checkfirst=True)

        # Create additional indexes
        with engine.connect() as conn:
            # Composite index for session + file_type queries
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS ix_vision_assets_session_type ON vision_assets (session_id, file_type)"
                )
            except Exception:
                pass  # Index may already exist

            # Index for parent asset lookups (video frames)
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS ix_vision_assets_parent ON vision_assets (parent_asset_id) WHERE parent_asset_id IS NOT NULL"
                )
            except Exception:
                pass

            conn.commit()

        logger.info("✓ vision_assets table created successfully")

        # Also create vision_sessions table for tracking multimodal sessions
        logger.info("Creating vision_sessions table...")

        vision_sessions = Table(
            "vision_sessions",
            metadata,
            Column("id", String(100), primary_key=True),
            Column("user_id", String(100), nullable=True, index=True),
            Column("name", String(200), nullable=True),
            Column("asset_count", Integer, default=0),
            Column("total_size", Integer, default=0),  # Total bytes
            Column("modalities", JSON, nullable=True),  # List of modalities used
            Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
            Column(
                "updated_at",
                DateTime,
                default=datetime.utcnow,
                onupdate=datetime.utcnow,
            ),
            Column("expires_at", DateTime, nullable=True, index=True),
        )

        metadata.create_all(engine, checkfirst=True)

        logger.info("✓ vision_sessions table created successfully")
        logger.info("✓ Migration completed successfully")

        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def rollback_migration():
    """Drop vision tables (use with caution)"""
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        return False

    try:
        engine = create_engine(database_url)

        with engine.connect() as conn:
            conn.execute("DROP TABLE IF EXISTS vision_assets CASCADE")
            conn.execute("DROP TABLE IF EXISTS vision_sessions CASCADE")
            conn.commit()

        logger.info("✓ Tables dropped successfully")
        return True

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vision assets migration")
    parser.add_argument(
        "--rollback", action="store_true", help="Rollback migration (drop tables)"
    )
    args = parser.parse_args()

    if args.rollback:
        confirm = input("This will DROP all vision tables. Type 'yes' to confirm: ")
        if confirm.lower() == "yes":
            success = rollback_migration()
        else:
            print("Rollback cancelled")
            success = True
    else:
        success = run_migration()

    sys.exit(0 if success else 1)
