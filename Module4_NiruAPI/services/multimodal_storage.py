"""
Multimodal Storage Service - Persistent storage for vision/media assets

Provides:
- Database-backed storage for images, videos, audio, and their embeddings
- Session-based asset management
- Cloudinary integration for file storage
- Asset lifecycle management (expiration, cleanup)
"""

import os
import uuid
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    DateTime,
    Float,
    Integer,
    JSON,
    Boolean,
)
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class VisionAsset(Base):
    """Vision/Media Asset model"""

    __tablename__ = "vision_assets"

    id = Column(String(100), primary_key=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=True, index=True)

    # File information
    file_type = Column(String(20), nullable=False, index=True)
    file_name = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=True)
    cloudinary_url = Column(String(1000), nullable=True)
    cloudinary_public_id = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String(100), nullable=True)

    # Embedding data
    embedding = Column(JSON, nullable=True)
    embedding_model = Column(String(100), nullable=True)

    # Extracted content
    extracted_text = Column(Text, nullable=True)
    extraction_confidence = Column(Float, nullable=True)

    # Video/audio specific
    duration = Column(Float, nullable=True)
    timestamp = Column(Float, nullable=True)
    frame_number = Column(Integer, nullable=True)
    parent_asset_id = Column(String(100), nullable=True, index=True)

    # Transcription
    segments = Column(JSON, nullable=True)
    language = Column(String(20), nullable=True)

    # Metadata
    metadata_json = Column(JSON, nullable=True)
    processing_status = Column(String(50), default="pending", index=True)
    processing_error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True, index=True)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "file_type": self.file_type,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "cloudinary_url": self.cloudinary_url,
            "embedding": self.embedding,
            "extracted_text": self.extracted_text,
            "extraction_confidence": self.extraction_confidence,
            "duration": self.duration,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "parent_asset_id": self.parent_asset_id,
            "segments": self.segments,
            "language": self.language,
            "metadata": self.metadata_json,
            "processing_status": self.processing_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def to_rag_format(self) -> Dict:
        """Convert to format expected by VisionRAGService"""
        return {
            "id": self.id,
            "file_path": self.file_path or self.cloudinary_url,
            "embedding": self.embedding,
            "metadata": {
                **(self.metadata_json or {}),
                "filename": self.file_name,
                "file_type": self.file_type,
                "source_type": self.file_type,
            },
        }


class VisionSession(Base):
    """Vision Session model for tracking multimodal sessions"""

    __tablename__ = "vision_sessions"

    id = Column(String(100), primary_key=True)
    user_id = Column(String(100), nullable=True, index=True)
    name = Column(String(200), nullable=True)
    asset_count = Column(Integer, default=0)
    total_size = Column(Integer, default=0)
    modalities = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True, index=True)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "asset_count": self.asset_count,
            "total_size": self.total_size,
            "modalities": self.modalities,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class MultimodalStorageService:
    """
    Service for managing multimodal assets with database persistence

    Replaces in-memory vision_storage dict with proper database storage
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        default_expiry_hours: int = 24,
        cloudinary_service: Optional[any] = None,
    ):
        """
        Initialize multimodal storage service

        Args:
            database_url: Database connection URL
            default_expiry_hours: Default asset expiration in hours
            cloudinary_service: Optional Cloudinary service for file upload
        """
        if database_url is None:
            database_url = os.getenv(
                "DATABASE_URL", "postgresql://localhost/amaniquery"
            )

        # Handle Neon connection pooling
        if "neon.tech" in database_url and "pooler" in database_url:
            unpooled_url = os.getenv("DATABASE_URL_UNPOOLED")
            if unpooled_url:
                database_url = unpooled_url

        self.engine = create_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=300,
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.default_expiry_hours = default_expiry_hours
        self.cloudinary_service = cloudinary_service

        logger.info("Multimodal storage service initialized")

    def _get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    def generate_asset_id(self) -> str:
        """Generate unique asset ID"""
        return f"asset_{uuid.uuid4().hex[:16]}"

    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"vsession_{uuid.uuid4().hex[:12]}"

    # ==================== SESSION MANAGEMENT ====================

    def create_session(
        self,
        user_id: Optional[str] = None,
        name: Optional[str] = None,
        expiry_hours: Optional[int] = None,
    ) -> Dict:
        """
        Create a new vision session

        Args:
            user_id: Optional user ID
            name: Optional session name
            expiry_hours: Hours until session expires

        Returns:
            Session dict
        """
        session_id = self.generate_session_id()
        expiry = expiry_hours or self.default_expiry_hours

        db = self._get_session()
        try:
            session = VisionSession(
                id=session_id,
                user_id=user_id,
                name=name or f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                asset_count=0,
                total_size=0,
                modalities=[],
                expires_at=datetime.utcnow() + timedelta(hours=expiry),
            )

            db.add(session)
            db.commit()
            db.refresh(session)

            logger.info(f"Created vision session: {session_id}")
            return session.to_dict()

        except Exception as e:
            db.rollback()
            logger.error(f"Error creating session: {e}")
            raise
        finally:
            db.close()

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID"""
        db = self._get_session()
        try:
            session = (
                db.query(VisionSession).filter(VisionSession.id == session_id).first()
            )

            return session.to_dict() if session else None
        finally:
            db.close()

    def delete_session(self, session_id: str) -> bool:
        """Delete session and all its assets"""
        db = self._get_session()
        try:
            # Delete all assets in session
            db.query(VisionAsset).filter(VisionAsset.session_id == session_id).delete()

            # Delete session
            deleted = (
                db.query(VisionSession).filter(VisionSession.id == session_id).delete()
            )

            db.commit()

            if deleted:
                logger.info(f"Deleted session: {session_id}")

            return deleted > 0

        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting session: {e}")
            return False
        finally:
            db.close()

    # ==================== ASSET MANAGEMENT ====================

    def add_asset(
        self,
        session_id: str,
        file_type: str,
        file_name: str,
        file_path: Optional[str] = None,
        cloudinary_url: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        extracted_text: Optional[str] = None,
        metadata: Optional[Dict] = None,
        user_id: Optional[str] = None,
        file_size: Optional[int] = None,
        mime_type: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Add an asset to the storage

        Args:
            session_id: Session ID to add asset to
            file_type: Type of file (image, pdf, audio, video, video_frame)
            file_name: Original file name
            file_path: Local file path
            cloudinary_url: CDN URL
            embedding: Vector embedding
            extracted_text: OCR or transcription text
            metadata: Additional metadata
            user_id: User ID
            file_size: Size in bytes
            mime_type: MIME type
            **kwargs: Additional fields (duration, timestamp, frame_number, etc.)

        Returns:
            Asset dict
        """
        asset_id = self.generate_asset_id()

        db = self._get_session()
        try:
            asset = VisionAsset(
                id=asset_id,
                session_id=session_id,
                user_id=user_id,
                file_type=file_type,
                file_name=file_name,
                file_path=file_path,
                cloudinary_url=cloudinary_url,
                file_size=file_size,
                mime_type=mime_type,
                embedding=embedding,
                embedding_model=kwargs.get("embedding_model", "cohere-embed-4"),
                extracted_text=extracted_text,
                extraction_confidence=kwargs.get("extraction_confidence"),
                duration=kwargs.get("duration"),
                timestamp=kwargs.get("timestamp"),
                frame_number=kwargs.get("frame_number"),
                parent_asset_id=kwargs.get("parent_asset_id"),
                segments=kwargs.get("segments"),
                language=kwargs.get("language"),
                metadata_json=metadata,
                processing_status=kwargs.get("processing_status", "completed"),
                expires_at=datetime.utcnow()
                + timedelta(hours=self.default_expiry_hours),
            )

            db.add(asset)

            # Update session stats
            session = (
                db.query(VisionSession).filter(VisionSession.id == session_id).first()
            )

            if session:
                session.asset_count = (session.asset_count or 0) + 1
                session.total_size = (session.total_size or 0) + (file_size or 0)

                # Update modalities
                modalities = set(session.modalities or [])
                modalities.add(file_type)
                session.modalities = list(modalities)
                session.updated_at = datetime.utcnow()

            db.commit()
            db.refresh(asset)

            logger.debug(
                f"Added asset: {asset_id} ({file_type}) to session {session_id}"
            )
            return asset.to_dict()

        except Exception as e:
            db.rollback()
            logger.error(f"Error adding asset: {e}")
            raise
        finally:
            db.close()

    def get_asset(self, asset_id: str) -> Optional[Dict]:
        """Get asset by ID"""
        db = self._get_session()
        try:
            asset = db.query(VisionAsset).filter(VisionAsset.id == asset_id).first()

            return asset.to_dict() if asset else None
        finally:
            db.close()

    def get_session_assets(
        self,
        session_id: str,
        file_type: Optional[str] = None,
        include_children: bool = True,
    ) -> List[Dict]:
        """
        Get all assets in a session

        Args:
            session_id: Session ID
            file_type: Optional filter by file type
            include_children: Include child assets (e.g., video frames)

        Returns:
            List of asset dicts
        """
        db = self._get_session()
        try:
            query = db.query(VisionAsset).filter(VisionAsset.session_id == session_id)

            if file_type:
                query = query.filter(VisionAsset.file_type == file_type)

            if not include_children:
                query = query.filter(VisionAsset.parent_asset_id == None)

            query = query.order_by(VisionAsset.created_at.desc())

            return [asset.to_dict() for asset in query.all()]

        finally:
            db.close()

    def get_session_assets_for_rag(
        self,
        session_id: str,
        include_images: bool = True,
        include_video_frames: bool = True,
        include_transcripts: bool = True,
    ) -> Dict[str, List[Dict]]:
        """
        Get assets formatted for RAG queries

        Args:
            session_id: Session ID
            include_images: Include image assets
            include_video_frames: Include video frame assets
            include_transcripts: Include audio transcripts

        Returns:
            Dict with 'images' and 'transcripts' lists
        """
        db = self._get_session()
        try:
            result = {
                "images": [],
                "transcripts": [],
            }

            # Get image assets
            if include_images or include_video_frames:
                image_types = []
                if include_images:
                    image_types.extend(["image", "pdf"])
                if include_video_frames:
                    image_types.append("video_frame")

                images = (
                    db.query(VisionAsset)
                    .filter(
                        VisionAsset.session_id == session_id,
                        VisionAsset.file_type.in_(image_types),
                        VisionAsset.embedding != None,
                    )
                    .all()
                )

                result["images"] = [img.to_rag_format() for img in images]

            # Get transcript assets
            if include_transcripts:
                transcripts = (
                    db.query(VisionAsset)
                    .filter(
                        VisionAsset.session_id == session_id,
                        VisionAsset.file_type.in_(["audio", "video"]),
                        VisionAsset.extracted_text != None,
                    )
                    .all()
                )

                for t in transcripts:
                    result["transcripts"].append(
                        {
                            "id": t.id,
                            "text": t.extracted_text,
                            "embedding": t.embedding,
                            "segments": t.segments,
                            "language": t.language,
                            "duration": t.duration,
                            "source_path": t.file_path or t.cloudinary_url,
                        }
                    )

            return result

        finally:
            db.close()

    def update_asset(
        self,
        asset_id: str,
        **updates,
    ) -> Optional[Dict]:
        """
        Update asset fields

        Args:
            asset_id: Asset ID
            **updates: Fields to update

        Returns:
            Updated asset dict or None
        """
        db = self._get_session()
        try:
            asset = db.query(VisionAsset).filter(VisionAsset.id == asset_id).first()

            if not asset:
                return None

            # Update allowed fields
            allowed_fields = {
                "embedding",
                "extracted_text",
                "extraction_confidence",
                "segments",
                "language",
                "processing_status",
                "processing_error",
                "metadata_json",
                "cloudinary_url",
                "cloudinary_public_id",
            }

            for key, value in updates.items():
                if key in allowed_fields:
                    setattr(asset, key, value)

            asset.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(asset)

            return asset.to_dict()

        except Exception as e:
            db.rollback()
            logger.error(f"Error updating asset: {e}")
            return None
        finally:
            db.close()

    def delete_asset(self, asset_id: str) -> bool:
        """Delete asset and its children"""
        db = self._get_session()
        try:
            # Delete child assets first
            db.query(VisionAsset).filter(
                VisionAsset.parent_asset_id == asset_id
            ).delete()

            # Delete asset
            deleted = db.query(VisionAsset).filter(VisionAsset.id == asset_id).delete()

            db.commit()
            return deleted > 0

        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting asset: {e}")
            return False
        finally:
            db.close()

    # ==================== CLEANUP & MAINTENANCE ====================

    def cleanup_expired(self) -> int:
        """
        Delete expired sessions and assets

        Returns:
            Number of items deleted
        """
        db = self._get_session()
        try:
            now = datetime.utcnow()

            # Get expired session IDs
            expired_sessions = (
                db.query(VisionSession.id).filter(VisionSession.expires_at < now).all()
            )

            expired_ids = [s.id for s in expired_sessions]

            # Delete assets in expired sessions
            assets_deleted = (
                db.query(VisionAsset)
                .filter(VisionAsset.session_id.in_(expired_ids))
                .delete(synchronize_session=False)
            )

            # Delete expired assets individually
            assets_deleted += (
                db.query(VisionAsset)
                .filter(VisionAsset.expires_at < now)
                .delete(synchronize_session=False)
            )

            # Delete expired sessions
            sessions_deleted = (
                db.query(VisionSession)
                .filter(VisionSession.expires_at < now)
                .delete(synchronize_session=False)
            )

            db.commit()

            total = assets_deleted + sessions_deleted
            if total > 0:
                logger.info(
                    f"Cleaned up {assets_deleted} assets and {sessions_deleted} sessions"
                )

            return total

        except Exception as e:
            db.rollback()
            logger.error(f"Error during cleanup: {e}")
            return 0
        finally:
            db.close()

    def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        db = self._get_session()
        try:
            total_sessions = db.query(func.count(VisionSession.id)).scalar()
            total_assets = db.query(func.count(VisionAsset.id)).scalar()
            total_size = db.query(func.sum(VisionAsset.file_size)).scalar() or 0

            # Assets by type
            type_counts = (
                db.query(VisionAsset.file_type, func.count(VisionAsset.id))
                .group_by(VisionAsset.file_type)
                .all()
            )

            return {
                "total_sessions": total_sessions,
                "total_assets": total_assets,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "assets_by_type": dict(type_counts),
            }

        finally:
            db.close()


# ==================== LEGACY COMPATIBILITY ====================


class InMemoryMultimodalStorage:
    """
    In-memory fallback storage for when database is not available

    Provides same interface as MultimodalStorageService for compatibility
    """

    def __init__(self):
        """Initialize in-memory storage"""
        self._sessions: Dict[str, Dict] = {}
        self._assets: Dict[str, Dict] = {}
        logger.warning("Using in-memory multimodal storage (not persistent)")

    def generate_asset_id(self) -> str:
        return f"asset_{uuid.uuid4().hex[:16]}"

    def generate_session_id(self) -> str:
        return f"vsession_{uuid.uuid4().hex[:12]}"

    def create_session(self, user_id=None, name=None, **kwargs) -> Dict:
        session_id = self.generate_session_id()
        session = {
            "id": session_id,
            "user_id": user_id,
            "name": name,
            "asset_count": 0,
            "modalities": [],
            "created_at": datetime.utcnow().isoformat(),
        }
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Dict]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        # Delete assets
        asset_ids = [
            aid for aid, a in self._assets.items() if a.get("session_id") == session_id
        ]
        for aid in asset_ids:
            del self._assets[aid]

        # Delete session
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def add_asset(
        self, session_id: str, file_type: str, file_name: str, **kwargs
    ) -> Dict:
        asset_id = self.generate_asset_id()
        asset = {
            "id": asset_id,
            "session_id": session_id,
            "file_type": file_type,
            "file_name": file_name,
            "created_at": datetime.utcnow().isoformat(),
            **kwargs,
        }
        self._assets[asset_id] = asset

        # Update session
        if session_id in self._sessions:
            self._sessions[session_id]["asset_count"] += 1
            modalities = set(self._sessions[session_id].get("modalities", []))
            modalities.add(file_type)
            self._sessions[session_id]["modalities"] = list(modalities)

        return asset

    def get_asset(self, asset_id: str) -> Optional[Dict]:
        return self._assets.get(asset_id)

    def get_session_assets(
        self, session_id: str, file_type=None, **kwargs
    ) -> List[Dict]:
        assets = [a for a in self._assets.values() if a.get("session_id") == session_id]
        if file_type:
            assets = [a for a in assets if a.get("file_type") == file_type]
        return assets

    def get_session_assets_for_rag(
        self, session_id: str, **kwargs
    ) -> Dict[str, List[Dict]]:
        assets = self.get_session_assets(session_id)

        return {
            "images": [
                {
                    "id": a["id"],
                    "file_path": a.get("file_path") or a.get("cloudinary_url"),
                    "embedding": a.get("embedding"),
                    "metadata": a.get("metadata", {}),
                }
                for a in assets
                if a.get("file_type") in ("image", "pdf", "video_frame")
                and a.get("embedding")
            ],
            "transcripts": [
                {
                    "id": a["id"],
                    "text": a.get("extracted_text", ""),
                    "embedding": a.get("embedding"),
                    "segments": a.get("segments", []),
                    "language": a.get("language"),
                    "duration": a.get("duration"),
                    "source_path": a.get("file_path"),
                }
                for a in assets
                if a.get("file_type") in ("audio", "video") and a.get("extracted_text")
            ],
        }

    def cleanup_expired(self) -> int:
        return 0  # No expiry in in-memory mode


def create_multimodal_storage(
    use_database: bool = True,
) -> Union[MultimodalStorageService, InMemoryMultimodalStorage]:
    """
    Factory function to create appropriate storage backend

    Args:
        use_database: Whether to use database storage

    Returns:
        Storage service instance
    """
    if use_database:
        try:
            return MultimodalStorageService()
        except Exception as e:
            logger.warning(f"Database storage unavailable, using in-memory: {e}")
            return InMemoryMultimodalStorage()
    else:
        return InMemoryMultimodalStorage()
