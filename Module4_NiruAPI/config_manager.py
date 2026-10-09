"""
Configuration Manager for Encrypted API Keys and Settings
"""

import os
import time
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from sqlalchemy import Column, String, Text, create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.exc import OperationalError, DisconnectionError
from loguru import logger

Base = declarative_base()


class ConfigEntry(Base):
    """Database model for configuration entries"""

    __tablename__ = "configurations"

    key = Column(String(255), primary_key=True)
    encrypted_value = Column(Text, nullable=False)
    description = Column(String(500))


class ConfigManager:
    """Manage encrypted configuration storage with robust error handling"""

    def __init__(
        self,
        database_url: Optional[str] = None,
        encryption_key: Optional[str] = None,
        enable_db: bool = True,
    ):
        """
        Initialize config manager

        Args:
            database_url: PostgreSQL connection URL
            encryption_key: Base64 encoded encryption key
            enable_db: If False, operate in memory-only mode (no database)
        """
        self.enabled = enable_db
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.engine = None
        self.SessionLocal = None
        self._connection_healthy = False
        self._in_memory_cache: Dict[str, str] = {}

        # Initialize encryption
        if encryption_key is None:
            encryption_key = os.getenv("CONFIG_ENCRYPTION_KEY")
            if not encryption_key:
                encryption_key = Fernet.generate_key().decode()
                logger.warning(
                    "Generated new encryption key (set CONFIG_ENCRYPTION_KEY to persist)"
                )

        try:
            self.cipher = Fernet(encryption_key.encode())
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise

        # Initialize database connection if enabled
        if self.enabled and self.database_url:
            try:
                self._init_database()
            except Exception as e:
                logger.warning(f"Failed to initialize database connection: {e}")
                logger.warning("Config manager will operate in memory-only mode")
                self.enabled = False
                self._connection_healthy = False
        else:
            logger.info("Config manager operating in memory-only mode (no database)")
            self.enabled = False

    def _init_database(self):
        """Initialize database connection with robust settings"""
        # Configure connection pool with proper timeouts and SSL handling
        poolclass = QueuePool if "postgresql" in self.database_url else NullPool

        self.engine = create_engine(
            self.database_url,
            poolclass=poolclass,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
            connect_args={
                "connect_timeout": 10,  # 10 second connection timeout
                "sslmode": "prefer",  # Prefer SSL but don't require it
                "application_name": "amaniquery_config_manager",
            },
            echo=False,
        )

        # Add connection event listeners for better error handling
        @event.listens_for(self.engine, "connect")
        def set_connection_timeout(dbapi_conn, connection_record):
            """Set connection timeout"""
            try:
                with dbapi_conn.cursor() as cursor:
                    cursor.execute("SET statement_timeout = '30s'")
            except Exception:
                pass  # Ignore if not supported

        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        # Test connection
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self._connection_healthy = True
            logger.info("Database connection established")
        except Exception as e:
            logger.warning(f"Database connection test failed: {e}")
            self._connection_healthy = False
            raise

        # Create tables with retry
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Config manager database tables created/verified")
        except Exception as e:
            logger.warning(f"Failed to create tables: {e}")
            self._connection_healthy = False

    def _check_connection(self) -> bool:
        """Check if database connection is healthy"""
        if not self.enabled or not self.engine:
            return False

        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self._connection_healthy = True
            return True
        except Exception as e:
            logger.debug(f"Connection check failed: {e}")
            self._connection_healthy = False
            return False

    def _get_db(self) -> Optional[Session]:
        """Get database session with error handling"""
        if not self.enabled or not self.SessionLocal:
            return None

        try:
            if not self._connection_healthy:
                self._check_connection()
            return self.SessionLocal()
        except Exception as e:
            logger.warning(f"Failed to get database session: {e}")
            self._connection_healthy = False
            return None

    def _retry_db_operation(self, operation, max_retries: int = 3):
        """Retry database operation with exponential backoff"""
        for attempt in range(max_retries):
            try:
                return operation()
            except (OperationalError, DisconnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Database operation failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                    self._connection_healthy = False
                    # Try to reconnect
                    try:
                        self._check_connection()
                    except Exception:
                        pass
                else:
                    logger.error(
                        f"Database operation failed after {max_retries} attempts: {e}"
                    )
                    raise
            except Exception as e:
                logger.error(f"Unexpected error in database operation: {e}")
                raise
        return None

    def set_config(self, key: str, value: str, description: str = ""):
        """Set encrypted configuration value"""
        try:
            encrypted_value = self.cipher.encrypt(value.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt config {key}: {e}")
            return

        # Store in memory cache
        self._in_memory_cache[key] = value

        # Store in database if available
        if self.enabled:

            def _set_operation():
                db = self._get_db()
                if not db:
                    return
                try:
                    entry = db.query(ConfigEntry).filter(ConfigEntry.key == key).first()
                    if entry:
                        entry.encrypted_value = encrypted_value
                        entry.description = description
                    else:
                        entry = ConfigEntry(
                            key=key,
                            encrypted_value=encrypted_value,
                            description=description,
                        )
                        db.add(entry)
                    db.commit()
                    logger.debug(f"Set config in database: {key}")
                except Exception as e:
                    db.rollback()
                    raise
                finally:
                    db.close()

            try:
                self._retry_db_operation(_set_operation)
                logger.info(f"Set config: {key}")
            except Exception as e:
                logger.warning(
                    f"Failed to set config in database (using memory cache): {e}"
                )
        else:
            logger.debug(f"Set config in memory: {key}")

    def get_config(self, key: str) -> Optional[str]:
        """Get decrypted configuration value"""
        # First check memory cache
        if key in self._in_memory_cache:
            return self._in_memory_cache[key]

        # Try database if available
        if self.enabled:

            def _get_operation():
                db = self._get_db()
                if not db:
                    return None
                try:
                    entry = db.query(ConfigEntry).filter(ConfigEntry.key == key).first()
                    if entry:
                        try:
                            value = self.cipher.decrypt(
                                entry.encrypted_value.encode()
                            ).decode()
                            # Cache in memory
                            self._in_memory_cache[key] = value
                            return value
                        except Exception as e:
                            logger.error(f"Failed to decrypt config {key}: {e}")
                            return None
                    return None
                finally:
                    db.close()

            try:
                return self._retry_db_operation(_get_operation)
            except Exception as e:
                logger.debug(f"Failed to get config from database: {e}")
                return None

        return None

    def delete_config(self, key: str):
        """Delete configuration entry"""
        # Remove from memory cache
        self._in_memory_cache.pop(key, None)

        # Delete from database if available
        if self.enabled:

            def _delete_operation():
                db = self._get_db()
                if not db:
                    return
                try:
                    entry = db.query(ConfigEntry).filter(ConfigEntry.key == key).first()
                    if entry:
                        db.delete(entry)
                        db.commit()
                        logger.debug(f"Deleted config from database: {key}")
                except Exception as e:
                    db.rollback()
                    raise
                finally:
                    db.close()

            try:
                self._retry_db_operation(_delete_operation)
                logger.info(f"Deleted config: {key}")
            except Exception as e:
                logger.warning(f"Failed to delete config from database: {e}")

    def list_configs(self) -> Dict[str, Dict[str, Any]]:
        """List all configuration keys with descriptions"""
        result = {}

        # Add memory cache entries
        for key in self._in_memory_cache:
            result[key] = {
                "description": "In-memory cache",
                "has_value": True,
                "source": "memory",
            }

        # Add database entries if available
        if self.enabled:

            def _list_operation():
                db = self._get_db()
                if not db:
                    return {}
                try:
                    entries = db.query(ConfigEntry).all()
                    return {
                        entry.key: {
                            "description": entry.description or "",
                            "has_value": bool(entry.encrypted_value),
                            "source": "database",
                        }
                        for entry in entries
                    }
                finally:
                    db.close()

            try:
                db_result = self._retry_db_operation(_list_operation)
                if db_result:
                    # Merge database results (database takes precedence)
                    result.update(db_result)
            except Exception as e:
                logger.debug(f"Failed to list configs from database: {e}")

        return result

    def is_available(self) -> bool:
        """Check if config manager is available and healthy"""
        if not self.enabled:
            return True  # Memory mode is always available
        return self._connection_healthy or self._check_connection()
