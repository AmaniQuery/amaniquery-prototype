"""
Unified Connection Pool Manager for NiruDB.
Single source of truth for all database connections at 1M+ concurrent scale.
Features:
- Single engine per database URL (eliminates module-specific pools)
- Read replica support (reader/writer split)
- Prometheus-compatible pool metrics
- Connection health monitoring
- Statement timeout enforcement
- Async engine support
"""

import os
import time
import threading
from typing import Dict, Optional, Any, List, Tuple
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache

from loguru import logger
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError

# Async support
try:
    from sqlalchemy.ext.asyncio import (
        create_async_engine,
        AsyncSession,
        async_sessionmaker,
    )

    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False


@dataclass
class PoolMetrics:
    size: int = 0
    checked_in: int = 0
    checked_out: int = 0
    overflow: int = 0
    wait_count: int = 0
    wait_ms: float = 0.0
    total_connections: int = 0


@dataclass
class EngineConfig:
    pool_size: int = 100
    max_overflow: int = 200
    pool_recycle: int = 300
    pool_pre_ping: bool = True
    pool_timeout: int = 30
    pool_use_lifo: bool = True
    connect_timeout: int = 10
    statement_timeout_ms: int = 30000
    echo: bool = False


class EngineGroup:
    """Holds writer + reader engines for a single database URL"""

    def __init__(
        self,
        writer_url: str,
        reader_urls: Optional[List[str]] = None,
        config: Optional[EngineConfig] = None,
    ):
        self.config = config or EngineConfig()
        self.writer_url = writer_url
        self.reader_urls = reader_urls or []

        self.writer_engine = self._build_engine(writer_url)
        self.writer_session_factory = scoped_session(
            sessionmaker(bind=self.writer_engine, expire_on_commit=False)
        )

        self.reader_engines: List[Any] = []
        self.reader_session_factories: List[Any] = []
        for url in self.reader_urls:
            eng = self._build_engine(url)
            self.reader_engines.append(eng)
            self.reader_session_factories.append(
                scoped_session(sessionmaker(bind=eng, expire_on_commit=False))
            )

        self._async_writer = None
        self._async_reader = None
        self._round_robin = 0
        self._lock = threading.Lock()

    def _build_engine(self, url: str):
        connect_args = {"connect_timeout": self.config.connect_timeout}
        if "neon.tech" in url or "postgresql" in url:
            connect_args["options"] = (
                f"-c statement_timeout={self.config.statement_timeout_ms}"
            )

        engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_pre_ping=self.config.pool_pre_ping,
            pool_recycle=self.config.pool_recycle,
            pool_timeout=self.config.pool_timeout,
            pool_use_lifo=self.config.pool_use_lifo,
            echo=self.config.echo,
            connect_args=connect_args,
        )

        @event.listens_for(engine, "connect")
        def _on_connect(dbapi_connection, connection_record):
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute("SET timezone = 'UTC'")
                cursor.close()
            except Exception:
                pass

        return engine

    def get_writer_session(self) -> Session:
        return self.writer_session_factory()

    def get_reader_session(self) -> Session:
        if not self.reader_engines:
            return self.writer_session_factory()
        with self._lock:
            idx = self._round_robin % len(self.reader_engines)
            self._round_robin += 1
        return self.reader_session_factories[idx]()

    def get_async_writer(self):
        if self._async_writer is None and ASYNC_AVAILABLE:
            async_url = self.writer_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
            async_url = async_url.replace("postgres://", "postgresql+asyncpg://")
            self._async_writer = create_async_engine(
                async_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=self.config.pool_pre_ping,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
            )
        return self._async_writer

    def get_reader_metrics(self) -> List[PoolMetrics]:
        metrics = []
        for eng in [self.writer_engine] + self.reader_engines:
            try:
                pool = eng.pool
                metrics.append(
                    PoolMetrics(
                        size=pool.size(),
                        checked_in=pool.checkedin(),
                        checked_out=pool.checkedout(),
                        overflow=pool.overflow(),
                    )
                )
            except Exception:
                pass
        return metrics

    def dispose(self):
        try:
            self.writer_engine.dispose()
        except Exception:
            pass
        for eng in self.reader_engines:
            try:
                eng.dispose()
            except Exception:
                pass


class ConnectionPoolManager:
    """
    Singleton managing ALL database connections across the entire application.
    Single engine per URL eliminates the ~145 separate connection limit.
    Supports read replicas for read/write splitting at 1M+ scale.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._groups: Dict[str, EngineGroup] = {}
        self._async_engines: Dict[str, Any] = {}
        self._async_factories: Dict[str, Any] = {}
        self._initialized = True
        logger.info(
            "ConnectionPoolManager initialized (pool_size=100, max_overflow=200)"
        )

    def get_engine_group(
        self,
        database_url: Optional[str] = None,
        reader_urls: Optional[List[str]] = None,
        config: Optional[EngineConfig] = None,
    ) -> EngineGroup:
        """Get or create an engine group for the given writer URL."""
        if database_url is None:
            database_url = os.getenv(
                "DATABASE_URL", "postgresql://localhost/amaniquery"
            )

        # For Neon, prefer unpooled for direct connections
        if "neon.tech" in database_url and "pooler" in database_url:
            unpooled = os.getenv("DATABASE_URL_UNPOOLED")
            if unpooled:
                database_url = unpooled

        if database_url not in self._groups:
            self._groups[database_url] = EngineGroup(
                writer_url=database_url,
                reader_urls=reader_urls or [],
                config=config,
            )
            logger.info(
                f"Engine group created for {database_url[:60]}... "
                f"(pool_size={config.pool_size if config else 100}, "
                f"readers={len(reader_urls or [])})"
            )

        return self._groups[database_url]

    def get_async_session_factory(self, database_url: Optional[str] = None):
        """Get or create async session factory for the given URL."""
        if not ASYNC_AVAILABLE:
            raise RuntimeError(
                "Async SQLAlchemy not available. Install: pip install sqlalchemy[asyncio] asyncpg"
            )

        if database_url is None:
            database_url = os.getenv(
                "DATABASE_URL", "postgresql://localhost/amaniquery"
            )

        if database_url not in self._async_engines:
            async_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
            async_url = async_url.replace("postgres://", "postgresql+asyncpg://")

            engine = create_async_engine(
                async_url,
                pool_size=100,
                max_overflow=200,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False,
            )
            self._async_engines[database_url] = engine
            self._async_factories[database_url] = async_sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )

        return self._async_factories[database_url]

    def get_health(self) -> Dict[str, Any]:
        """Aggregate pool health across all engine groups."""
        writer_metrics = []
        all_healthy = True
        for url, group in self._groups.items():
            metrics = group.get_reader_metrics()
            for m in metrics:
                if m.size > 0 and m.checked_out > m.size + m.overflow:
                    all_healthy = False
            writer_metrics.append(
                {
                    "url": url[:60] + "...",
                    "metrics": [
                        {
                            "size": m.size,
                            "checked_in": m.checked_in,
                            "checked_out": m.checked_out,
                            "overflow": m.overflow,
                            "utilization_pct": round(
                                m.checked_out / max(m.size, 1) * 100, 1
                            ),
                        }
                        for m in metrics
                    ],
                }
            )

        return {
            "status": "healthy" if all_healthy else "degraded",
            "engine_groups": len(self._groups),
            "writers": writer_metrics,
        }

    def dispose_all(self):
        """Dispose all engine groups (call on shutdown)."""
        for group in self._groups.values():
            try:
                group.dispose()
            except Exception:
                pass
        self._groups.clear()
        logger.info("All connection pools disposed")


# Global singleton
pool_manager = ConnectionPoolManager()
