import asyncpg
import json
from typing import Optional, Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from datetime import datetime
from ..config import settings
from ..monitoring import logger


class PostgresClient:
    """PostgreSQL client with connection pooling and retry logic"""

    def __init__(self):
        self.dsn = settings.DATABASE_URL
        self.pool: Optional[asyncpg.Pool] = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(asyncpg.PostgresConnectionError),
    )
    async def connect(self):
        """Connect to PostgreSQL with retry logic"""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.dsn,
                min_size=settings.DB_POOL_MIN_SIZE,
                max_size=settings.DB_POOL_MAX_SIZE,
                command_timeout=settings.DB_TIMEOUT,
            )
            await self._init_db()
            print(
                f"PostgreSQL connected: {settings.DB_POOL_MIN_SIZE}-{settings.DB_POOL_MAX_SIZE} pool"
            )

    async def _init_db(self):
        """Initialize database schema"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    url TEXT UNIQUE NOT NULL,
                    raw_content TEXT,
                    normalized_content TEXT,
                    source_domain VARCHAR(255),
                    published_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_domain);
                CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
                
                CREATE TABLE IF NOT EXISTS analysis_results (
                    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                    agent_id VARCHAR(50),
                    result_json JSONB,
                    model_version VARCHAR(50),
                    execution_time_ms INT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (document_id, agent_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_analysis_agent ON analysis_results(agent_id);
            """)

    async def save_document(self, doc: Dict[str, Any]) -> str:
        """Save document with proper field handling"""
        async with self.pool.acquire() as conn:
            # Parse published_at if available
            published_at = None
            if doc.get("published_at"):
                try:
                    published_at = datetime.fromisoformat(doc["published_at"])
                except (ValueError, TypeError):
                    pass

            row = await conn.fetchrow(
                """
                INSERT INTO documents (url, raw_content, normalized_content, source_domain, published_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (url) DO UPDATE 
                SET raw_content = EXCLUDED.raw_content,
                    normalized_content = EXCLUDED.normalized_content
                RETURNING id
            """,
                doc.get("url", "unknown"),
                doc.get("raw_content", doc.get("text", "")),
                doc.get("normalized_content", ""),
                doc.get("source", doc.get("source_domain", "unknown")),
                published_at,
            )
            return str(row["id"])

    async def save_analysis(self, doc_id: str, agent_id: str, result: Dict[str, Any]):
        """Save agent analysis results"""
        # Extract execution time from meta if available
        execution_time = result.get("_meta", {}).get("execution_time_ms", 0)
        model_version = result.get("_meta", {}).get("model_version", "v1")

        # Remove meta before storing
        result_clean = {k: v for k, v in result.items() if k != "_meta"}

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO analysis_results (document_id, agent_id, result_json, model_version, execution_time_ms)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (document_id, agent_id) 
                DO UPDATE SET 
                    result_json = EXCLUDED.result_json,
                    execution_time_ms = EXCLUDED.execution_time_ms,
                    created_at = NOW()
            """,
                doc_id,
                agent_id,
                json.dumps(result_clean),
                model_version,
                execution_time,
            )

    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return {
                    "status": "healthy" if result == 1 else "degraded",
                    "pool_size": self.pool.get_size(),
                    "pool_free": self.pool.get_idle_size(),
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def close(self):
        """Close database pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its UUID"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, url, raw_content, normalized_content, source_domain, published_at, created_at FROM documents WHERE id = $1",
                    doc_id,
                )
                if not row:
                    return None
                return {
                    "id": str(row["id"]),
                    "url": row["url"],
                    "raw_content": row["raw_content"],
                    "normalized_content": row["normalized_content"],
                    "source": row["source_domain"],
                    "published_at": (
                        row["published_at"].isoformat() if row["published_at"] else None
                    ),
                    "created_at": (
                        row["created_at"].isoformat() if row["created_at"] else None
                    ),
                }
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    async def get_document_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get a document by its source URL"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, url, raw_content, normalized_content, source_domain, published_at, created_at FROM documents WHERE url = $1",
                    url,
                )
                if not row:
                    return None
                return {
                    "id": str(row["id"]),
                    "url": row["url"],
                    "raw_content": row["raw_content"],
                    "normalized_content": row["normalized_content"],
                    "source": row["source_domain"],
                    "published_at": (
                        row["published_at"].isoformat() if row["published_at"] else None
                    ),
                    "created_at": (
                        row["created_at"].isoformat() if row["created_at"] else None
                    ),
                }
        except Exception as e:
            logger.error(f"Failed to get document by url {url}: {e}")
            return None

    async def get_analysis(self, doc_id: str) -> Dict[str, Any]:
        """Get all analysis results for a document"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT agent_id, result_json, model_version, execution_time_ms, created_at FROM analysis_results WHERE document_id = $1",
                    doc_id,
                )
                result = {}
                for row in rows:
                    result[row["agent_id"]] = {
                        "result": row["result_json"],
                        "model_version": row["model_version"],
                        "execution_time_ms": row["execution_time_ms"],
                        "created_at": (
                            row["created_at"].isoformat() if row["created_at"] else None
                        ),
                    }
                return result
        except Exception as e:
            logger.error(f"Failed to get analysis for {doc_id}: {e}")
            return {}

    async def cleanup_old_data(self, days: int = 90):
        """
        Cleanup old documents and analysis results

        Args:
            days: Delete documents older than this many days
        """
        try:
            async with self.pool.acquire() as conn:
                # Count and delete old analysis results first
                analysis_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM analysis_results
                    WHERE document_id IN (
                        SELECT id FROM documents
                        WHERE created_at < NOW() - $1 * INTERVAL '1 day'
                    )
                    """,
                    days,
                )

                await conn.execute(
                    """
                    DELETE FROM analysis_results
                    WHERE document_id IN (
                        SELECT id FROM documents
                        WHERE created_at < NOW() - $1 * INTERVAL '1 day'
                    )
                    """,
                    days,
                )

                # Count and delete old documents
                docs_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM documents
                    WHERE created_at < NOW() - $1 * INTERVAL '1 day'
                    """,
                    days,
                )

                await conn.execute(
                    """
                    DELETE FROM documents
                    WHERE created_at < NOW() - $1 * INTERVAL '1 day'
                    """,
                    days,
                )

                return {
                    "documents_deleted": docs_count,
                    "analysis_deleted": analysis_count,
                }
        except Exception as e:
            return {"error": str(e)}

    async def get_failed_documents(self, limit: int = 100):
        """
        Get documents that failed processing (no analysis results)

        Args:
            limit: Maximum number of failed documents to retrieve

        Returns:
            List of document data for reprocessing
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT d.id, d.url, d.raw_content, d.source_domain,
                           d.published_at, d.created_at
                    FROM documents d
                    LEFT JOIN analysis_results ar ON d.id = ar.document_id
                    WHERE ar.document_id IS NULL
                    AND d.created_at > NOW() - INTERVAL '7 days'
                    ORDER BY d.created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )

                return [
                    {
                        "id": str(row["id"]),
                        "url": row["url"],
                        "text": row["raw_content"],
                        "source": row["source_domain"],
                        "source_domain": row["source_domain"],
                        "published_at": (
                            row["published_at"].isoformat()
                            if row["published_at"]
                            else None
                        ),
                        "timestamp": (
                            int(row["created_at"].timestamp())
                            if row["created_at"]
                            else None
                        ),
                    }
                    for row in rows
                ]
        except Exception as e:
            return []


postgres = PostgresClient()
