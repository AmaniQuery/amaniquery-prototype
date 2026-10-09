"""
Database Migration Script for Agent Monitoring
Creates the agent_query_logs table
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
    """Create agent_query_logs table"""
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        return False

    try:
        logger.info(f"Connecting to database...")
        engine = create_engine(database_url)
        metadata = MetaData()

        logger.info("Creating agent_query_logs table...")

        # Create table without foreign keys to avoid dependency issues
        agent_query_logs = Table(
            "agent_query_logs",
            metadata,
            Column("id", String, primary_key=True),
            Column("session_id", String, nullable=False, index=True),
            Column("user_id", String, nullable=True, index=True),
            Column(
                "timestamp",
                DateTime,
                default=datetime.utcnow,
                nullable=False,
                index=True,
            ),
            Column("query", Text, nullable=False),
            Column("persona", String, nullable=False, index=True),
            Column("intent", String, nullable=False, index=True),
            Column("confidence", Float, nullable=False, index=True),
            Column("response_time_ms", Integer, nullable=False),
            Column("evidence_count", Integer, default=0),
            Column("reasoning_steps", Integer, default=0),
            Column("human_review_required", Boolean, default=False, index=True),
            Column("agent_path", JSON, nullable=True),
            Column("quality_issues", JSON, nullable=True),
            Column("reasoning_path", JSON, nullable=True),
            Column("user_feedback", String, nullable=True),
            Column("reviewed_at", DateTime, nullable=True),
            Column("reviewed_by", String, nullable=True),
            Column("review_status", String, nullable=True),
            Column("review_feedback", Text, nullable=True),
        )

        # Create table
        metadata.create_all(engine, checkfirst=True)

        logger.info("✓ Migration completed successfully")
        logger.info(
            "Note: Foreign key constraints not created. Add them manually if needed."
        )
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
