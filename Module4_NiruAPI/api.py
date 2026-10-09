"""
FastAPI Application - REST API for AmaniQuery
Refactored main entry point with modular routers
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
import asyncio
import threading
import time
import json
from datetime import datetime
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from loguru import logger

# Core services
from Module4_NiruAPI.rag_pipeline import RAGPipeline
from Module4_NiruAPI.alignment_pipeline import ConstitutionalAlignmentPipeline
from Module4_NiruAPI.sms_pipeline import SMSPipeline
from Module4_NiruAPI.sms_service import AfricasTalkingSMSService
from Module3_NiruDB.chat_manager import ChatDatabaseManager
from Module3_NiruDB.vector_store import VectorStore
from Module3_NiruDB.metadata_manager import MetadataManager

# Models
from Module4_NiruAPI.models import (
    HealthResponse,
    StatsResponse,
)

# Modules
from Module4_NiruAPI.research_module import ResearchModule  # Legacy fallback
from Module4_NiruAPI.research_module_agentic import AgenticResearchModule
from Module4_NiruAPI.config_manager import ConfigManager
from Module4_NiruAPI.report_generator import ReportGenerator
from Module4_NiruAPI.cache import get_cache_manager, CacheManager
from Module4_NiruAPI.crawler_manager import CrawlerManager
from Module4_NiruAPI.services.aggressive_cache import get_aggressive_cache
from Module4_NiruAPI.services.notification_service import NotificationService
from Module4_NiruAPI.services.research_bundle import ResearchBundleService
from Module4_NiruAPI.agents.tools.autocomplete import AutocompleteTool

# Routers - External modules
from Module5_NiruShare.api import router as share_router
from Module4_NiruAPI.routers.news_router import router as news_router
from Module4_NiruAPI.routers.websocket_router import (
    router as websocket_router,
    broadcast_new_article,
)
from Module4_NiruAPI.routers.notification_router import router as notification_router

# Routers - Internal modules (new refactored routers)
from Module4_NiruAPI.routers.query_router import router as query_router
from Module4_NiruAPI.routers.chat_router import router as chat_router
from Module4_NiruAPI.routers.admin_router import router as admin_router
from Module4_NiruAPI.routers.research_router import router as research_router
from Module4_NiruAPI.routers.sms_router import router as sms_router
from Module4_NiruAPI.routers.alignment_router import router as alignment_router
from Module4_NiruAPI.routers.monitoring_router import router as monitoring_router
from Module4_NiruAPI.routers.hybrid_rag_router import router as hybrid_rag_router
from Module4_NiruAPI.routers.nirusense_router import router as nirusense_router
from Module4_NiruAPI.routers.clustering_router import router as clustering_router
from Module4_NiruAPI.routers.finetuning_router import router as finetuning_router
from Module4_NiruAPI.routers.feedback_analytics_router import (
    router as feedback_analytics_router,
)
from Module4_NiruAPI.routers.voice_router import router as voice_router
from Module4_NiruAPI.routers.media_router import router as media_router
from Module4_NiruAPI.routers.admin_settings_router import (
    router as admin_settings_router,
)

# ============================================================
# Global service instances (shared across routers)
# ============================================================
vector_store: Optional[VectorStore] = None
rag_pipeline: Optional[RAGPipeline] = None
alignment_pipeline: Optional[ConstitutionalAlignmentPipeline] = None
sms_pipeline: Optional[SMSPipeline] = None
sms_service: Optional[AfricasTalkingSMSService] = None
metadata_manager: Optional[MetadataManager] = None
chat_manager: Optional[ChatDatabaseManager] = None
crawler_manager: Optional[CrawlerManager] = None
research_module: Optional[ResearchModule] = None
agentic_research_module: Optional[AgenticResearchModule] = None
report_generator: Optional[ReportGenerator] = None
research_bundle_service: Optional[ResearchBundleService] = None
config_manager: Optional[ConfigManager] = None
notification_service: Optional[NotificationService] = None
hybrid_rag_pipeline = None
autocomplete_tool: Optional[AutocompleteTool] = None
vision_storage: Dict = {}  # In-memory storage: {session_id: [image_data, ...]}
vision_rag_service = None
database_storage = None
cache_manager: Optional[CacheManager] = None
amaniq_v2_agent = None  # AmaniQ v2 agent instance
tool_registry = None  # Global tool registry initialized once at startup

# NiruSense service instances
nirusense_pipeline = None
nirusense_health_checker = None
nirusense_metrics = None
nirusense_scheduler = None
nirusense_settings = None
nirusense_orchestrator_task = None
nirusense_orchestrator_thread = None


# ============================================================
# Dependency Injection Functions
# ============================================================
def get_vector_store():
    """Dependency for vector store"""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return vector_store


def get_rag_pipeline():
    """Dependency for RAG pipeline"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return rag_pipeline


def get_alignment_pipeline():
    """Dependency for alignment pipeline"""
    if alignment_pipeline is None:
        raise HTTPException(
            status_code=503, detail="Alignment pipeline not initialized"
        )
    return alignment_pipeline


def get_sms_pipeline():
    """Dependency for SMS pipeline"""
    if sms_pipeline is None:
        raise HTTPException(status_code=503, detail="SMS pipeline not initialized")
    return sms_pipeline


def get_sms_service():
    """Dependency for SMS service"""
    return sms_service


def get_chat_manager():
    """Dependency for chat manager"""
    if chat_manager is None:
        raise HTTPException(status_code=503, detail="Chat manager not initialized")
    return chat_manager


def get_crawler_manager():
    """Dependency for crawler manager"""
    if crawler_manager is None:
        raise HTTPException(status_code=503, detail="Crawler manager not initialized")
    return crawler_manager


def get_research_module():
    """Dependency for research module (prefers agentic)"""
    return agentic_research_module or research_module


def get_report_generator():
    """Dependency for report generator"""
    return report_generator


def get_config_manager():
    """Dependency for config manager"""
    return config_manager


def get_cache_manager_dep():
    """Dependency for cache manager"""
    return cache_manager


def get_tool_registry():
    """Dependency for tool registry"""
    return tool_registry


def get_hybrid_rag_pipeline():
    """Dependency for hybrid RAG pipeline"""
    return hybrid_rag_pipeline


def get_vision_rag_service():
    """Dependency for vision RAG service"""
    return vision_rag_service


def get_vision_storage():
    """Dependency for vision storage"""
    return vision_storage


def get_database_storage():
    """Dependency for database storage"""
    return database_storage


def get_amaniq_v2_agent():
    """Dependency for Amaniq v2 agent"""
    return amaniq_v2_agent


def get_autocomplete_tool():
    """Dependency for autocomplete tool"""
    return autocomplete_tool


# ============================================================
# Lifespan Context Manager
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    global vector_store, rag_pipeline, alignment_pipeline, sms_pipeline, sms_service
    global metadata_manager, chat_manager, crawler_manager, research_module
    global agentic_research_module, report_generator, config_manager
    global notification_service, hybrid_rag_pipeline, autocomplete_tool
    global vision_storage, vision_rag_service, database_storage, cache_manager
    global amaniq_v2_agent, tool_registry, research_bundle_service
    global nirusense_pipeline, nirusense_health_checker, nirusense_metrics
    global nirusense_scheduler, nirusense_settings
    global nirusense_orchestrator_thread

    logger.info("Starting AmaniQuery API")

    # Initialize tool registry
    try:
        from Module4_NiruAPI.agents.tools.tool_registry import ToolRegistry

        logger.info("Initializing global tool registry...")
        tool_registry = ToolRegistry()
        logger.info(
            f"[OK] Tool registry initialized with {len(tool_registry.tools)} tools: {tool_registry.list_tools()}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize tool registry: {e}")
        tool_registry = None

    # Initialize config manager
    try:
        config_manager = ConfigManager()
        logger.info("Config manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize config manager: {e}")
        config_manager = None

    # Initialize vector store
    try:
        backend = os.getenv(
            "VECTOR_STORE_BACKEND", "auto"
        )  # Use "auto" for automatic fallback
        vector_store = VectorStore(backend=backend, config_manager=config_manager)
        logger.info(f"Vector store initialized with backend: {backend}")
    except Exception as e:
        logger.error(
            f"Failed to initialize vector store with backend '{os.getenv('VECTOR_STORE_BACKEND', 'auto')}': {e}"
        )
        # Try fallback to auto mode if specific backend was requested
        if (
            os.getenv("VECTOR_STORE_BACKEND")
            and os.getenv("VECTOR_STORE_BACKEND") != "auto"
        ):
            logger.info("Attempting fallback to auto backend selection...")
            try:
                vector_store = VectorStore(
                    backend="auto", config_manager=config_manager
                )
                logger.info(f"Vector store initialized with fallback (auto mode)")
            except Exception as e2:
                logger.error(f"Fallback vector store initialization also failed: {e2}")
                vector_store = None
        else:
            vector_store = None

    # Initialize metadata manager
    try:
        if vector_store:
            metadata_manager = MetadataManager(vector_store)
            logger.info("Metadata manager initialized")
        else:
            metadata_manager = None
    except Exception as e:
        logger.error(f"Failed to initialize metadata manager: {e}")
        metadata_manager = None

    # Initialize RAG pipeline
    try:
        if vector_store:
            llm_provider = os.getenv("LLM_PROVIDER", "moonshot")
            model = os.getenv("DEFAULT_MODEL", "moonshot-v1-8k")

            rag_pipeline = RAGPipeline(
                vector_store=vector_store,
                llm_provider=llm_provider,
                model=model,
            )
            logger.info("RAG pipeline initialized")
        else:
            rag_pipeline = None
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        rag_pipeline = None

    # Initialize Constitutional Alignment Pipeline
    try:
        if vector_store and rag_pipeline:
            alignment_pipeline = ConstitutionalAlignmentPipeline(
                vector_store=vector_store,
                rag_pipeline=rag_pipeline,
            )
            logger.info("Alignment pipeline initialized")
        else:
            alignment_pipeline = None
    except Exception as e:
        logger.error(f"Failed to initialize alignment pipeline: {e}")
        alignment_pipeline = None

    # Initialize SMS Pipeline
    try:
        if vector_store and rag_pipeline:
            sms_pipeline = SMSPipeline(
                vector_store=vector_store, llm_service=rag_pipeline.llm_service
            )
            logger.info("SMS pipeline initialized")
        else:
            sms_pipeline = None
    except Exception as e:
        logger.error(f"Failed to initialize SMS pipeline: {e}")
        sms_pipeline = None

    # Initialize SMS Service (Africa's Talking)
    try:
        sms_service = AfricasTalkingSMSService()
        if sms_service.available:
            logger.info("SMS service initialized and available")
        else:
            logger.warning(
                "SMS service initialized but not available (check credentials)"
            )
    except Exception as e:
        logger.error(f"Failed to initialize SMS service: {e}")
        sms_service = None

    # Initialize Hybrid RAG Pipeline (optional - for enhanced retrieval)
    try:
        if vector_store and rag_pipeline:
            from Module7_NiruHybrid.integration.rag_integration import HybridRAGPipeline
            from Module7_NiruHybrid.hybrid_encoder import HybridEncoder
            from Module7_NiruHybrid.retention.adaptive_retriever import (
                AdaptiveRetriever,
            )
            from Module7_NiruHybrid.config import default_config

            hybrid_encoder = HybridEncoder(config=default_config.encoder)
            adaptive_retriever = AdaptiveRetriever(
                hybrid_encoder=hybrid_encoder,
                vector_store=vector_store,
                config=default_config.retention,
            )

            hybrid_rag_pipeline = HybridRAGPipeline(
                base_rag_pipeline=rag_pipeline,
                hybrid_encoder=hybrid_encoder,
                adaptive_retriever=adaptive_retriever,
                use_hybrid=True,
                use_adaptive_retrieval=True,
                config=default_config,
            )
            logger.info("Hybrid RAG pipeline initialized")
        else:
            hybrid_rag_pipeline = None
            logger.warning(
                "Hybrid RAG pipeline not initialized: vector_store or rag_pipeline not available"
            )
    except Exception as e:
        logger.warning(f"Hybrid RAG pipeline not available: {e}")
        hybrid_rag_pipeline = None

    # Initialize chat database manager
    try:
        chat_manager = ChatDatabaseManager()
        logger.info("Chat manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize chat manager: {e}")
        chat_manager = None

    # Initialize cache manager
    try:
        cache_manager = get_cache_manager(config_manager)
        logger.info("Cache manager initialized")

        # Initialize Aggressive Semantic Cache with warmup
        try:
            agg_cache = get_aggressive_cache()
            logger.info("Aggressive semantic cache initialized")

            # Warm up with common Kenyan legal queries in background
            def warmup_worker():
                try:
                    import time as t

                    t.sleep(5)  # Defer warmup to let server finish starting
                    agg_cache.warmup()
                    logger.info("Aggressive semantic cache warmup complete")
                except Exception as e:
                    logger.warning(f"Cache warmup failed: {e}")

            import threading as _th

            warmup_thread = _th.Thread(target=warmup_worker, daemon=True)
            warmup_thread.start()
        except Exception as e:
            logger.warning(f"Aggressive semantic cache not available: {e}")

        # Start Redis Pub/Sub listener for invalidation
        if (
            cache_manager
            and cache_manager.redis_client
            and hasattr(cache_manager.redis_client, "pubsub")
        ):

            def redis_listener():
                try:
                    pubsub = cache_manager.redis_client.pubsub()
                    pubsub.subscribe("bill_updated")
                    logger.info(
                        "[LISTEN] Listening for cache invalidation events on 'bill_updated'"
                    )
                    for message in pubsub.listen():
                        if message["type"] == "message":
                            bill_name = message["data"]
                            if isinstance(bill_name, bytes):
                                bill_name = bill_name.decode("utf-8")
                            logger.info(
                                f"[CLEAN] Invalidation event received for: {bill_name}"
                            )
                            cache_manager.delete_pattern(f"*{bill_name}*")
                except Exception as e:
                    logger.error(f"Redis listener error: {e}")

            invalidation_thread = threading.Thread(target=redis_listener, daemon=True)
            invalidation_thread.start()
            logger.info("Redis invalidation listener started")

    except Exception as e:
        logger.warning(f"Failed to initialize cache manager: {e}")
        cache_manager = None

    # Initialize DatabaseStorage first (needed for crawler manager)
    try:
        from Module3_NiruDB.database_storage import DatabaseStorage

        database_storage = DatabaseStorage()
        logger.info("Database storage initialized")
    except Exception as e:
        logger.warning(f"Database storage not available: {e}")
        database_storage = None

    # Initialize crawler manager
    try:
        crawler_manager = CrawlerManager(database_storage=database_storage)
        logger.info("Crawler manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize crawler manager: {e}")
        crawler_manager = None

    # Initialize research module (try agentic first, fallback to legacy)
    agentic_research_module = None
    research_module = None
    try:
        # Pass shared tool_registry for faster initialization
        agentic_research_module = AgenticResearchModule(
            config_manager=config_manager, tool_registry=tool_registry
        )
        logger.info("Agentic research module initialized")
    except Exception as e:
        logger.warning(
            f"Agentic research module not available: {e}, falling back to legacy module"
        )
        try:
            research_module = ResearchModule()
            logger.info("Legacy research module initialized")
        except Exception as e2:
            logger.warning(f"Legacy research module also not available: {e2}")
            research_module = None

    # Initialize report generator (optional - only if Gemini API key is available)
    try:
        report_generator = ReportGenerator()
        logger.info("Report generator initialized")
    except Exception as e:
        logger.warning(f"Report generator not available: {e}")
        report_generator = None

    # Initialize Research Bundle Service
    try:
        research_bundle_service = ResearchBundleService(
            research_module=research_module,
            agentic_research_module=agentic_research_module,
            report_generator=report_generator,
            cache_manager=cache_manager,
        )
        logger.info("Research bundle service initialized")
    except Exception as e:
        logger.warning(f"Research bundle service not available: {e}")
        research_bundle_service = None

    # Initialize authentication module (if enabled)
    auth_enabled = os.getenv("ENABLE_AUTH", "false").lower() == "true"
    if auth_enabled:
        try:
            logger.info("Initializing authentication module...")

            # Check and create auth database tables if needed
            try:
                from Module3_NiruDB.chat_models import create_database_engine
                from Module8_NiruAuth.models.auth_models import Base
                from sqlalchemy import inspect

                database_url = os.getenv("DATABASE_URL")
                if database_url:
                    engine = create_database_engine(database_url)
                    with engine.connect() as conn:
                        inspector = inspect(conn)
                        existing_tables = inspector.get_table_names()

                        required_tables = ["users", "roles", "api_keys"]
                        missing_tables = [
                            t for t in required_tables if t not in existing_tables
                        ]

                        if missing_tables:
                            logger.info(
                                f"Creating missing auth tables: {missing_tables}"
                            )
                            Base.metadata.create_all(engine)
                            logger.info("[OK] Auth tables created successfully")

                            try:
                                from sqlalchemy.orm import sessionmaker
                                from Module8_NiruAuth.authorization.role_manager import (
                                    RoleManager,
                                )

                                Session = sessionmaker(bind=engine)
                                db = Session()
                                try:
                                    RoleManager.get_or_create_default_roles(db)
                                    logger.info("[OK] Default roles initialized")
                                finally:
                                    db.close()
                            except Exception as e:
                                logger.warning(
                                    f"Could not initialize default roles: {e}"
                                )
                        else:
                            logger.info("[OK] Auth tables already exist")
            except Exception as e:
                logger.warning(f"Auth database initialization check failed: {e}")
                logger.warning(
                    "You may need to run 'python migrate_auth_db.py' manually"
                )

            logger.info("[OK] Authentication module initialized")
        except Exception as e:
            logger.error(f"Failed to initialize authentication module: {e}")
            logger.error("Auth features will not be available")
    else:
        logger.info("Authentication module disabled (set ENABLE_AUTH=true to enable)")

    # Initialize notification service
    try:
        notification_service = NotificationService(config_manager=config_manager)
        # Set global instance for router
        from Module4_NiruAPI.routers import notification_router as nr_module

        nr_module.notification_service = notification_service
        nr_module.news_service = None  # Will be lazy-loaded

        # Create notification callback function for database storage
        def notification_callback(article: Dict):
            """Callback function to send notifications for new articles"""
            try:
                if notification_service:
                    notification_service.send_article_notification(article)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")

        # Make callback available globally for database storage
        import Module3_NiruDB.database_storage as db_storage_module

        db_storage_module.default_notification_callback = notification_callback

        logger.info("Notification service initialized")

        # Start background task for daily digest
        def daily_digest_worker():
            """Background worker for daily digest notifications"""
            import time

            while True:
                try:
                    time.sleep(3600)  # Check every hour
                    current_hour = datetime.utcnow().hour
                    if current_hour == 8:  # Send at 8 AM UTC
                        notification_service.send_digest_notifications()
                except Exception as e:
                    logger.error(f"Error in daily digest worker: {e}")

        digest_thread = threading.Thread(target=daily_digest_worker, daemon=True)
        digest_thread.start()
        logger.info("Daily digest background worker started")

    except Exception as e:
        logger.warning(f"Notification service not available: {e}")
        notification_service = None

    # Initialize autocomplete tool
    try:
        autocomplete_tool = AutocompleteTool()
        logger.info("Autocomplete tool initialized")
    except Exception as e:
        logger.warning(f"Autocomplete tool not available: {e}")
        autocomplete_tool = None

    # Initialize Vision RAG service
    vision_storage = {}  # In-memory storage: {session_id: [image_data, ...]}
    try:
        from Module4_NiruAPI.services.vision_rag import VisionRAGService

        vision_rag_service = VisionRAGService()
        logger.info("Vision RAG service initialized")
    except Exception as e:
        logger.warning(
            f"Vision RAG service not available: {e}. Vision RAG features will be disabled."
        )
        vision_rag_service = None

    # ============================================================
    # Initialize Amaniq v2 Agent (REQUIRED - The Brain of the System)
    # ============================================================
    logger.info("=" * 80)
    logger.info("INITIALIZING AMANIQ V2 AGENT (SYSTEM BRAIN)")
    logger.info("=" * 80)

    try:
        logger.info("Step 1: Importing AmaniQ v2 modules...")
        from Module4_NiruAPI.agents.amaniq_v2 import AmaniQAgent, AmaniQConfig

        logger.info("[OK] Import successful")

        logger.info("Step 2: Validating dependencies...")
        if not vector_store:
            missing_deps = []
            missing_deps.append(
                "Vector store (check QDRANT_URL, QDRANT_API_KEY or UPSTASH_VECTOR_URL, UPSTASH_VECTOR_TOKEN)"
            )
            logger.error("=" * 80)
            logger.error("[ERROR] VECTOR STORE NOT AVAILABLE")
            logger.error("=" * 80)
            logger.error(
                "The vector store failed to initialize. This could be because:"
            )
            logger.error(
                "  1. Cloud vector store (Qdrant/Upstash) is unavailable or misconfigured"
            )
            logger.error("  2. ChromaDB local fallback also failed")
            logger.error("")
            logger.error("Required environment variables (at least one set):")
            logger.error("  - QDRANT_URL and QDRANT_API_KEY (for Qdrant Cloud)")
            logger.error(
                "  - UPSTASH_VECTOR_URL and UPSTASH_VECTOR_TOKEN (for Upstash)"
            )
            logger.error("  - Or ensure ChromaDB can be initialized locally")
            logger.error("=" * 80)
            raise RuntimeError(
                f"Vector store is required for AmaniQ v2 agent. Missing: {', '.join(missing_deps)}"
            )
        logger.info("  [OK] Vector store available")

        if not rag_pipeline:
            raise RuntimeError(
                "RAG pipeline is required for AmaniQ v2 agent but was not initialized"
            )
        logger.info("  [OK] RAG pipeline available")

        if not rag_pipeline.llm_service:
            raise RuntimeError(
                "LLM service is required for AmaniQ v2 agent but was not initialized (check MOONSHOT_API_KEY)"
            )
        logger.info("  [OK] LLM service available")

        logger.info("Step 3: Creating agent configuration...")
        # Create config for the agent
        agent_config = AmaniQConfig(
            enable_caching=cache_manager is not None,
            enable_prefetch=True,
            enable_telemetry=False,
            enable_persistence=False,  # Disable persistence for faster startup
        )
        logger.info(f"  [OK] Config created (caching={cache_manager is not None})")

        logger.info("Step 4: Creating AmaniQAgent instance...")
        amaniq_v2_agent = AmaniQAgent(config=agent_config)
        logger.info("  [OK] Instance created")

        logger.info("Step 5: Initializing agent (building graph, etc.)...")
        await amaniq_v2_agent.initialize()
        logger.info("  [OK] Initialization complete")

        logger.info("Step 6: Verifying agent state...")
        if amaniq_v2_agent is None:
            raise RuntimeError("Agent instance is None after initialization!")
        if not amaniq_v2_agent._initialized:
            raise RuntimeError("Agent._initialized is False after initialization!")
        if amaniq_v2_agent.graph is None:
            raise RuntimeError("Agent.graph is None after initialization!")
        logger.info(
            f"  [OK] Agent verified (graph type: {type(amaniq_v2_agent.graph).__name__})"
        )

        logger.info("=" * 80)
        logger.info("[OK] AMANIQ V2 AGENT INITIALIZED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error("[ERROR] CRITICAL ERROR: AMANIQ V2 AGENT INITIALIZATION FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        logger.error("The API cannot start without the AmaniQ v2 agent (system brain)")
        logger.error("=" * 80)
        raise RuntimeError(f"Failed to initialize required AmaniQ v2 agent: {e}") from e

    # ============================================================
    # Initialize NiruSense Processing Pipeline (Optional)
    # ============================================================
    enable_nirusense = os.getenv("ENABLE_NIRUSENSE", "false").lower() == "true"
    if enable_nirusense:
        try:
            logger.info("Initializing NiruSense processing pipeline...")
            from Module9_NiruSense.processing.config import Settings
            from Module9_NiruSense.processing.health import HealthChecker
            from Module9_NiruSense.processing.monitoring import MetricsCollector
            from Module9_NiruSense.scheduler import NiruSenseScheduler

            nirusense_settings = Settings()
            nirusense_health_checker = HealthChecker()
            nirusense_metrics = MetricsCollector()

            # Start NiruSense orchestrator in background thread
            from Module9_NiruSense.nirusense_service import start_nirusense_thread

            nirusense_orchestrator_thread = threading.Thread(
                target=start_nirusense_thread, daemon=True, name="NiruSenseOrchestrator"
            )
            nirusense_orchestrator_thread.start()
            logger.info("[OK] NiruSense orchestrator thread started")

            # Start scheduler if configured
            nirusense_scheduler = NiruSenseScheduler()
            if nirusense_scheduler.start():
                logger.info("[OK] NiruSense scheduler started")

            # Initialize the pipeline reference (lazy - orchestrator handles actual processing)
            from Module9_NiruSense.processing.orchestrator import (
                pipeline as ns_pipeline,
            )

            nirusense_pipeline = ns_pipeline

            logger.info("[OK] NiruSense processing pipeline initialized")
        except Exception as e:
            logger.warning(f"NiruSense pipeline not available: {e}")
            nirusense_pipeline = None
            nirusense_health_checker = None
            nirusense_metrics = None
            nirusense_scheduler = None
    else:
        logger.info("NiruSense disabled (set ENABLE_NIRUSENSE=true to enable)")

    # Inject dependencies into routers
    _inject_router_dependencies()

    logger.info("[OK] AmaniQuery API startup complete")

    # Yield control to FastAPI
    yield

    # Shutdown cleanup
    logger.info("Shutting down AmaniQuery API")

    # Stop NiruSense scheduler
    if nirusense_scheduler is not None:
        try:
            nirusense_scheduler.stop()
            logger.info("[OK] NiruSense scheduler stopped")
        except Exception as e:
            logger.warning(f"Failed to stop NiruSense scheduler: {e}")

    logger.info("AmaniQuery API shutdown complete")


def _inject_router_dependencies():
    """Inject service dependencies into routers"""
    # Import router modules using sys.modules to avoid naming conflicts
    import sys

    query_router_module = sys.modules["Module4_NiruAPI.routers.query_router"]
    chat_router_module = sys.modules["Module4_NiruAPI.routers.chat_router"]
    admin_router_module = sys.modules["Module4_NiruAPI.routers.admin_router"]
    research_router_module = sys.modules["Module4_NiruAPI.routers.research_router"]
    sms_router_module = sys.modules["Module4_NiruAPI.routers.sms_router"]
    alignment_router_module = sys.modules["Module4_NiruAPI.routers.alignment_router"]
    monitoring_router_module = sys.modules["Module4_NiruAPI.routers.monitoring_router"]
    hybrid_rag_router_module = sys.modules["Module4_NiruAPI.routers.hybrid_rag_router"]

    # Set dependencies on query router using state container
    query_router_module._state.vector_store = vector_store
    query_router_module._state.rag_pipeline = rag_pipeline
    query_router_module._state.cache_manager = cache_manager
    query_router_module._state.amaniq_v2_agent = amaniq_v2_agent
    query_router_module._state.database_storage = database_storage
    query_router_module._state.chat_manager = chat_manager
    query_router_module._state.vision_rag_service = vision_rag_service
    query_router_module._state.vision_storage = vision_storage

    # Set dependencies on chat router using state container
    chat_router_module._state.chat_manager = chat_manager
    chat_router_module._state.vision_storage = vision_storage
    chat_router_module._state.vision_rag_service = vision_rag_service
    chat_router_module._state.rag_pipeline = rag_pipeline
    chat_router_module._state.vector_store = vector_store
    chat_router_module._state.amaniq_v2_agent = (
        amaniq_v2_agent  # Inject the full agent, not just the graph
    )
    chat_router_module._state.amaniq_v2_graph = (
        amaniq_v2_agent.graph if amaniq_v2_agent else None
    )
    chat_router_module._state.research_bundle_service = research_bundle_service

    # Verify critical dependencies
    if amaniq_v2_agent is None:
        logger.error("CRITICAL: amaniq_v2_agent is None during dependency injection!")
    elif amaniq_v2_agent.graph is None:
        logger.error(
            "CRITICAL: amaniq_v2_agent.graph is None during dependency injection!"
        )
    else:
        logger.info(
            f"[OK] AmaniQ v2 graph injected into chat_router (type={type(amaniq_v2_agent.graph).__name__})"
        )

    if chat_manager is None:
        logger.warning("chat_manager is None during dependency injection")

    # Set dependencies on admin router
    logger.info(
        f"Injecting dependencies into admin_router. crawler_manager is {'None' if crawler_manager is None else 'Set'}"
    )
    admin_router_module.crawler_manager = crawler_manager
    admin_router_module.vector_store = vector_store
    admin_router_module.config_manager = config_manager
    admin_router_module.database_storage = database_storage
    admin_router_module.cache_manager = cache_manager

    # Set dependencies on research router using state container
    research_router_module._state.agentic_research_module = agentic_research_module
    research_router_module._state.research_module = research_module
    research_router_module._state.report_generator = report_generator
    research_router_module._state.cache_manager = cache_manager
    research_router_module._state.chat_manager = chat_manager
    research_router_module._state.research_bundle_service = research_bundle_service

    # Set dependencies on SMS router
    sms_router_module.sms_pipeline = sms_pipeline
    sms_router_module.sms_service = sms_service

    # Set dependencies on alignment router
    alignment_router_module.alignment_pipeline = alignment_pipeline
    alignment_router_module.rag_pipeline = rag_pipeline
    alignment_router_module.cache_manager = cache_manager

    # Set dependencies on hybrid RAG router
    hybrid_rag_router_module.hybrid_rag_pipeline = hybrid_rag_pipeline
    hybrid_rag_router_module.rag_pipeline = rag_pipeline
    hybrid_rag_router_module.cache_manager = cache_manager
    hybrid_rag_router_module.chat_manager = chat_manager

    # Set dependencies on monitoring router
    if database_storage:
        monitoring_router_module.db_session_factory = database_storage.SessionLocal

    # Set dependencies on NiruSense router
    try:
        nirusense_router_module = sys.modules[
            "Module4_NiruAPI.routers.nirusense_router"
        ]
        nirusense_router_module._state.health_checker = nirusense_health_checker
        nirusense_router_module._state.pipeline = nirusense_pipeline
        nirusense_router_module._state.metrics = nirusense_metrics
        nirusense_router_module._state.scheduler = nirusense_scheduler
        nirusense_router_module._state.settings = nirusense_settings
        # Actually check if thread is alive
        nirusense_router_module._state.orchestrator_running = (
            nirusense_orchestrator_thread is not None
            and nirusense_orchestrator_thread.is_alive()
        )
        logger.info("NiruSense dependencies injected")
    except Exception as e:
        logger.warning(f"Failed to inject NiruSense dependencies: {e}")

    logger.info("Router dependencies injected")


# ============================================================
# FastAPI App Initialization
# ============================================================
app = FastAPI(
    title="AmaniQuery API",
    description="RAG-powered API for Kenyan legal, parliamentary, and news intelligence",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS
cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://localhost:8000,http://frontend:3000,"
    "https://amaniquery.vercel.app,https://www.amaniquery.vercel.app,https://api-amaniquery.onrender.com",
)
origins = [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add authentication middleware (optional - can be enabled via env var)
if os.getenv("ENABLE_AUTH", "false").lower() == "true":
    from Module8_NiruAuth.middleware.auth_middleware import AuthMiddleware
    from Module8_NiruAuth.middleware.rate_limit_middleware import RateLimitMiddleware
    from Module8_NiruAuth.middleware.usage_tracking_middleware import (
        UsageTrackingMiddleware,
    )

    # Add middleware in order (FastAPI executes in reverse order)
    app.add_middleware(UsageTrackingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)  # This executes first

    # Include auth routers
    from Module8_NiruAuth.routers import (
        user_router,
        admin_router as auth_admin_router,
        integration_router,
        api_key_router,
        oauth_router,
        analytics_router,
        blog_router,
        session_router,
        mfa_router,
    )
    from Module8_NiruAuth.routers.phone_verification_router import (
        router as phone_verification_router,
    )

    app.include_router(user_router)
    app.include_router(auth_admin_router)
    app.include_router(integration_router)
    app.include_router(api_key_router)
    app.include_router(oauth_router)
    app.include_router(analytics_router)
    app.include_router(blog_router)
    app.include_router(phone_verification_router)
    app.include_router(session_router)
    app.include_router(mfa_router)

# ============================================================
# Include Routers
# ============================================================
# External module routers
app.include_router(share_router)
app.include_router(news_router)
app.include_router(websocket_router)
app.include_router(notification_router)

# Internal refactored routers
app.include_router(query_router)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(research_router)
app.include_router(sms_router)
app.include_router(alignment_router)
app.include_router(monitoring_router)
app.include_router(hybrid_rag_router)
app.include_router(nirusense_router)
app.include_router(clustering_router)
app.include_router(finetuning_router)
app.include_router(feedback_analytics_router)
app.include_router(voice_router)
app.include_router(media_router)
app.include_router(admin_settings_router)


# ============================================================
#     Endpoints
# ============================================================
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "name": "AmaniQuery API",
        "version": "2.0.0",
        "description": "RAG-powered API for Kenyan intelligence with Constitutional Alignment Analysis",
        "agent": "AmaniQ v2",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /health",
            "stats": "GET /stats",
            "docs": "GET /docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint - cached for 30 seconds"""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Try to get from cache
    cache_key = "cache:health"
    if cache_manager:
        cached_result = await cache_manager.get(cache_key)
        if cached_result is not None:
            logger.debug("Cache hit for health check")
            return HealthResponse(**cached_result)

    stats = vector_store.get_stats()

    # Handle case where total_chunks might be "unknown" string
    total_chunks = stats.get("total_chunks", 0)
    if isinstance(total_chunks, str) and total_chunks == "unknown":
        total_chunks = 0
    elif not isinstance(total_chunks, int):
        total_chunks = 0

    result = HealthResponse(
        status="healthy",
        database_chunks=total_chunks,
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        llm_provider=os.getenv("LLM_PROVIDER", "moonshot"),
    )

    # Cache for 30 seconds
    if cache_manager:
        await cache_manager.set(cache_key, result.model_dump(), ttl=30)

    return result


@app.get("/debug/files", tags=["General"])
async def list_data_files():
    """List files in data directory for debugging"""
    try:
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / "data"
        chroma_dir = data_dir / "chroma_db"

        result = {
            "base_dir": str(base_dir),
            "data_dir_exists": data_dir.exists(),
            "chroma_dir_exists": chroma_dir.exists(),
            "files": [],
        }

        if data_dir.exists():
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(base_dir)
                    size = file_path.stat().st_size
                    result["files"].append(
                        {
                            "path": str(rel_path),
                            "size": size,
                            "size_mb": round(size / (1024 * 1024), 2),
                        }
                    )

        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/autocomplete", tags=["Query"])
async def get_autocomplete(
    q: str, max_results: int = 10, location: Optional[str] = None, language: str = "en"
):
    """
    Get Google autocomplete suggestions

    Args:
        q: Partial search query
        max_results: Maximum number of suggestions (default: 10)
        location: Location for localized suggestions (e.g., "Kenya")
        language: Language code (e.g., "en", "sw")

    Returns:
        Autocomplete suggestions
    """
    if autocomplete_tool is None:
        raise HTTPException(
            status_code=503, detail="Autocomplete service not available"
        )

    try:
        result = autocomplete_tool.execute(
            query=q, max_results=max_results, location=location, language=language
        )
        return result
    except Exception as e:
        logger.error(f"Error in autocomplete: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/news/health", tags=["News"])
async def news_crawler_health():
    """Health check for news crawler system"""
    try:
        from Module1_NiruSpider.niruspider.monitoring import CrawlerMonitor

        monitor = CrawlerMonitor()
        health = monitor.get_health_status()
        return health
    except Exception as e:
        logger.error(f"Error getting crawler health: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@app.get("/api/v1/news/sources/status", tags=["News"])
async def news_sources_status(days: int = 7):
    """Get status of news sources"""
    try:
        from Module1_NiruSpider.niruspider.monitoring import CrawlerMonitor

        monitor = CrawlerMonitor()
        status = monitor.get_source_status(days=days)
        return status
    except Exception as e:
        logger.error(f"Error getting source status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/news/stats", tags=["News"])
async def news_crawler_stats():
    """Get crawler statistics"""
    try:
        from Module1_NiruSpider.niruspider.monitoring import CrawlerMonitor

        monitor = CrawlerMonitor()
        stats = monitor.get_crawler_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting crawler stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse, tags=["General"])
async def get_stats():
    """Get database statistics - cached for 60 seconds"""
    try:
        if vector_store is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        # Try to get from cache
        cache_key = "cache:stats"
        if cache_manager:
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug("Cache hit for stats")
                return StatsResponse(**cached_result)

        # Run blocking operations in thread pool
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, vector_store.get_stats)

        # Ensure stats is a dict and has expected keys
        if not isinstance(stats, dict):
            logger.warning(
                f"vector_store.get_stats() returned non-dict type: {type(stats)}"
            )
            stats = {"sample_categories": {}, "total_chunks": 0}

        if "sample_categories" not in stats:
            stats["sample_categories"] = {}
        if "total_chunks" not in stats:
            stats["total_chunks"] = 0

        # Get categories and sources
        try:

            def get_metadata():
                meta_manager = MetadataManager(vector_store)
                categories_list = meta_manager.get_categories()
                sources_list = meta_manager.get_sources()
                return categories_list, sources_list

            categories_list, sources_list = await loop.run_in_executor(
                None, get_metadata
            )
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
            categories_list = ["Unknown"]
            sources_list = ["Unknown"]

        # Convert to dict with counts
        sample_categories = stats.get("sample_categories", {})
        categories_dict = {
            cat: sample_categories.get(cat, 0) for cat in categories_list
        }

        # Handle case where total_chunks might be "unknown" string
        total_chunks = stats["total_chunks"]
        if isinstance(total_chunks, str) and total_chunks == "unknown":
            total_chunks = 0
        elif not isinstance(total_chunks, int):
            total_chunks = 0

        result = StatsResponse(
            total_chunks=total_chunks,
            categories=categories_dict,
            sources=sources_list,
        )

        # Cache for 60 seconds
        if cache_manager:
            await cache_manager.set(cache_key, result.model_dump(), ttl=60)

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        # Return default stats to prevent fetch failure
        return StatsResponse(
            total_chunks=0,
            categories={},
            sources=["Unknown"],
        )


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    import platform

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    # Disable reload on Windows to avoid multiprocessing import issues
    is_windows = platform.system() == "Windows"
    default_reload = "False" if is_windows else "True"
    reload = os.getenv("API_RELOAD", default_reload).lower() == "true"

    print("=" * 60)
    print("🚀 Starting AmaniQuery API v2.0")
    print("=" * 60)
    print(f"📍 Server: http://{host}:{port}")
    print(f"📚 Docs: http://{host}:{port}/docs")
    print(f"🔧 Provider: {os.getenv('LLM_PROVIDER', 'moonshot')}")
    print(f"🤖 Agent: AmaniQ v2 (LangGraph)")
    print(f"🔄 Reload: {'Enabled' if reload else 'Disabled'}")
    if is_windows and reload:
        print("⚠️  Warning: Reload enabled on Windows may cause import issues")
    print("=" * 60)

    # Exclude setup.py and other non-source files from reload watch
    reload_excludes = (
        [
            "setup.py",
            "*.pyc",
            "__pycache__",
            "*.log",
            ".env",
            "venv/**",
            "node_modules/**",
        ]
        if reload
        else None
    )

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        reload_excludes=reload_excludes,
    )
