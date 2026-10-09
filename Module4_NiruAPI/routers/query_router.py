"""
Query Router - Main query endpoints for AmaniQuery
Uses AmanIQ v2 agent orchestration for intelligent query processing
"""

import os
import json
import asyncio
from typing import Optional, Dict, List, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

# User profile store for persistent personalization
try:
    from ..services.user_profile_store import UserProfileStore, get_profile_store

    PROFILE_STORE_AVAILABLE = True
except ImportError:
    PROFILE_STORE_AVAILABLE = False
    logger.warning("UserProfileStore not available")

# Aggressive semantic cache
try:
    from ..services.aggressive_cache import (
        get_aggressive_cache,
        AggressiveSemanticCache,
    )

    AGGRESSIVE_CACHE_AVAILABLE = True
except ImportError:
    AGGRESSIVE_CACHE_AVAILABLE = False
    logger.warning("AggressiveSemanticCache not available")


router = APIRouter(prefix="/api/v1", tags=["Query"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class QueryRequest(BaseModel):
    """Query request model"""

    query: str
    top_k: int = 5
    category: Optional[str] = None
    source: Optional[str] = None
    include_sources: bool = True
    temperature: float = 0.7
    max_tokens: int = 2000
    session_id: Optional[str] = None
    user_id: Optional[str] = None  # Added for user profiling


class Source(BaseModel):
    """Source model"""

    title: str
    url: str
    source_name: str
    category: str
    excerpt: Optional[str] = None


class QueryResponse(BaseModel):
    """Query response model"""

    answer: str
    sources: List[Source] = []
    query_time: float
    retrieved_chunks: int
    model_used: str
    structured_data: Optional[Dict[str, Any]] = None


# =============================================================================
# DEPENDENCIES - State container to avoid global variable issues
# =============================================================================


class QueryRouterState:
    """State container for query router dependencies"""

    vector_store = None
    rag_pipeline = None
    cache_manager = None
    amaniq_v2_agent = None
    database_storage = None
    chat_manager = None
    vision_rag_service = None
    vision_storage = None
    user_profile_store: Optional[UserProfileStore] = None  # Added for profiles


_state = QueryRouterState()

# Aggressive semantic cache instance (initialized once)
_aggressive_cache: Optional[AggressiveSemanticCache] = None


def get_aggressive_cache_instance() -> Optional[AggressiveSemanticCache]:
    global _aggressive_cache
    if _aggressive_cache is None and AGGRESSIVE_CACHE_AVAILABLE:
        try:
            _aggressive_cache = get_aggressive_cache()
        except Exception as e:
            logger.warning(f"Failed to init aggressive cache: {e}")
    return _aggressive_cache


def get_rag_pipeline():
    """Get the RAG pipeline instance"""
    if _state.rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return _state.rag_pipeline


def get_amaniq_v2_agent():
    """Get the AmanIQ v2 agent instance"""
    if _state.amaniq_v2_agent is None:
        raise HTTPException(status_code=503, detail="AmanIQ v2 agent not initialized")
    return _state.amaniq_v2_agent


def get_cache_manager():
    """Get the cache manager instance"""
    return _state.cache_manager


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def save_query_to_chat(session_id: str, query: str, result: Dict, role: str = "user"):
    """Helper function to save query and response to chat database"""
    if _state.chat_manager is None or not session_id:
        return

    try:
        # Validate session exists
        session = _state.chat_manager.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found, skipping message save")
            return

        # Save user message
        _state.chat_manager.add_message(
            session_id=session_id, content=query, role="user"
        )

        # Save assistant response
        _state.chat_manager.add_message(
            session_id=session_id,
            content=result.get("answer", ""),
            role="assistant",
            token_count=result.get("retrieved_chunks", 0),
            model_used=result.get("model_used", "unknown"),
            sources=result.get("sources", []),
        )

        # Generate session title if needed
        if not session.title:
            title = _state.chat_manager.generate_session_title(session_id)
            _state.chat_manager.update_session_title(session_id, title)

        logger.debug(f"Saved query to chat session {session_id}")
    except Exception as e:
        logger.warning(f"Failed to save query to chat: {e}")


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main query endpoint - Ask questions about Kenyan law, parliament, and news

    **Uses AmanIQ v2 local agents when available for:**
    - Intent classification (wanjiku/wakili/mwanahabari)
    - Multi-step reasoning with tool orchestration
    - Clarification handling for ambiguous queries
    - Persona-optimized retrieval
    - JSON-enforced structured responses
    - Self-correcting validation

    **Example queries:**
    - "What does the Kenyan Constitution say about freedom of speech?"
    - "What are the recent parliamentary debates on finance?"
    - "Latest news on AI policy in Kenya"
    - "Kanjo wameongeza parking fees aje?" (Sheng)

    **Vision RAG:** If session has uploaded images/PDFs, automatically uses Vision RAG.
    """
    try:
        # Check if session has vision data and use Vision RAG if available
        use_vision_rag = False
        session_images = []
        if request.session_id and _state.vision_rag_service and _state.vision_storage:
            session_images = _state.vision_storage.get(request.session_id, [])
            if session_images:
                use_vision_rag = True
                logger.info(
                    f"Using Vision RAG for session {request.session_id} with {len(session_images)} image(s)"
                )

        # Define computation function for caching
        async def compute_response():
            if use_vision_rag:
                # Use Vision RAG
                result = _state.vision_rag_service.query(
                    question=request.query,
                    session_images=session_images,
                    top_k=min(request.top_k, 3),
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )

                # Convert vision sources to Source format
                sources = []
                for src in result.get("sources", []):
                    sources.append(
                        {
                            "title": src.get("filename", "Image"),
                            "url": "",
                            "source_name": src.get("source_file", "Uploaded Image"),
                            "category": "vision",
                            "excerpt": f"Image similarity: {src.get('similarity', 0):.2f}",
                        }
                    )

                result["sources"] = sources
                return result

            # Use AmaniQ v2 agent for all non-vision queries (REQUIRED)
            logger.info("[AmaniQ v2] Using agent orchestration (System Brain)")

            try:
                # Get conversation history if session exists
                conversation_history = []
                if request.session_id and _state.chat_manager:
                    try:
                        messages = _state.chat_manager.get_messages(
                            request.session_id, limit=10
                        )
                        conversation_history = [
                            {"role": msg.role, "content": msg.content}
                            for msg in messages
                        ]
                    except Exception:
                        conversation_history = []

                # Load or create user profile for personalization
                user_profile = None
                user_id = request.user_id or request.session_id
                if user_id and PROFILE_STORE_AVAILABLE:
                    try:
                        if _state.user_profile_store is None:
                            _state.user_profile_store = get_profile_store()

                        profile = await _state.user_profile_store.get_profile(user_id)
                        user_profile = profile.to_dict()
                        logger.debug(
                            f"Loaded user profile for {user_id}: queries={profile.total_queries}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load user profile: {e}")

                # Execute AmaniQ v2 pipeline (THE BRAIN)
                amaniq_result = await _state.amaniq_v2_agent.chat(
                    message=request.query,
                    thread_id=request.session_id,
                    message_history=conversation_history,
                    user_profile=user_profile,  # Pass profile for personalization
                )

                # Track query type for user profiling
                if user_id and PROFILE_STORE_AVAILABLE and _state.user_profile_store:
                    try:
                        intent = amaniq_result.get("intent", "GENERAL_CHAT")
                        await _state.user_profile_store.track_query(user_id, intent)
                    except Exception as e:
                        logger.warning(f"Failed to track query: {e}")

                # Extract response data
                answer = amaniq_result.get("answer", "")
                confidence = amaniq_result.get("confidence", 0.0)
                sources_data = amaniq_result.get("sources", [])

                # Format sources
                sources = []
                for src in sources_data:
                    sources.append(
                        {
                            "title": src.get("title", "Source"),
                            "url": src.get("url", ""),
                            "source_name": src.get("source_type", "Unknown"),
                            "category": src.get("source_type", "general"),
                            "excerpt": (
                                src.get("content", "")[:200]
                                if src.get("content")
                                else ""
                            ),
                        }
                    )

                result = {
                    "answer": answer,
                    "sources": sources,
                    "query_time": amaniq_result.get("latency_ms", 0) / 1000,
                    "retrieved_chunks": len(sources_data),
                    "model_used": f"AmaniQ-v2-{amaniq_result.get('persona', 'wanjiku')}",
                    "structured_data": {
                        "confidence": confidence,
                        "persona": amaniq_result.get("persona"),
                        "intent": amaniq_result.get("intent"),
                        "reasoning_steps": amaniq_result.get("reasoning_steps", 0),
                    },
                }

                logger.info(
                    f"[AmaniQ v2] Query completed with confidence {confidence:.2f}"
                )
                return result

            except Exception as e:
                # ONLY on error: Fall back to standard RAG pipeline
                logger.error(f"[AmaniQ v2] CRITICAL ERROR: {e}")
                logger.warning("[RAG] Emergency fallback to standard RAG pipeline")
                result = get_rag_pipeline().query(
                    query=request.query,
                    top_k=request.top_k,
                    category=request.category,
                    source=request.source,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    session_id=request.session_id,
                )
                return result

        # Aggressive semantic caching — check before compute, write after
        agg_cache = get_aggressive_cache_instance() if not use_vision_rag else None

        # Step 1: Try aggressive semantic cache first (normalized + semantic matching)
        if agg_cache:
            cached = agg_cache.get(request.query)
            if cached is not None:
                logger.info(f"Aggressive cache HIT for: {request.query[:60]}...")
                result = cached
            else:
                # Step 2: Compute
                result = await compute_response()
                # Auto-write every response
                if result.get("answer"):
                    agg_cache.set(request.query, result)
                    logger.debug(f"Aggressive cache WRITE for: {request.query[:60]}...")
        else:
            # Fallback: use existing CacheManager with get_or_compute
            if _state.cache_manager and not use_vision_rag:
                ttl_type = "default"
                q_lower = request.query.lower()
                if "finance bill" in q_lower or "tax" in q_lower:
                    ttl_type = "trending"
                elif "constitution" in q_lower or "act" in q_lower or "law" in q_lower:
                    ttl_type = "wakili"
                elif "news" in q_lower or "update" in q_lower:
                    ttl_type = "mwanahabari"
                elif "how to" in q_lower or "calculate" in q_lower:
                    ttl_type = "widget"
                result = await _state.cache_manager.get_or_compute(
                    request.query, compute_response, ttl_type
                )
            else:
                result = await compute_response()

        # Save to chat if session_id provided
        if request.session_id:
            save_query_to_chat(request.session_id, request.query, result)

        # Format sources for response
        sources_data = result.get("sources", [])
        sources = [Source(**src) for src in sources_data]

        return QueryResponse(
            answer=result["answer"],
            sources=sources if request.include_sources else [],
            query_time=result.get("query_time", 0),
            retrieved_chunks=result.get("retrieved_chunks", 0),
            model_used=result.get("model_used", "unknown"),
            structured_data=result.get("structured_data"),
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Main query endpoint with streaming response - Fastest perceived speed

    **Streaming Benefits:**
    - Time to first token: <1 second (vs 5-10 seconds)
    - User sees response immediately as it's generated
    - Best for user experience
    """
    try:
        # Run RAG query with streaming
        result = get_rag_pipeline().query_stream(
            query=request.query,
            top_k=request.top_k,
            category=request.category,
            source=request.source,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            session_id=request.session_id,
        )

        if not result.get("stream", False):
            # Fallback to regular response
            if request.session_id:
                save_query_to_chat(request.session_id, request.query, result)

            sources = [Source(**src) for src in result["sources"]]
            return QueryResponse(
                answer=result["answer"],
                sources=sources if request.include_sources else [],
                query_time=result["query_time"],
                retrieved_chunks=result["retrieved_chunks"],
                model_used=result["model_used"],
            )

        # Return streaming response
        async def generate():
            full_answer = ""
            try:
                answer_stream = result["answer_stream"]
                rag_pipeline = get_rag_pipeline()

                if rag_pipeline.llm_provider in ["openai", "moonshot"]:
                    # OpenAI-style streaming
                    async for chunk in answer_stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_answer += content
                            yield f"data: {content}\n\n"

                elif rag_pipeline.llm_provider == "anthropic":
                    # Anthropic streaming
                    async for chunk in answer_stream:
                        if chunk.type == "content_block_delta" and chunk.delta.text:
                            text = chunk.delta.text
                            full_answer += text
                            yield f"data: {text}\n\n"

                # Send sources at the end
                sources_data = {
                    "sources": (
                        [Source(**src).model_dump() for src in result["sources"]]
                        if request.include_sources
                        else []
                    ),
                    "query_time": result["query_time"],
                    "retrieved_chunks": result["retrieved_chunks"],
                    "model_used": result["model_used"],
                }
                yield f"data: [DONE]{json.dumps(sources_data)}\n\n"

                # Save to chat if session_id provided
                if request.session_id and full_answer:
                    result["answer"] = full_answer
                    save_query_to_chat(request.session_id, request.query, result)

            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                yield f"data: [ERROR]{str(e)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        logger.error(f"Error processing streaming query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
