"""
Chat Router - Chat session and message endpoints for AmaniQuery
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path
from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Depends,
    File,
    UploadFile,
    BackgroundTasks,
)
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

# Lazy import for tool schema (avoids circular imports at module level)
_tool_schema_loaded = False
_registry_openai_tools = []
_registry_tool_schemas = []

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class ChatSessionCreate(BaseModel):
    """Create chat session request"""

    title: Optional[str] = None
    user_id: Optional[str] = None


class ChatSessionResponse(BaseModel):
    """Chat session response"""

    id: str
    title: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    message_count: int = 0


class ChatMessageCreate(BaseModel):
    """Create chat message request"""

    content: str
    role: str = "user"
    stream: bool = False
    attachment_ids: Optional[List[str]] = None


class ChatMessageResponse(BaseModel):
    """Chat message response"""

    id: str
    session_id: str
    content: str
    role: str
    created_at: datetime
    token_count: Optional[int] = None
    model_used: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    attachments: Optional[List[Dict[str, Any]]] = None


class CompletionsCreate(BaseModel):
    """Chat completions request (used by frontend sendMessage)"""

    session_id: str
    message: str
    use_hybrid: bool = False
    is_research: bool = False
    attachments: Optional[List[Dict[str, Any]]] = None


class FeedbackCreate(BaseModel):
    """Create feedback request"""

    message_id: str
    feedback_type: str  # "positive", "negative"
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Feedback response"""

    id: str
    message_id: str
    feedback_type: str
    comment: Optional[str] = None
    created_at: datetime


# =============================================================================
# DEPENDENCIES - State container to avoid global variable issues
# =============================================================================


class RouterState:
    """State container for router dependencies to avoid Python global variable issues"""

    chat_manager = None
    vision_storage = {}
    vision_rag_service = None
    rag_pipeline = None
    vector_store = None
    amaniq_v2_agent = None
    amaniq_v2_graph = None
    research_bundle_service = None


# Single global instance
_state = RouterState()


def get_rag_pipeline():
    """Get RAG pipeline with lazy initialization fallback"""
    if _state.rag_pipeline is None:
        logger.warning(
            "RAG pipeline not initialized via dependency injection, attempting lazy initialization"
        )
        try:
            from Module4_NiruAPI.rag_pipeline import RAGPipeline

            _state.rag_pipeline = RAGPipeline()
            logger.info("RAG pipeline lazily initialized successfully")
        except Exception as e:
            logger.error(f"Failed to lazily initialize RAG pipeline: {e}")
            raise HTTPException(
                status_code=503, detail=f"RAG service not initialized: {e}"
            )
    return _state.rag_pipeline


def get_chat_manager():
    """Get the chat manager instance"""
    if _state.chat_manager is None:
        logger.warning(
            "Chat manager not initialized via dependency injection, attempting lazy initialization"
        )
        try:
            from Module3_NiruDB.chat_manager import ChatDatabaseManager

            _state.chat_manager = ChatDatabaseManager()
            logger.info("Chat manager lazily initialized successfully")
        except Exception as e:
            logger.error(f"Failed to lazily initialize chat manager: {e}")
            raise HTTPException(
                status_code=503, detail=f"Chat service not initialized: {e}"
            )
    return _state.chat_manager


def get_amaniq_v2_graph():
    """Get the AmaniQ v2 compiled graph directly (avoids global model issues)"""
    if _state.amaniq_v2_graph is None:
        logger.error("CRITICAL: AmaniQ v2 graph is None - this should never happen!")
        logger.error("The graph should have been initialized during API startup.")
        logger.error("Check API startup logs for initialization errors.")

        raise HTTPException(
            status_code=503,
            detail="AmaniQ v2 graph not initialized. This is a critical system error. Please contact support.",
        )

    return _state.amaniq_v2_graph


# =============================================================================
# TOOL SCHEMA ACCESS
# =============================================================================


def _load_tool_schemas():
    """Lazy-load tool schemas from the global ToolRegistry."""
    global _tool_schema_loaded, _registry_openai_tools, _registry_tool_schemas
    if _tool_schema_loaded:
        return

    try:
        from Module4_NiruAPI.agents.tools.tool_schema import (
            get_registry_openai_tools,
            get_registry_schemas,
        )
        from Module4_NiruAPI.agents.tools.tool_registry import ToolRegistry

        registry = ToolRegistry()
        _registry_openai_tools = get_registry_openai_tools(registry)
        _registry_tool_schemas = get_registry_schemas(registry)
        _tool_schema_loaded = True
        logger.info(f"Loaded {len(_registry_openai_tools)} tool schemas for API")
    except Exception as e:
        logger.warning(f"Failed to load tool schemas: {e}")


def get_openai_tools() -> List[Dict[str, Any]]:
    """Get all tools in OpenAI-compatible format."""
    _load_tool_schemas()
    return _registry_openai_tools


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_tool_description(tool_name: str) -> str:
    """Get a human-readable description for a tool by name."""
    descriptions = {
        "kb_search": "Searching knowledge base for legal documents",
        "web_search": "Searching the web for information",
        "news_search": "Searching recent news articles",
        "calculator": "Performing calculations",
        "url_fetch": "Fetching content from URL",
        "youtube_search": "Searching YouTube",
        "twitter_search": "Searching X/Twitter",
        "file_write": "Writing results to file",
        "email_draft": "Drafting email",
        "bill_status": "Looking up bill status",
        "hansard": "Retrieving parliamentary debates",
        "fees_calculator": "Calculating statutory fees",
        "legal_citation": "Formatting legal citation",
    }
    return descriptions.get(tool_name, f"Running {tool_name}")


def get_current_user_id(request: Request) -> Optional[str]:
    """Get current user ID from auth context if available"""
    try:
        auth_context = getattr(request.state, "auth_context", None)
        if auth_context and auth_context.user_id:
            return auth_context.user_id
    except Exception:
        pass
    return None


def verify_session_ownership(
    session_id: str, user_id: Optional[str], chat_manager
) -> bool:
    """Verify that a session belongs to the specified user"""
    if not user_id:
        # If no user_id provided, allow access (for backward compatibility)
        return True

    session = chat_manager.get_session_with_user(session_id)
    if not session:
        return False

    return session.user_id == user_id


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/sessions", response_model=ChatSessionResponse)
async def create_chat_session(session: ChatSessionCreate, request: Request):
    """Create a new chat session"""
    chat_manager = get_chat_manager()

    try:
        # Get user_id from auth context if available
        user_id = get_current_user_id(request) or session.user_id

        session_id = chat_manager.create_session(session.title, user_id)
        session_data = chat_manager.get_session(session_id)
        return session_data
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_chat_sessions(request: Request, limit: int = 50):
    """List chat sessions for the current user"""
    chat_manager = get_chat_manager()

    try:
        user_id = get_current_user_id(request)
        return chat_manager.list_sessions(user_id, limit)
    except Exception as e:
        logger.error(f"Error listing chat sessions: {e}")
        return []


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(session_id: str, request: Request):
    """Get a specific chat session"""
    chat_manager = get_chat_manager()

    try:
        user_id = get_current_user_id(request)
        if not verify_session_ownership(session_id, user_id, chat_manager):
            raise HTTPException(status_code=403, detail="Access denied")

        session = chat_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str, request: Request):
    """Delete a chat session"""
    chat_manager = get_chat_manager()

    try:
        user_id = get_current_user_id(request)
        if not verify_session_ownership(session_id, user_id, chat_manager):
            raise HTTPException(status_code=403, detail="Access denied")

        chat_manager.delete_session(session_id)
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/sessions/{session_id}", response_model=ChatSessionResponse)
async def rename_chat_session(
    session_id: str, payload: Dict[str, str], request: Request
):
    """Rename a chat session"""
    chat_manager = get_chat_manager()

    try:
        user_id = get_current_user_id(request)
        if not verify_session_ownership(session_id, user_id, chat_manager):
            raise HTTPException(status_code=403, detail="Access denied")

        title = payload.get("title")
        if not title:
            raise HTTPException(status_code=400, detail="Title is required")

        chat_manager.update_session_title(session_id, title)
        session = chat_manager.get_session(session_id)
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# TOOL DISCOVERY ENDPOINT
# =============================================================================


@router.get("/tools", tags=["Tools"])
async def list_available_tools():
    """
    Discover all available tools in OpenAI-compatible format.

    Returns tools in the standard `tools` API format used by
    GPT-4, Gemini (via OpenAI-compat), Claude, and Moonshot AI.
    Each tool includes name, description, and JSON Schema parameters.
    """
    try:
        tools = get_openai_tools()
        return {
            "object": "list",
            "data": tools,
            "total": len(tools),
        }
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail="Failed to load tool definitions")


# =============================================================================
# CHAT COMPLETIONS ENDPOINT (SSE Streaming)
# =============================================================================


@router.post("/completions")
async def chat_completions(request: Request, body: CompletionsCreate):
    """
    Streaming chat completions endpoint.

    Called by frontend sendMessage(). Returns SSE stream with events:
      - tool_start:  Tool execution begins (name, query)
      - tool_result: Tool execution result (name, status, latency_ms)
      - sources:     Retrieved sources/citations
      - content:     Text content token
      - done:        Stream complete with full_answer
      - error:       Error occurred

    Uses AmaniQ v2 graph under the hood.
    """
    chat_manager = get_chat_manager()
    user_id = get_current_user_id(request)

    # Ensure session exists
    session = chat_manager.get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Add user message to history
    chat_manager.add_message(
        session_id=body.session_id,
        content=body.message,
        role="user",
    )

    # Auto-generate title if needed
    if not session.title or session.title == "New Chat":
        try:
            chat_manager.generate_session_title(body.session_id)
        except Exception as e:
            logger.warning(f"Failed to auto-generate session title: {e}")

    async def event_stream():
        full_answer = ""
        try:
            # ===== RESEARCH MODE =====
            if body.is_research:
                bundle_service = _state.research_bundle_service
                if bundle_service is None:
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Research service not available'})}\n\n"
                    return

                yield f"data: {json.dumps({'type': 'tool_start', 'tool_name': 'research', 'query': 'Starting deep research...'})}\n\n"

                # Run the full research flow
                result = await bundle_service.conduct_research(
                    query=body.message,
                    generate_pdf=True,
                    generate_docx=True,
                )

                if result.get("status") == "completed":
                    full_answer = result.get("analysis_summary", "Research complete.")
                    sources = result.get("sources", [])
                    download_urls = result.get("download_urls", {})

                    # Emit tool_result
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool_name': 'research', 'status': 'success', 'latency_ms': 0})}\n\n"

                    # Emit sources
                    if sources:
                        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

                    # Emit content with download links
                    research_content = f"""## Research Results

{result.get('analysis_summary', 'Analysis complete.')}

---

### Download Research Bundle

- [Download PDF]({download_urls.get('pdf', '#')})
- [Download DOCX]({download_urls.get('docx', '#')})

*This research was generated by AmaniQuery Research Intelligence.*
"""
                    yield f"data: {json.dumps({'type': 'content', 'content': research_content})}\n\n"

                    # Emit done with bundle metadata
                    done_payload = {
                        "type": "done",
                        "full_answer": research_content,
                        "is_research": True,
                        "bundle": {
                            "bundle_id": result.get("bundle_id"),
                            "has_pdf": result.get("has_pdf", False),
                            "has_docx": result.get("has_docx", False),
                            "download_urls": download_urls,
                        },
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'error': result.get('error', 'Research failed')})}\n\n"
                return

            # ===== STANDARD AmaniQ v2 MODE =====
            graph = get_amaniq_v2_graph()

            # Build conversation history
            messages = chat_manager.get_messages(body.session_id, limit=5)
            conversation_history = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            initial_state = {
                "current_query": body.message,
                "original_question": body.message,
                "messages": conversation_history
                + [{"role": "user", "content": body.message}],
                "thread_id": body.session_id,
                "user_id": user_id,
            }

            # Run the graph
            config = {"configurable": {"thread_id": body.session_id}}
            final_state = await graph.ainvoke(initial_state, config=config)

            supervisor_decision = final_state.get("supervisor_decision", {})
            tool_plan = supervisor_decision.get("tool_plan", []) or final_state.get(
                "tool_plan", []
            )
            tool_results = final_state.get("tool_results", [])

            # Emit tool_start events
            for tc in tool_plan:
                tool_name = tc.get("tool_name", tc.get("tool", "unknown"))
                yield f"data: {json.dumps({'type': 'tool_start', 'tool_name': tool_name, 'query': tc.get('query', '')})}\n\n"

            # Emit tool_result events
            for tr in tool_results:
                yield f"data: {json.dumps({'type': 'tool_result', 'tool_name': tr.get('tool_name', ''), 'status': tr.get('status', ''), 'latency_ms': tr.get('latency_ms', 0)})}\n\n"

            # Emit sources
            citations = final_state.get("citations", [])
            if citations:
                yield f"data: {json.dumps({'type': 'sources', 'sources': citations})}\n\n"

            # Emit content
            answer = final_state.get("final_response", "")
            if answer:
                full_answer = answer
                yield f"data: {json.dumps({'type': 'content', 'content': answer})}\n\n"

            # Emit completion
            yield f"data: {json.dumps({'type': 'done', 'full_answer': full_answer, 'sources': citations})}\n\n"

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[Completions] Error: {e}")
            import traceback

            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            # Save assistant message
            if full_answer.strip():
                chat_manager.add_message(
                    session_id=body.session_id,
                    content=full_answer,
                    role="assistant",
                    model_used=(
                        "AmaniQ-v2" if not body.is_research else "research-bundle"
                    ),
                )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# SESSION MESSAGES ENDPOINT
# =============================================================================


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def add_chat_message(
    session_id: str, message: ChatMessageCreate, request: Request
):
    """Add a message to a chat session"""
    chat_manager = get_chat_manager()

    try:
        user_id = get_current_user_id(request)
        if not verify_session_ownership(session_id, user_id, chat_manager):
            raise HTTPException(status_code=403, detail="Access denied")

        session = chat_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if message.role == "user":
            # Check for vision data
            use_vision_rag = False
            session_images = []
            if _state.vision_rag_service and _state.vision_storage:
                session_images = _state.vision_storage.get(session_id, [])
                if session_images:
                    use_vision_rag = True

            if not use_vision_rag:
                get_rag_pipeline()  # Ensure rag_pipeline is initialized

            if message.stream:
                # Return streaming response
                return await _handle_streaming_message(
                    session_id,
                    session,
                    message,
                    chat_manager,
                    use_vision_rag,
                    session_images,
                    user_id=user_id,
                )
            else:
                # Return regular response
                return await _handle_regular_message(
                    session_id,
                    session,
                    message,
                    chat_manager,
                    use_vision_rag,
                    session_images,
                    user_id=user_id,
                )
        else:
            # Non-user message (e.g., system)
            attachments_data = _get_attachments(
                message.attachment_ids, session_id, chat_manager
            )

            msg_id = chat_manager.add_message(
                session_id=session_id,
                content=message.content,
                role=message.role,
                attachments=attachments_data,
            )
            messages = chat_manager.get_messages(session_id, limit=1)
            return messages[-1] if messages else None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _handle_streaming_message(
    session_id: str,
    session,
    message: ChatMessageCreate,
    chat_manager,
    use_vision_rag: bool,
    session_images: list,
    user_id: Optional[str] = None,
):
    """Handle streaming message response"""
    # Process query
    if use_vision_rag:
        if _state.vision_rag_service is None or not hasattr(
            _state.vision_rag_service, "query"
        ):
            raise HTTPException(
                status_code=503, detail="Vision RAG service not initialized"
            )
        result = _state.vision_rag_service.query(
            question=message.content,
            session_images=session_images,
            top_k=3,
            temperature=0.7,
            max_tokens=1000,
            stream=True,
        )
        # Convert vision sources
        vision_sources = []
        for src in result.get("sources", []):
            vision_sources.append(
                {
                    "title": src.get("filename", "Image"),
                    "url": "",
                    "source_name": src.get("source_file", "Uploaded Image"),
                    "category": "vision",
                    "excerpt": f"Image similarity: {src.get('similarity', 0):.2f}",
                }
            )
        result["sources"] = vision_sources
    else:
        # Use AmaniQ v2 graph directly for all non-vision queries (REQUIRED)
        logger.info("[Chat] Using AmaniQ v2 graph (System Brain)")
        try:
            # Get the AmaniQ v2 compiled graph directly
            graph = get_amaniq_v2_graph()

            # Get conversation history
            messages = chat_manager.get_messages(session_id, limit=5)
            conversation_history = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            # Build initial state for the graph - only include required fields
            initial_state = {
                "current_query": message.content,
                "original_question": message.content,
                "messages": conversation_history
                + [{"role": "user", "content": message.content}],
                "thread_id": session_id,
                "user_id": user_id,
            }

            # Execute graph directly (THE BRAIN)
            config = {"configurable": {"thread_id": session_id}}
            final_state = await graph.ainvoke(initial_state, config=config)

            # Extract result from final state
            supervisor_decision = final_state.get("supervisor_decision", {})
            tool_plan = supervisor_decision.get("tool_plan", []) or final_state.get(
                "tool_plan", []
            )
            tool_results = final_state.get("tool_results", [])

            amaniq_result = {
                "answer": final_state.get("final_response", ""),
                "sources": final_state.get("citations", []),
                "confidence": final_state.get("response_confidence", 0.0),
                "persona": supervisor_decision.get("persona"),
                "intent": final_state.get("intent"),
                "tool_plan": tool_plan,
                "tool_results": tool_results,
                "tool_execution_latency_ms": final_state.get(
                    "tool_execution_latency_ms", 0
                ),
                "tool_success_rate": final_state.get("tool_success_rate", 0),
            }

            # Format for chat response
            result = {
                "answer": amaniq_result.get("answer", ""),
                "sources": amaniq_result.get("sources", []),
                "retrieved_chunks": len(amaniq_result.get("sources", [])),
                "model_used": f"AmaniQ-v2-{amaniq_result.get('persona', 'wanjiku')}",
                "structured_data": {
                    "confidence": amaniq_result.get("confidence", 0.0),
                    "persona": amaniq_result.get("persona"),
                    "intent": amaniq_result.get("intent"),
                },
                "tool_plan": amaniq_result.get("tool_plan", []),
                "tool_results": amaniq_result.get("tool_results", []),
                "tool_execution_latency_ms": amaniq_result.get(
                    "tool_execution_latency_ms", 0
                ),
                "tool_success_rate": amaniq_result.get("tool_success_rate", 0),
                "answer_stream": None,  # Non-streaming for now
            }
            logger.info(
                f"[Chat] AmaniQ v2 completed with confidence {amaniq_result.get('confidence', 0):.2f}"
            )
        except Exception as e:
            # ONLY on error: Fall back to standard RAG pipeline
            logger.error(f"[Chat] AmaniQ v2 CRITICAL ERROR: {e}")
            import traceback

            logger.error(traceback.format_exc())
            logger.warning("[RAG] Emergency fallback to standard RAG pipeline")
            if _state.rag_pipeline is not None:
                result = _state.rag_pipeline.query_stream(
                    query=message.content,
                    top_k=3,
                    max_tokens=1000,
                    temperature=0.7,
                    session_id=session_id,
                )
            else:
                raise HTTPException(
                    status_code=503, detail="No query service available"
                )

    # Add user message
    attachments_data = _get_attachments(
        message.attachment_ids, session_id, chat_manager
    )
    chat_manager.add_message(
        session_id=session_id,
        content=message.content,
        role="user",
        attachments=attachments_data,
    )

    # Auto-generate title if needed
    if not session.title or session.title == "New Chat":
        try:
            new_title = chat_manager.generate_session_title(session_id)
            logger.info(f"Auto-generated title: {new_title}")
        except Exception as e:
            logger.warning(f"Failed to auto-generate session title: {e}")

    async def generate_stream():
        full_answer = ""
        try:
            # Send tool execution events first (Gemini-like tool cards)
            tool_plan = result.get("tool_plan", [])
            tool_results = result.get("tool_results", [])

            if tool_plan:
                for tc in tool_plan:
                    tool_name = tc.get("tool_name", tc.get("tool", "unknown"))
                    query = tc.get("query", "")
                    tool_event = {
                        "type": "tool_start",
                        "tool_name": tool_name,
                        "query": query,
                        "description": _get_tool_description(tool_name),
                    }
                    yield f"data: {json.dumps(tool_event)}\n\n"

            if tool_results:
                for tr in tool_results:
                    tool_event = {
                        "type": "tool_result",
                        "tool_name": tr.get("tool_name", "unknown"),
                        "query": tr.get("query", ""),
                        "status": tr.get("status", "unknown"),
                        "latency_ms": tr.get("latency_ms", 0),
                        "cached": tr.get("cached", False),
                        "error": tr.get("error"),
                    }
                    yield f"data: {json.dumps(tool_event)}\n\n"

            # Send sources
            sources_data = {
                "type": "sources",
                "sources": result.get("sources", []),
                "retrieved_chunks": result.get("retrieved_chunks", 0),
                "model_used": result.get("model_used", "unknown"),
            }
            yield f"data: {json.dumps(sources_data)}\n\n"

            # Stream the answer
            if "answer_stream" in result and result["answer_stream"] is not None:
                for chunk in result["answer_stream"]:
                    if isinstance(chunk, str):
                        content = chunk
                    elif hasattr(chunk, "choices") and chunk.choices:
                        delta = chunk.choices[0].delta
                        content = delta.content if hasattr(delta, "content") else ""
                    elif hasattr(chunk, "text"):
                        content = chunk.text
                    else:
                        content = ""

                    if content:
                        full_answer += content
                        chunk_data = {"type": "content", "content": content}
                        yield f"data: {json.dumps(chunk_data)}\n\n"
            elif "answer" in result:
                content = str(result["answer"])
                full_answer = content
                chunk_data = {"type": "content", "content": content}
                yield f"data: {json.dumps(chunk_data)}\n\n"

            # Send completion
            completion_data = {
                "type": "done",
                "full_answer": full_answer,
                "structured_data": result.get("structured_data"),
            }
            yield f"data: {json.dumps(completion_data)}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            # Save assistant message
            if full_answer.strip():
                chat_manager.add_message(
                    session_id=session_id,
                    content=full_answer,
                    role="assistant",
                    token_count=result.get("retrieved_chunks", 0),
                    model_used=result.get("model_used", "unknown"),
                    sources=result.get("sources", []),
                )

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _handle_regular_message(
    session_id: str,
    session,
    message: ChatMessageCreate,
    chat_manager,
    use_vision_rag: bool,
    session_images: list,
    user_id: Optional[str] = None,
):
    """Handle regular (non-streaming) message response"""
    if use_vision_rag:
        result = _state.vision_rag_service.query(
            question=message.content,
            session_images=session_images,
            top_k=3,
            temperature=0.7,
            max_tokens=1000,
        )
        # Convert vision sources
        vision_sources = []
        for src in result.get("sources", []):
            vision_sources.append(
                {
                    "title": src.get("filename", "Image"),
                    "url": "",
                    "source_name": src.get("source_file", "Uploaded Image"),
                    "category": "vision",
                    "excerpt": f"Image similarity: {src.get('similarity', 0):.2f}",
                }
            )
        result["sources"] = vision_sources
        result["retrieved_chunks"] = result.get("retrieved_images", 0)
    else:
        # Use AmaniQ v2 agent for all non-vision queries (REQUIRED)
        logger.info("[Chat] Using AmaniQ v2 agent (System Brain)")
        try:
            # Get conversation history
            messages = chat_manager.get_messages(session_id, limit=5)
            conversation_history = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            # Execute AmaniQ v2 pipeline (THE BRAIN)
            amaniq_result = await _state.amaniq_v2_agent.chat(
                message=message.content,
                thread_id=session_id,
                message_history=conversation_history,
                user_id=user_id,
            )

            # Format for chat response
            result = {
                "answer": amaniq_result.get("content", amaniq_result.get("answer", "")),
                "reasoning_content": amaniq_result.get("reasoning_content", ""),
                "sources": amaniq_result.get("sources", []),
                "retrieved_chunks": len(amaniq_result.get("sources", [])),
                "model_used": f"AmaniQ-v2-{amaniq_result.get('persona', 'wanjiku')}",
                "structured_data": {
                    "confidence": amaniq_result.get("confidence", 0.0),
                    "persona": amaniq_result.get("persona"),
                    "intent": amaniq_result.get("intent"),
                },
            }
            logger.info(
                f"[Chat] AmaniQ v2 completed with confidence {amaniq_result.get('confidence', 0):.2f}"
            )
        except Exception as e:
            # ONLY on error: Fall back to standard RAG pipeline
            logger.error(f"[Chat] AmaniQ v2 CRITICAL ERROR: {e}")
            logger.warning("[RAG] Emergency fallback to standard RAG pipeline")
            if _state.rag_pipeline is not None:
                result = _state.rag_pipeline.query(
                    query=message.content,
                    top_k=3,
                    max_tokens=1000,
                    temperature=0.7,
                    session_id=session_id,
                )
            else:
                raise HTTPException(
                    status_code=503, detail="No query service available"
                )

    # Add user message
    attachments_data = _get_attachments(
        message.attachment_ids, session_id, chat_manager
    )
    chat_manager.add_message(
        session_id=session_id,
        content=message.content,
        role="user",
        attachments=attachments_data,
    )

    # Add assistant response
    # Embed reasoning in content if present
    final_content = result["answer"]
    if result.get("reasoning_content"):
        final_content = (
            f"<reasoning>{result['reasoning_content']}</reasoning>\n\n{final_content}"
        )

    chat_manager.add_message(
        session_id=session_id,
        content=final_content,
        role="assistant",
        token_count=result.get("retrieved_chunks", 0),
        model_used=result.get("model_used", "unknown"),
        sources=result.get("sources", []),
    )

    # Generate title if needed
    session = chat_manager.get_session(session_id)
    if session and not session.title:
        title = chat_manager.generate_session_title(session_id)
        chat_manager.update_session_title(session_id, title)

    messages = chat_manager.get_messages(session_id, limit=1)
    return messages[-1] if messages else None


def _get_attachments(
    attachment_ids: Optional[List[str]], session_id: str, chat_manager
) -> Optional[List[Dict]]:
    """Retrieve attachment metadata from session messages"""
    if not attachment_ids:
        return None

    session_messages = chat_manager.get_messages(session_id)
    attachments_data = []
    for msg in session_messages:
        if msg.attachments:
            for att in msg.attachments:
                if att.get("id") in attachment_ids:
                    attachments_data.append(att)
    return attachments_data if attachments_data else None


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_chat_messages(session_id: str, request: Request, limit: int = 100):
    """Get messages for a chat session"""
    chat_manager = get_chat_manager()

    try:
        user_id = get_current_user_id(request)
        if not verify_session_ownership(session_id, user_id, chat_manager):
            raise HTTPException(status_code=403, detail="Access denied")

        return chat_manager.get_messages(session_id, limit)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/attachments")
async def upload_chat_attachment(
    request: Request, session_id: str, file: UploadFile = File(...)
):
    """Upload a document attachment for a chat session"""
    chat_manager = get_chat_manager()

    if _state.vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    user_id = get_current_user_id(request)
    if not verify_session_ownership(session_id, user_id, chat_manager):
        raise HTTPException(status_code=403, detail="Access denied")

    session = chat_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate file size (10MB limit)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds 10MB limit")

    # Validate file type
    allowed_extensions = [".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"]
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}",
        )

    try:
        from Module4_NiruAPI.services.document_processor import DocumentProcessor

        processor = DocumentProcessor()

        result = processor.process_file(
            file_content=file_content, filename=file.filename, session_id=session_id
        )

        # Store chunks in vector store
        if result["chunks"]:
            collection_name = f"chat_session_{session_id}"
            processor.store_chunks_in_vector_store(
                chunks=result["chunks"],
                vector_store=_state.vector_store,
                collection_name=collection_name,
            )

        # Store vision data if available
        vision_data = result.get("vision_data")
        if vision_data and vision_data.get("images"):
            if session_id not in _state.vision_storage:
                _state.vision_storage[session_id] = []
            _state.vision_storage[session_id].extend(vision_data["images"])

        return {
            "attachment": result["attachment"],
            "message": "File processed successfully",
            "vision_processed": vision_data is not None
            and vision_data.get("count", 0) > 0,
        }

    except Exception as e:
        logger.error(f"Error processing attachment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@router.get("/sessions/{session_id}/vision-content")
async def get_vision_content(session_id: str, request: Request):
    """Get vision content for a session"""
    chat_manager = get_chat_manager()

    user_id = get_current_user_id(request)
    if not verify_session_ownership(session_id, user_id, chat_manager):
        raise HTTPException(status_code=403, detail="Access denied")

    session_images = _state.vision_storage.get(session_id, [])

    content_list = []
    for img_data in session_images:
        content_list.append(
            {
                "id": img_data.get("id"),
                "filename": img_data.get("metadata", {}).get("filename", ""),
                "file_path": img_data.get("file_path", ""),
                "type": img_data.get("metadata", {}).get("type", ""),
                "page_number": img_data.get("metadata", {}).get("page_number"),
                "source_file": img_data.get("metadata", {}).get("source_file", ""),
            }
        )

    return {
        "session_id": session_id,
        "count": len(content_list),
        "content": content_list,
    }


@router.post("/messages/{message_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    message_id: str,
    feedback: FeedbackCreate,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Submit feedback for a chat message.
    Auto-scores positive feedback for training dataset.
    """
    chat_manager = get_chat_manager()

    try:
        user_id = get_current_user_id(request)

        # Validate message exists first
        try:
            from Module3_NiruDB.chat_models import ChatMessage

            with chat_manager._get_db_session() as db:
                message = (
                    db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
                )

                if not message:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Message '{message_id}' not found. Please check the message ID.",
                    )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking message: {e}")
            raise HTTPException(
                status_code=500, detail="Error validating message. Please try again."
            )

        # Add feedback
        feedback_id = chat_manager.add_feedback(
            message_id=message_id,
            feedback_type=feedback.feedback_type,
            comment=feedback.comment,
            user_id=user_id,
        )

        # Auto-score on positive feedback (background task)
        if feedback.feedback_type in ["like", "positive"]:
            logger.info(
                f"[Feedback] Positive feedback for {message_id} - triggering quality scoring"
            )
            background_tasks.add_task(
                auto_score_from_feedback,
                message_id=message_id,
                chat_manager=chat_manager,
            )

        return FeedbackResponse(
            id=feedback_id,
            message_id=message_id,
            feedback_type=feedback.feedback_type,
            comment=feedback.comment,
            created_at=datetime.utcnow(),
        )
    except HTTPException:
        raise
    except ValueError as e:
        # Message not found in add_feedback
        raise HTTPException(status_code=404, detail=f"Invalid message ID: {str(e)}")
    except Exception as e:
        logger.error(f"Error adding feedback: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Failed to submit feedback: {str(e)}"
        )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_general_feedback(
    feedback: FeedbackCreate, request: Request, background_tasks: BackgroundTasks
):
    """
    Submit general feedback for a chat message.

    This endpoint accepts message_id in the request body for simpler API usage.
    Auto-scores positive feedback for training dataset.
    """
    try:
        chat_manager = get_chat_manager()

        # Validate message exists
        logger.info(f"Checking message_id: {feedback.message_id}")
        messages = chat_manager.get_messages_by_message_id(feedback.message_id)
        logger.info(
            f"Found {len(messages)} messages for message_id: {feedback.message_id}"
        )
        if not messages:
            logger.warning(f"Message not found: {feedback.message_id}")
            raise HTTPException(status_code=404, detail="Message not found")

        # Add feedback
        feedback_id = chat_manager.add_feedback(
            message_id=feedback.message_id,
            feedback_type=feedback.feedback_type,
            comment=feedback.comment,
        )

        # Auto-score on positive feedback (background task)
        if feedback.feedback_type == "positive":
            background_tasks.add_task(
                auto_score_from_feedback,
                message_id=feedback.message_id,
                chat_manager=chat_manager,
            )

        return FeedbackResponse(
            id=feedback_id,
            message_id=feedback.message_id,
            feedback_type=feedback.feedback_type,
            comment=feedback.comment,
            created_at=datetime.utcnow(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting general feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def auto_score_from_feedback(message_id: str, chat_manager):
    """Background task to score interaction when user gives positive feedback"""
    try:
        from Module4_NiruAPI.agents.quality_scorer import score_and_save_interaction

        result = await score_and_save_interaction(message_id, chat_manager)

        if result:
            logger.info(
                f"[Feedback] Auto-scored {message_id} → training dataset ID: {result}"
            )
        else:
            logger.info(
                f"[Feedback] Scored {message_id} but didn't meet training threshold"
            )

    except Exception as e:
        logger.error(f"[Feedback] Failed to auto-score {message_id}: {e}")


@router.get("/feedback/stats")
async def get_feedback_stats():
    """Get feedback statistics"""
    chat_manager = get_chat_manager()

    try:
        return chat_manager.get_feedback_stats()
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/share")
async def share_chat_session(
    session_id: str, request: Request, share_type: str = "link"
):
    """Generate a shareable link for a chat session"""
    chat_manager = get_chat_manager()

    try:
        user_id = get_current_user_id(request)
        if not verify_session_ownership(session_id, user_id, chat_manager):
            raise HTTPException(status_code=403, detail="Access denied")

        session = chat_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        share_link = f"/shared/{session_id}"

        return {
            "share_link": share_link,
            "session_title": session.title,
            "message_count": session.message_count,
            "share_type": share_type,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sharing chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shared/{session_id}")
async def get_shared_session(session_id: str):
    """Get a shared chat session (public access)"""
    chat_manager = get_chat_manager()

    try:
        session = chat_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        messages = chat_manager.get_messages(session_id)

        return {
            "title": session.title,
            "created_at": session.created_at,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at,
                    "model_used": msg.model_used,
                    "sources": msg.sources,
                    "attachments": msg.attachments,
                }
                for msg in messages
            ],
        }
    except Exception as e:
        logger.error(f"Error getting shared session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
