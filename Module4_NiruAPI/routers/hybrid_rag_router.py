"""
Hybrid RAG Router - Hybrid RAG and advanced retrieval endpoints for AmaniQuery
"""

import json
from typing import Optional, Dict, List, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

router = APIRouter(tags=["Hybrid RAG"])


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
# DEPENDENCIES - Module-level globals set by main app
# =============================================================================

hybrid_rag_pipeline = None
rag_pipeline = None
cache_manager = None
chat_manager = None


def get_hybrid_rag_pipeline():
    """Get the hybrid RAG pipeline instance"""
    if hybrid_rag_pipeline is None:
        raise HTTPException(
            status_code=503, detail="Hybrid RAG pipeline not initialized"
        )
    return hybrid_rag_pipeline


def save_query_to_chat(session_id: str, query: str, result: Dict):
    """Helper function to save query and response to chat database"""
    if chat_manager is None or not session_id:
        return

    try:
        session = chat_manager.get_session(session_id)
        if not session:
            return

        chat_manager.add_message(session_id=session_id, content=query, role="user")

        chat_manager.add_message(
            session_id=session_id,
            content=result.get("answer", ""),
            role="assistant",
            token_count=result.get("retrieved_chunks", 0),
            model_used=result.get("model_used", "unknown"),
            sources=result.get("sources", []),
        )
    except Exception as e:
        logger.warning(f"Failed to save query to chat: {e}")


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/query/hybrid", response_model=QueryResponse)
async def query_hybrid(request: QueryRequest):
    """
    Hybrid RAG query with enhanced encoder and adaptive retrieval

    Uses hybrid convolutional-transformer encoder for improved embeddings
    and adaptive retrieval for context-aware document selection.
    """
    pipeline = get_hybrid_rag_pipeline()

    try:
        result = pipeline.query(
            query=request.query,
            top_k=request.top_k,
            category=request.category,
            source=request.source,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            use_hybrid=True,
            use_adaptive=True,
        )

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
    except Exception as e:
        logger.error(f"Error processing hybrid query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diffusion/generate")
async def generate_synthetic_documents(
    query: Optional[str] = None, num_docs: int = 10, add_to_store: bool = True
):
    """
    Generate synthetic documents using diffusion models

    Args:
        query: Optional query context for generation
        num_docs: Number of documents to generate
        add_to_store: Whether to add generated documents to vector store
    """
    pipeline = get_hybrid_rag_pipeline()

    try:
        generated_texts = pipeline.generate_synthetic_documents(
            query=query, num_docs=num_docs, add_to_store=add_to_store
        )

        return {
            "generated_documents": generated_texts,
            "count": len(generated_texts),
            "added_to_store": add_to_store,
        }
    except Exception as e:
        logger.error(f"Error generating synthetic documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retention/update")
async def trigger_retention_update():
    """
    Trigger retention update (continual learning)

    Updates model weights using generated data for dynamic retention.
    """
    pipeline = get_hybrid_rag_pipeline()

    try:
        pipeline.trigger_retention_update()
        return {"status": "success", "message": "Retention update completed"}
    except Exception as e:
        logger.error(f"Error updating retention: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream/query")
async def stream_query(request: QueryRequest):
    """
    Real-time streaming query endpoint

    Processes queries in real-time with streaming response for both
    queries and generated data.
    """
    pipeline = get_hybrid_rag_pipeline()

    try:
        result = pipeline.query_stream(
            query=request.query,
            top_k=request.top_k,
            category=request.category,
            source=request.source,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            use_hybrid=True,
        )

        if not result.get("stream", False):
            # Fallback to regular response
            if request.session_id:
                save_query_to_chat(request.session_id, request.query, result)

            sources = [Source(**src) for src in result["sources"]]
            return QueryResponse(
                answer=result.get("answer", ""),
                sources=sources if request.include_sources else [],
                query_time=result["query_time"],
                retrieved_chunks=result["retrieved_chunks"],
                model_used=result["model_used"],
            )

        async def generate():
            full_answer = ""
            try:
                answer_stream = result["answer_stream"]
                llm_provider = pipeline.base_rag.llm_provider

                import asyncio

                for chunk in answer_stream:
                    if isinstance(chunk, str):
                        full_answer += chunk
                        yield f"data: {chunk}\n\n"
                        await asyncio.sleep(0)
                    elif llm_provider in ["openai", "moonshot"]:
                        if (
                            hasattr(chunk, "choices")
                            and chunk.choices
                            and chunk.choices[0].delta.content
                        ):
                            content = chunk.choices[0].delta.content
                            full_answer += content
                            yield f"data: {content}\n\n"
                            await asyncio.sleep(0)
                    elif llm_provider == "anthropic":
                        if chunk.type == "content_block_delta" and chunk.delta.text:
                            text = chunk.delta.text
                            full_answer += text
                            yield f"data: {text}\n\n"
                            await asyncio.sleep(0)

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
                    "hybrid_used": result.get("hybrid_used", False),
                }
                yield f"data: [DONE]{json.dumps(sources_data)}\n\n"

                if request.session_id and full_answer:
                    result["answer"] = full_answer
                    save_query_to_chat(request.session_id, request.query, result)

            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                yield f"data: [ERROR]{str(e)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    except Exception as e:
        logger.error(f"Error processing streaming query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hybrid/stats")
async def get_hybrid_stats():
    """Get statistics for hybrid RAG pipeline"""
    pipeline = get_hybrid_rag_pipeline()

    try:
        stats = pipeline.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting hybrid stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
