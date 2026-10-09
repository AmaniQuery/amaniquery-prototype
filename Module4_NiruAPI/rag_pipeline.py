"""
RAG Pipeline - Retrieval-Augmented Generation
Best Practices: Async retrieval, intelligent re-ranking, query optimization
"""

import os
import asyncio
from typing import List, Dict, Optional, Any, AsyncGenerator
import time
import httpx
from loguru import logger
from openai import OpenAI, AsyncOpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import hashlib
import json
import concurrent.futures
from dataclasses import dataclass
from enum import Enum

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Module3_NiruDB.vector_store import VectorStore
from Module3_NiruDB.metadata_manager import MetadataManager

# Reranker and Query Optimizer
try:
    from .services.reranker import IntelligentReranker, QueryOptimizer

    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("Reranker not available, using basic retrieval")

# Global thread pools shared across ALL pipeline instances for 1M+ scale
_global_query_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=10, thread_name_prefix="RAGQuery"
)
_global_retrieval_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=5, thread_name_prefix="RAGRetrieval"
)
_rag_pipeline_clients = {}  # Shared async HTTP clients


class StreamingMode(Enum):
    """Streaming modes for real-time responses"""

    FULL = "full"  # Wait for complete response
    CHUNKS = "chunks"  # Stream chunks as they arrive
    PROGRESSIVE = "progressive"  # Progressive rendering with typing indicator


@dataclass
class StreamingConfig:
    """Configuration for streaming responses"""

    mode: StreamingMode = StreamingMode.PROGRESSIVE
    chunk_size: int = 50  # Characters per chunk
    initial_delay: float = 0.1  # Initial delay before first chunk
    chunk_delay: float = 0.05  # Delay between chunks
    enable_typing_indicator: bool = True
    max_buffer_size: int = 1000  # Max characters to buffer


class RAGPipeline:
    """Blazing fast RAG pipeline with streaming, parallel retrieval, and advanced caching"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm_provider: str = "moonshot",
        model: str = "moonshot-v1-8k",
        config_manager: Optional[Any] = None,
    ):
        """
        Initialize RAG pipeline

        Args:
            vector_store: VectorStore instance (creates new if None)
            llm_provider: LLM provider (openai, anthropic, moonshot, local)
            model: Model name to use
        """
        # Load environment variables
        load_dotenv()

        self.config_manager = config_manager

        # Initialize vector store
        self.vector_store = vector_store or VectorStore(config_manager=config_manager)
        self.metadata_manager = MetadataManager(self.vector_store)

        # Initialize advanced caching system
        self.cache = {}
        self.cache_max_size = 1000  # Increased cache size
        self.semantic_cache = {}  # Semantic similarity cache
        self.query_cache = {}  # Query pattern cache
        self.response_cache = {}  # Full response cache

        # Streaming configuration
        self.streaming_config = StreamingConfig()

        # Connection pooling for parallel operations
        self.query_executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.retrieval_executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

        # Initialize LLM
        self.llm_provider = llm_provider
        self.model = model

        if llm_provider == "openai":
            api_key = self._get_secret("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            self.client = OpenAI(api_key=api_key)
        elif llm_provider == "anthropic":
            api_key = self._get_secret("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            self.client = Anthropic(api_key=api_key)
        elif llm_provider == "gemini":
            api_key = self._get_secret("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set in environment")
            try:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                # Update deprecated model names
                gemini_model = self.model
                if gemini_model == "gemini-pro":
                    gemini_model = "gemini-2.5-flash"  # Use flash as default (faster)
                elif gemini_model not in [
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-flash",
                ]:
                    # If model not specified or unknown, default to flash
                    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
                self.model = gemini_model  # Update self.model to the correct name
                self.client = genai.GenerativeModel(self.model)
                logger.info(f"Using Google Gemini AI with model: {self.model}")

                # Test the connection
                test_response = self.client.generate_content("test")
                logger.info("Gemini AI connection test successful")

            except ImportError:
                raise ValueError(
                    "google-generativeai package not installed. Install with: pip install google-generativeai"
                )
            except Exception as e:
                error_msg = str(e)
                if "API_KEY_INVALID" in error_msg or "PERMISSION_DENIED" in error_msg:
                    raise ValueError(
                        "Gemini API key is invalid. Please get a new API key from "
                        "https://makersuite.google.com/app/apikey and update GEMINI_API_KEY in your .env file"
                    ) from e
                else:
                    raise
        elif llm_provider == "nvidia_nim":
            api_key = self._get_secret("NVIDIA_NIM_API_KEY")
            base_url = (
                os.getenv("NVIDIA_NIM_BASE_URL")
                or self._get_secret("NVIDIA_NIM_BASE_URL")
                or "https://integrate.api.nvidia.com/v1"
            )
            if not api_key:
                raise ValueError("NVIDIA_NIM_API_KEY not set in environment")

            try:
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
                logger.info(f"Using NVIDIA NIM at {base_url}")

                test_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5,
                )
                logger.info("NVIDIA NIM connection test successful")

            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "Invalid Authentication" in error_msg:
                    raise ValueError(
                        "NVIDIA NIM API key is invalid or expired. "
                        "Check NVIDIA_NIM_API_KEY in your .env file"
                    ) from e
                else:
                    logger.error(f"NVIDIA NIM connection failed: {type(e).__name__}")
                    raise
        elif llm_provider == "moonshot":
            api_key = self._get_secret("MOONSHOT_API_KEY")
            base_url = (
                os.getenv("MOONSHOT_BASE_URL")
                or self._get_secret("MOONSHOT_BASE_URL")
                or "https://api.moonshot.ai/v1"
            )
            if not api_key:
                raise ValueError("MOONSHOT_API_KEY not set in environment")

            # Initialize OpenAI client for Moonshot
            try:
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
                logger.info(f"Using Moonshot AI at {base_url}")

                # Test the connection with a simple request
                test_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5,
                )
                logger.info("Moonshot AI connection test successful")

            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "Invalid Authentication" in error_msg:
                    logger.error("Moonshot AI API key is invalid or expired")
                    raise ValueError(
                        "Moonshot AI API key is invalid or expired. "
                        "Check MOONSHOT_API_KEY in your .env file"
                    ) from e
                else:
                    logger.error(
                        f"Moonshot AI connection failed (check API key): {type(e).__name__}"
                    )
                    raise
        else:
            # Check for OpenRouter first for any other provider/model
            openrouter_key = self._get_secret("OPENROUTER_API_KEY")
            if openrouter_key:
                try:
                    self.client = OpenAI(
                        api_key=openrouter_key,
                        base_url="https://openrouter.ai/api/v1",
                        default_headers={
                            "HTTP-Referer": "https://amaniquery.vercel.app",
                            "X-Title": "AmaniQuery",
                        },
                    )
                    self.llm_provider = "openrouter"
                    logger.info(f"Using OpenRouter for model: {self.model}")

                    # Test connection
                    # self.client.models.list() # Optional check

                except Exception as e:
                    logger.error(f"Failed to initialize OpenRouter client: {e}")
                    # Fallback to local/moonshot logic below if needed, or raise
                    raise ValueError(f"Failed to initialize OpenRouter: {e}")
            else:
                # Local model support or fallback
                logger.warning(
                    f"Unknown provider {llm_provider} and no OPENROUTER_API_KEY found. Defaulting to Moonshot configuration check."
                )
                api_key = self._get_secret("MOONSHOT_API_KEY")
                base_url = (
                    os.getenv("MOONSHOT_BASE_URL")
                    or self._get_secret("MOONSHOT_BASE_URL")
                    or "https://api.moonshot.ai/v1"
                )
                if api_key:
                    try:
                        self.client = OpenAI(
                            api_key=api_key,
                            base_url=base_url,
                        )
                        logger.info(f"Using Moonshot AI at {base_url} (Fallback)")
                    except Exception as e:
                        logger.error(f"Failed to initialize Moonshot client: {e}")
                        raise ValueError(
                            "MOONSHOT_API_KEY not set in environment or invalid"
                        )
                else:
                    raise ValueError(
                        f"Provider '{llm_provider}' not supported and OPENROUTER_API_KEY not set."
                    )

        logger.info(
            f"🚀 Blazing Fast RAG Pipeline initialized with {self.llm_provider}/{self.model}"
        )
        logger.info(
            f"📊 Cache capacity: {self.cache_max_size} | Workers: {self.query_executor._max_workers}"
        )

        # Ensemble clients initialized lazily on first use (avoids 4x wasted API client creation)
        self.ensemble_clients: Dict[str, Any] = {}
        self._ensemble_initialized = False

        # Initialize reranker and query optimizer
        self.reranker = None
        self.query_optimizer = None
        if RERANKER_AVAILABLE:
            try:
                self.reranker = IntelligentReranker(llm_client=self.client)
                self.query_optimizer = QueryOptimizer(llm_client=self.client)
                logger.info("Initialized intelligent reranker and query optimizer")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
        else:
            logger.info("Intelligent reranker not available")

    def _get_ensemble_clients(self) -> Dict[str, Any]:
        """Lazy-init ensemble clients only when actually needed (context-limited queries)"""
        if self._ensemble_initialized:
            return self.ensemble_clients

        clients: Dict[str, Any] = {}

        # Primary provider first - most likely to be used
        try:
            if self.llm_provider in ("openai", "moonshot", "nvidia_nim"):
                api_key = None
                base_url = None
                if self.llm_provider == "openai":
                    api_key = self._get_secret("OPENAI_API_KEY")
                elif self.llm_provider == "moonshot":
                    api_key = self._get_secret("MOONSHOT_API_KEY")
                    base_url = (
                        os.getenv("MOONSHOT_BASE_URL") or "https://api.moonshot.ai/v1"
                    )
                elif self.llm_provider == "nvidia_nim":
                    api_key = self._get_secret("NVIDIA_NIM_API_KEY")
                    base_url = (
                        os.getenv("NVIDIA_NIM_BASE_URL")
                        or "https://integrate.api.nvidia.com/v1"
                    )
                if api_key:
                    clients[self.llm_provider] = {
                        "client": (
                            OpenAI(api_key=api_key, base_url=base_url)
                            if base_url
                            else OpenAI(api_key=api_key)
                        ),
                        "model": self.model,
                    }
        except Exception:
            pass

        # Secondary providers only when primary is functional
        if clients:
            if "openai" not in clients:
                try:
                    api_key = self._get_secret("OPENAI_API_KEY")
                    if api_key:
                        clients["openai"] = {
                            "client": OpenAI(api_key=api_key),
                            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        }
                except Exception:
                    pass

            if "moonshot" not in clients:
                try:
                    api_key = self._get_secret("MOONSHOT_API_KEY")
                    if api_key:
                        base_url = (
                            os.getenv("MOONSHOT_BASE_URL")
                            or "https://api.moonshot.ai/v1"
                        )
                        clients["moonshot"] = {
                            "client": OpenAI(api_key=api_key, base_url=base_url),
                            "model": os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k"),
                        }
                except Exception:
                    pass

            if "nvidia_nim" not in clients:
                try:
                    api_key = self._get_secret("NVIDIA_NIM_API_KEY")
                    if api_key:
                        base_url = (
                            os.getenv("NVIDIA_NIM_BASE_URL")
                            or "https://integrate.api.nvidia.com/v1"
                        )
                        clients["nvidia_nim"] = {
                            "client": OpenAI(api_key=api_key, base_url=base_url),
                            "model": os.getenv(
                                "NVIDIA_NIM_MODEL", "meta/llama-3.1-70b-instruct"
                            ),
                        }
                except Exception:
                    pass

            try:
                api_key = self._get_secret("ANTHROPIC_API_KEY")
                if api_key:
                    clients["anthropic"] = {
                        "client": Anthropic(api_key=api_key),
                        "model": os.getenv(
                            "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"
                        ),
                    }
            except Exception:
                pass

        self.ensemble_clients = clients
        self._ensemble_initialized = True
        logger.info(f"Lazy ensemble: {len(clients)} model(s) available")
        return clients

    def _is_context_limited(
        self, retrieved_docs: List[Dict], min_relevance: float = 0.5
    ) -> bool:
        """Check if retrieved context is limited or insufficient"""
        if not retrieved_docs:
            return True

        # Check if relevance scores are too low
        if len(retrieved_docs) < 3:
            return True

        # Check average relevance (if available)
        scores = [doc.get("score", 0) for doc in retrieved_docs if "score" in doc]
        if scores and sum(scores) / len(scores) < min_relevance:
            return True

        return False

    def _get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Fetch secret values from environment variables with ConfigManager fallback."""
        value = os.getenv(key)
        if value:
            return value

        if self.config_manager:
            try:
                secret = self.config_manager.get_config(key)
                if secret:
                    return secret
            except Exception as exc:
                logger.warning(f"Failed to fetch {key} from config store: {exc}")

        return default

    def _get_cache_key(
        self, query: str, top_k: int, category: Optional[str], source: Optional[str]
    ) -> str:
        """Generate cache key for query"""
        key_data = {
            "query": query.strip().lower(),
            "top_k": top_k,
            "category": category,
            "source": source,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache(self, key: str) -> Optional[Dict]:
        """Get cached result if exists"""
        if key in self.cache:
            logger.info("Cache hit!")
            return self.cache[key]
        return None

    def _set_cache(self, key: str, result: Dict):
        """Store result in cache"""
        if len(self.cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = result
        logger.info("Result cached")

    @property
    def llm_service(self):
        """Expose LLM client for other components"""
        return self.client

    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> str:
        """
        Generate answer from context (public wrapper for _generate_answer)

        Args:
            query: User question
            context: Retrieved context
            system_prompt: Optional system prompt override
            temperature: LLM temperature
            max_tokens: Max tokens

        Returns:
            Generated answer string
        """
        # Use the internal method which now supports system_prompt override
        result = self._generate_answer(
            query=query,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        # Return just the answer text as expected by callers
        if isinstance(result, dict):
            return result.get("answer", "")
        return str(result)

    async def query_stream(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        source: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        max_context_length: int = 3000,
        session_id: Optional[str] = None,
        enable_typing_indicator: bool = True,
    ) -> AsyncGenerator[Dict, None]:
        """
        🔥 Blazing fast streaming RAG query with real-time response generation

        Features:
        - Progressive retrieval and generation
        - Real-time typing indicators
        - Parallel namespace search
        - Intelligent caching
        """
        start_time = time.time()

        # Send typing indicator immediately
        if enable_typing_indicator:
            yield {
                "type": "typing_start",
                "timestamp": start_time,
                "message": "Thinking...",
            }

        # 1. 🚀 Ultra-fast cache check (semantic + exact)
        cache_key = self._get_cache_key(query, top_k, category, source)
        cached_result = self._get_cache(cache_key)
        if cached_result:
            logger.info("⚡ Cache hit! Streaming cached response")
            if enable_typing_indicator:
                await asyncio.sleep(0.1)  # Small delay for natural feel

            # Stream cached response in chunks
            answer_chunks = cached_result["answer"].split()
            current_chunk = ""
            for i, word in enumerate(answer_chunks):
                current_chunk += word + " "
                if (
                    len(current_chunk) >= self.streaming_config.chunk_size
                    or i == len(answer_chunks) - 1
                ):
                    yield {
                        "type": "chunk",
                        "content": current_chunk.strip(),
                        "sources": cached_result["sources"],
                        "complete": i == len(answer_chunks) - 1,
                    }
                    current_chunk = ""
                    await asyncio.sleep(self.streaming_config.chunk_delay)

            yield {
                "type": "complete",
                "answer": cached_result["answer"],
                "sources": cached_result["sources"],
                "query_time": time.time() - start_time,
                "retrieved_chunks": cached_result["retrieved_chunks"],
                "model_used": cached_result["model_used"],
                "cached": True,
            }
            return

        # 2. 🏃‍♂️ Parallel namespace determination and retrieval
        filter_dict = {}
        if category:
            filter_dict["category"] = category
        if source:
            filter_dict["source_name"] = source

        # Determine namespaces in parallel
        namespaces_future = asyncio.create_task(
            self._determine_namespaces_async(query, category, source)
        )

        # Start retrieval while determining namespaces
        retrieval_start = time.time()

        # 3. ⚡ Progressive retrieval with streaming
        async for retrieval_result in self._progressive_retrieval(
            query, top_k, filter_dict, namespaces_future, session_id
        ):
            if retrieval_result["type"] == "documents_found":
                docs = retrieval_result["documents"]
                logger.info(f"🎯 Found {len(docs)} documents, starting generation")

                # Prepare context progressively
                context = self._prepare_context(docs, max_context_length)

                # Start streaming generation
                async for gen_result in self._stream_generate_answer(
                    query, context, temperature, max_tokens
                ):
                    yield gen_result

                # Format and yield final result
                sources = self._format_sources(docs)
                query_time = time.time() - start_time

                # Cache the complete result
                final_result = {
                    "answer": gen_result.get("complete_answer", ""),
                    "sources": sources,
                    "query_time": query_time,
                    "retrieved_chunks": len(docs),
                    "model_used": self.model,
                }
                self._set_cache(cache_key, final_result)

                yield {
                    "type": "complete",
                    "answer": final_result["answer"],
                    "sources": sources,
                    "query_time": query_time,
                    "retrieved_chunks": len(docs),
                    "model_used": self.model,
                    "cached": False,
                }
                return

            elif retrieval_result["type"] == "no_documents":
                yield {
                    "type": "complete",
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "query_time": time.time() - start_time,
                    "retrieved_chunks": 0,
                    "model_used": self.model,
                    "cached": False,
                }
                return

    async def _progressive_retrieval(
        self,
        query: str,
        top_k: int,
        filter_dict: Dict,
        namespaces_future: asyncio.Task,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict, None]:
        """Progressive retrieval with early results streaming"""

        # Wait for namespace determination
        namespaces = await namespaces_future
        logger.info(f"🔍 Searching namespaces: {namespaces}")

        # Start with quick local search (ChromaDB)
        quick_results = await self._quick_local_search(
            query, min(top_k, 3), filter_dict
        )
        if quick_results:
            yield {
                "type": "documents_found",
                "documents": quick_results,
                "source": "quick_local",
            }
            return

        # If no quick results, do comprehensive parallel search
        retrieval_tasks = []
        for namespace in namespaces:
            task = asyncio.create_task(
                self._retrieve_from_namespace_async(
                    query, namespace, top_k * 2, filter_dict
                )
            )
            retrieval_tasks.append(task)

        # Also search session-specific collection
        if session_id:
            session_task = asyncio.create_task(
                self._retrieve_session_docs_async(query, session_id, min(3, top_k))
            )
            retrieval_tasks.append(session_task)

        # Collect results as they complete
        all_docs = []
        for task in asyncio.as_completed(retrieval_tasks):
            try:
                docs = await task
                if docs:
                    all_docs.extend(docs)
                    # If we get enough good results, yield early
                    if len(all_docs) >= top_k:
                        # Sort by relevance and take top_k
                        all_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
                        yield {
                            "type": "documents_found",
                            "documents": all_docs[:top_k],
                            "source": "comprehensive",
                        }
                        return
            except Exception as e:
                logger.warning(f"Retrieval task failed: {e}")

        # Final result
        if all_docs:
            all_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
            yield {
                "type": "documents_found",
                "documents": all_docs[:top_k],
                "source": "final",
            }
        else:
            yield {"type": "no_documents"}

    async def _quick_local_search(
        self, query: str, n_results: int, filter_dict: Dict
    ) -> List[Dict]:
        """Ultra-fast local search using ChromaDB"""
        try:
            if (
                hasattr(self.vector_store, "chromadb_collection")
                and self.vector_store.chromadb_collection
            ):
                # Quick embedding
                query_embedding = self.vector_store.embedding_model.encode(
                    query
                ).tolist()

                # Fast local query
                results = self.vector_store.chromadb_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=filter_dict if filter_dict else None,
                )

                # Format results
                docs = []
                if results and results.get("documents"):
                    for i, doc_text in enumerate(results["documents"][0]):
                        docs.append(
                            {
                                "text": doc_text,
                                "source": (
                                    results["metadatas"][0][i]
                                    if results.get("metadatas")
                                    else {}
                                ),
                                "score": 1.0
                                - (
                                    results["distances"][0][i]
                                    if results.get("distances")
                                    else 0.0
                                ),
                            }
                        )

                return docs
        except Exception as e:
            logger.warning(f"Quick local search failed: {e}")

        return []

    async def _stream_generate_answer(
        self, query: str, context: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[Dict, None]:
        """Stream answer generation with real-time chunks"""

        # Use streaming if available
        if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
            try:
                system_prompt = """You are AmaniQuery, an AI assistant specialized in Kenyan law, parliamentary proceedings, and current affairs.
                Provide accurate, concise answers based on the provided context."""

                user_prompt = f"""Context: {context}
                
                Question: {query}
                
                Provide a clear, accurate answer based on the context above."""

                # Start streaming
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )

                complete_answer = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        complete_answer += content

                        yield {
                            "type": "chunk",
                            "content": content,
                            "complete_answer": complete_answer,
                        }

                        await asyncio.sleep(0.01)  # Small delay for natural flow

                return

            except Exception as e:
                logger.warning(
                    f"Streaming failed, falling back to regular generation: {e}"
                )

        # Fallback to regular generation
        result = self._generate_answer(query, context, temperature, max_tokens)
        if isinstance(result, dict):
            answer = result.get("answer", "")
        else:
            answer = str(result)

        # Stream the complete answer as one chunk
        yield {"type": "chunk", "content": answer, "complete_answer": answer}

    async def _determine_namespaces_async(
        self, query: str, category: Optional[str] = None, source: Optional[str] = None
    ) -> List[str]:
        """Async version of namespace determination"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._determine_namespaces, query, category, source
        )

    async def _retrieve_session_docs_async(
        self, query: str, session_id: str, n_results: int
    ) -> List[Dict]:
        """Async session document retrieval"""
        loop = asyncio.get_event_loop()

        def _sync_retrieve():
            try:
                collection_name = f"chat_session_{session_id}"
                if hasattr(self.vector_store, "get_collection"):
                    collection = self.vector_store.get_collection(collection_name)
                    if collection:
                        results = collection.query(
                            query_texts=[query],
                            n_results=n_results,
                        )

                        docs = []
                        if results and len(results) > 0:
                            for i, doc_text in enumerate(
                                results["documents"][0] if results["documents"] else []
                            ):
                                docs.append(
                                    {
                                        "text": doc_text,
                                        "source": (
                                            results.get("metadatas", [[]])[0][i]
                                            if results.get("metadatas")
                                            else {}
                                        ),
                                        "score": 1.0
                                        - (
                                            results.get("distances", [[]])[0][i]
                                            if results.get("distances")
                                            else 0.0
                                        ),
                                    }
                                )
                        return docs
            except Exception as e:
                logger.warning(f"Session retrieval failed: {e}")
            return []

        return await loop.run_in_executor(None, _sync_retrieve)

    def query(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        source: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        max_context_length: int = 3000,
        session_id: Optional[str] = None,
        use_optimized: bool = True,
    ) -> Dict:
        """
        Optimized RAG query with parallel retrieval and intelligent caching

        Args:
            query: User question
            top_k: Number of documents to retrieve
            category: Filter by category
            source: Filter by source
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
            use_optimized: Use optimized parallel retrieval (default: True)

        Returns:
            Dictionary with answer and sources
        """
        start_time = time.time()

        # 1. Ultra-fast cache check
        cache_key = self._get_cache_key(query, top_k, category, source)
        cached_result = self._get_cache(cache_key)
        if cached_result:
            logger.info("[INFO] Cache hit!")
            return {**cached_result, "cached": True}

        # 2. Optimized parallel retrieval
        if use_optimized:
            try:
                return self._optimized_query(
                    query,
                    top_k,
                    category,
                    source,
                    temperature,
                    max_tokens,
                    max_context_length,
                    session_id,
                    start_time,
                )
            except Exception as e:
                logger.warning(f"Optimized query failed, falling back to standard: {e}")

        # 3. Fallback to standard sequential retrieval
        return self._standard_query(
            query,
            top_k,
            category,
            source,
            temperature,
            max_tokens,
            max_context_length,
            session_id,
            start_time,
        )

    def _optimized_query(
        self,
        query: str,
        top_k: int,
        category: Optional[str],
        source: Optional[str],
        temperature: float,
        max_tokens: int,
        max_context_length: int,
        session_id: Optional[str],
        start_time: float,
    ) -> Dict:
        """Optimized query with parallel retrieval and smart caching"""

        # Parallel namespace determination and retrieval
        filter_dict = {}
        if category:
            filter_dict["category"] = category
        if source:
            filter_dict["source_name"] = source

        # Quick local search first (sub-100ms)
        quick_results = self._quick_local_search_sync(query, min(top_k, 3), filter_dict)
        if quick_results:
            logger.info(f"[INFO] Quick search found {len(quick_results)} results")
            return self._generate_answer_from_docs(
                query,
                quick_results,
                temperature,
                max_tokens,
                max_context_length,
                start_time,
                cache_key=self._get_cache_key(query, top_k, category, source),
            )

        # Parallel comprehensive search
        namespaces = self._determine_namespaces(query, category, source)

        # Use thread pool for parallel retrieval
        retrieval_futures = []
        for namespace in namespaces:
            future = self.retrieval_executor.submit(
                self._retrieve_from_namespace_sync,
                query,
                namespace,
                top_k * 2,
                filter_dict,
            )
            retrieval_futures.append(future)

        # Also search session documents
        if session_id:
            session_future = self.retrieval_executor.submit(
                self._retrieve_session_docs_sync, query, session_id, min(3, top_k)
            )
            retrieval_futures.append(session_future)

        # Collect results as they complete
        all_docs = []
        for future in concurrent.futures.as_completed(retrieval_futures):
            try:
                docs = future.result(timeout=5.0)  # 5 second timeout
                if docs:
                    all_docs.extend(docs)
                    # Early exit if we have enough good results
                    if len(all_docs) >= top_k:
                        break
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")

        # Sort by relevance and process
        if all_docs:
            all_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
            return self._generate_answer_from_docs(
                query,
                all_docs[:top_k],
                temperature,
                max_tokens,
                max_context_length,
                start_time,
                cache_key=self._get_cache_key(query, top_k, category, source),
            )

        # Fallback to ensemble if no documents found
        if len(self._get_ensemble_clients()) > 0:
            logger.info("Using ensemble fallback")
            return self._query_with_ensemble(query, temperature, max_tokens, start_time)

        return {
            "answer": "I couldn't find any relevant information to answer your question.",
            "sources": [],
            "query_time": time.time() - start_time,
            "retrieved_chunks": 0,
            "model_used": self.model,
            "cached": False,
        }

    def _standard_query(
        self,
        query: str,
        top_k: int,
        category: Optional[str],
        source: Optional[str],
        temperature: float,
        max_tokens: int,
        max_context_length: int,
        session_id: Optional[str],
        start_time: float,
    ) -> Dict:
        """Standard sequential query (fallback method)"""

        # Sequential namespace retrieval (original method)
        filter_dict = {}
        if category:
            filter_dict["category"] = category
        if source:
            filter_dict["source_name"] = source

        namespaces_to_search = self._determine_namespaces(query, category, source)

        retrieved_docs = []
        for namespace in namespaces_to_search:
            try:
                namespace_docs = self.vector_store.query(
                    query_text=query,
                    n_results=top_k // len(namespaces_to_search),
                    filter=filter_dict if filter_dict else None,
                    namespace=namespace,
                )
                retrieved_docs.extend(namespace_docs)
                logger.info(
                    f"Retrieved {len(namespace_docs)} documents from namespace: {namespace}"
                )
            except Exception as e:
                logger.warning(f"Failed to query namespace {namespace}: {e}")

        # Session documents
        if session_id:
            try:
                collection_name = f"chat_session_{session_id}"
                if hasattr(self.vector_store, "get_collection"):
                    collection = self.vector_store.get_collection(collection_name)
                    if collection:
                        session_docs = collection.query(
                            query_texts=[query], n_results=min(3, top_k)
                        )
                        if session_docs and len(session_docs) > 0:
                            for i, doc_text in enumerate(
                                session_docs["documents"][0]
                                if session_docs["documents"]
                                else []
                            ):
                                session_doc = {
                                    "text": doc_text,
                                    "source": (
                                        session_docs.get("metadatas", [[]])[0][i]
                                        if session_docs.get("metadatas")
                                        else {}
                                    ),
                                    "score": (
                                        session_docs.get("distances", [[]])[0][i]
                                        if session_docs.get("distances")
                                        else 0.0
                                    ),
                                }
                                retrieved_docs.append(session_doc)
            except Exception as e:
                logger.warning(f"Failed to retrieve session documents: {e}")

        return self._generate_answer_from_docs(
            query,
            retrieved_docs,
            temperature,
            max_tokens,
            max_context_length,
            start_time,
            cache_key=self._get_cache_key(query, top_k, category, source),
        )

    def _generate_answer_from_docs(
        self,
        query: str,
        retrieved_docs: List[Dict],
        temperature: float,
        max_tokens: int,
        max_context_length: int,
        start_time: float,
        cache_key: str,
    ) -> Dict:
        """Generate answer from retrieved documents"""

        # Check if context is limited
        use_ensemble = self._is_context_limited(retrieved_docs)

        if not retrieved_docs or use_ensemble:
            if use_ensemble and len(self._get_ensemble_clients()) > 0:
                logger.info("Context limited - using multi-model ensemble")
                return self._query_with_ensemble(
                    query, temperature, max_tokens, start_time
                )
            else:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "query_time": time.time() - start_time,
                    "retrieved_chunks": 0,
                    "model_used": self.model,
                    "cached": False,
                }

        # Prepare context and generate answer
        context = self._prepare_context(retrieved_docs, max_context_length)
        answer = self._generate_answer(query, context, temperature, max_tokens)
        sources = self._format_sources(retrieved_docs)

        query_time = time.time() - start_time

        result = {
            "answer": answer,
            "sources": sources,
            "query_time": query_time,
            "retrieved_chunks": len(retrieved_docs),
            "model_used": self.model,
            "cached": False,
        }

        # Cache the result
        self._set_cache(cache_key, result)

        return result

    def _quick_local_search_sync(
        self, query: str, n_results: int, filter_dict: Dict
    ) -> List[Dict]:
        """Ultra-fast local search using ChromaDB (sync version)"""
        try:
            if (
                hasattr(self.vector_store, "chromadb_collection")
                and self.vector_store.chromadb_collection
            ):
                # Quick embedding
                query_embedding = self.vector_store.embedding_model.encode(
                    query
                ).tolist()

                # Fast local query
                results = self.vector_store.chromadb_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=filter_dict if filter_dict else None,
                )

                # Format results
                docs = []
                if results and results.get("documents"):
                    for i, doc_text in enumerate(results["documents"][0]):
                        docs.append(
                            {
                                "text": doc_text,
                                "source": (
                                    results["metadatas"][0][i]
                                    if results.get("metadatas")
                                    else {}
                                ),
                                "score": 1.0
                                - (
                                    results["distances"][0][i]
                                    if results.get("distances")
                                    else 0.0
                                ),
                            }
                        )

                return docs
        except Exception as e:
            logger.warning(f"Quick local search failed: {e}")

        return []

    def _retrieve_from_namespace_sync(
        self,
        query: str,
        namespace: str,
        n_results: int,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """Sync wrapper for namespace retrieval"""
        try:
            return self.vector_store.query(
                query_text=query,
                n_results=n_results,
                filter=filter_dict if filter_dict else None,
                namespace=namespace,
            )
        except Exception as e:
            logger.warning(f"Failed to query namespace {namespace}: {e}")
            return []

    def _retrieve_session_docs_sync(
        self, query: str, session_id: str, n_results: int
    ) -> List[Dict]:
        """Sync session document retrieval"""
        try:
            collection_name = f"chat_session_{session_id}"
            if hasattr(self.vector_store, "get_collection"):
                collection = self.vector_store.get_collection(collection_name)
                if collection:
                    results = collection.query(query_texts=[query], n_results=n_results)

                    docs = []
                    if results and len(results) > 0:
                        for i, doc_text in enumerate(
                            results["documents"][0] if results["documents"] else []
                        ):
                            docs.append(
                                {
                                    "text": doc_text,
                                    "source": (
                                        results.get("metadatas", [[]])[0][i]
                                        if results.get("metadatas")
                                        else {}
                                    ),
                                    "score": 1.0
                                    - (
                                        results.get("distances", [[]])[0][i]
                                        if results.get("distances")
                                        else 0.0
                                    ),
                                }
                            )
                    return docs
        except Exception as e:
            logger.warning(f"Session retrieval failed: {e}")
        return []

    async def aquery(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        source: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        max_context_length: int = 3000,
        session_id: Optional[str] = None,
        use_reranking: bool = True,
        use_query_expansion: bool = False,
    ) -> Dict:
        """
        Async RAG query with 2025 best practices.

        Features:
        - Parallel namespace retrieval (2x speedup)
        - Intelligent re-ranking
        - Optional query expansion (HyDE)

        Args:
            query: User question
            top_k: Number of documents to retrieve
            category: Filter by category
            source: Filter by source
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
            use_reranking: Apply intelligent re-ranking
            use_query_expansion: Use HyDE query expansion

        Returns:
            Dictionary with answer and sources
        """
        start_time = time.time()

        # Check cache first
        cache_key = self._get_cache_key(query, top_k, category, source)
        cached_result = self._get_cache(cache_key)
        if cached_result:
            logger.info("Cache hit for async query!")
            return cached_result

        # Optional: Query expansion with HyDE
        search_query = query
        if use_query_expansion and self.query_optimizer:
            search_query = await self.query_optimizer.hyde_transform(query)
            logger.info(f"HyDE expanded query: {search_query[:100]}...")

        # Determine namespaces
        filter_dict = {}
        if category:
            filter_dict["category"] = category
        if source:
            filter_dict["source_name"] = source

        namespaces_to_search = self._determine_namespaces(query, category, source)

        # PARALLEL RETRIEVAL - 2x speedup
        logger.info(
            f"Async parallel retrieval from {len(namespaces_to_search)} namespaces"
        )

        retrieval_tasks = [
            self._retrieve_from_namespace_async(
                search_query,
                namespace,
                top_k * 2,  # Over-retrieve for reranking
                filter_dict,
            )
            for namespace in namespaces_to_search
        ]

        # Run all retrievals in parallel
        namespace_results = await asyncio.gather(
            *retrieval_tasks, return_exceptions=True
        )

        # Flatten results
        retrieved_docs = []
        for i, result in enumerate(namespace_results):
            if isinstance(result, Exception):
                logger.warning(f"Namespace {namespaces_to_search[i]} failed: {result}")
            elif result:
                retrieved_docs.extend(result)
                logger.info(
                    f"Retrieved {len(result)} docs from {namespaces_to_search[i]}"
                )

        # INTELLIGENT RE-RANKING
        if use_reranking and self.reranker and len(retrieved_docs) > top_k:
            logger.info(f"Re-ranking {len(retrieved_docs)} documents")
            retrieved_docs = await self.reranker.rerank(
                query=query, documents=retrieved_docs, top_k=top_k
            )
            logger.info(f"After reranking: {len(retrieved_docs)} documents")
        else:
            # Fallback: sort by score and take top_k
            retrieved_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
            retrieved_docs = retrieved_docs[:top_k]

        # Check if context is limited
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "query_time": time.time() - start_time,
                "retrieved_chunks": 0,
                "model_used": self.model,
            }

        # Prepare context
        context = self._prepare_context(retrieved_docs, max_context_length)

        # Generate answer (still sync, could be made async)
        logger.info("Generating answer with LLM")
        answer = self._generate_answer(query, context, temperature, max_tokens)

        # Format sources
        sources = self._format_sources(retrieved_docs)

        query_time = time.time() - start_time

        logger.info(f"Async query completed in {query_time:.2f}s")

        result = {
            "answer": answer,
            "sources": sources,
            "query_time": query_time,
            "retrieved_chunks": len(retrieved_docs),
            "model_used": self.model,
            "used_reranking": use_reranking and self.reranker is not None,
            "used_query_expansion": use_query_expansion
            and self.query_optimizer is not None,
        }

        # Store in cache
        self._set_cache(cache_key, result)

        return result

    async def _retrieve_from_namespace_async(
        self,
        query: str,
        namespace: str,
        n_results: int,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """Async wrapper for namespace retrieval."""
        loop = asyncio.get_event_loop()

        def _sync_retrieve():
            try:
                return self.vector_store.query(
                    query_text=query,
                    n_results=n_results,
                    filter=filter_dict if filter_dict else None,
                    namespace=namespace,
                )
            except Exception as e:
                logger.warning(f"Failed to query namespace {namespace}: {e}")
                return []

        return await loop.run_in_executor(None, _sync_retrieve)

    def _determine_namespaces(
        self, query: str, category: Optional[str] = None, source: Optional[str] = None
    ) -> List[str]:
        """Determine which namespaces to search based on query content and filters"""
        namespaces = []

        # If specific category is provided, map to namespace
        if category:
            category_lower = category.lower()
            if any(
                keyword in category_lower
                for keyword in [
                    "law",
                    "constitution",
                    "act",
                    "legislation",
                    "judgment",
                    "case law",
                ]
            ):
                namespaces.append("kenya_law")
            elif any(
                keyword in category_lower for keyword in ["news", "current affairs"]
            ):
                namespaces.append("kenya_news")
            elif any(
                keyword in category_lower
                for keyword in ["parliament", "bill", "hansard", "budget"]
            ):
                namespaces.append("kenya_parliament")
            elif any(keyword in category_lower for keyword in ["global", "trend"]):
                namespaces.append("global_trends")

        # Analyze query content for keywords
        query_lower = query.lower()

        # Legal keywords
        legal_keywords = [
            "law",
            "constitution",
            "act",
            "bill",
            "legislation",
            "court",
            "judge",
            "judgment",
            "case",
            "statute",
            "amendment",
            "clause",
            "section",
            "article",
        ]
        if any(keyword in query_lower for keyword in legal_keywords):
            if "kenya_law" not in namespaces:
                namespaces.append("kenya_law")

        # News keywords
        news_keywords = [
            "news",
            "current",
            "recent",
            "today",
            "latest",
            "breaking",
            "update",
            "report",
        ]
        if any(keyword in query_lower for keyword in news_keywords):
            if "kenya_news" not in namespaces:
                namespaces.append("kenya_news")

        # Parliament keywords
        parliament_keywords = [
            "parliament",
            "mp",
            "bill",
            "debate",
            "hansard",
            "budget",
            "vote",
            "speaker",
            "committee",
        ]
        if any(keyword in query_lower for keyword in parliament_keywords):
            if "kenya_parliament" not in namespaces:
                namespaces.append("kenya_parliament")

        # Global/international keywords
        global_keywords = [
            "global",
            "international",
            "world",
            "foreign",
            "diplomatic",
            "treaty",
        ]
        if any(keyword in query_lower for keyword in global_keywords):
            if "global_trends" not in namespaces:
                namespaces.append("global_trends")

        # Historical keywords (check for years before 2010)
        import re

        years = re.findall(r"\b(19\d{2}|20[01]\d)\b", query)
        if years and any(int(year) < 2010 for year in years):
            if "historical" not in namespaces:
                namespaces.append("historical")

        # If no specific namespaces determined, search all relevant ones
        if not namespaces:
            # Default search across main namespaces
            namespaces = ["kenya_law", "kenya_news", "kenya_parliament"]

        logger.info(f"Query '{query[:50]}...' will search namespaces: {namespaces}")
        return namespaces

    def _query_with_ensemble(
        self,
        query: str,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        start_time: float = None,
    ) -> Dict:
        """
        Sequential fallback chain — tries ONE model at a time.
        Replaces the old parallel-all-models ensemble that cost N+1 LLM calls.
        Fallback order: primary → moonshot → openai/gpt-4o-mini → anthropic/claude-haiku.
        """
        if start_time is None:
            start_time = time.time()

        clients = self._get_ensemble_clients()
        if not clients:
            return {
                "answer": "I couldn't generate a response. Please try again.",
                "sources": [],
                "query_time": time.time() - start_time,
                "retrieved_chunks": 0,
                "model_used": "none",
            }

        # Build a cost-ordered fallback chain
        primary = self.llm_provider
        fallback_order = [primary]
        for candidate in ["moonshot", "openai", "nvidia_nim", "anthropic", "gemini"]:
            if candidate != primary and candidate in clients:
                fallback_order.append(candidate)

        system_prompt = (
            "You are AmaniQuery, an AI assistant specialized in Kenyan law, "
            "parliamentary proceedings, and current affairs. "
            "Provide a concise, accurate answer. Be brief and factual."
        )
        user_prompt = (
            f"Question: {query}\n\nProvide a concise answer based on your knowledge."
        )

        last_error = None
        for provider in fallback_order:
            info = clients[provider]
            try:
                client = info["client"]
                model = info["model"]

                if provider in ("openai", "moonshot", "nvidia_nim", "openrouter"):
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    answer = resp.choices[0].message.content
                elif provider == "anthropic":
                    resp = client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    answer = resp.content[0].text
                elif provider == "gemini":
                    import google.generativeai as genai

                    resp = client.generate_content(
                        f"{system_prompt}\n\n{user_prompt}",
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature, max_output_tokens=max_tokens
                        ),
                    )
                    answer = resp.text
                else:
                    continue

                if answer:
                    logger.info(
                        f"Fallback chain: {provider} answered (tried {fallback_order.index(provider)+1}/{len(fallback_order)})"
                    )
                    return {
                        "answer": answer,
                        "sources": [],
                        "query_time": time.time() - start_time,
                        "retrieved_chunks": 0,
                        "model_used": f"fallback({provider})",
                    }
            except Exception as e:
                last_error = e
                logger.warning(f"Fallback chain: {provider} failed: {e}")
                continue

        logger.error(
            f"All {len(fallback_order)} fallback models failed. Last error: {last_error}"
        )
        return {
            "answer": "I'm having trouble generating a response right now. Please try again later.",
            "sources": [],
            "query_time": time.time() - start_time,
            "retrieved_chunks": 0,
            "model_used": "error",
        }

    def _create_simple_stream(self, text: str):
        """Create a simple stream from text"""

        class SimpleStream:
            def __init__(self, text):
                self.words = text.split()
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.index >= len(self.words):
                    raise StopIteration

                chunk_size = 3
                end_idx = min(self.index + chunk_size, len(self.words))
                chunk_text = " ".join(self.words[self.index : end_idx])
                if end_idx < len(self.words):
                    chunk_text += " "
                self.index = end_idx

                # Return in OpenAI format
                class Chunk:
                    def __init__(self, content):
                        class Delta:
                            def __init__(self, c):
                                self.content = c

                        class Choice:
                            def __init__(self, c):
                                self.delta = Delta(c)

                        self.choices = [Choice(content)]

                return Chunk(chunk_text)

        return SimpleStream(text)

    def _query_stream_with_ensemble(
        self,
        query: str,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        start_time: float = None,
    ) -> Dict:
        """Stream fallback via sequential fallback chain — never more than one LLM call."""
        result = self._query_with_ensemble(query, temperature, max_tokens, start_time)
        answer = result.get("answer", "")
        qtime = result.get("query_time", 0)
        model = result.get("model_used", "fallback")
        return {
            "answer_stream": self._create_simple_stream(answer),
            "sources": [],
            "query_time": qtime,
            "retrieved_chunks": 0,
            "model_used": model,
            "stream": True,
        }

    def query_stream(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        source: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        max_context_length: int = 3000,
        session_id: Optional[str] = None,
    ):
        """
        Run RAG query with streaming response

        Args:
            query: User question
            top_k: Number of documents to retrieve
            category: Filter by category
            source: Filter by source
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
            max_context_length: Maximum context length
            session_id: Optional session ID for session-specific document retrieval

        Returns:
            Dictionary with answer stream and sources
        """
        start_time = time.time()

        try:
            # 1. Retrieve relevant documents
            logger.info(f"Retrieving documents for query: {query[:50]}...")

            filter_dict = {}
            if category:
                filter_dict["category"] = category
            if source:
                filter_dict["source_name"] = source

            # Retrieve from main vector store with namespace support
            namespaces_to_search = self._determine_namespaces(query, category, source)

            retrieved_docs = []
            for namespace in namespaces_to_search:
                try:
                    namespace_docs = self.vector_store.query(
                        query_text=query,
                        n_results=top_k
                        // len(
                            namespaces_to_search
                        ),  # Distribute top_k across namespaces
                        filter=filter_dict if filter_dict else None,
                        namespace=namespace,
                    )
                    retrieved_docs.extend(namespace_docs)
                    logger.info(
                        f"Retrieved {len(namespace_docs)} documents from namespace: {namespace}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to query namespace {namespace}: {e}")
                    # Fallback to default namespace if namespace query fails
                    try:
                        fallback_docs = self.vector_store.query(
                            query_text=query,
                            n_results=top_k // len(namespaces_to_search),
                            filter=filter_dict if filter_dict else None,
                        )
                        retrieved_docs.extend(fallback_docs)
                        logger.info(
                            f"Fallback: Retrieved {len(fallback_docs)} documents from default namespace"
                        )
                    except Exception as fallback_e:
                        logger.warning(f"Fallback query also failed: {fallback_e}")

            # If session_id provided, also retrieve from session-specific collection
            if session_id:
                try:
                    collection_name = f"chat_session_{session_id}"
                    # Try to query session-specific collection
                    if hasattr(self.vector_store, "get_collection"):
                        collection = self.vector_store.get_collection(collection_name)
                        if collection:
                            session_docs = collection.query(
                                query_texts=[query],
                                n_results=min(3, top_k),  # Limit session docs
                            )
                            # Merge session docs with main docs
                            if session_docs and len(session_docs) > 0:
                                # Format session docs to match main docs format
                                if (
                                    isinstance(session_docs, dict)
                                    and "documents" in session_docs
                                ):
                                    for i, doc_text in enumerate(
                                        session_docs["documents"][0]
                                        if session_docs["documents"]
                                        else []
                                    ):
                                        session_doc = {
                                            "text": doc_text,
                                            "source": (
                                                session_docs.get("metadatas", [[]])[0][
                                                    i
                                                ]
                                                if session_docs.get("metadatas")
                                                else {}
                                            ),
                                            "score": (
                                                session_docs.get("distances", [[]])[0][
                                                    i
                                                ]
                                                if session_docs.get("distances")
                                                else 0.0
                                            ),
                                        }
                                        retrieved_docs.append(session_doc)
                                logger.info(
                                    f"Retrieved {len(session_docs) if isinstance(session_docs, list) else 1} documents from session collection"
                                )
                except Exception as e:
                    logger.warning(f"Failed to retrieve session documents: {e}")

            # Check if context is limited - use ensemble if so
            use_ensemble = self._is_context_limited(retrieved_docs)

            if use_ensemble and len(self._get_ensemble_clients()) > 0:
                logger.info(
                    "Context limited - using multi-model ensemble with streaming"
                )
                return self._query_stream_with_ensemble(
                    query=query,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    start_time=start_time,
                )

            # 2. Prepare context (use available docs or empty)
            context = self._prepare_context(retrieved_docs, max_context_length)

            # 3. Generate answer with streaming (always provide response)
            logger.info("Generating streaming answer with LLM")
            answer_stream = self._generate_answer_stream(
                query, context, temperature, max_tokens
            )

            # 4. Format sources
            sources = self._format_sources(retrieved_docs)

            query_time = time.time() - start_time

            return {
                "answer_stream": answer_stream,
                "sources": sources,
                "query_time": query_time,
                "retrieved_chunks": len(retrieved_docs),
                "model_used": self.model,
                "stream": True,
            }

        except Exception as e:
            logger.error(f"Error in query_stream: {e}")
            # Return non-streaming fallback response
            return {
                "answer": f"I encountered an error while processing your query: {str(e)}. Please try again.",
                "sources": [],
                "query_time": time.time() - start_time,
                "retrieved_chunks": 0,
                "model_used": self.model,
                "stream": False,
            }

    def _prepare_context(self, docs: List[Dict], max_context_length: int = 3000) -> str:
        """Prepare context from retrieved documents with Prompt Pruning"""
        context_parts = []
        total_length = 0

        for i, doc in enumerate(docs, 1):
            meta = doc["metadata"]
            text = doc["text"]

            # Optimization: Prune very short chunks
            if len(text) < 50:
                continue

            # Optimization: Tighter truncation
            chunk_limit = 400
            if len(text) > chunk_limit:
                text = text[:chunk_limit] + "..."

            # Optimization: Minimal metadata format
            # Format: [i] Title: Text
            title = meta.get("title", "Untitled")
            context_part = f"[{i}] {title}: {text}\n"

            # Check if adding this would exceed max length
            if total_length + len(context_part) > max_context_length:
                break

            context_parts.append(context_part)
            total_length += len(context_part)

        return "\n".join(context_parts)

    def _generate_answer(
        self,
        query: str,
        context: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate answer using LLM, potentially with interactive widgets"""

        # System prompt with Impact Agent instructions (default if not provided)
        if system_prompt is None:
            system_prompt = """You are AmaniQuery, an AI assistant specialized in Kenyan law, parliamentary proceedings, and current affairs.

CRITICAL INSTRUCTION: DETECT QUANTITATIVE POLICY QUERIES
If the user's query involves calculating costs, levies, taxes, fines, or statutory deductions (Housing Levy, NSSF, NHIF/SHIF, PAYE, Fuel Levy, Parking Fees, etc.), you MUST output a JSON response containing an interactive widget definition.

CRITICAL INSTRUCTION: DETECT LEGAL AMENDMENTS (GIT-DIFF)
If the user's query asks about changes, amendments, new bills, or comparisons (e.g., "changed from X to Y", "amendment to section", "what is new in the bill"), you MUST output a JSON response containing a `github_diff` object.

OUTPUT FORMAT:
If a widget or diff is needed, output ONLY a JSON object with this structure:
{
  "answer": "Brief text explanation...",
  "interactive_widgets": [ ... ],
  "github_diff": {
    "old_text": "Original legal text...",
    "new_text": "Amended legal text...",
    "title": "Bill Name → Section/Clause",
    "highlight_type": "side_by_side"
  }
}

If NO widget or diff is needed, output a standard text response.

FEW-SHOT EXAMPLES FOR WIDGETS:
... (keep existing widget examples) ...

FEW-SHOT EXAMPLES FOR LEGAL DIFFS:

Example 1: Housing Levy Change
User: "How did the housing levy change in the new Finance Bill?"
Response:
{
  "answer": "The Affordable Housing Levy was amended to clarify the deduction rate and matching contribution. The rate remains 1.5%, but the text now explicitly mandates the employer's matching contribution.",
  "github_diff": {
    "title": "Finance Bill 2024 → Clause 31B",
    "old_text": "An employer shall pay the levy deducted under this section to the collector...",
    "new_text": "An employer shall pay the levy deducted under this section and an equal amount as the employer's contribution to the collector...",
    "highlight_type": "side_by_side"
  }
}

Example 2: SHIF Rates
User: "What is the new SHIF rate compared to NHIF?"
Response:
{
  "answer": "The Social Health Insurance Fund (SHIF) introduces a flat 2.75% rate on gross household income, replacing the graduated NHIF scale.",
  "github_diff": {
    "title": "Social Health Insurance Act → Contribution Rate",
    "old_text": "Contributions shall be paid at the rates specified in the Schedule (Graduated Scale: KES 150 - KES 1,700)",
    "new_text": "Every household shall contribute to the Fund at a rate of 2.75% of the gross household income.",
    "highlight_type": "side_by_side"
  }
}

Example 3: VAT on Fuel
User: "Did they increase VAT on fuel?"
Response:
{
  "answer": "Yes, the Finance Act 2023 increased the VAT on petroleum products from 8% to 16%.",
  "github_diff": {
    "title": "Finance Act 2023 → VAT Act Amendment",
    "old_text": "The tax shall be charged at the rate of 8 percent on the supply of petroleum products...",
    "new_text": "The tax shall be charged at the rate of 16 percent on the supply of petroleum products...",
    "highlight_type": "side_by_side"
  }
}

Example 4: Traffic Fines Amendment
User: "Amendment to traffic fines for speeding"
Response:
{
  "answer": "The Traffic (Amendment) Bill proposes increasing the maximum fine for speeding offenses.",
  "github_diff": {
    "title": "Traffic (Amendment) Bill → Section 42",
    "old_text": "Any person who contravenes this section shall be liable to a fine not exceeding twenty thousand shillings...",
    "new_text": "Any person who contravenes this section shall be liable to a fine not exceeding one hundred thousand shillings...",
    "highlight_type": "side_by_side"
  }
}

Example 5: Excise Duty on Betting
User: "Change in excise duty for betting"
Response:
{
  "answer": "The excise duty on betting stakes was increased from 7.5% to 12.5%.",
  "github_diff": {
    "title": "Excise Duty Act → Betting Tax",
    "old_text": "Excise duty on betting shall be at the rate of 7.5 percent of the amount wagered or staked.",
    "new_text": "Excise duty on betting shall be at the rate of 12.5 percent of the amount wagered or staked.",
    "highlight_type": "side_by_side"
  }
}"""

        # User prompt
        user_prompt = f"""Context from relevant documents:

{context}

Question: {query}

Provide a concise answer. If the query is quantitative (taxes/levies), output JSON with `interactive_widgets`. If it asks about amendments/changes, output JSON with `github_diff`. Otherwise, output standard text."""

        try:
            raw_answer = ""
            if self.llm_provider in ["openai", "moonshot", "nvidia_nim"]:
                # OpenAI-compatible API (OpenAI, Moonshot, NVIDIA NIM)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw_answer = response.choices[0].message.content

            elif self.llm_provider == "gemini":
                # Gemini uses a different API format
                import google.generativeai as genai

                # Configure generation parameters
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

                # Combine system prompt with user prompt
                full_prompt = f"{system_prompt}\n\n{user_prompt}"

                response = self.client.generate_content(
                    full_prompt, generation_config=generation_config
                )

                raw_answer = response.text

            elif self.llm_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                raw_answer = response.content[0].text

            else:
                return {"answer": "LLM provider not supported"}

            # Parse response
            logger.info(f"LLM response length: {len(raw_answer) if raw_answer else 0}")
            if not raw_answer or not raw_answer.strip():
                logger.warning("LLM returned empty or whitespace-only response")
                return {
                    "answer": "I apologize, but I was unable to generate a response. Please try rephrasing your question."
                }

            # Try to parse as JSON
            try:
                # Clean up potential markdown code blocks
                clean_answer = raw_answer.strip()
                if clean_answer.startswith("```json"):
                    clean_answer = clean_answer[7:]
                if clean_answer.endswith("```"):
                    clean_answer = clean_answer[:-3]

                parsed_json = json.loads(clean_answer)

                # If valid JSON with answer and widgets/diff
                if isinstance(parsed_json, dict) and "answer" in parsed_json:
                    return {
                        "answer": parsed_json["answer"],
                        "interactive_widgets": parsed_json.get("interactive_widgets"),
                        "github_diff": parsed_json.get("github_diff"),
                    }
                else:
                    # JSON but not our expected format, treat as text
                    return {"answer": raw_answer}
            except json.JSONDecodeError:
                # Not JSON, treat as standard text response
                return {"answer": raw_answer}

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating answer: {error_msg}")
            return {"answer": f"Error generating answer: {error_msg}"}

    def _generate_answer_stream(
        self,
        query: str,
        context: str,
        temperature: float,
        max_tokens: int,
    ):
        """Generate answer using LLM with streaming"""
        # NOTE: For Impact Agent (widgets), we currently prioritize correctness over streaming.
        # We generate the full response (including widgets) and then yield the text answer.
        # This ensures widgets are generated if needed, even if we can't stream them yet.

        # Generate full response using the widget-aware method
        response = self._generate_answer(query, context, temperature, max_tokens)

        # If it's a dict (standard response), extract answer
        if isinstance(response, dict):
            answer = response.get("answer", "")
            # We yield the answer as a single chunk.
            # This simulates streaming but allows us to reuse the widget logic.
            yield answer
        else:
            # Fallback if something went wrong and it returned a string
            yield str(response)

    def _format_sources(self, docs: List[Dict]) -> List[Dict]:
        """Format source citations"""
        sources = []

        for doc in docs:
            meta = doc["metadata"]

            source = {
                "title": meta.get("title", "Untitled"),
                "url": meta.get("source_url", ""),
                "source_name": meta.get("source_name", "Unknown"),
                "category": meta.get("category", "Unknown"),
                "author": meta.get("author"),
                "publication_date": meta.get("publication_date"),
                "relevance_score": 1.0
                - doc.get("distance", 0.0),  # Convert distance to similarity
                "excerpt": (
                    doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"]
                ),
            }

            sources.append(source)

        return sources
