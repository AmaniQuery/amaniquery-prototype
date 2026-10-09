"""
AmaniQ v2 Performance Optimization - Background Agents
======================================================

Token cost and latency optimization through:
1. Conversation summarization (every 12 messages)
2. Query → Answer caching (Redis, 48h TTL)
3. Vector search result caching (Redis, 24h TTL)
4. Pre-warming top 50 legal documents at startup

Redis Key Patterns:
- amq:v2:answer:{query_hash}     → Full answer cache (48h)
- amq:v2:vector:{query_hash}     → Vector search results (24h)
- amq:v2:summary:{thread_id}     → Conversation summaries (7d)
- amq:v2:warm:status             → Pre-warm status (1h)

Author: Eng. Onyango Benard
Version: 2.0
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class CacheConfig:
    """Cache configuration with TTLs"""

    # TTLs in seconds
    ANSWER_CACHE_TTL: int = 48 * 3600  # 48 hours
    VECTOR_CACHE_TTL: int = 24 * 3600  # 24 hours
    SUMMARY_CACHE_TTL: int = 7 * 24 * 3600  # 7 days
    WARM_STATUS_TTL: int = 3600  # 1 hour

    # Key prefixes
    ANSWER_PREFIX: str = "amq:v2:answer"
    VECTOR_PREFIX: str = "amq:v2:vector"
    SUMMARY_PREFIX: str = "amq:v2:summary"
    WARM_PREFIX: str = "amq:v2:warm"

    # Summarization trigger
    MESSAGES_BEFORE_SUMMARY: int = 12

    # Pre-warm settings
    PREWARM_BATCH_SIZE: int = 10
    PREWARM_DELAY_BETWEEN_BATCHES: float = 0.5


# Top 50 most-queried legal documents for pre-warming
TOP_LEGAL_QUERIES = [
    # Constitution
    "Constitution of Kenya 2010",
    "Bill of Rights Kenya Constitution",
    "Article 27 equality Constitution Kenya",
    "Article 40 property rights Constitution",
    "Article 43 economic social rights",
    "Article 47 fair administrative action",
    "Article 48 access to justice",
    "Article 49 rights of arrested persons",
    "Article 50 fair hearing rights",
    "Article 159 judicial authority principles",
    # Criminal Law
    "Penal Code Kenya Cap 63",
    "Sexual Offences Act 2006",
    "Prevention of Terrorism Act Kenya",
    "Computer Misuse and Cybercrimes Act 2018",
    "Anti-Corruption and Economic Crimes Act",
    "Proceeds of Crime and Anti-Money Laundering Act",
    "Narcotic Drugs and Psychotropic Substances Act",
    # Commercial Law
    "Companies Act 2015 Kenya",
    "Insolvency Act 2015 Kenya",
    "Sale of Goods Act Kenya",
    "Law of Contract Act Kenya",
    "Partnership Act Kenya",
    "Trade Marks Act Kenya",
    "Copyright Act Kenya",
    # Employment Law
    "Employment Act 2007 Kenya",
    "Labour Relations Act Kenya",
    "Work Injury Benefits Act",
    "Occupational Safety and Health Act",
    "unfair dismissal Employment Act",
    "termination notice Employment Act section 35",
    # Land Law
    "Land Registration Act 2012",
    "Land Act 2012 Kenya",
    "Land Control Act Kenya",
    "Community Land Act 2016",
    "land transfer procedure Kenya",
    "stamp duty land Kenya",
    # Family Law
    "Marriage Act 2014 Kenya",
    "Matrimonial Property Act 2013",
    "Children Act Kenya",
    "Succession Act Kenya",
    "inheritance law Kenya",
    "divorce procedure Kenya",
    # Tax & Finance
    "Income Tax Act Kenya",
    "Value Added Tax Act Kenya",
    "Finance Act 2024 Kenya",
    "Tax Procedures Act Kenya",
    "Excise Duty Act Kenya",
    # Civil Procedure
    "Civil Procedure Act Kenya",
    "limitation of actions Kenya",
    "Evidence Act Kenya",
    "Arbitration Act Kenya",
]


# =============================================================================
# REDIS CLIENT WRAPPER
# =============================================================================


class RedisCache:
    """
    Redis cache wrapper with connection pooling.
    Falls back to in-memory cache if Redis unavailable.
    """

    def __init__(self):
        self._redis = None
        self._memory_fallback: Dict[str, Tuple[Any, float]] = {}
        self._initialized = False
        self.config = CacheConfig()

    async def initialize(self) -> bool:
        """Initialize Redis connection"""
        if self._initialized:
            return self._redis is not None

        try:
            import redis.asyncio as aioredis

            # HARDCODED: Backend always uses local Docker Redis
            redis_url = "redis://localhost:6379/0"
            self._redis = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
            )
            # Test connection
            await self._redis.ping()
            logger.info(f"✅ Backend Redis connected: {redis_url}")
            self._initialized = True
            return True

        except Exception as e:
            logger.warning(f"Redis unavailable, using memory fallback: {e}")
            self._redis = None
            self._initialized = True
            return False

    def _hash_key(self, text: str) -> str:
        """Create consistent hash for cache keys"""
        normalized = text.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if self._redis:
            try:
                return await self._redis.get(key)
            except Exception as e:
                logger.debug(f"Redis get error: {e}")

        # Memory fallback
        if key in self._memory_fallback:
            value, expiry = self._memory_fallback[key]
            if time.time() < expiry:
                return value
            del self._memory_fallback[key]
        return None

    async def set(self, key: str, value: str, ttl: int) -> bool:
        """Set value with TTL"""
        if self._redis:
            try:
                await self._redis.setex(key, ttl, value)
                return True
            except Exception as e:
                logger.debug(f"Redis set error: {e}")

        # Memory fallback
        self._memory_fallback[key] = (value, time.time() + ttl)
        # Cleanup old entries (simple LRU)
        if len(self._memory_fallback) > 10000:
            oldest = sorted(self._memory_fallback.items(), key=lambda x: x[1][1])[:1000]
            for k, _ in oldest:
                del self._memory_fallback[k]
        return True

    async def delete(self, key: str) -> bool:
        """Delete key"""
        if self._redis:
            try:
                await self._redis.delete(key)
                return True
            except Exception:
                pass

        if key in self._memory_fallback:
            del self._memory_fallback[key]
        return True

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if self._redis:
            try:
                return await self._redis.exists(key) > 0
            except Exception:
                pass
        return key in self._memory_fallback

    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()


# Global cache instance
_cache: Optional[RedisCache] = None


async def get_cache() -> RedisCache:
    """Get or create global cache instance"""
    global _cache
    if _cache is None:
        _cache = RedisCache()
        await _cache.initialize()
    return _cache


# =============================================================================
# ANSWER CACHING
# =============================================================================


class AnswerCache:
    """
    Cache complete query → answer pairs.

    Key Pattern: amq:v2:answer:{query_hash}
    TTL: 48 hours

    Value Structure:
    {
        "query": "original query",
        "answer": "full response",
        "citations": [...],
        "created_at": "ISO timestamp",
        "hit_count": 0
    }
    """

    def __init__(self, cache: RedisCache):
        self.cache = cache
        self.config = CacheConfig()

    def _make_key(self, query: str) -> str:
        """Create cache key from query"""
        query_hash = self.cache._hash_key(query)
        return f"{self.config.ANSWER_PREFIX}:{query_hash}"

    async def get_cached_answer(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached answer for exact query match.

        Args:
            query: User's query (exact match)

        Returns:
            Cached answer dict or None
        """
        key = self._make_key(query)
        cached = await self.cache.get(key)

        if cached:
            try:
                data = json.loads(cached)
                # Update hit count (fire and forget)
                data["hit_count"] = data.get("hit_count", 0) + 1
                asyncio.create_task(
                    self.cache.set(key, json.dumps(data), self.config.ANSWER_CACHE_TTL)
                )
                logger.debug(f"Answer cache HIT: {key[:50]}...")
                return data
            except json.JSONDecodeError:
                pass

        logger.debug(f"Answer cache MISS: {key[:50]}...")
        return None

    async def cache_answer(
        self,
        query: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Cache a query → answer pair.

        Args:
            query: User's original query
            answer: Complete response
            citations: List of citations used
            metadata: Additional metadata

        Returns:
            Success status
        """
        key = self._make_key(query)

        data = {
            "query": query,
            "answer": answer,
            "citations": citations or [],
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "hit_count": 0,
        }

        success = await self.cache.set(
            key, json.dumps(data), self.config.ANSWER_CACHE_TTL
        )

        if success:
            logger.debug(f"Cached answer: {key[:50]}... (TTL: 48h)")

        return success


# =============================================================================
# VECTOR SEARCH CACHING
# =============================================================================


class VectorSearchCache:
    """
    Cache vector search results by query hash.

    Key Pattern: amq:v2:vector:{query_hash}:{namespace}
    TTL: 24 hours

    Value Structure:
    {
        "query": "search query",
        "namespace": "kenya_law",
        "results": [...],
        "top_k": 5,
        "created_at": "ISO timestamp"
    }
    """

    def __init__(self, cache: RedisCache):
        self.cache = cache
        self.config = CacheConfig()

    def _make_key(self, query: str, namespace: Optional[str] = None) -> str:
        """Create cache key from query and namespace"""
        query_hash = self.cache._hash_key(query)
        ns_part = namespace or "all"
        return f"{self.config.VECTOR_PREFIX}:{query_hash}:{ns_part}"

    async def get_cached_results(
        self, query: str, namespace: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached vector search results.

        Args:
            query: Search query
            namespace: Optional namespace filter

        Returns:
            List of cached results or None
        """
        key = self._make_key(query, namespace)
        cached = await self.cache.get(key)

        if cached:
            try:
                data = json.loads(cached)
                logger.debug(
                    f"Vector cache HIT: {query[:30]}... ({len(data.get('results', []))} results)"
                )
                return data.get("results", [])
            except json.JSONDecodeError:
                pass

        return None

    async def cache_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        top_k: int = 5,
    ) -> bool:
        """
        Cache vector search results.

        Args:
            query: Search query
            results: Search results to cache
            namespace: Namespace that was searched
            top_k: Number of results requested

        Returns:
            Success status
        """
        key = self._make_key(query, namespace)

        data = {
            "query": query,
            "namespace": namespace,
            "results": results,
            "top_k": top_k,
            "created_at": datetime.utcnow().isoformat(),
        }

        success = await self.cache.set(
            key, json.dumps(data, default=str), self.config.VECTOR_CACHE_TTL
        )

        if success:
            logger.debug(
                f"Cached {len(results)} vector results for: {query[:30]}... (TTL: 24h)"
            )

        return success


# =============================================================================
# CONVERSATION SUMMARIZATION
# =============================================================================


class ConversationSummarizer:
    """
    Summarize conversations every N messages to reduce token costs.

    Key Pattern: amq:v2:summary:{thread_id}
    TTL: 7 days

    Strategy:
    - After every 12 messages, summarize the oldest 10
    - Keep last 2 messages + summary for context
    - Store summary in Redis for session recovery
    """

    def __init__(self, cache: RedisCache):
        self.cache = cache
        self.config = CacheConfig()

    def _make_key(self, thread_id: str) -> str:
        """Create cache key for thread summary"""
        return f"{self.config.SUMMARY_PREFIX}:{thread_id}"

    async def should_summarize(self, messages: List[Dict]) -> bool:
        """Check if we should trigger summarization"""
        return len(messages) >= self.config.MESSAGES_BEFORE_SUMMARY

    async def get_summary(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get existing summary for thread"""
        key = self._make_key(thread_id)
        cached = await self.cache.get(key)

        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass
        return None

    async def save_summary(
        self,
        thread_id: str,
        summary: str,
        summarized_count: int,
        key_entities: List[str],
    ) -> bool:
        """Save conversation summary"""
        key = self._make_key(thread_id)

        data = {
            "summary": summary,
            "summarized_message_count": summarized_count,
            "key_entities": key_entities,
            "created_at": datetime.utcnow().isoformat(),
        }

        return await self.cache.set(
            key, json.dumps(data), self.config.SUMMARY_CACHE_TTL
        )

    def create_summary_prompt(self, messages: List[Dict]) -> str:
        """Create prompt for LLM to summarize conversation"""
        conversation_text = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            conversation_text.append(f"{role.upper()}: {content}")

        return f"""Summarize this legal research conversation concisely. 
Preserve: key legal questions, case names, statutes mentioned, and conclusions reached.
Keep it under 200 words.

CONVERSATION:
{chr(10).join(conversation_text)}

SUMMARY:"""

    async def summarize_and_compress(
        self, thread_id: str, messages: List[Dict], llm_client=None
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Summarize old messages and return compressed message list.

        Args:
            thread_id: Conversation thread ID
            messages: Full message list
            llm_client: Optional LLM client for summarization

        Returns:
            Tuple of (compressed_messages, summary_text)
        """
        if len(messages) < self.config.MESSAGES_BEFORE_SUMMARY:
            return messages, None

        # Split: summarize oldest N-2, keep last 2
        messages_to_summarize = messages[:-2]
        messages_to_keep = messages[-2:]

        # Generate summary
        summary_text = None
        if llm_client:
            try:
                prompt = self.create_summary_prompt(messages_to_summarize)
                # This would call the actual LLM
                # summary_text = await llm_client.generate(prompt)
                summary_text = self._fallback_summary(messages_to_summarize)
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}")
                summary_text = self._fallback_summary(messages_to_summarize)
        else:
            summary_text = self._fallback_summary(messages_to_summarize)

        # Extract key entities
        key_entities = self._extract_entities(messages_to_summarize)

        # Save summary to cache
        await self.save_summary(
            thread_id, summary_text, len(messages_to_summarize), key_entities
        )

        # Create compressed message list
        summary_message = {
            "role": "system",
            "content": f"[CONVERSATION SUMMARY - {len(messages_to_summarize)} messages]\n{summary_text}",
            "metadata": {
                "type": "summary",
                "summarized_count": len(messages_to_summarize),
                "key_entities": key_entities,
            },
        }

        compressed = [summary_message] + messages_to_keep

        logger.info(
            f"Compressed {len(messages)} messages → {len(compressed)} "
            f"(saved ~{len(messages_to_summarize) * 500} tokens)"
        )

        return compressed, summary_text

    def _fallback_summary(self, messages: List[Dict]) -> str:
        """Simple fallback summary without LLM"""
        topics = set()
        questions = []

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Extract potential topics
            legal_terms = [
                "constitution",
                "article",
                "section",
                "act",
                "case",
                "court",
                "judgment",
                "rights",
                "law",
                "legal",
            ]
            for term in legal_terms:
                if term in content.lower():
                    topics.add(term)

            # Capture user questions
            if role == "user" and "?" in content:
                questions.append(content[:100])

        summary_parts = []
        if questions:
            summary_parts.append(f"Questions discussed: {'; '.join(questions[:3])}")
        if topics:
            summary_parts.append(f"Topics covered: {', '.join(list(topics)[:5])}")

        return (
            " | ".join(summary_parts) if summary_parts else "General legal discussion."
        )

    def _extract_entities(self, messages: List[Dict]) -> List[str]:
        """Extract key legal entities from messages"""
        import re

        entities = set()

        patterns = [
            r"\b(Article\s+\d+)",  # Article 27
            r"\b(Section\s+\d+)",  # Section 35
            r"\b(Cap\s+\d+)",  # Cap 63
            r"\[\d{4}\]\s*(?:e)?KLR",  # [2022] eKLR
            r"\b([A-Z][a-z]+\s+v\s+[A-Z][a-z]+)",  # Case names
        ]

        for msg in messages:
            content = msg.get("content", "")
            for pattern in patterns:
                matches = re.findall(pattern, content)
                entities.update(matches)

        return list(entities)[:10]


# =============================================================================
# PRE-WARMING SERVICE
# =============================================================================


class PreWarmService:
    """
    Pre-warm vector search cache with top legal documents at startup.

    Key Pattern: amq:v2:warm:status
    TTL: 1 hour (status only)

    Warms up TOP_LEGAL_QUERIES into vector cache.
    """

    def __init__(self, cache: RedisCache, vector_cache: VectorSearchCache):
        self.cache = cache
        self.vector_cache = vector_cache
        self.config = CacheConfig()
        self._is_warming = False

    def _status_key(self) -> str:
        return f"{self.config.WARM_PREFIX}:status"

    async def get_warm_status(self) -> Optional[Dict[str, Any]]:
        """Get current pre-warm status"""
        cached = await self.cache.get(self._status_key())
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass
        return None

    async def _update_status(self, status: Dict[str, Any]):
        """Update warm status in cache"""
        await self.cache.set(
            self._status_key(), json.dumps(status), self.config.WARM_STATUS_TTL
        )

    async def prewarm_cache(self, vector_store=None) -> Dict[str, Any]:
        """
        Pre-warm the vector cache with top legal queries.

        Args:
            vector_store: VectorStore instance to query

        Returns:
            Status dict with results
        """
        if self._is_warming:
            return {"status": "already_running"}

        # Check if recently warmed
        status = await self.get_warm_status()
        if status and status.get("status") == "complete":
            age_minutes = (
                datetime.utcnow()
                - datetime.fromisoformat(status.get("completed_at", "2000-01-01"))
            ).total_seconds() / 60
            if age_minutes < 60:
                return {"status": "recently_completed", "age_minutes": age_minutes}

        self._is_warming = True
        start_time = time.time()

        results = {
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "total_queries": len(TOP_LEGAL_QUERIES),
            "completed": 0,
            "cached": 0,
            "errors": 0,
        }

        await self._update_status(results)
        logger.info(f"Starting pre-warm of {len(TOP_LEGAL_QUERIES)} legal queries...")

        try:
            # Process in batches
            for i in range(0, len(TOP_LEGAL_QUERIES), self.config.PREWARM_BATCH_SIZE):
                batch = TOP_LEGAL_QUERIES[i : i + self.config.PREWARM_BATCH_SIZE]

                for query in batch:
                    try:
                        # Check if already cached
                        existing = await self.vector_cache.get_cached_results(query)
                        if existing:
                            results["completed"] += 1
                            continue

                        # Query vector store if available
                        if vector_store:
                            search_results = vector_store.query(
                                query_text=query,
                                n_results=5,
                                namespace=None,  # Search all
                            )

                            # Cache the results
                            await self.vector_cache.cache_results(
                                query=query,
                                results=search_results,
                                namespace=None,
                                top_k=5,
                            )
                            results["cached"] += 1

                        results["completed"] += 1

                    except Exception as e:
                        logger.warning(f"Pre-warm error for '{query[:30]}...': {e}")
                        results["errors"] += 1
                        results["completed"] += 1

                # Update status periodically
                await self._update_status(results)

                # Small delay between batches to avoid overwhelming the system
                await asyncio.sleep(self.config.PREWARM_DELAY_BETWEEN_BATCHES)

            # Mark complete
            results["status"] = "complete"
            results["completed_at"] = datetime.utcnow().isoformat()
            results["duration_seconds"] = round(time.time() - start_time, 2)

            await self._update_status(results)

            logger.info(
                f"Pre-warm complete: {results['cached']} cached, "
                f"{results['errors']} errors, {results['duration_seconds']}s"
            )

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            await self._update_status(results)
            logger.error(f"Pre-warm failed: {e}")

        finally:
            self._is_warming = False

        return results


# =============================================================================
# CACHED KB SEARCH WRAPPER
# =============================================================================


class CachedKBSearch:
    """
    Wrapper around kb_search that adds caching layer.

    Usage:
        cached_search = CachedKBSearch()
        results = await cached_search.search("land rights Kenya")
    """

    def __init__(self):
        self._cache: Optional[RedisCache] = None
        self._vector_cache: Optional[VectorSearchCache] = None
        self._answer_cache: Optional[AnswerCache] = None
        self._kb_search = None

    async def initialize(self):
        """Initialize cache and search tool"""
        self._cache = await get_cache()
        self._vector_cache = VectorSearchCache(self._cache)
        self._answer_cache = AnswerCache(self._cache)

        try:
            from Module4_NiruAPI.agents.tools.kb_search import KnowledgeBaseSearchTool

            self._kb_search = KnowledgeBaseSearchTool()
        except ImportError:
            logger.warning("KnowledgeBaseSearchTool not available")

    async def search(
        self,
        query: str,
        top_k: int = 5,
        namespace: Optional[List[str]] = None,
        bypass_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        Search with caching layer.

        Args:
            query: Search query
            top_k: Number of results
            namespace: Optional namespace filter
            bypass_cache: Skip cache lookup

        Returns:
            Search results (from cache or fresh)
        """
        if not self._cache:
            await self.initialize()

        ns_str = ",".join(sorted(namespace)) if namespace else None

        # Check cache first
        if not bypass_cache:
            cached = await self._vector_cache.get_cached_results(query, ns_str)
            if cached:
                return {
                    "query": query,
                    "search_results": cached,
                    "from_cache": True,
                }

        # Execute fresh search
        if not self._kb_search:
            return {
                "query": query,
                "search_results": [],
                "error": "Search not available",
            }

        result = self._kb_search.execute(query=query, top_k=top_k, namespace=namespace)

        # Cache results
        search_results = result.get("search_results", [])
        if search_results:
            await self._vector_cache.cache_results(
                query=query, results=search_results, namespace=ns_str, top_k=top_k
            )

        result["from_cache"] = False
        return result

    async def get_cached_answer(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached complete answer"""
        if not self._cache:
            await self.initialize()
        return await self._answer_cache.get_cached_answer(query)

    async def cache_answer(
        self, query: str, answer: str, citations: Optional[List[Dict]] = None
    ) -> bool:
        """Cache complete answer"""
        if not self._cache:
            await self.initialize()
        return await self._answer_cache.cache_answer(query, answer, citations)


# =============================================================================
# STARTUP HOOK
# =============================================================================


async def startup_prewarm(vector_store=None):
    """
    Call this at application startup to pre-warm caches.

    Usage:
        from Module4_NiruAPI.agents.optimization import startup_prewarm

        @app.on_event("startup")
        async def startup():
            asyncio.create_task(startup_prewarm(vector_store))
    """
    try:
        cache = await get_cache()
        vector_cache = VectorSearchCache(cache)
        prewarm = PreWarmService(cache, vector_cache)

        # Run in background
        result = await prewarm.prewarm_cache(vector_store)
        logger.info(f"Pre-warm result: {result}")

    except Exception as e:
        logger.error(f"Startup pre-warm failed: {e}")


# =============================================================================
# MIDDLEWARE FOR AUTOMATIC CACHING
# =============================================================================


class CachingMiddleware:
    """
    Middleware to automatically cache queries and answers.

    Usage in LangGraph:
        middleware = CachingMiddleware()

        # Before processing
        cached = await middleware.check_cache(query)
        if cached:
            return cached

        # After processing
        await middleware.save_to_cache(query, response)
    """

    def __init__(self):
        self._answer_cache: Optional[AnswerCache] = None
        self._summarizer: Optional[ConversationSummarizer] = None

    async def initialize(self):
        """Initialize caches"""
        cache = await get_cache()
        self._answer_cache = AnswerCache(cache)
        self._summarizer = ConversationSummarizer(cache)

    async def check_answer_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if we have a cached answer"""
        if not self._answer_cache:
            await self.initialize()
        return await self._answer_cache.get_cached_answer(query)

    async def save_answer(
        self, query: str, answer: str, citations: Optional[List[Dict]] = None
    ):
        """Save answer to cache"""
        if not self._answer_cache:
            await self.initialize()
        await self._answer_cache.cache_answer(query, answer, citations)

    async def maybe_summarize(
        self, thread_id: str, messages: List[Dict], llm_client=None
    ) -> List[Dict]:
        """Summarize if needed, return compressed messages"""
        if not self._summarizer:
            await self.initialize()

        if await self._summarizer.should_summarize(messages):
            compressed, _ = await self._summarizer.summarize_and_compress(
                thread_id, messages, llm_client
            )
            return compressed

        return messages


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    "CacheConfig",
    "TOP_LEGAL_QUERIES",
    # Cache
    "RedisCache",
    "get_cache",
    # Caching services
    "AnswerCache",
    "VectorSearchCache",
    "ConversationSummarizer",
    "PreWarmService",
    # Wrappers
    "CachedKBSearch",
    "CachingMiddleware",
    # Startup
    "startup_prewarm",
]
