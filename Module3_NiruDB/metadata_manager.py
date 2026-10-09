"""
Metadata Manager - Query and manage document metadata
"""

from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, date
from loguru import logger
import json
import hashlib
from functools import lru_cache
import time


class MetadataManager:
    """Production-ready metadata manager with caching, batch operations, and cross-backend support"""

    def __init__(self, vector_store, redis_client=None, cache_ttl: int = 3600):
        """
        Initialize metadata manager

        Args:
            vector_store: VectorStore instance
            redis_client: Optional Redis client for caching
            cache_ttl: Cache TTL in seconds (default: 1 hour)
        """
        self.vector_store = vector_store
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl

        # Cache keys
        self.CACHE_CATEGORIES = "metadata:categories"
        self.CACHE_SOURCES = "metadata:sources"
        self.CACHE_STATS = "metadata:stats"

        logger.info(
            "MetadataManager initialized with caching and cross-backend support"
        )

    def _get_cache_key(self, operation: str, **params) -> str:
        """Generate cache key for operation"""
        param_str = json.dumps(params, sort_keys=True, default=str)
        return f"metadata:{operation}:{hashlib.md5(param_str.encode()).hexdigest()[:8]}"

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        try:
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Set value in cache"""
        if not self.redis_client:
            return
        try:
            self.redis_client.set(
                key, json.dumps(value, default=str), ex=self.cache_ttl
            )
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def _invalidate_cache(self, pattern: str = "*") -> None:
        """Invalidate cache keys matching pattern"""
        if not self.redis_client:
            return
        try:
            # Simple pattern matching - in production you'd use Redis SCAN
            keys_to_delete = []
            if pattern == "*":
                keys_to_delete = [
                    self.CACHE_CATEGORIES,
                    self.CACHE_SOURCES,
                    self.CACHE_STATS,
                ]
            elif "categories" in pattern:
                keys_to_delete = [self.CACHE_CATEGORIES]
            elif "sources" in pattern:
                keys_to_delete = [self.CACHE_SOURCES]

            for key in keys_to_delete:
                self.redis_client.delete(key)
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")

    def filter_by_category(
        self, category: str, limit: int = 100, namespace: Optional[str] = None
    ) -> List[Dict]:
        """
        Get documents by category with namespace support

        Args:
            category: Category to filter by
            limit: Maximum results to return
            namespace: Optional namespace filter

        Returns:
            List of document dictionaries
        """
        if not category or not category.strip():
            raise ValueError("Category cannot be empty")

        cache_key = self._get_cache_key(
            "filter_category", category=category, limit=limit, namespace=namespace
        )
        cached_result = self._get_cached(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for category filter: {category}")
            return cached_result

        try:
            # Use vector store query with filter across all namespaces if none specified
            namespaces = (
                [namespace]
                if namespace
                else [
                    "kenya_law",
                    "kenya_news",
                    "kenya_parliament",
                    "historical",
                    "global_trends",
                ]
            )

            all_results = []
            for ns in namespaces:
                results = self.vector_store.query(
                    query_text="",  # Empty query for metadata-only search
                    n_results=limit,
                    filter={"category": category},
                    namespace=ns,
                )
                all_results.extend(results)
                if len(all_results) >= limit:
                    break

            # Sort by relevance (if scores available) and limit
            if all_results and "distance" in all_results[0]:
                all_results.sort(key=lambda x: x.get("distance", float("inf")))

            results = all_results[:limit]
            self._set_cached(cache_key, results)
            logger.info(f"Found {len(results)} documents for category: {category}")
            return results

        except Exception as e:
            logger.error(f"Error filtering by category {category}: {e}")
            return []

    def filter_by_source(
        self, source_name: str, limit: int = 100, namespace: Optional[str] = None
    ) -> List[Dict]:
        """
        Get documents by source with namespace support

        Args:
            source_name: Source name to filter by
            limit: Maximum results to return
            namespace: Optional namespace filter

        Returns:
            List of document dictionaries
        """
        if not source_name or not source_name.strip():
            raise ValueError("Source name cannot be empty")

        cache_key = self._get_cache_key(
            "filter_source", source_name=source_name, limit=limit, namespace=namespace
        )
        cached_result = self._get_cached(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for source filter: {source_name}")
            return cached_result

        try:
            # Use vector store query with filter across all namespaces if none specified
            namespaces = (
                [namespace]
                if namespace
                else [
                    "kenya_law",
                    "kenya_news",
                    "kenya_parliament",
                    "historical",
                    "global_trends",
                ]
            )

            all_results = []
            for ns in namespaces:
                results = self.vector_store.query(
                    query_text="",  # Empty query for metadata-only search
                    n_results=limit,
                    filter={"source_name": source_name},
                    namespace=ns,
                )
                all_results.extend(results)
                if len(all_results) >= limit:
                    break

            # Sort by relevance and limit
            if all_results and "distance" in all_results[0]:
                all_results.sort(key=lambda x: x.get("distance", float("inf")))

            results = all_results[:limit]
            self._set_cached(cache_key, results)
            logger.info(f"Found {len(results)} documents for source: {source_name}")
            return results

        except Exception as e:
            logger.error(f"Error filtering by source {source_name}: {e}")
            return []

    def filter_by_date_range(
        self,
        start_date: str,
        end_date: str,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get documents within date range across all namespaces

        Args:
            start_date: ISO format date string (YYYY-MM-DD)
            end_date: ISO format date string (YYYY-MM-DD)
            limit: Maximum results to return
            namespace: Optional namespace filter

        Returns:
            List of document dictionaries sorted by date
        """
        try:
            # Parse and validate dates
            start = (
                datetime.fromisoformat(start_date).date()
                if isinstance(start_date, str)
                else start_date
            )
            end = (
                datetime.fromisoformat(end_date).date()
                if isinstance(end_date, str)
                else end_date
            )

            if start > end:
                raise ValueError("Start date cannot be after end date")

            cache_key = self._get_cache_key(
                "filter_date",
                start_date=str(start),
                end_date=str(end),
                limit=limit,
                namespace=namespace,
            )
            cached_result = self._get_cached(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for date range filter: {start} to {end}")
                return cached_result

        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            return []

        try:
            # Get sample documents from all relevant namespaces
            namespaces = (
                [namespace]
                if namespace
                else [
                    "kenya_law",
                    "kenya_news",
                    "kenya_parliament",
                    "historical",
                    "global_trends",
                ]
            )

            all_docs = []
            for ns in namespaces:
                try:
                    # Use sample documents for date filtering (more efficient than vector search)
                    sample_docs = self._get_sample_documents_from_namespace(
                        ns, limit=1000
                    )
                    all_docs.extend(sample_docs)
                except Exception as ns_error:
                    logger.warning(
                        f"Error getting documents from namespace {ns}: {ns_error}"
                    )
                    continue

            # Filter by date range
            filtered_docs = []
            for doc in all_docs:
                pub_date_str = doc.get("metadata", {}).get("publication_date", "")
                if pub_date_str:
                    try:
                        # Handle various date formats
                        if "T" in pub_date_str:
                            pub_date = datetime.fromisoformat(
                                pub_date_str.replace("Z", "+00:00")
                            ).date()
                        else:
                            pub_date = datetime.strptime(
                                pub_date_str, "%Y-%m-%d"
                            ).date()

                        if start <= pub_date <= end:
                            filtered_docs.append(doc)
                    except (ValueError, TypeError):
                        # Skip documents with invalid dates
                        continue

            # Sort by date (newest first) and limit
            filtered_docs.sort(
                key=lambda x: x.get("metadata", {}).get(
                    "publication_date", "1970-01-01"
                ),
                reverse=True,
            )

            results = filtered_docs[:limit]
            self._set_cached(cache_key, results)
            logger.info(
                f"Found {len(results)} documents in date range {start} to {end}"
            )
            return results

        except Exception as e:
            logger.error(f"Error filtering by date range: {e}")
            return []

    def _get_sample_documents_from_namespace(
        self, namespace: str, limit: int = 1000
    ) -> List[Dict]:
        """Get sample documents from a specific namespace"""
        try:
            # Temporarily modify collection name for namespace
            original_collection = self.vector_store.collection_name
            if self.vector_store.backend in ["qdrant", "chromadb"]:
                self.vector_store.collection_name = f"{original_collection}_{namespace}"

            sample_docs = self.vector_store.get_sample_documents(limit=limit)

            # Restore original collection name
            self.vector_store.collection_name = original_collection

            return sample_docs

        except Exception as e:
            logger.warning(
                f"Error getting sample documents from namespace {namespace}: {e}"
            )
            return []

    def get_categories(
        self, namespace: Optional[str] = None, force_refresh: bool = False
    ) -> List[str]:
        """
        Get list of all unique categories with optional namespace filtering

        Args:
            namespace: Optional namespace to filter categories
            force_refresh: Force refresh cache

        Returns:
            Sorted list of unique categories
        """
        cache_key = self._get_cache_key("categories", namespace=namespace)
        if not force_refresh:
            cached_result = self._get_cached(cache_key)
            if cached_result:
                logger.debug("Cache hit for categories")
                return cached_result

        try:
            categories = set()
            namespaces = (
                [namespace]
                if namespace
                else [
                    "kenya_law",
                    "kenya_news",
                    "kenya_parliament",
                    "historical",
                    "global_trends",
                ]
            )

            for ns in namespaces:
                try:
                    sample_docs = self._get_sample_documents_from_namespace(
                        ns, limit=2000
                    )
                    for doc in sample_docs:
                        category = doc.get("metadata", {}).get("category", "").strip()
                        if category and category != "Unknown":
                            categories.add(category)
                except Exception as ns_error:
                    logger.warning(
                        f"Error getting categories from namespace {ns}: {ns_error}"
                    )
                    continue

            result = sorted(list(categories))
            self._set_cached(cache_key, result)
            logger.info(f"Found {len(result)} unique categories")
            return result

        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return ["Unknown"]

    def get_sources(
        self, namespace: Optional[str] = None, force_refresh: bool = False
    ) -> List[str]:
        """
        Get list of all unique sources with optional namespace filtering

        Args:
            namespace: Optional namespace to filter sources
            force_refresh: Force refresh cache

        Returns:
            Sorted list of unique sources
        """
        cache_key = self._get_cache_key("sources", namespace=namespace)
        if not force_refresh:
            cached_result = self._get_cached(cache_key)
            if cached_result:
                logger.debug("Cache hit for sources")
                return cached_result

        try:
            sources = set()
            namespaces = (
                [namespace]
                if namespace
                else [
                    "kenya_law",
                    "kenya_news",
                    "kenya_parliament",
                    "historical",
                    "global_trends",
                ]
            )

            for ns in namespaces:
                try:
                    sample_docs = self._get_sample_documents_from_namespace(
                        ns, limit=2000
                    )
                    for doc in sample_docs:
                        source = doc.get("metadata", {}).get("source_name", "").strip()
                        if source and source != "Unknown":
                            sources.add(source)
                except Exception as ns_error:
                    logger.warning(
                        f"Error getting sources from namespace {ns}: {ns_error}"
                    )
                    continue

            result = sorted(list(sources))
            self._set_cached(cache_key, result)
            logger.info(f"Found {len(result)} unique sources")
            return result

        except Exception as e:
            logger.error(f"Error getting sources: {e}")
            return ["Unknown"]

    def get_citation(
        self, chunk_id: str, namespace: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get citation information for a chunk across all backends and namespaces

        Args:
            chunk_id: Chunk identifier
            namespace: Optional namespace hint

        Returns:
            Citation dictionary or None if not found
        """
        if not chunk_id or not chunk_id.strip():
            return None

        cache_key = self._get_cache_key(
            "citation", chunk_id=chunk_id, namespace=namespace
        )
        cached_result = self._get_cached(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for citation: {chunk_id}")
            return cached_result

        try:
            # Try different namespaces if none specified
            namespaces = (
                [namespace]
                if namespace
                else [
                    "kenya_law",
                    "kenya_news",
                    "kenya_parliament",
                    "historical",
                    "global_trends",
                ]
            )

            for ns in namespaces:
                try:
                    citation = self._get_citation_from_namespace(chunk_id, ns)
                    if citation:
                        self._set_cached(cache_key, citation)
                        return citation
                except Exception as ns_error:
                    logger.debug(f"Citation not found in namespace {ns}: {ns_error}")
                    continue

            logger.warning(f"Citation not found for chunk_id: {chunk_id}")
            return None

        except Exception as e:
            logger.error(f"Error getting citation for {chunk_id}: {e}")
            return None

    def _get_citation_from_namespace(
        self, chunk_id: str, namespace: str
    ) -> Optional[Dict]:
        """Get citation from a specific namespace"""
        try:
            if self.vector_store.backend == "chromadb":
                # Temporarily modify collection name
                original_collection = self.vector_store.collection_name
                self.vector_store.collection_name = f"{original_collection}_{namespace}"

                try:
                    result = self.vector_store.collection.get(
                        ids=[chunk_id], include=["metadatas", "documents"]
                    )

                    # Restore collection name
                    self.vector_store.collection_name = original_collection

                    if not result["ids"]:
                        return None

                    meta = result["metadatas"][0] if result["metadatas"] else {}

                except Exception as chroma_error:
                    # Restore collection name on error
                    self.vector_store.collection_name = original_collection
                    raise chroma_error

            elif self.vector_store.backend == "qdrant":
                # Use QDrant scroll with ID filter
                original_collection = self.vector_store.collection_name
                self.vector_store.collection_name = f"{original_collection}_{namespace}"

                try:
                    from qdrant_client.http import models

                    # Try to find by payload chunk_id first
                    scroll_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="chunk_id", match=models.MatchValue(value=chunk_id)
                            )
                        ]
                    )

                    scroll_result, _ = self.vector_store.client.scroll(
                        collection_name=self.vector_store.collection_name,
                        scroll_filter=scroll_filter,
                        limit=1,
                        with_payload=True,
                    )

                    # Restore collection name
                    self.vector_store.collection_name = original_collection

                    if not scroll_result:
                        return None

                    payload = (
                        scroll_result[0].payload
                        if hasattr(scroll_result[0], "payload")
                        else {}
                    )
                    meta = {
                        k: str(v) if not isinstance(v, str) else v
                        for k, v in payload.items()
                    }

                except Exception as qdrant_error:
                    # Restore collection name on error
                    self.vector_store.collection_name = original_collection
                    raise qdrant_error

            elif self.vector_store.backend == "upstash":
                # Upstash doesn't support direct ID lookup easily
                # We'll need to use metadata filtering
                filter_dict = {"metadata.chunk_id": chunk_id}
                if namespace:
                    filter_dict["metadata.namespace"] = namespace

                # Use a dummy vector for metadata-only query
                dummy_vector = [0.0] * 384
                results = self.vector_store.client.query(
                    vector=dummy_vector,
                    top_k=1,
                    filter=filter_dict,
                    include_metadata=True,
                )

                if not results:
                    return None

                meta = {
                    k: str(v) if not isinstance(v, str) else v
                    for k, v in results[0].metadata.items()
                }

            else:
                logger.warning(
                    f"get_citation not implemented for backend: {self.vector_store.backend}"
                )
                return None

            # Build citation from metadata
            citation = {
                "title": meta.get("title", "Untitled"),
                "source_url": meta.get("source_url", ""),
                "source_name": meta.get("source_name", "Unknown"),
                "author": meta.get("author", ""),
                "publication_date": meta.get("publication_date", ""),
                "category": meta.get("category", ""),
                "chunk_id": chunk_id,
                "namespace": namespace,
            }

            return citation

        except Exception as e:
            logger.debug(f"Error getting citation from namespace {namespace}: {e}")
            return None

    def format_citation(self, citation: Dict) -> str:
        """Format citation as a string with enhanced formatting"""
        if not citation:
            return "Citation unavailable"

        parts = []

        # Author (if available)
        if citation.get("author"):
            parts.append(citation["author"])

        # Title
        title = citation.get("title", "Untitled")
        if title != "Untitled":
            parts.append(f'"{title}"')
        else:
            parts.append(title)

        # Source
        source = citation.get("source_name", "Unknown Source")
        parts.append(source)

        # Date (if available)
        if citation.get("publication_date"):
            try:
                # Try to format date nicely
                date_str = citation["publication_date"]
                if "T" in date_str:
                    date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    parts.append(date_obj.strftime("%B %d, %Y"))
                else:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    parts.append(date_obj.strftime("%B %d, %Y"))
            except (ValueError, TypeError):
                parts.append(citation["publication_date"][:10])  # Just date part

        # URL (if available)
        if citation.get("source_url"):
            parts.append(citation["source_url"])

        # Category (if available and not generic)
        category = citation.get("category", "")
        if category and category not in ["", "Unknown"]:
            parts.append(f"[{category}]")

        return ". ".join(parts)

    def get_statistics(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive metadata statistics

        Args:
            namespace: Optional namespace filter

        Returns:
            Statistics dictionary
        """
        cache_key = self._get_cache_key("stats", namespace=namespace)
        cached_result = self._get_cached(cache_key)
        if cached_result:
            logger.debug("Cache hit for statistics")
            return cached_result

        try:
            stats = {
                "total_documents": 0,
                "categories": {},
                "sources": {},
                "date_range": {"oldest": None, "newest": None},
                "namespaces": {},
                "backend": self.vector_store.backend,
                "cached": self.redis_client is not None,
                "generated_at": datetime.utcnow().isoformat(),
            }

            namespaces = (
                [namespace]
                if namespace
                else [
                    "kenya_law",
                    "kenya_news",
                    "kenya_parliament",
                    "historical",
                    "global_trends",
                ]
            )

            all_dates = []

            for ns in namespaces:
                try:
                    sample_docs = self._get_sample_documents_from_namespace(
                        ns, limit=1000
                    )
                    ns_count = len(sample_docs)
                    stats["total_documents"] += ns_count
                    stats["namespaces"][ns] = ns_count

                    for doc in sample_docs:
                        meta = doc.get("metadata", {})

                        # Category stats
                        category = meta.get("category", "Unknown")
                        stats["categories"][category] = (
                            stats["categories"].get(category, 0) + 1
                        )

                        # Source stats
                        source = meta.get("source_name", "Unknown")
                        stats["sources"][source] = stats["sources"].get(source, 0) + 1

                        # Date range
                        pub_date = meta.get("publication_date", "")
                        if pub_date:
                            try:
                                if "T" in pub_date:
                                    date_obj = datetime.fromisoformat(
                                        pub_date.replace("Z", "+00:00")
                                    ).date()
                                else:
                                    date_obj = datetime.strptime(
                                        pub_date, "%Y-%m-%d"
                                    ).date()
                                all_dates.append(date_obj)
                            except (ValueError, TypeError):
                                continue

                except Exception as ns_error:
                    logger.warning(
                        f"Error getting stats from namespace {ns}: {ns_error}"
                    )
                    continue

            # Calculate date range
            if all_dates:
                stats["date_range"]["oldest"] = min(all_dates).isoformat()
                stats["date_range"]["newest"] = max(all_dates).isoformat()

            # Sort category and source stats
            stats["categories"] = dict(
                sorted(stats["categories"].items(), key=lambda x: x[1], reverse=True)
            )
            stats["sources"] = dict(
                sorted(stats["sources"].items(), key=lambda x: x[1], reverse=True)
            )

            self._set_cached(cache_key, stats)
            logger.info(
                f"Generated statistics for {stats['total_documents']} documents"
            )
            return stats

        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {"error": str(e), "generated_at": datetime.utcnow().isoformat()}

    def batch_get_citations(
        self, chunk_ids: List[str], namespace: Optional[str] = None
    ) -> Dict[str, Optional[Dict]]:
        """
        Batch get citations for multiple chunk IDs

        Args:
            chunk_ids: List of chunk identifiers
            namespace: Optional namespace hint

        Returns:
            Dictionary mapping chunk_id to citation or None
        """
        if not chunk_ids:
            return {}

        results = {}
        cache_hits = 0
        cache_misses = 0

        # Check cache first
        for chunk_id in chunk_ids:
            cache_key = self._get_cache_key(
                "citation", chunk_id=chunk_id, namespace=namespace
            )
            cached = self._get_cached(cache_key)
            if cached is not None:
                results[chunk_id] = cached
                cache_hits += 1
            else:
                results[chunk_id] = None  # Placeholder for cache miss
                cache_misses += 1

        # Get missing citations
        missing_ids = [cid for cid, citation in results.items() if citation is None]

        if missing_ids:
            logger.debug(
                f"Batch citation lookup: {cache_hits} hits, {cache_misses} misses"
            )

            # Try different namespaces
            namespaces = (
                [namespace]
                if namespace
                else [
                    "kenya_law",
                    "kenya_news",
                    "kenya_parliament",
                    "historical",
                    "global_trends",
                ]
            )

            for ns in namespaces:
                if not missing_ids:  # All found
                    break

                try:
                    found_citations = self._batch_get_citations_from_namespace(
                        missing_ids, ns
                    )
                    for chunk_id, citation in found_citations.items():
                        if citation:
                            results[chunk_id] = citation
                            # Cache the result
                            cache_key = self._get_cache_key(
                                "citation", chunk_id=chunk_id, namespace=namespace
                            )
                            self._set_cached(cache_key, citation)
                            missing_ids.remove(chunk_id)
                except Exception as ns_error:
                    logger.debug(
                        f"Error in batch citation lookup for namespace {ns}: {ns_error}"
                    )
                    continue

        logger.info(
            f"Batch citation lookup completed: {len(results)} total, {cache_hits} cached, {len([r for r in results.values() if r is not None]) - cache_hits} fetched"
        )
        return results

    def _batch_get_citations_from_namespace(
        self, chunk_ids: List[str], namespace: str
    ) -> Dict[str, Optional[Dict]]:
        """Batch get citations from a specific namespace"""
        results = {cid: None for cid in chunk_ids}

        try:
            if self.vector_store.backend == "chromadb":
                # Temporarily modify collection name
                original_collection = self.vector_store.collection_name
                self.vector_store.collection_name = f"{original_collection}_{namespace}"

                try:
                    result = self.vector_store.collection.get(
                        ids=chunk_ids, include=["metadatas", "documents"]
                    )

                    # Restore collection name
                    self.vector_store.collection_name = original_collection

                    if result["ids"] and result["metadatas"]:
                        for i, chunk_id in enumerate(result["ids"]):
                            meta = (
                                result["metadatas"][i]
                                if i < len(result["metadatas"])
                                else {}
                            )
                            results[chunk_id] = {
                                "title": meta.get("title", "Untitled"),
                                "source_url": meta.get("source_url", ""),
                                "source_name": meta.get("source_name", "Unknown"),
                                "author": meta.get("author", ""),
                                "publication_date": meta.get("publication_date", ""),
                                "category": meta.get("category", ""),
                                "chunk_id": chunk_id,
                                "namespace": namespace,
                            }

                except Exception as chroma_error:
                    # Restore collection name on error
                    self.vector_store.collection_name = original_collection
                    raise chroma_error

            elif self.vector_store.backend == "qdrant":
                # QDrant batch lookup using scroll with filters
                original_collection = self.vector_store.collection_name
                self.vector_store.collection_name = f"{original_collection}_{namespace}"

                try:
                    from qdrant_client.http import models

                    # Create filter for chunk_ids
                    conditions = [
                        models.FieldCondition(
                            key="chunk_id", match=models.MatchValue(value=chunk_id)
                        )
                        for chunk_id in chunk_ids
                    ]

                    # QDrant doesn't support OR conditions easily in scroll
                    # We'll do individual queries for each ID
                    for chunk_id in chunk_ids:
                        try:
                            scroll_filter = models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="chunk_id",
                                        match=models.MatchValue(value=chunk_id),
                                    )
                                ]
                            )

                            scroll_result, _ = self.vector_store.client.scroll(
                                collection_name=self.vector_store.collection_name,
                                scroll_filter=scroll_filter,
                                limit=1,
                                with_payload=True,
                            )

                            if scroll_result:
                                payload = (
                                    scroll_result[0].payload
                                    if hasattr(scroll_result[0], "payload")
                                    else {}
                                )
                                meta = {
                                    k: str(v) if not isinstance(v, str) else v
                                    for k, v in payload.items()
                                }
                                results[chunk_id] = {
                                    "title": meta.get("title", "Untitled"),
                                    "source_url": meta.get("source_url", ""),
                                    "source_name": meta.get("source_name", "Unknown"),
                                    "author": meta.get("author", ""),
                                    "publication_date": meta.get(
                                        "publication_date", ""
                                    ),
                                    "category": meta.get("category", ""),
                                    "chunk_id": chunk_id,
                                    "namespace": namespace,
                                }
                        except Exception as single_error:
                            logger.debug(
                                f"Error getting citation for {chunk_id}: {single_error}"
                            )
                            continue

                    # Restore collection name
                    self.vector_store.collection_name = original_collection

                except Exception as qdrant_error:
                    # Restore collection name on error
                    self.vector_store.collection_name = original_collection
                    raise qdrant_error

            # Upstash doesn't support efficient batch operations
            # We'll skip it for now to avoid performance issues

        except Exception as e:
            logger.debug(
                f"Error in batch citation lookup for namespace {namespace}: {e}"
            )

        return results

    def clear_cache(self, pattern: str = "*") -> int:
        """
        Clear cache entries matching pattern

        Args:
            pattern: Cache key pattern to clear

        Returns:
            Number of keys cleared
        """
        if not self.redis_client:
            logger.warning("No Redis client available for cache clearing")
            return 0

        try:
            if not self.redis_client:
                logger.warning("No Redis client available for cache clearing")
                return 0

            # Use Redis SCAN to count actual deletions
            deleted_count = 0
            cursor = 0
            pattern_str = f"metadata:{pattern}"
            while True:
                cursor, keys = self.redis_client.scan(
                    cursor=cursor, match=pattern_str, count=100
                )
                for key in keys:
                    try:
                        self.redis_client.delete(key)
                        deleted_count += 1
                    except Exception as del_error:
                        logger.warning(f"Error deleting cache key {key}: {del_error}")
                if cursor == 0:
                    break
            logger.info(f"Cleared {deleted_count} cache keys for pattern: {pattern}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on metadata manager

        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "vector_store": "unknown",
                "cache": "disabled",
                "namespaces": [],
            },
            "metrics": {"cache_hit_rate": 0.0, "avg_response_time": 0.0},
        }

        try:
            # Check vector store
            if self.vector_store:
                try:
                    stats = self.vector_store.get_stats()
                    health["components"]["vector_store"] = "healthy"
                    health["components"]["namespaces"] = list(
                        stats.get("sample_categories", {}).keys()
                    )[
                        :5
                    ]  # Sample
                except Exception as vs_error:
                    health["components"]["vector_store"] = f"unhealthy: {str(vs_error)}"
                    health["status"] = "degraded"

            # Check cache
            if self.redis_client:
                try:
                    self.redis_client.ping()
                    health["components"]["cache"] = "healthy"
                except Exception as cache_error:
                    health["components"]["cache"] = f"unhealthy: {str(cache_error)}"
                    if health["status"] == "healthy":
                        health["status"] = "degraded"

            # Test basic operations
            try:
                categories = self.get_categories()
                if categories:
                    health["components"]["namespaces"] = categories[:3]
            except Exception as op_error:
                logger.warning(f"Operation test failed: {op_error}")
                if health["status"] == "healthy":
                    health["status"] = "degraded"

        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)

        return health
