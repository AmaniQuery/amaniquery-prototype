"""
AmaniQuery v2.0 - Persona-Specific Retrieval Strategies
========================================================

Production-ready retrieval logic using VectorStore abstraction layer
for multi-backend support (Qdrant, ChromaDB, Upstash).

Optimized for three Kenyan civic AI personas:
- wanjiku: Hybrid search with recency boost (general citizens)
- wakili: Precision semantic search on legal documents (lawyers)
- mwanahabari: Keyword + metadata filtering for data/statistics (journalists)

Namespaces:
- kenya_law: Legal documents (constitution, acts, judgments, case law)
- kenya_parliament: Parliamentary records (bills, hansard, budget)
- kenya_news: News articles
- global_trends: Global content
- historical: Pre-2010 documents

Usage:
    from Module4_NiruAPI.agents.retrieval_strategies import (
        AmaniQueryRetriever,
        Namespace,
        PERSONA_NAMESPACES
    )
"""

from typing import List, Dict, Any, Optional, Literal, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# ============================================================================
# NAMESPACE DEFINITIONS
# ============================================================================


class Namespace(str, Enum):
    """Available namespaces in AmaniQuery vector store"""

    KENYA_LAW = "kenya_law"
    KENYA_PARLIAMENT = "kenya_parliament"
    KENYA_NEWS = "kenya_news"
    GLOBAL_TRENDS = "global_trends"
    HISTORICAL = "historical"


# Persona → Namespaces mapping (which collections each persona searches)
PERSONA_NAMESPACES: Dict[str, List[Namespace]] = {
    "wanjiku": [Namespace.KENYA_LAW, Namespace.KENYA_NEWS],
    "wakili": [Namespace.KENYA_LAW, Namespace.KENYA_PARLIAMENT],
    "mwanahabari": [
        Namespace.KENYA_PARLIAMENT,
        Namespace.KENYA_NEWS,
        Namespace.GLOBAL_TRENDS,
    ],
}

# Document type → Namespace mapping
DOCTYPE_NAMESPACE: Dict[str, Namespace] = {
    "constitution": Namespace.KENYA_LAW,
    "act": Namespace.KENYA_LAW,
    "bill": Namespace.KENYA_PARLIAMENT,
    "judgment": Namespace.KENYA_LAW,
    "case_law": Namespace.KENYA_LAW,
    "hansard": Namespace.KENYA_PARLIAMENT,
    "budget": Namespace.KENYA_PARLIAMENT,
    "news": Namespace.KENYA_NEWS,
    "global": Namespace.GLOBAL_TRENDS,
}

# Sheng/Swahili → English mapping for query normalization
SHENG_ENGLISH_MAP = {
    "kanjo": "nairobi city county",
    "bunge": "parliament",
    "mheshimiwa": "member of parliament MP",
    "doh": "money fees",
    "serikali": "government",
    "wabunge": "members of parliament MPs",
    "sheria": "law",
    "katiba": "constitution",
    "korti": "court",
    "hakimu": "magistrate judge",
    "wakili": "lawyer advocate",
    "polisi": "police",
    "askari": "officer guard",
    "raia": "citizen",
    "haki": "rights justice",
    "ushuru": "tax taxes",
    "kodi": "tax rent",
    "ardhi": "land",
    "mali": "property wealth",
    "biashara": "business trade",
    "kazi": "work employment job",
    "mishahara": "salary wages",
}


# ============================================================================
# RETRIEVAL RESULT MODELS
# ============================================================================


@dataclass
class RetrievalResult:
    """Standardized retrieval result"""

    id: str
    text: str
    score: float
    source: str
    namespace: str
    doc_type: str = ""
    citation: str = ""
    date_published: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations"""

    limit: int = 8
    score_threshold: float = 0.5
    recency_boost: float = 1.2
    explainer_boost: float = 1.15
    diversity: float = 0.3
    include_historical: bool = False


# ============================================================================
# MAIN RETRIEVER CLASS
# ============================================================================


class AmaniQueryRetriever:
    """
    Main retriever for AmaniQuery using VectorStore abstraction.

    Supports:
    - Multi-namespace queries (searches across multiple collections)
    - Persona-specific retrieval strategies
    - Automatic score boosting (recency, explainer tags)
    - Query normalization (Sheng/Swahili → English)

    Example:
        from Module3_NiruDB.vector_store import VectorStore

        vs = VectorStore(backend="qdrant")
        retriever = AmaniQueryRetriever(vs)

        results = retriever.retrieve_wanjiku("What are my rights during arrest?")
    """

    def __init__(
        self,
        vector_store,  # VectorStore instance
        default_config: Optional[RetrievalConfig] = None,
    ):
        """
        Initialize retriever with VectorStore.

        Args:
            vector_store: Initialized VectorStore instance
            default_config: Default retrieval configuration
        """
        self.vector_store = vector_store
        self.config = default_config or RetrievalConfig()
        self._executor = ThreadPoolExecutor(max_workers=4)

        logger.info(
            f"AmaniQueryRetriever initialized with backend: {vector_store.backend}"
        )

    # ========================================================================
    # WANJIKU RETRIEVAL (General Citizen)
    # ========================================================================

    def retrieve_wanjiku(
        self,
        query: str,
        limit: int = 8,
        recency_months: int = 6,
        namespaces: Optional[List[Namespace]] = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        """
        Wanjiku retrieval: Citizen-friendly search with recency boost.

        Strategy:
        - Search kenya_law + kenya_news namespaces
        - Normalize Sheng/Swahili queries
        - Boost recent documents (< 6 months)
        - Prioritize explainer/summary content

        Args:
            query: User query (can be Sheng/Swahili)
            limit: Max results to return
            recency_months: Boost docs within this timeframe
            namespaces: Override default namespaces
            filter_dict: Additional metadata filters

        Returns:
            List of RetrievalResult objects
        """
        # Use persona defaults if namespaces not specified
        if namespaces is None:
            namespaces = PERSONA_NAMESPACES["wanjiku"]

        # Normalize query (Sheng → English)
        normalized_query = self._normalize_query(query)
        logger.debug(f"Wanjiku query normalized: '{query}' → '{normalized_query}'")

        # Calculate recency cutoff
        recency_cutoff = datetime.now() - timedelta(days=recency_months * 30)

        # Search across namespaces
        all_results = []
        for namespace in namespaces:
            try:
                results = self._search_namespace(
                    query=normalized_query,
                    namespace=namespace.value,
                    limit=limit * 2,  # Get more for post-processing
                    filter_dict=filter_dict,
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Failed to search namespace {namespace.value}: {e}")

        # Apply boosts
        boosted_results = self._apply_wanjiku_boosts(
            all_results,
            recency_cutoff,
            recency_boost=self.config.recency_boost,
            explainer_boost=self.config.explainer_boost,
        )

        # Sort by boosted score and return top results
        boosted_results.sort(key=lambda x: x.score, reverse=True)
        return boosted_results[:limit]

    # ========================================================================
    # WAKILI RETRIEVAL (Legal Professional)
    # ========================================================================

    def retrieve_wakili(
        self,
        query: str,
        limit: int = 6,
        doc_types: Optional[List[str]] = None,
        namespaces: Optional[List[Namespace]] = None,
        include_historical: bool = True,
    ) -> List[RetrievalResult]:
        """
        Wakili retrieval: High-precision legal document search.

        Strategy:
        - Search kenya_law + kenya_parliament namespaces
        - Filter by legal doc_types (act, bill, judgment, constitution)
        - Include historical (pre-2010) for legal precedents
        - Format citations properly

        Args:
            query: Legal query (formal language preferred)
            limit: Max results to return
            doc_types: Filter by document types
            namespaces: Override default namespaces
            include_historical: Include pre-2010 documents

        Returns:
            List of RetrievalResult objects with citations
        """
        if namespaces is None:
            namespaces = PERSONA_NAMESPACES["wakili"]

        if doc_types is None:
            doc_types = ["constitution", "act", "bill", "judgment", "case_law"]

        # Add historical namespace if requested
        search_namespaces = list(namespaces)
        if include_historical and Namespace.HISTORICAL not in search_namespaces:
            search_namespaces.append(Namespace.HISTORICAL)

        # Build filter for legal doc types
        filter_dict = {"category": {"$in": doc_types}} if doc_types else None

        # Search across namespaces
        all_results = []
        for namespace in search_namespaces:
            try:
                results = self._search_namespace(
                    query=query,
                    namespace=namespace.value,
                    limit=limit,
                    filter_dict=filter_dict,
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Failed to search namespace {namespace.value}: {e}")

        # Add citations to results
        for result in all_results:
            result.citation = self._build_citation(result)

        # Sort by score and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:limit]

    # ========================================================================
    # MWANAHABARI RETRIEVAL (Journalist/Data Analyst)
    # ========================================================================

    def retrieve_mwanahabari(
        self,
        query: str,
        limit: int = 12,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        mp_name: Optional[str] = None,
        committee: Optional[str] = None,
        namespaces: Optional[List[Namespace]] = None,
    ) -> List[RetrievalResult]:
        """
        Mwanahabari retrieval: Data-focused search for journalists.

        Strategy:
        - Search kenya_parliament + kenya_news + global_trends namespaces
        - Strong metadata filtering (dates, MPs, committees)
        - Prioritize documents with tables/statistics

        Args:
            query: Data-focused query
            limit: Max results to return
            date_from: Filter docs after this date
            date_to: Filter docs before this date
            mp_name: Filter by MP name
            committee: Filter by parliamentary committee
            namespaces: Override default namespaces

        Returns:
            List of data-rich RetrievalResult objects
        """
        if namespaces is None:
            namespaces = PERSONA_NAMESPACES["mwanahabari"]

        # Build complex filter
        filter_dict = {}

        if mp_name:
            filter_dict["mp_name"] = mp_name
        if committee:
            filter_dict["committee"] = committee

        # Search across namespaces
        all_results = []
        for namespace in namespaces:
            try:
                results = self._search_namespace(
                    query=query,
                    namespace=namespace.value,
                    limit=limit,
                    filter_dict=filter_dict if filter_dict else None,
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Failed to search namespace {namespace.value}: {e}")

        # Apply date filtering (post-search since VectorStore doesn't support range)
        if date_from or date_to:
            all_results = self._filter_by_date_range(all_results, date_from, date_to)

        # Sort by score and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:limit]

    # ========================================================================
    # UNIFIED RETRIEVE METHOD
    # ========================================================================

    def retrieve(
        self,
        query: str,
        persona: Literal["wanjiku", "wakili", "mwanahabari"] = "wanjiku",
        limit: int = 8,
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        Unified retrieval method that routes to persona-specific strategy.

        Args:
            query: Search query
            persona: User persona (wanjiku, wakili, mwanahabari)
            limit: Max results
            **kwargs: Persona-specific arguments

        Returns:
            List of RetrievalResult objects
        """
        if persona == "wanjiku":
            return self.retrieve_wanjiku(query, limit=limit, **kwargs)
        elif persona == "wakili":
            return self.retrieve_wakili(query, limit=limit, **kwargs)
        elif persona == "mwanahabari":
            return self.retrieve_mwanahabari(query, limit=limit, **kwargs)
        else:
            raise ValueError(f"Unknown persona: {persona}")

    # ========================================================================
    # ASYNC RETRIEVAL METHODS
    # ========================================================================

    async def aretrieve(
        self,
        query: str,
        persona: Literal["wanjiku", "wakili", "mwanahabari"] = "wanjiku",
        limit: int = 8,
        **kwargs,
    ) -> List[RetrievalResult]:
        """Async version of retrieve"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, lambda: self.retrieve(query, persona, limit, **kwargs)
        )

    async def aretrieve_multi_persona(
        self,
        query: str,
        personas: List[str] = ["wanjiku", "wakili"],
        limit_per_persona: int = 4,
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieve from multiple personas in parallel.

        Useful for comprehensive research queries.
        """
        tasks = [
            self.aretrieve(
                query,
                persona=p if p in ["wanjiku", "wakili", "mwanahabari"] else "wanjiku",
                limit=limit_per_persona,
            )
            for p in personas
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            persona: result if isinstance(result, list) else []
            for persona, result in zip(personas, results)
        }

    # ========================================================================
    # INTERNAL HELPER METHODS
    # ========================================================================

    def _search_namespace(
        self, query: str, namespace: str, limit: int, filter_dict: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Search a single namespace using VectorStore.query()

        Args:
            query: Search query text
            namespace: Namespace to search
            limit: Max results
            filter_dict: Metadata filters

        Returns:
            List of RetrievalResult objects
        """
        try:
            # Use VectorStore.query() with namespace
            raw_results = self.vector_store.query(
                query_text=query,
                n_results=limit,
                filter=filter_dict,
                namespace=namespace,
            )

            # Convert to RetrievalResult objects
            results = []
            for r in raw_results:
                metadata = r.get("metadata", {})
                results.append(
                    RetrievalResult(
                        id=r.get("id", ""),
                        text=r.get("text", ""),
                        score=r.get("distance", 0.0),  # VectorStore returns distance
                        source=metadata.get(
                            "source_name", metadata.get("source_url", "")
                        ),
                        namespace=namespace,
                        doc_type=metadata.get("category", ""),
                        date_published=metadata.get("publication_date"),
                        metadata=metadata,
                    )
                )

            logger.debug(f"Namespace {namespace}: found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching namespace {namespace}: {e}")
            return []

    def _normalize_query(self, query: str) -> str:
        """
        Normalize Sheng/Swahili query to English for better retrieval.
        """
        normalized = query.lower()
        for sheng, english in SHENG_ENGLISH_MAP.items():
            if sheng in normalized:
                normalized = normalized.replace(sheng, english)
        return normalized

    def _apply_wanjiku_boosts(
        self,
        results: List[RetrievalResult],
        recency_cutoff: datetime,
        recency_boost: float = 1.2,
        explainer_boost: float = 1.15,
    ) -> List[RetrievalResult]:
        """
        Apply score boosts for Wanjiku persona.

        - Recent documents get recency_boost
        - Explainer/summary content gets explainer_boost
        """
        for result in results:
            boost = 1.0

            # Recency boost
            if result.date_published:
                try:
                    # Parse date string
                    doc_date = self._parse_date(result.date_published)
                    if doc_date and doc_date >= recency_cutoff:
                        boost *= recency_boost
                except Exception:
                    pass

            # Explainer boost
            tags = result.metadata.get("metadata_tags", [])
            if isinstance(tags, list) and any(
                t in tags for t in ["explainer", "summary", "guide"]
            ):
                boost *= explainer_boost

            # Also check title/category for explainer content
            title = result.metadata.get("title", "").lower()
            if any(
                word in title for word in ["explained", "guide", "how to", "what is"]
            ):
                boost *= 1.1

            result.score *= boost

        return results

    def _filter_by_date_range(
        self,
        results: List[RetrievalResult],
        date_from: Optional[datetime],
        date_to: Optional[datetime],
    ) -> List[RetrievalResult]:
        """Filter results by date range"""
        filtered = []
        for result in results:
            if not result.date_published:
                continue

            doc_date = self._parse_date(result.date_published)
            if not doc_date:
                continue

            if date_from and doc_date < date_from:
                continue
            if date_to and doc_date > date_to:
                continue

            filtered.append(result)

        return filtered

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_str:
            return None

        import re

        # Try ISO format first
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00").split("T")[0])
        except ValueError:
            pass

        # Try extracting year
        year_match = re.search(r"(\d{4})", date_str)
        if year_match:
            try:
                return datetime(int(year_match.group(1)), 1, 1)
            except ValueError:
                pass

        return None

    def _build_citation(self, result: RetrievalResult) -> str:
        """Build proper legal citation from result"""
        doc_type = result.doc_type.lower()
        source = result.source
        metadata = result.metadata

        if doc_type == "constitution":
            article = metadata.get("article_number", metadata.get("section_number", ""))
            if article:
                return f"Constitution of Kenya, 2010, Article {article}"
            return "Constitution of Kenya, 2010"

        elif doc_type in ["act", "legislation"]:
            section = metadata.get("section_number", "")
            title = metadata.get("title", source)
            if section:
                return f"{title}, Section {section}"
            return title

        elif doc_type == "bill":
            return f"{metadata.get('title', source)} (Bill)"

        elif doc_type in ["judgment", "case_law"]:
            return source or metadata.get("title", "Case Law")

        elif doc_type == "hansard":
            date = metadata.get("publication_date", "")
            return f"Kenya Hansard, {date}"

        else:
            return source or metadata.get("title", "")


# ============================================================================
# CONVENIENCE FACTORY FUNCTION
# ============================================================================


def create_retriever(
    backend: str = "auto", collection_name: str = "amaniquery_docs", config_manager=None
) -> AmaniQueryRetriever:
    """
    Factory function to create retriever with VectorStore.

    Args:
        backend: Vector store backend ('auto', 'qdrant', 'chromadb', 'upstash')
        collection_name: Collection name
        config_manager: Optional ConfigManager for credentials

    Returns:
        Configured AmaniQueryRetriever instance
    """
    from Module3_NiruDB.vector_store import VectorStore

    vector_store = VectorStore(
        backend=backend, collection_name=collection_name, config_manager=config_manager
    )

    return AmaniQueryRetriever(vector_store)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RETRIEVAL STRATEGIES - USAGE EXAMPLES")
    print("=" * 80)

    print("""
    # Basic Usage
    from Module4_NiruAPI.agents.retrieval_strategies import create_retriever
    
    # Create retriever (auto-detects backend)
    retriever = create_retriever()
    
    # Wanjiku: General citizen queries
    results = retriever.retrieve_wanjiku(
        query="What are my rights during arrest?",
        limit=8
    )
    
    # Wakili: Legal professional queries
    results = retriever.retrieve_wakili(
        query="Section 23 of Employment Act termination provisions",
        limit=6,
        doc_types=["act", "judgment"]
    )
    
    # Mwanahabari: Journalist/data queries
    results = retriever.retrieve_mwanahabari(
        query="Budget allocation health sector 2024",
        mp_name="Opiyo Wandayi",
        limit=12
    )
    
    # Unified interface
    results = retriever.retrieve(
        query="Land registration process",
        persona="wanjiku",
        limit=8
    )
    
    # Async multi-persona search
    import asyncio
    
    async def search():
        results = await retriever.aretrieve_multi_persona(
            query="Constitutional amendment procedures",
            personas=["wanjiku", "wakili"],
            limit_per_persona=4
        )
        return results
    
    all_results = asyncio.run(search())
    """)

    print("\n" + "=" * 80)
    print("NAMESPACES AVAILABLE:")
    for ns in Namespace:
        print(f"  - {ns.value}")

    print("\nPERSONA NAMESPACE MAPPINGS:")
    for persona, namespaces in PERSONA_NAMESPACES.items():
        ns_list = [ns.value for ns in namespaces]
        print(f"  - {persona}: {ns_list}")

    print("=" * 80)
