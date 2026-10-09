"""
Adaptive RAG Pipeline - AI-Powered Self-Improving System
Advanced task identification, predictive retrieval, and continuous learning with real-time optimization
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from loguru import logger
from collections import defaultdict, Counter
import asyncio
import concurrent.futures
from dataclasses import dataclass
from typing import AsyncGenerator

from Module3_NiruDB.vector_store import VectorStore
from Module3_NiruDB.metadata_manager import MetadataManager
from Module4_NiruAPI.cache import RAGCache, get_rag_cache, BlazingFastCache


@dataclass
class AdaptiveQuery:
    """Enhanced query with adaptive metadata"""

    query: str
    intent: str
    confidence: float
    keywords: List[str]
    semantic_embedding: np.ndarray
    predicted_task_type: str
    complexity_score: float
    timestamp: datetime
    metadata: Dict = None


@dataclass
class RetrievalStrategy:
    """Optimized retrieval strategy"""

    task_type: str
    namespaces: List[str]
    top_k: int
    rerank: bool
    use_expansion: bool
    timeout: float
    fallback_strategy: str


class TaskCluster:
    """Enhanced task cluster with performance metrics"""

    def __init__(
        self,
        cluster_id: str,
        name: str,
        description: str,
        keywords: List[str],
        sample_queries: List[str],
        document_ids: List[str],
        avg_response_time: float = 0.0,
        success_rate: float = 0.0,
        popularity_score: float = 0.0,
    ):
        self.cluster_id = cluster_id
        self.name = name
        self.description = description
        self.keywords = keywords
        self.sample_queries = sample_queries
        self.document_ids = document_ids
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.query_count = len(sample_queries)
        self.avg_response_time = avg_response_time
        self.success_rate = success_rate
        self.popularity_score = popularity_score
        self.performance_history = []  # Track response times
        self.last_used = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "sample_queries": self.sample_queries,
            "document_ids": self.document_ids,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "query_count": self.query_count,
            "avg_response_time": self.avg_response_time,
            "success_rate": self.success_rate,
            "popularity_score": self.popularity_score,
            "performance_history": self.performance_history[-10:],  # Last 10 entries
            "last_used": self.last_used.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TaskCluster":
        cluster = cls(
            cluster_id=data["cluster_id"],
            name=data["name"],
            description=data["description"],
            keywords=data["keywords"],
            sample_queries=data["sample_queries"],
            document_ids=data["document_ids"],
        )
        cluster.created_at = datetime.fromisoformat(data["created_at"])
        cluster.updated_at = datetime.fromisoformat(data["updated_at"])
        cluster.query_count = data["query_count"]
        return cluster


class BlazingFastTaskIdentificationEngine:
    """🔥 High-performance task identification with predictive capabilities"""

    def __init__(self, log_directory: str = "logs", min_cluster_size: int = 5):
        self.log_directory = Path(log_directory)
        self.min_cluster_size = min_cluster_size
        self.task_clusters: Dict[str, TaskCluster] = {}
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )
        self.cluster_model = None
        self.load_existing_clusters()

    def load_existing_clusters(self):
        """Load existing task clusters from storage"""
        cluster_file = Path("data/task_clusters.json")
        if cluster_file.exists():
            try:
                with open(cluster_file, "r") as f:
                    data = json.load(f)
                    for cluster_data in data.get("clusters", []):
                        cluster = TaskCluster.from_dict(cluster_data)
                        self.task_clusters[cluster.cluster_id] = cluster
                logger.info(f"Loaded {len(self.task_clusters)} existing task clusters")
            except Exception as e:
                logger.error(f"Failed to load task clusters: {e}")

    def save_clusters(self):
        """Save task clusters to storage"""
        cluster_file = Path("data/task_clusters.json")
        cluster_file.parent.mkdir(exist_ok=True)

        data = {
            "clusters": [cluster.to_dict() for cluster in self.task_clusters.values()],
            "last_updated": datetime.now().isoformat(),
        }

        with open(cluster_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.task_clusters)} task clusters")

    def extract_queries_from_logs(self, days_back: int = 7) -> List[Dict]:
        """Extract user queries from recent log files"""
        queries = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        # Find all log files
        log_files = list(self.log_directory.glob("*.log"))
        log_files.extend(self.log_directory.glob("**/*.log"))

        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        # Look for query-related log entries
                        if (
                            "Retrieving documents for query:" in line
                            or "Incoming SMS" in line
                        ):
                            try:
                                # Parse timestamp and extract query
                                parts = line.split("|")
                                if len(parts) >= 3:
                                    timestamp_str = parts[0].strip()
                                    timestamp = datetime.strptime(
                                        timestamp_str, "%Y-%m-%d %H:%M:%S.%f"
                                    )

                                    if timestamp >= cutoff_date:
                                        # Extract query text
                                        if "Retrieving documents for query:" in line:
                                            query_start = (
                                                line.find(
                                                    "Retrieving documents for query:"
                                                )
                                                + 32
                                            )
                                            query_text = (
                                                line[query_start:].strip().rstrip("...")
                                            )
                                        elif "Incoming SMS" in line:
                                            query_start = line.find("Incoming SMS") + 12
                                            query_text = (
                                                line[query_start:]
                                                .split(":", 1)[-1]
                                                .strip()
                                            )
                                        else:
                                            continue

                                        queries.append(
                                            {
                                                "timestamp": timestamp,
                                                "query": query_text,
                                                "source": (
                                                    "api"
                                                    if "Retrieving documents" in line
                                                    else "sms"
                                                ),
                                            }
                                        )
                            except Exception as e:
                                continue
            except Exception as e:
                logger.warning(f"Failed to process log file {log_file}: {e}")

        logger.info(f"Extracted {len(queries)} queries from logs")
        return queries

    def analyze_query_intents(self, queries: List[Dict]) -> List[Dict]:
        """Analyze queries to extract intents and key concepts using LLM"""
        analyzed_queries = []

        # Group queries for batch processing
        query_texts = [q["query"] for q in queries]

        # Use LLM to analyze intents (simplified version - in practice would use actual LLM)
        for i, query_data in enumerate(queries):
            query_text = query_data["query"]

            # Simple keyword-based intent detection (replace with LLM analysis)
            intent = self._classify_intent_simple(query_text)
            keywords = self._extract_keywords_simple(query_text)

            analyzed_queries.append(
                {
                    **query_data,
                    "intent": intent,
                    "keywords": keywords,
                    "processed_text": query_text.lower(),
                }
            )

        return analyzed_queries

    def _classify_intent_simple(self, query: str) -> str:
        """Simple rule-based intent classification"""
        query_lower = query.lower()

        # Legal intents
        if any(
            word in query_lower
            for word in ["law", "constitution", "act", "bill", "court", "judge"]
        ):
            return "legal_analysis"

        # News/Current affairs
        if any(
            word in query_lower
            for word in ["news", "current", "recent", "today", "latest"]
        ):
            return "news_inquiry"

        # Parliament
        if any(
            word in query_lower
            for word in ["parliament", "mp", "debate", "budget", "vote"]
        ):
            return "parliamentary_inquiry"

        # Sentiment/Public opinion
        if any(
            word in query_lower
            for word in ["sentiment", "public opinion", "people think", "popular"]
        ):
            return "sentiment_analysis"

        # General information
        return "general_inquiry"

    def _extract_keywords_simple(self, query: str) -> List[str]:
        """Simple keyword extraction"""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "what",
            "is",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "how",
            "why",
            "when",
            "where",
            "who",
        }
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Limit to top 10 keywords

    def cluster_queries(
        self, analyzed_queries: List[Dict], n_clusters: int = None
    ) -> Dict[str, List[Dict]]:
        """Cluster similar queries using TF-IDF and K-means"""
        if len(analyzed_queries) < self.min_cluster_size:
            logger.warning(
                f"Not enough queries for clustering: {len(analyzed_queries)} < {self.min_cluster_size}"
            )
            return {}

        # Prepare text data for clustering
        texts = [q["processed_text"] for q in analyzed_queries]

        # Create TF-IDF matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
        except Exception as e:
            logger.error(f"Failed to create TF-IDF matrix: {e}")
            return {}

        # Determine number of clusters
        if n_clusters is None:
            n_clusters = min(len(analyzed_queries) // 3, 10)  # Adaptive clustering
            n_clusters = max(n_clusters, 2)  # At least 2 clusters

        # Perform clustering
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.cluster_model.fit_predict(tfidf_matrix.toarray())

        # Group queries by cluster
        clustered_queries = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            clustered_queries[str(cluster_id)].append(analyzed_queries[i])

        logger.info(
            f"Clustered {len(analyzed_queries)} queries into {n_clusters} groups"
        )
        return dict(clustered_queries)

    def create_task_clusters(
        self, clustered_queries: Dict[str, List[Dict]]
    ) -> List[TaskCluster]:
        """Create TaskCluster objects from clustered queries"""
        task_clusters = []

        for cluster_id, queries in clustered_queries.items():
            if len(queries) < self.min_cluster_size:
                continue

            # Analyze cluster characteristics
            intents = Counter(q["intent"] for q in queries)
            dominant_intent = intents.most_common(1)[0][0]

            # Collect all keywords
            all_keywords = []
            for q in queries:
                all_keywords.extend(q["keywords"])
            keyword_counts = Counter(all_keywords)
            top_keywords = [k for k, v in keyword_counts.most_common(10)]

            # Sample queries (up to 5)
            sample_queries = [q["query"] for q in queries[:5]]

            # Generate cluster name and description
            cluster_name = self._generate_cluster_name(dominant_intent, top_keywords)
            description = self._generate_cluster_description(dominant_intent, queries)

            # For now, document_ids will be populated during retrieval phase
            document_ids = []

            cluster = TaskCluster(
                cluster_id=f"cluster_{cluster_id}",
                name=cluster_name,
                description=description,
                keywords=top_keywords,
                sample_queries=sample_queries,
                document_ids=document_ids,
            )

            task_clusters.append(cluster)

        return task_clusters

    def _generate_cluster_name(self, intent: str, keywords: List[str]) -> str:
        """Generate a human-readable name for the cluster"""
        intent_names = {
            "legal_analysis": "Legal Analysis",
            "news_inquiry": "News & Current Affairs",
            "parliamentary_inquiry": "Parliamentary Affairs",
            "sentiment_analysis": "Public Sentiment",
            "general_inquiry": "General Information",
        }

        base_name = intent_names.get(intent, "General Inquiry")
        if keywords:
            base_name += f" - {keywords[0].title()}"

        return base_name

    def _generate_cluster_description(self, intent: str, queries: List[Dict]) -> str:
        """Generate description for the cluster"""
        intent_descriptions = {
            "legal_analysis": "Queries related to Kenyan laws, constitution, court cases, and legal matters",
            "news_inquiry": "Questions about current events, recent news, and ongoing developments",
            "parliamentary_inquiry": "Inquiries about parliamentary proceedings, bills, debates, and government actions",
            "sentiment_analysis": "Analysis of public opinion and sentiment on various topics",
            "general_inquiry": "General questions and information requests",
        }

        return intent_descriptions.get(intent, "Various types of information requests")

    def update_task_clusters(self, new_clusters: List[TaskCluster]):
        """Update existing task clusters with new ones"""
        for cluster in new_clusters:
            if cluster.cluster_id in self.task_clusters:
                # Update existing cluster
                existing = self.task_clusters[cluster.cluster_id]
                existing.sample_queries.extend(cluster.sample_queries)
                existing.query_count += cluster.query_count
                existing.keywords = list(set(existing.keywords + cluster.keywords))
                existing.updated_at = datetime.now()
            else:
                # Add new cluster
                self.task_clusters[cluster.cluster_id] = cluster

        self.save_clusters()

    def identify_task_group(self, query: str) -> Optional[str]:
        """Identify which task group a new query belongs to"""
        if not self.task_clusters:
            return None

        # Preprocess query
        processed_query = query.lower()
        query_keywords = self._extract_keywords_simple(query)

        # Find best matching cluster
        best_match = None
        best_score = 0

        for cluster in self.task_clusters.values():
            # Calculate similarity score based on keyword overlap and intent
            keyword_overlap = len(set(query_keywords) & set(cluster.keywords))
            intent_match = (
                1
                if self._classify_intent_simple(query)
                == cluster.name.split(" - ")[0].lower().replace(" ", "_")
                else 0
            )

            score = keyword_overlap * 0.7 + intent_match * 0.3

            if score > best_score:
                best_score = score
                best_match = cluster.cluster_id

        return best_match if best_score > 0.3 else None  # Threshold for matching

    def run_task_identification(self, days_back: int = 7):
        """Run the complete task identification pipeline"""
        logger.info("Starting task identification process...")

        # Extract queries from logs
        queries = self.extract_queries_from_logs(days_back)

        if not queries:
            logger.warning("No queries found in logs")
            return

        # Analyze query intents
        analyzed_queries = self.analyze_query_intents(queries)

        # Cluster queries
        clustered_queries = self.cluster_queries(analyzed_queries)

        if not clustered_queries:
            logger.warning("No clusters created")
            return

        # Create task clusters
        new_task_clusters = self.create_task_clusters(clustered_queries)

        # Update existing clusters
        self.update_task_clusters(new_task_clusters)

        logger.info(
            f"Task identification completed. Created/updated {len(new_task_clusters)} clusters"
        )


class AdaptiveRetrievalEngine:
    """Engine for adaptive retrieval using task groups"""

    def __init__(
        self,
        vector_store: VectorStore,
        task_engine: BlazingFastTaskIdentificationEngine,
    ):
        self.vector_store = vector_store
        self.task_engine = task_engine
        self.semantic_cache = {}
        self.cache_max_size = 500

    def retrieve_adaptive(
        self, query: str, top_k: int = 5, task_group: Optional[str] = None
    ) -> List[Dict]:
        """Perform adaptive retrieval using task groups and semantic caching"""

        # Check semantic cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self.semantic_cache:
            logger.info("Semantic cache hit!")
            return self.semantic_cache[cache_key]

        # Identify task group if not provided
        if task_group is None:
            task_group = self.task_engine.identify_task_group(query)

        retrieved_docs = []

        if task_group and task_group in self.task_engine.task_clusters:
            cluster = self.task_engine.task_clusters[task_group]

            # Targeted retrieval within task group documents
            if cluster.document_ids:
                # Query specific documents in the cluster
                cluster_docs = self.vector_store.query_by_ids(
                    document_ids=cluster.document_ids, query_text=query, n_results=top_k
                )
                retrieved_docs.extend(cluster_docs)
                logger.info(
                    f"Retrieved {len(cluster_docs)} documents from task group {task_group}"
                )

            # Also do broader search with task group keywords as context
            keyword_filter = " ".join(cluster.keywords[:5])
            enhanced_query = f"{query} {keyword_filter}"

            broader_docs = self.vector_store.query(
                query_text=enhanced_query,
                n_results=top_k // 2,
                filter={"category": cluster.name.split(" - ")[0].lower()},
            )
            retrieved_docs.extend(broader_docs)
            logger.info(
                f"Retrieved {len(broader_docs)} additional documents with enhanced query"
            )

        else:
            # Fallback to standard retrieval
            retrieved_docs = self.vector_store.query(query_text=query, n_results=top_k)
            logger.info("Used standard retrieval (no task group match)")

        # Remove duplicates and sort by relevance
        seen_ids = set()
        unique_docs = []
        for doc in retrieved_docs:
            doc_id = doc.get("id", doc.get("document_id", str(hash(str(doc)))))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

        # Sort by score if available
        unique_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
        final_docs = unique_docs[:top_k]

        # Update semantic cache
        self._update_cache(cache_key, final_docs)

        return final_docs

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for semantic caching"""
        # Normalize query for better cache hits
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _update_cache(self, key: str, documents: List[Dict]):
        """Update semantic cache with new results"""
        if len(self.semantic_cache) >= self.cache_max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.semantic_cache))
            del self.semantic_cache[oldest_key]

        self.semantic_cache[key] = documents

    def update_task_group_documents(self, task_group: str, document_ids: List[str]):
        """Update the document IDs associated with a task group"""
        if task_group in self.task_engine.task_clusters:
            cluster = self.task_engine.task_clusters[task_group]
            cluster.document_ids = list(set(cluster.document_ids + document_ids))
            cluster.updated_at = datetime.now()
            self.task_engine.save_clusters()


class ContinuousLearningEngine:
    """Engine for continuous learning and model adaptation"""

    def __init__(
        self,
        task_engine: BlazingFastTaskIdentificationEngine,
        adaptation_interval_days: int = 7,
    ):
        self.task_engine = task_engine
        self.adaptation_interval_days = adaptation_interval_days
        self.last_adaptation = datetime.now() - timedelta(
            days=adaptation_interval_days + 1
        )
        self.performance_metrics = {}

    def should_adapt(self) -> bool:
        """Check if it's time for model adaptation"""
        return (
            datetime.now() - self.last_adaptation
        ).days >= self.adaptation_interval_days

    def collect_feedback_data(self) -> Dict:
        """Collect feedback data from user interactions"""
        # This would integrate with user feedback systems
        # For now, return mock data structure
        feedback_data = {
            "successful_queries": [],
            "failed_queries": [],
            "user_ratings": [],
            "response_times": [],
            "task_group_performance": {},
        }

        # In practice, this would read from a feedback database
        return feedback_data

    def adapt_model(self, feedback_data: Dict):
        """Adapt the model based on feedback data"""
        logger.info("Starting model adaptation...")

        # Analyze task group performance
        task_performance = self._analyze_task_performance(feedback_data)

        # Update task clusters based on performance
        self._update_clusters_from_feedback(task_performance)

        # Fine-tune retrieval strategies
        self._fine_tune_retrieval_strategies(task_performance)

        self.last_adaptation = datetime.now()
        logger.info("Model adaptation completed")

    def _analyze_task_performance(self, feedback_data: Dict) -> Dict:
        """Analyze performance of different task groups"""
        task_performance = defaultdict(dict)

        # Calculate metrics for each task group
        for cluster_id, cluster in self.task_engine.task_clusters.items():
            # Mock performance calculation
            successful_queries = len(
                [
                    q
                    for q in feedback_data["successful_queries"]
                    if self.task_engine.identify_task_group(q) == cluster_id
                ]
            )
            total_queries = cluster.query_count

            if total_queries > 0:
                success_rate = successful_queries / total_queries
                avg_response_time = np.mean(
                    [
                        rt
                        for i, rt in enumerate(feedback_data["response_times"])
                        if i < len(feedback_data["successful_queries"])
                        and self.task_engine.identify_task_group(
                            feedback_data["successful_queries"][i]
                        )
                        == cluster_id
                    ]
                    or [1.0]
                )

                task_performance[cluster_id] = {
                    "success_rate": success_rate,
                    "avg_response_time": avg_response_time,
                    "query_volume": total_queries,
                }

        return dict(task_performance)

    def _update_clusters_from_feedback(self, task_performance: Dict):
        """Update task clusters based on performance feedback"""
        for cluster_id, performance in task_performance.items():
            if cluster_id in self.task_engine.task_clusters:
                cluster = self.task_engine.task_clusters[cluster_id]

                # If performance is poor, mark for review
                if performance["success_rate"] < 0.7:
                    logger.warning(
                        f"Task cluster {cluster_id} has low success rate: {performance['success_rate']}"
                    )

                # Update cluster metadata
                cluster.updated_at = datetime.now()

        self.task_engine.save_clusters()

    def _fine_tune_retrieval_strategies(self, task_performance: Dict):
        """Fine-tune retrieval strategies based on performance"""
        # Adjust retrieval parameters based on performance
        for cluster_id, performance in task_performance.items():
            if performance["success_rate"] > 0.8:
                # High performing clusters - can reduce retrieval breadth
                logger.info(
                    f"Optimizing retrieval for high-performing cluster {cluster_id}"
                )
            elif performance["success_rate"] < 0.6:
                # Low performing clusters - increase retrieval breadth
                logger.info(
                    f"Expanding retrieval for low-performing cluster {cluster_id}"
                )

    def run_continuous_learning(self):
        """Run the continuous learning cycle"""
        if not self.should_adapt():
            return

        logger.info("Running continuous learning cycle...")

        # Collect feedback data
        feedback_data = self.collect_feedback_data()

        # Adapt model
        self.adapt_model(feedback_data)

        logger.info("Continuous learning cycle completed")


class AdaptiveRAGPipeline:
    """Main adaptive RAG pipeline implementing the proposed algorithm"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        config_manager: Optional[Any] = None,
    ):
        self.vector_store = vector_store or VectorStore(config_manager=config_manager)
        self.metadata_manager = MetadataManager(self.vector_store)

        # Initialize components
        self.task_engine = BlazingFastTaskIdentificationEngine()
        self.retrieval_engine = AdaptiveRetrievalEngine(
            self.vector_store, self.task_engine
        )
        self.learning_engine = ContinuousLearningEngine(self.task_engine)

        # Initialize cache
        self.response_cache = {}
        self.cache_max_size = 100

        logger.info("Adaptive RAG Pipeline initialized")

    def query(self, query: str, top_k: int = 5, **kwargs) -> Dict:
        """Main query method implementing the adaptive RAG algorithm"""
        start_time = time.time()

        # Stage 1: Task Identification and Grouping (run periodically)
        if len(self.task_engine.task_clusters) == 0:
            self.task_engine.run_task_identification()

        # Stage 2: Adaptive Retrieval
        logger.info(f"Performing adaptive retrieval for query: {query[:50]}...")

        retrieved_docs = self.retrieval_engine.retrieve_adaptive(query, top_k)

        if not retrieved_docs:
            return {
                "answer": "I couldn't find relevant information for your query.",
                "sources": [],
                "query_time": time.time() - start_time,
                "task_group": None,
                "retrieved_chunks": 0,
            }

        # Generate answer using retrieved documents
        context = self._prepare_context(retrieved_docs)
        answer = self._generate_answer(query, context)

        # Identify task group for this query
        task_group = self.task_engine.identify_task_group(query)

        # Update task group with relevant document IDs
        if task_group:
            doc_ids = [
                doc.get("id", doc.get("document_id", ""))
                for doc in retrieved_docs
                if doc.get("id") or doc.get("document_id")
            ]
            if doc_ids:
                self.retrieval_engine.update_task_group_documents(task_group, doc_ids)

        # Stage 3: Continuous Learning (run periodically)
        self.learning_engine.run_continuous_learning()

        query_time = time.time() - start_time

        result = {
            "answer": answer,
            "sources": self._format_sources(retrieved_docs),
            "query_time": query_time,
            "task_group": task_group,
            "retrieved_chunks": len(retrieved_docs),
            "model_used": "adaptive-rag",
        }

        return result

    def _prepare_context(self, documents: List[Dict], max_length: int = 3000) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        current_length = 0

        for doc in documents:
            text = doc.get("text", doc.get("content", ""))
            if text and current_length + len(text) <= max_length:
                context_parts.append(text)
                current_length += len(text)
            else:
                break

        return "\n\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer from query and context"""
        # This would use an LLM - simplified version for now
        if not context:
            return "I don't have enough information to answer this question."

        # Mock answer generation
        return f"Based on the available information: {context[:500]}..."

    def _format_sources(self, documents: List[Dict]) -> List[Dict]:
        """Format sources for response"""
        sources = []
        for doc in documents:
            source = {
                "title": doc.get("title", doc.get("source", "Unknown")),
                "content": doc.get("text", doc.get("content", ""))[:200] + "...",
                "score": doc.get("score", 0.0),
                "metadata": doc.get("metadata", {}),
            }
            sources.append(source)
        return sources

    def get_task_groups(self) -> Dict:
        """Get information about current task groups"""
        return {
            "total_groups": len(self.task_engine.task_clusters),
            "groups": [
                cluster.to_dict() for cluster in self.task_engine.task_clusters.values()
            ],
        }

    def force_task_identification(self, days_back: int = 7):
        """Force run task identification"""
        self.task_engine.run_task_identification(days_back)

    def force_continuous_learning(self):
        """Force run continuous learning cycle"""
