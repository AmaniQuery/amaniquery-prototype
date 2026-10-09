"""
Elasticsearch storage client for NiruSense
Replaces MinIO for document retrieval
"""

import os
from typing import Optional, Dict, Any
from elasticsearch import Elasticsearch
from loguru import logger


class ElasticsearchClient:
    """Elasticsearch client for document storage and retrieval"""

    def __init__(self):
        self.client: Optional[Elasticsearch] = None
        self._connect()

    def _connect(self):
        """Initialize Elasticsearch connection"""
        es_url = os.getenv("ELASTICSEARCH_URL")
        es_api_key = os.getenv("ELASTICSEARCH_API_KEY")

        if not es_url:
            logger.warning(
                "ELASTICSEARCH_URL not configured - document retrieval disabled"
            )
            return

        try:
            if es_api_key:
                self.client = Elasticsearch(es_url, api_key=es_api_key)
            else:
                # Try with username/password
                es_user = os.getenv("ELASTICSEARCH_USERNAME", "")
                es_pass = os.getenv("ELASTICSEARCH_PASSWORD", "")
                if es_user and es_pass:
                    self.client = Elasticsearch(es_url, basic_auth=(es_user, es_pass))
                else:
                    self.client = Elasticsearch(es_url)

            # Test connection
            if self.client.ping():
                logger.info("[OK] Elasticsearch connected successfully")
            else:
                logger.error("Elasticsearch ping failed")
                self.client = None

        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            self.client = None

    def get_document(self, index: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from Elasticsearch

        Args:
            index: Elasticsearch index name
            doc_id: Document ID

        Returns:
            Document data or None if not found
        """
        if not self.client:
            logger.error("Elasticsearch client not initialized")
            return None

        try:
            response = self.client.get(index=index, id=doc_id)

            if response and response.get("found"):
                doc_source = response.get("_source", {})

                # Return document with metadata
                return {
                    "id": response.get("_id"),
                    "index": response.get("_index"),
                    "text": doc_source.get("text", doc_source.get("content", "")),
                    "url": doc_source.get("url", ""),
                    "source": doc_source.get("source", ""),
                    "source_domain": doc_source.get("source_domain", ""),
                    "published_at": doc_source.get("published_at"),
                    "metadata": doc_source.get("metadata", {}),
                    **doc_source,  # Include all other fields
                }
            else:
                logger.warning(f"Document not found in Elasticsearch: {index}/{doc_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve document from Elasticsearch: {e}")
            return None

    def search_documents(self, index: str, query: str, size: int = 10) -> list:
        """
        Search for documents in Elasticsearch

        Args:
            index: Elasticsearch index name
            query: Search query
            size: Maximum number of results

        Returns:
            List of matching documents
        """
        if not self.client:
            return []

        try:
            response = self.client.search(
                index=index, query={"match": {"text": query}}, size=size
            )

            hits = response.get("hits", {}).get("hits", [])
            return [
                {
                    "id": hit.get("_id"),
                    "score": hit.get("_score"),
                    **hit.get("_source", {}),
                }
                for hit in hits
            ]

        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check Elasticsearch connection health"""
        if not self.client:
            return {
                "status": "unavailable",
                "message": "Elasticsearch client not initialized",
            }

        try:
            if self.client.ping():
                cluster_health = self.client.cluster.health()
                return {
                    "status": "healthy",
                    "cluster_status": cluster_health.get("status"),
                    "number_of_nodes": cluster_health.get("number_of_nodes"),
                }
            else:
                return {"status": "unhealthy", "message": "Elasticsearch ping failed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Global instance
es_client = ElasticsearchClient()
