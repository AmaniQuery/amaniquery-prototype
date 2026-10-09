from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import AutoTokenizer, AutoModel
import torch
import os
import numpy as np
from datetime import datetime, timedelta
from .query_parser import QueryParser


class AmaniQueryRetriever(BaseRetriever):
    """
    Custom Retriever for AmaniQuery that performs hybrid search,
    sentiment aggregation, and intensity ranking.
    """

    qdrant_client: Any = None
    query_parser: Any = None
    tokenizer: Any = None
    model: Any = None
    collection_name: str = "kenyan_sentiments_2025"
    device: str = "cpu"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize Qdrant Cloud
        qdrant_url = os.getenv("QDRANT_URL")  # Cloud-based, no localhost fallback
        qdrant_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

        # Initialize Query Parser
        self.query_parser = QueryParser()

        # Initialize Encoder (AfriBERTa)
        model_name = "Davlan/afriberta-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, text: str) -> List[float]:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings[0].tolist()

    def _build_filters(self, parsed_query: dict) -> models.Filter:
        conditions = []

        # Topic Filter
        if parsed_query.get("topic"):
            conditions.append(
                models.FieldCondition(
                    key="topic", match=models.MatchText(text=parsed_query["topic"])
                )
            )

        # Platform Filter
        if parsed_query.get("platform"):
            conditions.append(
                models.FieldCondition(
                    key="platform",
                    match=models.MatchValue(value=parsed_query["platform"]),
                )
            )

        # Sentiment Filter (e.g., "angry" -> negative + high intensity)
        sentiment = parsed_query.get("sentiment", "").lower()
        if sentiment:
            if "neg" in sentiment or "angry" in sentiment:
                conditions.append(
                    models.FieldCondition(
                        key="sentiment_score", range=models.Range(lt=-0.1)
                    )
                )
            elif "pos" in sentiment or "happy" in sentiment:
                conditions.append(
                    models.FieldCondition(
                        key="sentiment_score", range=models.Range(gt=0.1)
                    )
                )

        # Date Range
        date_range = parsed_query.get("date_range")
        if date_range:
            now = datetime.now()
            start_date = None

            if "last_24_hours" in date_range:
                start_date = now - timedelta(hours=24)
            elif "last_7_days" in date_range or "last_week" in date_range:
                start_date = now - timedelta(days=7)
            elif "last_30_days" in date_range:
                start_date = now - timedelta(days=30)

            if start_date:
                conditions.append(
                    models.FieldCondition(
                        key="created_at", range=models.Range(gte=start_date.isoformat())
                    )
                )

        if not conditions:
            return None

        return models.Filter(must=conditions)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:

        # 1. Parse Query
        parsed = self.query_parser.parse_query(query)
        print(f"Parsed Query: {parsed}")

        # 2. Build Filters
        qdrant_filter = self._build_filters(parsed)

        # 3. Encode Query
        query_vector = self._encode(query)

        # 4. Search Qdrant
        limit = 20
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )

        if not results:
            return [Document(page_content="No relevant results found.")]

        # 5. Aggregate Stats
        scores = [r.payload.get("sentiment_score", 0) for r in results]
        intensities = [r.payload.get("intensity", 0) for r in results]

        avg_score = np.mean(scores) if scores else 0
        avg_intensity = np.mean(intensities) if intensities else 0

        sentiment_desc = "Neutral"
        if avg_score > 0.3:
            sentiment_desc = "Positive"
        elif avg_score < -0.3:
            sentiment_desc = "Negative"
        if avg_score < -0.7:
            sentiment_desc = "Strongly Negative"
        if avg_score > 0.7:
            sentiment_desc = "Strongly Positive"

        summary_text = (
            f"Public sentiment is {sentiment_desc}, average {avg_score:.2f} "
            f"with intensity {avg_intensity:.2f} based on {len(results)} results."
        )

        # 6. Re-rank (if requested)
        sort_by = parsed.get("sort_by", "relevance")
        if "intensity" in sort_by:
            # Sort by intensity (descending)
            results.sort(key=lambda x: x.payload.get("intensity", 0), reverse=True)
        elif "recency" in sort_by:
            # Sort by created_at (descending)
            results.sort(key=lambda x: x.payload.get("created_at", ""), reverse=True)

        # 7. Construct Documents
        docs = []

        # Summary Document (First)
        docs.append(
            Document(
                page_content=summary_text,
                metadata={"type": "summary", "avg_score": avg_score},
            )
        )

        # Result Documents
        for res in results:
            docs.append(
                Document(
                    page_content=res.payload.get("text", ""),
                    metadata={
                        "score": res.score,
                        "sentiment": res.payload.get("sentiment_score"),
                        "platform": res.payload.get("platform"),
                        "created_at": res.payload.get("created_at"),
                    },
                )
            )

        return docs


if __name__ == "__main__":
    # Example Usage
    retriever = AmaniQueryRetriever()
    docs = retriever.invoke("What is the current Kenyan mood on Finance Bill?")
    for d in docs:
        print(f"[{d.metadata.get('type', 'quote')}] {d.page_content}")
