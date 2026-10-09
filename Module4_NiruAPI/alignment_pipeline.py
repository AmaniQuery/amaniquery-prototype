"""
Constitutional Alignment Pipeline
Performs dual-retrieval RAG for constitutional alignment analysis
"""

from typing import Dict, List, Optional, Tuple
from loguru import logger
import re
from dataclasses import dataclass

from Module3_NiruDB.vector_store import VectorStore
from Module4_NiruAPI.rag_pipeline import RAGPipeline


@dataclass
class AlignmentContext:
    """Stores contexts for alignment analysis"""

    bill_chunks: List[Dict]
    constitution_chunks: List[Dict]
    query: str
    bill_name: Optional[str] = None
    legal_concept: Optional[str] = None


class QueryAnalyzer:
    """Analyzes queries to extract Bill references and legal concepts"""

    def __init__(self):
        # Patterns for bill identification
        self.bill_patterns = [
            r"(?:the\s+)?([A-Z][a-zA-Z\s]+(?:Bill|Act)(?:\s*,?\s*20\d{2})?)",
            r"(?:bill|act)\s+(?:on|about|regarding)\s+([a-z\s]+)",
        ]

        # Constitutional concepts
        self.constitutional_concepts = {
            "rights": ["right", "rights", "freedom", "liberty"],
            "taxation": ["tax", "levy", "duty", "revenue", "taxation"],
            "property": ["property", "ownership", "land"],
            "housing": ["housing", "shelter", "residential"],
            "economic_rights": ["economic", "social rights", "welfare"],
            "governance": ["government", "parliament", "executive"],
            "elections": ["election", "vote", "voting", "electoral"],
            "citizenship": ["citizen", "citizenship", "nationality"],
            "devolution": ["county", "devolution", "local government"],
            "public_finance": ["budget", "finance", "appropriation"],
        }

    def analyze(self, query: str) -> Dict[str, any]:
        """
        Analyze query to identify:
        1. Bill/Act being referenced
        2. Constitutional concepts involved
        3. Type of analysis needed

        Returns:
            {
                "bill_name": str,
                "legal_concepts": List[str],
                "keywords": List[str],
                "analysis_type": str  # "alignment", "conflict", "compatibility"
            }
        """
        result = {
            "bill_name": None,
            "legal_concepts": [],
            "keywords": [],
            "analysis_type": "alignment",
        }

        # Extract Bill/Act name
        result["bill_name"] = self._extract_bill_name(query)

        # Identify constitutional concepts
        result["legal_concepts"] = self._identify_concepts(query)

        # Extract keywords for retrieval
        result["keywords"] = self._extract_keywords(query)

        # Determine analysis type
        result["analysis_type"] = self._determine_analysis_type(query)

        logger.info(
            f"Query analysis: Bill={result['bill_name']}, Concepts={result['legal_concepts']}"
        )
        return result

    def _extract_bill_name(self, query: str) -> Optional[str]:
        """Extract bill/act name from query"""
        for pattern in self.bill_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                bill_name = match.group(1).strip()
                return bill_name

        # Check for common bill references
        if "finance bill" in query.lower():
            # Extract year if present
            year_match = re.search(r"20\d{2}", query)
            year = year_match.group(0) if year_match else "2025"
            return f"Finance Bill {year}"

        return None

    def _identify_concepts(self, query: str) -> List[str]:
        """Identify constitutional concepts in query"""
        query_lower = query.lower()
        concepts = []

        for concept, keywords in self.constitutional_concepts.items():
            if any(keyword in query_lower for keyword in keywords):
                concepts.append(concept)

        return concepts

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords for retrieval"""
        # Remove common words
        stop_words = {
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
            "from",
            "how",
            "does",
            "what",
            "is",
            "are",
        }

        words = re.findall(r"\b[a-z]+\b", query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]

        return keywords[:10]  # Top 10 keywords

    def _determine_analysis_type(self, query: str) -> str:
        """Determine type of analysis requested"""
        query_lower = query.lower()

        if any(
            word in query_lower
            for word in ["conflict", "contradict", "violate", "breach"]
        ):
            return "conflict"
        elif any(
            word in query_lower for word in ["compatible", "consistent", "harmony"]
        ):
            return "compatibility"
        else:
            return "alignment"


class ConstitutionalAlignmentPipeline:
    """
    Dual-retrieval RAG pipeline for constitutional alignment analysis
    """

    def __init__(self, vector_store: VectorStore, rag_pipeline: RAGPipeline):
        self.vector_store = vector_store
        self.rag_pipeline = rag_pipeline
        self.query_analyzer = QueryAnalyzer()

    async def analyze_alignment(
        self, query: str, bill_top_k: int = 3, constitution_top_k: int = 3
    ) -> Dict[str, any]:
        """
        Perform constitutional alignment analysis

        Steps:
        1. Analyze query to extract Bill and legal concepts
        2. Retrieve Bill context (filtered by category='Bill')
        3. Retrieve Constitution context (filtered by category='Constitution')
        4. Construct structured alignment prompt
        5. Generate analysis using LLM

        Args:
            query: User's question about constitutional alignment
            bill_top_k: Number of Bill chunks to retrieve
            constitution_top_k: Number of Constitution chunks to retrieve

        Returns:
            {
                "analysis": str,  # LLM's comparative analysis
                "bill_context": List[Dict],
                "constitution_context": List[Dict],
                "metadata": Dict
            }
        """
        logger.info(f"Starting constitutional alignment analysis: {query}")

        # Step 1: Analyze query
        query_analysis = self.query_analyzer.analyze(query)

        # Step 2: Dual retrieval
        alignment_context = await self._dual_retrieval(
            query=query,
            query_analysis=query_analysis,
            bill_top_k=bill_top_k,
            constitution_top_k=constitution_top_k,
        )

        # Step 3: Construct alignment prompt
        alignment_prompt = self._construct_alignment_prompt(
            context=alignment_context, analysis_type=query_analysis["analysis_type"]
        )

        # Step 4: Generate analysis with LLM
        analysis = self.rag_pipeline._generate_answer(
            query=query,
            context=alignment_prompt["user_prompt"],
            temperature=0.3,
            max_tokens=1500,
        )

        # Step 5: Format response
        response = {
            "analysis": analysis,
            "bill_context": [
                {
                    "text": chunk.get("text", ""),
                    "clause_number": chunk.get("clause_number"),
                    "subject": chunk.get("subject"),
                    "title": chunk.get("title", ""),
                }
                for chunk in alignment_context.bill_chunks
            ],
            "constitution_context": [
                {
                    "text": chunk.get("text", ""),
                    "article_number": chunk.get("article_number"),
                    "article_title": chunk.get("article_title"),
                    "clause": chunk.get("clause"),
                }
                for chunk in alignment_context.constitution_chunks
            ],
            "metadata": {
                "bill_name": query_analysis["bill_name"],
                "legal_concepts": query_analysis["legal_concepts"],
                "analysis_type": query_analysis["analysis_type"],
                "bill_chunks_count": len(alignment_context.bill_chunks),
                "constitution_chunks_count": len(alignment_context.constitution_chunks),
            },
        }

        logger.info("Constitutional alignment analysis completed")
        return response

    async def _dual_retrieval(
        self, query: str, query_analysis: Dict, bill_top_k: int, constitution_top_k: int
    ) -> AlignmentContext:
        """
        Perform dual retrieval: Bill context + Constitution context
        Uses async parallel retrieval for 2x speedup.
        """
        import asyncio

        # Branch 1: Retrieve Bill/Act context
        bill_search_query = self._construct_bill_search_query(query, query_analysis)
        logger.info(f"Bill search query: {bill_search_query}")

        # Branch 2: Retrieve Constitution context
        constitution_search_query = self._construct_constitution_search_query(
            query, query_analysis
        )
        logger.info(f"Constitution search query: {constitution_search_query}")

        # Async retrieval with run_in_executor for blocking calls
        async def retrieve_bill():
            loop = asyncio.get_event_loop()

            def _sync_query():
                bill_results = self.vector_store.query(
                    query_text=bill_search_query,
                    n_results=bill_top_k,
                    filter={"category": "Bill"},
                    namespace=["kenya_law"],
                )

                # If no Bills found, try Acts
                if not bill_results:
                    logger.warning("No Bills found, searching Acts...")
                    bill_results = self.vector_store.query(
                        query_text=bill_search_query,
                        n_results=bill_top_k,
                        filter={"category": "Act"},
                        namespace=["kenya_law"],
                    )
                return bill_results

            return await loop.run_in_executor(None, _sync_query)

        async def retrieve_constitution():
            loop = asyncio.get_event_loop()

            def _sync_query():
                return self.vector_store.query(
                    query_text=constitution_search_query,
                    n_results=constitution_top_k,
                    filter={"category": "Constitution"},
                    namespace=["kenya_law"],
                )

            return await loop.run_in_executor(None, _sync_query)

        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context
            bill_task = loop.create_task(retrieve_bill())
            const_task = loop.create_task(retrieve_constitution())
            bill_results, constitution_results = await asyncio.gather(
                bill_task, const_task
            )
        except RuntimeError:
            # No running loop, create one
            bill_results, constitution_results = asyncio.run(
                asyncio.gather(retrieve_bill(), retrieve_constitution())
            )

        return AlignmentContext(
            bill_chunks=bill_results or [],
            constitution_chunks=constitution_results or [],
            query=query,
            bill_name=query_analysis["bill_name"],
            legal_concept=(
                query_analysis["legal_concepts"][0]
                if query_analysis["legal_concepts"]
                else None
            ),
        )

    def _construct_bill_search_query(self, original_query: str, analysis: Dict) -> str:
        """Construct optimized search query for Bill retrieval"""
        parts = []

        if analysis["bill_name"]:
            parts.append(analysis["bill_name"])

        # Add legal concepts
        if analysis["legal_concepts"]:
            parts.extend(analysis["legal_concepts"][:2])

        # Add original query keywords
        parts.extend(analysis["keywords"][:3])

        return " ".join(parts)

    def _construct_constitution_search_query(
        self, original_query: str, analysis: Dict
    ) -> str:
        """Construct optimized search query for Constitution retrieval"""
        parts = []

        # Focus on constitutional concepts
        if analysis["legal_concepts"]:
            for concept in analysis["legal_concepts"]:
                # Map concepts to constitutional terms
                if concept == "taxation":
                    parts.extend(["taxation", "revenue", "public finance"])
                elif concept == "housing":
                    parts.extend(["housing", "shelter", "economic rights"])
                elif concept == "property":
                    parts.extend(["property", "ownership", "right to property"])
                else:
                    parts.append(concept)

        # Add keywords
        parts.extend(analysis["keywords"][:3])

        return " ".join(parts)

    def _construct_alignment_prompt(
        self, context: AlignmentContext, analysis_type: str
    ) -> Dict[str, str]:
        """
        Construct structured prompt for alignment analysis

        Returns:
            {
                "system_prompt": str,
                "user_prompt": str
            }
        """
        # System prompt - defines role and responsibilities
        system_prompt = """You are a specialized Kenyan legal analyst with expertise in constitutional law. Your task is to objectively compare the provided sections from a new Bill/Act with the relevant articles from the Constitution of Kenya.

IMPORTANT: You must NOT provide a final legal opinion or judgment. Instead, present a factual, structured analysis by:

1. **Summarizing the Bill's Proposal**: Clearly state what the Bill proposes, citing specific clauses.

2. **Stating Constitutional Provisions**: Quote or paraphrase the relevant constitutional articles that apply.

3. **Identifying Relationships**: Highlight areas of:
   - Alignment (where Bill supports constitutional provisions)
   - Potential tension (where Bill may raise constitutional questions)
   - Overlap (where Bill and Constitution address same subject)

4. **Citation Requirements**: Every statement must cite the source document (Bill clause or Constitution article).

5. **Objectivity**: Avoid words like "violates," "unconstitutional," or "legal." Use neutral terms like "may raise questions regarding," "appears to relate to," or "could be analyzed in light of."

Your analysis should enable legal experts to make informed judgments."""

        # Construct user prompt with structured context
        bill_context_str = self._format_bill_context(context.bill_chunks)
        constitution_context_str = self._format_constitution_context(
            context.constitution_chunks
        )

        user_prompt = f"""CONTEXT FROM BILL/ACT:

{bill_context_str}

---

CONTEXT FROM CONSTITUTION:

{constitution_context_str}

---

USER QUESTION: {context.query}

---

YOUR ANALYSIS:
Provide a structured analysis following this format:

**1. The Bill's Proposal:**
[Summarize what the bill proposes, with citations]

**2. Relevant Constitutional Provisions:**
[List applicable constitutional articles and what they guarantee/regulate]

**3. Alignment Analysis:**
[Discuss how the bill relates to constitutional provisions - areas of alignment, potential tensions, overlaps]

**4. Key Considerations:**
[Highlight important points for legal review]

Remember: Cite every claim with [Source: Document, Clause/Article X]"""

        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    def _format_bill_context(self, chunks: List[Dict]) -> str:
        """Format Bill chunks for prompt"""
        if not chunks:
            return "[No Bill context found]"

        formatted = []
        for i, chunk in enumerate(chunks, 1):
            clause = chunk.get("clause_number", "N/A")
            subject = chunk.get("subject", "")
            title = chunk.get("bill_title") or chunk.get("title", "Unknown Bill")
            text = chunk.get("text", "")

            formatted.append(f"""[Bill Chunk {i}]
Source: {title}, Clause {clause}
Subject: {subject}
Text: {text}
""")

        return "\n".join(formatted)

    def _format_constitution_context(self, chunks: List[Dict]) -> str:
        """Format Constitution chunks for prompt"""
        if not chunks:
            return "[No Constitution context found]"

        formatted = []
        for i, chunk in enumerate(chunks, 1):
            article = chunk.get("article_number", "N/A")
            article_title = chunk.get("article_title", "")
            clause = chunk.get("clause", "")
            text = chunk.get("text", "")

            formatted.append(f"""[Constitution Chunk {i}]
Source: Constitution of Kenya 2010, Article {article}
Title: {article_title}
Clause: {clause if clause else "Main provision"}
Text: {text}
""")

        return "\n".join(formatted)

    async def quick_check(
        self, bill_name: str, constitutional_topic: str
    ) -> Dict[str, any]:
        """
        Quick alignment check for a specific bill and constitutional topic

        Args:
            bill_name: Name of the bill (e.g., "Finance Bill 2025")
            constitutional_topic: Topic to check (e.g., "taxation", "housing rights")

        Returns:
            Alignment analysis
        """
        query = f"How does the {bill_name} align with the Constitution regarding {constitutional_topic}?"
        return await self.analyze_alignment(query)
