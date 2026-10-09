"""
Query Expansion - Expands queries for better retrieval
"""

from typing import List, Dict, Any
from loguru import logger


class QueryExpansion:
    """
    Expands queries with synonyms, related terms, and context
    """

    def expand(self, query: str, method: str = "synonyms") -> List[str]:
        """
        Expand a query

        Args:
            query: Original query
            method: Expansion method (synonyms, related, context)

        Returns:
            List of expanded query variations
        """
        expansions = [query]  # Always include original

        if method == "synonyms":
            # Simple synonym expansion (in production, use a thesaurus or LLM)
            expansions.extend(self._synonym_expansion(query))
        elif method == "related":
            expansions.extend(self._related_terms(query))
        elif method == "context":
            expansions.extend(self._context_expansion(query))

        return expansions

    def _synonym_expansion(self, query: str) -> List[str]:
        """Expand with synonyms"""
        # Simple keyword-based expansion
        # In production, use WordNet, LLM, or domain-specific thesaurus
        expansions = []

        # Common legal/parliamentary term expansions
        term_mappings = {
            "law": ["legislation", "statute", "act"],
            "bill": ["proposal", "legislation", "draft"],
            "constitution": ["constitutional", "charter", "fundamental law"],
            "parliament": ["legislature", "assembly", "congress"],
        }

        query_lower = query.lower()
        for term, synonyms in term_mappings.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded = query_lower.replace(term, synonym)
                    if expanded != query_lower:
                        expansions.append(expanded)

        return expansions

    def _related_terms(self, query: str) -> List[str]:
        """Add related terms"""
        # Add related terms based on query structure
        expansions = []

        # If query is about a specific topic, add related aspects
        if "kenya" in query.lower():
            expansions.append(query + " legal framework")
            expansions.append(query + " recent developments")

        return expansions

    def _context_expansion(self, query: str) -> List[str]:
        """Expand with context"""
        expansions = []

        # Add context-specific terms
        expansions.append(query + " Kenya")
        expansions.append(query + " 2024")
        expansions.append(query + " analysis")

        return expansions
