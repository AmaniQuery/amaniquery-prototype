"""
Context Compression Node for Agentic RAG
Semantic deduplication and token optimization
"""

import re
from typing import Dict, Any, List, Tuple
from loguru import logger

# =============================================================================
# CONTEXT COMPRESSION
# =============================================================================


def compression_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compress retrieved contexts by removing redundancy

    Args:
        state: AmaniQState with tool_results or refinement_results

    Returns:
        Updated state with compressed_context
    """
    # Gather all contexts
    tool_results = state.get("tool_results", [])
    refinement_results = state.get("refinement_results", [])

    logger.info(
        f"[Compression] Processing {len(tool_results)} tool results + {len(refinement_results)} refinements"
    )

    # Extract text chunks
    chunks = []

    # From tool results
    for result in tool_results:
        content = _extract_content(result)
        if content:
            chunks.append(
                {
                    "text": content,
                    "source": result.get("tool_name", "unknown"),
                    "metadata": result.get("metadata", {}),
                }
            )

    # From refinement results
    for result in refinement_results:
        if result.get("success"):
            content = _extract_content(result.get("result", {}))
            if content:
                chunks.append(
                    {
                        "text": content,
                        "source": "refinement",
                        "query": result.get("query", ""),
                    }
                )

    if not chunks:
        logger.warning("[Compression] No content to compress")
        return {
            "compressed_context": "",
            "compression_stats": {"original_chunks": 0, "compressed_chunks": 0},
        }

    # Deduplicate and compress
    compressed_chunks = _semantic_deduplicate(chunks)
    compressed_text = _format_compressed_context(compressed_chunks, max_tokens=3000)

    stats = {
        "original_chunks": len(chunks),
        "compressed_chunks": len(compressed_chunks),
        "compression_ratio": len(compressed_chunks) / len(chunks) if chunks else 0,
        "estimated_tokens": len(compressed_text.split()) * 1.3,  # Rough estimate
    }

    logger.info(
        f"[Compression] {stats['original_chunks']} → {stats['compressed_chunks']} chunks ({stats['compression_ratio']:.1%} retained)"
    )

    return {"compressed_context": compressed_text, "compression_stats": stats}


def _extract_content(result: Dict) -> str:
    """Extract text content from tool result"""
    if isinstance(result, str):
        return result

    # Try various keys
    for key in ["content", "text", "result", "output", "data"]:
        if key in result:
            val = result[key]
            if isinstance(val, str):
                return val
            elif isinstance(val, list):
                return "\n".join([str(v) for v in val if v])

    return ""


def _semantic_deduplicate(chunks: List[Dict]) -> List[Dict]:
    """
    Remove semantically duplicate chunks
    Simple approach: exact and near-duplicate removal
    """
    if not chunks:
        return []

    unique_chunks = []
    seen_texts = set()

    for chunk in chunks:
        text = chunk["text"].strip()

        # Normalize for comparison
        normalized = _normalize_text(text)

        # Skip exact duplicates
        if normalized in seen_texts:
            continue

        # Check for near-duplicates (simple substring check)
        is_duplicate = False
        for seen in seen_texts:
            # If 80%+ overlap, consider duplicate
            if _calculate_overlap(normalized, seen) > 0.8:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_chunks.append(chunk)
            seen_texts.add(normalized)

    return unique_chunks


def _normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _calculate_overlap(text1: str, text2: str) -> float:
    """Calculate text overlap ratio"""
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.split())
    words2 = set(text2.split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def _format_compressed_context(chunks: List[Dict], max_tokens: int = 3000) -> str:
    """Format compressed chunks into final context"""
    formatted_parts = []
    current_tokens = 0

    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"]
        source = chunk.get("source", "unknown")

        # Estimate tokens (rough: 1 word ≈ 1.3 tokens)
        chunk_tokens = len(text.split()) * 1.3

        if current_tokens + chunk_tokens > max_tokens:
            logger.info(f"[Compression] Reached token limit at chunk {i}/{len(chunks)}")
            break

        # Format with source
        formatted = f"[Source: {source}]\n{text}\n"
        formatted_parts.append(formatted)
        current_tokens += chunk_tokens

    return "\n---\n".join(formatted_parts)


# =============================================================================
# UTILITY: Extract Citations
# =============================================================================


def extract_citations(compressed_context: str) -> List[Dict[str, str]]:
    """Extract citations from compressed context"""
    citations = []

    # Find all [Source: ...] markers
    source_pattern = r"\[Source: ([^\]]+)\]"
    matches = re.finditer(source_pattern, compressed_context)

    for match in matches:
        source = match.group(1)
        citations.append({"source": source})

    return citations
