"""
Alignment Router - Constitutional alignment analysis endpoints for AmaniQuery
"""

import time
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

router = APIRouter(tags=["Constitutional Alignment"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class AlignmentRequest(BaseModel):
    """Alignment request model"""

    query: str
    bill_top_k: int = 5
    constitution_top_k: int = 5


class BillContext(BaseModel):
    """Bill context model"""

    section: str
    content: str
    relevance_score: float


class ConstitutionContext(BaseModel):
    """Constitution context model"""

    article: str
    content: str
    relevance_score: float


class AlignmentMetadata(BaseModel):
    """Alignment metadata model"""

    bill_name: Optional[str] = None
    constitutional_topics: List[str] = []
    analysis_type: str = "comparative"


class AlignmentResponse(BaseModel):
    """Alignment response model"""

    analysis: str
    bill_context: List[BillContext]
    constitution_context: List[ConstitutionContext]
    metadata: AlignmentMetadata
    query_time: Optional[float] = None


# =============================================================================
# DEPENDENCIES - Module-level globals set by main app
# =============================================================================

alignment_pipeline = None
rag_pipeline = None
cache_manager = None


def get_alignment_pipeline():
    """Get the alignment pipeline instance"""
    if alignment_pipeline is None:
        raise HTTPException(status_code=503, detail="Alignment service not initialized")
    return alignment_pipeline


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/alignment-check", response_model=AlignmentResponse)
async def check_constitutional_alignment(request: AlignmentRequest):
    """
    Constitutional Alignment Analysis - Compare Bills/Acts with Constitution

    This endpoint performs specialized dual-retrieval RAG analysis to compare
    proposed or enacted legislation with relevant constitutional provisions.

    **How it works:**
    1. Analyzes your query to identify the Bill and constitutional concepts
    2. Retrieves relevant Bill/Act sections
    3. Retrieves relevant Constitutional articles
    4. Generates structured comparative analysis with citations

    **Example queries:**
    - "How does the Finance Bill 2025 housing levy align with the constitution?"
    - "Does the Data Protection Act comply with constitutional privacy rights?"
    - "What does the Constitution say about the new taxation measures?"

    **Important:** This provides factual analysis, NOT legal opinions.
    """
    alignment_pipeline = get_alignment_pipeline()

    try:
        start_time = time.time()

        result = await alignment_pipeline.analyze_alignment(
            query=request.query,
            bill_top_k=request.bill_top_k,
            constitution_top_k=request.constitution_top_k,
        )

        query_time = time.time() - start_time

        return AlignmentResponse(
            analysis=result["analysis"],
            bill_context=[BillContext(**ctx) for ctx in result["bill_context"]],
            constitution_context=[
                ConstitutionContext(**ctx) for ctx in result["constitution_context"]
            ],
            metadata=AlignmentMetadata(**result["metadata"]),
            query_time=query_time,
        )

    except Exception as e:
        logger.error(f"Error in constitutional alignment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alignment-quick-check", response_model=AlignmentResponse)
async def quick_alignment_check(bill_name: str, constitutional_topic: str):
    """
    Quick Constitutional Alignment Check

    Simplified endpoint for checking specific bill against constitutional topic.

    **Parameters:**
    - bill_name: Name of the bill (e.g., "Finance Bill 2025")
    - constitutional_topic: Topic to check (e.g., "taxation", "housing rights", "privacy")

    **Example:**
    - bill_name: "Finance Bill 2025"
    - constitutional_topic: "taxation and revenue"
    """
    alignment_pipeline = get_alignment_pipeline()

    try:
        result = await alignment_pipeline.quick_check(
            bill_name=bill_name,
            constitutional_topic=constitutional_topic,
        )

        return AlignmentResponse(
            analysis=result["analysis"],
            bill_context=[BillContext(**ctx) for ctx in result["bill_context"]],
            constitution_context=[
                ConstitutionContext(**ctx) for ctx in result["constitution_context"]
            ],
            metadata=AlignmentMetadata(**result["metadata"]),
            query_time=result.get("query_time"),
        )

    except Exception as e:
        logger.error(f"Error in quick alignment check: {e}")
        raise HTTPException(status_code=500, detail=str(e))
