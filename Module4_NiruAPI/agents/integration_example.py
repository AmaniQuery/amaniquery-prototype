"""
AmaniQuery v2.0 - Complete Integration Example
===============================================

This file demonstrates how to integrate the Intent Router and Sheng Translator
into a complete query processing pipeline.

Pipeline Flow:
1. User sends query → Detect Intent (wanjiku/wakili/mwanahabari)
2. If wanjiku → Translate Sheng to Formal
3. Execute RAG search with formal query
4. If wanjiku → Re-inject Sheng into response
5. Return personalized answer

"""

import json
from typing import Dict, Any, Optional

# ============================================================================
# MOCK FUNCTIONS (Replace with your actual implementations)
# ============================================================================


def mock_gemini_llm(prompt: str) -> str:
    """
    Mock LLM function. Replace with actual Gemini/Groq API call.

    In production:
        import google.generativeai as genai
        genai.configure(api_key="YOUR_KEY")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt, temperature=0.1)
        return response.text
    """
    # For testing purposes, return mock responses
    if "classify" in prompt.lower() or "query_type" in prompt.lower():
        # Mock intent classification
        if (
            "kanjo" in prompt.lower()
            or "doh" in prompt.lower()
            or "bana" in prompt.lower()
        ):
            return json.dumps(
                {
                    "query_type": "wanjiku",
                    "confidence": 0.9,
                    "detected_language": "sheng",
                    "reasoning": "Informal Sheng query about practical civic issue",
                }
            )
        elif "section" in prompt.lower() and "act" in prompt.lower():
            return json.dumps(
                {
                    "query_type": "wakili",
                    "confidence": 0.95,
                    "detected_language": "en",
                    "reasoning": "Formal legal query requesting specific statutory provision",
                }
            )
        else:
            return json.dumps(
                {
                    "query_type": "mwanahabari",
                    "confidence": 0.85,
                    "detected_language": "en",
                    "reasoning": "Data-oriented query seeking statistics",
                }
            )

    elif "formal query" in prompt.lower() or "translate" in prompt.lower():
        # Mock Sheng → Formal translation
        return "What has the Nairobi City County Government resolved regarding parking fees?"

    elif "sheng-infused" in prompt.lower() or "conversational" in prompt.lower():
        # Mock Formal → Sheng translation
        return "Kanjo wameongeza parking fees kwa town centre to KES 300 per day, kutoka March 1, 2024."

    return "Mock LLM response"


def mock_rag_search(formal_query: str) -> str:
    """
    Mock RAG search function. Replace with actual RAG implementation.

    In production:
        from Module4_NiruAPI.rag_pipeline import RAGPipeline
        rag = RAGPipeline()
        results = rag.query(formal_query)
        return results['answer']
    """
    # Return mock parliamentary/government data
    mock_responses = {
        "parking fees": "The Nairobi City County Government passed Resolution No. 42/2024 on February 15, 2024, increasing parking fees in the Central Business District from KES 200 to KES 300 per day, effective March 1, 2024. The resolution was passed by the County Assembly with 45 votes in favor and 12 against.",
        "finance bill": "The Finance Bill 2024 was debated in Parliament on January 20, 2024. The Member of Parliament for Starehe Constituency, Hon. Charles Kanyi, opposed the bill citing increased tax burden on small businesses. The bill proposes a 2% increase in VAT and introduction of a digital services tax.",
        "mp attendance": "According to parliamentary records for Q3 2024 (July-September), the average MP attendance rate was 67%. The Finance Committee had an attendance rate of 72%, while the Health Committee recorded 61% attendance. The highest individual attendance was 95% and the lowest was 34%.",
        "default": "Based on parliamentary records and government documents, relevant information has been retrieved from official sources including Hansard, county resolutions, and budget documents.",
    }

    # Simple keyword matching for mock
    query_lower = formal_query.lower()
    if "parking" in query_lower or "fees" in query_lower:
        return mock_responses["parking fees"]
    elif "finance bill" in query_lower or "starehe" in query_lower:
        return mock_responses["finance bill"]
    elif "attendance" in query_lower or "mp" in query_lower:
        return mock_responses["mp attendance"]
    else:
        return mock_responses["default"]


# ============================================================================
# MAIN INTEGRATION PIPELINE
# ============================================================================


def process_amaniquery(
    user_query: str,
    llm_function: callable = None,
    rag_function: callable = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Complete AmaniQuery processing pipeline integrating:
    - Intent classification
    - Sheng translation (if needed)
    - RAG retrieval
    - Response personalization

    Args:
        user_query: The user's original query
        llm_function: LLM callable (defaults to mock)
        rag_function: RAG search callable (defaults to mock)
        verbose: If True, prints pipeline steps

    Returns:
        {
            "query": str,
            "intent": Dict[str, Any],
            "translation": Optional[Dict[str, Any]],
            "formal_query": str,
            "raw_answer": str,
            "final_answer": str,
            "pipeline_steps": List[str],
            "metadata": Dict[str, Any]
        }
    """
    # Use defaults if not provided
    if llm_function is None:
        llm_function = mock_gemini_llm
    if rag_function is None:
        rag_function = mock_rag_search

    pipeline_steps = []
    metadata = {}

    # Import modules
    try:
        import sys
        import os

        sys.path.insert(0, os.path.dirname(__file__))

        from intent_router import classify_query
        from sheng_translator import (
            translate_to_formal,
            translate_to_sheng,
            detect_sheng,
        )
    except ImportError as e:
        return {
            "error": f"Failed to import AmaniQuery modules: {e}",
            "suggestion": "Ensure intent_router.py and sheng_translator.py are in the same directory",
        }

    # ========================================================================
    # STEP 1: CLASSIFY INTENT
    # ========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 1: INTENT CLASSIFICATION")
        print("=" * 80)

    intent_result = classify_query(user_query, llm_function)
    pipeline_steps.append(f"Classified as {intent_result['query_type']}")

    if verbose:
        print(f"Query Type: {intent_result['query_type']}")
        print(f"Confidence: {intent_result['confidence']}")
        print(f"Language: {intent_result['detected_language']}")
        print(f"Reasoning: {intent_result['reasoning']}")

    metadata["intent_confidence"] = intent_result["confidence"]
    metadata["detected_language"] = intent_result["detected_language"]

    # ========================================================================
    # STEP 2: TRANSLATE SHENG (if wanjiku)
    # ========================================================================
    translation_result = None
    formal_query = user_query

    if intent_result["query_type"] == "wanjiku":
        if verbose:
            print("\n" + "=" * 80)
            print("STEP 2: SHENG TRANSLATION (Wanjiku Persona Detected)")
            print("=" * 80)

        # Detect Sheng
        is_sheng, sheng_conf, sheng_terms = detect_sheng(user_query)

        if is_sheng:
            translation_result = translate_to_formal(user_query, llm_function)
            formal_query = translation_result["formal_query"]
            pipeline_steps.append(
                f"Translated Sheng → Formal ({translation_result['method']})"
            )

            if verbose:
                print(f"Sheng Detected: Yes (confidence: {sheng_conf:.2f})")
                print(f"Detected Terms: {', '.join(sheng_terms)}")
                print(f"Original Query: {user_query}")
                print(f"Formal Query: {formal_query}")

            metadata["sheng_detected"] = True
            metadata["sheng_confidence"] = sheng_conf
            metadata["sheng_terms"] = sheng_terms
        else:
            pipeline_steps.append("No Sheng detected - using original query")
            if verbose:
                print("No significant Sheng detected - proceeding with original query")
            metadata["sheng_detected"] = False
    else:
        if verbose:
            print("\n" + "=" * 80)
            print(
                f"STEP 2: SKIPPED (Non-Wanjiku Persona: {intent_result['query_type']})"
            )
            print("=" * 80)
        pipeline_steps.append("No translation needed - formal user")
        metadata["sheng_detected"] = False

    # ========================================================================
    # STEP 3: RAG RETRIEVAL
    # ========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 3: RAG RETRIEVAL")
        print("=" * 80)
        print(f"Search Query: {formal_query}")

    raw_answer = rag_function(formal_query)
    pipeline_steps.append("Retrieved answer from RAG")

    if verbose:
        print(
            f"Retrieved Answer: {raw_answer[:200]}..."
            if len(raw_answer) > 200
            else raw_answer
        )

    # ========================================================================
    # STEP 4: PERSONALIZE RESPONSE
    # ========================================================================
    final_answer = raw_answer

    if intent_result["query_type"] == "wanjiku" and metadata.get(
        "sheng_detected", False
    ):
        if verbose:
            print("\n" + "=" * 80)
            print("STEP 4: RESPONSE PERSONALIZATION (Sheng Re-injection)")
            print("=" * 80)

        sheng_response = translate_to_sheng(
            user_query,
            raw_answer,
            llm_function,
            detected_style=intent_result["detected_language"],
        )
        final_answer = sheng_response["sheng_response"]
        pipeline_steps.append("Re-injected Sheng style into response")

        if verbose:
            print(f"Personalized Response: {final_answer}")

    elif intent_result["query_type"] == "wakili":
        # For legal professionals, ensure citations and formal tone
        pipeline_steps.append("Formatted for legal professional (formal)")
        if verbose:
            print("\n" + "=" * 80)
            print("STEP 4: LEGAL FORMATTING (Wakili Persona)")
            print("=" * 80)
            print("Maintaining formal tone and adding citations...")

    elif intent_result["query_type"] == "mwanahabari":
        # For journalists, ensure data and statistics are highlighted
        pipeline_steps.append("Formatted for journalist (data-focused)")
        if verbose:
            print("\n" + "=" * 80)
            print("STEP 4: DATA FORMATTING (Mwanahabari Persona)")
            print("=" * 80)
            print("Highlighting statistics and trends...")

    # ========================================================================
    # STEP 5: RETURN RESULTS
    # ========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"\nFinal Answer:\n{final_answer}\n")

    return {
        "query": user_query,
        "intent": intent_result,
        "translation": translation_result,
        "formal_query": formal_query,
        "raw_answer": raw_answer,
        "final_answer": final_answer,
        "pipeline_steps": pipeline_steps,
        "metadata": metadata,
    }


# ============================================================================
# EXAMPLE USE CASES
# ============================================================================


def example_wanjiku_query():
    """Example: Ordinary citizen asking in Sheng"""
    print("\n" + "🧑‍🌾 EXAMPLE 1: WANJIKU (Ordinary Citizen)")
    print("=" * 80)

    result = process_amaniquery(
        "Kanjo wameamua nini kuhusu parking doh? Naskia wameongeza bana!", verbose=True
    )

    return result


def example_wakili_query():
    """Example: Legal professional asking formally"""
    print("\n" + "⚖️  EXAMPLE 2: WAKILI (Legal Professional)")
    print("=" * 80)

    result = process_amaniquery(
        "Can you provide the verbatim text of Section 3(b) of the Finance Act 2023 and any subsequent amendments?",
        verbose=True,
    )

    return result


def example_mwanahabari_query():
    """Example: Journalist asking for data"""
    print("\n" + "📰 EXAMPLE 3: MWANAHABARI (Journalist)")
    print("=" * 80)

    result = process_amaniquery(
        "What was the MP attendance rate for the Finance Committee in Q3 2024? I need the statistics breakdown.",
        verbose=True,
    )

    return result


def example_batch_queries():
    """Process multiple queries and show summary"""
    print("\n" + "📊 BATCH PROCESSING EXAMPLE")
    print("=" * 80)

    test_queries = [
        "Kanjo wameamua nini kuhusu parking doh?",
        "Section 12 of the Constitution regarding devolution",
        "MP attendance statistics for 2024",
        "Bunge wanapanga kuongeza tax ya mat ama?",
        "What did the Starehe MP say about healthcare funding?",
    ]

    results = []
    for query in test_queries:
        result = process_amaniquery(query, verbose=False)
        results.append(result)

        print(f"\nQuery: {query}")
        print(f"  → Intent: {result['intent']['query_type']}")
        print(f"  → Sheng Detected: {result['metadata'].get('sheng_detected', False)}")
        print(f"  → Steps: {' → '.join(result['pipeline_steps'])}")

    return results


# ============================================================================
# FASTAPI INTEGRATION TEMPLATE
# ============================================================================


def create_fastapi_app():
    """
    Template for FastAPI integration

    Usage:
        app = create_fastapi_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="AmaniQuery v2.0 API")

    class QueryRequest(BaseModel):
        query: str
        verbose: bool = False

    @app.post("/query")
    async def query_endpoint(request: QueryRequest):
        """Process a user query through the complete pipeline"""
        try:
            result = process_amaniquery(request.query, verbose=request.verbose)

            return {
                "success": True,
                "answer": result["final_answer"],
                "intent": result["intent"]["query_type"],
                "pipeline": result["pipeline_steps"],
                "metadata": result["metadata"],
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": "2.0"}

    return app


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AMANIQUERY v2.0 - COMPLETE INTEGRATION EXAMPLES")
    print("=" * 80)

    # Run examples
    example_wanjiku_query()
    print("\n\n")

    example_wakili_query()
    print("\n\n")

    example_mwanahabari_query()
    print("\n\n")

    example_batch_queries()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Replace mock_gemini_llm() with actual Gemini/Groq API")
    print("2. Replace mock_rag_search() with actual RAG pipeline")
    print("3. Import this into your FastAPI application")
    print("4. Test with real parliamentary data")
    print("=" * 80)
