"""
AmaniQuery Engine v2.0 - Intent Router Module
==============================================

Zero-shot classification system that routes incoming queries to the correct user persona:
- wanjiku: ordinary Kenyan citizen (curious, non-expert, uses Sheng/mixed language)
- wakili: lawyer, judge, legal academic (asks for clauses, citations, precedents)
- mwanahabari: journalist, researcher, analyst (asks for trends, data, statistics)

The classifier runs on fast models (gemini-1.5-flash or Groq's llama-3-70b) and returns
structured JSON output with query type, confidence, detected language, and reasoning.

Usage:
    from Module4_NiruAPI.agents.intent_router import classify_query

    result = classify_query("Naskia kuna sheria mpya za motorbikes, ni nini inasema?")
    # Returns: {"query_type": "wanjiku", "confidence": 0.9, ...}
"""

import json
from typing import Dict, Any, Literal
from enum import Enum


class QueryType(str, Enum):
    """User persona types for query classification"""

    WANJIKU = "wanjiku"  # Ordinary citizen
    WAKILI = "wakili"  # Legal professional
    MWANAHABARI = "mwanahabari"  # Journalist/researcher


class DetectedLanguage(str, Enum):
    """Language detection categories"""

    ENGLISH = "en"
    SWAHILI = "sw"
    SHENG = "sheng"
    MIXED = "mixed"


# ============================================================================
# ZERO-SHOT CLASSIFICATION PROMPT
# ============================================================================

INTENT_ROUTER_SYSTEM_PROMPT = """You are an intelligent query classifier for AmaniQuery, a Kenyan civic AI system that helps citizens understand parliamentary proceedings, laws, and government data.

Your task is to classify EVERY incoming query into EXACTLY ONE of these three user personas:

1. **wanjiku** - Ordinary Kenyan citizen
   - Characteristics: curious but non-expert, wants simple explanations
   - Language: often uses Sheng (Kenyan slang), mixed Swahili-English, informal tone
   - Questions about: how laws affect daily life, what MPs said about local issues, simple translations
   - Examples: "Naskia kuna tax mpya?", "What did Raila say about housing?", typos common
   
2. **wakili** - Legal professional (lawyer, judge, legal academic)
   - Characteristics: expert-level legal knowledge, formal queries
   - Language: formal English or Swahili, precise legal terminology
   - Questions about: specific clauses, precedents, citations, legal interpretations, amendments
   - Examples: "Section 3(b) of the Finance Act 2023", "judicial precedent on land tenure"
   
3. **mwanahabari** - Journalist, researcher, or analyst
   - Characteristics: looking for data, trends, evidence, accountability
   - Language: professional but accessible, can be English or Swahili
   - Questions about: voting records, attendance statistics, timelines, trends, comparative data
   - Examples: "MP attendance rates Q3 2024", "voting patterns on finance bills", "budget allocation trends"

**EDGE CASES TO WATCH:**
- A law student asking casually → still "wakili" if using legal terminology
- A journalist using Sheng → could still be "mwanahabari" if asking for data/trends
- A citizen asking about "Section X" → if asking for practical impact, likely "wanjiku"
- Mixed language doesn't automatically mean "wanjiku" - check the CONTENT and INTENT

**OUTPUT FORMAT (strict JSON):**
{
  "query_type": "wanjiku" | "wakili" | "mwanahabari",
  "confidence": 0.0-1.0,
  "detected_language": "en" | "sw" | "sheng" | "mixed",
  "reasoning": "One sentence explaining the classification"
}

**RULES:**
1. Respond ONLY with valid JSON, no markdown or extra text
2. Base classification on INTENT and CONTENT, not just language
3. Confidence > 0.8 for clear cases, 0.5-0.8 for edge cases, < 0.5 for ambiguous
4. "sheng" = heavy slang mixing, "mixed" = code-switching Eng/Swa, "sw" = pure Swahili, "en" = pure English
5. When uncertain, default to "wanjiku" (most common user type)

Now classify the following query:"""


# ============================================================================
# FEW-SHOT EXAMPLES (15 diverse cases)
# ============================================================================

FEW_SHOT_EXAMPLES = [
    # === WANJIKU EXAMPLES (ordinary citizens) ===
    {
        "query": "Naskia kuna tax mpya kwa bodaboda, ni ukweli ama uongo?",
        "classification": {
            "query_type": "wanjiku",
            "confidence": 0.95,
            "detected_language": "sheng",
            "reasoning": "Informal Sheng question asking if motorcycle tax rumors are true, typical wanjiku concern about daily impact",
        },
    },
    {
        "query": "What did Raila say about the housing levy last week?",
        "classification": {
            "query_type": "wanjiku",
            "confidence": 0.9,
            "detected_language": "en",
            "reasoning": "Citizen asking what a politician said about a current issue affecting ordinary people",
        },
    },
    {
        "query": "Hii sheria ya Finance Act inasema nini kuhusu mama mboga?",
        "classification": {
            "query_type": "wanjiku",
            "confidence": 0.92,
            "detected_language": "mixed",
            "reasoning": "Mixed language asking how the Finance Act affects small traders (mama mboga), practical concern",
        },
    },
    {
        "query": "Why is the goverment taxing us so much bana??? This is to much!!",
        "classification": {
            "query_type": "wanjiku",
            "confidence": 0.88,
            "detected_language": "en",
            "reasoning": "Emotional, informal complaint with typos (goverment, to much), typical frustrated citizen query",
        },
    },
    {
        "query": "Niambie tu kwa simple Kiswahili, hii bill ya climate change inasema nn?",
        "classification": {
            "query_type": "wanjiku",
            "confidence": 0.93,
            "detected_language": "mixed",
            "reasoning": "Explicitly requesting simple Swahili explanation, non-expert wanting accessible information",
        },
    },
    # === WAKILI EXAMPLES (legal professionals) ===
    {
        "query": "Can you provide the full text of Section 3(b) of the Finance Act 2023 and any subsequent amendments?",
        "classification": {
            "query_type": "wakili",
            "confidence": 0.98,
            "detected_language": "en",
            "reasoning": "Precise request for specific legal clause and amendments, formal legal terminology",
        },
    },
    {
        "query": "What is the judicial precedent regarding land tenure disputes in Kenya following the 2010 Constitution?",
        "classification": {
            "query_type": "wakili",
            "confidence": 0.97,
            "detected_language": "en",
            "reasoning": "Expert-level query about judicial precedent and constitutional law, typical legal academic question",
        },
    },
    {
        "query": "Je, kifungu cha 47 katika Katiba kinasema nini kuhusu ugatuzi?",
        "classification": {
            "query_type": "wakili",
            "confidence": 0.95,
            "detected_language": "sw",
            "reasoning": "Formal Swahili asking about specific constitutional article on devolution, legal professional tone",
        },
    },
    {
        "query": "I need the verbatim hansard record of the debate on clause 12 during the second reading.",
        "classification": {
            "query_type": "wakili",
            "confidence": 0.96,
            "detected_language": "en",
            "reasoning": "Request for verbatim parliamentary record of specific clause debate, legal research question",
        },
    },
    {
        "query": "hey quick Q - wat does section 23A say about public procurement? need it for a case",
        "classification": {
            "query_type": "wakili",
            "confidence": 0.85,
            "detected_language": "en",
            "reasoning": "Informal language but asking for specific legal section for case work, law student or junior lawyer",
        },
    },
    # === MWANAHABARI EXAMPLES (journalists/researchers) ===
    {
        "query": "What is the MP attendance rate for the Finance Committee in Q3 2024?",
        "classification": {
            "query_type": "mwanahabari",
            "confidence": 0.97,
            "detected_language": "en",
            "reasoning": "Specific request for statistical data on MP attendance over a time period, researcher/journalist query",
        },
    },
    {
        "query": "Nipe data ya voting patterns kwa county MPs on healthcare bills from 2020-2024",
        "classification": {
            "query_type": "mwanahabari",
            "confidence": 0.94,
            "detected_language": "mixed",
            "reasoning": "Request for voting data trends over time period, analytical journalism or research question",
        },
    },
    {
        "query": "How many bills has the National Assembly passed this session compared to the last three sessions?",
        "classification": {
            "query_type": "mwanahabari",
            "confidence": 0.96,
            "detected_language": "en",
            "reasoning": "Comparative statistical analysis request across multiple sessions, typical investigative journalism",
        },
    },
    {
        "query": "Timeline ya budget allocation kwa health sector toka 2018 - nataka trends na figures",
        "classification": {
            "query_type": "mwanahabari",
            "confidence": 0.95,
            "detected_language": "mixed",
            "reasoning": "Request for timeline data with trends and figures, data journalism or policy research",
        },
    },
    # === EDGE CASES ===
    {
        "query": "Alafu hiyo Section 12 inasema aje exactly? Sisi wananchi tunakaa confused bana",
        "classification": {
            "query_type": "wanjiku",
            "confidence": 0.78,
            "detected_language": "sheng",
            "reasoning": "Although mentions Section 12, the informal Sheng and self-identification as 'wananchi' indicates ordinary citizen seeking clarity",
        },
    },
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_full_prompt(user_query: str) -> str:
    """
    Constructs the complete prompt including system instructions and user query.

    Args:
        user_query: The incoming user query to classify

    Returns:
        Complete prompt string ready for LLM inference
    """
    return f"{INTENT_ROUTER_SYSTEM_PROMPT}\n\n{user_query}"


def get_few_shot_prompt(user_query: str, num_examples: int = 5) -> str:
    """
    Constructs a few-shot prompt with example classifications.

    Args:
        user_query: The incoming user query to classify
        num_examples: Number of example classifications to include (default: 5)

    Returns:
        Complete few-shot prompt string
    """
    examples_text = "\n\n".join(
        [
            f"Example {i+1}:\nQuery: {ex['query']}\nOutput: {json.dumps(ex['classification'], indent=2)}"
            for i, ex in enumerate(FEW_SHOT_EXAMPLES[:num_examples])
        ]
    )

    return f"""{INTENT_ROUTER_SYSTEM_PROMPT}

Here are some example classifications:

{examples_text}

Now classify this query:
{user_query}"""


def validate_classification_output(response: Dict[str, Any]) -> bool:
    """
    Validates that the LLM response matches the expected schema.

    Args:
        response: The parsed JSON response from the LLM

    Returns:
        True if valid, False otherwise
    """
    required_keys = {"query_type", "confidence", "detected_language", "reasoning"}

    if not all(key in response for key in required_keys):
        return False

    if response["query_type"] not in [qt.value for qt in QueryType]:
        return False

    if not isinstance(response["confidence"], (int, float)) or not (
        0.0 <= response["confidence"] <= 1.0
    ):
        return False

    if response["detected_language"] not in [lang.value for lang in DetectedLanguage]:
        return False

    if not isinstance(response["reasoning"], str) or len(response["reasoning"]) == 0:
        return False

    return True


def classify_query(
    query: str,
    llm_function: callable = None,
    use_few_shot: bool = False,
    num_examples: int = 5,
) -> Dict[str, Any]:
    """
    Classifies a user query into one of three personas.

    Args:
        query: The user query to classify
        llm_function: A callable that takes a prompt string and returns JSON response.
                     Should be your Gemini/Groq API wrapper function.
                     If None, returns the prompt for manual testing.
        use_few_shot: Whether to include few-shot examples in the prompt
        num_examples: Number of examples to include if use_few_shot=True

    Returns:
        Dictionary with classification results matching the schema:
        {
            "query_type": str,
            "confidence": float,
            "detected_language": str,
            "reasoning": str
        }

    Example:
        >>> from Module4_NiruAPI.agents.intent_router import classify_query
        >>> def my_llm_call(prompt):
        >>>     # Your Gemini/Groq API call here
        >>>     return {"query_type": "wanjiku", "confidence": 0.9, ...}
        >>>
        >>> result = classify_query("Naskia tax inakuja?", llm_function=my_llm_call)
        >>> print(result["query_type"])  # "wanjiku"
    """
    # Construct the appropriate prompt
    if use_few_shot:
        prompt = get_few_shot_prompt(query, num_examples)
    else:
        prompt = get_full_prompt(query)

    # If no LLM function provided, return the prompt for testing
    if llm_function is None:
        return {
            "prompt": prompt,
            "note": "No LLM function provided. Use this prompt with your Gemini/Groq API.",
        }

    # Call the LLM
    try:
        response = llm_function(prompt)

        # Handle string responses that need JSON parsing
        if isinstance(response, str):
            # Remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = json.loads(response.strip())

        # Validate the response
        if not validate_classification_output(response):
            raise ValueError("Invalid classification output schema")

        return response

    except Exception as e:
        # Fallback to wanjiku with low confidence if classification fails
        return {
            "query_type": "wanjiku",
            "confidence": 0.3,
            "detected_language": "mixed",
            "reasoning": f"Classification failed ({str(e)}), defaulting to most common user type",
            "error": str(e),
        }


# ============================================================================
# INTEGRATION EXAMPLES
# ============================================================================


def example_gemini_integration():
    """
    Example showing how to integrate with Google Gemini Flash.
    Replace with your actual API setup.
    """
    import google.generativeai as genai

    # Configure your API key
    genai.configure(api_key="YOUR_API_KEY")
    model = genai.GenerativeModel("gemini-1.5-flash")

    def gemini_llm_call(prompt: str) -> Dict[str, Any]:
        """Wrapper for Gemini API calls"""
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,  # Low temperature for consistent classification
                "response_mime_type": "application/json",
            },
        )
        return json.loads(response.text)

    # Use the classifier
    result = classify_query(
        "Niambie tu kuna wabunge wangapi walikuwa absent jana?",
        llm_function=gemini_llm_call,
    )
    print(f"Query Type: {result['query_type']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Language: {result['detected_language']}")
    print(f"Reasoning: {result['reasoning']}")


def example_groq_integration():
    """
    Example showing how to integrate with Groq's Llama-3-70B.
    Replace with your actual API setup.
    """
    from groq import Groq

    client = Groq(api_key="YOUR_API_KEY")

    def groq_llm_call(prompt: str) -> Dict[str, Any]:
        """Wrapper for Groq API calls"""
        response = client.chat.completions.create(
            model="llama-3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    # Use the classifier
    result = classify_query(
        "What is the precedent for section 23 interpretation?",
        llm_function=groq_llm_call,
    )
    return result


# ============================================================================
# TESTING & VALIDATION
# ============================================================================


def test_all_examples():
    """
    Validates all few-shot examples have correct schema.
    Run this to ensure example quality.
    """
    print("Testing all few-shot examples for schema compliance...\n")

    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        classification = example["classification"]
        is_valid = validate_classification_output(classification)

        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"{status} Example {i}: {example['query'][:50]}...")

        if not is_valid:
            print(f"  Invalid classification: {classification}")

    print(f"\nTotal examples: {len(FEW_SHOT_EXAMPLES)}")
    print("All validations complete.")


if __name__ == "__main__":
    # Run validation tests
    test_all_examples()

    # Example: Get prompt without LLM call (for manual testing)
    print("\n" + "=" * 80)
    print("EXAMPLE PROMPT (Zero-Shot):")
    print("=" * 80)
    test_query = "Naskia kuna tax mpya kwa motorbikes?"
    result = classify_query(test_query)
    if "prompt" in result:
        print(result["prompt"])

    print("\n" + "=" * 80)
    print("EXAMPLE PROMPT (Few-Shot with 3 examples):")
    print("=" * 80)
    result_few_shot = classify_query(test_query, use_few_shot=True, num_examples=3)
    if "prompt" in result_few_shot:
        print(result_few_shot["prompt"][:1000] + "...\n[truncated]")
