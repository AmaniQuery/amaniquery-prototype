"""
AmaniQuery Engine v2.0 - Sheng Translator Layer
================================================

Bidirectional translator that converts Kenyan Sheng/slang to formal institutional
language for accurate RAG retrieval, then re-injects the user's original vibe
into responses.

Flow:
1. User query in Sheng → Formal search query (for RAG)
2. RAG retrieves formal answer → Sheng-infused response (for user)

Example:
  User: "Kanjo wameamua nini kuhusu parking doh?"
  → Internal: "Nairobi City County latest resolution on parking fees"
  → Response: "Kanjo wameongeza parking fees kwa town centre to KES 300 per day..."

Usage:
    from Module4_NiruAPI.agents.sheng_translator import (
        translate_to_formal,
        translate_to_sheng,
        detect_sheng
    )
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from enum import Enum


class TranslationDirection(str, Enum):
    """Direction of translation"""

    TO_FORMAL = "to_formal"
    TO_SHENG = "to_sheng"


# ============================================================================
# COMPREHENSIVE SHENG ↔ FORMAL DICTIONARY (80+ TERMS)
# ============================================================================

SHENG_FORMAL_DICTIONARY = {
    # Government & Institutions
    "kanjo": "Nairobi City County Government",
    "bunge": "Parliament of Kenya",
    "mashamba": "County Assembly",
    "serikali": "National Government of Kenya",
    "county": "County Government",
    "gavaa": "Government",
    "state house": "Office of the President",
    "statehouse": "Office of the President",
    # Political Figures
    "mhesh": "Member of Parliament",
    "mheshimiwa": "Honourable Member of Parliament",
    "MCA": "Member of County Assembly",
    "gavana": "Governor",
    "gava": "Governor",
    "rais": "President",
    "prezzo": "President",
    "DP": "Deputy President",
    "seneta": "Senator",
    "wabunge": "Members of Parliament",
    "viongozi": "leaders",
    "mbunge": "Member of Parliament",
    # Money & Finance
    "doh": "money",
    "doe": "money",
    "mullah": "money",
    "ganji": "money",
    "mbeca": "money",
    "pesa": "money",
    "doo": "money",
    "budget": "budget allocation",
    "taxes": "taxation",
    "cess": "county levy",
    # Laws & Bills
    "sheria": "law",
    "bill": "proposed legislation",
    "act": "enacted law",
    "katiba": "Constitution of Kenya",
    "finance bill": "Finance Bill",
    "amendment": "constitutional amendment",
    # Locations
    "tao": "town",
    "town": "Central Business District",
    "CBD": "Central Business District",
    "mtaa": "neighborhood",
    "hood": "constituency",
    "ghetto": "informal settlement",
    "slums": "informal settlements",
    "shags": "rural areas",
    "ocha": "rural home areas",
    # Transport
    "mat": "public service vehicle",
    "matatu": "public service vehicle",
    "boda": "motorcycle taxi",
    "bodaboda": "motorcycle taxi services",
    "nduthi": "motorcycle",
    "gari": "vehicle",
    "barabara": "road",
    "jam": "traffic congestion",
    "traffic": "traffic management",
    # Services & Infrastructure
    "maji": "water services",
    "stima": "electricity supply",
    "umeme": "electrical power",
    "wifi": "internet connectivity",
    "garbage": "solid waste management",
    "takataka": "waste management services",
    "kanju": "Nairobi City County enforcement",
    "askari": "security officers",
    "polisi": "Kenya Police Service",
    # Healthcare & Education
    "hospital": "healthcare facility",
    "hospitali": "public hospital",
    "shule": "educational institution",
    "chuo": "university",
    "college": "tertiary institution",
    "daktari": "medical practitioner",
    "mwalimu": "teacher",
    # Social Issues
    "uchizi": "corruption",
    "rushwa": "bribery",
    "corruption": "corruption and economic crimes",
    "demo": "public demonstration",
    "maandamano": "public protests",
    "protest": "public demonstration",
    "strikes": "industrial action",
    "hustler": "small-scale entrepreneur",
    "hustlers": "ordinary citizens",
    "wanjiku": "ordinary Kenyan citizen",
    "mwananchi": "citizen",
    "wananchi": "citizens",
    # Food & Agriculture
    "chakula": "food security",
    "unga": "maize flour",
    "mahindi": "maize",
    "kilimo": "agriculture",
    "wakulima": "farmers",
    "mama mboga": "vegetable vendors",
    # Housing
    "nyumba": "housing",
    "plot": "residential plot",
    "title deed": "land title",
    "rent": "rental charges",
    "kodi": "rent payment",
    # Employment
    "kazi": "employment",
    "job": "employment opportunity",
    "hustle": "income-generating activity",
    "biashara": "business enterprise",
    "duka": "retail shop",
    # Time & Events
    "saa": "hour",
    "leo": "today",
    "kesho": "tomorrow",
    "juzi": "recently",
    "siku hizi": "currently",
    "session": "parliamentary session",
    # Actions & Processes
    "kupiga kura": "voting",
    "kura": "vote",
    "uchaguzi": "election",
    "campaign": "political campaign",
    "debate": "parliamentary debate",
    "approve": "legislative approval",
    "reject": "legislative rejection",
    "pass": "enact legislation",
    # Common Verbs
    "wameamua": "has resolved",
    "wamesema": "has stated",
    "wanapanga": "is planning",
    "wanaongeza": "is increasing",
    "wanapunguza": "is reducing",
    "wamebadilisha": "has changed",
    # Questions & References
    "nini": "what",
    "aje": "how",
    "lini": "when",
    "wapi": "where",
    "nani": "who",
    "ngapi": "how much",
    "je": "question marker",
    "ama": "or",
    "au": "or",
    # Emphasis & Slang
    "si": "isn't it",
    "kwani": "why",
    "bana": "emphasis/frustration marker",
    "alafu": "and then",
    "lakini": "but",
    "kwa": "for/at",
    "ya": "of",
    "kuhusu": "about/regarding",
    "hii": "this",
    "hiyo": "that",
}

# Create reverse mapping (formal → sheng) for bidirectional translation
FORMAL_SHENG_DICTIONARY = {v.lower(): k for k, v in SHENG_FORMAL_DICTIONARY.items()}


# ============================================================================
# SHENG DETECTION PATTERNS
# ============================================================================

SHENG_INDICATORS = [
    # Common Sheng words
    r"\b(kanjo|bunge|mashamba|mhesh|doh|doe|ganji|mat|boda|nduthi)\b",
    # Swahili question words
    r"\b(nini|aje|lini|wapi|nani|ngapi|je)\b",
    # Common verbs
    r"\b(wame|wana|ame|ana)(amua|sema|panga|ongeza|punguza|badilisha)\b",
    # Emphasis markers
    r"\b(bana|si|kwani|alafu)\b",
    # Mixed Swahili-English
    r"(kwa |ya |kuhusu |hii |hiyo )",
]

SHENG_PATTERN = re.compile("|".join(SHENG_INDICATORS), re.IGNORECASE)


# ============================================================================
# TRANSLATION PROMPTS
# ============================================================================

SHENG_TO_FORMAL_PROMPT = """You are a Kenyan civic language translator for AmaniQuery, a parliamentary and government information system.

Your task is to translate a user's informal Kenyan Sheng/slang query into a formal, precise search query suitable for retrieving official government documents, hansard records, and legal texts.

**TRANSLATION RULES:**
1. Replace ALL Sheng slang with proper institutional names (use the dictionary below)
2. Convert informal phrasing to formal queries
3. Preserve the ORIGINAL INTENT and question focus
4. Remove filler words (bana, si, alafu) but keep question essence
5. Expand abbreviations to full terms
6. Output ONLY the formal query, no explanations

**SHENG → FORMAL DICTIONARY:**
{dictionary}

**EXAMPLES:**

User Query: "Kanjo wameamua nini kuhusu parking doh?"
Formal Query: "What has the Nairobi City County Government resolved regarding parking fees?"

User Query: "Mheshimiwa wa Starehe alisema aje kuhusu Finance Bill?"
Formal Query: "What did the Member of Parliament for Starehe Constituency state about the Finance Bill?"

User Query: "Bunge wanapanga kuongeza tax ya mat ama?"
Formal Query: "Is Parliament planning to increase taxation on public service vehicles?"

User Query: "Naskia serikali wanapunguza doh ya healthcare, ni ukweli?"
Formal Query: "Is it true that the National Government is reducing healthcare budget allocation?"

User Query: "Gavana wa Nairobi amesema nini kuhusu maji na stima?"
Formal Query: "What has the Nairobi County Governor stated about water services and electricity supply?"

Now translate this query:
User Query: {user_query}
Formal Query:"""


FORMAL_TO_SHENG_PROMPT = """You are a Kenyan civic AI assistant that communicates naturally with ordinary Kenyans.

Your task is to take a formal institutional answer and rephrase it in a natural, conversational tone that matches the user's original Sheng/slang style. The goal is to make government information accessible and relatable.

**RE-INJECTION RULES:**
1. Use the user's original slang terms where they appeared (e.g., if user said "kanjo", use "kanjo" not "Nairobi City County")
2. Keep the tone conversational and friendly
3. Maintain ALL factual accuracy - don't change numbers, dates, or official decisions
4. Use short sentences and simple explanations
5. Mix Sheng/Swahili/English naturally as the user did
6. Add clarifying context for complex terms
7. Keep it concise - Kenyans want quick, clear answers

**USER'S ORIGINAL QUERY:** {user_query}
**USER'S LANGUAGE STYLE:** {detected_style}

**FORMAL ANSWER:**
{formal_answer}

**YOUR SHENG-INFUSED RESPONSE:**"""


# ============================================================================
# CORE TRANSLATION FUNCTIONS
# ============================================================================


def detect_sheng(text: str) -> Tuple[bool, float, List[str]]:
    """
    Detects if text contains Kenyan Sheng or heavy slang.

    Args:
        text: The input text to analyze

    Returns:
        Tuple of (is_sheng, confidence, detected_terms)
        - is_sheng: True if Sheng detected
        - confidence: Score 0.0-1.0 based on term frequency
        - detected_terms: List of Sheng words found

    Example:
        >>> detect_sheng("Kanjo wameamua nini kuhusu parking doh?")
        (True, 0.85, ["kanjo", "wameamua", "nini", "kuhusu", "doh"])
    """
    text_lower = text.lower()
    detected_terms = []

    # Find all Sheng terms
    for sheng_term in SHENG_FORMAL_DICTIONARY.keys():
        if re.search(r"\b" + re.escape(sheng_term) + r"\b", text_lower):
            detected_terms.append(sheng_term)

    # Check regex patterns
    pattern_matches = SHENG_PATTERN.findall(text_lower)
    detected_terms.extend(
        [match[0] if isinstance(match, tuple) else match for match in pattern_matches]
    )

    # Remove duplicates
    detected_terms = list(set(detected_terms))

    # Calculate confidence based on term density
    word_count = len(text.split())
    if word_count == 0:
        confidence = 0.0
    else:
        # Confidence = (sheng_terms / total_words) normalized to 0-1
        confidence = min(len(detected_terms) / word_count * 2, 1.0)

    is_sheng = confidence > 0.2  # Threshold: >20% Sheng terms

    return is_sheng, confidence, detected_terms


def get_translation_dictionary_str() -> str:
    """Returns formatted dictionary string for prompts"""
    dict_items = [
        f'"{sheng}" → "{formal}"'
        for sheng, formal in sorted(SHENG_FORMAL_DICTIONARY.items())
    ]
    return "\n".join(dict_items)


def translate_to_formal(
    user_query: str, llm_function: callable = None, use_dictionary_only: bool = False
) -> Dict[str, any]:
    """
    Translates Sheng query to formal search query.

    Args:
        user_query: The user's original query in Sheng/slang
        llm_function: Callable that takes prompt and returns text response
        use_dictionary_only: If True, only does dictionary replacement (no LLM)

    Returns:
        {
            "original_query": str,
            "formal_query": str,
            "detected_sheng": bool,
            "confidence": float,
            "detected_terms": List[str],
            "method": "llm" | "dictionary" | "passthrough"
        }

    Example:
        >>> translate_to_formal("Kanjo wameamua nini kuhusu parking doh?", my_llm)
        {
            "original_query": "Kanjo wameamua nini kuhusu parking doh?",
            "formal_query": "What has Nairobi City County resolved regarding parking fees?",
            "detected_sheng": True,
            "confidence": 0.75,
            "detected_terms": ["kanjo", "wameamua", "nini", "kuhusu", "doh"],
            "method": "llm"
        }
    """
    # Detect Sheng
    is_sheng, confidence, detected_terms = detect_sheng(user_query)

    # If no Sheng detected, return original query
    if not is_sheng:
        return {
            "original_query": user_query,
            "formal_query": user_query,
            "detected_sheng": False,
            "confidence": confidence,
            "detected_terms": [],
            "method": "passthrough",
        }

    # Method 1: Dictionary-only replacement
    if use_dictionary_only or llm_function is None:
        formal_query = user_query
        for sheng, formal in SHENG_FORMAL_DICTIONARY.items():
            # Use word boundaries for accurate replacement
            pattern = r"\b" + re.escape(sheng) + r"\b"
            formal_query = re.sub(pattern, formal, formal_query, flags=re.IGNORECASE)

        return {
            "original_query": user_query,
            "formal_query": formal_query,
            "detected_sheng": True,
            "confidence": confidence,
            "detected_terms": detected_terms,
            "method": "dictionary",
        }

    # Method 2: LLM-based translation
    try:
        prompt = SHENG_TO_FORMAL_PROMPT.format(
            dictionary=get_translation_dictionary_str(), user_query=user_query
        )

        formal_query = llm_function(prompt).strip()

        # Remove any markdown formatting
        formal_query = formal_query.replace("**", "").replace("```", "")

        return {
            "original_query": user_query,
            "formal_query": formal_query,
            "detected_sheng": True,
            "confidence": confidence,
            "detected_terms": detected_terms,
            "method": "llm",
        }

    except Exception as e:
        # Fallback to dictionary method
        return translate_to_formal(user_query, use_dictionary_only=True)


def translate_to_sheng(
    user_query: str,
    formal_answer: str,
    llm_function: callable,
    detected_style: str = "mixed Swahili-English with Sheng slang",
) -> Dict[str, any]:
    """
    Translates formal institutional answer back to Sheng-infused response.

    Args:
        user_query: The user's original query (for style matching)
        formal_answer: The formal answer from RAG system
        llm_function: Callable that takes prompt and returns text response
        detected_style: Description of user's language style

    Returns:
        {
            "original_query": str,
            "formal_answer": str,
            "sheng_response": str,
            "style": str
        }

    Example:
        >>> translate_to_sheng(
        ...     "Kanjo wameamua nini kuhusu parking doh?",
        ...     "Nairobi City County has increased parking fees to KES 300.",
        ...     my_llm
        ... )
        {
            "sheng_response": "Kanjo wameongeza parking fees kwa town centre to KES 300 per day..."
        }
    """
    try:
        prompt = FORMAL_TO_SHENG_PROMPT.format(
            user_query=user_query,
            detected_style=detected_style,
            formal_answer=formal_answer,
        )

        sheng_response = llm_function(prompt).strip()

        return {
            "original_query": user_query,
            "formal_answer": formal_answer,
            "sheng_response": sheng_response,
            "style": detected_style,
        }

    except Exception as e:
        # Fallback: return formal answer
        return {
            "original_query": user_query,
            "formal_answer": formal_answer,
            "sheng_response": formal_answer,
            "style": detected_style,
            "error": str(e),
        }


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================


def full_translation_pipeline(
    user_query: str, rag_function: callable, llm_function: callable
) -> Dict[str, any]:
    """
    Complete pipeline: Sheng query → Formal search → RAG → Sheng response

    Args:
        user_query: User's original query
        rag_function: Function that takes formal query and returns formal answer
        llm_function: LLM function for translation

    Returns:
        {
            "user_query": str,
            "formal_query": str,
            "formal_answer": str,
            "final_response": str,
            "detected_sheng": bool,
            "pipeline_steps": List[str]
        }

    Example:
        >>> def my_rag(query):
        ...     return "Nairobi City County increased parking fees to KES 300."
        >>>
        >>> result = full_translation_pipeline(
        ...     "Kanjo wameamua nini kuhusu parking doh?",
        ...     my_rag,
        ...     my_llm
        ... )
        >>> print(result["final_response"])
        "Kanjo wameongeza parking fees to KES 300..."
    """
    pipeline_steps = []

    # Step 1: Translate to formal
    translation = translate_to_formal(user_query, llm_function)
    pipeline_steps.append(f"Translated Sheng → Formal: {translation['method']}")

    formal_query = translation["formal_query"]

    # Step 2: RAG retrieval
    formal_answer = rag_function(formal_query)
    pipeline_steps.append("Retrieved formal answer from RAG")

    # Step 3: Translate back to Sheng (only if original was Sheng)
    if translation["detected_sheng"]:
        sheng_response = translate_to_sheng(user_query, formal_answer, llm_function)
        final_response = sheng_response["sheng_response"]
        pipeline_steps.append("Re-injected Sheng style into response")
    else:
        final_response = formal_answer
        pipeline_steps.append("No Sheng re-injection (formal query)")

    return {
        "user_query": user_query,
        "formal_query": formal_query,
        "formal_answer": formal_answer,
        "final_response": final_response,
        "detected_sheng": translation["detected_sheng"],
        "detected_terms": translation.get("detected_terms", []),
        "pipeline_steps": pipeline_steps,
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_dictionary_stats() -> Dict[str, any]:
    """Returns statistics about the Sheng dictionary"""
    return {
        "total_terms": len(SHENG_FORMAL_DICTIONARY),
        "categories": {
            "government": len(
                [
                    k
                    for k in SHENG_FORMAL_DICTIONARY
                    if any(
                        x in SHENG_FORMAL_DICTIONARY[k].lower()
                        for x in ["government", "parliament", "county", "assembly"]
                    )
                ]
            ),
            "money": len(
                [
                    k
                    for k in SHENG_FORMAL_DICTIONARY
                    if "money" in SHENG_FORMAL_DICTIONARY[k].lower()
                    or k in ["doh", "doe", "ganji", "mullah", "mbeca"]
                ]
            ),
            "transport": len(
                [
                    k
                    for k in SHENG_FORMAL_DICTIONARY
                    if any(x in k for x in ["mat", "boda", "nduthi", "gari"])
                ]
            ),
            "people": len(
                [
                    k
                    for k in SHENG_FORMAL_DICTIONARY
                    if any(
                        x in SHENG_FORMAL_DICTIONARY[k].lower()
                        for x in ["member", "governor", "president", "senator"]
                    )
                ]
            ),
        },
        "sample_terms": list(SHENG_FORMAL_DICTIONARY.items())[:10],
    }


def export_dictionary_json(filepath: str = "sheng_dictionary.json"):
    """Exports the dictionary to JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(SHENG_FORMAL_DICTIONARY, f, indent=2, ensure_ascii=False)
    return f"Dictionary exported to {filepath}"


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SHENG TRANSLATOR - DICTIONARY STATS")
    print("=" * 80)
    stats = get_dictionary_stats()
    print(f"\nTotal terms: {stats['total_terms']}")
    print("\nCategories:")
    for cat, count in stats["categories"].items():
        print(f"  - {cat}: {count} terms")

    print("\n" + "=" * 80)
    print("SHENG DETECTION TESTS")
    print("=" * 80)

    test_queries = [
        "Kanjo wameamua nini kuhusu parking doh?",
        "What did the MP for Starehe say about the Finance Bill?",
        "Bunge wanapanga kuongeza tax ya mat ama?",
        "Hii sheria ya housing levy inasema aje?",
        "When is the next parliamentary session?",
    ]

    for query in test_queries:
        is_sheng, conf, terms = detect_sheng(query)
        print(f'\nQuery: "{query}"')
        print(f"  Sheng Detected: {is_sheng}")
        print(f"  Confidence: {conf:.2f}")
        print(f"  Terms: {', '.join(terms) if terms else 'None'}")

    print("\n" + "=" * 80)
    print("DICTIONARY-ONLY TRANSLATION TEST")
    print("=" * 80)

    test_query = "Kanjo wameamua nini kuhusu parking doh na mat?"
    result = translate_to_formal(test_query, use_dictionary_only=True)
    print(f"\nOriginal: {result['original_query']}")
    print(f"Formal: {result['formal_query']}")
    print(f"Method: {result['method']}")
    print(f"Detected terms: {', '.join(result['detected_terms'])}")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
