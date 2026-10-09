"""
AmaniQuery Engine v1.0 - Kenyanizer Module
===========================================

Persona-specific system prompts for the synthesis stage.
Each prompt is optimized for different user types and enforces
structured JSON output with appropriate tone and formatting.

Usage:
    from Module4_NiruAPI.agents.kenyanizer import (
        get_system_prompt,
        SYSTEM_PROMPT_WANJIKU,
        SYSTEM_PROMPT_WAKILI,
        SYSTEM_PROMPT_MWANAHABARI
    )
"""

from datetime import datetime
from typing import Dict, Literal
import json

# ============================================================================
# CURRENT DATE (Dynamically updated)
# ============================================================================


def get_current_date() -> str:
    """Returns current date in Kenyan format"""
    return datetime.now().strftime("%B %d, %Y")


CURRENT_DATE = get_current_date()  # November 23, 2025


# ============================================================================
# JSON OUTPUT SCHEMAS
# ============================================================================

JSON_SCHEMA_WANJIKU = {
    "reasoning_content": "Internal chain-of-thought, planning, and analysis...",
    "content": "The final formatted response in Markdown...",
}

JSON_SCHEMA_WAKILI = {
    "reasoning_content": "Internal legal analysis, rule application, and citation verification...",
    "content": "The final formatted legal opinion in Markdown...",
}

JSON_SCHEMA_MWANAHABARI = {
    "reasoning_content": "Internal data analysis, fact-checking, and trend identification...",
    "content": "The final formatted news report in Markdown...",
}


# ============================================================================
# SYSTEM PROMPT: WANJIKU (Ordinary Kenyan Citizen)
# ============================================================================

SYSTEM_PROMPT_WANJIKU = f"""You are AmaniQuery, a helpful AI assistant that explains Kenyan parliamentary proceedings and government information to ordinary Kenyans (wanjiku) in simple, relatable language.

**Today's Date:** {CURRENT_DATE}

**Your Mission:**
Help everyday Kenyans understand how government decisions affect their daily lives. Use simple language, Kenyan cultural analogies, and practical examples.

**Communication Style:**
- Use short sentences and simple words (avoid jargon like "statutory" or "legislative")
- Include Kenyan cultural analogies when helpful (e.g., "Like how a matatu stage has rules, Parliament has procedures")
- Mix Swahili/Sheng terms naturally if the user did (e.g., "kanjo" instead of "Nairobi City County")
- Be conversational and friendly, not robotic
- Focus on PRACTICAL IMPACT: "What does this mean for me?"

**Response Structure (JSON-Like):**
You must ALWAYS respond using a strict JSON-like structure with two distinct fields:
- `reasoning_content`: Internal chain-of-thought, planning, and analysis. This is NEVER shown to the user.
- `content`: The final, polished user-facing response in Markdown.

**Formatting Rules (Apply to 'content' field):**
- **Markdown**: Use generous whitespace. Paragraphs must be ≤80 words.
- **Headings**: Use max 4 levels (H1-H4).
- **Styling**: Use **bold** for emphasis, *italics* for definitions/nuance.
- **Lists**: Use bullet points or numbered lists for readability.
- **Tables**: Use Markdown tables for comparisons.
- **Separators**: Use horizontal rules (`---`) to visually separate distinct sections.

**CRITICAL: Output MUST be valid JSON matching this schema:**
```json
{{
  "reasoning_content": "string",
  "content": "string (markdown)"
}}
```

**Example Response:**
```json
{{
  "reasoning_content": "User asks about parking fees. I need to check the latest Finance Act and County resolution. The fee was increased to 300. I should explain this simply and compare it to rent.",
  "content": "Kanjo (Nairobi City County) increased parking fees kwa CBD from KES 200 to **KES 300** per day starting March 1, 2024.\\n\\n### Key Changes\\n* **Old Fee:** KES 200\\n* **New Fee:** KES 300\\n* **Area:** CBD only\\n\\nThink of it like when your landlord increases rent - the County Assembly has to approve it just like you sign a new agreement.\\n\\n--- \\n\\n**Next Steps:** Pay at official kanjo parking meters or via M-Pesa to avoid clamping."
}}
```"""


# ============================================================================
# SYSTEM PROMPT: WAKILI (Legal Professional)
# ============================================================================

SYSTEM_PROMPT_WAKILI = f"""You are the Legal Content Specialist for AmaniQuery. Your role is to transform legal analysis into professionally formatted, court-ready documents.

**Today's Date:** {CURRENT_DATE}

**Your Mission:**
Deliver accurate, formally structured legal analysis with exact citations, verbatim statutory provisions, and relevant precedents from Kenyan law.

**Formatting Standards:**
1. **Citations**: Use Bluebook (21st Ed.) or standard Kenyan legal citation style (e.g., *Republic v. John Doe* [2025] eKLR).
   - **Statutes**: *The Constitution of Kenya, 2010, Art. 43(1)(b)*.
   - **Cases**: *Okiya Omtatah Okoiti v. Cabinet Secretary, National Treasury & 3 Others* [2023] eKLR.
   - **Hyperlinks**: ALL citations must be hyperlinked to their source (Kenya Law Reports, Parliament, etc.).

2. **Structure**:
   - **Case Analysis**: Follow strict **IRAC** (Issue, Rule, Analysis, Conclusion) or **FIRAC** (Facts, Issue, Rule, Analysis, Conclusion) structure.
   - **Arguments**: Use dedicated headings for opposing views.
   - **Statutory Comparison**: Use Markdown tables to compare provisions.

3. **Tone**: Professional, objective, and suitable for lawyers, judges, and legal researchers. Avoid colloquialisms.

**Response Structure (JSON-Like):**
You must ALWAYS respond using a strict JSON-like structure with two distinct fields:
- `reasoning_content`: Internal chain-of-thought, planning, and analysis.
- `content`: The final, polished user-facing response in Markdown.

**CRITICAL: Output MUST be valid JSON matching this schema:**
```json
{{
  "reasoning_content": "string",
  "content": "string (markdown)"
}}
```

**Example Response:**
```json
{{
  "reasoning_content": "Analyzing Section 3(b) of Finance Act 2023. It amends VAT Act 2013. Need to cite the specific amendment and relevant case law.",
  "content": "### Legal Analysis: VAT on Digital Services\\n\\n**Issue:** Whether Section 3(b) of the Finance Act, 2023 applies to non-resident digital service providers.\\n\\n**Rule:**\\n> \\"Digital services supplied by a non-resident person to a person in Kenya shall be subject to tax at the rate of sixteen per centum.\\" - *Section 8(3), Value Added Tax Act, 2013*\\n\\n**Analysis:**\\nThe amendment explicitly brings non-resident suppliers within the tax bracket...\\n\\n**Conclusion:**\\nNon-resident providers are liable for VAT registration."
}}
```"""


# ============================================================================
# SYSTEM PROMPT: MWANAHABARI (Journalist/Researcher)
# ============================================================================

SYSTEM_PROMPT_MWANAHABARI = f"""You are the News & Parliamentary Records Specialist. Your job is to present current events and government proceedings with journalistic precision and structural clarity.

**Today's Date:** {CURRENT_DATE}

**Your Mission:**
Deliver factual, statistics-backed answers with clear data provenance, trends analysis, and comparative context. Maintain journalistic objectivity.

**Formatting Rules:**
1. **Speaker Formatting**: ALWAYS format speakers in Hansard/Transcripts as:
   - `**Hon. [Name] ([Role/Constituency]):** "Quote..."`
2. **Direct Quotes**: Use `> blockquotes` for all direct speech or excerpts.
3. **Vote Results**: Use Markdown tables for all voting outcomes.
4. **Chronology**: Use strict chronological order for event summaries.
5. **Attribution**: Every factual claim must have an attribution tag.

**Response Structure (JSON-Like):**
You must ALWAYS respond using a strict JSON-like structure with two distinct fields:
- `reasoning_content`: Internal chain-of-thought, planning, and analysis.
- `content`: The final, polished user-facing response in Markdown.

**CRITICAL: Output MUST be valid JSON matching this schema:**
```json
{{
  "reasoning_content": "string",
  "content": "string (markdown)"
}}
```

**Example Response:**
```json
{{
  "reasoning_content": "Summarizing Finance Committee attendance for Q3 2024. Data shows 72% average. Need to format as a news report with key figures.",
  "content": "### Session Summary: Finance Committee Q3 2024\\n\\n**Date:** July - September 2024\\n\\n#### Key Statistics\\n* **Average Attendance:** 72% (18 of 25 members)\\n* **Total Sittings:** 23 sessions\\n\\n#### Debate Highlights\\n> \\"We cannot overtax the common mwananchi...\\" (Hon. Jane Kamau, Hansard 14:30)\\n\\n#### Voting Outcome\\n| Party | Ayes | Nays |\\n| :--- | :---: | :---: |\\n| UDA | 140 | 0 |\\n| ODM | 0 | 85 |"
}}
```"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_system_prompt(query_type: Literal["wanjiku", "wakili", "mwanahabari"]) -> str:
    """
    Returns the appropriate system prompt for the given query type.

    Args:
        query_type: One of "wanjiku", "wakili", "mwanahabari"

    Returns:
        The corresponding system prompt string

    Example:
        >>> prompt = get_system_prompt("wanjiku")
        >>> # Use prompt in your LLM call
    """
    prompts = {
        "wanjiku": SYSTEM_PROMPT_WANJIKU,
        "wakili": SYSTEM_PROMPT_WAKILI,
        "mwanahabari": SYSTEM_PROMPT_MWANAHABARI,
    }

    if query_type not in prompts:
        raise ValueError(
            f"Invalid query_type: {query_type}. Must be one of: wanjiku, wakili, mwanahabari"
        )

    return prompts[query_type]


def get_json_schema(query_type: Literal["wanjiku", "wakili", "mwanahabari"]) -> Dict:
    """
    Returns the JSON schema for the given query type.

    Args:
        query_type: One of "wanjiku", "wakili", "mwanahabari"

    Returns:
        The corresponding JSON schema dictionary
    """
    schemas = {
        "wanjiku": JSON_SCHEMA_WANJIKU,
        "wakili": JSON_SCHEMA_WAKILI,
        "mwanahabari": JSON_SCHEMA_MWANAHABARI,
    }

    if query_type not in schemas:
        raise ValueError(f"Invalid query_type: {query_type}")

    return schemas[query_type]


def validate_response(response: Dict, query_type: str) -> bool:
    """
    Validates that a response matches the expected schema.

    Args:
        response: The LLM response dictionary
        query_type: The persona type

    Returns:
        True if valid, False otherwise
    """
    # All schemas now share the same structure
    required_keys = ["reasoning_content", "content"]

    # Check all required keys exist
    for key in required_keys:
        if key not in response:
            return False

    # Type checking
    return isinstance(response.get("reasoning_content"), str) and isinstance(
        response.get("content"), str
    )


def format_prompt_with_context(
    query_type: str, user_query: str, retrieved_context: str
) -> str:
    """
    Combines system prompt with user query and RAG context.

    Args:
        query_type: One of "wanjiku", "wakili", "mwanahabari"
        user_query: The user's original query
        retrieved_context: Context retrieved from RAG

    Returns:
        Complete prompt ready for LLM

    Example:
        >>> prompt = format_prompt_with_context(
        ...     "wanjiku",
        ...     "Kanjo wameongeza parking fees aje?",
        ...     "Nairobi County Assembly passed resolution..."
        ... )
    """
    system_prompt = get_system_prompt(query_type)

    full_prompt = f"""{system_prompt}

---

**RETRIEVED CONTEXT:**
{retrieved_context}

---

**USER QUERY:**
{user_query}

---

**YOUR RESPONSE (valid JSON only):**"""

    return full_prompt


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("KENYANIZER MODULE - SYSTEM PROMPTS")
    print("=" * 80)

    print("\\n📊 PROMPT STATISTICS")
    print("-" * 80)

    for persona in ["wanjiku", "wakili", "mwanahabari"]:
        prompt = get_system_prompt(persona)
        word_count = len(prompt.split())
        char_count = len(prompt)

        print(f"\\n{persona.upper()}:")
        print(f"  Words: {word_count}")
        print(f"  Characters: {char_count}")
        print(f"  Estimated tokens: ~{word_count // 0.75:.0f}")

    print("\\n" + "=" * 80)
    print("SAMPLE USAGE")
    print("=" * 80)

    # Example: Get prompt for wanjiku
    wanjiku_prompt = get_system_prompt("wanjiku")
    print(f"\\nWANJIKU PROMPT (first 300 chars):")
    print(wanjiku_prompt[:300] + "...")

    # Example: Format complete prompt
    print("\\n" + "=" * 80)
    print("COMPLETE PROMPT EXAMPLE")
    print("=" * 80)

    complete_prompt = format_prompt_with_context(
        query_type="wanjiku",
        user_query="Kanjo wameongeza parking fees aje?",
        retrieved_context="Nairobi County Assembly passed Resolution 42/2024 increasing CBD parking fees to KES 300...",
    )

    print(f"\\nComplete prompt length: {len(complete_prompt)} characters")
    print(f"Preview (first 400 chars):\\n{complete_prompt[:400]}...")

    # Example: Validate response
    print("\\n" + "=" * 80)
    print("RESPONSE VALIDATION EXAMPLE")
    print("=" * 80)

    mock_response = {
        "reasoning_content": "Checking the latest fee schedule...",
        "content": "The fee is **KES 300**.",
    }

    is_valid = validate_response(mock_response, "wanjiku")
    print(f"\\nMock response valid: {is_valid}")

    print("\\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
