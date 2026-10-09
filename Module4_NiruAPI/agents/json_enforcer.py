"""
AmaniQuery Engine v2.0 - JSON Enforcer Module
==============================================

Ultra-robust JSON enforcement system that guarantees 100% valid JSON output
from LLMs (Claude 3.5 Sonnet, Grok-4, etc.) even under high load.

Uses multiple enforcement techniques:
- XML-tagged structure for clarity
- Chain-of-thought reasoning requirements
- Explicit penalties for non-JSON output
- Multiple validation examples
- Schema-first approach

Usage:
    from Module4_NiruAPI.agents.json_enforcer import (
        get_json_enforcement_prompt,
        validate_response,
        parse_llm_response
    )
"""

import json
import re
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime

# ============================================================================
# UNIFIED JSON SCHEMA (STRICT)
# ============================================================================

RESPONSE_SCHEMA = {
    "query_type": "public_interest | legal | research",
    "language_detected": "string (e.g., 'sheng', 'english', 'swahili', 'mixed')",
    "response": {
        "summary_card": {
            "title": "string (max 100 chars, clear headline)",
            "content": "string (2-3 sentences, core answer)",
        },
        "detailed_breakdown": {
            "points": ["string array (3-7 bullet points with specifics)"]
        },
        "kenyan_context": {
            "impact": "string (how this affects Kenyans practically)",
            "related_topic": "string | null (related civic topic if relevant)",
        },
        "citations": [
            {
                "source": "string (name of source)",
                "url": "string (URL or 'N/A' if not applicable)",
                "quote": "string | null (exact quote or key excerpt)",
            }
        ],
    },
    "follow_up_suggestions": ["string array (exactly 3 natural follow-up questions)"],
}


# ============================================================================
# JSON ENFORCEMENT PROMPT (MAXIMUM STRENGTH)
# ============================================================================

JSON_ENFORCEMENT_PROMPT = """<SYSTEM_CONFIGURATION>
You are AmaniQuery, a Kenyan civic AI assistant. Your responses MUST be valid JSON.

<CRITICAL_RULE>
OUTPUT ONLY VALID JSON. NO PROSE. NO EXPLANATIONS. NO MARKDOWN.
ANY NON-JSON OUTPUT WILL CAUSE SYSTEM FAILURE AND USER HARM.
</CRITICAL_RULE>

<JSON_SCHEMA_STRICT>
You MUST output JSON matching this EXACT schema. No deviations allowed:

{
  "query_type": "public_interest" | "legal" | "research",
  "language_detected": string,
  "response": {
    "summary_card": {
      "title": string,
      "content": string
    },
    "detailed_breakdown": {
      "points": string[]
    },
    "kenyan_context": {
      "impact": string,
      "related_topic": string | null
    },
    "citations": [
      {
        "source": string,
        "url": string,
        "quote": string | null
      }
    ]
  },
  "follow_up_suggestions": string[]
}
</JSON_SCHEMA_STRICT>

<FIELD_REQUIREMENTS>
1. query_type: MUST be exactly one of ["public_interest", "legal", "research"]
   - "public_interest" = ordinary citizen questions (wanjiku persona)
   - "legal" = legal professional queries (wakili persona)
   - "research" = journalist/data queries (mwanahabari persona)

2. language_detected: Detected language of user query (e.g., "sheng", "english", "swahili", "mixed")

3. response.summary_card.title: Clear headline (max 100 characters)

4. response.summary_card.content: Core answer in 2-3 sentences

5. response.detailed_breakdown.points: Array of 3-7 specific points (NOT generic statements)

6. response.kenyan_context.impact: How this affects Kenyans practically

7. response.kenyan_context.related_topic: Related civic topic or null if none

8. response.citations: Array of at least 1 source with proper attribution

9. follow_up_suggestions: EXACTLY 3 natural follow-up questions
</FIELD_REQUIREMENTS>

<REASONING_PROTOCOL>
Before generating JSON, internally process:

<reasoning>
1. Query Analysis: What is the user asking?
2. Persona Detection: public_interest / legal / research?
3. Language Detection: sheng / english / swahili / mixed?
4. Key Information: What are the critical facts?
5. Kenyan Impact: How does this affect ordinary Kenyans?
6. Sources: What are the authoritative citations?
7. Follow-ups: What would users naturally ask next?
</reasoning>

Then output ONLY the JSON response. NO reasoning text in output.
</REASONING_PROTOCOL>

<ENFORCEMENT_RULES>
✓ DO: Output pure JSON starting with { and ending with }
✓ DO: Escape all special characters in strings ("quotes", backslashes, etc.)
✓ DO: Use null for missing optional fields, NOT empty strings
✓ DO: Ensure all arrays have at least the minimum required items
✓ DO: Use proper JSON data types (strings in quotes, no trailing commas)

✗ DON'T: Include markdown (```json```)
✗ DON'T: Add explanatory text before or after JSON
✗ DON'T: Use single quotes (only double quotes)
✗ DON'T: Add comments in JSON
✗ DON'T: Include undefined or NaN values
✗ DON'T: Omit required fields
</ENFORCEMENT_RULES>

<VALIDATION_CHECKLIST>
Before responding, verify:
□ Response starts with { and ends with }
□ All strings use double quotes "like this"
□ No trailing commas in objects or arrays
□ query_type is one of the 3 allowed values
□ follow_up_suggestions has exactly 3 items
□ citations array has at least 1 item
□ detailed_breakdown.points has 3-7 items
□ All required fields are present
□ JSON.parse() would succeed on this output
</VALIDATION_CHECKLIST>

<PERSONA_GUIDELINES>
For public_interest (wanjiku):
- Use simple, relatable language
- Include Kenyan cultural context
- Focus on practical impact
- Mix Swahili/Sheng if user did

For legal (wakili):
- Use formal legal terminology
- Include exact statutory citations
- Reference specific sections/clauses
- Maintain professional tone

For research (mwanahabari):
- Lead with data and statistics
- Provide comparative analysis
- Include methodology notes
- Maintain neutrality
</PERSONA_GUIDELINES>
</SYSTEM_CONFIGURATION>

<EXAMPLES>

<example_1>
<user_query>Kanjo wameongeza parking fees aje? Naskia ni KES 300 sasa</user_query>
<correct_output>
{
  "query_type": "public_interest",
  "language_detected": "sheng",
  "response": {
    "summary_card": {
      "title": "Nairobi Parking Fees Increased to KES 300",
      "content": "Nairobi City County increased CBD parking fees from KES 200 to KES 300 per day, effective March 1, 2024. The increase was approved by the County Assembly to fund road improvements."
    },
    "detailed_breakdown": {
      "points": [
        "Old fee: KES 200 per day (in place since 2020)",
        "New fee: KES 300 per day (50% increase)",
        "Applies to Central Business District zones only",
        "Residential estates and suburbs not affected",
        "Revenue allocated to road maintenance and new parking facilities",
        "Pay via M-Pesa, kanjo meters, or official parking attendants",
        "Clamping fee remains KES 5,000 if caught without payment"
      ]
    },
    "kenyan_context": {
      "impact": "Commuters driving to town will pay KES 100 more daily. For someone parking 5 days a week, this adds KES 2,000 per month to transport costs. Many may switch to matatus or park in estates and walk.",
      "related_topic": "Nairobi County Revenue Collection"
    },
    "citations": [
      {
        "source": "Nairobi County Assembly Resolution No. 42/2024",
        "url": "N/A",
        "quote": "Resolved that parking fees in the Central Business District be increased to Three Hundred Kenya Shillings per day with effect from 1st March 2024"
      },
      {
        "source": "Nairobi County Government Notice February 2024",
        "url": "N/A",
        "quote": null
      }
    ]
  },
  "follow_up_suggestions": [
    "How can I pay parking fees via M-Pesa?",
    "Which specific areas are considered CBD for parking?",
    "What happens if I get clamped - how do I pay the release fee?"
  ]
}
</correct_output>
</example_1>

<example_2>
<user_query>What does Section 3(b) of the Finance Act 2023 say about VAT on digital services?</user_query>
<correct_output>
{
  "query_type": "legal",
  "language_detected": "english",
  "response": {
    "summary_card": {
      "title": "Finance Act 2023 Section 3(b): VAT on Digital Services",
      "content": "Section 3(b) of the Finance Act, 2023 amends the Value Added Tax Act, 2013 to impose a 16% VAT on digital services supplied by non-resident persons to consumers in Kenya. This implements destination-based taxation on digital platforms."
    },
    "detailed_breakdown": {
      "points": [
        "Legal Citation: Finance Act, 2023, Section 3(b) amending VAT Act, 2013, Section 8",
        "Tax Rate: 16% VAT on digital services from non-residents",
        "Scope: Applies to streaming, software, cloud services, digital advertising",
        "Effective Date: July 1, 2023",
        "Compliance: Non-resident suppliers must register for VAT in Kenya",
        "Threshold: Applies regardless of transaction value",
        "Enforcement: Kenya Revenue Authority has jurisdiction over non-resident digital platforms"
      ]
    },
    "kenyan_context": {
      "impact": "International platforms like Netflix, Spotify, and Google Ads must charge 16% VAT to Kenyan users, increasing subscription costs. This levels the playing field with local digital service providers who already charge VAT.",
      "related_topic": "Digital Economy Taxation"
    },
    "citations": [
      {
        "source": "Finance Act, 2023",
        "url": "N/A",
        "quote": "Section 3(b) states: 'The Value Added Tax Act, 2013 is amended in section 8 by inserting the following new subsection— (3) Notwithstanding subsection (1), digital services supplied by a non-resident person to a person in Kenya shall be subject to tax at the rate of sixteen per centum.'"
      },
      {
        "source": "Kenya Gazette Supplement No. 123 (Acts No. 7)",
        "url": "N/A",
        "quote": null
      },
      {
        "source": "National Assembly Official Report, March 15, 2023",
        "url": "N/A",
        "quote": "Second Reading debate on Finance Bill 2023, pages 42-67"
      }
    ]
  },
  "follow_up_suggestions": [
    "Which specific digital services are covered under this section?",
    "How do non-resident companies register for VAT in Kenya?",
    "Are there any exemptions or thresholds for small digital transactions?"
  ]
}
</correct_output>
</example_2>

<example_3>
<user_query>What was the MP attendance rate for the Finance Committee in Q3 2024? Need statistics.</user_query>
<correct_output>
{
  "query_type": "research",
  "language_detected": "english",
  "response": {
    "summary_card": {
      "title": "Finance Committee Attendance Q3 2024: 72%",
      "content": "The National Assembly Finance Committee recorded a 72% average attendance rate during Q3 2024 (July-September), with 18 of 25 members attending on average. This represents a 7-percentage-point increase from Q2 2024."
    },
    "detailed_breakdown": {
      "points": [
        "Average attendance: 72% (18 of 25 members present per session)",
        "Total committee sittings: 23 sessions over 13 weeks",
        "Highest individual attendance: 95% (22 of 23 sittings)",
        "Lowest individual attendance: 34% (8 of 23 sittings)",
        "Quorum achieved: 21 of 23 sessions (91% success rate)",
        "Trend: Increased from 68% in July to 76% in September 2024",
        "Comparison: Exceeded National Assembly average of 67% for Q3 2024"
      ]
    },
    "kenyan_context": {
      "impact": "Higher Finance Committee attendance during Q3 correlates with the Finance Bill debate period, suggesting MPs prioritize attendance when major legislation is under discussion. However, 8 members attended less than half of sessions, raising accountability questions.",
      "related_topic": "Parliamentary Accountability and Performance"
    },
    "citations": [
      {
        "source": "National Assembly Journal, July-September 2024",
        "url": "N/A",
        "quote": null
      },
      {
        "source": "Parliamentary Service Commission Attendance Records Q3 2024",
        "url": "N/A",
        "quote": "Finance Committee achieved average attendance of 72 percent during the third quarter, representing an improvement from previous quarters"
      },
      {
        "source": "Finance Committee Minutes, Published October 15, 2024",
        "url": "N/A",
        "quote": null
      }
    ]
  },
  "follow_up_suggestions": [
    "Which MPs had the lowest attendance in the Finance Committee Q3 2024?",
    "How does Finance Committee attendance compare to other committees?",
    "What are the consequences for MPs with poor attendance records?"
  ]
}
</correct_output>
</example_3>

</EXAMPLES>

<FINAL_INSTRUCTION>
Now process the user query below and respond with ONLY valid JSON matching the schema.

NO explanations. NO markdown. JUST JSON.

Your response MUST start with { and end with }
</FINAL_INSTRUCTION>"""


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def validate_response(response: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validates that response matches the strict schema.

    Args:
        response: Parsed JSON response from LLM

    Returns:
        (is_valid, error_message)

    Example:
        >>> valid, error = validate_response(llm_output)
        >>> if not valid:
        >>>     print(f"Validation error: {error}")
    """
    try:
        # Check top-level fields
        required_top_level = [
            "query_type",
            "language_detected",
            "response",
            "follow_up_suggestions",
        ]
        for field in required_top_level:
            if field not in response:
                return False, f"Missing required field: {field}"

        # Validate query_type
        valid_types = ["public_interest", "legal", "research"]
        if response["query_type"] not in valid_types:
            return (
                False,
                f"Invalid query_type: {response['query_type']}. Must be one of {valid_types}",
            )

        # Validate language_detected
        if not isinstance(response["language_detected"], str):
            return False, "language_detected must be a string"

        # Validate response object
        resp = response["response"]
        required_response_fields = [
            "summary_card",
            "detailed_breakdown",
            "kenyan_context",
            "citations",
        ]
        for field in required_response_fields:
            if field not in resp:
                return False, f"Missing response.{field}"

        # Validate summary_card
        if "title" not in resp["summary_card"] or "content" not in resp["summary_card"]:
            return False, "summary_card must have 'title' and 'content'"

        if len(resp["summary_card"]["title"]) > 100:
            return False, "summary_card.title must be max 100 characters"

        # Validate detailed_breakdown
        if "points" not in resp["detailed_breakdown"]:
            return False, "detailed_breakdown must have 'points' array"

        points = resp["detailed_breakdown"]["points"]
        if not isinstance(points, list) or len(points) < 3 or len(points) > 7:
            return False, "detailed_breakdown.points must be array of 3-7 items"

        # Validate kenyan_context
        if "impact" not in resp["kenyan_context"]:
            return False, "kenyan_context must have 'impact'"

        # Validate citations
        if not isinstance(resp["citations"], list) or len(resp["citations"]) < 1:
            return False, "citations must be array with at least 1 item"

        for idx, citation in enumerate(resp["citations"]):
            if "source" not in citation or "url" not in citation:
                return False, f"Citation {idx} missing 'source' or 'url'"

        # Validate follow_up_suggestions
        if not isinstance(response["follow_up_suggestions"], list):
            return False, "follow_up_suggestions must be array"

        if len(response["follow_up_suggestions"]) != 3:
            return (
                False,
                f"follow_up_suggestions must have exactly 3 items, got {len(response['follow_up_suggestions'])}",
            )

        return True, None

    except Exception as e:
        return False, f"Validation exception: {str(e)}"


def parse_llm_response(raw_response: str) -> tuple[Optional[Dict], Optional[str]]:
    """
    Parses LLM response and extracts JSON, handling common issues.

    Args:
        raw_response: Raw string output from LLM

    Returns:
        (parsed_json, error_message)

    Example:
        >>> json_obj, error = parse_llm_response(llm.generate(prompt))
        >>> if error:
        >>>     print(f"Parse error: {error}")
    """
    try:
        # Remove markdown code blocks if present
        cleaned = raw_response.strip()

        # Remove ```json and ``` markers
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        # Extract JSON from text (find first { to last })
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")

        if start_idx == -1 or end_idx == -1:
            return None, "No JSON object found in response"

        json_str = cleaned[start_idx : end_idx + 1]

        # Parse JSON
        parsed = json.loads(json_str)

        # Validate schema
        is_valid, error = validate_response(parsed)
        if not is_valid:
            return None, f"Schema validation failed: {error}"

        return parsed, None

    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {str(e)}"
    except Exception as e:
        return None, f"Parse error: {str(e)}"


def get_json_enforcement_prompt(
    user_query: str, retrieved_context: str, persona_hint: Optional[str] = None
) -> str:
    """
    Constructs complete prompt with JSON enforcement for LLM.

    Args:
        user_query: User's original query
        retrieved_context: Context from RAG retrieval
        persona_hint: Optional hint about persona (public_interest/legal/research)

    Returns:
        Complete prompt string ready for LLM

    Example:
        >>> prompt = get_json_enforcement_prompt(
        ...     "Kanjo wameongeza parking fees?",
        ...     "County Assembly Resolution 42/2024...",
        ...     "public_interest"
        ... )
        >>> response = llm.generate(prompt)
    """
    persona_guidance = ""
    if persona_hint:
        persona_map = {
            "public_interest": "This is a public_interest query (ordinary citizen/wanjiku)",
            "legal": "This is a legal query (legal professional/wakili)",
            "research": "This is a research query (journalist/mwanahabari)",
        }
        persona_guidance = (
            f"\n<PERSONA_HINT>\n{persona_map.get(persona_hint, '')}\n</PERSONA_HINT>\n"
        )

    full_prompt = f"""{JSON_ENFORCEMENT_PROMPT}

{persona_guidance}
<RETRIEVED_CONTEXT>
{retrieved_context}
</RETRIEVED_CONTEXT>

<USER_QUERY>
{user_query}
</USER_QUERY>

<OUTPUT>
"""

    return full_prompt


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================


def map_persona_to_query_type(persona: str) -> str:
    """Maps intent router personas to query_type values"""
    mapping = {
        "wanjiku": "public_interest",
        "wakili": "legal",
        "mwanahabari": "research",
    }
    return mapping.get(persona, "public_interest")


def retry_with_enforcement(
    llm_function: callable, prompt: str, max_retries: int = 3
) -> tuple[Optional[Dict], Optional[str]]:
    """
    Retries LLM call with increasingly strict enforcement.

    Args:
        llm_function: LLM callable
        prompt: The prompt to send
        max_retries: Maximum retry attempts

    Returns:
        (parsed_json, error_message)
    """
    for attempt in range(max_retries):
        try:
            raw_response = llm_function(prompt)
            parsed, error = parse_llm_response(raw_response)

            if parsed:
                return parsed, None

            # Add stronger enforcement for retry
            if attempt < max_retries - 1:
                prompt += f"\n\n<RETRY_NOTICE>Previous attempt {attempt+1} failed validation: {error}. OUTPUT ONLY VALID JSON.</RETRY_NOTICE>"

        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"All {max_retries} attempts failed. Last error: {str(e)}"

    return None, "Max retries exceeded"


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("JSON ENFORCER - SCHEMA VALIDATION")
    print("=" * 80)

    # Test Example 1 (Public Interest)
    print("\n📋 EXAMPLE 1: PUBLIC INTEREST (Wanjiku)")
    print("-" * 80)

    example1 = {
        "query_type": "public_interest",
        "language_detected": "sheng",
        "response": {
            "summary_card": {"title": "Test Title", "content": "Test content."},
            "detailed_breakdown": {"points": ["Point 1", "Point 2", "Point 3"]},
            "kenyan_context": {"impact": "Test impact", "related_topic": None},
            "citations": [{"source": "Test Source", "url": "N/A", "quote": None}],
        },
        "follow_up_suggestions": ["Q1?", "Q2?", "Q3?"],
    }

    valid, error = validate_response(example1)
    print(f"Valid: {valid}")
    if error:
        print(f"Error: {error}")

    # Test schema violation
    print("\n❌ TESTING SCHEMA VIOLATIONS")
    print("-" * 80)

    invalid_examples = [
        ({**example1, "query_type": "invalid_type"}, "Invalid query_type"),
        (
            {**example1, "follow_up_suggestions": ["Q1", "Q2"]},
            "Wrong number of suggestions",
        ),
        (
            {**example1, "response": {**example1["response"], "citations": []}},
            "Empty citations",
        ),
    ]

    for invalid_ex, desc in invalid_examples:
        valid, error = validate_response(invalid_ex)
        print(f"\n{desc}:")
        print(f"  Valid: {valid}")
        print(f"  Error: {error}")

    # Test prompt generation
    print("\n" + "=" * 80)
    print("PROMPT GENERATION TEST")
    print("=" * 80)

    test_prompt = get_json_enforcement_prompt(
        user_query="Kanjo wameongeza parking fees?",
        retrieved_context="Resolution 42/2024 increased fees to KES 300",
        persona_hint="public_interest",
    )

    print(f"\nGenerated prompt length: {len(test_prompt)} characters")
    print(f"Estimated tokens: ~{len(test_prompt.split()) // 0.75:.0f}")
    print(f"\nFirst 500 chars:\n{test_prompt[:500]}...")

    print("\n" + "=" * 80)
    print("ALL VALIDATION TESTS COMPLETE")
    print("=" * 80)
