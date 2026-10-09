"""
Test suite and demonstration for AmaniQuery Intent Router

This file shows example outputs and validates the intent classification system.
Run this to see how different queries are classified.
"""

import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from intent_router import (
    classify_query,
    FEW_SHOT_EXAMPLES,
    validate_classification_output,
    QueryType,
    DetectedLanguage,
)


def demo_classification_outputs():
    """
    Demonstrates what the classifier output looks like for various queries.
    Since we don't have an actual LLM here, we'll show the expected outputs.
    """
    print("=" * 80)
    print("AMANIQUERY INTENT ROUTER - CLASSIFICATION DEMONSTRATIONS")
    print("=" * 80)
    print()

    # Group examples by persona
    wanjiku_examples = [
        ex
        for ex in FEW_SHOT_EXAMPLES
        if ex["classification"]["query_type"] == "wanjiku"
    ]
    wakili_examples = [
        ex for ex in FEW_SHOT_EXAMPLES if ex["classification"]["query_type"] == "wakili"
    ]
    mwanahabari_examples = [
        ex
        for ex in FEW_SHOT_EXAMPLES
        if ex["classification"]["query_type"] == "mwanahabari"
    ]

    # WANJIKU (Ordinary Citizens)
    print("🧑‍🌾 WANJIKU EXAMPLES (Ordinary Kenyan Citizens)")
    print("-" * 80)
    for i, ex in enumerate(wanjiku_examples, 1):
        print(f"\n{i}. Query: \"{ex['query']}\"")
        print(f"   Classification:")
        print(f"   └─ Type: {ex['classification']['query_type']}")
        print(f"   └─ Confidence: {ex['classification']['confidence']}")
        print(f"   └─ Language: {ex['classification']['detected_language']}")
        print(f"   └─ Reasoning: {ex['classification']['reasoning']}")

    print("\n" + "=" * 80)

    # WAKILI (Legal Professionals)
    print("\n⚖️  WAKILI EXAMPLES (Legal Professionals)")
    print("-" * 80)
    for i, ex in enumerate(wakili_examples, 1):
        print(f"\n{i}. Query: \"{ex['query']}\"")
        print(f"   Classification:")
        print(f"   └─ Type: {ex['classification']['query_type']}")
        print(f"   └─ Confidence: {ex['classification']['confidence']}")
        print(f"   └─ Language: {ex['classification']['detected_language']}")
        print(f"   └─ Reasoning: {ex['classification']['reasoning']}")

    print("\n" + "=" * 80)

    # MWANAHABARI (Journalists/Researchers)
    print("\n📰 MWANAHABARI EXAMPLES (Journalists & Researchers)")
    print("-" * 80)
    for i, ex in enumerate(mwanahabari_examples, 1):
        print(f"\n{i}. Query: \"{ex['query']}\"")
        print(f"   Classification:")
        print(f"   └─ Type: {ex['classification']['query_type']}")
        print(f"   └─ Confidence: {ex['classification']['confidence']}")
        print(f"   └─ Language: {ex['classification']['detected_language']}")
        print(f"   └─ Reasoning: {ex['classification']['reasoning']}")

    print("\n" + "=" * 80)
    print()


def demo_edge_cases():
    """Highlights specific edge cases and how they're handled"""
    print("\n🔍 EDGE CASE ANALYSIS")
    print("=" * 80)

    edge_cases = [
        {
            "scenario": "Law Student Being Casual",
            "query": "hey quick Q - wat does section 23A say about public procurement? need it for a case",
            "expected": "wakili",
            "notes": "Informal language but legal context (section + case) → wakili",
        },
        {
            "scenario": "Journalist Using Sheng",
            "query": "Nipe data ya voting patterns kwa county MPs on healthcare bills from 2020-2024",
            "expected": "mwanahabari",
            "notes": "Sheng/mixed but asking for data trends → mwanahabari",
        },
        {
            "scenario": "Citizen Mentioning Legal Section",
            "query": "Alafu hiyo Section 12 inasema aje exactly? Sisi wananchi tunakaa confused bana",
            "expected": "wanjiku",
            "notes": "Mentions Section but heavy Sheng + self-ID as 'wananchi' → wanjiku",
        },
        {
            "scenario": "Mixed Language Researcher",
            "query": "Timeline ya budget allocation kwa health sector toka 2018 - nataka trends na figures",
            "expected": "mwanahabari",
            "notes": "Code-switching but requesting timeline/trends/data → mwanahabari",
        },
    ]

    for i, case in enumerate(edge_cases, 1):
        print(f"\n{i}. {case['scenario']}")
        print(f"   Query: \"{case['query']}\"")
        print(f"   Expected Type: {case['expected']}")
        print(f"   Analysis: {case['notes']}")

    print("\n" + "=" * 80)
    print()


def demo_language_detection():
    """Shows how different languages are detected"""
    print("\n🌍 LANGUAGE DETECTION PATTERNS")
    print("=" * 80)

    language_examples = {
        "sheng": [
            "Naskia kuna tax mpya kwa bodaboda, ni ukweli ama uongo?",
            "Alafu hiyo Section 12 inasema aje exactly? Sisi wananchi tunakaa confused bana",
        ],
        "mixed": [
            "Hii sheria ya Finance Act inasema nini kuhusu mama mboga?",
            "Nipe data ya voting patterns kwa county MPs on healthcare bills",
        ],
        "sw": ["Je, kifungu cha 47 katika Katiba kinasema nini kuhusu ugatuzi?"],
        "en": [
            "What did Raila say about the housing levy last week?",
            "Can you provide the full text of Section 3(b) of the Finance Act 2023?",
        ],
    }

    for lang, examples in language_examples.items():
        print(f"\n{lang.upper()}:")
        print("-" * 40)
        for ex in examples:
            print(f"  • {ex}")

    print("\n" + "=" * 80)
    print()


def demo_confidence_scoring():
    """Explains confidence score ranges"""
    print("\n📊 CONFIDENCE SCORE GUIDE")
    print("=" * 80)

    confidence_ranges = [
        {
            "range": "0.9 - 1.0",
            "level": "Very High",
            "description": "Clear-cut classification with strong indicators",
            "examples": [
                "Query: 'Section 3(b) of Finance Act' → wakili (0.98)",
                "Query: 'MP attendance rate Q3 2024' → mwanahabari (0.97)",
            ],
        },
        {
            "range": "0.8 - 0.89",
            "level": "High",
            "description": "Strong classification but with minor ambiguity",
            "examples": [
                "Query: 'Why is the government taxing us so much bana???' → wanjiku (0.88)",
                "Query: 'hey quick Q - wat does section 23A say?' → wakili (0.85)",
            ],
        },
        {
            "range": "0.6 - 0.79",
            "level": "Medium",
            "description": "Likely correct but notable edge case characteristics",
            "examples": [
                "Query: 'Alafu hiyo Section 12 inasema aje exactly?' → wanjiku (0.78)"
            ],
        },
        {
            "range": "< 0.6",
            "level": "Low",
            "description": "Ambiguous - may need clarification or fallback",
            "examples": [
                "Very short queries like 'Section 5?'",
                "Extremely mixed context queries",
            ],
        },
    ]

    for conf in confidence_ranges:
        print(f"\n{conf['range']} - {conf['level']}")
        print(f"  {conf['description']}")
        print(f"  Examples:")
        for ex in conf["examples"]:
            print(f"    • {ex}")

    print("\n" + "=" * 80)
    print()


def demo_prompt_preview():
    """Shows what the actual prompt looks like"""
    print("\n📝 SAMPLE CLASSIFICATION PROMPT (Zero-Shot)")
    print("=" * 80)

    # Get a sample prompt
    sample_query = "Naskia kuna tax mpya kwa motorbikes, ni ukweli?"
    result = classify_query(sample_query)

    if "prompt" in result:
        # Show first 1500 characters
        prompt_preview = result["prompt"][:1500]
        print(prompt_preview)
        print("\n[... truncated for brevity ...]")
        print(f"\nFull prompt length: {len(result['prompt'])} characters")

    print("\n" + "=" * 80)
    print()


def demo_json_output_format():
    """Shows the expected JSON output format"""
    print("\n💾 JSON OUTPUT FORMAT")
    print("=" * 80)

    sample_outputs = [
        {
            "query_type": "wanjiku",
            "confidence": 0.92,
            "detected_language": "sheng",
            "reasoning": "Informal Sheng question asking if motorcycle tax rumors are true",
        },
        {
            "query_type": "wakili",
            "confidence": 0.98,
            "detected_language": "en",
            "reasoning": "Precise request for specific legal clause and amendments, formal legal terminology",
        },
        {
            "query_type": "mwanahabari",
            "confidence": 0.97,
            "detected_language": "en",
            "reasoning": "Specific request for statistical data on MP attendance over a time period",
        },
    ]

    for i, output in enumerate(sample_outputs, 1):
        print(f"\nExample {i}:")
        print(json.dumps(output, indent=2))

    print("\n" + "=" * 80)
    print()


def demo_routing_logic():
    """Shows how to use classifications for routing"""
    print("\n🔀 ROUTING LOGIC EXAMPLE")
    print("=" * 80)

    routing_code = """
def route_query(user_query: str, llm_function):
    # Classify the query
    result = classify_query(user_query, llm_function)
    
    # Route based on persona
    if result["query_type"] == "wanjiku":
        # Ordinary citizen - use simple language
        return {
            "handler": "citizen_query_handler",
            "config": {
                "simplify_language": True,
                "focus_on_impact": True,
                "language": result["detected_language"],
                "include_sheng_glossary": result["detected_language"] == "sheng"
            }
        }
    
    elif result["query_type"] == "wakili":
        # Legal professional - precise citations
        return {
            "handler": "legal_query_handler",
            "config": {
                "include_citations": True,
                "verbatim_text": True,
                "case_law_references": True,
                "format": "formal"
            }
        }
    
    elif result["query_type"] == "mwanahabari":
        # Journalist - data and trends
        return {
            "handler": "data_query_handler",
            "config": {
                "include_statistics": True,
                "visualizations": True,
                "export_csv": True,
                "comparative_analysis": True
            }
        }
"""

    print(routing_code)
    print("\n" + "=" * 80)
    print()


def run_all_demos():
    """Run all demonstration functions"""
    demo_classification_outputs()
    demo_edge_cases()
    demo_language_detection()
    demo_confidence_scoring()
    demo_prompt_preview()
    demo_json_output_format()
    demo_routing_logic()

    print("\n✅ ALL DEMONSTRATIONS COMPLETE")
    print("=" * 80)
    print("\nNext Steps:")
    print(
        "1. Integrate with your LLM (Gemini/Groq) using examples in INTENT_ROUTER_GUIDE.md"
    )
    print("2. Test with real queries from your users")
    print("3. Monitor confidence scores and adjust thresholds")
    print("4. Add logging to track classification accuracy")
    print("=" * 80)


if __name__ == "__main__":
    run_all_demos()
