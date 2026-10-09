"""
Task-Aware Style Guidelines for AmaniQ Responses
Defines response styles per task cluster/group
"""

from typing import Dict

# =============================================================================
# STYLE GUIDELINES PER TASK GROUP
# =============================================================================

TASK_STYLE_GUIDELINES = {
    "Constitutional Law Research": {
        "tone": "formal_academic",
        "structure": "hierarchical_legal",
        "citation_format": "full_kenyan_legal",
        "guidelines": """
- Use formal legal language and terminology
- Cite Constitution articles in full: "Article 27(1) of the Constitution of Kenya 2010"
- Reference relevant case law and precedents
- Organize with clear hierarchical structure
- Include constitutional interpretation principles
- Cite *Constitution of Kenya 2010* with article numbers
        """,
    },
    "Employment Law Queries": {
        "tone": "practical_advisory",
        "structure": "step_by_step",
        "citation_format": "act_and_section",
        "guidelines": """
- Use clear, practical language accessible to non-lawyers
- Cite Employment Act sections: "Section 45 of the Employment Act, 2007"
- Provide actionable steps when applicable
- Include remedies and dispute resolution mechanisms
- Reference relevant Labour Relations Act provisions
- Explain procedures in chronological order
        """,
    },
    "Land Disputes & Property Law": {
        "tone": "procedural_clear",
        "structure": "process_oriented",
        "citation_format": "land_act_format",
        "guidelines": """
- Focus on procedural steps and requirements
- Cite Land Act and Land Registration Act provisions
- Explain registration and title processes clearly
- Include required documentation
- Reference relevant tribunal/court jurisdictions
- Use diagrams or step-by-step flows when helpful
        """,
    },
    "Bill Tracking & Parliamentary Process": {
        "tone": "factual_chronological",
        "structure": "timeline_based",
        "citation_format": "hansard_parliamentary",
        "guidelines": """
- Present events chronologically
- Cite Hansard debates with dates: "Hansard, 15th June 2024"
- Include vote counts and key dates
- Mention committee stages and readings
- Reference specific Bill sections
- Track amendments and their sponsors
        """,
    },
    "Case Law Lookup": {
        "tone": "analytical_judicial",
        "structure": "case_analysis",
        "citation_format": "case_citation_full",
        "guidelines": """
- Cite cases in full: *ABC Ltd v XYZ [2023] eKLR*
- Include court level and year
- Summarize key holdings and ratio decidendi
- Distinguish or apply precedents
- Reference judge names when significant
- Follow Kenyan case citation standards
        """,
    },
    "General Legal Research": {
        "tone": "balanced_professional",
        "structure": "comprehensive",
        "citation_format": "mixed_appropriate",
        "guidelines": """
- Use professional but accessible language
- Cite all sources appropriately
- Provide balanced analysis
- Include relevant statutes, cases, and commentary
- Organize logically with clear sections
- Adapt detail level to query complexity
        """,
    },
}


# Default style for unknown task groups
DEFAULT_STYLE = {
    "tone": "professional_clear",
    "structure": "logical",
    "citation_format": "standard",
    "guidelines": """
- Use clear, professional language
- Cite sources in standard Kenyan legal format
- Organize information logically
- Provide comprehensive but concise answers
- Include relevant citations and references
    """,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_style_for_task_group(task_group: str) -> Dict[str, str]:
    """
    Get style guidelines for a specific task group

    Args:
        task_group: Task group name from user profile

    Returns:
        Style dict with tone, structure, format, guidelines
    """
    return TASK_STYLE_GUIDELINES.get(task_group, DEFAULT_STYLE)


def format_style_instructions(user_profile: Dict) -> str:
    """
    Format style instructions based on user profile

    Args:
        user_profile: User profile with task_groups and expertise_level

    Returns:
        Formatted style instructions for prompt
    """
    task_groups = user_profile.get("task_groups", [])
    expertise_level = user_profile.get("expertise_level", "general")

    # Get primary task group (first one)
    primary_task = task_groups[0] if task_groups else None
    style = get_style_for_task_group(primary_task) if primary_task else DEFAULT_STYLE

    # Adjust for expertise level
    expertise_notes = {
        "lawyer": "Use technical legal terminology. Assume knowledge of legal principles and procedures.",
        "researcher": "Provide detailed analysis with academic rigor. Include nuances and multiple perspectives.",
        "journalist": "Focus on clarity and accessibility. Highlight public interest angles.",
        "layperson": "Use plain language. Explain legal concepts clearly without jargon.",
    }

    expertise_note = expertise_notes.get(expertise_level, expertise_notes["layperson"])

    return f"""
**Response Style for {primary_task or 'General Query'}:**

{style['guidelines']}

**Expertise Adaptation ({expertise_level}):**
{expertise_note}

**Citation Format:** {style['citation_format']}
**Tone:** {style['tone']}
"""


def get_citation_example(task_group: str) -> str:
    """Get citation example for task group"""
    examples = {
        "Constitutional Law Research": "Article 27(1) of the Constitution of Kenya 2010",
        "Employment Law Queries": "Section 45 of the Employment Act, 2007",
        "Case Law Lookup": "*ABC Ltd v XYZ [2023] eKLR*",
        "Bill Tracking & Parliamentary Process": "Finance Bill 2024, Second Reading, Hansard 15th June 2024",
        "Land Disputes & Property Law": "Section 12 of the Land Act, 2012",
    }
    return examples.get(task_group, "Standard legal citation format")
