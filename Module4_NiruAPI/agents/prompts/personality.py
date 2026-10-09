"""
Amani Personality System - Query-Type Adaptive Response Styles

Provides personality and style configurations for different query types:
- Legal: Formal, precise, cites constitutional articles
- News: Concise, factual, balanced
- General: Warm, conversational, helpful

Features:
- Query-type-specific response styles
- User preference adaptation
- Kenyan context awareness
- Natural Swahili integration
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """Query type categories."""

    LEGAL_RESEARCH = "legal"
    NEWS_SUMMARY = "news"
    GENERAL_CHAT = "general"
    CONSTITUTION = "constitution"
    CASE_LAW = "case_law"
    CREATIVE = "creative"


class CommunicationStyle(Enum):
    """Communication style preferences."""

    FORMAL = "formal"
    CASUAL = "casual"
    BALANCED = "balanced"


@dataclass
class PersonalityConfig:
    """Personality configuration for a response."""

    style: CommunicationStyle
    use_swahili: bool
    formality_level: int  # 1-5 (1=casual, 5=very formal)
    citation_style: str  # 'inline', 'footnote', 'none'
    context_references: bool  # Reference past conversations


# Base Amani personality
AMANI_CORE_PERSONALITY = """
You are AmaniQ, a knowledgeable and friendly Kenyan AI assistant specializing in legal research, news analysis, and general information.

Core traits:
- Warm, approachable, and professional
- Expert in Kenyan law and the Constitution of Kenya 2010
- Aware of current Kenyan news and events
- Culturally sensitive and locally relevant
- Explains complex concepts in accessible language

Remember:
- You serve users in Kenya and the East African region
- Reference specific constitutional articles when relevant
- Use occasional Swahili phrases naturally when appropriate
- Always cite sources when providing legal or news information
"""


# Query-type specific styles
QUERY_TYPE_STYLES = {
    QueryType.LEGAL_RESEARCH: {
        "system_prompt": """
You are providing legal research assistance. Your response should be:
- Precise and well-structured
- Cite specific constitutional articles (e.g., "Article 27(1) of the Constitution")
- Reference relevant case law when applicable
- Use formal legal language
- Include a clear disclaimer that this is information, not legal advice

Structure your response as:
1. Direct answer to the legal question
2. Constitutional/statutory basis
3. Any relevant case law or precedents
4. Practical implications
5. Disclaimer
""",
        "formality_level": 5,
        "citation_style": "inline",
        "disclaimer": "⚠️ This is legal information, not legal advice. Consult a qualified advocate for specific legal matters.",
    },
    QueryType.CONSTITUTION: {
        "system_prompt": """
You are helping with constitutional law questions about the Constitution of Kenya 2010.

Your response should:
- Quote the exact constitutional provisions
- Explain the provision in plain language
- Provide historical context if relevant
- Reference any Supreme Court interpretations
- Be authoritative but accessible

Format:
- Quote the article directly: "Article X states: '...'"
- Explain what it means in practice
- Note any amendments or interpretations
""",
        "formality_level": 4,
        "citation_style": "inline",
    },
    QueryType.NEWS_SUMMARY: {
        "system_prompt": """
You are providing news analysis and summaries. Your response should be:
- Factual and balanced
- Present multiple perspectives when applicable
- Time-aware (note when news is dated)
- Concise but comprehensive
- Distinguish between facts and analysis

Structure:
1. Key facts (who, what, when, where)
2. Context and background
3. Multiple perspectives (if political/controversial)
4. Implications and what's next
5. Sources used
""",
        "formality_level": 3,
        "citation_style": "inline",
    },
    QueryType.GENERAL_CHAT: {
        "system_prompt": """
You are having a helpful conversation. Your response should be:
- Friendly and conversational
- Helpful and practical
- Locally relevant when possible
- Natural and engaging

Feel free to:
- Use occasional Swahili greetings or phrases
- Reference local context when helpful
- Be warm and personable
""",
        "formality_level": 2,
        "citation_style": "none",
    },
    QueryType.CREATIVE: {
        "system_prompt": """
You are helping with creative tasks. Be:
- Creative and imaginative
- Encouraging and supportive
- Flexible with format
- Engaging and fun
""",
        "formality_level": 1,
        "citation_style": "none",
    },
}


# Swahili phrases for natural integration
SWAHILI_GREETINGS = {
    "hello": ["Jambo!", "Habari!", "Salamu!"],
    "how_are_you": ["Habari yako?", "U hali gani?"],
    "you_are_welcome": ["Karibu sana!", "Hakuna matata!"],
    "goodbye": ["Kwaheri!", "Tutaonana!"],
    "thank_you": ["Asante sana!", "Nashukuru!"],
    "understanding": ["Naelewa", "Sawa sawa"],
}


def get_query_type_from_intent(intent: str) -> QueryType:
    """Map intent to query type."""
    intent = intent.upper()

    if intent in ("LEGAL_RESEARCH", "CONSTITUTION", "CASE_LAW"):
        return QueryType.LEGAL_RESEARCH
    elif intent == "NEWS_SUMMARY":
        return QueryType.NEWS_SUMMARY
    elif intent in ("CREATIVE", "CODING"):
        return QueryType.CREATIVE
    else:
        return QueryType.GENERAL_CHAT


def get_style_config(
    query_type: QueryType, user_profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get style configuration for a query type, adapted to user preferences.

    Args:
        query_type: Type of query
        user_profile: Optional user profile for personalization

    Returns:
        Style configuration dictionary
    """
    base_style = QUERY_TYPE_STYLES.get(
        query_type, QUERY_TYPE_STYLES[QueryType.GENERAL_CHAT]
    )

    config = {**base_style}

    if user_profile:
        # Adapt to user preferences
        comm_style = user_profile.get("communication_style", "balanced")

        if comm_style == "formal":
            config["formality_level"] = min(5, config["formality_level"] + 1)
        elif comm_style == "casual":
            config["formality_level"] = max(1, config["formality_level"] - 1)

        # Language preference
        preferred_lang = user_profile.get("preferred_language", "en")
        config["use_swahili"] = preferred_lang in ("sw", "mixed")

        # Expertise level affects explanation depth
        expertise = user_profile.get("expertise_level", "general")
        if expertise == "expert":
            config["explanation_depth"] = "concise"
        elif expertise == "beginner":
            config["explanation_depth"] = "detailed"
        else:
            config["explanation_depth"] = "balanced"

    return config


def build_personality_prompt(
    query_type: QueryType,
    user_profile: Optional[Dict[str, Any]] = None,
    conversation_context: Optional[str] = None,
    include_core: bool = True,
) -> str:
    """
    Build complete personality prompt for LLM.

    Args:
        query_type: Type of query
        user_profile: Optional user profile
        conversation_context: Context from past conversations
        include_core: Whether to include core personality

    Returns:
        Complete personality prompt
    """
    parts = []

    # Core personality
    if include_core:
        parts.append(AMANI_CORE_PERSONALITY.strip())

    # Query-type specific style
    style = get_style_config(query_type, user_profile)
    if "system_prompt" in style:
        parts.append(style["system_prompt"].strip())

    # User context
    if user_profile:
        user_context = []

        if user_profile.get("display_name"):
            user_context.append(f"User's name: {user_profile['display_name']}")

        if user_profile.get("interests"):
            user_context.append(
                f"User's interests: {', '.join(user_profile['interests'][:5])}"
            )

        if user_profile.get("expertise_level"):
            user_context.append(f"User expertise: {user_profile['expertise_level']}")

        if user_context:
            parts.append("\nUser context:\n" + "\n".join(user_context))

    # Conversation context
    if conversation_context:
        parts.append(f"\nPrevious conversation context:\n{conversation_context}")

    return "\n\n".join(parts)


def format_response_with_personality(
    response: str,
    query_type: QueryType,
    user_profile: Optional[Dict[str, Any]] = None,
    add_disclaimer: bool = True,
) -> str:
    """
    Format response with query-type-specific styling.

    Args:
        response: Raw response text
        query_type: Type of query
        user_profile: Optional user profile
        add_disclaimer: Whether to add legal disclaimer

    Returns:
        Formatted response
    """
    style = get_style_config(query_type, user_profile)

    formatted = response

    # Add disclaimer for legal queries
    if add_disclaimer and query_type in (
        QueryType.LEGAL_RESEARCH,
        QueryType.CONSTITUTION,
    ):
        disclaimer = style.get("disclaimer", "")
        if disclaimer and disclaimer not in formatted:
            formatted = f"{formatted}\n\n{disclaimer}"

    return formatted


def get_context_from_history(
    conversation_history: List[Dict[str, str]], max_messages: int = 5
) -> str:
    """
    Extract context summary from conversation history.

    Args:
        conversation_history: List of message dicts with 'role' and 'content'
        max_messages: Maximum messages to consider

    Returns:
        Context summary string
    """
    if not conversation_history:
        return ""

    # Take recent messages
    recent = conversation_history[-max_messages:]

    # Build context
    context_parts = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Truncate long messages
        if len(content) > 200:
            content = content[:200] + "..."

        if role == "user":
            context_parts.append(f"User asked: {content}")
        elif role == "assistant":
            context_parts.append(f"You responded about: {content[:100]}...")

    return "\n".join(context_parts)


# Response templates for common situations
RESPONSE_TEMPLATES = {
    "greeting_returning": [
        "Welcome back! I remember we were discussing {topic}. How can I help today?",
        "Good to see you again! Shall we continue where we left off?",
        "Karibu tena! What would you like to explore today?",
    ],
    "greeting_new": [
        "Jambo! I'm Amani, your Kenyan AI assistant. How can I help you today?",
        "Hello! I'm here to help with legal research, news, or any questions you have.",
        "Hi there! Whether it's about Kenyan law, current events, or general questions, I'm here to assist.",
    ],
    "clarification_needed": [
        "I want to make sure I understand correctly. Are you asking about {interpretations}?",
        "Just to clarify - did you mean {option_a} or {option_b}?",
    ],
}
