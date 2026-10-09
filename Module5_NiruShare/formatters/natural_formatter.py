"""
Natural formatter using LLM to create conversational posts
"""

import os
from typing import List, Dict, Optional
from .base_formatter import BaseFormatter


class NaturalFormatter(BaseFormatter):
    """Formatter that uses LLM to create natural, conversational posts"""

    def __init__(self, llm_provider: str = "openai", model: Optional[str] = None):
        """
        Initialize natural formatter

        Args:
            llm_provider: LLM provider (openai, anthropic, gemini, moonshot)
            model: Model name (uses defaults if not provided)
        """
        super().__init__(char_limit=None)
        self.llm_provider = llm_provider.lower()
        self.model = model or self._get_default_model()
        self.client = self._initialize_client()

    def _get_default_model(self) -> str:
        """Get default model for provider"""
        defaults = {
            "openai": "gpt-3.5-turbo",
            "anthropic": "claude-3-haiku-20240307",
            "gemini": "gemini-1.5-pro",
            "moonshot": "moonshot-v1-8k",
        }
        return defaults.get(self.llm_provider, "gpt-3.5-turbo")

    def _initialize_client(self):
        """Initialize LLM client"""
        if self.llm_provider == "openai":
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None
            return OpenAI(api_key=api_key)

        elif self.llm_provider == "anthropic":
            from anthropic import Anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            return Anthropic(api_key=api_key)

        elif self.llm_provider == "gemini":
            try:
                import google.generativeai as genai

                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    return None
                genai.configure(api_key=api_key)
                return genai.GenerativeModel(self.model)
            except ImportError:
                return None

        elif self.llm_provider == "moonshot":
            from openai import OpenAI

            api_key = os.getenv("MOONSHOT_API_KEY")
            if not api_key:
                return None
            return OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")

        return None

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
        style: Optional[str] = "casual",
        char_limit: Optional[int] = None,
    ) -> Dict:
        """
        Format response using LLM for natural language

        Args:
            answer: The RAG answer
            sources: List of source dictionaries
            query: Original query
            include_hashtags: Whether to include hashtags
            style: Writing style (professional, casual, engaging)
            char_limit: Optional character limit

        Returns:
            Dictionary with formatted post and metadata
        """
        self._validate_input(answer, sources)

        # If no LLM client available, fall back to simple formatting
        if not self.client:
            return self._fallback_format(
                answer, sources, query, include_hashtags, char_limit
            )

        # Generate natural post using LLM
        try:
            natural_post = self._generate_natural_post(
                answer=answer,
                sources=sources,
                query=query,
                style=style or "casual",
                char_limit=char_limit,
            )

            # Generate hashtags if requested
            hashtags = []
            if include_hashtags:
                hashtags = self._generate_hashtags(answer, sources)

            # Add hashtags to post if there's room
            if hashtags and char_limit:
                hashtag_text = " " + " ".join(hashtags[:3])
                if len(natural_post) + len(hashtag_text) <= char_limit:
                    natural_post += hashtag_text
            elif hashtags and not char_limit:
                natural_post += " " + " ".join(hashtags[:5])

            return {
                "platform": "natural",
                "content": natural_post.strip(),
                "character_count": len(natural_post.strip()),
                "hashtags": hashtags,
                "style": style,
                "llm_provider": self.llm_provider,
            }

        except Exception as e:
            # Fall back to simple formatting on error
            return self._fallback_format(
                answer, sources, query, include_hashtags, char_limit
            )

    def _generate_natural_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str],
        style: str,
        char_limit: Optional[int],
    ) -> str:
        """Generate natural post using LLM"""

        style_instructions = {
            "professional": "Write in a professional, authoritative tone suitable for LinkedIn or business contexts. Use clear, structured language.",
            "casual": "Write in a casual, conversational tone like you're talking to a friend. Be engaging and relatable.",
            "engaging": "Write in an engaging, enthusiastic tone that captures attention. Use questions, interesting facts, and compelling language.",
        }

        style_guide = style_instructions.get(style, style_instructions["casual"])

        # Build prompt
        prompt = f"""Rewrite the following information as a natural, engaging social media post.

Style: {style_guide}

Original query: {query if query else "General information"}

Information to share:
{answer}

Instructions:
- Write naturally, as if a human wrote it (not AI-generated)
- Keep it conversational and engaging
- Don't use excessive emojis or formatting symbols
- Make it flow naturally from sentence to sentence
- If there are sources, mention them naturally at the end
- Write in first or second person when appropriate
"""

        if char_limit:
            prompt += f"\n- Keep it under {char_limit} characters\n"

        prompt += "\nWrite the post now:"

        # Call LLM
        if self.llm_provider in ["openai", "moonshot"]:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500 if char_limit else 1000,
            )
            post = response.choices[0].message.content.strip()

        elif self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500 if char_limit else 1000,
                messages=[{"role": "user", "content": prompt}],
            )
            post = response.content[0].text.strip()

        elif self.llm_provider == "gemini":
            response = self.client.generate_content(prompt)
            post = response.text.strip()

        else:
            post = answer

        # Apply character limit if specified
        if char_limit and len(post) > char_limit:
            post = self._truncate_smart(post, char_limit)

        # Add sources naturally if available
        if sources:
            source_text = self._format_sources_naturally(sources)
            if char_limit:
                if len(post) + len(source_text) <= char_limit:
                    post += "\n\n" + source_text
            else:
                post += "\n\n" + source_text

        return post

    def _format_sources_naturally(self, sources: List[Dict]) -> str:
        """Format sources in a natural way"""
        if not sources:
            return ""

        source = sources[0]
        title = source.get("title", "")
        url = source.get("url", "")

        if url:
            return f"Learn more: {url}"
        elif title:
            return f"Source: {title}"
        else:
            return ""

    def _fallback_format(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str],
        include_hashtags: bool,
        char_limit: Optional[int],
    ) -> Dict:
        """Fallback formatting when LLM is not available"""
        # Create more natural, conversational post
        post_parts = []

        if query:
            # Make it conversational
            post_parts.append(f"Question: {query}")
            post_parts.append("")  # Empty line

        # Add the answer
        post_parts.append(answer)

        # Add source if available
        if sources and len(sources) > 0:
            source = sources[0]
            url = source.get("url", "")
            title = source.get("title", "")
            if url:
                post_parts.append("")
                post_parts.append(f"Source: {url}")

        post = "\n".join(post_parts)

        # Truncate if needed
        if char_limit and len(post) > char_limit:
            post = self._truncate_smart(post, char_limit)

        hashtags = []
        if include_hashtags:
            hashtags = self._generate_hashtags(answer, sources)
            if hashtags:
                # Add hashtags at the end
                hashtag_text = " " + " ".join(f"#{tag}" for tag in hashtags[:3])
                if char_limit:
                    if len(post) + len(hashtag_text) <= char_limit:
                        post += hashtag_text
                else:
                    post += hashtag_text

        return {
            "platform": "natural",
            "content": post.strip(),
            "character_count": len(post.strip()),
            "hashtags": hashtags,
            "style": "fallback",
            "llm_provider": None,
        }
