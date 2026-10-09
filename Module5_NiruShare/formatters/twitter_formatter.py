"""
Twitter/X formatter
"""

from typing import List, Dict, Optional
from .base_formatter import BaseFormatter


class TwitterFormatter(BaseFormatter):
    """Format responses for X (Twitter)"""

    # Twitter character limits
    SINGLE_TWEET_LIMIT = 280
    TWEET_WITH_LINK_LIMIT = 257  # Reserve space for URL

    def __init__(self):
        super().__init__(char_limit=self.SINGLE_TWEET_LIMIT)

    def format_post(
        self,
        answer: str,
        sources: List[Dict],
        query: Optional[str] = None,
        include_hashtags: bool = True,
    ) -> Dict:
        """
        Format response for Twitter/X

        Returns thread if answer is too long for single tweet
        """
        # Validate input
        self._validate_input(answer, sources)

        # Generate hashtags
        hashtags = self._generate_hashtags(answer, sources) if include_hashtags else []
        hashtag_text = " " + " ".join(hashtags[:3]) if hashtags else ""

        # Try single tweet first
        single_tweet = self._format_single_tweet(answer, sources, hashtag_text, query)

        if len(single_tweet) <= self.SINGLE_TWEET_LIMIT:
            return {
                "platform": "twitter",
                "format": "single",
                "content": single_tweet,
                "character_count": len(single_tweet),
                "hashtags": hashtags[:3],
            }

        # Create thread
        thread = self._format_thread(answer, sources, hashtags, query)

        return {
            "platform": "twitter",
            "format": "thread",
            "content": thread,
            "tweet_count": len(thread),
            "hashtags": hashtags,
        }

    def _format_single_tweet(
        self,
        answer: str,
        sources: List[Dict],
        hashtag_text: str,
        query: Optional[str] = None,
    ) -> str:
        """Format as single tweet"""
        # Reserve space for source link (more natural)
        source_link = ""
        if sources and isinstance(sources[0], dict):
            url = sources[0].get("url", "")
            if url:
                # More natural link format
                source_link = f"\n\n{url}"

        # Calculate available space more accurately
        base_space = self.SINGLE_TWEET_LIMIT
        base_space -= len(hashtag_text)
        base_space -= len(source_link)

        # Add query if provided
        prefix = ""
        if query:
            query_text = str(query).strip()
            prefix = f"{query_text}\n\n"
            base_space -= len(prefix)

        # Ensure we have positive space
        available_space = max(50, base_space)  # Minimum 50 chars for content

        # Truncate answer
        truncated_answer = self._truncate_smart(answer, available_space)

        # Build tweet efficiently
        parts = []
        if prefix:
            parts.append(prefix)
        parts.append(truncated_answer)
        if source_link:
            parts.append(source_link)
        if hashtag_text:
            parts.append(hashtag_text)

        return "".join(parts).strip()

    def _format_thread(
        self,
        answer: str,
        sources: List[Dict],
        hashtags: List[str],
        query: Optional[str] = None,
    ) -> List[str]:
        """Format as Twitter thread with better handling for long content"""
        tweets = []

        # First tweet: Query + hook
        if query:
            query_text = str(query).strip()
            # If query is short, use it as hook
            if len(query_text) < 100:
                intro = f"🧵 {query_text}\n\nHere's what I found:"
            else:
                intro = "🧵 Here's a thread on this:"
        else:
            intro = "🧵 Here's what I found:"

        tweets.append(intro)

        # Split answer into sentences first to avoid breaking mid-sentence
        # This is a simple split, for production might want nltk or spacy but keeping dependencies low
        sentences = (
            answer.replace("!", "!<STOP>")
            .replace("?", "?<STOP>")
            .replace(".", ".<STOP>")
            .split("<STOP>")
        )
        sentences = [s.strip() for s in sentences if s.strip()]

        current_tweet = ""
        # Reserve space for numbering like "1/xx " (approx 5-6 chars)
        NUMBERING_RESERVE = 8
        limit_with_numbering = self.SINGLE_TWEET_LIMIT - NUMBERING_RESERVE

        # Buffer tweets content first, then number them later
        tweet_contents = []

        for sentence in sentences:
            # If sentence itself is longer than limit, we must force split it
            # But usually sentences are shorter.
            if len(sentence) > limit_with_numbering:
                # If specific sentence is huge, chunk it by words
                words = sentence.split()
                for word in words:
                    if len(current_tweet) + len(word) + 1 < limit_with_numbering:
                        current_tweet += word + " "
                    else:
                        if current_tweet:
                            tweet_contents.append(current_tweet.strip())
                        current_tweet = word + " "
            else:
                # Normal sentence handling
                if len(current_tweet) + len(sentence) + 1 < limit_with_numbering:
                    current_tweet += sentence + " "
                else:
                    if current_tweet:
                        tweet_contents.append(current_tweet.strip())
                    current_tweet = sentence + " "

        if current_tweet:
            tweet_contents.append(current_tweet.strip())

        # Add buffered contents to tweets with numbering
        # Note: tweets[0] is the intro.
        # The content tweets start from index 1.

        total_content_tweets = len(tweet_contents)
        for i, content in enumerate(tweet_contents, 1):
            tweets.append(f"{i}/{total_content_tweets} {content}")

        # Last tweet: Sources and hashtags
        sources_parts = []
        if sources:
            sources_parts.append("Sources:")
            for i, source in enumerate(sources[:2], 1):
                if isinstance(source, dict):
                    title = str(source.get("title", "Untitled")).strip()
                    url = str(source.get("url", "")).strip()
                    if len(title) > 40:
                        title = title[:37] + "..."
                    if url:
                        sources_parts.append(f"• {title} - {url}")
                    else:
                        sources_parts.append(f"• {title}")

        sources_text = "\n".join(sources_parts)
        hashtag_text = " ".join(hashtags[:5]) if hashtags else ""

        final_footer = ""
        if sources_text:
            final_footer += f"\n\n{sources_text}"
        if hashtag_text:
            final_footer += f"\n\n{hashtag_text}"

        final_footer = final_footer.strip()

        if final_footer:
            # Check if footer fits in the very last tweet or needs a new one
            if len(tweets[-1]) + len(final_footer) + 2 < self.SINGLE_TWEET_LIMIT:
                tweets[-1] += f"\n\n{final_footer}"
            else:
                tweets.append(final_footer)

        return tweets

    def format_quote_tweet(self, answer: str, original_tweet_url: str) -> str:
        """Format as quote tweet"""
        available = self.TWEET_WITH_LINK_LIMIT - len(original_tweet_url) - 10

        truncated = self._truncate_smart(answer, available)
        return f"{truncated}\n\n💬 {original_tweet_url}"
