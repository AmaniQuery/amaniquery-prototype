"""
Article Quality Scorer
Scores articles based on content length, metadata completeness, source reputation, and freshness
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from loguru import logger


class QualityScorer:
    """
    Score articles for quality and relevance
    """

    # Source reputation scores (0-1 scale)
    SOURCE_REPUTATION = {
        # Major newspapers - high reputation
        "nation.africa": 0.95,
        "www.nation.co.ke": 0.95,
        "standardmedia.co.ke": 0.95,
        "www.standardmedia.co.ke": 0.95,
        "the-star.co.ke": 0.90,
        "www.the-star.co.ke": 0.90,
        "businessdailyafrica.com": 0.90,
        "www.businessdailyafrica.com": 0.90,
        "theeastafrican.co.ke": 0.90,
        "www.theeastafrican.co.ke": 0.90,
        # TV stations - high reputation
        "citizen.digital": 0.92,
        "www.citizen.digital": 0.92,
        "ktnnews.co.ke": 0.90,
        "www.ktnnews.co.ke": 0.90,
        "ntv.co.ke": 0.90,
        "www.ntv.co.ke": 0.90,
        # Radio stations - good reputation
        "capitalfm.co.ke": 0.85,
        "www.capitalfm.co.ke": 0.85,
        "radiocitizen.co.ke": 0.85,
        "www.radiocitizen.co.ke": 0.85,
        # Online portals - variable reputation
        "tuko.co.ke": 0.75,
        "www.tuko.co.ke": 0.75,
        "hivisasa.com": 0.70,
        "www.hivisasa.com": 0.70,
        "pulselive.co.ke": 0.75,
        "www.pulselive.co.ke": 0.75,
        "kenyans.co.ke": 0.70,
        "www.kenyans.co.ke": 0.70,
        # Government sources - high authority
        "president.go.ke": 0.95,
        "www.president.go.ke": 0.95,
        "information.go.ke": 0.90,
        "www.information.go.ke": 0.90,
        "kenyanews.go.ke": 0.90,
        "www.kenyanews.go.ke": 0.90,
    }

    # Minimum content length for quality articles
    MIN_CONTENT_LENGTH = 200
    IDEAL_CONTENT_LENGTH = 500
    MAX_CONTENT_LENGTH = 10000

    def score_article(self, article: Dict) -> Dict:
        """
        Score an article for quality

        Args:
            article: Article dictionary with url, content, title, author, publication_date, source_name

        Returns:
            Dictionary with:
            - total_score: Overall quality score (0-1)
            - content_score: Content quality (0-1)
            - metadata_score: Metadata completeness (0-1)
            - source_score: Source reputation (0-1)
            - freshness_score: Article freshness (0-1)
            - breakdown: Detailed breakdown of scores
        """
        url = article.get("url", "")
        content = article.get("content", "")
        title = article.get("title", "")
        author = article.get("author", "")
        publication_date = article.get("publication_date", "")
        source_name = article.get("source_name", "")

        # Score components
        content_score = self._score_content(content)
        metadata_score = self._score_metadata(title, author, publication_date, content)
        source_score = self._score_source(url, source_name)
        freshness_score = self._score_freshness(publication_date)

        # Weighted total score
        total_score = (
            content_score * 0.35  # Content is most important
            + metadata_score * 0.25  # Metadata completeness
            + source_score * 0.25  # Source reputation
            + freshness_score * 0.15  # Freshness
        )

        return {
            "total_score": round(total_score, 3),
            "content_score": round(content_score, 3),
            "metadata_score": round(metadata_score, 3),
            "source_score": round(source_score, 3),
            "freshness_score": round(freshness_score, 3),
            "breakdown": {
                "content_length": len(content),
                "has_title": bool(title),
                "has_author": bool(author),
                "has_date": bool(publication_date),
                "source_reputation": source_score,
            },
        }

    def _score_content(self, content: str) -> float:
        """Score content based on length and quality"""
        if not content:
            return 0.0

        content_length = len(content.strip())

        # Too short
        if content_length < self.MIN_CONTENT_LENGTH:
            return 0.2

        # Ideal length
        if self.MIN_CONTENT_LENGTH <= content_length <= self.IDEAL_CONTENT_LENGTH:
            # Linear scale from 0.5 to 1.0
            ratio = (content_length - self.MIN_CONTENT_LENGTH) / (
                self.IDEAL_CONTENT_LENGTH - self.MIN_CONTENT_LENGTH
            )
            return 0.5 + (ratio * 0.5)

        # Good length (above ideal but not too long)
        if self.IDEAL_CONTENT_LENGTH < content_length <= self.MAX_CONTENT_LENGTH:
            return 1.0

        # Too long (might be scraping errors or full pages)
        if content_length > self.MAX_CONTENT_LENGTH:
            return 0.8

    def _score_metadata(
        self, title: str, author: str, publication_date: str, content: str
    ) -> float:
        """Score metadata completeness"""
        score = 0.0

        # Title (required)
        if title and len(title.strip()) > 10:
            score += 0.3
        elif title:
            score += 0.15

        # Author (optional but preferred)
        if author and len(author.strip()) > 2:
            score += 0.25
        else:
            score += 0.1  # Partial credit if missing

        # Publication date (important)
        if publication_date:
            try:
                # Try to parse date
                date_parser.parse(publication_date)
                score += 0.3
            except:
                score += 0.15
        else:
            score += 0.1

        # Content quality indicators
        if content:
            # Check for common quality indicators
            has_paragraphs = "\n\n" in content or len(content.split(".")) > 5
            if has_paragraphs:
                score += 0.15
            else:
                score += 0.05

        return min(score, 1.0)

    def _score_source(self, url: str, source_name: str) -> float:
        """Score based on source reputation"""
        from urllib.parse import urlparse

        # Try to get domain from URL
        domain = None
        if url:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
            except:
                pass

        # Check domain reputation
        if domain and domain in self.SOURCE_REPUTATION:
            return self.SOURCE_REPUTATION[domain]

        # Check source name reputation
        if source_name:
            source_lower = source_name.lower()
            for known_domain, reputation in self.SOURCE_REPUTATION.items():
                if (
                    known_domain.replace("www.", "") in source_lower
                    or source_lower in known_domain
                ):
                    return reputation

        # Default reputation for unknown sources
        return 0.6

    def _score_freshness(self, publication_date: str) -> float:
        """Score based on article freshness"""
        if not publication_date:
            return 0.5  # Neutral score if no date

        try:
            pub_date = date_parser.parse(publication_date)
            now = datetime.utcnow()

            # Handle timezone-aware dates
            if pub_date.tzinfo:
                now = datetime.now(pub_date.tzinfo)

            age = now - pub_date

            # Very fresh (less than 1 hour)
            if age < timedelta(hours=1):
                return 1.0

            # Fresh (less than 24 hours)
            if age < timedelta(hours=24):
                return 0.9

            # Recent (less than 7 days)
            if age < timedelta(days=7):
                return 0.7

            # Recent (less than 30 days)
            if age < timedelta(days=30):
                return 0.5

            # Old (more than 30 days)
            if age < timedelta(days=90):
                return 0.3

            # Very old
            return 0.1

        except Exception as e:
            logger.debug(f"Error parsing publication date: {e}")
            return 0.5

    def is_high_quality(self, article: Dict, threshold: float = 0.7) -> bool:
        """
        Check if article meets quality threshold

        Args:
            article: Article dictionary
            threshold: Minimum score to be considered high quality (default 0.7)

        Returns:
            True if article meets quality threshold
        """
        score_result = self.score_article(article)
        return score_result["total_score"] >= threshold

    def filter_by_quality(self, articles: list, min_score: float = 0.6) -> list:
        """
        Filter articles by quality score

        Args:
            articles: List of article dictionaries
            min_score: Minimum quality score to keep

        Returns:
            List of articles that meet quality threshold, sorted by score
        """
        scored_articles = []
        for article in articles:
            score_result = self.score_article(article)
            if score_result["total_score"] >= min_score:
                article["quality_score"] = score_result
                scored_articles.append(article)

        # Sort by total score (descending)
        scored_articles.sort(
            key=lambda x: x["quality_score"]["total_score"], reverse=True
        )
        return scored_articles
