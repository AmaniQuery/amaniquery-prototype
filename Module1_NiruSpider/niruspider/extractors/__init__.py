"""
Article Extractors for News Aggregation
"""

from .article_extractor import ArticleExtractor
from .site_specific import SiteSpecificExtractor

__all__ = ["ArticleExtractor", "SiteSpecificExtractor"]
