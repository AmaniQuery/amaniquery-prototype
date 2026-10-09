"""
NiruSpider - Custom Middlewares
"""

from scrapy import signals
from scrapy.http import Response
from loguru import logger


class PoliteDelayMiddleware:
    """
    Rate-limiting middleware using Redis-based per-domain tracking.
    Does NOT use time.sleep() — relies on Scrapy's built-in DOWNLOAD_DELAY + AUTOTHROTTLE
    for actual request pacing. Redis tracking is for cross-pod coordination.
    """

    def __init__(self, rate_limiter=None):
        self.rate_limiter = rate_limiter

    @classmethod
    def from_crawler(cls, crawler):
        rate_limiter = None
        try:
            from ..rate_limiter import RateLimiter

            redis_url = crawler.settings.get("REDIS_URL")
            default_rate = crawler.settings.getfloat("DEFAULT_RATE_LIMIT", 2.0)
            rate_limiter = RateLimiter(redis_url=redis_url, default_rate=default_rate)
        except Exception as e:
            logger.debug(f"Rate limiter not available: {e}")

        middleware = cls(rate_limiter=rate_limiter)
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        return middleware

    def spider_opened(self, spider):
        if self.rate_limiter and self.rate_limiter.redis_client:
            spider.logger.info("PoliteDelayMiddleware: Redis rate tracking active")
        else:
            spider.logger.info(
                "PoliteDelayMiddleware: No Redis — relying on Scrapy AUTOTHROTTLE"
            )

    def process_request(self, request, spider):
        if self.rate_limiter and self.rate_limiter.redis_client:
            allowed = self.rate_limiter.allow_request(request.url)
            if not allowed:
                spider.logger.debug(f"Rate limit hit for {request.url}, deferring")
                return Response(url=request.url, status=429, request=request)
        return None
