"""
Twitter Scraper Tool using twikit
Supports both authenticated and unauthenticated modes with robust error handling
"""

import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

try:
    from twikit import Client
    from twikit.errors import TwitterException
except ImportError:
    Client = None
    TwitterException = None
    logger.warning("twikit not installed. Install with: pip install twikit")


class TwitterScraperTool:
    """
    Twitter scraping tool using twikit

    Supports optional authentication, rate limiting, retry logic,
    data validation, and comprehensive error handling.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        email: Optional[str] = None,
        enable_auth: bool = False,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """
        Initialize Twitter scraper

        Args:
            username: Twitter username for authentication (optional)
            password: Twitter password for authentication (optional)
            email: Twitter email for authentication (optional)
            enable_auth: Whether to enable authentication
            rate_limit_delay: Delay between requests to respect rate limits (seconds)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        if Client is None:
            raise ImportError(
                "twikit package required. Install with: pip install twikit"
            )

        self.client = Client()
        self.enable_auth = enable_auth
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_request_time = 0
        self.request_count = 0
        self.request_window_start = datetime.utcnow()

        # Authentication (optional - twikit can work without it for basic scraping)
        self.authenticated = False
        if enable_auth:
            self._authenticate(username, password, email)

        logger.info(f"Twitter scraper initialized (auth: {self.authenticated})")

    def _authenticate(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        email: Optional[str] = None,
    ):
        """
        Authenticate with Twitter (optional but recommended for production)

        Args:
            username: Twitter username
            password: Twitter password
            email: Twitter email
        """
        try:
            # Try to get credentials from environment if not provided
            username = username or os.getenv("TWITTER_USERNAME")
            password = password or os.getenv("TWITTER_PASSWORD")
            email = email or os.getenv("TWITTER_EMAIL")

            if not username or not password:
                logger.warning(
                    "Twitter credentials not provided. Running in unauthenticated mode."
                )
                self.authenticated = False
                return

            # Authenticate with twikit
            # Note: twikit authentication may vary by version
            try:
                self.client.login(
                    username=username,
                    password=password,
                    email=email if email else username,
                )
                self.authenticated = True
                logger.info("Twitter authentication successful")
            except Exception as auth_error:
                logger.warning(
                    f"Twitter authentication failed: {auth_error}. Continuing in unauthenticated mode."
                )
                self.authenticated = False
        except Exception as e:
            logger.error(f"Error during Twitter authentication: {e}")
            self.authenticated = False

    def _rate_limit_check(self):
        """Implement rate limiting to respect Twitter's limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1

    def _extract_tweet_data(self, tweet: Any) -> Optional[Dict[str, Any]]:
        """
        Extract and validate tweet data safely

        Args:
            tweet: Tweet object from twikit

        Returns:
            Extracted tweet data or None if invalid
        """
        try:
            # Safely extract tweet attributes with fallbacks
            tweet_id = getattr(tweet, "id", None) or getattr(tweet, "id_str", None)
            if not tweet_id:
                logger.warning("Tweet missing ID, skipping")
                return None

            # Extract text content
            text = (
                getattr(tweet, "full_text", None)
                or getattr(tweet, "text", None)
                or getattr(tweet, "content", None)
                or str(tweet)
            )

            if not text or len(text.strip()) == 0:
                logger.warning(f"Tweet {tweet_id} has no text content, skipping")
                return None

            # Extract user information
            user = getattr(tweet, "user", None)
            if user:
                author = (
                    getattr(user, "screen_name", None)
                    or getattr(user, "username", None)
                    or getattr(user, "name", None)
                    or "unknown"
                )
                author_id = (
                    getattr(user, "id", None) or getattr(user, "id_str", None) or None
                )
            else:
                author = "unknown"
                author_id = None

            # Extract timestamp
            created_at = None
            if hasattr(tweet, "created_at"):
                created_at_attr = getattr(tweet, "created_at")
                if created_at_attr:
                    if isinstance(created_at_attr, str):
                        created_at = created_at_attr
                    elif hasattr(created_at_attr, "isoformat"):
                        created_at = created_at_attr.isoformat()
                    else:
                        created_at = str(created_at_attr)

            # Build tweet URL
            if author and author != "unknown" and tweet_id:
                url = f"https://twitter.com/{author}/status/{tweet_id}"
            else:
                url = f"https://twitter.com/i/web/status/{tweet_id}" if tweet_id else ""

            # Extract engagement metrics
            retweet_count = getattr(tweet, "retweet_count", 0) or 0
            favorite_count = (
                getattr(tweet, "favorite_count", 0)
                or getattr(tweet, "like_count", 0)
                or 0
            )
            reply_count = getattr(tweet, "reply_count", 0) or 0

            return {
                "id": str(tweet_id),
                "text": text.strip(),
                "author": author,
                "author_id": str(author_id) if author_id else None,
                "created_at": created_at or datetime.utcnow().isoformat(),
                "url": url,
                "retweet_count": int(retweet_count),
                "favorite_count": int(favorite_count),
                "reply_count": int(reply_count),
                "is_retweet": getattr(tweet, "retweeted", False) or False,
                "is_reply": getattr(tweet, "in_reply_to_status_id", None) is not None,
            }
        except Exception as e:
            logger.error(f"Error extracting tweet data: {e}")
            return None

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Retry function with exponential backoff

        Args:
            func: Function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                self._rate_limit_check()
                return func(*args, **kwargs)
            except TwitterException as e:
                last_exception = e
                error_code = getattr(e, "code", None) or getattr(e, "status_code", None)

                # Don't retry on authentication errors
                if error_code in [401, 403]:
                    logger.error(f"Authentication error (code {error_code}): {e}")
                    raise

                # Don't retry on rate limit errors immediately
                if error_code == 429:
                    wait_time = (2**attempt) * 60  # Exponential backoff in minutes
                    logger.warning(
                        f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{self.max_retries}"
                    )
                    time.sleep(min(wait_time, 300))  # Cap at 5 minutes
                    continue

                # Retry on other errors
                if attempt < self.max_retries - 1:
                    wait_time = (2**attempt) * self.rate_limit_delay
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{self.max_retries}: {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached. Last error: {e}")
                    raise
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = (2**attempt) * self.rate_limit_delay
                    logger.warning(
                        f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached. Last error: {e}")
                    raise

        if last_exception:
            raise last_exception

    def execute(
        self,
        query: str,
        max_results: int = 20,
        search_type: str = "tweet",
        lang: Optional[str] = None,
        result_type: str = "recent",
    ) -> Dict[str, Any]:
        """
        Search Twitter for tweets

        Args:
            query: Search query
            max_results: Maximum number of results (max 100)
            search_type: Type of search (tweet, user, etc.)
            lang: Language code (e.g., 'en', 'sw' for Swahili)
            result_type: Result type (deprecated - no longer supported by twikit)

        Returns:
            Twitter search results with sources and metadata
        """
        if not query or not query.strip():
            return {
                "query": query,
                "results": [],
                "sources": [],
                "error": "Empty query provided",
                "count": 0,
            }

        # Validate and limit max_results
        max_results = min(max(1, max_results), 100)  # Clamp between 1 and 100

        try:
            # Execute search with retry logic
            def _search():
                if search_type == "tweet":
                    # Build search parameters
                    search_params = {"count": max_results}

                    # Add optional parameters
                    if lang:
                        search_params["lang"] = lang

                    # Note: result_type parameter removed as it's not supported in current twikit version

                    # Perform search
                    tweets = self.client.search_tweet(query, "latest", **search_params)
                    return tweets
                else:
                    raise ValueError(f"Unsupported search type: {search_type}")

            tweets = self._retry_with_backoff(_search)

            # Process and validate results
            formatted_results = []
            sources = []
            processed_count = 0

            # Handle both iterable and single tweet responses
            if not hasattr(tweets, "__iter__"):
                tweets = [tweets] if tweets else []

            for tweet in tweets:
                if processed_count >= max_results:
                    break

                tweet_data = self._extract_tweet_data(tweet)
                if tweet_data:
                    formatted_results.append(tweet_data)

                    # Create source entry
                    sources.append(
                        {
                            "type": "twitter",
                            "title": f"Tweet by @{tweet_data['author']}",
                            "text": tweet_data["text"][:300],  # Truncate for storage
                            "author": tweet_data["author"],
                            "url": tweet_data["url"],
                            "created_at": tweet_data["created_at"],
                            "engagement": {
                                "retweets": tweet_data["retweet_count"],
                                "likes": tweet_data["favorite_count"],
                                "replies": tweet_data["reply_count"],
                            },
                        }
                    )
                    processed_count += 1

            return {
                "query": query,
                "results": formatted_results,
                "sources": sources,
                "count": len(formatted_results),
                "metadata": {
                    "authenticated": self.authenticated,
                    "search_type": search_type,
                    "lang": lang,
                    "result_type": result_type,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

        except TwitterException as e:
            error_code = getattr(e, "code", None) or getattr(e, "status_code", None)
            error_msg = str(e)

            logger.error(f"Twitter API error (code {error_code}): {error_msg}")

            return {
                "query": query,
                "results": [],
                "sources": [],
                "error": f"Twitter API error: {error_msg}",
                "error_code": error_code,
                "count": 0,
                "metadata": {
                    "authenticated": self.authenticated,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }
        except Exception as e:
            logger.error(f"Unexpected error in Twitter search: {e}", exc_info=True)
            return {
                "query": query,
                "results": [],
                "sources": [],
                "error": f"Unexpected error: {str(e)}",
                "count": 0,
                "metadata": {
                    "authenticated": self.authenticated,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }
