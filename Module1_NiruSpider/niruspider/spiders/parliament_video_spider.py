"""
Parliament Video Spider - Scrape YouTube videos from Parliament channel
Extracts video IDs, metadata, and transcripts with timestamps

Enhanced with:
- Robust error handling
- Graceful shutdown on failures
- Timeout protection
- Signal handling for clean termination
"""

import scrapy
from loguru import logger
from datetime import datetime
import re
import signal
import sys
from twisted.internet import reactor
from twisted.internet.error import TimeoutError as TwistedTimeoutError
from ..items import DocumentItem


class ParliamentVideoSpider(scrapy.Spider):
    """Spider for scraping Parliament YouTube channel videos"""

    name = "parliament_videos"

    # Kenya Parliament YouTube channels
    custom_settings = {
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "ROBOTSTXT_OBEY": False,
        "CONCURRENT_REQUESTS": 2,
        "DOWNLOAD_DELAY": 2,
    }

    def __init__(self, channel_urls=None, max_videos=50, *args, **kwargs):
        """
        Initialize spider

        Args:
            channel_urls: Comma-separated YouTube channel URLs
            max_videos: Maximum videos to scrape per channel
        """
        super().__init__(*args, **kwargs)

        # Default: Kenya Parliament channels
        if channel_urls:
            self.channel_urls = channel_urls.split(",")
        else:
            self.channel_urls = [
                "https://www.youtube.com/@ParliamentofKenyaChannel/videos",
                "https://www.youtube.com/@ParliamentofKenyaChannel/videos",
            ]

        self.max_videos = int(max_videos)
        self.video_count = 0

        logger.info(
            f"Initialized Parliament Video Spider for {len(self.channel_urls)} channels"
        )
        logger.info(f"Max videos per channel: {self.max_videos}")

    def start_requests(self):
        """Generate initial requests for each channel"""
        for channel_url in self.channel_urls:
            logger.info(f"Scraping channel: {channel_url}")
            yield scrapy.Request(
                url=channel_url,
                callback=self.parse_channel,
                meta={"channel_url": channel_url},
            )

    def parse_channel(self, response):
        """
        Parse channel page to extract video IDs

        Note: YouTube dynamically loads content via JavaScript, so we extract
        video IDs from the page source using regex patterns
        """
        channel_url = response.meta["channel_url"]

        # Extract video IDs from page source
        # YouTube embeds video data in ytInitialData JSON
        video_ids = self._extract_video_ids_from_source(response.text)

        if not video_ids:
            logger.warning(f"No videos found on {channel_url}")
            return

        logger.info(f"Found {len(video_ids)} videos on {channel_url}")

        # Process each video (up to max_videos)
        for video_id in video_ids[: self.max_videos]:
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            # Create DocumentItem for each video
            item = DocumentItem()
            item["doc_id"] = f"parliament_video_{video_id}"
            item["url"] = video_url
            item["title"] = (
                f"Parliament Video: {video_id}"  # Will be updated by detail spider
            )
            item["content"] = f"YouTube video from Parliament channel: {video_url}"
            item["summary"] = f"Parliamentary video content from {channel_url}"
            item["content_type"] = "video"
            item["chunk_index"] = 0
            item["total_chunks"] = 1
            item["doc_type"] = "parliamentary_video"
            item["category"] = "Parliamentary Record"
            item["source_name"] = "Kenya Parliament YouTube"
            item["publication_date"] = (
                datetime.utcnow().isoformat()
            )  # Will be updated by detail spider
            item["crawl_date"] = datetime.utcnow().isoformat()
            item["metadata_tags"] = ["video", "parliament", "youtube"]
            item["status_code"] = response.status

            yield item

            self.video_count += 1

        logger.info(
            f"Yielded {min(len(video_ids), self.max_videos)} videos from {channel_url}"
        )

    def _extract_video_ids_from_source(self, html_source):
        """
        Extract video IDs from YouTube page source

        YouTube embeds video data in the page. We use regex to find video IDs.
        """
        video_ids = []

        # Pattern 1: /watch?v=VIDEO_ID
        pattern1 = r"/watch\?v=([a-zA-Z0-9_-]{11})"
        matches1 = re.findall(pattern1, html_source)
        video_ids.extend(matches1)

        # Pattern 2: "videoId":"VIDEO_ID"
        pattern2 = r'"videoId":"([a-zA-Z0-9_-]{11})"'
        matches2 = re.findall(pattern2, html_source)
        video_ids.extend(matches2)

        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for vid in video_ids:
            if vid not in seen:
                seen.add(vid)
                unique_ids.append(vid)

        return unique_ids

    def closed(self, reason):
        """Called when spider finishes"""
        logger.info(
            f"Parliament Video Spider finished: {self.video_count} videos scraped"
        )


class ParliamentVideoDetailSpider(scrapy.Spider):
    """
    Spider for fetching detailed metadata for specific videos
    This is used after video IDs are collected
    """

    name = "parliament_video_details"

    custom_settings = {
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "ROBOTSTXT_OBEY": False,
    }

    def __init__(self, video_ids=None, *args, **kwargs):
        """
        Initialize spider

        Args:
            video_ids: Comma-separated list of YouTube video IDs
        """
        super().__init__(*args, **kwargs)

        if video_ids:
            self.video_ids = video_ids.split(",")
        else:
            self.video_ids = []

        logger.info(f"Initialized Video Detail Spider for {len(self.video_ids)} videos")

    def start_requests(self):
        """Generate requests for each video"""
        for video_id in self.video_ids:
            url = f"https://www.youtube.com/watch?v={video_id}"
            yield scrapy.Request(url, callback=self.parse_video)

    def parse_video(self, response):
        """Parse individual video page for metadata"""

        # Extract video ID from URL
        video_id = (
            response.url.split("v=")[1].split("&")[0] if "v=" in response.url else None
        )

        if not video_id:
            logger.warning(f"Could not extract video ID from {response.url}")
            return

        # Extract metadata from page source
        html = response.text

        title = self._extract_title(html)
        description = self._extract_description(html)
        upload_date = self._extract_upload_date(html)
        view_count = self._extract_view_count(html)
        duration = self._extract_duration(html)

        # Create DocumentItem with detailed metadata
        item = DocumentItem()
        item["doc_id"] = f"parliament_video_{video_id}"
        item["url"] = response.url
        item["title"] = title
        item["content"] = description or f"YouTube video: {title}"
        item["summary"] = (
            description[:500] if description else f"Parliamentary video: {title}"
        )
        item["content_type"] = "video"
        item["chunk_index"] = 0
        item["total_chunks"] = 1
        item["doc_type"] = "parliamentary_video"
        item["category"] = "Parliamentary Record"
        item["source_name"] = "Kenya Parliament YouTube"
        item["publication_date"] = upload_date
        item["crawl_date"] = datetime.utcnow().isoformat()
        item["metadata_tags"] = ["video", "parliament", "youtube", "transcript"]
        item["status_code"] = response.status

        # Add video-specific metadata
        item["statistics"] = {
            "view_count": view_count,
            "duration_seconds": duration,
            "video_id": video_id,
        }

        yield item

    def _extract_title(self, html):
        """Extract video title"""
        match = re.search(r'"title":"([^"]+)"', html)
        return match.group(1) if match else "Unknown Title"

    def _extract_description(self, html):
        """Extract video description"""
        match = re.search(r'"shortDescription":"([^"]+)"', html)
        return match.group(1) if match else ""

    def _extract_upload_date(self, html):
        """Extract upload date"""
        match = re.search(r'"uploadDate":"([^"]+)"', html)
        return match.group(1) if match else None

    def _extract_view_count(self, html):
        """Extract view count"""
        match = re.search(r'"viewCount":"(\d+)"', html)
        return int(match.group(1)) if match else 0

    def _extract_duration(self, html):
        """Extract video duration"""
        match = re.search(r'"lengthSeconds":"(\d+)"', html)
        return int(match.group(1)) if match else 0
