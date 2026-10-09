"""
NiruSpider - Item Pipelines
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import sys
import os

import scrapy
from scrapy.pipelines.files import FilesPipeline
from itemadapter import ItemAdapter

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class VectorStorePipeline:
    """Save scraped items directly to vector store"""

    def __init__(self):
        self.vector_store = None
        self.text_embedder = None

    def _determine_namespace(self, category: str, publication_date: str) -> str:
        """Determine the appropriate namespace based on category and publication date"""
        # Check for historical data (pre-2010)
        if publication_date:
            try:
                # Extract year from publication date
                if isinstance(publication_date, str):
                    # Handle various date formats
                    import re

                    year_match = re.search(r"(\d{4})", publication_date)
                    if year_match:
                        year = int(year_match.group(1))
                        if year < 2010:
                            return "historical"
            except (ValueError, AttributeError):
                pass

        # Map categories to namespaces
        category_lower = category.lower()

        # Kenya Law namespace
        if any(
            keyword in category_lower
            for keyword in [
                "kenyan law",
                "case law",
                "kenya gazette",
                "kenya law blog",
                "cause lists",
                "constitution",
                "act",
                "legislation",
                "judgment",
            ]
        ):
            return "kenya_law"

        # Kenya News namespace
        elif any(keyword in category_lower for keyword in ["kenyan news", "news"]):
            return "kenya_news"

        # Kenya Parliament namespace
        elif any(
            keyword in category_lower
            for keyword in [
                "parliament",
                "parliamentary record",
                "bill",
                "hansard",
                "budget",
            ]
        ):
            return "kenya_parliament"

        # Global Trends namespace (default for global content)
        elif any(keyword in category_lower for keyword in ["global trend"]):
            return "global_trends"

        # Default fallback
        else:
            return "kenya_law"  # Default to kenya_law for legal content

    def open_spider(self, spider):
        """Initialize vector store connection"""
        try:
            from Module3_NiruDB.vector_store import VectorStore
            from Module2_NiruParser.embedders.text_embedder import TextEmbedder

            self.vector_store = VectorStore()
            self.text_embedder = TextEmbedder()

            spider.logger.info("Vector store pipeline initialized")
        except Exception as e:
            spider.logger.error(f"Failed to initialize vector store: {e}")
            raise

    def process_item(self, item, spider):
        """Process and save item to vector store"""
        if not self.vector_store or not self.text_embedder:
            spider.logger.warning(
                "Vector store or embedder not available, skipping item"
            )
            return item

        try:
            adapter = ItemAdapter(item)

            # Prepare document data
            doc_id = hashlib.md5(adapter["url"].encode()).hexdigest()

            # Prepare metadata
            metadata = {
                "title": adapter.get("title", ""),
                "source_url": adapter["url"],
                "source_name": adapter.get("source_name", ""),
                "category": adapter.get("category", "Unknown"),
                "publication_date": adapter.get("publication_date", ""),
                "author": adapter.get("author", ""),
                "content_type": adapter.get("content_type", "html"),
                "crawl_date": adapter.get("crawl_date", datetime.utcnow().isoformat()),
            }

            # Determine namespace based on category and publication date
            namespace = self._determine_namespace(
                metadata["category"], metadata["publication_date"]
            )

            # Prepare content for embedding
            content = adapter.get("content", "")
            if not content and adapter.get("summary"):
                content = adapter.get("summary")

            if content:
                # Chunk the content if it's too long
                chunk_size = 800  # Match the chunk size in settings
                chunks = []

                # Simple chunking by sentences/paragraphs
                paragraphs = content.split("\n\n")
                current_chunk = ""

                for para in paragraphs:
                    if len(current_chunk + para) <= chunk_size:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para + "\n\n"

                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If still no chunks, create one big chunk
                if not chunks:
                    chunks = [content[:chunk_size]]

                # Prepare chunks for embedding
                chunk_dicts = []
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    chunk_dict = {
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "title": metadata["title"],
                        "category": metadata["category"],
                        "source_url": metadata["source_url"],
                        "source_name": metadata["source_name"],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }

                    # Add optional metadata
                    if metadata.get("author"):
                        chunk_dict["author"] = metadata["author"]
                    if metadata.get("publication_date"):
                        chunk_dict["publication_date"] = metadata["publication_date"]

                    chunk_dicts.append(chunk_dict)

                # Generate embeddings
                embedded_chunks = self.text_embedder.embed_chunks(chunk_dicts)

                # Add to vector store with namespace
                self.vector_store.add_documents(embedded_chunks, namespace=namespace)

                spider.logger.info(
                    f"Added {len(embedded_chunks)} chunks for: {adapter['title'][:50]}... (namespace: {namespace})"
                )

            return item

        except Exception as e:
            spider.logger.error(f"Error processing item for vector store: {e}")
            return item


class DeduplicationPipeline:
    """Deduplicate articles using URL and content hashing"""

    def __init__(self):
        self.dedup_engine = None

    def open_spider(self, spider):
        """Initialize deduplication engine"""
        try:
            from .deduplication import DeduplicationEngine

            self.dedup_engine = DeduplicationEngine()
            spider.logger.info("Deduplication pipeline initialized")
        except Exception as e:
            spider.logger.error(f"Failed to initialize deduplication: {e}")
            self.dedup_engine = None

    def process_item(self, item, spider):
        """Check for duplicates and register article"""
        if not self.dedup_engine:
            return item

        adapter = ItemAdapter(item)
        url = adapter.get("url")
        content = adapter.get("content", "")
        title = adapter.get("title", "")
        source_name = adapter.get("source_name", "")

        # Parse publication date
        pub_date = None
        pub_date_str = adapter.get("publication_date")
        if pub_date_str:
            try:
                from dateutil import parser as date_parser

                pub_date = date_parser.parse(pub_date_str)
            except:
                pass

        # Check if duplicate
        is_dup, reason = self.dedup_engine.is_duplicate(url, content, title)
        if is_dup:
            spider.logger.info(
                f"[FAIL] Dropping duplicate article: {title[:50]}... (reason: {reason})"
            )
            raise scrapy.exceptions.DropItem(f"Duplicate article: {reason}")

        # Register article
        registered = self.dedup_engine.register_article(
            url=url,
            content=content,
            title=title,
            source_name=source_name,
            publication_date=pub_date,
        )

        if not registered:
            spider.logger.info(
                f"[FAIL] Failed to register article (likely duplicate): {title[:50]}..."
            )
            raise scrapy.exceptions.DropItem("Failed to register article")

        # Successfully passed deduplication
        spider.logger.info(f"[OK] New article registered: {title[:50]}...")

        return item


class QualityScoringPipeline:
    """Score articles for quality and filter low-quality articles"""

    def __init__(self, min_quality_score=0.6):
        self.quality_scorer = None
        self.min_quality_score = min_quality_score

    def open_spider(self, spider):
        """Initialize quality scorer"""
        try:
            from .quality_scorer import QualityScorer

            self.quality_scorer = QualityScorer()
            # Get min score from settings
            self.min_quality_score = spider.settings.getfloat("MIN_QUALITY_SCORE", 0.6)
            spider.logger.info(
                f"Quality scoring pipeline initialized (min_score={self.min_quality_score})"
            )
        except Exception as e:
            spider.logger.error(f"Failed to initialize quality scorer: {e}")
            self.quality_scorer = None

    def process_item(self, item, spider):
        """Score article and filter if below threshold"""
        if not self.quality_scorer:
            return item

        adapter = ItemAdapter(item)

        # Skip quality scoring for news_rss spider (make quality_score optional)
        if spider.name == "news_rss":
            adapter["quality_score"] = None
            adapter["quality_breakdown"] = None
            return item

        # Prepare article dict for scoring
        article = {
            "url": adapter.get("url", ""),
            "content": adapter.get("content", ""),
            "title": adapter.get("title", ""),
            "author": adapter.get("author", ""),
            "publication_date": adapter.get("publication_date", ""),
            "source_name": adapter.get("source_name", ""),
        }

        # Score article
        score_result = self.quality_scorer.score_article(article)

        # Add quality score to item
        adapter["quality_score"] = score_result["total_score"]
        adapter["quality_breakdown"] = score_result["breakdown"]

        # Filter if below threshold
        if score_result["total_score"] < self.min_quality_score:
            spider.logger.debug(
                f"Filtering low-quality article: {adapter.get('title', 'Unknown')[:50]} "
                f"(score: {score_result['total_score']:.2f})"
            )
            raise scrapy.exceptions.DropItem(
                f"Low quality score: {score_result['total_score']:.2f} < {self.min_quality_score}"
            )

        spider.logger.debug(
            f"Article quality score: {score_result['total_score']:.2f} - {adapter.get('title', 'Unknown')[:50]}"
        )
        return item


class DataValidationPipeline:
    """Validate and clean scraped data"""

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        # Ensure required fields
        if not adapter.get("url"):
            raise scrapy.exceptions.DropItem("Missing URL")

        if not adapter.get("title"):
            raise scrapy.exceptions.DropItem("Missing title")

        # Add crawl timestamp
        adapter["crawl_date"] = datetime.utcnow().isoformat()

        # Clean title
        if adapter.get("title"):
            adapter["title"] = adapter["title"].strip()

        # Ensure category is set
        if not adapter.get("category"):
            adapter["category"] = "Unknown"

        spider.logger.info(f"Validated item: {adapter['title'][:50]}...")
        return item


class PDFDownloadPipeline(FilesPipeline):
    """Handle PDF downloads"""

    def get_media_requests(self, item, info):
        adapter = ItemAdapter(item)

        # If content_type is PDF, download it
        if adapter.get("content_type") == "pdf" and adapter.get("url"):
            info.spider.logger.info(
                f"[DOC] Queueing PDF download: {adapter.get('title', '')[:50]}..."
            )
            yield scrapy.Request(adapter["url"], meta={"item": item})
        else:
            # Not a PDF, just pass through
            info.spider.logger.debug(
                f"Skipping PDF download for non-PDF item: {adapter.get('content_type', 'unknown')}"
            )

    def file_path(self, request, response=None, info=None, *, item=None):
        """Generate file path for PDFs"""
        adapter = ItemAdapter(item)

        # Create hash of URL for unique filename
        url_hash = hashlib.md5(adapter["url"].encode()).hexdigest()[:12]

        # Extract filename or use hash
        parsed = urlparse(adapter["url"])
        filename = Path(parsed.path).name or f"{url_hash}.pdf"

        # Organize by category
        category = adapter.get("category", "unknown").replace(" ", "_").lower()
        return f"{category}/{filename}"

    def item_completed(self, results, item, info):
        """Update item with PDF path"""
        adapter = ItemAdapter(item)

        # results is a list of (success, file_info) tuples
        file_paths = [x["path"] for ok, x in results if ok]

        if file_paths:
            adapter["pdf_path"] = file_paths[0]
            info.spider.logger.info(f"Downloaded PDF: {file_paths[0]}")

        return item


class FileStoragePipeline:
    """Save items to structured files"""

    def __init__(self, raw_data_path):
        self.raw_data_path = Path(raw_data_path)
        self.files = {}  # Keyed by spider name (string), not spider object

    @classmethod
    def from_crawler(cls, crawler):
        return cls(raw_data_path=crawler.settings.get("RAW_DATA_PATH"))

    def open_spider(self, spider):
        """Create output directory for this spider"""
        self.spider_dir = self.raw_data_path / spider.name
        self.spider_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.spider_dir / f"{spider.name}_{timestamp}.jsonl"

        # Close existing handle for same spider name if it leaked
        if spider.name in self.files:
            try:
                self.files[spider.name].close()
            except Exception:
                pass

        self.files[spider.name] = open(self.output_file, "w", encoding="utf-8")
        spider.logger.info(f"Saving data to: {self.output_file}")

    def close_spider(self, spider):
        """Close output file"""
        fh = self.files.pop(spider.name, None)
        if fh:
            try:
                fh.close()
            except Exception as e:
                spider.logger.warning(f"Error closing file for {spider.name}: {e}")

    def process_item(self, item, spider):
        """Write item to JSONL file"""
        fh = self.files.get(spider.name)
        if not fh:
            spider.logger.error(f"No open file handle for {spider.name}")
            return item
        adapter = ItemAdapter(item)
        line = json.dumps(dict(item), ensure_ascii=False, default=str) + "\n"
        fh.write(line)
        spider.logger.info(f"[SAVE] Saved to file: {adapter['title'][:50]}...")
        return item
