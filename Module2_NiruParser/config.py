"""
Configuration for NiruParser
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Processing pipeline configuration"""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
    EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings"

    # Create directories
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)

    # Chunking settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
    MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", 100))

    # Text splitting separators (in order of preference)
    CHUNK_SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",  # Lines
        ". ",  # Sentences
        "! ",
        "? ",
        "; ",
        ", ",
        " ",  # Words
        "",  # Characters
    ]

    # Embedding settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_BATCH_SIZE = 32
    NORMALIZE_EMBEDDINGS = True

    # Processing
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 50))

    # Text extraction
    MIN_TEXT_LENGTH = 100  # Minimum characters for a valid document
    MAX_TEXT_LENGTH = 1_000_000  # Maximum characters per document

    # Metadata
    REQUIRED_METADATA = ["source_url", "title", "category", "chunk_id"]
    OPTIONAL_METADATA = ["author", "publication_date", "summary", "keywords"]

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def get_output_path(cls, category: str, filename: str) -> Path:
        """Get output path for processed file"""
        category_dir = cls.PROCESSED_DATA_PATH / category.replace(" ", "_").lower()
        category_dir.mkdir(parents=True, exist_ok=True)
        return category_dir / filename

    @classmethod
    def get_embedding_path(cls, category: str, filename: str) -> Path:
        """Get output path for embeddings"""
        category_dir = cls.EMBEDDINGS_PATH / category.replace(" ", "_").lower()
        category_dir.mkdir(parents=True, exist_ok=True)
        return category_dir / filename
