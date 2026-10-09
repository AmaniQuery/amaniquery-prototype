"""
Text Chunker using LangChain's RecursiveCharacterTextSplitter
"""

from typing import List, Dict
from loguru import logger
import hashlib

# Try different import paths for langchain text splitter (version compatibility)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain_community.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError(
                "RecursiveCharacterTextSplitter not found. "
                "Install with: pip install langchain-text-splitters or langchain"
            )


class TextChunker:
    """Chunk text into smaller pieces with overlap"""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: List[str] = None,
    ):
        """
        Initialize chunker

        Args:
            chunk_size: Target size of each chunk (characters)
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to split on (in order of preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if separators is None:
            separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

    def chunk(
        self,
        text: str,
        metadata: Dict = None,
    ) -> List[Dict]:
        """
        Chunk text with metadata

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunks with metadata
        """
        if not text:
            logger.warning("Empty text provided to chunker")
            return []

        if metadata is None:
            metadata = {}

        try:
            # Split text
            chunks = self.splitter.split_text(text)

            # Create chunk dictionaries with metadata
            chunk_dicts = []
            for i, chunk_text in enumerate(chunks):
                # Generate unique chunk ID
                chunk_id = self._generate_chunk_id(metadata.get("url", "unknown"), i)

                chunk_dict = {
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **metadata,  # Include all original metadata
                }

                chunk_dicts.append(chunk_dict)

            logger.info(f"Created {len(chunk_dicts)} chunks from document")
            return chunk_dicts

        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return []

    def _generate_chunk_id(self, url: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        # Create hash of URL for shorter ID
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        return f"{url_hash}_chunk_{chunk_index}"

    def chunk_with_semantic_boundaries(
        self,
        text: str,
        metadata: Dict = None,
    ) -> List[Dict]:
        """
        Chunk text while respecting semantic boundaries
        (e.g., don't split in the middle of a sentence)
        """
        # This is already handled by RecursiveCharacterTextSplitter
        # with appropriate separators
        return self.chunk(text, metadata)
