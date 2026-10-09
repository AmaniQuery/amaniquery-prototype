"""
Module 2: NiruParser - ETL & Embedding Pipeline
"""

__version__ = "2.0.0"

from .pipeline import ProcessingPipeline
from .config import Config
from .extractors import HTMLExtractor, PDFExtractor, TranscriptExtractor
from .cleaners import TextCleaner
from .chunkers import TextChunker
from .enrichers import MetadataEnricher, LegalMetadataEnricher
from .embedders import TextEmbedder, VisionEmbedder

__all__ = [
    "ProcessingPipeline",
    "Config",
    "HTMLExtractor",
    "PDFExtractor",
    "TranscriptExtractor",
    "TextCleaner",
    "TextChunker",
    "MetadataEnricher",
    "LegalMetadataEnricher",
    "TextEmbedder",
    "VisionEmbedder",
]
