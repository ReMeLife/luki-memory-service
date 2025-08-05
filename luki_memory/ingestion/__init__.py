"""
ELR Ingestion Module
Handles the ingestion and processing of ELR (Episodic Life Record) data.
"""

from .pipeline import ELRPipeline, ELRProcessingResult, create_pipeline
from .chunker import ELRChunk, TextChunker, create_chunker
from .embedder import TextEmbedder, create_embedder
from .redact import TextRedactor, create_redactor

__all__ = [
    "ELRPipeline",
    "ELRProcessingResult", 
    "ELRChunk",
    "TextChunker",
    "TextEmbedder",
    "TextRedactor",
    "create_pipeline",
    "create_chunker", 
    "create_embedder",
    "create_redactor"
]
