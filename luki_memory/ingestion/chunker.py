"""
Text/Media Chunking Module
Split text into chunks suitable for embedding storage.

Responsibilities:
- Split text into manageable chunks for embedding
- Handle token limits and overlap
- Preserve semantic boundaries
- Add chunk metadata
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Union

import spacy
from spacy.lang.en import English

logger = logging.getLogger(__name__)


@dataclass
class ELRChunk:
    """Represents a processed chunk of ELR data ready for embedding."""
    
    content: str
    metadata: Dict[str, Union[str, int, float, List[str]]]
    consent_level: str = "private"  # private, family, research
    chunk_id: str = ""
    source_file: str = ""
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class TextChunker:
    """Handles text chunking for embedding storage."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize text chunker.
        
        Args:
            spacy_model: Name of spaCy model to use for sentence segmentation
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"spaCy model {spacy_model} not found, using basic English")
            self.nlp = English()
        
        self.chunk_size = 512  # Max tokens per chunk
        self.overlap_size = 50  # Token overlap between chunks
    
    def chunk_text(self, text: str, metadata: Dict) -> List[ELRChunk]:
        """
        Split text into chunks suitable for embedding.
        
        Args:
            text: Input text to chunk
            metadata: Base metadata for chunks
            
        Returns:
            List of ELR chunks
        """
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.nlp(sentence))
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "token_count": current_tokens,
                })
                
                chunks.append(ELRChunk(
                    content=current_chunk.strip(),
                    metadata=chunk_metadata,
                    consent_level=metadata.get("consent_level", "private")
                ))
                
                # Start new chunk with overlap
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk if any content remains
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "token_count": current_tokens,
            })
            
            chunks.append(ELRChunk(
                content=current_chunk.strip(),
                metadata=chunk_metadata,
                consent_level=metadata.get("consent_level", "private")
            ))
        
        return chunks


def create_chunker(spacy_model: str = "en_core_web_sm") -> TextChunker:
    """
    Factory function to create a text chunker.
    
    Args:
        spacy_model: spaCy model to use
        
    Returns:
        Initialized TextChunker instance
    """
    return TextChunker(spacy_model)
