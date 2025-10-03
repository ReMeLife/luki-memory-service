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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Union, Optional

# Optional spacy import for advanced text processing
try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    English = None

from ..schemas.elr import ConsentLevel, SensitivityLevel, ELRContentType, ELRChunk as SchemaELRChunk

logger = logging.getLogger(__name__)


# Use the schema ELRChunk as the canonical definition
ELRChunk = SchemaELRChunk


class TextChunker:
    """Handles text chunking for embedding storage."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize chunker with spaCy model."""
        self.spacy_model = spacy_model
        self.nlp = None
        
        if SPACY_AVAILABLE and spacy is not None:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                try:
                    # Try the small model as fallback
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.warning(f"spaCy model {spacy_model} not found, using en_core_web_sm")
                except OSError:
                    logger.warning(f"spaCy model {spacy_model} not found, using basic English")
                    if English is not None:
                        self.nlp = English()
                        # Add sentencizer to basic English pipeline
                        self.nlp.add_pipe('sentencizer')
        else:
            logger.warning("spaCy not available, using simple text splitting")
        
        self.chunk_size = 512  # Max tokens per chunk
        self.overlap_size = 50  # Token overlap between chunks
    
    def chunk_text(self, text: str, metadata: Dict, parent_item_id: str = "", user_id: str = "") -> List[ELRChunk]:
        """
        Split text into chunks suitable for embedding.
        
        Args:
            text: Input text to chunk
            metadata: Base metadata for chunks
            parent_item_id: ID of the parent ELR item
            user_id: User ID for the chunks
            
        Returns:
            List of ELR chunks
        """
        # Use spacy if available, otherwise simple sentence splitting
        if self.nlp is not None:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
        else:
            # Simple sentence splitting fallback
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        # Extract metadata values with defaults
        consent_level = ConsentLevel(metadata.get("consent_level", "private"))
        content_type = ELRContentType(metadata.get("content_type", "memory"))
        sensitivity_level = SensitivityLevel(metadata.get("sensitivity_level", "personal"))
        
        for sentence in sentences:
            # Use spacy for token counting if available, otherwise estimate
            if self.nlp is not None:
                sentence_tokens = len(self.nlp(sentence))
            else:
                # Simple word-based token estimation
                sentence_tokens = len(sentence.split())
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "token_count": current_tokens,
                })
                
                chunks.append(ELRChunk(
                    text=current_chunk.strip(),
                    chunk_id=f"{parent_item_id}_chunk_{chunk_index}",
                    parent_item_id=parent_item_id,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated after all chunks are created
                    user_id=user_id,
                    content_type=content_type,
                    consent_level=consent_level,
                    sensitivity_level=sensitivity_level,
                    start_char=None,
                    end_char=None,
                    embedding_model=None,
                    chunk_quality_score=None,
                    source_file=metadata.get("source_file"),
                    metadata=chunk_metadata
                ))
                
                chunk_index += 1
                # Start new chunk with overlap
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk if any content remains
        if current_chunk.strip():
            chunk_text = current_chunk.strip()
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "token_count": current_tokens,
            })
            
            chunk = ELRChunk(
                text=chunk_text,
                chunk_id=f"{parent_item_id}_chunk_{chunk_index}",
                parent_item_id=parent_item_id,
                chunk_index=chunk_index,
                total_chunks=len(chunks) + 1,  # Include this final chunk
                user_id=user_id,
                content_type=content_type,
                consent_level=consent_level,
                sensitivity_level=sensitivity_level,
                start_char=None,
                end_char=None,
                embedding_model=None,
                chunk_quality_score=None,
                source_file=metadata.get("source_file"),
                metadata=chunk_metadata
            )
            
            chunks.append(chunk)
        
        # Update total_chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
        
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
