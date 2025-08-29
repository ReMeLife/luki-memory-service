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

import spacy
from spacy.lang.en import English

from ..schemas.elr import ConsentLevel, SensitivityLevel, ELRContentType

logger = logging.getLogger(__name__)


@dataclass
class ELRChunk:
    """Represents a processed chunk of ELR data ready for embedding."""
    
    # Core content
    text: str
    chunk_id: str
    
    # Chunk metadata
    parent_item_id: str
    chunk_index: int
    total_chunks: int
    
    # Content boundaries
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    # Inherited metadata
    user_id: str = ""
    content_type: ELRContentType = ELRContentType.MEMORY
    consent_level: ConsentLevel = ConsentLevel.PRIVATE
    sensitivity_level: SensitivityLevel = SensitivityLevel.PERSONAL
    
    # Processing info
    created_at: datetime = field(default_factory=datetime.utcnow)
    embedding_model: Optional[str] = None
    
    # Quality metrics
    chunk_quality_score: Optional[float] = None
    
    # Source tracking
    source_file: Optional[str] = None
    
    # Legacy support for existing code
    metadata: Dict[str, Union[str, int, float, List[str]]] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        # Ensure chunk_id is set if empty
        if not self.chunk_id:
            import uuid
            self.chunk_id = str(uuid.uuid4())
    
    @property
    def content(self) -> str:
        """Legacy property for backward compatibility."""
        return self.text


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
            # Add sentencizer if not present
            if 'sentencizer' not in self.nlp.pipe_names:
                self.nlp.add_pipe('sentencizer')
        except OSError:
            logger.warning(f"spaCy model {spacy_model} not found, using basic English")
            self.nlp = English()
            # Add sentencizer to basic English pipeline
            self.nlp.add_pipe('sentencizer')
        
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
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        # Extract metadata values with defaults
        consent_level = ConsentLevel(metadata.get("consent_level", "private"))
        content_type = ELRContentType(metadata.get("content_type", "memory"))
        sensitivity_level = SensitivityLevel(metadata.get("sensitivity_level", "personal"))
        
        for sentence in sentences:
            sentence_tokens = len(self.nlp(sentence))
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "token_count": current_tokens,
                })
                
                chunks.append(ELRChunk(
                    text=current_chunk.strip(),
                    chunk_id="",  # Will be generated in __post_init__
                    parent_item_id=parent_item_id,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated after all chunks are created
                    user_id=user_id,
                    content_type=content_type,
                    consent_level=consent_level,
                    sensitivity_level=sensitivity_level,
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
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "token_count": current_tokens,
            })
            
            chunks.append(ELRChunk(
                text=current_chunk.strip(),
                chunk_id="",  # Will be generated in __post_init__
                parent_item_id=parent_item_id,
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated below
                user_id=user_id,
                content_type=content_type,
                consent_level=consent_level,
                sensitivity_level=sensitivity_level,
                metadata=chunk_metadata
            ))
        
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
