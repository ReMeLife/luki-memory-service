"""
ELR Ingestion Module
Load raw ELR JSON → NLP parse → text chunks

Responsibilities:
- Parse Electronic Life Record (ELR®) data from JSON format
- Extract entities, sentiment, and metadata using spaCy
- Chunk text for embedding storage
- Handle consent tags and privacy flags
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

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


@dataclass
class ELRProcessingResult:
    """Result of ELR processing operation."""
    
    chunks: List[ELRChunk]
    total_processed: int
    errors: List[str]
    processing_time: float


class ELRIngestionError(Exception):
    """Custom exception for ELR ingestion errors."""
    pass


class ELRProcessor:
    """Main class for processing ELR data."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize ELR processor.
        
        Args:
            spacy_model: Name of spaCy model to use for NLP processing
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"spaCy model {spacy_model} not found, using basic English")
            self.nlp = English()
        
        self.chunk_size = 512  # Max tokens per chunk
        self.overlap_size = 50  # Token overlap between chunks
    
    def load_elr_file(self, file_path: Union[str, Path]) -> Dict:
        """
        Load ELR data from JSON file.
        
        Args:
            file_path: Path to ELR JSON file
            
        Returns:
            Parsed ELR data dictionary
            
        Raises:
            ELRIngestionError: If file cannot be loaded or parsed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded ELR file: {file_path}")
            return data
            
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ELRIngestionError(f"Failed to load ELR file {file_path}: {e}")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary of entity types and their values
        """
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text (placeholder for sentiment analysis).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Sentiment scores dictionary
        """
        # TODO: Implement proper sentiment analysis
        # For MVP, return neutral sentiment
        return {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "confidence": 0.5
        }
    
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
                    "entities": self.extract_entities(current_chunk),
                    "sentiment": self.analyze_sentiment(current_chunk)
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
                "entities": self.extract_entities(current_chunk),
                "sentiment": self.analyze_sentiment(current_chunk)
            })
            
            chunks.append(ELRChunk(
                content=current_chunk.strip(),
                metadata=chunk_metadata,
                consent_level=metadata.get("consent_level", "private")
            ))
        
        return chunks
    
    def process_elr_data(self, elr_data: Dict, source_file: str = "") -> ELRProcessingResult:
        """
        Process complete ELR data structure into chunks.
        
        Args:
            elr_data: Raw ELR data dictionary
            source_file: Source file name for tracking
            
        Returns:
            Processing result with chunks and metadata
        """
        start_time = datetime.now()
        all_chunks = []
        errors = []
        total_processed = 0
        
        try:
            # Process different ELR sections
            sections = [
                ("life_story", elr_data.get("life_story", {})),
                ("preferences", elr_data.get("preferences", {})),
                ("memories", elr_data.get("memories", [])),
                ("family", elr_data.get("family", {})),
                ("interests", elr_data.get("interests", []))
            ]
            
            for section_name, section_data in sections:
                try:
                    chunks = self._process_section(section_name, section_data, source_file)
                    all_chunks.extend(chunks)
                    total_processed += len(chunks)
                    
                except Exception as e:
                    error_msg = f"Error processing section {section_name}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ELRProcessingResult(
                chunks=all_chunks,
                total_processed=total_processed,
                errors=errors,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise ELRIngestionError(f"Failed to process ELR data: {e}")
    
    def _process_section(self, section_name: str, section_data: Union[Dict, List], 
                        source_file: str) -> List[ELRChunk]:
        """
        Process a specific section of ELR data.
        
        Args:
            section_name: Name of the ELR section
            section_data: Data for this section
            source_file: Source file name
            
        Returns:
            List of chunks for this section
        """
        chunks = []
        
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                if isinstance(value, str) and value.strip():
                    metadata = {
                        "section": section_name,
                        "subsection": key,
                        "data_type": "text",
                        "source_file": source_file
                    }
                    chunks.extend(self.chunk_text(value, metadata))
                    
        elif isinstance(section_data, list):
            for i, item in enumerate(section_data):
                if isinstance(item, str) and item.strip():
                    metadata = {
                        "section": section_name,
                        "item_index": i,
                        "data_type": "list_item",
                        "source_file": source_file
                    }
                    chunks.extend(self.chunk_text(item, metadata))
                elif isinstance(item, dict):
                    # Handle nested dictionaries in lists
                    for key, value in item.items():
                        if isinstance(value, str) and value.strip():
                            metadata = {
                                "section": section_name,
                                "item_index": i,
                                "subsection": key,
                                "data_type": "nested_text",
                                "source_file": source_file
                            }
                            chunks.extend(self.chunk_text(value, metadata))
        
        return chunks


def process_elr_file(file_path: Union[str, Path], 
                    spacy_model: str = "en_core_web_sm") -> ELRProcessingResult:
    """
    Convenience function to process a single ELR file.
    
    Args:
        file_path: Path to ELR JSON file
        spacy_model: spaCy model to use
        
    Returns:
        Processing result
    """
    processor = ELRProcessor(spacy_model)
    elr_data = processor.load_elr_file(file_path)
    return processor.process_elr_data(elr_data, str(file_path))


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would be used like:
    # result = process_elr_file("path/to/elr_data.json")
    # print(f"Processed {result.total_processed} chunks in {result.processing_time:.2f}s")
