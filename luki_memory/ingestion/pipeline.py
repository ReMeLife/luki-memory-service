"""
Orchestration Module
Coordinate the ELR ingestion pipeline from JSON to processed chunks.

Responsibilities:
- Load and parse ELR JSON files
- Orchestrate chunking, embedding, and redaction
- Handle different ELR data sections
- Manage processing results and errors
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from .chunker import ELRChunk, TextChunker, create_chunker
from .redact import TextRedactor, create_redactor

logger = logging.getLogger(__name__)


# Import ELRProcessingResult from schemas instead of defining here
from ..schemas.elr import ELRProcessingResult


class ELRIngestionError(Exception):
    """Custom exception for ELR ingestion errors."""
    pass


class ELRPipeline:
    """Main orchestration class for ELR ingestion pipeline."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize ELR pipeline.
        
        Args:
            spacy_model: Name of spaCy model to use for NLP processing
        """
        self.chunker = create_chunker(spacy_model)
        self.redactor = create_redactor(spacy_model)
    
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
                    chunks = self._process_section(section_name, section_data, source_file, elr_data)
                    all_chunks.extend(chunks)
                    total_processed += len(chunks)
                    
                except Exception as e:
                    error_msg = f"Error processing section {section_name}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ELRProcessingResult(
                success=len(errors) == 0,
                processed_items=total_processed,
                failed_items=len(errors),
                chunks_created=len(all_chunks),
                chunks=all_chunks,
                errors=errors,
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            raise ELRIngestionError(f"Failed to process ELR data: {e}")
    
    def _process_section(self, section_name: str, section_data: Union[Dict, List], 
                        source_file: str, elr_data: Optional[Dict] = None) -> List[ELRChunk]:
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
                        "source_file": source_file,
                        "entities": self.redactor.extract_entities(value),
                        "sentiment": self.redactor.analyze_sentiment(value)
                    }
                    # Extract user_id from ELR data or use default
                    user_id = elr_data.get("user_id", "system") if elr_data else "system"
                    parent_item_id = f"{user_id}_{section_name}_{key}"
                    chunks.extend(self.chunker.chunk_text(value, metadata, parent_item_id, user_id))
                    
        elif isinstance(section_data, list):
            for i, item in enumerate(section_data):
                if isinstance(item, str) and item.strip():
                    metadata = {
                        "section": section_name,
                        "item_index": i,
                        "data_type": "list_item",
                        "source_file": source_file,
                        "entities": self.redactor.extract_entities(item),
                        "sentiment": self.redactor.analyze_sentiment(item)
                    }
                    # Extract user_id from ELR data or use default
                    user_id = elr_data.get("user_id", "system") if elr_data else "system"
                    parent_item_id = f"{user_id}_{section_name}_{i}"
                    chunks.extend(self.chunker.chunk_text(item, metadata, parent_item_id, user_id))
                elif isinstance(item, dict):
                    # Handle nested dictionaries in lists
                    for key, value in item.items():
                        if isinstance(value, str) and value.strip():
                            metadata = {
                                "section": section_name,
                                "item_index": i,
                                "subsection": key,
                                "data_type": "nested_text",
                                "source_file": source_file,
                                "entities": self.redactor.extract_entities(value),
                                "sentiment": self.redactor.analyze_sentiment(value)
                            }
                            # Extract user_id from ELR data or use default
                            user_id = elr_data.get("user_id", "system") if elr_data else "system"
                            parent_item_id = f"{user_id}_{section_name}_{i}_{key}"
                            chunks.extend(self.chunker.chunk_text(value, metadata, parent_item_id, user_id))
        
        return chunks
    
    def process_file(self, file_path: Union[str, Path]) -> ELRProcessingResult:
        """
        Process a single ELR file from start to finish.
        
        Args:
            file_path: Path to ELR JSON file
            
        Returns:
            Processing result with chunks and metadata
        """
        elr_data = self.load_elr_file(file_path)
        return self.process_elr_data(elr_data, str(file_path))


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
    pipeline = ELRPipeline(spacy_model)
    elr_data = pipeline.load_elr_file(file_path)
    return pipeline.process_elr_data(elr_data, str(file_path))


def create_pipeline(spacy_model: str = "en_core_web_sm") -> ELRPipeline:
    """
    Factory function to create an ELR pipeline.
    
    Args:
        spacy_model: spaCy model to use
        
    Returns:
        Initialized ELRPipeline instance
    """
    return ELRPipeline(spacy_model)
