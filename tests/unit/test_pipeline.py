"""
Test suite for ELR ingestion pipeline.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from luki_memory.ingestion.chunker import ELRChunk, TextChunker
from luki_memory.ingestion.pipeline import (
    ELRProcessingResult,
    ELRIngestionError,
    ELRPipeline,
    process_elr_file
)
from luki_memory.ingestion.redact import TextRedactor


class TestELRChunk:
    """Test ELRChunk dataclass."""
    
    def test_elr_chunk_creation(self):
        """Test basic ELR chunk creation."""
        chunk = ELRChunk(
            text="Test content",
            chunk_id="test_chunk_1",
            parent_item_id="test_item_1",
            chunk_index=0,
            total_chunks=1,
            user_id="test_user"
        )
        
        assert chunk.text == "Test content"
        assert chunk.chunk_id == "test_chunk_1"
        assert chunk.parent_item_id == "test_item_1"
        assert isinstance(chunk.created_at, datetime)


class TestELRPipeline:
    """Test ELRPipeline class."""
    
    @pytest.fixture
    def pipeline(self):
        """Create ELR pipeline for testing."""
        with patch('luki_memory.ingestion.chunker.create_chunker') as mock_chunker_factory:
            with patch('luki_memory.ingestion.redact.create_redactor') as mock_redactor_factory:
                mock_chunker = Mock()
                mock_redactor = Mock()
                
                # Configure chunker mock to return proper chunks
                mock_chunker.chunk_text.return_value = [ELRChunk(
                    text="test content",
                    chunk_id="test_chunk_1", 
                    parent_item_id="test_item_1",
                    chunk_index=0,
                    total_chunks=1,
                    user_id="test_user"
                )]
                
                # Configure redactor mock to return proper data structures
                mock_redactor.extract_entities.return_value = {}
                mock_redactor.analyze_sentiment.return_value = {"polarity": 0.0}
                
                mock_chunker_factory.return_value = mock_chunker
                mock_redactor_factory.return_value = mock_redactor
                
                pipeline = ELRPipeline("en_core_web_sm")
                return pipeline
    
    @pytest.fixture
    def sample_elr_data(self):
        """Sample ELR data for testing."""
        return {
            "life_story": {
                "childhood": "I grew up in a small town with my parents and sister.",
                "career": "I worked as a teacher for 30 years."
            },
            "preferences": {
                "music": "I love classical music and jazz.",
                "food": "Italian cuisine is my favorite."
            },
            "memories": [
                "My wedding day in 1975 was the happiest day of my life.",
                "The birth of my first child brought me immense joy."
            ],
            "family": {
                "spouse": "Alice",
                "children": ["John", "Mary"]
            },
            "interests": ["gardening", "reading", "cooking"]
        }
    
    def test_load_elr_file_success(self, pipeline, tmp_path):
        """Test successful ELR file loading."""
        # Create test file
        test_data = {"test": "data"}
        test_file = tmp_path / "test_elr.json"
        test_file.write_text(json.dumps(test_data))
        
        result = pipeline.load_elr_file(test_file)
        
        assert result == test_data
    
    def test_load_elr_file_not_found(self, pipeline):
        """Test ELR file loading with missing file."""
        with pytest.raises(ELRIngestionError, match="Failed to load ELR file"):
            pipeline.load_elr_file("nonexistent.json")
    
    def test_process_elr_data(self, pipeline, sample_elr_data):
        """Test processing complete ELR data."""
        # Mock the chunker and redactor
        with patch.object(pipeline.chunker, 'chunk_text') as mock_chunk:
            with patch.object(pipeline.redactor, 'extract_entities') as mock_entities:
                with patch.object(pipeline.redactor, 'analyze_sentiment') as mock_sentiment:
                    mock_chunk.return_value = [ELRChunk(
                        text="test content",
                        chunk_id="test_chunk_1",
                        parent_item_id="test_item_1",
                        chunk_index=0,
                        total_chunks=1,
                        user_id="test_user"
                    )]
                    mock_entities.return_value = {}
                    mock_sentiment.return_value = {"polarity": 0.0}
                    
                    result = pipeline.process_elr_data(sample_elr_data, "test_file.json")
                    
                    assert isinstance(result, ELRProcessingResult)
                    assert result.processed_items > 0
                    assert result.chunks_created > 0
                    assert isinstance(result.processing_time_seconds, float)


class TestProcessELRFile:
    """Test process_elr_file convenience function."""
    
    def test_process_elr_file(self, tmp_path):
        """Test the convenience function."""
        # Create test file
        test_data = {
            "life_story": {"test": "content"},
            "preferences": {},
            "memories": [],
            "family": {},
            "interests": []
        }
        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(test_data))
        
        with patch('luki_memory.ingestion.pipeline.ELRPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            result = ELRProcessingResult(
                success=True,
                processed_items=1,
                chunks_created=1,
                errors=[],
                processing_time_seconds=0.5
            )
            mock_pipeline.load_elr_file.return_value = test_data
            mock_pipeline.process_elr_data.return_value = result
            mock_pipeline_class.return_value = mock_pipeline
            
            result = process_elr_file(test_file)
            
            assert isinstance(result, ELRProcessingResult)
            mock_pipeline.load_elr_file.assert_called_once_with(test_file)
            mock_pipeline.process_elr_data.assert_called_once_with(test_data, str(test_file))


if __name__ == "__main__":
    pytest.main([__file__])
