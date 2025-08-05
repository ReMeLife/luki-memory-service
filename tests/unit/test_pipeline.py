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
            content="Test content",
            metadata={"section": "test"},
            consent_level="private"
        )
        
        assert chunk.content == "Test content"
        assert chunk.metadata["section"] == "test"
        assert chunk.consent_level == "private"
        assert isinstance(chunk.created_at, datetime)


class TestELRPipeline:
    """Test ELRPipeline class."""
    
    @pytest.fixture
    def pipeline(self):
        """Create ELR pipeline instance for testing."""
        with patch('luki_memory.ingestion.chunker.spacy.load') as mock_load:
            with patch('luki_memory.ingestion.redact.spacy.load') as mock_redact_load:
                mock_nlp = Mock()
                mock_load.return_value = mock_nlp
                mock_redact_load.return_value = mock_nlp
                pipeline = ELRPipeline()
                yield pipeline
    
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
                    mock_chunk.return_value = [ELRChunk("test content", {"section": "test"})]
                    mock_entities.return_value = {}
                    mock_sentiment.return_value = {"polarity": 0.0}
                    
                    result = pipeline.process_elr_data(sample_elr_data, "test_file.json")
                    
                    assert isinstance(result, ELRProcessingResult)
                    assert result.total_processed > 0
                    assert len(result.chunks) > 0
                    assert isinstance(result.processing_time, float)


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
            mock_result = ELRProcessingResult(
                chunks=[],
                total_processed=0,
                errors=[],
                processing_time=0.1
            )
            mock_pipeline.load_elr_file.return_value = test_data
            mock_pipeline.process_elr_data.return_value = mock_result
            mock_pipeline_class.return_value = mock_pipeline
            
            result = process_elr_file(test_file)
            
            assert isinstance(result, ELRProcessingResult)
            mock_pipeline.load_elr_file.assert_called_once_with(test_file)
            mock_pipeline.process_elr_data.assert_called_once_with(test_data, str(test_file))


if __name__ == "__main__":
    pytest.main([__file__])
