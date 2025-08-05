"""
Test suite for ELR ingestion module.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from ingestion.elr_ingestion import (
    ELRChunk,
    ELRProcessor,
    ELRProcessingResult,
    ELRIngestionError,
    process_elr_file
)


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
    
    def test_elr_chunk_with_custom_datetime(self):
        """Test ELR chunk with custom creation time."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        chunk = ELRChunk(
            content="Test content",
            metadata={},
            created_at=custom_time
        )
        
        assert chunk.created_at == custom_time


class TestELRProcessor:
    """Test ELRProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create ELR processor instance for testing."""
        with patch('ingestion.elr_ingestion.spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp
            processor = ELRProcessor()
            yield processor
    
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
    
    def test_processor_initialization(self):
        """Test processor initialization with spaCy model."""
        with patch('ingestion.elr_ingestion.spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp
            
            processor = ELRProcessor("en_core_web_sm")
            
            mock_load.assert_called_once_with("en_core_web_sm")
            assert processor.nlp == mock_nlp
    
    def test_processor_fallback_to_english(self):
        """Test processor fallback when spaCy model not found."""
        with patch('ingestion.elr_ingestion.spacy.load', side_effect=OSError):
            with patch('ingestion.elr_ingestion.English') as mock_english:
                mock_nlp = Mock()
                mock_english.return_value = mock_nlp
                
                processor = ELRProcessor("nonexistent_model")
                
                mock_english.assert_called_once()
                assert processor.nlp == mock_nlp
    
    def test_load_elr_file_success(self, processor, tmp_path):
        """Test successful ELR file loading."""
        # Create test file
        test_data = {"test": "data"}
        test_file = tmp_path / "test_elr.json"
        test_file.write_text(json.dumps(test_data))
        
        result = processor.load_elr_file(test_file)
        
        assert result == test_data
    
    def test_load_elr_file_not_found(self, processor):
        """Test ELR file loading with missing file."""
        with pytest.raises(ELRIngestionError, match="Failed to load ELR file"):
            processor.load_elr_file("nonexistent.json")
    
    def test_load_elr_file_invalid_json(self, processor, tmp_path):
        """Test ELR file loading with invalid JSON."""
        test_file = tmp_path / "invalid.json"
        test_file.write_text("invalid json content")
        
        with pytest.raises(ELRIngestionError, match="Failed to load ELR file"):
            processor.load_elr_file(test_file)
    
    def test_extract_entities(self, processor):
        """Test entity extraction from text."""
        # Mock spaCy processing
        mock_doc = Mock()
        mock_entity = Mock()
        mock_entity.label_ = "PERSON"
        mock_entity.text = "John"
        mock_doc.ents = [mock_entity]
        processor.nlp.return_value = mock_doc
        
        entities = processor.extract_entities("John is a person")
        
        assert "PERSON" in entities
        assert "John" in entities["PERSON"]
    
    def test_analyze_sentiment(self, processor):
        """Test sentiment analysis (placeholder implementation)."""
        sentiment = processor.analyze_sentiment("This is a test")
        
        assert "polarity" in sentiment
        assert "subjectivity" in sentiment
        assert "confidence" in sentiment
        assert sentiment["polarity"] == 0.0  # Neutral for MVP
    
    def test_chunk_text_basic(self, processor):
        """Test basic text chunking."""
        # Mock spaCy processing
        mock_doc = Mock()
        mock_sent1 = Mock()
        mock_sent1.text = "First sentence."
        mock_sent2 = Mock()
        mock_sent2.text = "Second sentence."
        mock_doc.sents = [mock_sent1, mock_sent2]
        processor.nlp.return_value = mock_doc
        
        # Mock token counting
        def mock_nlp_call(text):
            if "First sentence" in text:
                return [1] * 10  # 10 tokens
            elif "Second sentence" in text:
                return [1] * 10  # 10 tokens
            return [1] * len(text.split())
        
        processor.nlp.side_effect = mock_nlp_call
        
        metadata = {"section": "test"}
        chunks = processor.chunk_text("First sentence. Second sentence.", metadata)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, ELRChunk) for chunk in chunks)
    
    def test_process_elr_data(self, processor, sample_elr_data):
        """Test processing complete ELR data."""
        # Mock the _process_section method
        with patch.object(processor, '_process_section') as mock_process:
            mock_chunk = ELRChunk(
                content="test content",
                metadata={"section": "test"}
            )
            mock_process.return_value = [mock_chunk]
            
            result = processor.process_elr_data(sample_elr_data, "test_file.json")
            
            assert isinstance(result, ELRProcessingResult)
            assert result.total_processed > 0
            assert len(result.chunks) > 0
            assert isinstance(result.processing_time, float)
    
    def test_process_section_dict(self, processor):
        """Test processing dictionary section."""
        section_data = {
            "key1": "Some text content",
            "key2": "More text content"
        }
        
        # Mock chunking
        with patch.object(processor, 'chunk_text') as mock_chunk:
            mock_chunk.return_value = [ELRChunk("test", {})]
            
            chunks = processor._process_section("test_section", section_data, "test_file")
            
            assert len(chunks) > 0
            assert mock_chunk.call_count == 2  # Called for each text value
    
    def test_process_section_list(self, processor):
        """Test processing list section."""
        section_data = ["First item", "Second item"]
        
        # Mock chunking
        with patch.object(processor, 'chunk_text') as mock_chunk:
            mock_chunk.return_value = [ELRChunk("test", {})]
            
            chunks = processor._process_section("test_section", section_data, "test_file")
            
            assert len(chunks) > 0
            assert mock_chunk.call_count == 2  # Called for each list item


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
        
        with patch('ingestion.elr_ingestion.ELRProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_result = ELRProcessingResult(
                chunks=[],
                total_processed=0,
                errors=[],
                processing_time=0.1
            )
            mock_processor.load_elr_file.return_value = test_data
            mock_processor.process_elr_data.return_value = mock_result
            mock_processor_class.return_value = mock_processor
            
            result = process_elr_file(test_file)
            
            assert isinstance(result, ELRProcessingResult)
            mock_processor.load_elr_file.assert_called_once_with(test_file)
            mock_processor.process_elr_data.assert_called_once_with(test_data, str(test_file))


class TestELRProcessingResult:
    """Test ELRProcessingResult dataclass."""
    
    def test_processing_result_creation(self):
        """Test processing result creation."""
        chunks = [ELRChunk("test", {})]
        result = ELRProcessingResult(
            chunks=chunks,
            total_processed=1,
            errors=[],
            processing_time=0.5
        )
        
        assert result.chunks == chunks
        assert result.total_processed == 1
        assert result.errors == []
        assert result.processing_time == 0.5


# Integration tests
class TestELRIngestionIntegration:
    """Integration tests for ELR ingestion."""
    
    @pytest.mark.integration
    def test_full_pipeline(self, tmp_path):
        """Test full ELR ingestion pipeline."""
        # This test would require actual spaCy model
        # Skip in unit tests, run only in integration test suite
        pytest.skip("Integration test - requires spaCy model")
    
    @pytest.mark.integration
    def test_large_file_processing(self, tmp_path):
        """Test processing large ELR files."""
        pytest.skip("Integration test - requires large test data")


if __name__ == "__main__":
    pytest.main([__file__])
