"""
Test suite for embedding store module.
"""

import numpy as np
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from luki_memory.storage.vector_store import (
    EmbeddingStore,
    EmbeddingStoreError,
    create_embedding_store
)
from luki_memory.ingestion.chunker import ELRChunk


class TestEmbeddingStore:
    """Test EmbeddingStore class."""
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer for testing."""
        with patch('luki_memory.storage.vector_store.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = np.random.rand(384)
            mock_st.return_value = mock_model
            yield mock_model
    
    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client for testing."""
        with patch('luki_memory.storage.vector_store.chromadb.PersistentClient') as mock_client:
            mock_instance = Mock()
            mock_collection = Mock()
            
            # Configure collection mock methods
            mock_collection.count.return_value = 0
            mock_collection.add.return_value = None
            mock_collection.query.return_value = {
                'ids': [['test_id_1']],
                'distances': [[0.1]],
                'metadatas': [[{'user_id': 'test_user'}]],
                'documents': [['test document']]
            }
            mock_collection.name = 'test_collection'
            
            # Configure client mock methods
            mock_instance.get_or_create_collection.return_value = mock_collection
            mock_client.return_value = mock_instance
            yield mock_instance, mock_collection
    
    @pytest.fixture
    def embedding_store(self, mock_sentence_transformer, mock_chroma_client):
        """Create embedding store instance for testing."""
        client, collection = mock_chroma_client
        
        with patch('luki_memory.storage.vector_store.chromadb.PersistentClient') as mock_persistent:
            mock_persistent.return_value = client
            store = EmbeddingStore(
                model_name="test-model",
                persist_directory="./test_db",
                collection_name="test_collection"
            )
            store.collection = collection
            store.embedding_model = mock_sentence_transformer
            return store
    
    @pytest.fixture
    def sample_elr_chunk(self):
        """Sample ELR chunk for testing."""
        return ELRChunk(
            text="I love gardening and spending time outdoors.",
            chunk_id="test_chunk_1",
            parent_item_id="test_item_1",
            chunk_index=0,
            total_chunks=1,
            user_id="test_user_123"
        )
    
    def test_embedding_store_initialization(self, mock_sentence_transformer, mock_chroma_client):
        """Test embedding store initialization."""
        client, collection = mock_chroma_client
        
        store = EmbeddingStore(
            model_name="test-model",
            persist_directory="./test_db",
            collection_name="test_collection"
        )
        
        assert store.model_name == "test-model"
        assert store.collection_name == "test_collection"
        assert store.embedding_dim == 384
    
    def test_embedding_store_init_model_error(self, mock_chroma_client):
        """Test embedding store initialization with model loading error."""
        with patch('luki_memory.storage.vector_store.SentenceTransformer', side_effect=Exception("Model error")):
            with pytest.raises(EmbeddingStoreError, match="Failed to load embedding model"):
                EmbeddingStore()
    
    def test_embedding_store_init_chroma_error(self, mock_sentence_transformer):
        """Test embedding store initialization with ChromaDB error."""
        with patch('luki_memory.storage.vector_store.chromadb.PersistentClient', side_effect=Exception("DB error")):
            with pytest.raises(EmbeddingStoreError, match="Failed to initialize ChromaDB"):
                EmbeddingStore()
    
    def test_generate_embedding(self, embedding_store):
        """Test single embedding generation."""
        test_text = "This is a test sentence."
        
        embedding = embedding_store.generate_embedding(test_text)
        
        assert isinstance(embedding, np.ndarray)
        embedding_store.embedding_model.encode.assert_called_once_with(
            test_text, convert_to_numpy=True
        )
    
    def test_generate_embedding_error(self, embedding_store):
        """Test embedding generation with error."""
        embedding_store.embedding_model.encode.side_effect = Exception("Encoding error")
        
        with pytest.raises(EmbeddingStoreError, match="Failed to generate embedding"):
            embedding_store.generate_embedding("test")
    
    def test_generate_embeddings_batch(self, embedding_store):
        """Test batch embedding generation."""
        test_texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embedding_store.embedding_model.encode.return_value = np.random.rand(3, 384)
        
        embeddings = embedding_store.generate_embeddings_batch(test_texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
        embedding_store.embedding_model.encode.assert_called_once_with(
            test_texts, convert_to_numpy=True, show_progress_bar=False
        )
    
    def test_generate_embeddings_batch_with_progress(self, embedding_store):
        """Test batch embedding generation with progress bar."""
        test_texts = ["Text"] * 15  # More than 10 texts
        embedding_store.embedding_model.encode.return_value = np.random.rand(15, 384)
        
        embedding_store.generate_embeddings_batch(test_texts)
        
        embedding_store.embedding_model.encode.assert_called_once_with(
            test_texts, convert_to_numpy=True, show_progress_bar=True
        )
    
    def test_add_chunk(self, embedding_store, sample_elr_chunk):
        """Test adding single chunk to store."""
        # Mock the generate_embedding method properly
        with patch.object(embedding_store, 'generate_embedding') as mock_gen_embedding:
            mock_gen_embedding.return_value = np.random.rand(384)
            
            chunk_id = embedding_store.add_chunk(sample_elr_chunk)
            
            assert chunk_id == "test_chunk_1"
            embedding_store.collection.add.assert_called_once()
        
        # Verify the call arguments
        call_args = embedding_store.collection.add.call_args
        assert call_args[1]["ids"] == ["test_chunk_1"]
        assert call_args[1]["documents"] == [sample_elr_chunk.text]
    
    def test_add_chunk_without_id(self, embedding_store):
        """Test adding chunk without predefined ID."""
        chunk = ELRChunk(
            text="Test content",
            chunk_id="test_chunk_2",
            parent_item_id="test_item_2",
            chunk_index=0,
            total_chunks=1,
            user_id="test_user"
        )
        embedding_store.generate_embedding.return_value = np.random.rand(384)
        
        chunk_id = embedding_store.add_chunk(chunk)
        
        assert chunk_id is not None
        assert len(chunk_id) > 0  # UUID should be generated
        embedding_store.collection.add.assert_called_once()
    
    def test_add_chunk_error(self, embedding_store, sample_elr_chunk):
        """Test adding chunk with error."""
        embedding_store.collection.add.side_effect = Exception("Add error")
        
        with pytest.raises(EmbeddingStoreError, match="Failed to add chunk to store"):
            embedding_store.add_chunk(sample_elr_chunk)
    
    def test_add_chunks_batch(self, embedding_store):
        """Test adding multiple chunks in batch."""
        chunks = [
            ELRChunk(
                text="First chunk",
                chunk_id="chunk1",
                parent_item_id="item1",
                chunk_index=0,
                total_chunks=2,
                user_id="test_user"
            ),
            ELRChunk(
                text="Second chunk",
                chunk_id="chunk2",
                parent_item_id="item1",
                chunk_index=1,
                total_chunks=2,
                user_id="test_user"
            )
        ]
        embedding_store.generate_embeddings_batch.return_value = np.random.rand(2, 384)
        
        chunk_ids = embedding_store.add_chunks_batch(chunks)
        
        assert len(chunk_ids) == 2
        embedding_store.collection.add.assert_called_once()
        
        # Verify batch call
        call_args = embedding_store.collection.add.call_args
        assert len(call_args[1]["ids"]) == 2
        assert len(call_args[1]["documents"]) == 2
    
    def test_add_chunks_batch_empty(self, embedding_store):
        """Test adding empty batch."""
        chunk_ids = embedding_store.add_chunks_batch([])
        
        assert chunk_ids == []
        embedding_store.collection.add.assert_not_called()
    
    def test_search_similar(self, embedding_store):
        """Test similarity search."""
        # Mock ChromaDB query response
        mock_results = {
            "ids": [["chunk1", "chunk2"]],
            "documents": [["First document", "Second document"]],
            "metadatas": [[{"section": "test1"}, {"section": "test2"}]],
            "distances": [[0.1, 0.3]]  # High similarity
        }
        embedding_store.collection.query.return_value = mock_results
        embedding_store.generate_embedding.return_value = np.random.rand(384)
        
        results = embedding_store.search_similar("test query", k=2)
        
        assert len(results) == 2
        assert results[0]["similarity"] > results[1]["similarity"]  # Sorted by similarity
        assert all("id" in result for result in results)
        assert all("content" in result for result in results)
        assert all("similarity" in result for result in results)
    
    def test_search_similar_with_filters(self, embedding_store):
        """Test similarity search with consent and metadata filters."""
        mock_results = {
            "ids": [["chunk1"]],
            "documents": [["Test document"]],
            "metadatas": [[{"section": "test"}]],
            "distances": [[0.2]]
        }
        embedding_store.collection.query.return_value = mock_results
        embedding_store.generate_embedding.return_value = np.random.rand(384)
        
        results = embedding_store.search_similar(
            "test query",
            consent_filter=["private", "family"],
            metadata_filter={"section": "interests"}
        )
        
        # Verify where clause was constructed
        call_args = embedding_store.collection.query.call_args
        where_clause = call_args[1]["where"]
        assert "consent_level" in where_clause
        assert "section" in where_clause
    
    def test_search_similar_threshold_filtering(self, embedding_store):
        """Test similarity search with threshold filtering."""
        # Mock results with varying distances
        mock_results = {
            "ids": [["chunk1", "chunk2", "chunk3"]],
            "documents": [["Doc1", "Doc2", "Doc3"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.1, 0.5, 0.8]]  # Only first two should pass threshold 0.25
        }
        embedding_store.collection.query.return_value = mock_results
        embedding_store.generate_embedding.return_value = np.random.rand(384)
        
        results = embedding_store.search_similar("test", similarity_threshold=0.25)
        
        assert len(results) == 2  # Only chunks with similarity >= 0.75
        assert all(result["similarity"] >= 0.25 for result in results)
    
    def test_get_chunk_by_id(self, embedding_store):
        """Test retrieving chunk by ID."""
        mock_results = {
            "ids": ["chunk1"],
            "documents": ["Test document"],
            "metadatas": [{"section": "test"}]
        }
        embedding_store.collection.get.return_value = mock_results
        
        chunk = embedding_store.get_chunk_by_id("chunk1")
        
        assert chunk is not None
        assert chunk["id"] == "chunk1"
        assert chunk["content"] == "Test document"
        assert chunk["metadata"]["section"] == "test"
    
    def test_get_chunk_by_id_not_found(self, embedding_store):
        """Test retrieving non-existent chunk."""
        embedding_store.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        
        chunk = embedding_store.get_chunk_by_id("nonexistent")
        
        assert chunk is None
    
    def test_delete_chunk(self, embedding_store):
        """Test deleting chunk."""
        result = embedding_store.delete_chunk("chunk1")
        
        assert result is True
        embedding_store.collection.delete.assert_called_once_with(ids=["chunk1"])
    
    def test_delete_chunk_error(self, embedding_store):
        """Test deleting chunk with error."""
        embedding_store.collection.delete.side_effect = Exception("Delete error")
        
        result = embedding_store.delete_chunk("chunk1")
        
        assert result is False
    
    def test_get_collection_stats(self, embedding_store):
        """Test getting collection statistics."""
        embedding_store.collection.count.return_value = 42
        
        stats = embedding_store.get_collection_stats()
        
        assert stats["total_chunks"] == 42
        assert stats["collection_name"] == "test_collection"
        assert stats["model_name"] == "test-model"
        assert stats["embedding_dimension"] == 384
    
    def test_get_collection_stats_error(self, embedding_store):
        """Test getting collection stats with error."""
        embedding_store.collection.count.side_effect = Exception("Count error")
        
        stats = embedding_store.get_collection_stats()
        
        assert stats == {}
    
    def test_reset_collection(self, embedding_store):
        """Test resetting collection."""
        # Mock the chroma client methods
        embedding_store.chroma_client.delete_collection = Mock()
        embedding_store.chroma_client.create_collection = Mock()
        new_collection = Mock()
        embedding_store.chroma_client.create_collection.return_value = new_collection
        
        result = embedding_store.reset_collection()
        
        assert result is True
        embedding_store.chroma_client.delete_collection.assert_called_once_with("test_collection")
        embedding_store.chroma_client.create_collection.assert_called_once()
        assert embedding_store.collection == new_collection
    
    def test_reset_collection_error(self, embedding_store):
        """Test resetting collection with error."""
        embedding_store.chroma_client.delete_collection = Mock(side_effect=Exception("Reset error"))
        
        result = embedding_store.reset_collection()
        
        assert result is False


class TestCreateEmbeddingStore:
    """Test create_embedding_store factory function."""
    
    def test_create_embedding_store_default(self):
        """Test factory function with default parameters."""
        with patch('luki_memory.storage.vector_store.EmbeddingStore') as mock_store:
            create_embedding_store()
            
            mock_store.assert_called_once_with(
                "all-MiniLM-L12-v2",
                "./chroma_db", 
                "elr_embeddings"
            )
    
    def test_create_embedding_store_custom(self):
        """Test factory function with custom parameters."""
        with patch('luki_memory.storage.vector_store.EmbeddingStore') as mock_store:
            create_embedding_store(
                model_name="custom-model",
                persist_directory="./custom_db",
                collection_name="custom_collection"
            )
            
            mock_store.assert_called_once_with(
                "custom-model",
                "./custom_db",
                "custom_collection"
            )


# Integration tests
class TestEmbeddingStoreIntegration:
    """Integration tests for embedding store."""
    
    @pytest.mark.integration
    def test_full_pipeline_with_real_models(self, tmp_path):
        """Test full pipeline with real SentenceTransformer and ChromaDB."""
        pytest.skip("Integration test - requires actual models and ChromaDB")
    
    @pytest.mark.integration
    def test_large_batch_processing(self, tmp_path):
        """Test processing large batches of embeddings."""
        pytest.skip("Integration test - requires performance testing")
    
    @pytest.mark.integration
    def test_persistence_across_sessions(self, tmp_path):
        """Test that embeddings persist across store instances."""
        pytest.skip("Integration test - requires file system persistence")


if __name__ == "__main__":
    pytest.main([__file__])
