"""
Embedding Store Module
SentenceTransformer embeddings → ChromaDB vectors

Responsibilities:
- Generate embeddings using SentenceTransformers
- Store and retrieve vectors in ChromaDB
- Handle similarity search with consent filtering
- Manage vector database lifecycle
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ..ingestion.chunker import ELRChunk

logger = logging.getLogger(__name__)


class EmbeddingStoreError(Exception):
    """Custom exception for embedding store errors."""
    pass


class EmbeddingStore:
    """Main class for managing embeddings and vector storage."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L12-v2",
        persist_directory: str = "./chroma_db",
        collection_name: str = "elr_embeddings"
    ):
        """
        Initialize embedding store.
        
        Args:
            model_name: SentenceTransformer model name
            persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
        """
        self.model_name = model_name
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Initialize SentenceTransformer model
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            raise EmbeddingStoreError(f"Failed to load embedding model {model_name}: {e}")
        
        # Initialize ChromaDB client
        self._init_chroma_client()
    
    def _init_chroma_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persist directory if it doesn't exist
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with persistence - FORCE persistence, no fallback
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    persist_directory=str(self.persist_directory)
                )
            )
            logger.info(f"Using persistent ChromaDB at {self.persist_directory}")
            
            # Verify persistence is working by checking if directory contains data
            db_files = list(self.persist_directory.glob("*"))
            if db_files:
                logger.info(f"Found existing ChromaDB files: {[f.name for f in db_files]}")
            else:
                logger.info("Initialized new persistent ChromaDB instance")
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "ELR embeddings for LUKi agent"}
            )
            
            logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            raise EmbeddingStoreError(f"Failed to initialize ChromaDB: {e}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            raise EmbeddingStoreError(f"Failed to generate embedding: {e}")
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            Embedding matrix as numpy array
        """
        try:
            embeddings = self.embedding_model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10
            )
            return embeddings
        except Exception as e:
            raise EmbeddingStoreError(f"Failed to generate batch embeddings: {e}")
    
    def add_chunk(self, chunk: ELRChunk) -> str:
        """
        Add a single ELR chunk to the vector store.
        
        Args:
            chunk: ELR chunk to add
            
        Returns:
            Unique ID of the stored chunk
        """
        chunk_id = chunk.chunk_id or str(uuid.uuid4())
        
        try:
            # Generate embedding
            embedding = self.generate_embedding(chunk.text)
            
            # Prepare metadata for ChromaDB
            metadata = chunk.metadata.copy()
            elr_metadata = {
                "user_id": chunk.user_id,
                "consent_level": str(chunk.consent_level),
                "content_type": str(chunk.content_type),
                "sensitivity_level": str(chunk.sensitivity_level),
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "parent_item_id": chunk.parent_item_id,
                "chunk_id": chunk_id,
                "created_at": chunk.created_at.isoformat(),
                "source_file": chunk.source_file or "",
                "chunk_quality_score": chunk.chunk_quality_score or 0.0,
                "embedding_model": chunk.embedding_model or self.model_name
            }
            metadata.update(elr_metadata)
            
            # Convert any non-string metadata values to strings
            metadata = {k: str(v) if not isinstance(v, str) else v 
                       for k, v in metadata.items()}
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding.tolist()],
                documents=[chunk.text],
                metadatas=[metadata]
            )
            
            logger.debug(f"Added chunk {chunk_id} to embedding store")
            return chunk_id
            
        except Exception as e:
            raise EmbeddingStoreError(f"Failed to add chunk to store: {e}")
    
    def add_chunks_batch(self, chunks: List[ELRChunk]) -> List[str]:
        """
        Add multiple ELR chunks to the vector store in batch.
        
        Args:
            chunks: List of ELR chunks to add
            
        Returns:
            List of unique IDs of the stored chunks
        """
        if not chunks:
            return []
        
        try:
            # Generate chunk IDs
            chunk_ids = [chunk.chunk_id or str(uuid.uuid4()) for chunk in chunks]
            
            # Generate embeddings in batch
            texts = [chunk.text for chunk in chunks]
            embeddings = self.generate_embeddings_batch(texts)
            
            # Prepare metadata
            metadatas = []
            for chunk in chunks:
                metadata = chunk.metadata.copy()
                # Ensure required fields are present and canonicalized
                metadata.setdefault("user_id", chunk.user_id or "")
                batch_metadata = {
                    "user_id": chunk.user_id or metadata.get("user_id", ""),
                    "consent_level": getattr(chunk.consent_level, "value", str(chunk.consent_level)) if chunk.consent_level else metadata.get("consent_level", "private"),
                    "content_type": getattr(chunk.content_type, "value", str(chunk.content_type)) if chunk.content_type else metadata.get("content_type", "memory"),
                    "sensitivity_level": getattr(chunk.sensitivity_level, "value", str(chunk.sensitivity_level)) if chunk.sensitivity_level else metadata.get("sensitivity_level", "personal"),
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "parent_item_id": chunk.parent_item_id,
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file or metadata.get("source_file"),
                    "created_at": chunk.created_at.isoformat(),
                    "content_length": len(chunk.text)
                }
                metadata.update(batch_metadata)
                # Convert any non-string metadata values to strings
                metadata = {k: str(v) if not isinstance(v, str) else v 
                           for k, v in metadata.items()}
                metadatas.append(metadata)
            
            # Add to ChromaDB collection in batch
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks to embedding store")
            return chunk_ids
            
        except Exception as e:
            raise EmbeddingStoreError(f"Failed to add chunks batch to store: {e}")
    
    def search_similar(
        self,
        query: str,
        k: int = 6,
        similarity_threshold: float = 0.25,
        consent_filter: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Search for similar chunks in the vector store.
        
        Args:
            query: Search query text
            k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            consent_filter: List of allowed consent levels
            metadata_filter: Additional metadata filters
            
        Returns:
            List of similar chunks with metadata and scores
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Prepare where clause for filtering
            where_clause = {}
            
            if consent_filter:
                where_clause["consent_level"] = {"$in": consent_filter}
            
            if metadata_filter:
                where_clause.update(metadata_filter)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k * 2,  # Get more results to filter by threshold
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results and apply similarity threshold
            similar_chunks = []
            
            if results and results.get("ids") and results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results.get("distances") and results["distances"] and results["distances"][0] else 1.0
                    # For cosine distance, convert to similarity: similarity = 1 - (distance / 2)
                    # This handles cases where cosine distance > 1.0
                    similarity = max(0.0, 1.0 - (distance / 2.0))
                    
                    if similarity >= similarity_threshold:
                        similar_chunks.append({
                            "id": chunk_id,
                            "content": results["documents"][0][i] if results.get("documents") and results["documents"] and results["documents"][0] else "",
                            "metadata": results["metadatas"][0][i] if results.get("metadatas") and results["metadatas"] and results["metadatas"][0] else {},
                            "similarity": similarity,
                            "distance": distance
                        })
            
            # Sort by similarity and limit to k results
            similar_chunks.sort(key=lambda x: x["similarity"], reverse=True)
            similar_chunks = similar_chunks[:k]
            
            logger.debug(f"Found {len(similar_chunks)} similar chunks for query")
            return similar_chunks
            
        except Exception as e:
            raise EmbeddingStoreError(f"Failed to search similar chunks: {e}")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a specific chunk by its ID.
        
        Args:
            chunk_id: Unique ID of the chunk
            
        Returns:
            Chunk data or None if not found
        """
        try:
            results = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            
            if results and results.get("ids") and results["ids"]:
                return {
                    "id": chunk_id,
                    "content": results["documents"][0] if results.get("documents") and results["documents"] else "",
                    "metadata": results["metadatas"][0] if results.get("metadatas") and results["metadatas"] else {}
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk from the vector store.
        
        Args:
            chunk_id: Unique ID of the chunk to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[chunk_id])
            logger.debug(f"Deleted chunk {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Union[int, str]]:
        """Get basic statistics about the vector store."""
        count = self.collection.count() if self.collection else 0
        name = self.collection.name if self.collection else "unknown"
        return {
            "total_chunks": count,
            "collection_name": name,
            "embedding_model": self.model_name
        }
    
    def get_collection_stats(self) -> Dict[str, Union[int, str]]:
        """Get collection statistics - alias for get_stats for compatibility."""
        return self.get_stats()
    
    def reset_collection(self) -> bool:
        """
        Reset (clear) the entire collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "ELR embeddings for LUKi agent"}
            )
            logger.info(f"Reset collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False


def create_embedding_store(
    model_name: str = "all-MiniLM-L12-v2",
    persist_directory: str = "./chroma_db",
    collection_name: str = "elr_embeddings"
) -> EmbeddingStore:
    """
    Factory function to create an embedding store.
    
    Args:
        model_name: SentenceTransformer model name
        persist_directory: Directory to persist ChromaDB
        collection_name: Name of the ChromaDB collection
        
    Returns:
        Initialized EmbeddingStore instance
    """
    return EmbeddingStore(model_name, persist_directory, collection_name)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would be used like:
    # store = create_embedding_store()
    # results = store.search_similar("What are my favorite activities?", k=5)
    # print(f"Found {len(results)} similar chunks")
