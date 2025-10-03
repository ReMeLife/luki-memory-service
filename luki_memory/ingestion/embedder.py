"""
Embedding Calls Module
Generate embeddings for text chunks using SentenceTransformers.

Responsibilities:
- Generate embeddings for individual texts
- Handle batch embedding generation
- Manage embedding model lifecycle
- Provide embedding utilities
"""

import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding errors."""
    pass


class TextEmbedder:
    """Handles text embedding generation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        """
        Initialize text embedder.
        
        Args:
            model_name: SentenceTransformer model name
        """
        self.model_name = model_name
        
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model {model_name}: {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}")
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            Embedding matrix as numpy array
        """
        try:
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10
            )
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}")


def create_embedder(model_name: str = "all-MiniLM-L12-v2") -> TextEmbedder:
    """
    Factory function to create a text embedder.
    
    Args:
        model_name: SentenceTransformer model name
        
    Returns:
        Initialized TextEmbedder instance
    """
    return TextEmbedder(model_name)
