"""
Project Knowledge Store - Dedicated storage for system context documents.

This module provides a completely separate storage system for project knowledge,
independent from the ELR (Electronic Life Records) pipeline.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib
import uuid

from .vector_store import create_embedding_store

logger = logging.getLogger(__name__)


class ProjectKnowledgeStore:
    """
    Dedicated store for project knowledge and system context.
    
    This is completely separate from ELR user memories to ensure:
    1. Project facts are immutable and authoritative
    2. No mixing of user data with system knowledge
    3. Optimized retrieval for ecosystem queries
    4. Systematic perfection in knowledge management
    """
    
    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "project_knowledge"):
        """
        Initialize the project knowledge store.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection for project knowledge
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "./chroma_db"
        
        # Create dedicated embedding store for project knowledge
        from ..api.config import get_settings
        settings = get_settings()
        self.store = create_embedding_store(
            model_name=settings.embedding_model,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        # Track loaded documents to prevent duplicates
        self.loaded_documents = set()
        
        logger.info(f"ProjectKnowledgeStore initialized with collection: {collection_name}")
    
    def add_chunk(self, elr_chunk) -> str:
        """
        Add an ELR chunk to the project knowledge store.
        
        Args:
            elr_chunk: ELR chunk object with text and metadata attributes
            
        Returns:
            The ID of the stored chunk
        """
        # Handle ELRChunk object (not dictionary)
        content = elr_chunk.text if hasattr(elr_chunk, 'text') else str(elr_chunk)
        
        # Build metadata from ELRChunk attributes
        metadata = {}
        if hasattr(elr_chunk, 'metadata'):
            metadata = elr_chunk.metadata
        if hasattr(elr_chunk, 'user_id'):
            metadata['user_id'] = elr_chunk.user_id
        if hasattr(elr_chunk, 'content_type'):
            metadata['content_type'] = str(elr_chunk.content_type)
        if hasattr(elr_chunk, 'chunk_id'):
            metadata['chunk_id'] = elr_chunk.chunk_id
            
        return self.add_knowledge(content, metadata)
    
    def add_knowledge(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Add a piece of project knowledge to the store.
        
        Args:
            content: The knowledge content
            metadata: Metadata about the knowledge
            
        Returns:
            The ID of the stored knowledge chunk
        """
        # Generate unique ID based on content
        chunk_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Ensure required metadata
        metadata.update({
            "source": metadata.get("source", "project_knowledge"),
            "type": "project_context",
            "priority": metadata.get("priority", "high"),
            "immutable": True,
            "created_at": datetime.utcnow().isoformat(),
            "chunk_id": chunk_id
        })
        
        # Generate embedding
        embedding = self.store.generate_embedding(content)
        
        # Store directly in ChromaDB collection
        self.store.collection.add(
            ids=[chunk_id],
            embeddings=[embedding.tolist()],
            documents=[content],
            metadatas=[metadata]
        )
        
        return chunk_id
    
    def search(self, query: str, k: int = 10, priority_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search project knowledge with priority filtering.
        
        Args:
            query: Search query
            k: Number of results to return
            priority_filter: Filter by priority (critical, high, medium, low)
            
        Returns:
            List of matching knowledge chunks
        """
        # Perform semantic search
        results = self.store.search_similar(
            query=query,
            k=k * 2 if priority_filter else k,  # Get more results if filtering
            similarity_threshold=0.1  # Lower threshold for project knowledge
        )
        
        # Apply priority filtering if specified
        if priority_filter:
            filtered_results = []
            for result in results:
                if result.get("metadata", {}).get("priority") == priority_filter:
                    filtered_results.append(result)
                    if len(filtered_results) >= k:
                        break
            results = filtered_results
        
        # Sort by similarity and priority
        def sort_key(result):
            priority_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            priority = result.get("metadata", {}).get("priority", "medium")
            priority_score = priority_weights.get(priority, 2)
            similarity = 1.0 - result.get("distance", 0.0)
            return (priority_score * similarity, similarity)
        
        results.sort(key=sort_key, reverse=True)
        
        return results[:k]
    
    def get_authoritative_facts(self, topics: List[str]) -> Dict[str, str]:
        """
        Get authoritative facts about specific topics.
        
        Args:
            topics: List of topics to get facts about
            
        Returns:
            Dictionary of topic -> authoritative fact
        """
        facts = {}
        
        # Predefined authoritative facts (hardcoded for absolute certainty)
        authoritative_facts = {
            "founder": "Simon Hooper is the Founder and CEO of ReMeLife",
            "team": "Core team: Mike Anderson (Technical Lead), Orbit (Software Engineer), Oliver (SEO/Community), Johnny (Design/Art), Asif (Backend)",
            "luki_token": "$LUKI is a Solana SPL token with 1B total supply that powers AI Avatar interactions",
            "elr": "ELR stands for Electronic Life Records - comprehensive personal profiles including preferences, habits, and memories",
            "caps": "CAPs are Care Action Points - unlimited, non-transferable points earned for care actions",
            "reme": "REME tokens are tradeable utility tokens with 1B fixed supply for premium services",
            "database": "PostgreSQL and MongoDB (NOT MySQL)",
            "ai_model": "OpenAI GPT-OSS 120B via Together AI (NOT LLaMA)",
            "blockchain": "ReMeGrid uses Convex Lattice with Proof-of-Authority consensus"
        }
        
        for topic in topics:
            topic_lower = topic.lower()
            for key, fact in authoritative_facts.items():
                if key in topic_lower or topic_lower in key:
                    facts[topic] = fact
                    break
        
        return facts
    
    def load_document(self, file_path: str, priority: str = "medium") -> int:
        """
        Load a document into the project knowledge store.
        
        Args:
            file_path: Path to the document
            priority: Priority level (critical, high, medium, low)
            
        Returns:
            Number of chunks created
        """
        # Check if already loaded
        doc_hash = hashlib.sha256(file_path.encode()).hexdigest()
        if doc_hash in self.loaded_documents:
            logger.info(f"Document already loaded: {file_path}")
            return 0
        
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Document not found: {file_path}")
                return 0
            
            content = path.read_text(encoding='utf-8')
            
            # Chunk the document
            chunks = self._chunk_document(content, chunk_size=1000, overlap=200)
            
            # Add each chunk to the store
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": str(file_path),
                    "priority": priority,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "document_type": "context_document"
                }
                self.add_knowledge(chunk, metadata)
            
            self.loaded_documents.add(doc_hash)
            logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            return 0
    
    def _chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Chunk a document into smaller pieces with overlap.
        
        Args:
            text: Document text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to find a good break point
            if end < len(text):
                # Look for sentence end
                for sep in ['. ', '.\n', '! ', '? ', '\n\n']:
                    pos = text.rfind(sep, start, end)
                    if pos > start:
                        end = pos + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def clear(self):
        """Clear all project knowledge from the store."""
        logger.warning("Clearing all project knowledge from store")
        # This would need to be implemented in the vector store
        self.loaded_documents.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the project knowledge store."""
        return {
            "collection_name": self.collection_name,
            "loaded_documents": len(self.loaded_documents),
            "persist_directory": self.persist_directory,
            "status": "operational"
        }


# Global instance for singleton pattern
_project_knowledge_store: Optional[ProjectKnowledgeStore] = None


def get_project_knowledge_store() -> ProjectKnowledgeStore:
    """Get or create the global project knowledge store instance."""
    global _project_knowledge_store
    if _project_knowledge_store is None:
        _project_knowledge_store = ProjectKnowledgeStore()
    return _project_knowledge_store


def initialize_project_knowledge():
    """Initialize project knowledge with core documents."""
    store = get_project_knowledge_store()
    
    # Core identity - CRITICAL priority
    store.add_knowledge(
        content="""
        AUTHORITATIVE FACTS - NEVER DEVIATE FROM THESE:
        
        FOUNDER: Simon Hooper is the Founder and CEO of ReMeLife and creator of LUKi
        TEAM: Mike Anderson (Technical Lead), Orbit (Software Engineer), Oliver (SEO/Community), Johnny (Design/Art), Asif (Backend)
        
        $LUKI TOKEN: EXISTS on Solana blockchain with 1B total supply. Powers AI Avatar interactions, rewards engagement, grants NFT access.
        
        ELR: Electronic Life Records (ONLY this definition) - comprehensive personal profiles
        CAPs: Care Action Points (NOT "Care Points") - unlimited, non-transferable
        REME: Tradeable utility tokens with 1B fixed supply
        
        TECHNOLOGY: OpenAI GPT-OSS 120B via Together AI, PostgreSQL, MongoDB, ChromaDB
        """,
        metadata={
            "source": "authoritative_facts",
            "priority": "critical",
            "category": "core_identity"
        }
    )
    
    logger.info("Project knowledge store initialized with core facts")
