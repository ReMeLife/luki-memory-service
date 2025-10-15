"""
ELR Store - Dedicated storage for Electronic Life Records (user memories).

This module provides a completely separate storage system for user ELR data,
independent from project knowledge to ensure data isolation and privacy.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json

from .vector_store import create_embedding_store
# Settings will be passed as parameters instead of importing

logger = logging.getLogger(__name__)


class ELRStore:
    """
    Dedicated store for Electronic Life Records (user memories).
    
    This is completely separate from project knowledge to ensure:
    1. User data privacy and isolation
    2. Personalized memory retrieval
    3. Consent-based data management
    4. No contamination with system knowledge
    """
    
    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "user_elr"):
        """
        Initialize the ELR store.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection for ELR data
        """
        self.collection_name = collection_name
        
        # Get settings to read ChromaDB path from environment
        from ..api.config import get_settings
        settings = get_settings()
        
        # Use provided path, or fall back to settings (which reads from CHROMA_PERSIST_DIR env var)
        self.persist_directory = persist_directory or settings.vector_db_path
        self.store = create_embedding_store(
            model_name=settings.embedding_model,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        # Track user data statistics
        self.user_stats = {}
        
        logger.info(f"ELRStore initialized with collection: {collection_name}")
    
    def add_memory(self, user_id: str, content: str, metadata: Dict[str, Any]) -> str:
        """
        Add a user memory to the ELR store.
        
        Args:
            user_id: User identifier
            content: Memory content
            metadata: Memory metadata
            
        Returns:
            The ID of the stored memory chunk
        """
        # Generate unique ID based on user and content
        chunk_id = f"{user_id}_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
        
        # Ensure required metadata
        metadata.update({
            "user_id": user_id,
            "type": "elr_memory",
            "content_type": metadata.get("content_type", "general"),
            "sensitivity_level": metadata.get("sensitivity_level", "low"),
            "created_at": datetime.utcnow().isoformat(),
            "chunk_id": chunk_id,
            "consent_level": metadata.get("consent_level", "private")
        })
        
        # Generate embedding and store directly
        embedding = self.store.generate_embedding(content)
        self.store.collection.add(
            ids=[chunk_id],
            embeddings=[embedding.tolist()],
            documents=[content],
            metadatas=[metadata]
        )
        
        # Update user statistics
        if user_id not in self.user_stats:
            self.user_stats[user_id] = {"memory_count": 0, "last_updated": None}
        self.user_stats[user_id]["memory_count"] += 1
        self.user_stats[user_id]["last_updated"] = datetime.utcnow().isoformat()
        
        return chunk_id
    
    def search_user_memories(
        self, 
        user_id: str, 
        query: str, 
        k: int = 10,
        content_types: Optional[List[str]] = None,
        sensitivity_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search memories for a specific user.
        
        Args:
            user_id: User identifier
            query: Search query
            k: Number of results to return
            content_types: Filter by content types
            sensitivity_filter: Filter by sensitivity levels
            
        Returns:
            List of matching memory chunks
        """
        # Special case: if query is just whitespace, get all user memories
        if not query or query.strip() == "":
            # Get all documents for this user directly from ChromaDB
            try:
                # Use large limit to get ALL memories for listing
                actual_limit = max(k, 1000)  # Ensure we get all memories, not just k
                results = self.store.collection.get(
                    where={"user_id": user_id},
                    limit=actual_limit,
                    include=["documents", "metadatas", "embeddings"]
                )
                
                user_results = []
                if results and results.get("ids"):
                    docs = results.get("documents") or []
                    metas = results.get("metadatas") or []
                    logger.info(f"Direct retrieval found {len(results['ids'])} memories for user {user_id}")
                    for i, chunk_id in enumerate(results["ids"]):
                        user_results.append({
                            "id": chunk_id,
                            "content": docs[i] if i < len(docs) else "",
                            "metadata": metas[i] if i < len(metas) else {},
                            "similarity_score": 1.0,  # Max score for direct retrieval
                            "distance": 0.0
                        })
                else:
                    logger.warning(f"No memories found in ChromaDB for user {user_id}")
                return user_results
            except Exception as e:
                logger.warning(f"Direct retrieval failed for user {user_id}: {e}")
                # Fall back to regular search
        
        # Perform semantic search with user isolation
        all_results = self.store.search_similar(
            query=query,
            k=k * 3,  # Get more results for filtering
            similarity_threshold=0.3
        )
        
        # Filter for user's memories only
        user_results = []
        for result in all_results:
            metadata = result.get("metadata", {})
            
            # User isolation - CRITICAL for privacy
            if metadata.get("user_id") != user_id:
                continue
            
            # Apply content type filter
            if content_types and metadata.get("content_type") not in content_types:
                continue
            
            # Apply sensitivity filter
            if sensitivity_filter and metadata.get("sensitivity_level") not in sensitivity_filter:
                continue
            
            user_results.append(result)
            if len(user_results) >= k:
                break
        
        return user_results
    
    def get_user_context(self, user_id: str, max_items: int = 5) -> Dict[str, Any]:
        """
        Get contextual information about a user from their ELR.
        
        Args:
            user_id: User identifier
            max_items: Maximum number of context items per category
            
        Returns:
            Structured user context
        """
        context = {
            "preferences": [],
            "recent_memories": [],
            "important_people": [],
            "health_notes": [],
            "activities": []
        }
        
        # Search for different types of memories
        categories = {
            "preferences": "user preferences likes dislikes favorite",
            "recent_memories": "recent today yesterday this week",
            "important_people": "family friend spouse child parent sibling",
            "health_notes": "health medical condition medication allergy",
            "activities": "hobby activity interest enjoy fun"
        }
        
        for category, search_query in categories.items():
            results = self.search_user_memories(
                user_id=user_id,
                query=search_query,
                k=max_items
            )
            
            for result in results:
                context[category].append({
                    "content": result.get("content", ""),
                    "relevance": 1.0 - result.get("distance", 0.0),
                    "created_at": result.get("metadata", {}).get("created_at")
                })
        
        return context
    
    def update_memory(self, user_id: str, chunk_id: str, new_content: str, new_metadata: Dict[str, Any]) -> bool:
        """
        Update an existing memory.
        
        Args:
            user_id: User identifier
            chunk_id: ID of the memory to update
            new_content: Updated content
            new_metadata: Updated metadata
            
        Returns:
            True if successful, False otherwise
        """
        # Verify the memory belongs to the user
        if not chunk_id.startswith(f"{user_id}_"):
            logger.warning(f"User {user_id} attempted to update memory {chunk_id} they don't own")
            return False
        
        try:
            # Delete old memory
            self.store.collection.delete(ids=[chunk_id])
            
            # Add updated memory
            new_metadata["updated_at"] = datetime.utcnow().isoformat()
            self.add_memory(user_id, new_content, new_metadata)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update memory {chunk_id}: {e}")
            return False
    
    def delete_user_memories(self, user_id: str) -> int:
        """
        Delete all memories for a user (GDPR compliance).
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of memories deleted
        """
        # Search for all user memories
        all_memories = self.search_user_memories(
            user_id=user_id,
            query="",  # Empty query to get all
            k=10000  # Large number to get all
        )
        
        # Extract chunk IDs
        chunk_ids = [m.get("metadata", {}).get("chunk_id") for m in all_memories]
        chunk_ids = [cid for cid in chunk_ids if cid]  # Filter None values
        
        if chunk_ids:
            self.store.collection.delete(ids=chunk_ids)
            
        # Clear user statistics
        if user_id in self.user_stats:
            del self.user_stats[user_id]
        
        logger.info(f"Deleted {len(chunk_ids)} memories for user {user_id}")
        return len(chunk_ids)
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for a user's ELR data.
        
        Args:
            user_id: User identifier
            
        Returns:
            User statistics
        """
        if user_id not in self.user_stats:
            # Calculate stats if not cached
            memories = self.search_user_memories(user_id, "", k=10000)
            
            content_types = {}
            sensitivity_levels = {}
            
            for memory in memories:
                metadata = memory.get("metadata", {})
                
                # Count content types
                ct = metadata.get("content_type", "unknown")
                content_types[ct] = content_types.get(ct, 0) + 1
                
                # Count sensitivity levels
                sl = metadata.get("sensitivity_level", "unknown")
                sensitivity_levels[sl] = sensitivity_levels.get(sl, 0) + 1
            
            self.user_stats[user_id] = {
                "memory_count": len(memories),
                "content_types": content_types,
                "sensitivity_levels": sensitivity_levels,
                "last_updated": datetime.utcnow().isoformat()
            }
        
        return self.user_stats[user_id]
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Export all user data for portability (GDPR compliance).
        
        Args:
            user_id: User identifier
            
        Returns:
            Complete user data export
        """
        memories = self.search_user_memories(user_id, "", k=10000)
        
        export_data = {
            "user_id": user_id,
            "export_date": datetime.utcnow().isoformat(),
            "total_memories": len(memories),
            "memories": []
        }
        
        for memory in memories:
            export_data["memories"].append({
                "content": memory.get("content"),
                "metadata": memory.get("metadata"),
                "chunk_id": memory.get("metadata", {}).get("chunk_id")
            })
        
        return export_data


# Global instance for singleton pattern
_elr_store: Optional[ELRStore] = None


def get_elr_store() -> ELRStore:
    """Get or create the global ELR store instance."""
    global _elr_store
    if _elr_store is None:
        _elr_store = ELRStore()
    return _elr_store
