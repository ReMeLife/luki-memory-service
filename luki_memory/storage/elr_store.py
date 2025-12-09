"""
ELR Store - Dedicated storage for Electronic Life Records (user memories).

This module provides a completely separate storage system for user ELR data,
independent from project knowledge to ensure data isolation and privacy.
"""

import logging
import os
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .vector_store import create_embedding_store
# Settings will be passed as parameters instead of importing

logger = logging.getLogger(__name__)


_ENCRYPTION_KEY: Optional[bytes] = None


def _get_encryption_key() -> bytes:
    global _ENCRYPTION_KEY
    if _ENCRYPTION_KEY is not None:
        return _ENCRYPTION_KEY
    key_str = os.getenv("ENCRYPTION_KEY")
    if not key_str:
        _ENCRYPTION_KEY = AESGCM.generate_key(bit_length=256)
        logger.warning("ENCRYPTION_KEY not set; using ephemeral in-memory encryption key")
        return _ENCRYPTION_KEY
    key_bytes: Optional[bytes] = None
    try:
        candidate = bytes.fromhex(key_str)
        if len(candidate) == 32:
            key_bytes = candidate
    except ValueError:
        key_bytes = None
    if key_bytes is None:
        try:
            candidate = base64.b64decode(key_str)
            if len(candidate) == 32:
                key_bytes = candidate
        except Exception:
            key_bytes = None
    if key_bytes is None:
        key_bytes = hashlib.sha256(key_str.encode("utf-8")).digest()
        logger.info("Derived encryption key from ENCRYPTION_KEY value using SHA-256")
    _ENCRYPTION_KEY = key_bytes
    return _ENCRYPTION_KEY


def _encrypt_text(plaintext: str) -> str:
    if not plaintext:
        return plaintext
    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def _decrypt_text(value: str) -> str:
    if not value:
        return value
    key = _get_encryption_key()
    try:
        raw = base64.b64decode(value.encode("ascii"))
        if len(raw) <= 12:
            return value
        nonce = raw[:12]
        ciphertext = raw[12:]
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")
    except Exception as e:
        logger.warning(
            "Failed to decrypt ELR memory content; returning stored value",
            extra={"error": str(e)},
        )
        return value


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
        encrypted_content = _encrypt_text(content)
        self.store.collection.add(
            ids=[chunk_id],
            embeddings=[embedding.tolist()],
            documents=[encrypted_content],
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
        # Special case: if query is empty or just whitespace, do a fast
        # direct retrieval of this user's memories without semantic search.
        if not query or query.strip() == "":
            try:
                # Respect the requested limit k and avoid fetching embeddings,
                # since listing memories only needs content + metadata.
                actual_limit = max(1, k or 10)
                results = self.store.collection.get(
                    where={"user_id": user_id},
                    limit=actual_limit,
                    include=["documents", "metadatas"],
                )

                user_results: List[Dict[str, Any]] = []
                if results and results.get("ids"):
                    docs = results.get("documents") or []
                    metas = results.get("metadatas") or []
                    logger.info(
                        "Direct retrieval found %d memories for user %s (limit=%d)",
                        len(results["ids"]),
                        user_id,
                        actual_limit,
                    )
                    for i, chunk_id in enumerate(results["ids"]):
                        content_value = docs[i] if i < len(docs) else ""
                        content_value = _decrypt_text(content_value)
                        user_results.append(
                            {
                                "id": chunk_id,
                                "content": content_value,
                                "metadata": metas[i] if i < len(metas) else {},
                                # For listing, treat all direct results as max similarity
                                # so downstream consumers can rely on this field.
                                "similarity_score": 1.0,
                                "distance": 0.0,
                            }
                        )
                else:
                    logger.info("No memories found in ChromaDB for user %s", user_id)
                return user_results
            except Exception as e:
                logger.warning(f"Direct retrieval failed for user {user_id}: {e}")
                # Fall back to regular semantic search if direct get fails
        
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
            
            content_value = result.get("content", "")
            result["content"] = _decrypt_text(content_value)
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
    
    async def update_memory_metadata(
        self, 
        memory_id: str, 
        user_id: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for an existing memory.
        
        Args:
            memory_id: The memory ID to update
            user_id: User ID for verification
            metadata: New metadata to set (replaces existing)
            
        Returns:
            True if successful, False if memory not found
        """
        try:
            # Get existing memory to verify ownership
            result = self.store.collection.get(
                ids=[memory_id],
                include=["metadatas", "documents"]
            )
            
            if not result["ids"]:
                logger.warning(f"Memory {memory_id} not found for update")
                return False
            
            existing_metadata = result["metadatas"][0] if result["metadatas"] else {}
            
            # Verify user ownership
            if existing_metadata.get("user_id") != user_id:
                logger.warning(f"User {user_id} does not own memory {memory_id}")
                return False
            
            # Merge new metadata with existing, preserving user_id and other system fields
            updated_metadata = {
                **existing_metadata,
                **metadata,
                "user_id": user_id,  # Ensure user_id is preserved
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Update in ChromaDB
            self.store.collection.update(
                ids=[memory_id],
                metadatas=[updated_metadata]
            )
            
            logger.info(f"Updated metadata for memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory metadata: {e}")
            return False
    
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
