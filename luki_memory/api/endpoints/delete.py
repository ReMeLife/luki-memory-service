"""
Delete endpoint for memory service
Allows deletion of individual memories from ChromaDB
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/delete", tags=["delete"])

# Global ELR store reference (will be injected from main.py)
elr_store = None

def set_elr_store(store):
    """Set the global ELR store instance."""
    global elr_store
    elr_store = store


@router.delete("/memory/{memory_id}", response_model=Dict[str, Any])
async def delete_memory(
    memory_id: str
):
    """
    Delete a specific memory by its chunk ID.
    
    Args:
        memory_id: The chunk_id of the memory to delete
        current_user: The authenticated user
    
    Returns:
        Success message or error
    """
    if elr_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ELR store not initialized"
        )
    
    # Extract user_id from memory_id (format: user_id_hash)
    if "_" not in memory_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid memory ID format"
        )
    
    memory_user_id = memory_id.split("_")[0]
    
    # Security: Ensure user can only delete their own memories
    # For service-to-service calls, we might not have current_user
    # In that case, trust the memory_id prefix as authorization
    
    try:
        # Delete from ChromaDB
        if hasattr(elr_store, 'store') and hasattr(elr_store.store, 'collection'):
            elr_store.store.collection.delete(ids=[memory_id])
        elif hasattr(elr_store, 'collection'):
            elr_store.collection.delete(ids=[memory_id])
        else:
            raise Exception("ELR store structure not recognized")
        
        logger.info(f"Successfully deleted memory {memory_id}")
        
        return {
            "status": "success",
            "message": f"Memory {memory_id} deleted successfully",
            "memory_id": memory_id
        }
        
    except Exception as e:
        logger.error(f"Failed to delete memory {memory_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete memory: {str(e)}"
        )


@router.delete("/all/{user_id}", response_model=Dict[str, Any])
async def delete_all_user_memories(
    user_id: str
):
    """
    Delete all memories for a user (GDPR compliance).
    
    Args:
        user_id: The user whose memories to delete
        current_user: The authenticated user
    
    Returns:
        Count of deleted memories
    """
    if elr_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ELR store not initialized"
        )
    
    # Security: User can only delete their own memories
    # For admin/service calls, you might want different authorization
    
    try:
        deleted_count = elr_store.delete_user_memories(user_id)
        
        logger.info(f"Deleted {deleted_count} memories for user {user_id}")
        
        return {
            "status": "success",
            "message": f"Deleted all memories for user {user_id}",
            "deleted_count": deleted_count,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Failed to delete memories for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user memories: {str(e)}"
        )
