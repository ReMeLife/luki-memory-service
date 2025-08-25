#!/usr/bin/env python3
"""
Memory Search and Retrieval API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional, Dict, Any
import logging
import time
from datetime import datetime

from ..models import (
    MemorySearchRequest, MemorySearchResponse, MemorySearchResult,
    UserMemoryStats, ErrorResponse
)
from ..auth import get_current_active_user, User
from ..config import get_settings
from ...schemas.elr import ELRContentType, SensitivityLevel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])
settings = get_settings()

# Global pipeline reference (will be injected)
pipeline = None

def set_pipeline(elr_pipeline):
    """Set the global pipeline instance."""
    global pipeline
    pipeline = elr_pipeline


@router.post("/memories", response_model=MemorySearchResponse)
async def search_memories(
    request: MemorySearchRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Search user memories using semantic similarity.
    
    This endpoint provides intelligent memory retrieval:
    - Semantic search using vector embeddings
    - Content type and sensitivity filtering
    - Date range filtering
    - User data isolation
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ELR pipeline not initialized"
        )
    
    # Validate user authorization
    if current_user.user_id != request.user_id and current_user.user_id != "api_service":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to search memories for this user"
        )
    
    start_time = time.time()
    
    try:
        # Perform semantic search
        search_results = pipeline.search_user_memories(
            user_id=request.user_id,
            query=request.query,
            k=min(request.k, settings.max_search_results)
        )
        
        # Convert results to API format
        formatted_results = []
        for result in search_results:
            # Apply content type filtering if specified
            if request.content_types:
                result_content_type = result.get('metadata', {}).get('content_type')
                if result_content_type and result_content_type not in [ct.value for ct in request.content_types]:
                    continue
            
            # Apply sensitivity filtering if specified
            if request.sensitivity_filter:
                result_sensitivity = result.get('metadata', {}).get('sensitivity_level')
                if result_sensitivity and result_sensitivity not in [sl.value for sl in request.sensitivity_filter]:
                    continue
            
            # Apply date filtering if specified
            result_created_at = result.get('metadata', {}).get('created_at')
            if result_created_at:
                try:
                    result_date = datetime.fromisoformat(result_created_at.replace('Z', '+00:00'))
                    if request.date_from and result_date < request.date_from:
                        continue
                    if request.date_to and result_date > request.date_to:
                        continue
                except (ValueError, AttributeError):
                    pass  # Skip date filtering if date parsing fails
            
            formatted_results.append(MemorySearchResult(
                content=result.get('content', ''),
                similarity_score=result.get('similarity_score', 0.0),
                metadata=result.get('metadata', {}),
                chunk_id=result.get('id', ''),
                created_at=datetime.fromisoformat(result_created_at.replace('Z', '+00:00')) if result_created_at else datetime.utcnow()
            ))
        
        query_time = time.time() - start_time
        
        logger.info(f"Memory search completed for user {request.user_id}: "
                   f"{len(formatted_results)} results in {query_time:.3f}s")
        
        return MemorySearchResponse(
            success=True,
            results=formatted_results,
            total_results=len(formatted_results),
            query_time_seconds=query_time,
            user_id=request.user_id
        )
    
    except Exception as e:
        query_time = time.time() - start_time
        error_msg = f"Memory search failed: {str(e)}"
        logger.error(error_msg)
        
        return MemorySearchResponse(
            success=False,
            results=[],
            total_results=0,
            query_time_seconds=query_time,
            user_id=request.user_id
        )


@router.get("/memories/{user_id}/stats", response_model=UserMemoryStats)
async def get_user_memory_stats(
    user_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get comprehensive statistics about a user's stored memories.
    
    Returns detailed breakdown of:
    - Total memories and chunks
    - Content type distribution
    - Sensitivity level distribution
    - Date ranges
    - Storage usage
    """
    # Validate user authorization
    if current_user.user_id != user_id and current_user.user_id != "api_service":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access stats for this user"
        )
    
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ELR pipeline not initialized"
        )
    
    try:
        # Get all memories for the user (large k value to get everything)
        all_memories = pipeline.search_user_memories(
            user_id=user_id,
            query="",  # Empty query to get all memories
            k=10000  # Large number to get all results
        )
        
        # Calculate statistics
        total_memories = len(all_memories)
        content_type_breakdown = {}
        sensitivity_breakdown = {}
        dates = []
        
        for memory in all_memories:
            metadata = memory.get('metadata', {})
            
            # Content type breakdown
            content_type = metadata.get('content_type', 'UNKNOWN')
            content_type_breakdown[content_type] = content_type_breakdown.get(content_type, 0) + 1
            
            # Sensitivity breakdown
            sensitivity = metadata.get('sensitivity_level', 'UNKNOWN')
            sensitivity_breakdown[sensitivity] = sensitivity_breakdown.get(sensitivity, 0) + 1
            
            # Collect dates
            created_at = metadata.get('created_at')
            if created_at:
                try:
                    date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    dates.append(date)
                except (ValueError, AttributeError):
                    pass
        
        # Calculate date range
        earliest_memory = min(dates) if dates else None
        latest_memory = max(dates) if dates else None
        
        # Estimate storage size (rough approximation)
        total_content_length = sum(len(memory.get('content', '')) for memory in all_memories)
        storage_size_mb = total_content_length / (1024 * 1024)  # Convert to MB
        
        logger.info(f"Generated memory stats for user {user_id}: {total_memories} memories")
        
        return UserMemoryStats(
            user_id=user_id,
            total_memories=total_memories,
            total_chunks=total_memories,  # In our current implementation, 1 memory = 1 chunk
            content_type_breakdown=content_type_breakdown,
            sensitivity_breakdown=sensitivity_breakdown,
            earliest_memory=earliest_memory,
            latest_memory=latest_memory,
            storage_size_mb=storage_size_mb
        )
    
    except Exception as e:
        error_msg = f"Failed to get memory stats: {str(e)}"
        logger.error(error_msg)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )


@router.get("/memories/{user_id}/similar/{chunk_id}")
async def find_similar_memories(
    user_id: str,
    chunk_id: str,
    k: int = 5,
    current_user: User = Depends(get_current_active_user)
):
    """
    Find memories similar to a specific memory chunk.
    
    This endpoint helps discover related memories by finding
    semantically similar content to a given memory.
    """
    # Validate user authorization
    if current_user.user_id != user_id and current_user.user_id != "api_service":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access memories for this user"
        )
    
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ELR pipeline not initialized"
        )
    
    try:
        # First, get the content of the specified chunk
        all_memories = pipeline.search_user_memories(
            user_id=user_id,
            query="",
            k=10000
        )
        
        target_memory = None
        for memory in all_memories:
            if memory.get('id') == chunk_id:
                target_memory = memory
                break
        
        if not target_memory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory chunk not found"
            )
        
        # Use the content of the target memory as the search query
        target_content = target_memory.get('content', '')
        
        # Search for similar memories
        similar_memories = pipeline.search_user_memories(
            user_id=user_id,
            query=target_content,
            k=k + 1  # +1 to account for the original memory
        )
        
        # Filter out the original memory from results
        filtered_results = [
            memory for memory in similar_memories 
            if memory.get('id') != chunk_id
        ][:k]
        
        # Format results
        formatted_results = []
        for result in filtered_results:
            formatted_results.append(MemorySearchResult(
                content=result.get('content', ''),
                similarity_score=result.get('similarity_score', 0.0),
                metadata=result.get('metadata', {}),
                chunk_id=result.get('id', ''),
                created_at=datetime.utcnow()  # Default timestamp
            ))
        
        logger.info(f"Found {len(formatted_results)} similar memories for chunk {chunk_id}")
        
        return {
            "success": True,
            "target_chunk_id": chunk_id,
            "similar_memories": formatted_results,
            "total_found": len(formatted_results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to find similar memories: {str(e)}"
        logger.error(error_msg)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )


@router.get("/status")
async def search_status(
    current_user: User = Depends(get_current_active_user)
):
    """Get search service status."""
    return {
        "service": "Memory Search",
        "status": "operational" if pipeline is not None else "unavailable",
        "pipeline_ready": pipeline is not None,
        "max_search_results": settings.max_search_results,
        "supported_filters": [
            "content_types",
            "sensitivity_levels", 
            "date_range"
        ],
        "search_capabilities": [
            "semantic_similarity",
            "metadata_filtering",
            "user_isolation",
            "similar_memory_discovery"
        ]
    }
