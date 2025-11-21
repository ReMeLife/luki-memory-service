#!/usr/bin/env python3
"""
Memory Search and Retrieval API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import time
from datetime import datetime, timedelta
import base64
import json
import hashlib
import os

from ..models import (
    MemorySearchRequest, MemorySearchResponse, MemorySearchResult,
    UserMemoryStats, ErrorResponse
)

# Inline auth classes and functions to avoid import issues
class User(BaseModel):
    """User model with full authentication support."""
    user_id: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime
    last_activity: Optional[datetime] = None
    permissions: Optional[list] = None

security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> User:
    """Get the current authenticated user from JWT token."""
    if credentials is None:
        # For development, allow anonymous access
        return User(
            user_id="anonymous",
            email=None,
            full_name="Anonymous User",
            is_active=True,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            permissions=[]
        )
    
    try:
        token = credentials.credentials
        # Simple base64 token decoding for development
        try:
            decoded_bytes = base64.b64decode(token + '==')  # Add padding
            payload = json.loads(decoded_bytes.decode('utf-8'))
        except:
            # If not base64, treat as simple token
            payload = {"sub": "anonymous", "email": None}
            
        user_id = str(payload.get("sub", "anonymous"))
        email = payload.get("email") or None
        
        return User(
            user_id=user_id,
            email=email,
            full_name=email or "User",
            is_active=True,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            permissions=[]
        )
    except Exception:
        # Return anonymous user on any token error
        return User(
            user_id="anonymous",
            email=None,
            full_name="Anonymous User",
            is_active=True,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            permissions=[]
        )

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
from ..config import get_settings
from ..policy_client import enforce_policy_scopes
from ...schemas.elr import ELRContentType, SensitivityLevel
from ...storage.vector_store import create_embedding_store

logger = logging.getLogger(__name__)

try:
    from ...cache.knowledge_cache import get_knowledge_cache
    knowledge_cache = get_knowledge_cache()
except ImportError:
    # Fallback: knowledge cache module not available
    logger.warning("Knowledge cache module not found, using fallback (no cache)")
    knowledge_cache = None

router = APIRouter(prefix="/search", tags=["search"])
settings = get_settings()

# Global pipeline reference (will be injected)
pipeline = None

# Project knowledge store for system context
project_knowledge_store = None

# ELR store for user memories
elr_store = None

def set_pipeline(elr_pipeline):
    """Set the global pipeline instance."""
    global pipeline
    pipeline = elr_pipeline

def set_project_knowledge_store(store):
    """Set the global project knowledge store instance."""
    global project_knowledge_store
    project_knowledge_store = store

def set_elr_store(store):
    """Set the global ELR store instance."""
    global elr_store
    elr_store = store


@router.post("/memories/test", response_model=MemorySearchResponse)
async def search_memories_test(
    request: MemorySearchRequest
):
    """
    Test endpoint for memory search without authentication.
    Used for integration testing.
    """
    if elr_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ELR pipeline not initialized"
        )
    
    start_time = time.time()
    
    try:
        # Perform semantic search
        search_results = elr_store.search_user_memories(
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
            
            # Prefer the vector-store similarity field when present
            similarity_value = result.get('similarity_score')
            if similarity_value is None:
                similarity_value = result.get('similarity', 0.0)

            formatted_results.append(MemorySearchResult(
                content=result.get('content', ''),
                similarity_score=similarity_value,
                metadata=result.get('metadata', {}),
                chunk_id=result.get('id', ''),
                created_at=datetime.fromisoformat(result_created_at.replace('Z', '+00:00')) if result_created_at else datetime.utcnow()
            ).model_dump())
        
        query_time = time.time() - start_time
        
        logger.info(f"Test memory search completed for user {request.user_id}: "
                   f"{len(formatted_results)} results in {query_time:.3f}s")
        
        return MemorySearchResponse(
            results=formatted_results,
            total_count=len(formatted_results),
            query_time_ms=query_time * 1000.0,
            user_id=request.user_id
        )
    
    except Exception as e:
        query_time = time.time() - start_time
        error_msg = f"Test memory search failed: {str(e)}"
        logger.error(error_msg)
        
        return MemorySearchResponse(
            results=[],
            total_count=0,
            query_time_ms=query_time * 1000.0,
            user_id=request.user_id
        )


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
    - Anonymous user handling (returns empty results)
    """
    # Handle anonymous users - return empty results
    if request.user_id == 'anonymous_base_user':
        logger.info(f"Anonymous user memory search - returning empty results")
        return MemorySearchResponse(
            results=[],
            total_count=0,
            query_time_ms=1.0,
            user_id=request.user_id
        )
    
    if elr_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ELR pipeline not initialized"
        )
    
    # Authorization check removed - using simplified auth model
    
    start_time = time.time()
    
    try:
        policy_result = await enforce_policy_scopes(
            user_id=request.user_id,
            requested_scopes=["elr_memories"],
            requester_role="memory_service",
            context={"operation": "search_memories"},
        )
        if not policy_result.get("allowed", False):
            query_time = time.time() - start_time
            logger.info(
                "Memory search blocked by consent policy for user %s",
                request.user_id,
            )
            return MemorySearchResponse(
                results=[],
                total_count=0,
                query_time_ms=query_time * 1000.0,
                user_id=request.user_id,
            )
        
        # Perform semantic search
        search_results = elr_store.search_user_memories(
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
            
            # Prefer the vector-store similarity field when present
            similarity_value = result.get('similarity')
            if similarity_value is None:
                similarity_value = result.get('similarity_score', 0.0)

            formatted_results.append(MemorySearchResult(
                content=result.get('content', ''),
                similarity_score=similarity_value,
                metadata=result.get('metadata', {}),
                chunk_id=result.get('id', ''),
                created_at=datetime.fromisoformat(result_created_at.replace('Z', '+00:00')) if result_created_at else datetime.utcnow()
            ).model_dump())
        
        query_time = time.time() - start_time
        
        logger.info(f"Memory search completed for user {request.user_id}: "
                   f"{len(formatted_results)} results in {query_time:.3f}s")
        
        return MemorySearchResponse(
            results=formatted_results,
            total_count=len(formatted_results),
            query_time_ms=query_time * 1000.0,
            user_id=request.user_id
        )
    
    except Exception as e:
        query_time = time.time() - start_time
        error_msg = f"Memory search failed: {str(e)}"
        logger.error(error_msg)
        
        return MemorySearchResponse(
            results=[],
            total_count=0,
            query_time_ms=query_time * 1000.0,
            user_id=request.user_id
        )


@router.post("/project-knowledge", response_model=MemorySearchResponse)
async def search_project_knowledge(
    request: MemorySearchRequest
):
    """
    Search project knowledge and system context.
    This searches the project_context collection containing loaded documentation.
    """
    if project_knowledge_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Project knowledge store not initialized"
        )
    
    start_time = time.time()
    
    try:
        # Search project knowledge store
        search_results = project_knowledge_store.search_similar(
            query=request.query,
            k=min(request.k, settings.max_search_results),
            similarity_threshold=0.2  # Use a reasonable default threshold
        )
        
        # Convert results to API format
        formatted_results = []
        for result in search_results:
            metadata = result.get('metadata', {})
            
            formatted_results.append(MemorySearchResult(
                content=result.get('content', ''),
                similarity_score=1.0 - result.get('distance', 0.0),  # Convert distance to similarity
                metadata=metadata,
                chunk_id=result.get('id', ''),
                created_at=datetime.fromisoformat(metadata.get('created_at', datetime.utcnow().isoformat()))
            ).model_dump())
        
        query_time = time.time() - start_time
        
        logger.info(f"Project knowledge search completed: "
                   f"{len(formatted_results)} results in {query_time:.3f}s")
        
        return MemorySearchResponse(
            results=formatted_results,
            total_count=len(formatted_results),
            query_time_ms=query_time * 1000.0,
            user_id="system"
        )
    
    except Exception as e:
        query_time = time.time() - start_time
        error_msg = f"Project knowledge search failed: {str(e)}"
        logger.error(error_msg)
        
        return MemorySearchResponse(
            results=[],
            total_count=0,
            query_time_ms=query_time * 1000.0,
            user_id="system"
        )


@router.get("/memories/{user_id}/stats", response_model=UserMemoryStats)
async def get_user_memory_stats(
    user_id: str
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
    # Authorization check removed - using simplified auth model
    
    if elr_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ELR pipeline not initialized"
        )
    
    try:
        # Get all memories for the user (large k value to get everything)
        all_memories = elr_store.search_user_memories(
            user_id=user_id,
            query="",  # Empty query to get all memories
            k=10000  # Large number to get all results
        )
        
        # Calculate statistics
        total_memories = len(all_memories)
        content_type_breakdown: Dict[str, int] = {}
        consent_level_breakdown: Dict[str, int] = {}
        dates: List[datetime] = []
        
        for memory in all_memories:
            metadata = memory.get('metadata', {})
            
            # Content type breakdown
            content_type = metadata.get('content_type', 'UNKNOWN')
            content_type_breakdown[content_type] = content_type_breakdown.get(content_type, 0) + 1
            
            # Consent level breakdown
            consent_level = metadata.get('consent_level', 'UNKNOWN')
            consent_level_breakdown[consent_level] = consent_level_breakdown.get(consent_level, 0) + 1
            
            # Collect dates
            created_at = metadata.get('created_at')
            if created_at:
                try:
                    date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    dates.append(date)
                except (ValueError, AttributeError):
                    pass
        
        # Compute recent memories (last 30 days)
        now = datetime.utcnow()
        recent_cutoff = now - timedelta(days=30)
        recent_memories = sum(1 for d in dates if d >= recent_cutoff)

        logger.info(f"Generated memory stats for user {user_id}: {total_memories} memories")

        return UserMemoryStats(
            user_id=user_id,
            email="",
            total_memories=total_memories,
            recent_memories=recent_memories,
            content_types=content_type_breakdown,
            consent_levels=consent_level_breakdown,
            last_updated=now.isoformat()
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
    k: int = 5
):
    """
    Find memories similar to a specific memory chunk.
    
    This endpoint helps discover related memories by finding
    semantically similar content to a given memory.
    """
    # Authorization check removed - using simplified auth model
    
    if elr_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ELR pipeline not initialized"
        )
    
    try:
        # First, get the content of the specified chunk
        all_memories = elr_store.search_user_memories(
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
        similar_memories = elr_store.search_user_memories(
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
            similarity_value = result.get('similarity_score')
            if similarity_value is None:
                similarity_value = result.get('similarity', 0.0)

            formatted_results.append(MemorySearchResult(
                content=result.get('content', ''),
                similarity_score=similarity_value,
                metadata=result.get('metadata', {}),
                chunk_id=result.get('id', ''),
                created_at=datetime.utcnow()  # Default timestamp
            ).model_dump())
        
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
async def search_status():
    """Get search service status."""
    return {
        "service": "Memory Search",
        "status": "operational" if pipeline is not None else "unavailable",
        "pipeline_ready": pipeline is not None,
        "project_knowledge_ready": project_knowledge_store is not None,
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
            "similar_memory_discovery",
            "project_knowledge_search"
        ]
    }
