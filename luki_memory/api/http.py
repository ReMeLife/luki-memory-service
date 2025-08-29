"""
FastAPI HTTP routes for LUKi Memory Service.

Provides REST API endpoints for ELR ingestion, vector search, and memory management.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Query, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from ..config import MemoryServiceConfig, load_config
from ..schemas.elr import ELRItem, ELRChunk, ELRSearchFilter, ELRBatchOperation, ELRProcessingResult
from ..schemas.query import SearchQuery, SearchResponse, BatchSearchQuery, BatchSearchResponse
from ..schemas.kv import UserPreferences, SessionData
from ..ingestion.pipeline import ELRPipeline
from ..ingestion.embedding_integration import (
    ELRToVectorPipeline,
    create_elr_to_vector_pipeline,
)
from ..storage.vector_store import EmbeddingStore
from ..storage.kv_store import create_kv_store, UserPreferencesStore, SessionStore
from ..storage.session_store import ShortTermMemoryStore, ContextBuilder
from ..auth.rbac import RBACManager
from ..auth.consent import ConsentManager, ConsentAction
from ..audit.logger import AuditLogger

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LUKi Memory Service",
    description="Memory and embedding service for LUKi AI assistant",
    version="1.0.0"
)

# Configuration
config = load_config()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins or [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global dependencies (would be initialized properly in production)
embedding_store: Optional[EmbeddingStore] = None
elr_pipeline: Optional[ELRPipeline] = None
elr_to_vector_pipeline: Optional[ELRToVectorPipeline] = None
kv_store = None
session_store: Optional[SessionStore] = None
memory_store: Optional[ShortTermMemoryStore] = None
context_builder: Optional[ContextBuilder] = None
rbac_manager: Optional[RBACManager] = None
consent_manager: Optional[ConsentManager] = None
audit_logger: Optional[AuditLogger] = None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract user ID from authentication token."""
    # In production, this would validate JWT token and extract user ID
    # For now, return a placeholder
    return "user_123"


async def verify_consent(user_id: str, operation: str, resource_id: Optional[str] = None):
    """Verify user consent for operations."""
    if consent_manager:
        # Convert string to ConsentAction enum
        try:
            action_enum = ConsentAction(operation)
        except ValueError:
            action_enum = ConsentAction.DATA_PROCESSING  # Default fallback
        
        # Convert resource_id to ConsentLevel if it's a valid enum value
        consent_level = None
        if resource_id:
            try:
                from ..schemas.elr import ConsentLevel
                consent_level = ConsentLevel(resource_id)
            except ValueError:
                # If resource_id is not a valid ConsentLevel, use None
                consent_level = None
        
        has_consent = await consent_manager.check_consent(user_id, action_enum, consent_level)
        if not has_consent:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient consent for this operation"
            )


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global embedding_store, elr_pipeline, elr_to_vector_pipeline, kv_store, session_store, memory_store, context_builder
    global rbac_manager, consent_manager, audit_logger
    
    try:
        # Initialize stores
        embedding_store = EmbeddingStore()
        kv_store = create_kv_store(config.database)
        session_store = SessionStore(kv_store)
        memory_store = ShortTermMemoryStore(kv_store)
        context_builder = ContextBuilder(memory_store)
        
        # Initialize pipeline
        elr_pipeline = ELRPipeline()
        elr_to_vector_pipeline = create_elr_to_vector_pipeline()
        # Share the same EmbeddingStore instance between API and pipeline
        elr_to_vector_pipeline.embedding_store = embedding_store
        
        # Initialize auth and audit (placeholders for now)
        # rbac_manager = RBACManager()
        # consent_manager = ConsentManager()
        # audit_logger = AuditLogger()
        
        logger.info("Memory service started successfully")
    except Exception as e:
        logger.error(f"Failed to start memory service: {e}")
        raise


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# ELR Ingestion Endpoints
@app.post("/elr/ingest", response_model=ELRProcessingResult)
async def ingest_elr_file(
    file_path: str = Body(..., description="Path to ELR JSON file"),
    user_id: str = Depends(get_current_user)
):
    """Ingest an ELR file and process it into chunks."""
    try:
        await verify_consent(user_id, "ingest_elr")
        
        if not elr_to_vector_pipeline or not embedding_store:
            raise HTTPException(status_code=500, detail="Embedding pipeline not initialized")
        
        # Load ELR data then process to embeddings (stores chunks with user_id)
        elr_data = elr_to_vector_pipeline.elr_pipeline.load_elr_file(file_path)
        integration_result = elr_to_vector_pipeline.process_elr_to_embeddings(
            elr_data=elr_data,
            user_id=user_id,
            source_file=file_path,
        )
        
        # Map to ELRProcessingResult for response compatibility
        result = ELRProcessingResult(
            success=integration_result.success,
            processed_items=integration_result.processed_items,
            failed_items=integration_result.failed_embeddings,
            chunks_created=integration_result.embedded_chunks,
            errors=integration_result.errors,
            processing_time_seconds=integration_result.processing_time_seconds,
            created_item_ids=integration_result.created_item_ids,
        )
        
        if audit_logger:
            await audit_logger.log_operation(
                user_id=user_id,
                operation="elr_ingest",
                resource_id=file_path,
                success=result.success
            )
        
        return result
    except Exception as e:
        logger.error(f"Error ingesting ELR file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/elr/ingest-data", response_model=ELRProcessingResult)
async def ingest_elr_data(
    elr_data: Dict[str, Any] = Body(..., description="ELR data object"),
    source_file: str = Body("", description="Source file identifier"),
    user_id: str = Depends(get_current_user)
):
    """Ingest ELR data directly."""
    try:
        await verify_consent(user_id, "ingest_elr")
        
        if not elr_to_vector_pipeline or not embedding_store:
            raise HTTPException(status_code=500, detail="Embedding pipeline not initialized")
        
        integration_result = elr_to_vector_pipeline.process_elr_to_embeddings(
            elr_data=elr_data,
            user_id=user_id,
            source_file=source_file,
        )
        
        # Map to ELRProcessingResult for response compatibility
        result = ELRProcessingResult(
            success=integration_result.success,
            processed_items=integration_result.processed_items,
            failed_items=integration_result.failed_embeddings,
            chunks_created=integration_result.embedded_chunks,
            errors=integration_result.errors,
            processing_time_seconds=integration_result.processing_time_seconds,
            created_item_ids=integration_result.created_item_ids,
        )
        
        if audit_logger:
            await audit_logger.log_operation(
                user_id=user_id,
                operation="elr_ingest_data",
                resource_id=source_file,
                success=result.success
            )
        
        return result
    except Exception as e:
        logger.error(f"Error ingesting ELR data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Search Endpoints
@app.post("/search", response_model=SearchResponse)
async def search_memories(
    query: SearchQuery,
    user_id: str = Depends(get_current_user)
):
    """Search for memories using similarity or keyword search."""
    try:
        await verify_consent(user_id, "search_memories")
        
        if not embedding_store:
            raise HTTPException(status_code=500, detail="Embedding store not initialized")
        
        # Ensure user can only search their own data
        if not query.user_id:
            query.user_id = user_id
        elif query.user_id != user_id:
            raise HTTPException(status_code=403, detail="Cannot search other users' data")
        
        # Perform search
        metadata_filter = {"user_id": user_id}
        if query.content_types:
            metadata_filter.update({
                "content_type": {"$in": [ct.value for ct in query.content_types]}
            })
        results = embedding_store.search_similar(
            query=query.query_text,
            k=query.limit,
            similarity_threshold=query.similarity_threshold,
            consent_filter=[level.value for level in query.consent_levels] if query.consent_levels else None,
            metadata_filter=metadata_filter
        )
        
        # Convert to SearchResponse format
        from ..schemas.query import SearchResult
        search_results = []
        for i, result in enumerate(results):
            from ..schemas.elr import ELRContentType, SensitivityLevel, ConsentLevel
            metadata = result.get("metadata", {})
            
            search_result = SearchResult(
                item_id=result.get("id", ""),
                chunk_id=metadata.get("chunk_id"),
                content=result.get("content", ""),
                title=metadata.get("title"),
                relevance_score=1.0 - result.get("distance", 0.0),  # Convert distance to similarity
                rank=i + 1,
                content_type=ELRContentType(metadata.get("content_type", "memory")),
                consent_level=ConsentLevel(metadata.get("consent_level", "private")),
                sensitivity_level=SensitivityLevel(metadata.get("sensitivity_level", "personal")),
                created_at=datetime.fromisoformat(metadata.get("created_at", datetime.utcnow().isoformat())),
                event_date=datetime.fromisoformat(metadata.get("event_date")) if metadata.get("event_date") else None,
                metadata=metadata,
                highlighted_content=metadata.get("highlighted_content"),
                matched_terms=metadata.get("matched_terms", []),
                chunk_index=metadata.get("chunk_index"),
                total_chunks=metadata.get("total_chunks"),
                context_before=metadata.get("context_before"),
                context_after=metadata.get("context_after")
            )
            search_results.append(search_result)
        
        response = SearchResponse(
            query_id=f"query_{datetime.utcnow().timestamp()}",
            query_text=query.query_text,
            query_type=query.query_type,
            results=search_results,
            total_results=len(search_results),
            returned_results=len(search_results),
            offset=query.offset,
            limit=query.limit,
            has_more=False,  # Would implement proper pagination
            execution_time_ms=0.0,  # Would measure actual execution time
            processed_query=query.query_text,  # For now, same as original
            query_terms=query.query_text.split(),  # Simple tokenization
            aggregations={}  # No aggregations for now
        )
        
        if audit_logger:
            await audit_logger.log_operation(
                user_id=user_id,
                operation="search_memories",
                metadata={"query": query.query_text, "results_count": len(search_results)}
            )
        
        return response
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/batch", response_model=BatchSearchResponse)
async def batch_search(
    batch_query: BatchSearchQuery,
    user_id: str = Depends(get_current_user)
):
    """Execute multiple search queries in batch."""
    try:
        await verify_consent(user_id, "search_memories")
        
        responses = []
        successful_queries = 0
        failed_queries = 0
        
        for query in batch_query.queries:
            try:
                # Ensure user can only search their own data
                if not query.user_id:
                    query.user_id = user_id
                elif query.user_id != user_id:
                    responses.append({"error": "Cannot search other users' data"})
                    failed_queries += 1
                    continue
                
                # Execute individual search
                response = await search_memories(query, user_id)
                responses.append(response)
                successful_queries += 1
                
            except Exception as e:
                responses.append({"error": str(e)})
                failed_queries += 1
                
                if batch_query.fail_on_error:
                    break
        
        return BatchSearchResponse(
            batch_id=batch_query.batch_id,
            total_queries=len(batch_query.queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            responses=responses,
            total_execution_time_ms=0.0  # Would measure actual time
        )
    except Exception as e:
        logger.error(f"Error in batch search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Context and Session Endpoints
@app.post("/context/build")
async def build_context(
    session_id: str = Body(...),
    conversation_id: str = Body(...),
    query: str = Body(...),
    user_id: str = Depends(get_current_user)
):
    """Build context for LLM interaction."""
    try:
        if not context_builder:
            raise HTTPException(status_code=500, detail="Context builder not initialized")
        
        context = await context_builder.build_context(user_id, session_id, conversation_id, query)
        return context
    except Exception as e:
        logger.error(f"Error building context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context/update")
async def update_context(
    session_id: str = Body(...),
    conversation_id: str = Body(...),
    user_message: str = Body(...),
    assistant_response: str = Body(...),
    extracted_info: Optional[Dict] = Body(None),
    user_id: str = Depends(get_current_user)
):
    """Update context after interaction."""
    try:
        if not context_builder:
            raise HTTPException(status_code=500, detail="Context builder not initialized")
        
        success = await context_builder.update_context_after_interaction(
            user_id, session_id, conversation_id, user_message, assistant_response, extracted_info
        )
        
        return {"success": success}
    except Exception as e:
        logger.error(f"Error updating context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# User Preferences Endpoints
@app.get("/preferences", response_model=UserPreferences)
async def get_user_preferences(user_id: str = Depends(get_current_user)):
    """Get user preferences."""
    try:
        if not kv_store:
            raise HTTPException(status_code=500, detail="KV store not initialized")
        
        prefs_store = UserPreferencesStore(kv_store)
        preferences = await prefs_store.get_preferences(user_id)
        
        if not preferences:
            # Return default preferences
            preferences = UserPreferences(
                user_id=user_id,
                consent_level="private",
                data_retention_days=None,
                language="en",
                timezone="UTC",
                theme="light",
                email_notifications=True,
                push_notifications=True,
                ai_personality="balanced",
                response_length="medium"
            )
        
        return preferences
    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/preferences", response_model=UserPreferences)
async def update_user_preferences(
    preferences: UserPreferences,
    user_id: str = Depends(get_current_user)
):
    """Update user preferences."""
    try:
        # Ensure user can only update their own preferences
        if preferences.user_id != user_id:
            raise HTTPException(status_code=403, detail="Cannot update other users' preferences")
        
        if not kv_store:
            raise HTTPException(status_code=500, detail="KV store not initialized")
        
        prefs_store = UserPreferencesStore(kv_store)
        result = await prefs_store.set_preferences(preferences)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to update preferences")
        
        return preferences
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin Endpoints (would require admin role in production)
@app.get("/admin/stats")
async def get_service_stats(user_id: str = Depends(get_current_user)):
    """Get service statistics."""
    try:
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "embedding_store": {
                "collections": 1,  # Would get actual stats
                "total_embeddings": 0
            },
            "kv_store": {
                "total_keys": 0
            },
            "active_sessions": 0
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/cleanup")
async def cleanup_expired_data(user_id: str = Depends(get_current_user)):
    """Clean up expired data."""
    try:
        cleanup_results = {}
        
        if kv_store:
            expired_kv = await kv_store.cleanup_expired()
            cleanup_results["expired_kv_items"] = expired_kv
        
        if memory_store:
            expired_memory = await memory_store.cleanup_expired_data()
            cleanup_results["expired_memory_items"] = expired_memory
        
        return {
            "success": True,
            "cleanup_results": cleanup_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.api.host, port=config.api.port, reload=config.api.debug)
