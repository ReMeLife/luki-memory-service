#!/usr/bin/env python3
"""
Complete FastAPI application with all endpoints integrated
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import datetime

from .main import app
from .endpoints import auth, ingestion, search, users
from .endpoints import metrics
from .endpoints.ingestion import set_pipeline as set_ingestion_pipeline
from .endpoints.search import set_pipeline as set_search_pipeline
from .config import get_settings
from ..ingestion.embedding_integration import create_elr_to_vector_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Include routers
app.include_router(auth.router)
app.include_router(ingestion.router)
app.include_router(search.router)
app.include_router(users.router)
app.include_router(metrics.router)

@app.on_event("startup")
async def startup_event():
    """Initialize the ELR pipeline and project knowledge store."""
    logger.info("Initializing Memory Service API...")
    
    try:
        # Create ELR pipeline
        pipeline = create_elr_to_vector_pipeline(
            embedding_model=settings.embedding_model,
            spacy_model=settings.spacy_model,
            persist_directory=settings.vector_db_path,
            collection_name=settings.collection_name
        )
        
        # Inject pipeline into endpoints
        set_ingestion_pipeline(pipeline)
        set_search_pipeline(pipeline)
        
        # Initialize project knowledge store
        from ..storage.vector_store import create_embedding_store
        from .endpoints.search import set_project_knowledge_store
        project_store = create_embedding_store(
            model_name=settings.embedding_model,
            persist_directory=settings.vector_db_path,
            collection_name="project_context"
        )
        set_project_knowledge_store(project_store)
        
        logger.info("Memory Service API initialized successfully")
        logger.info("Project knowledge store initialized successfully")
        logger.info(f"API available at: http://{settings.host}:{settings.port}")
        logger.info(f"Documentation at: http://{settings.host}:{settings.port}/docs")
        
    except Exception as e:
        logger.error(f"Failed to initialize Memory Service API: {e}")
        raise

if __name__ == "__main__":
    uvicorn.run(
        "luki_memory.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        workers=settings.workers
    )
