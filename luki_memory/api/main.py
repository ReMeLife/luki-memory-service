#!/usr/bin/env python3
"""
LUKi Memory Service API
FastAPI application for ELR ingestion, storage, and retrieval
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import logging
from datetime import datetime
import asyncio

from ..schemas.elr import ConsentLevel, SensitivityLevel, ELRContentType
from ..ingestion.embedding_integration import create_elr_to_vector_pipeline
from .models import (
    ELRIngestionRequest, ELRIngestionResponse,
    MemorySearchRequest, MemorySearchResponse,
    UserMemoryStats, HealthResponse
)
from .auth import get_current_user, User
from .config import get_settings
from .endpoints import auth, ingestion, search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LUKi Memory Service API",
    description="Electronic Life Record (ELR) ingestion, processing, and retrieval service",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
settings = get_settings()

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize the ELR pipeline on startup."""
    global pipeline
    logger.info("Initializing ELR pipeline...")
    
    try:
        pipeline = create_elr_to_vector_pipeline(
            embedding_model=settings.embedding_model,
            spacy_model=settings.spacy_model,
            persist_directory=settings.vector_db_path,
            collection_name=settings.collection_name
        )
        # Set pipeline in ingestion and search endpoints
        ingestion.set_pipeline(pipeline)
        search.set_pipeline(pipeline)
        logger.info("ELR pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ELR pipeline: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Memory Service API")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="2.0.0",
        pipeline_ready=pipeline is not None
    )

# Include routers
app.include_router(auth.router)
app.include_router(ingestion.router)
app.include_router(search.router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "LUKi Memory Service API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "luki_memory.api.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=True,
        log_level="info"
    )
