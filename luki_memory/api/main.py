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
from ..ingestion.chunker import ELRChunk
from .models import (
    ELRIngestionRequest, ELRIngestionResponse,
    MemorySearchRequest, MemorySearchResponse,
    UserMemoryStats, HealthResponse
)
from .auth import get_current_active_user
from .config import get_settings
from ..integrations.supabase_integration import create_supabase_integration
from .endpoints import ingestion, search, supabase
from pathlib import Path
import logging

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
project_knowledge_loaded = False

@app.on_event("startup")
async def startup_event():
    """Initialize separated ELR and project knowledge stores on startup."""
    global pipeline, project_knowledge_loaded
    logger.info("Initializing separated ELR and project knowledge stores...")
    
    try:
        # Initialize PROJECT KNOWLEDGE store (completely separate from ELR)
        from ..storage.knowledge_store import get_project_knowledge_store, initialize_project_knowledge
        from ..storage.elr_store import get_elr_store
        from ..storage.vector_store import create_embedding_store
        
        logger.info("Initializing project knowledge store...")
        project_store = get_project_knowledge_store()
        initialize_project_knowledge()  # Load authoritative facts
        
        # Load project context documents
        logger.info("Loading project context documents...")
        await load_project_context_into_store(project_store)
        project_knowledge_loaded = True
        
        search.set_project_knowledge_store(project_store)
        logger.info("Project knowledge store initialized successfully")
        
        # Initialize ELR store (completely separate from project knowledge)
        logger.info("Initializing ELR store...")
        elr_store = get_elr_store()
        search.set_elr_store(elr_store)
        logger.info("ELR store initialized successfully")
        
        # Mark pipeline as ready with separated stores
        pipeline = {"status": "ready", "type": "separated", "elr_store": True, "knowledge_store": True}
        logger.info("System ready with separated ELR and project knowledge stores")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

async def load_project_context_into_store(project_store):
    """Load project context documents into the knowledge store."""
    base_path = Path(__file__).parent.parent.parent.parent  # Navigate to workspace root
    total_chunks = 0
    
    # Define context documents with priorities
    context_documents = [
        ("_context/00-System-Overview.md", "critical"),
        ("_context/01-MVP-Standalone.md", "high"),
        ("_context/02-Phase-2-Integration.md", "high"),
        ("_context/03-Security-Privacy.md", "high"),
        ("_context/LUKi-Blueprint.md", "critical"),
        ("_context/remelife-whitepaper", "medium"),
        ("_context/activities-summary", "medium"),
        ("context-documents/luki-website-info.txt", "critical"),
        ("context-documents/gitbook-info.txt", "high"),
        ("context-documents/Federated-Learning", "medium")
    ]
    
    # Load LUKi identity first
    identity_chunks = await load_luki_identity(project_store)
    total_chunks += identity_chunks
    
    # Load each document
    for file_path, priority in context_documents:
        chunks_created = await load_context_document(project_store, base_path, file_path, priority)
        total_chunks += chunks_created
    
    logger.info(f"Project context loading completed! Total chunks: {total_chunks}")

async def load_luki_identity(project_store) -> int:
    """Load LUKi's core identity into the store."""
    identity_content = """# LUKi AI Companion - Core Identity

## Revolutionary AI Companion
I am LUKi, your AI companion within the ReMeLife ecosystem. I'm designed to be your warmly mischievous, truth-telling companion who helps you navigate digital care and wellness.

## Core Identity
- Revolutionary AI companion for personalized digital care
- Memory Keeper: I remember your preferences, stories, and care needs from your Electronic Life Records (ELR)
- Truth-Telling Companion: I provide honest, evidence-based guidance while maintaining warmth
- Warmly Mischievous: I bring gentle humor and playfulness to our interactions
- Powered by GPT-OSS 120B via Together AI for sophisticated natural language understanding

## Mission
- Democratize data ownership in healthcare through blockchain technology
- Provide personalized care recommendations based on your unique ELR data
- Support family caregivers and professional care teams
- Bridge the gap between technology and human-centered care
- Facilitate meaningful connections within the ReMeLife community

## Tri-Token Economy Integration
- $LUKI Token: Primary utility token for AI interactions and premium features (Solana SPL token)
- REME Tokens: Reward tokens for care actions and community engagement (ReMeGrid blockchain)
- CAPs (Care Action Points): Points earned through care activities and AI participation

## Technical Architecture
- GPT-OSS 120B for natural language generation and reasoning
- Vector embeddings using Sentence-Transformers for semantic memory search
- ChromaDB for long-term memory storage and retrieval
- LangChain framework for conversation orchestration and tool routing
- SQLite for local data storage and ChromaDB for vector embeddings

## Electronic Life Records (ELR)
ELR stands for Electronic Life Records - this is the ONLY correct definition. ELR captures comprehensive personal profiles through activity engagement including likes/dislikes, memories, lifestyle, habits, interests, demographics, and family information.

## Care Action Points (CAPs)
CAPs stands for Care Action Points - NOT 'Care Points'. These are points earned through care activities and AI participation within the ReMeLife ecosystem.

## ReMeGrid Convex Lattice Blockchain
ReMeGrid is the custom blockchain infrastructure supporting the ReMeLife ecosystem with Proof-of-Authority consensus, CAD20 token standard, and cross-chain bridge capabilities."""
    
    chunks = chunk_text(identity_content, 1000, 200)
    
    for i, chunk in enumerate(chunks):
        chunk_metadata = {
            "source": "luki_core_identity",
            "document_type": "system_identity",
            "priority": "critical",
            "chunk_index": i,
            "total_chunks": len(chunks),
            "user_id": "system",
            "loaded_at": datetime.now().isoformat(),
            "content_type": "static_knowledge"
        }
        
        elr_chunk = ELRChunk(
            text=chunk,
            chunk_id="",  # Will be auto-generated
            parent_item_id=f"identity_{i}",
            chunk_index=i,
            total_chunks=len(chunks),
            user_id="system",
            content_type=ELRContentType.MEMORY,
            consent_level=ConsentLevel.RESEARCH,
            sensitivity_level=SensitivityLevel.PUBLIC,
            source_file="luki_core_identity",
            metadata=chunk_metadata
        )
        
        project_store.add_chunk(elr_chunk)
    
    logger.info(f"Loaded LUKi identity: {len(chunks)} chunks")
    return len(chunks)

async def load_context_document(project_store, base_path: Path, file_path: str, priority: str) -> int:
    """Load a single context document into the store."""
    full_path = base_path / file_path
    
    if not full_path.exists():
        logger.warning(f"Context document not found: {file_path}")
        return 0
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create document metadata
        doc_metadata = {
            "source": file_path,
            "document_type": "project_context",
            "priority": priority,
            "loaded_at": datetime.now().isoformat(),
            "content_type": "static_knowledge"
        }
        
        # Use smaller chunks for critical content
        chunk_size = 800 if priority == "critical" else 1000
        chunk_overlap = 150 if priority == "critical" else 200
        
        # Chunk the document
        chunks = chunk_text(content, chunk_size, chunk_overlap)
        
        # Add chunks to embedding store
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **doc_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "user_id": "system"
            }
            
            elr_chunk = ELRChunk(
                text=chunk,
                chunk_id="",  # Will be auto-generated
                parent_item_id=f"doc_{i}",
                chunk_index=i,
                total_chunks=len(chunks),
                user_id="system",
                content_type=ELRContentType.MEMORY,
                consent_level=ConsentLevel.RESEARCH,
                sensitivity_level=SensitivityLevel.PUBLIC,
                source_file=file_path,
                metadata=chunk_metadata
            )
            
            project_store.add_chunk(elr_chunk)
        
        logger.info(f"Loaded {file_path}: {len(chunks)} chunks")
        return len(chunks)
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return 0

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks with smart boundary detection."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at natural boundaries
        if end < len(text):
            # Look for sentence endings
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
            else:
                # Look for paragraph breaks
                para_break = text.rfind('\n\n', start, end)
                if para_break > start + chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for any line break
                    line_break = text.rfind('\n', start, end)
                    if line_break > start + chunk_size // 2:
                        end = line_break + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - chunk_overlap
        
    return chunks

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
        pipeline_ready=pipeline is not None and project_knowledge_loaded
    )

# Include routers
app.include_router(ingestion.router)
app.include_router(search.router)
app.include_router(supabase.router)

# Debug endpoint for collection statistics
@app.get("/debug/collections")
async def get_collection_stats():
    """Get statistics about ChromaDB collections."""
    try:
        import chromadb
        from pathlib import Path
        
        # Get ChromaDB client
        settings = get_settings()
        client = chromadb.PersistentClient(path=str(settings.vector_db_path))
        
        # Check project_context collection
        try:
            collection = client.get_collection("project_context")
            count = collection.count()
            
            # Sample some documents to verify content
            sample_results = collection.get(limit=5)
            
            return {
                "collections": [
                    {
                        "name": "project_context",
                        "count": count,
                        "sample_metadata": (sample_results.get("metadatas") or [])[:2] if sample_results and isinstance(sample_results, dict) else []
                    }
                ],
                "project_context": count,
                "total_documents": count
            }
        except Exception as e:
            logger.warning(f"Collection 'project_context' not found: {e}")
            return {
                "collections": [],
                "project_context": 0,
                "total_documents": 0,
                "error": "Collection not found"
            }
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}")
        return {"error": str(e), "collections": [], "project_context": 0}

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
