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
from contextlib import asynccontextmanager
import uvicorn
import logging
from datetime import datetime
import asyncio
import sys
import os

from ..schemas.elr import ConsentLevel, SensitivityLevel, ELRContentType
from ..ingestion.embedding_integration import create_elr_to_vector_pipeline
from ..ingestion.chunker import ELRChunk
from .models import (
    ELRIngestionRequest, ELRIngestionResponse,
    MemorySearchRequest, MemorySearchResponse,
    UserMemoryStats, HealthResponse
)
from .config import get_settings
from .endpoints import ingestion, search, delete
try:
    from .endpoints.supabase import router as supabase_router
except ImportError:
    supabase_router = None

try:
    from .endpoints.auth import router as auth_router
except ImportError:
    auth_router = None
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None
project_knowledge_loaded = False

async def delayed_startup_background():
    """Wait for health checks to pass, then start full initialization."""
    # Wait 10 seconds to let Railway health checks pass first (reduced from 30)
    await asyncio.sleep(10)
    logger.info("Starting delayed background initialization after health check window...")
    try:
        await full_startup_background()
        logger.info("✅ Background initialization completed successfully")
    except Exception as e:
        logger.error(f"❌ Background initialization failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ultra-minimal lifespan for instant health check response."""
    # Absolute minimal startup - just set flags for health checks
    global project_knowledge_loaded, pipeline
    project_knowledge_loaded = True  # Set immediately for health checks
    pipeline = {"status": "ready", "type": "minimal"}
    
    # Start initialization in background AFTER a delay to let health checks pass
    asyncio.create_task(delayed_startup_background())
    
    yield
    # Shutdown
    await shutdown_event()

# Create FastAPI app with metadata
app = FastAPI(
    title="LUKi Memory Service",
    description="Electronic Life Records (ELR) ingestion, storage, and retrieval API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
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

async def full_startup_background():
    """Complete startup initialization in background to avoid blocking FastAPI."""
    try:
        logger.info("Initializing separated ELR and project knowledge stores...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Python path: {sys.path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
        
        # Import knowledge store with comprehensive fallback strategies
        knowledge_store_imported = False
        import_errors = []
        
        # Strategy 1: Relative import
        try:
            from ..storage.knowledge_store import get_project_knowledge_store, initialize_project_knowledge
            knowledge_store_imported = True
            logger.info("Successfully imported knowledge_store using relative import")
        except ImportError as e1:
            import_errors.append(f"Relative import failed: {e1}")
            logger.warning(f"Relative import failed for knowledge_store: {e1}")
        
        # Strategy 2: Absolute import
        if not knowledge_store_imported:
            try:
                from luki_memory.storage.knowledge_store import get_project_knowledge_store, initialize_project_knowledge
                knowledge_store_imported = True
                logger.info("Successfully imported knowledge_store using absolute import")
            except ImportError as e2:
                import_errors.append(f"Absolute import failed: {e2}")
                logger.warning(f"Absolute import failed for knowledge_store: {e2}")
        
        # Strategy 3: Fallback knowledge store (if main knowledge_store fails)
        if not knowledge_store_imported:
            try:
                from ..api.fallback_knowledge import MinimalKnowledgeStore
                # Create a minimal fallback that provides basic functionality
                logger.warning("Using fallback knowledge store - main knowledge_store import failed")
                knowledge_store_imported = True
            except ImportError as e3:
                import_errors.append(f"Fallback import failed: {e3}")
                logger.warning(f"Fallback import failed: {e3}")
        
        if not knowledge_store_imported:
            all_errors = "; ".join(import_errors)
            logger.error(f"CRITICAL: Cannot import knowledge_store after all strategies. Errors: {all_errors}")
            return
        
        try:
            from ..storage.elr_store import get_elr_store
        except ImportError:
            from luki_memory.storage.elr_store import get_elr_store
            
        try:
            from ..storage.vector_store import create_embedding_store
        except ImportError:
            from luki_memory.storage.vector_store import create_embedding_store
        
        # Initialize stores
        await initialize_stores_background()
        
    except Exception as e:
        logger.error(f"Background startup failed: {e}")

async def initialize_stores_background():
    """Initialize all stores in background to avoid blocking HTTP server startup."""
    try:
        logger.info("Background store initialization started...")
        
        # Import functions with fallback
        try:
            from ..storage.knowledge_store import get_project_knowledge_store, initialize_project_knowledge
        except ImportError:
            from luki_memory.storage.knowledge_store import get_project_knowledge_store, initialize_project_knowledge
            
        try:
            from ..storage.elr_store import get_elr_store
        except ImportError:
            from luki_memory.storage.elr_store import get_elr_store
        
        # Initialize project knowledge store
        logger.info("Initializing project knowledge store...")
        project_store = get_project_knowledge_store()
        
        # Initialize Supabase integration if available
        try:
            from .endpoints.supabase import initialize_supabase_integration
            from .config import Settings
            settings = Settings()
            
            supabase_integration = initialize_supabase_integration(
                config=settings,
                elr_pipeline=None,  # Will be created in the integration if needed
                embedding_store=None  # Will be created in the integration if needed
            )
            logger.info("Supabase integration initialized")
        except ImportError as e:
            logger.warning(f"Supabase integration not available: {e}")
            supabase_integration = None
        
        if supabase_integration is None:
            initialize_project_knowledge()  # Load authoritative facts
        
        # Re-enabled project context ingestion for LUKi identity
        logger.info("Project context ingestion ENABLED - loading LUKi identity and knowledge base")
        
        # Project context ingestion
        settings = get_settings()
        if not settings.project_context_ingest_enabled:
            logger.info("Project context ingestion disabled by configuration (PROJECT_CONTEXT_INGEST_ENABLED=false)")
        else:
            # Determine marker path (defaults under vector_db_path)
            default_marker = Path(settings.vector_db_path) / "project_context_ingested.marker"
            marker_path = Path(settings.project_context_marker_path) if settings.project_context_marker_path else default_marker
            if settings.project_context_ingest_once and marker_path.exists():
                logger.info(f"Project context already ingested; skipping (marker found at {marker_path})")
            else:
                logger.info("Starting background context document loading...")
                override_dir = Path(settings.project_context_dir) if settings.project_context_dir else None
                await load_project_context_into_store(project_store, context_dir=override_dir)
                # Write marker to prevent re-ingestion on reloads/restarts
                if settings.project_context_ingest_once:
                    try:
                        marker_path.parent.mkdir(parents=True, exist_ok=True)
                        marker_path.touch(exist_ok=True)
                        logger.info(f"Created project context ingestion marker at {marker_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create ingestion marker at {marker_path}: {e}")
        
        search.set_project_knowledge_store(project_store)
        logger.info("Project knowledge store initialized successfully")
        
        # Initialize ELR store (completely separate from project knowledge)
        logger.info("Initializing ELR store...")
        elr_store = get_elr_store()
        search.set_elr_store(elr_store)
        delete.set_elr_store(elr_store)  # Inject ELR store into delete module
        logger.info("ELR store initialized successfully")
        
        # Initialize the ingestion pipeline with the ELR store
        logger.info("Setting up ingestion pipeline...")
        ingestion.set_pipeline(elr_store)
        logger.info("Ingestion pipeline configured successfully")
        
        logger.info("Background store initialization completed successfully")
    except Exception as e:
        logger.error(f"Background store initialization failed: {e}")

async def load_project_context_into_store(project_store, context_dir: Optional[Path] = None):
    """Load project context documents into the knowledge store exactly once.

    Only ingest files that currently exist under the local `_context/` directory
    of the memory service project. This avoids persistent loops on Railway when
    the server reloads.
    """
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")
    base_path = Path(current_dir)
    # Determine context directory
    if context_dir is not None:
        context_path = Path(context_dir)
    else:
        # Prefer local project `_context/` directory; fallback to /app/_context
        if (base_path / "_context").exists():
            context_path = base_path / "_context"
        else:
            context_path = Path("/app/_context")
    logger.info(f"Using context directory: {context_path} (exists: {context_path.exists()})")
    if not context_path.exists():
        logger.warning("Context directory not found; skipping project context ingestion")
        return
    total_chunks = 0
    
    # Build dynamic list of ingestible files (.md and .txt) within `_context/`
    allowed_suffixes = {".md", ".txt"}
    settings = get_settings()
    
    if settings.luki_personality_only:
        # Only load LUKi personality framework for production efficiency
        personality_files = [
            "10-LUKi-Personality-Framework.md",
            "LUKi-Blueprint.md"
        ]
        files_to_ingest = []
        for filename in personality_files:
            file_path = context_path / filename
            if file_path.exists():
                files_to_ingest.append(file_path)
        logger.info(f"Selective ingestion: Loading only LUKi personality files ({len(files_to_ingest)} found)")
    else:
        # Load all context files
        files_to_ingest = [p for p in sorted(context_path.rglob("*")) if p.is_file() and (p.suffix in allowed_suffixes)]
        logger.info(f"Full ingestion: Loading all context files ({len(files_to_ingest)} found)")
    
    # Load LUKi identity first
    luki_identity = """
    LUKi is an AI assistant focused on Electronic Life Records (ELR) and personal memory management.
    Core capabilities: ELR ingestion, semantic search, privacy-preserving storage, personalized responses.
    Built with FastAPI, ChromaDB, SentenceTransformers for embedding generation.
    Designed for ReMeLife platform integration with strong privacy and consent management.
    """
    
    chunks = chunk_text(luki_identity, chunk_size=1000, chunk_overlap=200)
    for i, chunk in enumerate(chunks):
        try:
            elr_chunk = ELRChunk(
                text=chunk,
                chunk_id="",  # Will be auto-generated
                parent_item_id=f"identity_{i}",
                chunk_index=i,
                total_chunks=len(chunks),
                start_char=0,
                end_char=len(chunk),
                user_id="system",
                content_type=ELRContentType.MEMORY,
                consent_level=ConsentLevel.RESEARCH,
                sensitivity_level=SensitivityLevel.PUBLIC,
                embedding_model="all-MiniLM-L12-v2",
                chunk_quality_score=1.0,
                source_file="luki_identity",
                metadata={
                    "source": "luki_identity",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_type": "identity",
                    "priority": "high"
                }
            )
            project_store.add_chunk(elr_chunk)
        except Exception as e:
            logger.error(f"Failed to add LUKi identity chunk {i}: {e}")
    
    logger.info(f"Loaded LUKi identity: {len(chunks)} chunks")
    total_chunks += len(chunks)
    
    # Load context documents dynamically
    for file_path in files_to_ingest:
        if file_path.exists():
            try:
                content = file_path.read_text(encoding='utf-8')
                chunks = chunk_text(content, chunk_size=1000, chunk_overlap=200)
                
                for i, chunk in enumerate(chunks):
                    try:
                        elr_chunk = ELRChunk(
                            text=chunk,
                            chunk_id="",  # Will be auto-generated
                            parent_item_id=f"doc_{file_path.name}_{i}",
                            chunk_index=i,
                            total_chunks=len(chunks),
                            start_char=0,
                            end_char=len(chunk),
                            user_id="system",
                            content_type=ELRContentType.MEMORY,
                            consent_level=ConsentLevel.RESEARCH,
                            sensitivity_level=SensitivityLevel.PUBLIC,
                            embedding_model="all-MiniLM-L12-v2",
                            chunk_quality_score=1.0,
                            source_file=str(file_path.relative_to(context_path)),
                            metadata={
                                "source": str(file_path.relative_to(context_path)),
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "content_type": "documentation",
                                "priority": "medium"
                            }
                        )
                        project_store.add_chunk(elr_chunk)
                    except Exception as e:
                        logger.error(f"Failed to add chunk {i} from {file_path}: {e}")
                
                logger.info(f"Loaded {file_path.name}: {len(chunks)} chunks")
                total_chunks += len(chunks)
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        else:
            logger.warning(f"Context file not found: {file_path}")
    
    logger.info(f"Total context chunks loaded: {total_chunks}")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        
        chunks.append(text[start:end])
        
        if end >= len(text):
            break
            
        start = end - chunk_overlap
        
    return chunks

async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down LUKi Memory Service...")
    # Add any cleanup logic here
    logger.info("Shutdown complete")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {"status": "ok", "service": "luki-memory-service"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Fast health check endpoint for Railway deployment."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "luki-memory-service",
        "version": "1.0.0"
    }

# Include routers
app.include_router(ingestion.router)
app.include_router(search.router)
app.include_router(delete.router)

# Include auth router
if auth_router:
    try:
        app.include_router(auth_router)
    except Exception as e:
        logger.warning(f"Failed to include auth router: {e}")
else:
    logger.warning("Auth router not available - skipping")

# Include supabase router
if supabase_router:
    try:
        app.include_router(supabase_router)
    except Exception as e:
        logger.warning(f"Failed to include supabase router: {e}")
else:
    logger.warning("Supabase router not available - skipping")

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "luki_memory.api.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=True,
        log_level="info"
    )
