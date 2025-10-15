#!/usr/bin/env python3
"""
Configuration settings for the Memory Service API
"""

from pydantic import Field, BaseModel
from typing import Optional, Any
import os
from pathlib import Path

# Import BaseSettings - use type: ignore to suppress type checker warnings
try:
    from pydantic_settings import BaseSettings  # type: ignore
except ImportError:
    try:
        from pydantic import BaseSettings  # type: ignore
    except ImportError:
        # Use BaseModel as fallback
        BaseSettings = BaseModel  # type: ignore


class Settings(BaseSettings):  # type: ignore
    """Application settings."""
    
    # API Configuration
    api_title: str = "LUKi Memory Service API"
    api_version: str = "2.0.0"
    debug: bool = False
    debug_mode: bool = True
    
    # Server Configuration
    host: str = "127.0.0.1"
    port: int = 8002
    workers: int = 1
    
    # ELR Pipeline Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"
    
    # Vector Database Configuration
    # Reads from CHROMA_PERSIST_DIR environment variable, falls back to /data/chroma_db
    vector_db_path: str = Field(
        default="/data/chroma_db",
        alias="CHROMA_PERSIST_DIR"
    )
    collection_name: str = "elr_embeddings"
    
    # Authentication Configuration
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    
    # Database Configuration (for user management)
    database_url: str = "sqlite:///./data/users.db"
    
    # Supabase Configuration
    supabase_url: Optional[str] = Field(default=None, alias="SUPABASE_URL")
    supabase_anon_key: Optional[str] = Field(default=None, alias="SUPABASE_ANON_KEY")
    supabase_service_role_key: Optional[str] = Field(default=None, alias="SUPABASE_SERVICE_ROLE_KEY")
    supabase_elr_table: str = "user_elr_data"
    supabase_profiles_table: str = "profiles"
    
    # Project Context Ingestion Controls
    # Enable/disable loading project context documents entirely
    # ENABLED: Loading curated LUKi personality and context documents
    project_context_ingest_enabled: bool = Field(default=True, alias="PROJECT_CONTEXT_INGEST_ENABLED")
    # Enable selective loading of only LUKi personality framework
    luki_personality_only: bool = Field(default=True, alias="LUKI_PERSONALITY_ONLY")
    # If true, perform ingestion only once per deploy (guarded by a marker file)
    project_context_ingest_once: bool = Field(default=True, alias="PROJECT_CONTEXT_INGEST_ONCE")
    # Optional override for the context directory (defaults to "/app/_context" in container)
    project_context_dir: Optional[str] = Field(default=None, alias="PROJECT_CONTEXT_DIR")
    # Optional override for the marker file path; defaults to f"{vector_db_path}/project_context_ingested.marker"
    project_context_marker_path: Optional[str] = Field(default=None, alias="PROJECT_CONTEXT_MARKER_PATH")
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Performance Configuration
    max_batch_size: int = 100
    max_search_results: int = 50
    request_timeout_seconds: int = 300
    
    # Storage Configuration
    max_memory_size_mb: int = 1000
    cleanup_interval_hours: int = 24
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = ""
    
    def __init__(self, **kwargs):
        # Skip environment parsing temporarily to avoid validation errors
        try:
            super().__init__(**kwargs)
        except Exception:
            # Fallback: initialize with defaults only
            for field_name, field_info in self.__fields__.items():
                if hasattr(field_info, 'default') and field_info.default is not ...:
                    setattr(self, field_name, field_info.default)
        # Ensure data directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            Path(self.vector_db_path).parent,
            Path(self.database_url.replace("sqlite:///", "")).parent if self.database_url.startswith("sqlite:///") else None
        ]
        
        for directory in directories:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Get application settings (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
