#!/usr/bin/env python3
"""
Configuration settings for the Memory Service API
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_title: str = "LUKi Memory Service API"
    api_version: str = "2.0.0"
    debug: bool = Field(False, description="Debug mode")
    debug_mode: bool = Field(True, description="Enable debug endpoints")
    
    # Server Configuration
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8002, env="PORT", description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    
    # ELR Pipeline Configuration
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Embedding model")
    spacy_model: str = Field("en_core_web_sm", description="SpaCy model")
    
    # Vector Database Configuration
    vector_db_path: str = Field("./data/chroma_db", description="Vector database path")
    collection_name: str = Field("elr_embeddings", description="Collection name")
    
    # Authentication Configuration
    jwt_secret_key: str = Field("your-secret-key-change-in-production", description="JWT secret key")
    jwt_algorithm: str = Field("HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(1440, description="Token expiry minutes")
    
    # Database Configuration (for user management)
    database_url: str = Field("sqlite:///./data/users.db", description="Database URL")
    
    # Logging Configuration
    log_level: str = Field("INFO", description="Log level")
    log_file: Optional[str] = Field(None, description="Log file path")
    
    # Performance Configuration
    max_batch_size: int = Field(100, description="Maximum batch size")
    max_search_results: int = Field(50, description="Maximum search results")
    request_timeout_seconds: int = Field(300, description="Request timeout")
    
    # Storage Configuration
    max_memory_size_mb: int = Field(1000, description="Maximum memory size MB")
    cleanup_interval_hours: int = Field(24, description="Cleanup interval hours")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
