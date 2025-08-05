"""
Configuration management for LUKi Memory Service.

Handles environment variables, database URLs, model choices, and feature flags.
"""

import os
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    # Vector store (ChromaDB)
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "luki_memories"
    
    # Key-value store (PostgreSQL/Redis)
    postgres_url: Optional[str] = None
    redis_url: Optional[str] = None
    
    # Session store
    session_store_type: str = "memory"  # memory, redis, postgres
    session_ttl_seconds: int = 3600  # 1 hour


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    
    model_name: str = "all-MiniLM-L12-v2"
    model_cache_dir: str = "./models/sentence_transformers"
    batch_size: int = 32
    max_seq_length: int = 512
    device: str = "cpu"  # cpu, cuda, mps


@dataclass
class SecurityConfig:
    """Security and privacy configuration."""
    
    encryption_key: Optional[str] = None
    consent_levels: Optional[List[str]] = None
    pii_detection_enabled: bool = True
    audit_logging_enabled: bool = True
    
    def __post_init__(self):
        if self.consent_levels is None:
            self.consent_levels = ["private", "family", "research"]


@dataclass
class APIConfig:
    """API server configuration."""
    
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = False
    cors_origins: Optional[List[str]] = None
    api_key_required: bool = True
    rate_limit_per_minute: int = 100
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000", "http://localhost:8000"]


@dataclass
class MemoryServiceConfig:
    """Main configuration class for LUKi Memory Service."""
    
    database: DatabaseConfig
    embedding: EmbeddingConfig
    security: SecurityConfig
    api: APIConfig
    
    # Feature flags
    enable_grpc: bool = False
    enable_audit_trail: bool = True
    enable_versioning: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"


def load_config() -> MemoryServiceConfig:
    """Load configuration from environment variables."""
    
    # Database config
    db_config = DatabaseConfig(
        chroma_persist_directory=os.getenv(
            "CHROMA_PERSIST_DIR", 
            "./data/chroma_db"
        ),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME", 
            "luki_memories"
        ),
        postgres_url=os.getenv("POSTGRES_URL"),
        redis_url=os.getenv("REDIS_URL"),
        session_store_type=os.getenv("SESSION_STORE_TYPE", "memory"),
        session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "3600"))
    )
    
    # Embedding config
    embedding_config = EmbeddingConfig(
        model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L12-v2"),
        model_cache_dir=os.getenv("MODEL_CACHE_DIR", "./models/sentence_transformers"),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        max_seq_length=int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "512")),
        device=os.getenv("EMBEDDING_DEVICE", "cpu")
    )
    
    # Security config
    security_config = SecurityConfig(
        encryption_key=os.getenv("ENCRYPTION_KEY"),
        consent_levels=os.getenv("CONSENT_LEVELS", "private,family,research").split(","),
        pii_detection_enabled=os.getenv("PII_DETECTION_ENABLED", "true").lower() == "true",
        audit_logging_enabled=os.getenv("AUDIT_LOGGING_ENABLED", "true").lower() == "true"
    )
    
    # API config
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
    api_config = APIConfig(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8001")),
        debug=os.getenv("API_DEBUG", "false").lower() == "true",
        cors_origins=cors_origins.split(",") if cors_origins else [],
        api_key_required=os.getenv("API_KEY_REQUIRED", "true").lower() == "true",
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    )
    
    # Main config
    return MemoryServiceConfig(
        database=db_config,
        embedding=embedding_config,
        security=security_config,
        api=api_config,
        enable_grpc=os.getenv("ENABLE_GRPC", "false").lower() == "true",
        enable_audit_trail=os.getenv("ENABLE_AUDIT_TRAIL", "true").lower() == "true",
        enable_versioning=os.getenv("ENABLE_VERSIONING", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_format=os.getenv("LOG_FORMAT", "json")
    )


def get_data_dir() -> Path:
    """Get the data directory path, creating it if it doesn't exist."""
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_models_dir() -> Path:
    """Get the models directory path, creating it if it doesn't exist."""
    models_dir = Path(os.getenv("MODELS_DIR", "./models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


# Global config instance
config = load_config()
