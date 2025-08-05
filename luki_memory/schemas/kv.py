"""
Key-Value store schema definitions.

Defines data models for key-value storage, session data, and caching.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator


class KVDataType(str, Enum):
    """Data types for key-value storage."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"
    BINARY = "binary"
    LIST = "list"
    SET = "set"


class KVNamespace(str, Enum):
    """Namespaces for organizing key-value data."""
    USER_PREFERENCES = "user_prefs"
    SESSION_DATA = "sessions"
    CACHE = "cache"
    COUNTERS = "counters"
    LOCKS = "locks"
    METADATA = "metadata"
    TEMPORARY = "temp"


class KVItem(BaseModel):
    """Key-value item model."""
    
    # Core fields
    key: str = Field(..., description="Unique key identifier")
    value: Any = Field(..., description="Value data")
    namespace: KVNamespace = Field(KVNamespace.CACHE, description="Data namespace")
    
    # Type information
    data_type: KVDataType = Field(KVDataType.STRING, description="Value data type")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    access_count: int = Field(0, description="Number of times accessed")
    
    # Expiration
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    ttl_seconds: Optional[int] = Field(None, description="Time to live in seconds")
    
    # User and permissions
    user_id: Optional[str] = Field(None, description="Owner user ID")
    is_public: bool = Field(False, description="Public read access")
    
    # Storage options
    compress: bool = Field(False, description="Compress value for storage")
    encrypt: bool = Field(False, description="Encrypt value for storage")
    
    # Tags and indexing
    tags: List[str] = Field(default_factory=list)
    
    @validator('key')
    def validate_key(cls, v):
        if not v or not v.strip():
            raise ValueError('Key cannot be empty')
        # Basic key validation - alphanumeric, underscore, dash, dot
        import re
        if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
            raise ValueError('Key can only contain alphanumeric characters, underscore, dash, and dot')
        return v.strip()
    
    @validator('ttl_seconds')
    def validate_ttl(cls, v):
        if v is not None and v <= 0:
            raise ValueError('TTL must be positive')
        return v
    
    def is_expired(self) -> bool:
        """Check if the item has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def set_ttl(self, seconds: int) -> None:
        """Set TTL and calculate expiration time."""
        self.ttl_seconds = seconds
        self.expires_at = datetime.utcnow() + timedelta(seconds=seconds)


class SessionData(BaseModel):
    """Session data model for temporary user state."""
    
    # Session identification
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    
    # Session metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(..., description="Session expiration")
    
    # Session data
    data: Dict[str, Any] = Field(default_factory=dict, description="Session key-value data")
    
    # Session state
    is_active: bool = Field(True, description="Session is active")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    
    # Security
    csrf_token: Optional[str] = Field(None, description="CSRF protection token")
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() > self.expires_at
    
    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = datetime.utcnow()
        self.last_activity = datetime.utcnow()
    
    def extend_expiration(self, seconds: int) -> None:
        """Extend session expiration."""
        self.expires_at = datetime.utcnow() + timedelta(seconds=seconds)


class CacheEntry(BaseModel):
    """Cache entry model with metadata."""
    
    # Cache key and value
    cache_key: str = Field(..., description="Cache key")
    value: Any = Field(..., description="Cached value")
    
    # Cache metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="Cache expiration")
    hit_count: int = Field(0, description="Number of cache hits")
    
    # Cache tags for invalidation
    tags: List[str] = Field(default_factory=list)
    
    # Size estimation (bytes)
    estimated_size: Optional[int] = Field(None, description="Estimated size in bytes")
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def increment_hits(self) -> None:
        """Increment hit counter."""
        self.hit_count += 1


class KVBatchOperation(BaseModel):
    """Batch operation for multiple key-value items."""
    
    operation: str = Field(..., description="Operation: get, set, delete, exists")
    items: List[Union[KVItem, str]] = Field(..., description="Items or keys to operate on")
    
    # Batch options
    namespace: Optional[KVNamespace] = Field(None, description="Default namespace")
    fail_on_error: bool = Field(False, description="Stop on first error")
    atomic: bool = Field(False, description="All operations succeed or all fail")
    
    # Batch metadata
    batch_id: Optional[str] = Field(None, description="Batch operation ID")
    timeout_seconds: Optional[int] = Field(30, description="Operation timeout")
    
    @validator('operation')
    def validate_operation(cls, v):
        allowed_ops = {'get', 'set', 'delete', 'exists', 'increment', 'expire'}
        if v not in allowed_ops:
            raise ValueError(f'Operation must be one of: {allowed_ops}')
        return v


class KVSearchFilter(BaseModel):
    """Filter criteria for key-value search operations."""
    
    # Key patterns
    key_pattern: Optional[str] = Field(None, description="Key pattern (supports wildcards)")
    key_prefix: Optional[str] = Field(None, description="Key prefix filter")
    
    # Namespace filtering
    namespaces: Optional[List[KVNamespace]] = Field(None, description="Namespaces to search")
    
    # User filtering
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    include_public: bool = Field(True, description="Include public items")
    
    # Time filtering
    created_after: Optional[datetime] = Field(None, description="Created after timestamp")
    created_before: Optional[datetime] = Field(None, description="Created before timestamp")
    accessed_after: Optional[datetime] = Field(None, description="Last accessed after")
    
    # Expiration filtering
    include_expired: bool = Field(False, description="Include expired items")
    expires_within_seconds: Optional[int] = Field(None, description="Expires within N seconds")
    
    # Data type filtering
    data_types: Optional[List[KVDataType]] = Field(None, description="Filter by data types")
    
    # Tag filtering
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    
    # Pagination
    limit: Optional[int] = Field(100, description="Maximum results to return")
    offset: Optional[int] = Field(0, description="Results offset")


@dataclass
class KVOperationResult:
    """Result of key-value operations."""
    
    success: bool
    operation: str
    key: Optional[str] = None
    value: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    
    # Batch operation results
    successful_operations: int = 0
    failed_operations: int = 0
    results: List['KVOperationResult'] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []


class UserPreferences(BaseModel):
    """User preferences stored in key-value store."""
    
    user_id: str = Field(..., description="User identifier")
    
    # Privacy preferences
    consent_level: str = Field("private", description="Default consent level")
    data_retention_days: Optional[int] = Field(None, description="Data retention period")
    
    # Interface preferences
    language: str = Field("en", description="Preferred language")
    timezone: str = Field("UTC", description="User timezone")
    theme: str = Field("light", description="UI theme preference")
    
    # Notification preferences
    email_notifications: bool = Field(True, description="Enable email notifications")
    push_notifications: bool = Field(True, description="Enable push notifications")
    
    # AI preferences
    ai_personality: str = Field("balanced", description="AI personality preference")
    response_length: str = Field("medium", description="Preferred response length")
    
    # Custom preferences
    custom_settings: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
