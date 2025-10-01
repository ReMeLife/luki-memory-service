#!/usr/bin/env python3
"""
Pydantic models for the Memory Service API
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

from ..schemas.elr import ConsentLevel, SensitivityLevel, ELRContentType


class ELRIngestionRequest(BaseModel):
    """Request model for ELR ingestion."""
    
    elr_data: Dict[str, Any] = Field(
        ..., 
        description="Electronic Life Record data in JSON format"
    )
    user_id: str = Field(
        ..., 
        min_length=1, 
        max_length=255,
        description="Unique identifier for the user"
    )
    source_file: Optional[str] = Field(
        None,
        description="Original source file name"
    )
    consent_level: ConsentLevel = Field(
        ConsentLevel.PRIVATE,
        description="Consent level for data processing"
    )
    sensitivity_level: SensitivityLevel = Field(
        SensitivityLevel.PERSONAL,
        description="Sensitivity level of the data"
    )
    
    @field_validator('elr_data')
    @classmethod
    def validate_elr_data(cls, v):
        """Validate ELR data structure."""
        if not isinstance(v, dict):
            raise ValueError("ELR data must be a dictionary")
        if not v:
            raise ValueError("ELR data cannot be empty")
        return v


class ELRIngestionResponse(BaseModel):
    """Response model for ELR ingestion."""
    model_config = ConfigDict(extra='forbid')
    
    success: bool = Field(..., description="Whether the ingestion was successful")
    chunks_created: int = Field(..., description="Number of chunks created")
    message: str = Field(..., description="Response message")
    chunk_ids: Optional[List[str]] = Field(None, description="IDs of created chunks")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    created_item_ids: List[str] = Field(default_factory=list, description="IDs of created items")


class MemorySearchRequest(BaseModel):
    """Request model for memory search."""
    
    user_id: str = Field(
        ..., 
        min_length=1, 
        max_length=255,
        description="User ID to search memories for"
    )
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Search query"
    )
    k: int = Field(
        5, 
        ge=1, 
        le=50,
        description="Number of results to return"
    )
    content_types: Optional[List[ELRContentType]] = Field(
        None,
        description="Filter by content types"
    )
    sensitivity_filter: Optional[List[SensitivityLevel]] = Field(
        None,
        description="Filter by sensitivity levels"
    )
    date_from: Optional[datetime] = Field(
        None,
        description="Filter memories from this date"
    )
    date_to: Optional[datetime] = Field(
        None,
        description="Filter memories to this date"
    )


class MemorySearchResult(BaseModel):
    """Individual memory search result."""
    
    content: str = Field(..., description="Memory content")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    metadata: Dict[str, Any] = Field(..., description="Memory metadata")
    chunk_id: str = Field(..., description="Chunk identifier")
    created_at: datetime = Field(..., description="Creation timestamp")


class MemorySearchResponse(BaseModel):
    """Response model for memory search."""
    model_config = ConfigDict(extra='forbid')
    
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of matching results")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")
    user_id: str = Field(..., description="User ID searched")


class UserMemoryStats(BaseModel):
    """User memory statistics model."""
    model_config = ConfigDict(extra='forbid')
    
    user_id: str
    email: str
    total_memories: int
    recent_memories: int
    content_types: Dict[str, int]
    consent_levels: Dict[str, int]
    last_updated: str


class HealthResponse(BaseModel):
    """Health check response model."""
    model_config = ConfigDict(extra='forbid')
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Response timestamp")
    version: str = Field(..., description="Service version")
    pipeline_ready: bool = Field(..., description="Whether the ELR pipeline is ready")


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class BatchIngestionRequest(BaseModel):
    """Request model for batch ELR ingestion."""
    
    batch_data: List[ELRIngestionRequest] = Field(
        ..., 
        description="Batch of ELR ingestion requests"
    )
    batch_id: Optional[str] = Field(
        None,
        description="Optional batch identifier"
    )
    
    @field_validator('batch_data')
    @classmethod
    def validate_batch_data(cls, v):
        """Validate batch data length constraints."""
        if not isinstance(v, list):
            raise ValueError("batch_data must be a list")
        if len(v) < 1:
            raise ValueError("batch_data must contain at least 1 item")
        if len(v) > 100:
            raise ValueError("batch_data cannot contain more than 100 items")
        return v


class BatchIngestionResponse(BaseModel):
    """Response model for batch ELR ingestion."""
    
    batch_id: str = Field(..., description="Batch identifier")
    total_requests: int = Field(..., description="Total number of requests in batch")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    individual_results: List[ELRIngestionResponse] = Field(..., description="Individual results")
    total_processing_time_seconds: float = Field(..., description="Total batch processing time")


class UserCreateRequest(BaseModel):
    """Request model for user creation."""
    
    user_id: str = Field(
        ..., 
        min_length=1, 
        max_length=255,
        description="Unique user identifier"
    )
    email: Optional[str] = Field(
        None,
        description="User email address"
    )
    full_name: Optional[str] = Field(
        None,
        max_length=500,
        description="User full name"
    )
    preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="User preferences"
    )


class UserResponse(BaseModel):
    """Response model for user operations."""
    
    user_id: str = Field(..., description="User identifier")
    email: Optional[str] = Field(None, description="User email")
    full_name: Optional[str] = Field(None, description="User full name")
    created_at: datetime = Field(..., description="User creation timestamp")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    memory_count: int = Field(0, description="Number of memories stored")
    is_active: bool = Field(True, description="Whether user is active")


class UserUpdateRequest(BaseModel):
    """Request model for user updates."""
    
    email: Optional[str] = Field(None, description="Updated email address")
    full_name: Optional[str] = Field(None, max_length=500, description="Updated full name")
    preferences: Optional[Dict[str, Any]] = Field(None, description="Updated preferences")
    is_active: Optional[bool] = Field(None, description="Updated active status")


class UserListResponse(BaseModel):
    """Response model for user list operations."""
    
    users: List[UserResponse] = Field(..., description="List of users")
    total_count: int = Field(..., description="Total number of users")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(50, description="Number of users per page")


class SupabaseMemoryRequest(BaseModel):
    """Request model for Supabase memory operations."""
    
    user_id: str = Field(..., min_length=1, max_length=255, description="User identifier")
    memory_data: Dict[str, Any] = Field(..., description="Memory data from Supabase")
    source_table: Optional[str] = Field(None, description="Source Supabase table")
    sync_timestamp: Optional[datetime] = Field(None, description="Synchronization timestamp")


class SupabaseMemoryResponse(BaseModel):
    """Response model for Supabase memory operations."""
    
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Operation status message")
    synced_records: int = Field(0, description="Number of records synchronized")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


class LoginRequest(BaseModel):
    """Request model for user login."""
    
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class LoginResponse(BaseModel):
    """Response model for user login."""
    
    success: bool = Field(..., description="Whether login was successful")
    access_token: str = Field(..., description="Access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: UserResponse = Field(..., description="User information")


class TokenResponse(BaseModel):
    """Response model for token operations."""
    
    access_token: str = Field(..., description="Access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


