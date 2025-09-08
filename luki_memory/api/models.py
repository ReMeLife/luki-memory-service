#!/usr/bin/env python3
"""
Pydantic models for the Memory Service API
"""

from pydantic import BaseModel, Field, field_validator
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
    
    success: bool = Field(..., description="Whether ingestion was successful")
    message: str = Field(..., description="Status message")
    processed_items: int = Field(..., description="Number of items processed")
    embedded_chunks: int = Field(..., description="Number of chunks embedded")
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
    
    success: bool = Field(..., description="Whether search was successful")
    results: List[MemorySearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    query_time_seconds: float = Field(..., description="Query execution time")
    user_id: str = Field(..., description="User ID searched")


class UserMemoryStats(BaseModel):
    """User memory statistics."""
    
    user_id: str = Field(..., description="User identifier")
    total_memories: int = Field(..., description="Total number of memories")
    total_chunks: int = Field(..., description="Total number of chunks")
    content_type_breakdown: Dict[str, int] = Field(..., description="Breakdown by content type")
    sensitivity_breakdown: Dict[str, int] = Field(..., description="Breakdown by sensitivity")
    earliest_memory: Optional[datetime] = Field(None, description="Earliest memory date")
    latest_memory: Optional[datetime] = Field(None, description="Latest memory date")
    storage_size_mb: float = Field(..., description="Approximate storage size in MB")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")
    pipeline_ready: bool = Field(..., description="Whether ELR pipeline is ready")


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
        min_length=1,
        max_length=100,
        description="Batch of ELR ingestion requests"
    )
    batch_id: Optional[str] = Field(
        None,
        description="Optional batch identifier"
    )


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
