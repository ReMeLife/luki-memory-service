"""
ELR (Episodic Life Record) schema definitions.

Defines data models for ELR items, consent levels, sensitivity enums, and metadata.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator


class ConsentLevel(str, Enum):
    """Consent levels for data usage."""
    PRIVATE = "private"      # Only user can access
    FAMILY = "family"        # User + family members
    RESEARCH = "research"    # Anonymized research use allowed


class SensitivityLevel(str, Enum):
    """Sensitivity classification for ELR content."""
    PUBLIC = "public"        # No sensitive information
    PERSONAL = "personal"    # Personal but not sensitive
    SENSITIVE = "sensitive"  # Contains PII or sensitive data
    CONFIDENTIAL = "confidential"  # Highly sensitive/medical


class ELRContentType(str, Enum):
    """Types of ELR content."""
    LIFE_STORY = "life_story"
    MEMORY = "memory"
    PREFERENCE = "preference"
    FAMILY = "family"
    INTEREST = "interest"
    HEALTH = "health"
    GOAL = "goal"
    RELATIONSHIP = "relationship"


class ELRMetadata(BaseModel):
    """Metadata for ELR items."""
    
    # Core identifiers
    user_id: str = Field(..., description="User identifier")
    source_file: Optional[str] = Field(None, description="Original source file")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    # Content classification
    content_type: ELRContentType
    consent_level: ConsentLevel = ConsentLevel.PRIVATE
    sensitivity_level: SensitivityLevel = SensitivityLevel.PERSONAL
    
    # Processing metadata
    chunk_index: Optional[int] = Field(None, description="Index if part of chunked content")
    total_chunks: Optional[int] = Field(None, description="Total chunks for this item")
    language: str = Field("en", description="Content language")
    
    # Semantic tags
    tags: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list, description="Named entities extracted")
    
    # Quality metrics
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    completeness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Audit trail
    version: int = Field(1, description="Version number for updates")
    checksum: Optional[str] = Field(None, description="Content hash for integrity")


class ELRItem(BaseModel):
    """Core ELR item model."""
    
    # Identifiers
    id: Optional[str] = Field(None, description="Unique item ID")
    external_id: Optional[str] = Field(None, description="External system ID")
    
    # Content
    content: str = Field(..., description="Main text content")
    title: Optional[str] = Field(None, description="Optional title/summary")
    
    # Metadata
    metadata: ELRMetadata
    
    # Relationships
    parent_id: Optional[str] = Field(None, description="Parent item ID")
    related_ids: List[str] = Field(default_factory=list, description="Related item IDs")
    
    # Temporal information
    event_date: Optional[datetime] = Field(None, description="When the event occurred")
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    
    # Geospatial (optional)
    location: Optional[str] = Field(None, description="Location description")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Lat/lng coordinates")
    
    # Custom fields
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()
    
    @validator('coordinates')
    def validate_coordinates(cls, v):
        if v is not None:
            required_keys = {'lat', 'lng'}
            if not required_keys.issubset(v.keys()):
                raise ValueError('Coordinates must contain lat and lng keys')
            if not (-90 <= v['lat'] <= 90):
                raise ValueError('Latitude must be between -90 and 90')
            if not (-180 <= v['lng'] <= 180):
                raise ValueError('Longitude must be between -180 and 180')
        return v


class ELRChunk(BaseModel):
    """Chunked ELR content for embedding storage."""
    
    # Core content
    text: str = Field(..., description="Chunk text content")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    
    # Chunk metadata
    parent_item_id: str = Field(..., description="Parent ELR item ID")
    chunk_index: int = Field(..., ge=0, description="Index within parent item")
    total_chunks: int = Field(..., ge=1, description="Total chunks for parent item")
    
    # Content boundaries
    start_char: Optional[int] = Field(None, ge=0, description="Start character position")
    end_char: Optional[int] = Field(None, ge=0, description="End character position")
    
    # Inherited metadata
    user_id: str
    content_type: ELRContentType
    consent_level: ConsentLevel
    sensitivity_level: SensitivityLevel
    
    # Processing info
    created_at: datetime = Field(default_factory=datetime.utcnow)
    embedding_model: Optional[str] = Field(None, description="Model used for embedding")
    
    # Quality metrics
    chunk_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Source tracking
    source_file: Optional[str] = Field(None, description="Source file path")
    
    # Legacy support for existing code
    metadata: Dict[str, Union[str, int, float, List[str]]] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('end_char')
    def end_char_greater_than_start(cls, v, values):
        if v is not None and 'start_char' in values and values['start_char'] is not None:
            if v <= values['start_char']:
                raise ValueError('end_char must be greater than start_char')
        return v


class ELRSearchFilter(BaseModel):
    """Filter criteria for ELR search operations."""
    
    # User and consent filtering
    user_id: Optional[str] = None
    consent_levels: Optional[List[ConsentLevel]] = None
    
    # Content filtering
    content_types: Optional[List[ELRContentType]] = None
    sensitivity_levels: Optional[List[SensitivityLevel]] = None
    
    # Temporal filtering
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    event_date_from: Optional[datetime] = None
    event_date_to: Optional[datetime] = None
    
    # Text filtering
    tags: Optional[List[str]] = None
    entities: Optional[List[str]] = None
    
    # Quality filtering
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_completeness: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Relationship filtering
    parent_id: Optional[str] = None
    related_to: Optional[str] = None
    
    # Custom field filtering
    custom_filters: Dict[str, Any] = Field(default_factory=dict)


class ELRBatchOperation(BaseModel):
    """Batch operation request for multiple ELR items."""
    
    operation: str = Field(..., description="Operation type: create, update, delete")
    items: List[Union[ELRItem, str]] = Field(..., description="Items or IDs to operate on")
    
    # Batch options
    fail_on_error: bool = Field(True, description="Stop on first error")
    return_results: bool = Field(True, description="Return operation results")
    
    # Metadata for the batch
    batch_id: Optional[str] = Field(None, description="Batch operation ID")
    requested_by: Optional[str] = Field(None, description="User requesting the batch")
    
    @validator('operation')
    def validate_operation(cls, v):
        allowed_ops = {'create', 'update', 'delete', 'upsert'}
        if v not in allowed_ops:
            raise ValueError(f'Operation must be one of: {allowed_ops}')
        return v


@dataclass
class ELRProcessingResult:
    """Result of ELR processing operations."""
    
    success: bool
    processed_items: int = 0
    failed_items: int = 0
    chunks_created: int = 0
    chunks: List['ELRChunk'] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_seconds: Optional[float] = None
    
    # Detailed results
    created_item_ids: List[str] = field(default_factory=list)
    updated_item_ids: List[str] = field(default_factory=list)
    failed_item_ids: List[str] = field(default_factory=list)
