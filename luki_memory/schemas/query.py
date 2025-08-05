"""
Query schema definitions for search requests and responses.

Defines data models for search operations, similarity queries, and result formatting.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from .elr import ConsentLevel, ELRContentType, SensitivityLevel


class QueryType(str, Enum):
    """Types of queries supported."""
    SIMILARITY = "similarity"      # Vector similarity search
    KEYWORD = "keyword"           # Text-based keyword search
    HYBRID = "hybrid"             # Combined similarity + keyword
    FILTER = "filter"             # Metadata filtering only
    AGGREGATE = "aggregate"       # Aggregation queries


class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


class SortField(str, Enum):
    """Fields available for sorting."""
    RELEVANCE = "relevance"       # Similarity score
    CREATED_AT = "created_at"     # Creation timestamp
    UPDATED_AT = "updated_at"     # Update timestamp
    EVENT_DATE = "event_date"     # Event occurrence date
    CONFIDENCE = "confidence"     # Content confidence score
    ACCESS_COUNT = "access_count" # Number of times accessed


class SearchQuery(BaseModel):
    """Search query request model."""
    
    # Query content
    query_text: str = Field(..., description="Search query text")
    query_type: QueryType = Field(QueryType.SIMILARITY, description="Type of query")
    
    # Result parameters
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Results offset for pagination")
    
    # Similarity parameters
    similarity_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Minimum similarity score")
    
    # Filtering
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    consent_levels: Optional[List[ConsentLevel]] = Field(None, description="Allowed consent levels")
    content_types: Optional[List[ELRContentType]] = Field(None, description="Filter by content types")
    sensitivity_levels: Optional[List[SensitivityLevel]] = Field(None, description="Filter by sensitivity")
    
    # Temporal filtering
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")
    
    # Metadata filtering
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    entities: Optional[List[str]] = Field(None, description="Filter by entities")
    custom_filters: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata filters")
    
    # Sorting
    sort_by: SortField = Field(SortField.RELEVANCE, description="Sort field")
    sort_order: SortOrder = Field(SortOrder.DESC, description="Sort order")
    
    # Advanced options
    include_metadata: bool = Field(True, description="Include item metadata in results")
    include_chunks: bool = Field(False, description="Include chunk information")
    highlight_matches: bool = Field(False, description="Highlight matching text")
    
    # Query metadata
    query_id: Optional[str] = Field(None, description="Unique query identifier")
    requested_by: Optional[str] = Field(None, description="User making the request")
    
    @validator('query_text')
    def validate_query_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Query text cannot be empty')
        return v.strip()


class SearchResult(BaseModel):
    """Individual search result item."""
    
    # Item identification
    item_id: str = Field(..., description="Unique item identifier")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier if applicable")
    
    # Content
    content: str = Field(..., description="Result content text")
    title: Optional[str] = Field(None, description="Result title")
    
    # Relevance scoring
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance/similarity score")
    rank: int = Field(..., ge=1, description="Result rank in the result set")
    
    # Metadata
    content_type: ELRContentType = Field(..., description="Type of content")
    consent_level: ConsentLevel = Field(..., description="Consent level")
    sensitivity_level: SensitivityLevel = Field(..., description="Sensitivity level")
    
    # Temporal information
    created_at: datetime = Field(..., description="Creation timestamp")
    event_date: Optional[datetime] = Field(None, description="Event date if applicable")
    
    # Additional metadata (optional)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    # Highlighting (if requested)
    highlighted_content: Optional[str] = Field(None, description="Content with highlights")
    matched_terms: Optional[List[str]] = Field(None, description="Terms that matched")
    
    # Chunk information (if applicable)
    chunk_index: Optional[int] = Field(None, description="Chunk index within parent")
    total_chunks: Optional[int] = Field(None, description="Total chunks in parent item")
    
    # Context
    context_before: Optional[str] = Field(None, description="Text before the match")
    context_after: Optional[str] = Field(None, description="Text after the match")


class SearchResponse(BaseModel):
    """Search query response model."""
    
    # Query information
    query_id: Optional[str] = Field(None, description="Query identifier")
    query_text: str = Field(..., description="Original query text")
    query_type: QueryType = Field(..., description="Type of query executed")
    
    # Results
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., ge=0, description="Total matching results")
    returned_results: int = Field(..., ge=0, description="Number of results returned")
    
    # Pagination
    offset: int = Field(..., ge=0, description="Results offset")
    limit: int = Field(..., ge=1, description="Results limit")
    has_more: bool = Field(..., description="More results available")
    
    # Performance metrics
    execution_time_ms: float = Field(..., ge=0, description="Query execution time")
    
    # Query analysis
    processed_query: Optional[str] = Field(None, description="Processed/normalized query")
    query_terms: Optional[List[str]] = Field(None, description="Extracted query terms")
    
    # Aggregations (if requested)
    aggregations: Optional[Dict[str, Any]] = Field(None, description="Aggregation results")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    @validator('returned_results')
    def validate_returned_results(cls, v, values):
        if 'results' in values and len(values['results']) != v:
            raise ValueError('returned_results must match the length of results list')
        return v


class AggregationQuery(BaseModel):
    """Aggregation query for analytics."""
    
    # Base query (optional)
    base_query: Optional[SearchQuery] = Field(None, description="Base query to aggregate over")
    
    # Aggregation specifications
    group_by: List[str] = Field(..., description="Fields to group by")
    aggregations: Dict[str, str] = Field(..., description="Aggregation functions")
    
    # Filtering
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")
    
    # Time-based aggregation
    time_bucket: Optional[str] = Field(None, description="Time bucket size (hour, day, week, month)")
    time_field: str = Field("created_at", description="Time field for bucketing")
    
    # Result options
    limit: int = Field(100, ge=1, le=1000, description="Maximum aggregation results")
    min_doc_count: int = Field(1, ge=0, description="Minimum document count per bucket")


class QuerySuggestion(BaseModel):
    """Query suggestion for autocomplete/search assistance."""
    
    suggestion: str = Field(..., description="Suggested query text")
    score: float = Field(..., ge=0.0, le=1.0, description="Suggestion relevance score")
    type: str = Field(..., description="Type of suggestion (completion, correction, related)")
    
    # Context
    matched_prefix: Optional[str] = Field(None, description="Prefix that was matched")
    category: Optional[str] = Field(None, description="Suggestion category")
    
    # Metadata
    frequency: Optional[int] = Field(None, description="How often this query is used")
    last_used: Optional[datetime] = Field(None, description="When this query was last used")


class QueryAnalytics(BaseModel):
    """Analytics data for query performance and usage."""
    
    # Query identification
    query_text: str = Field(..., description="Query text")
    query_hash: str = Field(..., description="Query hash for deduplication")
    
    # Performance metrics
    execution_time_ms: float = Field(..., ge=0, description="Execution time")
    results_count: int = Field(..., ge=0, description="Number of results returned")
    
    # User interaction
    user_id: Optional[str] = Field(None, description="User who made the query")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    # Result interaction
    clicked_results: List[int] = Field(default_factory=list, description="Ranks of clicked results")
    time_to_first_click_ms: Optional[float] = Field(None, description="Time to first result click")
    
    # Query context
    query_type: QueryType = Field(..., description="Type of query")
    filters_used: Dict[str, Any] = Field(default_factory=dict, description="Filters applied")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Query timestamp")
    
    # Quality metrics
    user_satisfaction: Optional[float] = Field(None, ge=0.0, le=1.0, description="User satisfaction score")
    query_success: bool = Field(True, description="Whether query was successful")


class BatchSearchQuery(BaseModel):
    """Batch search query for multiple queries."""
    
    queries: List[SearchQuery] = Field(..., description="List of search queries")
    
    # Batch options
    parallel_execution: bool = Field(True, description="Execute queries in parallel")
    fail_on_error: bool = Field(False, description="Stop on first query error")
    
    # Shared parameters (override individual query settings)
    shared_filters: Optional[Dict[str, Any]] = Field(None, description="Filters applied to all queries")
    shared_limit: Optional[int] = Field(None, description="Limit applied to all queries")
    
    # Batch metadata
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    timeout_seconds: Optional[int] = Field(60, description="Total batch timeout")


class BatchSearchResponse(BaseModel):
    """Response for batch search queries."""
    
    # Batch information
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    total_queries: int = Field(..., ge=0, description="Total number of queries")
    successful_queries: int = Field(..., ge=0, description="Number of successful queries")
    failed_queries: int = Field(..., ge=0, description="Number of failed queries")
    
    # Results
    responses: List[Union[SearchResponse, Dict[str, str]]] = Field(..., description="Individual query responses or errors")
    
    # Performance
    total_execution_time_ms: float = Field(..., ge=0, description="Total batch execution time")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Batch completion timestamp")
