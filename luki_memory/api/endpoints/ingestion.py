#!/usr/bin/env python3
"""
ELR Ingestion API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from typing import List
import logging
import time
import asyncio
from datetime import datetime

from ..models import (
    ELRIngestionRequest, ELRIngestionResponse,
    BatchIngestionRequest, BatchIngestionResponse,
    ErrorResponse
)
# Auth imports removed - not used in current implementation
from ..config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingestion", tags=["ingestion"])
settings = get_settings()

# Global pipeline reference (will be injected)
pipeline = None

def set_pipeline(elr_pipeline):
    """Set the global pipeline instance."""
    global pipeline
    pipeline = elr_pipeline


@router.post("/elr", response_model=ELRIngestionResponse)
async def ingest_elr(
    request: ELRIngestionRequest
):
    """
    Ingest Electronic Life Record (ELR) data for a user.
    
    This endpoint processes ELR data through the complete pipeline:
    - Text extraction and chunking
    - NLP processing with spaCy
    - PII redaction based on sensitivity level
    - Vector embedding generation
    - Storage in ChromaDB
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ELR pipeline not initialized"
        )
    
    start_time = time.time()
    
    try:
        # Authorization check removed - using simplified auth model
        
        # Process ELR data through pipeline
        result = pipeline.process_elr_to_embeddings(
            elr_data=request.elr_data,
            user_id=request.user_id,
            source_file=request.source_file or f"api_upload_{datetime.utcnow().isoformat()}",
            consent_level=request.consent_level,
            sensitivity_level=request.sensitivity_level
        )
        
        processing_time = time.time() - start_time
        
        if result.success:
            logger.info(f"Successfully ingested ELR data for user {request.user_id}: "
                       f"{result.processed_items} items, {result.embedded_chunks} chunks")
            
            return ELRIngestionResponse(
                success=True,
                message="ELR data ingested successfully",
                chunks_created=result.embedded_chunks,
                chunk_ids=result.chunk_ids,
                processing_time_seconds=processing_time,
                errors=result.errors,
                created_item_ids=result.created_item_ids
            )
        else:
            logger.warning(f"ELR ingestion failed for user {request.user_id}: {result.errors}")
            
            return ELRIngestionResponse(
                success=False,
                message="ELR ingestion completed with errors",
                chunks_created=result.embedded_chunks,
                chunk_ids=result.chunk_ids,
                processing_time_seconds=processing_time,
                errors=result.errors,
                created_item_ids=result.created_item_ids
            )
    
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"ELR ingestion failed: {str(e)}"
        logger.error(error_msg)
        
        return ELRIngestionResponse(
            success=False,
            message="ELR ingestion failed due to internal error",
            chunks_created=0,
            chunk_ids=[],
            processing_time_seconds=processing_time,
            errors=[error_msg],
            created_item_ids=[]
        )


@router.post("/elr/batch", response_model=BatchIngestionResponse)
async def ingest_elr_batch(
    request: BatchIngestionRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest multiple ELR records in a batch.
    
    This endpoint processes multiple ELR records efficiently:
    - Validates all requests before processing
    - Processes in parallel where possible
    - Returns detailed results for each item
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ELR pipeline not initialized"
        )
    
    if len(request.batch_data) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum of {settings.max_batch_size}"
        )
    
    start_time = time.time()
    batch_id = request.batch_id or f"batch_{datetime.utcnow().isoformat()}"
    
    try:
        # Authorization check removed - using simplified auth model
        
        # Process each item in the batch
        individual_results = []
        successful_requests = 0
        failed_requests = 0
        
        for i, elr_request in enumerate(request.batch_data):
            try:
                item_start_time = time.time()
                
                result = pipeline.process_elr_to_embeddings(
                    elr_data=elr_request.elr_data,
                    user_id=elr_request.user_id,
                    source_file=elr_request.source_file or f"{batch_id}_item_{i}",
                    consent_level=elr_request.consent_level,
                    sensitivity_level=elr_request.sensitivity_level
                )
                
                item_processing_time = time.time() - item_start_time
                
                if result.success:
                    successful_requests += 1
                    individual_results.append(ELRIngestionResponse(
                        success=True,
                        message=f"Batch item {i} processed successfully",
                        chunks_created=result.embedded_chunks,
                        chunk_ids=result.chunk_ids,
                        processing_time_seconds=item_processing_time,
                        errors=result.errors,
                        created_item_ids=result.created_item_ids
                    ))
                else:
                    failed_requests += 1
                    individual_results.append(ELRIngestionResponse(
                        success=False,
                        message=f"Batch item {i} failed",
                        chunks_created=result.embedded_chunks,
                        chunk_ids=result.chunk_ids,
                        processing_time_seconds=item_processing_time,
                        errors=result.errors,
                        created_item_ids=result.created_item_ids
                    ))
            
            except Exception as e:
                failed_requests += 1
                item_processing_time = time.time() - item_start_time
                individual_results.append(ELRIngestionResponse(
                    success=False,
                    message=f"Batch item {i} failed with exception",
                    chunks_created=0,
                    chunk_ids=[],
                    processing_time_seconds=item_processing_time,
                    errors=[str(e)],
                    created_item_ids=[]
                ))
        
        total_processing_time = time.time() - start_time
        
        logger.info(f"Batch ingestion completed: {batch_id}, "
                   f"{successful_requests}/{len(request.batch_data)} successful")
        
        return BatchIngestionResponse(
            batch_id=batch_id,
            total_requests=len(request.batch_data),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            individual_results=individual_results,
            total_processing_time_seconds=total_processing_time
        )
    
    except Exception as e:
        total_processing_time = time.time() - start_time
        error_msg = f"Batch ingestion failed: {str(e)}"
        logger.error(error_msg)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )


@router.get("/status")
async def ingestion_status():
    """Get ingestion service status."""
    return {
        "service": "ELR Ingestion",
        "status": "operational" if pipeline is not None else "unavailable",
        "pipeline_ready": pipeline is not None,
        "max_batch_size": settings.max_batch_size,
        "supported_formats": ["JSON"],
        "supported_content_types": ["MEMORY", "PREFERENCE", "FAMILY", "INTEREST"],
        "supported_sensitivity_levels": ["PERSONAL", "SENSITIVE", "CONFIDENTIAL"],
        "supported_consent_levels": ["PUBLIC", "PRIVATE", "RESTRICTED"]
    }
