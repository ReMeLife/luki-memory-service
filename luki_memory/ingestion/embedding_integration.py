"""
ELR Ingestion to Vector Embedding Integration

Connects the enhanced ELR ingestion pipeline to the vector embedding system
with automatic chunking, indexing, and metadata tagging for consent/privacy.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from ..schemas.elr import ELRItem, ELRProcessingResult, ConsentLevel, SensitivityLevel
from ..storage.vector_store import EmbeddingStore, create_embedding_store
from .elr_ingestion import AdvancedELRIngestionPipeline, create_advanced_pipeline
from .chunker import ELRChunk, create_chunker
from .redact import create_redactor

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingIntegrationResult:
    """Result of ELR ingestion to embedding integration."""
    success: bool
    processed_items: int = 0
    embedded_chunks: int = 0
    failed_embeddings: int = 0
    errors: List[str] = field(default_factory=list)
    processing_time_seconds: Optional[float] = None
    chunk_ids: List[str] = field(default_factory=list)
    created_item_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.chunk_ids is None:
            self.chunk_ids = []
        if self.created_item_ids is None:
            self.created_item_ids = []


class ELRToVectorPipeline:
    """Complete pipeline from ELR data to vector embeddings."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L12-v2",
        spacy_model: str = "en_core_web_lg",
        persist_directory: str = "./chroma_db",
        collection_name: str = "elr_embeddings",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """Initialize the complete ELR to vector pipeline."""
        self.embedding_model = embedding_model
        self.spacy_model = spacy_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.elr_pipeline = create_advanced_pipeline(spacy_model)
        self.embedding_store = create_embedding_store(
            model_name=embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        self.chunker = create_chunker(spacy_model)
        self.redactor = create_redactor(spacy_model)
        
        logger.info(f"Initialized ELR to Vector pipeline with {embedding_model} embeddings")
    
    def process_elr_to_embeddings(
        self,
        elr_data: Dict,
        user_id: str,
        source_file: str = "",
        consent_level: ConsentLevel = ConsentLevel.PRIVATE,
        sensitivity_level: SensitivityLevel = SensitivityLevel.PERSONAL
    ) -> EmbeddingIntegrationResult:
        """
        Complete pipeline: ELR data → structured items → chunks → embeddings.
        
        Args:
            elr_data: Raw ELR data dictionary
            user_id: User identifier
            source_file: Source file name
            consent_level: Default consent level for items
            sensitivity_level: Default sensitivity level for items
            
        Returns:
            Integration result with embedding statistics
        """
        start_time = datetime.now()
        errors = []
        chunk_ids = []
        
        try:
            # Step 1: Process ELR data into structured items
            logger.info("Step 1: Processing ELR data into structured items")
            processing_result = self.elr_pipeline.process_elr_data_enhanced(
                elr_data, user_id, source_file
            )
            
            if not processing_result.success:
                errors.extend(processing_result.errors)
                return EmbeddingIntegrationResult(
                    success=False,
                    errors=errors,
                    processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                    created_item_ids=[]
                )
            
            # Step 2: Convert ELR items to chunks with privacy controls
            logger.info("Step 2: Converting ELR items to chunks with privacy controls")
            all_chunks = []
            
            # Get ELR items from the processing result (we need to modify the pipeline to return items)
            elr_items = self._extract_items_from_processing_result(elr_data, user_id, source_file)
            
            for item in elr_items:
                try:
                    chunks = self._convert_item_to_chunks(
                        item, consent_level, sensitivity_level
                    )
                    all_chunks.extend(chunks)
                except Exception as e:
                    error_msg = f"Error converting item to chunks: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Step 3: Apply privacy redaction to chunks
            logger.info("Step 3: Applying privacy redaction to chunks")
            redacted_chunks = []
            for chunk in all_chunks:
                try:
                    redacted_chunk = self._apply_privacy_redaction(chunk)
                    redacted_chunks.append(redacted_chunk)
                except Exception as e:
                    error_msg = f"Error applying redaction to chunk: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Step 4: Generate embeddings and store in vector database
            logger.info("Step 4: Generating embeddings and storing in vector database")
            if redacted_chunks:
                try:
                    stored_chunk_ids = self.embedding_store.add_chunks_batch(redacted_chunks)
                    chunk_ids.extend(stored_chunk_ids)
                except Exception as e:
                    error_msg = f"Error storing chunks in vector database: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EmbeddingIntegrationResult(
                success=len(errors) == 0,
                processed_items=len(elr_items),
                embedded_chunks=len(chunk_ids),
                failed_embeddings=len(all_chunks) - len(chunk_ids),
                errors=errors,
                processing_time_seconds=processing_time,
                chunk_ids=chunk_ids,
                created_item_ids=[item.id or str(uuid.uuid4()) for item in elr_items]
            )
            
        except Exception as e:
            error_msg = f"Pipeline error: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return EmbeddingIntegrationResult(
                success=False,
                errors=errors,
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                created_item_ids=[]
            )
    
    def _extract_items_from_processing_result(
        self, elr_data: Dict, user_id: str, source_file: str
    ) -> List[ELRItem]:
        """Extract ELR items from processing result (temporary implementation)."""
        # This is a simplified version - in a full implementation, 
        # we'd modify the pipeline to return the actual ELR items
        items = []
        
        # Process life story section
        if "life_story" in elr_data:
            items.extend(
                self.elr_pipeline.process_life_story_section(
                    elr_data["life_story"], user_id, source_file
                )
            )
        
        return items
    
    def _convert_item_to_chunks(
        self,
        item: ELRItem,
        consent_level: ConsentLevel,
        sensitivity_level: SensitivityLevel
    ) -> List[ELRChunk]:
        """Convert an ELR item to chunks for embedding."""
        chunks = []
        
        # Create metadata for chunking
        chunk_metadata = {
            "user_id": item.metadata.user_id,
            "content_type": item.metadata.content_type.value,
            "source_file": item.metadata.source_file or "",
            "consent_level": consent_level.value,
            "sensitivity_level": sensitivity_level.value,
            "tags": item.metadata.tags,
            "entities": item.metadata.entities,
            "item_id": item.id or str(uuid.uuid4()),
            "title": item.title or "",
            "event_date": item.event_date.isoformat() if item.event_date else None,
            "location": item.location or "",
        }
        
        # Add custom fields to metadata
        if item.custom_fields:
            for key, value in item.custom_fields.items():
                if isinstance(value, (str, int, float, bool)):
                    chunk_metadata[f"custom_{key}"] = str(value)
        
        # Chunk the content
        text_chunks = self.chunker.chunk_text(
            text=item.content, 
            metadata=chunk_metadata,
            parent_item_id=item.id or str(uuid.uuid4()),
            user_id=item.metadata.user_id
        )
        
        # The chunker now returns properly structured ELRChunk objects
        # Update them with additional metadata
        for text_chunk in text_chunks:
            # Update the chunk with additional metadata
            text_chunk.consent_level = consent_level
            text_chunk.sensitivity_level = sensitivity_level
            text_chunk.embedding_model = self.embedding_model
            text_chunk.chunk_quality_score = self._calculate_chunk_quality(text_chunk.text)
            chunks.append(text_chunk)
        
        return chunks
    
    def _calculate_chunk_quality(self, text: str) -> float:
        """Calculate quality score for a chunk."""
        # Simple quality metrics
        word_count = len(text.split())
        char_count = len(text)
        
        # Base quality
        quality = 0.5
        
        # Prefer chunks with reasonable length
        if 50 <= word_count <= 200:
            quality += 0.2
        elif word_count < 10:
            quality -= 0.3
        
        # Prefer chunks with good character density
        if char_count > 0:
            avg_word_length = char_count / max(word_count, 1)
            if 4 <= avg_word_length <= 8:
                quality += 0.1
        
        # Check for meaningful content (not just punctuation/spaces)
        meaningful_chars = sum(1 for c in text if c.isalnum())
        if meaningful_chars / max(char_count, 1) > 0.7:
            quality += 0.2
        
        return min(max(quality, 0.0), 1.0)
    
    def _apply_privacy_redaction(self, chunk: ELRChunk) -> ELRChunk:
        """Apply privacy redaction to a chunk based on sensitivity level."""
        redacted_text = chunk.text
        
        # Apply redaction based on sensitivity level
        if chunk.sensitivity_level in [SensitivityLevel.SENSITIVE, SensitivityLevel.CONFIDENTIAL]:
            # Use the redactor to remove PII
            redacted_text = self.redactor.redact_pii(chunk.text)
            
            # For confidential content, apply additional redaction
            if chunk.sensitivity_level == SensitivityLevel.CONFIDENTIAL:
                redacted_text = self.redactor.redact_sensitive_terms(redacted_text)
        
        # Create new chunk with redacted content
        redacted_chunk = ELRChunk(
            text=redacted_text,
            chunk_id=chunk.chunk_id,
            parent_item_id=chunk.parent_item_id,
            chunk_index=chunk.chunk_index,
            total_chunks=chunk.total_chunks,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            user_id=chunk.user_id,
            content_type=chunk.content_type,
            consent_level=chunk.consent_level,
            sensitivity_level=chunk.sensitivity_level,
            created_at=chunk.created_at,
            embedding_model=chunk.embedding_model,
            chunk_quality_score=chunk.chunk_quality_score
        )
        
        return redacted_chunk
    
    def search_user_memories(
        self,
        user_id: str,
        query: str,
        k: int = 5,
        consent_levels: Optional[List[ConsentLevel]] = None,
        content_types: Optional[List[str]] = None,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search user memories using vector similarity.
        
        Args:
            user_id: User identifier
            query: Search query
            k: Number of results to return
            consent_levels: Allowed consent levels
            content_types: Filter by content types
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar memory chunks
        """
        try:
            # Prepare metadata filters
            metadata_filter = {"user_id": user_id}
            
            if content_types:
                # Note: ChromaDB doesn't support complex OR queries easily
                # This is a simplified implementation
                pass
            
            # Prepare consent filter
            consent_filter = None
            if consent_levels:
                consent_filter = [level.value for level in consent_levels]
            
            # Search in vector store
            results = self.embedding_store.search_similar(
                query=query,
                k=k,
                similarity_threshold=similarity_threshold,
                consent_filter=consent_filter,
                metadata_filter=metadata_filter
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching user memories: {e}")
            return []
    
    def get_user_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about a user's stored memories."""
        try:
            # This is a simplified implementation
            # In a full implementation, we'd query the vector store for user-specific stats
            collection_stats = self.embedding_store.get_collection_stats()
            
            return {
                "user_id": user_id,
                "total_chunks_in_collection": collection_stats.get("total_chunks", 0),
                "embedding_model": collection_stats.get("model_name", "unknown"),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user memory stats: {e}")
            return {}


def create_elr_to_vector_pipeline(
    embedding_model: str = "all-MiniLM-L12-v2",
    spacy_model: str = "en_core_web_lg",
    persist_directory: str = "./chroma_db",
    collection_name: str = "elr_embeddings"
) -> ELRToVectorPipeline:
    """Factory function to create ELR to vector pipeline."""
    return ELRToVectorPipeline(
        embedding_model=embedding_model,
        spacy_model=spacy_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )


# Async wrapper for batch processing
class AsyncELRProcessor:
    """Async wrapper for processing multiple ELR files."""
    
    def __init__(self, pipeline: ELRToVectorPipeline):
        self.pipeline = pipeline
    
    async def process_elr_files_batch(
        self,
        file_paths: List[str],
        user_ids: List[str],
        max_concurrent: int = 3
    ) -> List[EmbeddingIntegrationResult]:
        """Process multiple ELR files concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_file(file_path: str, user_id: str):
            async with semaphore:
                # Load ELR data (this would be async in a real implementation)
                elr_data = self.pipeline.elr_pipeline.load_elr_file(file_path)
                
                # Process to embeddings
                return self.pipeline.process_elr_to_embeddings(
                    elr_data, user_id, file_path
                )
        
        tasks = [
            process_single_file(file_path, user_id)
            for file_path, user_id in zip(file_paths, user_ids)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(
                    EmbeddingIntegrationResult(
                        success=False,
                        errors=[str(result)]
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results
