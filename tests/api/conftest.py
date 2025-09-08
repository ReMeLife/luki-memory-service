#!/usr/bin/env python3
"""
Pytest configuration for API integration tests
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime
import tempfile
import shutil
import os

from luki_memory.api.main import app
from luki_memory.api.auth import create_access_token
from luki_memory.ingestion.synthetic_elr_generator import create_synthetic_elr_generator


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = Mock()
    mock_store.add_chunks.return_value = None
    mock_store.search.return_value = []
    mock_store.get_collection_stats.return_value = {"total_chunks": 0}
    return mock_store


@pytest.fixture
def mock_pipeline():
    """Mock ELR pipeline for testing."""
    mock_pipeline = Mock()
    
    # Mock process_elr_data method
    from luki_memory.schemas.elr import ELRProcessingResult
    mock_result = ELRProcessingResult(
        success=True,
        processed_items=1,
        failed_items=0,
        chunks_created=3,
        errors=[],
        warnings=[],
        processing_time_seconds=0.1,
        created_item_ids=["test_item_1"],
        updated_item_ids=[],
        failed_item_ids=[]
    )
    mock_pipeline.process_elr_data.return_value = mock_result
    
    # Mock search method
    mock_pipeline.search.return_value = []
    
    return mock_pipeline


@pytest.fixture
def mock_embedding_store():
    """Mock embedding store for project knowledge."""
    mock_store = Mock()
    mock_store.search.return_value = []
    return mock_store


@pytest.fixture
def test_client(mock_pipeline, mock_vector_store, mock_embedding_store):
    """Create test client with mocked dependencies."""
    
    # Patch the dependencies used in startup_event and inject pipeline
    with patch('luki_memory.storage.knowledge_store.get_project_knowledge_store') as mock_get_project_store, \
         patch('luki_memory.storage.knowledge_store.initialize_project_knowledge') as mock_init_knowledge, \
         patch('luki_memory.api.main.load_project_context_into_store') as mock_load_context, \
         patch('luki_memory.storage.elr_store.get_elr_store') as mock_get_elr_store, \
         patch('luki_memory.api.endpoints.search.set_project_knowledge_store') as mock_set_project_store, \
         patch('luki_memory.api.endpoints.search.set_elr_store') as mock_set_elr_store:
        
        # Configure mocks
        mock_get_project_store.return_value = mock_embedding_store
        mock_get_elr_store.return_value = mock_vector_store
        mock_load_context.return_value = None
        
        # Inject mocked pipeline into endpoints
        from luki_memory.api.endpoints import ingestion, search
        ingestion.pipeline = mock_pipeline
        search.elr_store = mock_vector_store
        search.project_knowledge_store = mock_embedding_store
        search.pipeline = mock_pipeline
        
        # Create test client
        client = TestClient(app)
        
        yield client


@pytest.fixture
def auth_token():
    """Create test authentication token."""
    token_data = {
        "sub": "test_user_123",
        "email": "test@example.com",
        "full_name": "Test User"
    }
    return create_access_token(token_data)


@pytest.fixture
def auth_headers(auth_token):
    """Create authentication headers."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def sample_elr_data():
    """Create sample ELR data for testing."""
    generator = create_synthetic_elr_generator(seed=42)
    return generator.generate_person_elr()


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)
