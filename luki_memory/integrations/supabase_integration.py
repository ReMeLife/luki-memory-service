#!/usr/bin/env python3
"""
Supabase Integration for Automatic ELR Ingestion
Handles user authentication and automatic ELR data ingestion based on user sessions.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from ..ingestion.pipeline import ELRPipeline
from ..storage.vector_store import EmbeddingStore
from ..api.auth import User
from ..ingestion.chunker import ELRChunk

logger = logging.getLogger(__name__)


class SupabaseConfig(BaseModel):
    """Supabase configuration."""
    url: str = Field(..., description="Supabase project URL")
    anon_key: str = Field(..., description="Supabase anonymous key")
    service_role_key: Optional[str] = Field(None, description="Supabase service role key")
    elr_table: str = Field("user_elr_data", description="ELR data table name")
    profiles_table: str = Field("profiles", description="User profiles table name")


class SupabaseSession(BaseModel):
    """Supabase user session."""
    access_token: str
    refresh_token: str
    user_id: str
    email: Optional[str] = None
    expires_at: datetime


class ELRIngestionEvent(BaseModel):
    """ELR ingestion event data."""
    user_id: str
    event_type: str = Field(..., description="Type of event: login, chat, update")
    elr_data: Dict[str, Any] = Field(..., description="ELR data to ingest")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field("supabase", description="Data source")


class SupabaseIntegration:
    """Handles Supabase integration for automatic ELR ingestion."""
    
    def __init__(
        self,
        config: SupabaseConfig,
        elr_pipeline: ELRPipeline,
        embedding_store: EmbeddingStore
    ):
        self.config = config
        self.pipeline = elr_pipeline
        self.store = embedding_store
        self.client = httpx.AsyncClient()
        
        # Track processed users to avoid duplicate ingestion
        self.processed_users = set()
        
        logger.info("Initialized Supabase integration")
    
    async def verify_user_session(self, access_token: str) -> Optional[SupabaseSession]:
        """Verify user session with Supabase."""
        try:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "apikey": self.config.anon_key,
                "Content-Type": "application/json"
            }
            
            response = await self.client.get(
                f"{self.config.url}/auth/v1/user",
                headers=headers
            )
            
            if response.status_code == 200:
                user_data = response.json()
                return SupabaseSession(
                    access_token=access_token,
                    refresh_token="",  # Not provided in user endpoint
                    user_id=user_data["id"],
                    email=user_data.get("email"),
                    expires_at=datetime.fromisoformat(user_data["created_at"])
                )
            else:
                logger.warning(f"Failed to verify session: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error verifying user session: {e}")
            return None
    
    async def fetch_user_elr_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch user's ELR data from Supabase."""
        try:
            headers = {
                "Authorization": f"Bearer {self.config.service_role_key}",
                "apikey": self.config.anon_key,
                "Content-Type": "application/json"
            }
            
            # Query ELR data table
            response = await self.client.get(
                f"{self.config.url}/rest/v1/{self.config.elr_table}",
                headers=headers,
                params={"user_id": f"eq.{user_id}"}
            )
            
            if response.status_code == 200:
                elr_records = response.json()
                if elr_records:
                    # Combine all ELR records for the user
                    combined_elr = self._combine_elr_records(elr_records)
                    logger.info(f"Fetched ELR data for user {user_id}: {len(elr_records)} records")
                    return combined_elr
                else:
                    logger.info(f"No ELR data found for user {user_id}")
                    return None
            else:
                logger.error(f"Failed to fetch ELR data: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching ELR data for user {user_id}: {e}")
            return None
    
    def _combine_elr_records(self, records: List[Dict]) -> Dict[str, Any]:
        """Combine multiple ELR records into a single ELR data structure."""
        combined = {
            "user_id": records[0].get("user_id"),
            "life_story": {},
            "memories": [],
            "preferences": {},
            "relationships": {},
            "health_data": {},
            "goals": [],
            "achievements": []
        }
        
        for record in records:
            data = record.get("elr_data", {})
            
            # Merge different sections
            if "life_story" in data:
                combined["life_story"].update(data["life_story"])
            
            if "memories" in data:
                combined["memories"].extend(data["memories"])
            
            if "preferences" in data:
                combined["preferences"].update(data["preferences"])
            
            if "relationships" in data:
                combined["relationships"].update(data["relationships"])
            
            if "health_data" in data:
                combined["health_data"].update(data["health_data"])
            
            if "goals" in data:
                combined["goals"].extend(data["goals"])
            
            if "achievements" in data:
                combined["achievements"].extend(data["achievements"])
        
        return combined
    
    async def ingest_user_elr(self, user_id: str, elr_data: Dict[str, Any]) -> bool:
        """Ingest ELR data for a user into the memory service."""
        try:
            logger.info(f"Starting ELR ingestion for user {user_id}")
            
            # Process ELR data through pipeline
            processing_result = self.pipeline.process_elr_data(elr_data)
            chunks = processing_result.chunks
            
            if not chunks:
                logger.warning(f"No chunks generated from ELR data for user {user_id}")
                return False
            
            # Add chunks to embedding store
            chunk_ids = self.store.add_chunks_batch(chunks)
            
            logger.info(f"Successfully ingested {len(chunk_ids)} chunks for user {user_id}")
            
            # Mark user as processed
            self.processed_users.add(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest ELR data for user {user_id}: {e}")
            return False
    
    async def handle_user_login(self, access_token: str) -> Optional[User]:
        """Handle user login and trigger ELR ingestion if needed."""
        try:
            # Verify session
            session = await self.verify_user_session(access_token)
            if not session:
                return None
            
            user_id = session.user_id
            
            # Check if we've already processed this user
            if user_id in self.processed_users:
                logger.info(f"User {user_id} already processed, skipping ingestion")
            else:
                # Fetch and ingest ELR data
                elr_data = await self.fetch_user_elr_data(user_id)
                if elr_data:
                    await self.ingest_user_elr(user_id, elr_data)
                else:
                    logger.info(f"No ELR data to ingest for user {user_id}")
                    # Still mark as processed to avoid repeated attempts
                    self.processed_users.add(user_id)
            
            # Return User object for authentication
            return User(
                user_id=user_id,
                email=session.email,
                full_name=None,  # Could fetch from profiles table
                is_active=True,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error handling user login: {e}")
            return None
    
    async def handle_elr_update(self, event: ELRIngestionEvent) -> bool:
        """Handle ELR data updates from Supabase webhooks."""
        try:
            logger.info(f"Handling ELR update event for user {event.user_id}")
            
            # Process new/updated ELR data
            success = await self.ingest_user_elr(event.user_id, event.elr_data)
            
            if success:
                logger.info(f"Successfully processed ELR update for user {event.user_id}")
            else:
                logger.error(f"Failed to process ELR update for user {event.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error handling ELR update: {e}")
            return False
    
    async def get_user_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user."""
        try:
            # Search for user's chunks to get count
            results = self.store.search_similar(
                query="",  # Empty query to get all
                k=1000,  # Large number to get all chunks
                metadata_filter={"user_id": user_id}
            )
            
            # Analyze chunks
            total_chunks = len(results)
            content_types = {}
            consent_levels = {}
            
            for result in results:
                metadata = result.get("metadata", {})
                
                content_type = metadata.get("content_type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                consent_level = metadata.get("consent_level", "unknown")
                consent_levels[consent_level] = consent_levels.get(consent_level, 0) + 1
            
            return {
                "user_id": user_id,
                "total_chunks": total_chunks,
                "content_types": content_types,
                "consent_levels": consent_levels,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats for user {user_id}: {e}")
            return {"error": str(e)}
    
    async def cleanup_user_data(self, user_id: str) -> bool:
        """Clean up all data for a user (GDPR compliance)."""
        try:
            logger.info(f"Starting data cleanup for user {user_id}")
            
            # Get all user chunks
            results = self.store.search_similar(
                query="",
                k=10000,  # Large number to get all chunks
                metadata_filter={"user_id": user_id}
            )
            
            # Delete chunks
            deleted_count = 0
            for result in results:
                chunk_id = result.get("id")
                if chunk_id and self.store.delete_chunk(chunk_id):
                    deleted_count += 1
            
            # Remove from processed users
            self.processed_users.discard(user_id)
            
            logger.info(f"Cleaned up {deleted_count} chunks for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up data for user {user_id}: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


def create_supabase_integration(
    supabase_url: str,
    supabase_anon_key: str,
    service_role_key: Optional[str] = None,
    elr_table: str = "user_elr_data",
    profiles_table: str = "profiles",
    elr_pipeline: Optional[ELRPipeline] = None,
    embedding_store: Optional[EmbeddingStore] = None
) -> SupabaseIntegration:
    """Factory function to create Supabase integration."""
    
    config = SupabaseConfig(
        url=supabase_url,
        anon_key=supabase_anon_key,
        service_role_key=service_role_key,
        elr_table=elr_table,
        profiles_table=profiles_table
    )
    
    # Create default instances if not provided
    if elr_pipeline is None:
        elr_pipeline = ELRPipeline()
    
    if embedding_store is None:
        embedding_store = EmbeddingStore()
    
    return SupabaseIntegration(config, elr_pipeline, embedding_store)


# Example usage and testing
async def test_supabase_integration():
    """Test function for Supabase integration."""
    # This would be used in actual implementation
    config = SupabaseConfig(
        url="https://your-project.supabase.co",
        anon_key="your-anon-key",
        service_role_key="your-service-role-key",
        elr_table="user_elr_data",
        profiles_table="profiles"
    )
    
    pipeline = ELRPipeline()
    store = EmbeddingStore()
    
    integration = SupabaseIntegration(config, pipeline, store)
    
    # Test user login flow
    # user = await integration.handle_user_login("test-access-token")
    # if user:
    #     print(f"User authenticated: {user.user_id}")
    #     stats = await integration.get_user_memory_stats(user.user_id)
    #     print(f"Memory stats: {stats}")
    
    await integration.close()


if __name__ == "__main__":
    # Run test
    asyncio.run(test_supabase_integration())
