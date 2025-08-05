"""
Key-Value store implementation with PostgreSQL and Redis adapters.

Provides persistent and ephemeral key-value storage for user preferences,
session data, caching, and metadata.
"""

import json
import pickle
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..schemas.kv import (
    KVItem, SessionData, CacheEntry, KVDataType, KVNamespace,
    KVBatchOperation, KVSearchFilter, KVOperationResult, UserPreferences
)
from ..config import DatabaseConfig

logger = logging.getLogger(__name__)


class KVStoreInterface(ABC):
    """Abstract interface for key-value storage implementations."""
    
    @abstractmethod
    async def get(self, key: str, namespace: KVNamespace = KVNamespace.CACHE) -> Optional[KVItem]:
        """Get a value by key."""
        pass
    
    @abstractmethod
    async def set(self, item: KVItem) -> bool:
        """Set a key-value pair."""
        pass
    
    @abstractmethod
    async def delete(self, key: str, namespace: KVNamespace = KVNamespace.CACHE) -> bool:
        """Delete a key-value pair."""
        pass
    
    @abstractmethod
    async def exists(self, key: str, namespace: KVNamespace = KVNamespace.CACHE) -> bool:
        """Check if a key exists."""
        pass
    
    @abstractmethod
    async def search(self, filter_criteria: KVSearchFilter) -> List[KVItem]:
        """Search for key-value pairs matching criteria."""
        pass
    
    @abstractmethod
    async def batch_operation(self, operation: KVBatchOperation) -> List[KVOperationResult]:
        """Execute batch operations."""
        pass
    
    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired items and return count of removed items."""
        pass


class MemoryKVStore(KVStoreInterface):
    """In-memory key-value store implementation for development and testing."""
    
    def __init__(self):
        self._data: Dict[str, Dict[str, KVItem]] = {}
        self._setup_namespaces()
    
    def _setup_namespaces(self):
        """Initialize namespace storage."""
        for namespace in KVNamespace:
            self._data[namespace.value] = {}
    
    def _get_full_key(self, key: str, namespace: KVNamespace) -> str:
        """Get the full key including namespace."""
        return f"{namespace.value}:{key}"
    
    async def get(self, key: str, namespace: KVNamespace = KVNamespace.CACHE) -> Optional[KVItem]:
        """Get a value by key."""
        try:
            item = self._data[namespace.value].get(key)
            if item and item.is_expired():
                await self.delete(key, namespace)
                return None
            
            if item:
                item.accessed_at = datetime.utcnow()
                item.access_count += 1
            
            return item
        except Exception as e:
            logger.error(f"Error getting key {key} from namespace {namespace}: {e}")
            return None
    
    async def set(self, item: KVItem) -> bool:
        """Set a key-value pair."""
        try:
            # Set TTL expiration if specified
            if item.ttl_seconds and not item.expires_at:
                item.set_ttl(item.ttl_seconds)
            
            item.updated_at = datetime.utcnow()
            self._data[item.namespace.value][item.key] = item
            return True
        except Exception as e:
            logger.error(f"Error setting key {item.key}: {e}")
            return False
    
    async def delete(self, key: str, namespace: KVNamespace = KVNamespace.CACHE) -> bool:
        """Delete a key-value pair."""
        try:
            if key in self._data[namespace.value]:
                del self._data[namespace.value][key]
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
    
    async def exists(self, key: str, namespace: KVNamespace = KVNamespace.CACHE) -> bool:
        """Check if a key exists."""
        try:
            item = self._data[namespace.value].get(key)
            if item and item.is_expired():
                await self.delete(key, namespace)
                return False
            return item is not None
        except Exception as e:
            logger.error(f"Error checking existence of key {key}: {e}")
            return False
    
    async def search(self, filter_criteria: KVSearchFilter) -> List[KVItem]:
        """Search for key-value pairs matching criteria."""
        results = []
        
        try:
            namespaces = filter_criteria.namespaces or list(KVNamespace)
            
            for namespace in namespaces:
                for key, item in self._data[namespace.value].items():
                    if self._matches_filter(item, filter_criteria):
                        results.append(item)
            
            # Apply limit and offset
            if filter_criteria.offset:
                results = results[filter_criteria.offset:]
            if filter_criteria.limit:
                results = results[:filter_criteria.limit]
            
            return results
        except Exception as e:
            logger.error(f"Error searching key-value store: {e}")
            return []
    
    def _matches_filter(self, item: KVItem, filter_criteria: KVSearchFilter) -> bool:
        """Check if an item matches the filter criteria."""
        # Skip expired items unless explicitly included
        if not filter_criteria.include_expired and item.is_expired():
            return False
        
        # Key pattern matching
        if filter_criteria.key_pattern:
            import fnmatch
            if not fnmatch.fnmatch(item.key, filter_criteria.key_pattern):
                return False
        
        # Key prefix matching
        if filter_criteria.key_prefix and not item.key.startswith(filter_criteria.key_prefix):
            return False
        
        # User ID filtering
        if filter_criteria.user_id and item.user_id != filter_criteria.user_id:
            if not (filter_criteria.include_public and item.is_public):
                return False
        
        # Data type filtering
        if filter_criteria.data_types and item.data_type not in filter_criteria.data_types:
            return False
        
        # Tag filtering
        if filter_criteria.tags:
            if not any(tag in item.tags for tag in filter_criteria.tags):
                return False
        
        # Time filtering
        if filter_criteria.created_after and item.created_at < filter_criteria.created_after:
            return False
        if filter_criteria.created_before and item.created_at > filter_criteria.created_before:
            return False
        
        return True
    
    async def batch_operation(self, operation: KVBatchOperation) -> List[KVOperationResult]:
        """Execute batch operations."""
        results = []
        
        for item in operation.items:
            try:
                if operation.operation == "get":
                    key = item if isinstance(item, str) else item.key
                    namespace = operation.namespace or KVNamespace.CACHE
                    value = await self.get(key, namespace)
                    results.append(KVOperationResult(
                        success=value is not None,
                        operation="get",
                        key=key,
                        value=value
                    ))
                
                elif operation.operation == "set":
                    if isinstance(item, str):
                        results.append(KVOperationResult(
                            success=False,
                            operation="set",
                            error="Set operation requires KVItem object"
                        ))
                    else:
                        success = await self.set(item)
                        results.append(KVOperationResult(
                            success=success,
                            operation="set",
                            key=item.key
                        ))
                
                elif operation.operation == "delete":
                    key = item if isinstance(item, str) else item.key
                    namespace = operation.namespace or KVNamespace.CACHE
                    success = await self.delete(key, namespace)
                    results.append(KVOperationResult(
                        success=success,
                        operation="delete",
                        key=key
                    ))
                
                elif operation.operation == "exists":
                    key = item if isinstance(item, str) else item.key
                    namespace = operation.namespace or KVNamespace.CACHE
                    exists = await self.exists(key, namespace)
                    results.append(KVOperationResult(
                        success=True,
                        operation="exists",
                        key=key,
                        value=exists
                    ))
                
            except Exception as e:
                results.append(KVOperationResult(
                    success=False,
                    operation=operation.operation,
                    error=str(e)
                ))
                
                if operation.fail_on_error:
                    break
        
        return results
    
    async def cleanup_expired(self) -> int:
        """Clean up expired items and return count of removed items."""
        removed_count = 0
        
        try:
            for namespace_name, namespace_data in self._data.items():
                expired_keys = []
                for key, item in namespace_data.items():
                    if item.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del namespace_data[key]
                    removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} expired items")
            return removed_count
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0


class SessionStore:
    """Session management store for user sessions."""
    
    def __init__(self, kv_store: KVStoreInterface):
        self.kv_store = kv_store
        self.namespace = KVNamespace.SESSION_DATA
    
    async def create_session(self, user_id: str, session_id: str, ttl_seconds: int = 3600) -> SessionData:
        """Create a new user session."""
        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(seconds=ttl_seconds)
        )
        
        kv_item = KVItem(
            key=session_id,
            value=session.dict(),
            namespace=self.namespace,
            data_type=KVDataType.JSON,
            user_id=user_id,
            ttl_seconds=ttl_seconds
        )
        
        success = await self.kv_store.set(kv_item)
        if success:
            return session
        else:
            raise Exception("Failed to create session")
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by session ID."""
        kv_item = await self.kv_store.get(session_id, self.namespace)
        if kv_item:
            return SessionData(**kv_item.value)
        return None
    
    async def update_session(self, session: SessionData) -> bool:
        """Update session data."""
        session.touch()
        
        kv_item = KVItem(
            key=session.session_id,
            value=session.dict(),
            namespace=self.namespace,
            data_type=KVDataType.JSON,
            user_id=session.user_id
        )
        
        return await self.kv_store.set(kv_item)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return await self.kv_store.delete(session_id, self.namespace)
    
    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """Get all sessions for a user."""
        filter_criteria = KVSearchFilter(
            namespaces=[self.namespace],
            user_id=user_id,
            include_expired=False
        )
        
        kv_items = await self.kv_store.search(filter_criteria)
        sessions = []
        
        for item in kv_items:
            try:
                session = SessionData(**item.value)
                sessions.append(session)
            except Exception as e:
                logger.error(f"Error parsing session data: {e}")
        
        return sessions
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        return await self.kv_store.cleanup_expired()


class UserPreferencesStore:
    """Store for user preferences and settings."""
    
    def __init__(self, kv_store: KVStoreInterface):
        self.kv_store = kv_store
        self.namespace = KVNamespace.USER_PREFERENCES
    
    async def get_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Get user preferences."""
        kv_item = await self.kv_store.get(f"prefs_{user_id}", self.namespace)
        if kv_item:
            return UserPreferences(**kv_item.value)
        return None
    
    async def set_preferences(self, preferences: UserPreferences) -> bool:
        """Set user preferences."""
        preferences.updated_at = datetime.utcnow()
        
        kv_item = KVItem(
            key=f"prefs_{preferences.user_id}",
            value=preferences.dict(),
            namespace=self.namespace,
            data_type=KVDataType.JSON,
            user_id=preferences.user_id
        )
        
        return await self.kv_store.set(kv_item)
    
    async def update_preference(self, user_id: str, key: str, value: Any) -> bool:
        """Update a specific preference."""
        preferences = await self.get_preferences(user_id)
        if not preferences:
            preferences = UserPreferences(user_id=user_id)
        
        # Update the specific preference
        if hasattr(preferences, key):
            setattr(preferences, key, value)
        else:
            preferences.custom_settings[key] = value
        
        return await self.set_preferences(preferences)


def create_kv_store(config: DatabaseConfig) -> KVStoreInterface:
    """Factory function to create appropriate KV store implementation."""
    # For now, return memory store. In production, this would check config
    # and return PostgreSQL or Redis implementations
    return MemoryKVStore()
