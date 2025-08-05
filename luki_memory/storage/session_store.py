"""
Session store implementation for short-term memory and ephemeral data.

Manages conversation context, temporary state, and short-lived data that doesn't
need to be persisted long-term.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from ..schemas.kv import SessionData, KVItem, KVNamespace, KVDataType
from .kv_store import KVStoreInterface

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Context data for ongoing conversations."""
    
    user_id: str
    session_id: str
    conversation_id: str
    
    # Message history (limited)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    max_messages: int = 20
    
    # Current state
    current_topic: Optional[str] = None
    user_intent: Optional[str] = None
    context_summary: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Preferences for this conversation
    response_style: str = "balanced"
    language: str = "en"
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        
        # Keep only the most recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        self.last_updated = datetime.utcnow()
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent messages."""
        return self.messages[-count:] if count < len(self.messages) else self.messages
    
    def clear_messages(self):
        """Clear all messages from the conversation."""
        self.messages.clear()
        self.last_updated = datetime.utcnow()


@dataclass
class UserState:
    """Current user state and temporary data."""
    
    user_id: str
    session_id: str
    
    # Current activity
    current_activity: Optional[str] = None
    activity_started_at: Optional[datetime] = None
    
    # Mood and context
    current_mood: Optional[str] = None
    energy_level: Optional[str] = None
    stress_level: Optional[str] = None
    
    # Goals and intentions
    current_goals: List[str] = field(default_factory=list)
    immediate_needs: List[str] = field(default_factory=list)
    
    # Temporary preferences
    temp_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Context flags
    is_in_crisis: bool = False
    needs_support: bool = False
    privacy_mode: bool = False
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_mood(self, mood: str, confidence: float = 1.0):
        """Update current mood with confidence score."""
        self.current_mood = mood
        self.temp_preferences["mood_confidence"] = confidence
        self.last_updated = datetime.utcnow()
    
    def add_goal(self, goal: str):
        """Add a current goal."""
        if goal not in self.current_goals:
            self.current_goals.append(goal)
            self.last_updated = datetime.utcnow()
    
    def remove_goal(self, goal: str):
        """Remove a completed or cancelled goal."""
        if goal in self.current_goals:
            self.current_goals.remove(goal)
            self.last_updated = datetime.utcnow()


class ShortTermMemoryStore:
    """Store for short-term memory and ephemeral session data."""
    
    def __init__(self, kv_store: KVStoreInterface, default_ttl: int = 3600):
        self.kv_store = kv_store
        self.default_ttl = default_ttl  # 1 hour default
        self.namespace = KVNamespace.SESSION_DATA
    
    async def store_conversation_context(self, context: ConversationContext, ttl_seconds: Optional[int] = None) -> bool:
        """Store conversation context."""
        key = f"conv_ctx_{context.conversation_id}"
        ttl = ttl_seconds or self.default_ttl
        
        kv_item = KVItem(
            key=key,
            value=context.__dict__,
            namespace=self.namespace,
            data_type=KVDataType.JSON,
            user_id=context.user_id,
            ttl_seconds=ttl
        )
        
        return await self.kv_store.set(kv_item)
    
    async def get_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context by ID."""
        key = f"conv_ctx_{conversation_id}"
        kv_item = await self.kv_store.get(key, self.namespace)
        
        if kv_item:
            try:
                data = kv_item.value
                # Reconstruct datetime objects
                data['created_at'] = datetime.fromisoformat(data['created_at']) if isinstance(data['created_at'], str) else data['created_at']
                data['last_updated'] = datetime.fromisoformat(data['last_updated']) if isinstance(data['last_updated'], str) else data['last_updated']
                return ConversationContext(**data)
            except Exception as e:
                logger.error(f"Error reconstructing conversation context: {e}")
                return None
        
        return None
    
    async def update_conversation_context(self, context: ConversationContext) -> bool:
        """Update existing conversation context."""
        context.last_updated = datetime.utcnow()
        return await self.store_conversation_context(context)
    
    async def store_user_state(self, state: UserState, ttl_seconds: Optional[int] = None) -> bool:
        """Store current user state."""
        key = f"user_state_{state.user_id}_{state.session_id}"
        ttl = ttl_seconds or self.default_ttl
        
        # Convert datetime objects to ISO strings for JSON serialization
        state_dict = state.__dict__.copy()
        for field in ['created_at', 'last_updated', 'activity_started_at']:
            if field in state_dict and state_dict[field]:
                if isinstance(state_dict[field], datetime):
                    state_dict[field] = state_dict[field].isoformat()
        
        kv_item = KVItem(
            key=key,
            value=state_dict,
            namespace=self.namespace,
            data_type=KVDataType.JSON,
            user_id=state.user_id,
            ttl_seconds=ttl
        )
        
        return await self.kv_store.set(kv_item)
    
    async def get_user_state(self, user_id: str, session_id: str) -> Optional[UserState]:
        """Get current user state."""
        key = f"user_state_{user_id}_{session_id}"
        kv_item = await self.kv_store.get(key, self.namespace)
        
        if kv_item:
            try:
                data = kv_item.value
                # Reconstruct datetime objects
                for field in ['created_at', 'last_updated', 'activity_started_at']:
                    if field in data and data[field]:
                        if isinstance(data[field], str):
                            data[field] = datetime.fromisoformat(data[field])
                
                return UserState(**data)
            except Exception as e:
                logger.error(f"Error reconstructing user state: {e}")
                return None
        
        return None
    
    async def update_user_state(self, state: UserState) -> bool:
        """Update existing user state."""
        state.last_updated = datetime.utcnow()
        return await self.store_user_state(state)
    
    async def store_temporary_data(self, key: str, data: Any, user_id: str, ttl_seconds: int = 300) -> bool:
        """Store temporary data with short TTL (5 minutes default)."""
        full_key = f"temp_{key}"
        
        kv_item = KVItem(
            key=full_key,
            value=data,
            namespace=KVNamespace.TEMPORARY,
            data_type=KVDataType.JSON if isinstance(data, (dict, list)) else KVDataType.STRING,
            user_id=user_id,
            ttl_seconds=ttl_seconds
        )
        
        return await self.kv_store.set(kv_item)
    
    async def get_temporary_data(self, key: str) -> Optional[Any]:
        """Get temporary data by key."""
        full_key = f"temp_{key}"
        kv_item = await self.kv_store.get(full_key, KVNamespace.TEMPORARY)
        
        if kv_item:
            return kv_item.value
        return None
    
    async def store_context_summary(self, user_id: str, session_id: str, summary: str, ttl_seconds: Optional[int] = None) -> bool:
        """Store conversation context summary."""
        key = f"ctx_summary_{user_id}_{session_id}"
        ttl = ttl_seconds or (self.default_ttl * 2)  # Longer TTL for summaries
        
        kv_item = KVItem(
            key=key,
            value={
                "summary": summary,
                "created_at": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "session_id": session_id
            },
            namespace=self.namespace,
            data_type=KVDataType.JSON,
            user_id=user_id,
            ttl_seconds=ttl
        )
        
        return await self.kv_store.set(kv_item)
    
    async def get_context_summary(self, user_id: str, session_id: str) -> Optional[str]:
        """Get conversation context summary."""
        key = f"ctx_summary_{user_id}_{session_id}"
        kv_item = await self.kv_store.get(key, self.namespace)
        
        if kv_item:
            return kv_item.value.get("summary")
        return None
    
    async def clear_user_session_data(self, user_id: str, session_id: str) -> bool:
        """Clear all session data for a user."""
        try:
            # Delete conversation contexts
            conv_key = f"conv_ctx_{session_id}"
            await self.kv_store.delete(conv_key, self.namespace)
            
            # Delete user state
            state_key = f"user_state_{user_id}_{session_id}"
            await self.kv_store.delete(state_key, self.namespace)
            
            # Delete context summary
            summary_key = f"ctx_summary_{user_id}_{session_id}"
            await self.kv_store.delete(summary_key, self.namespace)
            
            return True
        except Exception as e:
            logger.error(f"Error clearing session data: {e}")
            return False
    
    async def get_active_sessions(self, user_id: str) -> List[str]:
        """Get list of active session IDs for a user."""
        from ..schemas.kv import KVSearchFilter
        
        filter_criteria = KVSearchFilter(
            key_prefix=f"user_state_{user_id}_",
            namespaces=[self.namespace],
            user_id=user_id,
            include_expired=False
        )
        
        kv_items = await self.kv_store.search(filter_criteria)
        session_ids = []
        
        for item in kv_items:
            # Extract session ID from key: user_state_{user_id}_{session_id}
            key_parts = item.key.split('_')
            if len(key_parts) >= 3:
                session_id = '_'.join(key_parts[3:])  # Handle session IDs with underscores
                session_ids.append(session_id)
        
        return session_ids
    
    async def cleanup_expired_data(self) -> int:
        """Clean up expired session data."""
        return await self.kv_store.cleanup_expired()


class ContextBuilder:
    """Builds context for LLM interactions using short-term memory."""
    
    def __init__(self, memory_store: ShortTermMemoryStore):
        self.memory_store = memory_store
    
    async def build_context(self, user_id: str, session_id: str, conversation_id: str, query: str) -> Dict[str, Any]:
        """Build comprehensive context for LLM interaction."""
        context = {
            "user_id": user_id,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "query": query,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Get conversation context
        conv_context = await self.memory_store.get_conversation_context(conversation_id)
        if conv_context:
            context["conversation"] = {
                "recent_messages": conv_context.get_recent_messages(5),
                "current_topic": conv_context.current_topic,
                "user_intent": conv_context.user_intent,
                "response_style": conv_context.response_style
            }
        
        # Get user state
        user_state = await self.memory_store.get_user_state(user_id, session_id)
        if user_state:
            context["user_state"] = {
                "current_mood": user_state.current_mood,
                "energy_level": user_state.energy_level,
                "current_goals": user_state.current_goals,
                "immediate_needs": user_state.immediate_needs,
                "is_in_crisis": user_state.is_in_crisis,
                "needs_support": user_state.needs_support,
                "privacy_mode": user_state.privacy_mode
            }
        
        # Get context summary
        summary = await self.memory_store.get_context_summary(user_id, session_id)
        if summary:
            context["context_summary"] = summary
        
        return context
    
    async def update_context_after_interaction(self, user_id: str, session_id: str, conversation_id: str, 
                                             user_message: str, assistant_response: str, 
                                             extracted_info: Optional[Dict] = None) -> bool:
        """Update context after an interaction."""
        try:
            # Update conversation context
            conv_context = await self.memory_store.get_conversation_context(conversation_id)
            if not conv_context:
                conv_context = ConversationContext(
                    user_id=user_id,
                    session_id=session_id,
                    conversation_id=conversation_id
                )
            
            conv_context.add_message("user", user_message)
            conv_context.add_message("assistant", assistant_response)
            
            # Update topic and intent if extracted
            if extracted_info:
                if "topic" in extracted_info:
                    conv_context.current_topic = extracted_info["topic"]
                if "intent" in extracted_info:
                    conv_context.user_intent = extracted_info["intent"]
            
            await self.memory_store.update_conversation_context(conv_context)
            
            # Update user state if mood/state info was extracted
            if extracted_info and any(key in extracted_info for key in ["mood", "energy", "goals", "needs"]):
                user_state = await self.memory_store.get_user_state(user_id, session_id)
                if not user_state:
                    user_state = UserState(user_id=user_id, session_id=session_id)
                
                if "mood" in extracted_info:
                    user_state.update_mood(extracted_info["mood"])
                if "energy" in extracted_info:
                    user_state.energy_level = extracted_info["energy"]
                if "goals" in extracted_info:
                    for goal in extracted_info["goals"]:
                        user_state.add_goal(goal)
                
                await self.memory_store.update_user_state(user_state)
            
            return True
        except Exception as e:
            logger.error(f"Error updating context after interaction: {e}")
            return False
