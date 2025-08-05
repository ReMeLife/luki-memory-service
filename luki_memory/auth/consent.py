"""
Consent management for LUKi Memory Service.

Handles user consent levels, data usage permissions, and GDPR compliance.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..schemas.elr import ConsentLevel

logger = logging.getLogger(__name__)


class ConsentAction(str, Enum):
    """Types of actions that require consent."""
    DATA_COLLECTION = "data_collection"
    DATA_PROCESSING = "data_processing"
    DATA_STORAGE = "data_storage"
    DATA_SHARING = "data_sharing"
    RESEARCH_USE = "research_use"
    FAMILY_SHARING = "family_sharing"
    AI_TRAINING = "ai_training"
    ANALYTICS = "analytics"
    MARKETING = "marketing"


class ConsentStatus(str, Enum):
    """Consent status values."""
    GRANTED = "granted"
    DENIED = "denied"
    PENDING = "pending"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


@dataclass
class ConsentRecord:
    """Individual consent record."""
    
    user_id: str
    action: ConsentAction
    status: ConsentStatus
    consent_level: ConsentLevel
    
    # Temporal information
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    
    # Context
    purpose: Optional[str] = None
    data_categories: List[str] = field(default_factory=list)
    third_parties: List[str] = field(default_factory=list)
    
    # Legal basis
    legal_basis: str = "consent"  # consent, legitimate_interest, contract, etc.
    
    # Audit trail
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    
    # Metadata
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    consent_method: str = "explicit"  # explicit, implicit, opt_out
    
    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.status != ConsentStatus.GRANTED:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        if self.withdrawn_at:
            return False
        
        return True
    
    def is_expired(self) -> bool:
        """Check if consent has expired."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return True
        return False


@dataclass
class ConsentPreferences:
    """User's consent preferences and settings."""
    
    user_id: str
    
    # Default consent levels
    default_consent_level: ConsentLevel = ConsentLevel.PRIVATE
    
    # Granular consent settings
    data_collection_consent: ConsentStatus = ConsentStatus.PENDING
    data_processing_consent: ConsentStatus = ConsentStatus.PENDING
    research_consent: ConsentStatus = ConsentStatus.DENIED
    family_sharing_consent: ConsentStatus = ConsentStatus.DENIED
    ai_training_consent: ConsentStatus = ConsentStatus.DENIED
    
    # Data retention preferences
    data_retention_days: Optional[int] = None
    auto_delete_enabled: bool = False
    
    # Communication preferences
    consent_reminders: bool = True
    privacy_updates: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_consent_status(self, action: ConsentAction) -> ConsentStatus:
        """Get consent status for a specific action."""
        action_mapping = {
            ConsentAction.DATA_COLLECTION: self.data_collection_consent,
            ConsentAction.DATA_PROCESSING: self.data_processing_consent,
            ConsentAction.RESEARCH_USE: self.research_consent,
            ConsentAction.FAMILY_SHARING: self.family_sharing_consent,
            ConsentAction.AI_TRAINING: self.ai_training_consent,
        }
        
        return action_mapping.get(action, ConsentStatus.PENDING)


class ConsentManager:
    """Manages user consent and data usage permissions."""
    
    def __init__(self):
        self._consent_records: Dict[str, Dict[ConsentAction, ConsentRecord]] = {}
        self._user_preferences: Dict[str, ConsentPreferences] = {}
    
    async def grant_consent(self, user_id: str, action: ConsentAction, 
                          consent_level: ConsentLevel = ConsentLevel.PRIVATE,
                          purpose: Optional[str] = None,
                          duration_days: Optional[int] = None,
                          context: Optional[Dict[str, Any]] = None) -> bool:
        """Grant consent for a specific action."""
        try:
            # Calculate expiration if duration specified
            expires_at = None
            if duration_days:
                expires_at = datetime.utcnow() + timedelta(days=duration_days)
            
            consent_record = ConsentRecord(
                user_id=user_id,
                action=action,
                status=ConsentStatus.GRANTED,
                consent_level=consent_level,
                granted_at=datetime.utcnow(),
                expires_at=expires_at,
                purpose=purpose,
                ip_address=context.get("ip_address") if context else None,
                user_agent=context.get("user_agent") if context else None
            )
            
            # Store consent record
            if user_id not in self._consent_records:
                self._consent_records[user_id] = {}
            
            self._consent_records[user_id][action] = consent_record
            
            # Update user preferences
            await self._update_user_preferences(user_id, action, ConsentStatus.GRANTED)
            
            logger.info(f"Granted consent for user {user_id}, action {action}, level {consent_level}")
            return True
        except Exception as e:
            logger.error(f"Error granting consent: {e}")
            return False
    
    async def withdraw_consent(self, user_id: str, action: ConsentAction) -> bool:
        """Withdraw consent for a specific action."""
        try:
            if user_id in self._consent_records and action in self._consent_records[user_id]:
                consent_record = self._consent_records[user_id][action]
                consent_record.status = ConsentStatus.WITHDRAWN
                consent_record.withdrawn_at = datetime.utcnow()
                consent_record.updated_at = datetime.utcnow()
                consent_record.version += 1
                
                # Update user preferences
                await self._update_user_preferences(user_id, action, ConsentStatus.WITHDRAWN)
                
                logger.info(f"Withdrew consent for user {user_id}, action {action}")
                return True
            
            logger.warning(f"No consent record found for user {user_id}, action {action}")
            return False
        except Exception as e:
            logger.error(f"Error withdrawing consent: {e}")
            return False
    
    async def check_consent(self, user_id: str, action: ConsentAction, 
                          resource_consent_level: Optional[ConsentLevel] = None) -> bool:
        """Check if user has valid consent for an action."""
        try:
            # Check if user has consent record for this action
            if user_id not in self._consent_records or action not in self._consent_records[user_id]:
                logger.debug(f"No consent record for user {user_id}, action {action}")
                return False
            
            consent_record = self._consent_records[user_id][action]
            
            # Check if consent is valid
            if not consent_record.is_valid():
                logger.debug(f"Invalid consent for user {user_id}, action {action}")
                return False
            
            # Check consent level compatibility
            if resource_consent_level:
                if not self._is_consent_level_compatible(consent_record.consent_level, resource_consent_level):
                    logger.debug(f"Consent level incompatible for user {user_id}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking consent: {e}")
            return False
    
    def _is_consent_level_compatible(self, user_consent: ConsentLevel, resource_consent: ConsentLevel) -> bool:
        """Check if user's consent level is compatible with resource consent level."""
        # Define consent level hierarchy
        level_hierarchy = {
            ConsentLevel.PRIVATE: 0,
            ConsentLevel.FAMILY: 1,
            ConsentLevel.RESEARCH: 2
        }
        
        user_level = level_hierarchy.get(user_consent, 0)
        resource_level = level_hierarchy.get(resource_consent, 0)
        
        # User consent must be at least as permissive as resource requirement
        return user_level >= resource_level
    
    async def get_user_consent_status(self, user_id: str) -> Dict[ConsentAction, ConsentStatus]:
        """Get consent status for all actions for a user."""
        try:
            status_map = {}
            
            for action in ConsentAction:
                if (user_id in self._consent_records and 
                    action in self._consent_records[user_id]):
                    consent_record = self._consent_records[user_id][action]
                    if consent_record.is_expired():
                        status_map[action] = ConsentStatus.EXPIRED
                    else:
                        status_map[action] = consent_record.status
                else:
                    status_map[action] = ConsentStatus.PENDING
            
            return status_map
        except Exception as e:
            logger.error(f"Error getting user consent status: {e}")
            return {}
    
    async def get_user_preferences(self, user_id: str) -> Optional[ConsentPreferences]:
        """Get user's consent preferences."""
        return self._user_preferences.get(user_id)
    
    async def update_user_preferences(self, preferences: ConsentPreferences) -> bool:
        """Update user's consent preferences."""
        try:
            preferences.updated_at = datetime.utcnow()
            self._user_preferences[preferences.user_id] = preferences
            
            logger.info(f"Updated consent preferences for user {preferences.user_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False
    
    async def _update_user_preferences(self, user_id: str, action: ConsentAction, status: ConsentStatus):
        """Update user preferences based on consent action."""
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = ConsentPreferences(user_id=user_id)
        
        preferences = self._user_preferences[user_id]
        
        # Update specific consent status
        if action == ConsentAction.DATA_COLLECTION:
            preferences.data_collection_consent = status
        elif action == ConsentAction.DATA_PROCESSING:
            preferences.data_processing_consent = status
        elif action == ConsentAction.RESEARCH_USE:
            preferences.research_consent = status
        elif action == ConsentAction.FAMILY_SHARING:
            preferences.family_sharing_consent = status
        elif action == ConsentAction.AI_TRAINING:
            preferences.ai_training_consent = status
        
        preferences.updated_at = datetime.utcnow()
    
    async def get_consent_audit_trail(self, user_id: str, action: Optional[ConsentAction] = None) -> List[ConsentRecord]:
        """Get audit trail of consent changes."""
        try:
            audit_trail = []
            
            if user_id in self._consent_records:
                user_consents = self._consent_records[user_id]
                
                if action:
                    if action in user_consents:
                        audit_trail.append(user_consents[action])
                else:
                    audit_trail.extend(user_consents.values())
            
            # Sort by creation time
            audit_trail.sort(key=lambda x: x.created_at, reverse=True)
            return audit_trail
        except Exception as e:
            logger.error(f"Error getting consent audit trail: {e}")
            return []
    
    async def cleanup_expired_consents(self) -> int:
        """Clean up expired consent records."""
        try:
            expired_count = 0
            
            for user_id, user_consents in self._consent_records.items():
                expired_actions = []
                
                for action, consent_record in user_consents.items():
                    if consent_record.is_expired():
                        consent_record.status = ConsentStatus.EXPIRED
                        consent_record.updated_at = datetime.utcnow()
                        expired_count += 1
                        
                        # Update user preferences
                        await self._update_user_preferences(user_id, action, ConsentStatus.EXPIRED)
            
            logger.info(f"Marked {expired_count} consent records as expired")
            return expired_count
        except Exception as e:
            logger.error(f"Error cleaning up expired consents: {e}")
            return 0
    
    async def export_user_consent_data(self, user_id: str) -> Dict[str, Any]:
        """Export all consent data for a user (GDPR compliance)."""
        try:
            export_data = {
                "user_id": user_id,
                "export_timestamp": datetime.utcnow().isoformat(),
                "consent_records": [],
                "preferences": None
            }
            
            # Export consent records
            if user_id in self._consent_records:
                for action, consent_record in self._consent_records[user_id].items():
                    export_data["consent_records"].append({
                        "action": action.value,
                        "status": consent_record.status.value,
                        "consent_level": consent_record.consent_level.value,
                        "granted_at": consent_record.granted_at.isoformat() if consent_record.granted_at else None,
                        "expires_at": consent_record.expires_at.isoformat() if consent_record.expires_at else None,
                        "withdrawn_at": consent_record.withdrawn_at.isoformat() if consent_record.withdrawn_at else None,
                        "purpose": consent_record.purpose,
                        "legal_basis": consent_record.legal_basis,
                        "created_at": consent_record.created_at.isoformat(),
                        "version": consent_record.version
                    })
            
            # Export preferences
            if user_id in self._user_preferences:
                preferences = self._user_preferences[user_id]
                export_data["preferences"] = {
                    "default_consent_level": preferences.default_consent_level.value,
                    "data_retention_days": preferences.data_retention_days,
                    "auto_delete_enabled": preferences.auto_delete_enabled,
                    "consent_reminders": preferences.consent_reminders,
                    "privacy_updates": preferences.privacy_updates,
                    "created_at": preferences.created_at.isoformat(),
                    "updated_at": preferences.updated_at.isoformat()
                }
            
            return export_data
        except Exception as e:
            logger.error(f"Error exporting user consent data: {e}")
            return {}
    
    async def delete_user_consent_data(self, user_id: str) -> bool:
        """Delete all consent data for a user (GDPR right to be forgotten)."""
        try:
            # Remove consent records
            if user_id in self._consent_records:
                del self._consent_records[user_id]
            
            # Remove preferences
            if user_id in self._user_preferences:
                del self._user_preferences[user_id]
            
            logger.info(f"Deleted all consent data for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting user consent data: {e}")
            return False


def create_default_consent_preferences(user_id: str) -> ConsentPreferences:
    """Create default consent preferences for a new user."""
    return ConsentPreferences(
        user_id=user_id,
        default_consent_level=ConsentLevel.PRIVATE,
        data_collection_consent=ConsentStatus.GRANTED,  # Required for basic functionality
        data_processing_consent=ConsentStatus.GRANTED,  # Required for basic functionality
        research_consent=ConsentStatus.PENDING,
        family_sharing_consent=ConsentStatus.PENDING,
        ai_training_consent=ConsentStatus.PENDING
    )
