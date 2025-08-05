"""
Role-Based Access Control (RBAC) implementation for LUKi Memory Service.

Manages user roles, permissions, and access control for memory operations.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"
    USER = "user"
    FAMILY_MEMBER = "family_member"
    RESEARCHER = "researcher"
    CAREGIVER = "caregiver"
    GUEST = "guest"


class Permission(str, Enum):
    """System permissions."""
    # ELR operations
    READ_ELR = "read_elr"
    WRITE_ELR = "write_elr"
    DELETE_ELR = "delete_elr"
    INGEST_ELR = "ingest_elr"
    
    # Search operations
    SEARCH_MEMORIES = "search_memories"
    ADVANCED_SEARCH = "advanced_search"
    
    # User data operations
    READ_USER_DATA = "read_user_data"
    WRITE_USER_DATA = "write_user_data"
    DELETE_USER_DATA = "delete_user_data"
    
    # Preferences
    READ_PREFERENCES = "read_preferences"
    WRITE_PREFERENCES = "write_preferences"
    
    # Session management
    CREATE_SESSION = "create_session"
    READ_SESSION = "read_session"
    DELETE_SESSION = "delete_session"
    
    # Admin operations
    ADMIN_STATS = "admin_stats"
    ADMIN_CLEANUP = "admin_cleanup"
    MANAGE_USERS = "manage_users"
    AUDIT_ACCESS = "audit_access"
    
    # Research operations
    ANONYMIZED_ACCESS = "anonymized_access"
    BULK_EXPORT = "bulk_export"


@dataclass
class UserContext:
    """User context for authorization decisions."""
    user_id: str
    roles: Set[Role]
    permissions: Set[Permission] = field(default_factory=set)
    
    # Relationship context
    family_members: Set[str] = field(default_factory=set)
    caregivers: Set[str] = field(default_factory=set)
    
    # Consent context
    consent_levels: Set[str] = field(default_factory=set)
    
    # Session context
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    # Temporal context
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)


class RBACManager:
    """Role-Based Access Control manager."""
    
    def __init__(self):
        self._role_permissions = self._initialize_role_permissions()
        self._user_contexts: Dict[str, UserContext] = {}
    
    def _initialize_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """Initialize default role-permission mappings."""
        return {
            Role.ADMIN: {
                # Full access to everything
                Permission.READ_ELR,
                Permission.WRITE_ELR,
                Permission.DELETE_ELR,
                Permission.INGEST_ELR,
                Permission.SEARCH_MEMORIES,
                Permission.ADVANCED_SEARCH,
                Permission.READ_USER_DATA,
                Permission.WRITE_USER_DATA,
                Permission.DELETE_USER_DATA,
                Permission.READ_PREFERENCES,
                Permission.WRITE_PREFERENCES,
                Permission.CREATE_SESSION,
                Permission.READ_SESSION,
                Permission.DELETE_SESSION,
                Permission.ADMIN_STATS,
                Permission.ADMIN_CLEANUP,
                Permission.MANAGE_USERS,
                Permission.AUDIT_ACCESS,
                Permission.ANONYMIZED_ACCESS,
                Permission.BULK_EXPORT,
            },
            
            Role.USER: {
                # Standard user permissions for own data
                Permission.READ_ELR,
                Permission.WRITE_ELR,
                Permission.DELETE_ELR,
                Permission.INGEST_ELR,
                Permission.SEARCH_MEMORIES,
                Permission.READ_USER_DATA,
                Permission.WRITE_USER_DATA,
                Permission.READ_PREFERENCES,
                Permission.WRITE_PREFERENCES,
                Permission.CREATE_SESSION,
                Permission.READ_SESSION,
                Permission.DELETE_SESSION,
            },
            
            Role.FAMILY_MEMBER: {
                # Limited access to family member's data based on consent
                Permission.READ_ELR,
                Permission.SEARCH_MEMORIES,
                Permission.READ_USER_DATA,
                Permission.READ_PREFERENCES,
                Permission.CREATE_SESSION,
                Permission.READ_SESSION,
            },
            
            Role.RESEARCHER: {
                # Anonymized access for research
                Permission.ANONYMIZED_ACCESS,
                Permission.ADVANCED_SEARCH,
                Permission.BULK_EXPORT,
                Permission.CREATE_SESSION,
                Permission.READ_SESSION,
            },
            
            Role.CAREGIVER: {
                # Healthcare provider access
                Permission.READ_ELR,
                Permission.WRITE_ELR,
                Permission.SEARCH_MEMORIES,
                Permission.READ_USER_DATA,
                Permission.WRITE_USER_DATA,
                Permission.READ_PREFERENCES,
                Permission.CREATE_SESSION,
                Permission.READ_SESSION,
            },
            
            Role.GUEST: {
                # Very limited access
                Permission.CREATE_SESSION,
                Permission.READ_SESSION,
            }
        }
    
    async def register_user_context(self, user_context: UserContext) -> bool:
        """Register a user context for authorization."""
        try:
            # Calculate effective permissions based on roles
            effective_permissions = set()
            for role in user_context.roles:
                role_perms = self._role_permissions.get(role, set())
                effective_permissions.update(role_perms)
            
            user_context.permissions = effective_permissions
            self._user_contexts[user_context.user_id] = user_context
            
            logger.info(f"Registered user context for {user_context.user_id} with roles: {user_context.roles}")
            return True
        except Exception as e:
            logger.error(f"Error registering user context: {e}")
            return False
    
    async def check_permission(self, user_id: str, permission: Permission, 
                             resource_owner_id: Optional[str] = None,
                             resource_consent_level: Optional[str] = None) -> bool:
        """Check if user has permission for a specific operation."""
        try:
            user_context = self._user_contexts.get(user_id)
            if not user_context:
                logger.warning(f"No user context found for {user_id}")
                return False
            
            # Check if user has the required permission
            if permission not in user_context.permissions:
                logger.debug(f"User {user_id} lacks permission {permission}")
                return False
            
            # Additional checks for resource access
            if resource_owner_id and resource_owner_id != user_id:
                # Check if user can access another user's resource
                if not await self._check_cross_user_access(user_context, resource_owner_id, resource_consent_level):
                    return False
            
            # Update last activity
            user_context.last_activity = datetime.utcnow()
            
            return True
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False
    
    async def _check_cross_user_access(self, user_context: UserContext, 
                                     resource_owner_id: str,
                                     resource_consent_level: Optional[str]) -> bool:
        """Check if user can access another user's resource based on relationships and consent."""
        try:
            # Admin can access everything
            if Role.ADMIN in user_context.roles:
                return True
            
            # Family member access
            if Role.FAMILY_MEMBER in user_context.roles:
                if resource_owner_id in user_context.family_members:
                    # Check consent level
                    if resource_consent_level in ["family", "research"]:
                        return True
            
            # Caregiver access
            if Role.CAREGIVER in user_context.roles:
                if resource_owner_id in user_context.caregivers:
                    # Caregivers can access based on consent
                    if resource_consent_level in ["family", "research"]:
                        return True
            
            # Researcher access (anonymized only)
            if Role.RESEARCHER in user_context.roles:
                if resource_consent_level == "research":
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking cross-user access: {e}")
            return False
    
    async def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user."""
        user_context = self._user_contexts.get(user_id)
        if user_context:
            return user_context.permissions.copy()
        return set()
    
    async def add_user_role(self, user_id: str, role: Role) -> bool:
        """Add a role to a user."""
        try:
            user_context = self._user_contexts.get(user_id)
            if not user_context:
                logger.warning(f"No user context found for {user_id}")
                return False
            
            user_context.roles.add(role)
            
            # Recalculate permissions
            effective_permissions = set()
            for user_role in user_context.roles:
                role_perms = self._role_permissions.get(user_role, set())
                effective_permissions.update(role_perms)
            
            user_context.permissions = effective_permissions
            
            logger.info(f"Added role {role} to user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding user role: {e}")
            return False
    
    async def remove_user_role(self, user_id: str, role: Role) -> bool:
        """Remove a role from a user."""
        try:
            user_context = self._user_contexts.get(user_id)
            if not user_context:
                logger.warning(f"No user context found for {user_id}")
                return False
            
            user_context.roles.discard(role)
            
            # Recalculate permissions
            effective_permissions = set()
            for user_role in user_context.roles:
                role_perms = self._role_permissions.get(user_role, set())
                effective_permissions.update(role_perms)
            
            user_context.permissions = effective_permissions
            
            logger.info(f"Removed role {role} from user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing user role: {e}")
            return False
    
    async def add_family_relationship(self, user_id: str, family_member_id: str) -> bool:
        """Add a family relationship."""
        try:
            user_context = self._user_contexts.get(user_id)
            if user_context:
                user_context.family_members.add(family_member_id)
                logger.info(f"Added family relationship: {user_id} -> {family_member_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding family relationship: {e}")
            return False
    
    async def add_caregiver_relationship(self, user_id: str, caregiver_id: str) -> bool:
        """Add a caregiver relationship."""
        try:
            user_context = self._user_contexts.get(user_id)
            if user_context:
                user_context.caregivers.add(caregiver_id)
                logger.info(f"Added caregiver relationship: {user_id} -> {caregiver_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding caregiver relationship: {e}")
            return False
    
    async def check_multiple_permissions(self, user_id: str, permissions: List[Permission],
                                       resource_owner_id: Optional[str] = None,
                                       resource_consent_level: Optional[str] = None) -> Dict[Permission, bool]:
        """Check multiple permissions at once."""
        results = {}
        for permission in permissions:
            results[permission] = await self.check_permission(
                user_id, permission, resource_owner_id, resource_consent_level
            )
        return results
    
    async def get_accessible_users(self, user_id: str) -> Set[str]:
        """Get list of users whose data this user can access."""
        try:
            user_context = self._user_contexts.get(user_id)
            if not user_context:
                return {user_id}  # Can only access own data
            
            accessible_users = {user_id}  # Always can access own data
            
            # Admin can access all users
            if Role.ADMIN in user_context.roles:
                # In production, this would query all users from database
                # For now, return the user's accessible set
                pass
            
            # Add family members
            if Role.FAMILY_MEMBER in user_context.roles:
                accessible_users.update(user_context.family_members)
            
            # Add caregiver relationships
            if Role.CAREGIVER in user_context.roles:
                accessible_users.update(user_context.caregivers)
            
            return accessible_users
        except Exception as e:
            logger.error(f"Error getting accessible users: {e}")
            return {user_id}
    
    async def cleanup_expired_contexts(self, max_age_hours: int = 24) -> int:
        """Clean up expired user contexts."""
        try:
            cutoff_time = datetime.utcnow().replace(hour=datetime.utcnow().hour - max_age_hours)
            expired_users = []
            
            for user_id, context in self._user_contexts.items():
                if context.last_activity < cutoff_time:
                    expired_users.append(user_id)
            
            for user_id in expired_users:
                del self._user_contexts[user_id]
            
            logger.info(f"Cleaned up {len(expired_users)} expired user contexts")
            return len(expired_users)
        except Exception as e:
            logger.error(f"Error cleaning up expired contexts: {e}")
            return 0


def create_user_context(user_id: str, roles: List[str], 
                       family_members: Optional[List[str]] = None,
                       caregivers: Optional[List[str]] = None,
                       consent_levels: Optional[List[str]] = None) -> UserContext:
    """Factory function to create a user context."""
    role_enums = []
    for role_str in roles:
        try:
            role_enums.append(Role(role_str))
        except ValueError:
            logger.warning(f"Invalid role: {role_str}")
    
    return UserContext(
        user_id=user_id,
        roles=set(role_enums),
        family_members=set(family_members or []),
        caregivers=set(caregivers or []),
        consent_levels=set(consent_levels or ["private"])
    )
