"""
Version tracking and audit for data changes in LUKi Memory Service.

Maintains version history of ELR items and other data for audit trails and rollback capability.
"""

import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Types of changes that can be tracked."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"
    MERGE = "merge"
    SPLIT = "split"


@dataclass
class VersionRecord:
    """Individual version record for a data item."""
    
    # Version identification
    version_id: str
    item_id: str
    version_number: int
    
    # Change information
    change_type: ChangeType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # User context
    changed_by: str
    session_id: Optional[str] = None
    
    # Data snapshot
    data_snapshot: Dict[str, Any] = field(default_factory=dict)
    data_hash: Optional[str] = None
    
    # Change details
    changes_summary: Optional[str] = None
    fields_changed: List[str] = field(default_factory=list)
    previous_version: Optional[str] = None
    
    # Metadata
    size_bytes: Optional[int] = None
    compression_used: bool = False
    
    # Audit information
    reason: Optional[str] = None
    automated: bool = False
    
    def calculate_hash(self) -> str:
        """Calculate hash of the data snapshot."""
        data_str = json.dumps(self.data_snapshot, sort_keys=True, default=str)
        return sha256(data_str.encode()).hexdigest()
    
    def __post_init__(self):
        """Calculate hash after initialization."""
        if not self.data_hash and self.data_snapshot:
            self.data_hash = self.calculate_hash()


@dataclass
class VersionHistory:
    """Complete version history for a data item."""
    
    item_id: str
    item_type: str
    current_version: int
    total_versions: int
    
    # Version records
    versions: List[VersionRecord] = field(default_factory=list)
    
    # Summary information
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    last_modified_by: Optional[str] = None
    
    # Retention policy
    max_versions: int = 50
    retention_days: Optional[int] = None
    
    def get_version(self, version_number: int) -> Optional[VersionRecord]:
        """Get a specific version by number."""
        for version in self.versions:
            if version.version_number == version_number:
                return version
        return None
    
    def get_latest_version(self) -> Optional[VersionRecord]:
        """Get the latest version."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.version_number)
    
    def get_versions_by_user(self, user_id: str) -> List[VersionRecord]:
        """Get all versions created by a specific user."""
        return [v for v in self.versions if v.changed_by == user_id]
    
    def get_versions_in_range(self, start_date: datetime, end_date: datetime) -> List[VersionRecord]:
        """Get versions within a date range."""
        return [v for v in self.versions if start_date <= v.timestamp <= end_date]


class VersionManager:
    """Manages version tracking and history for data items."""
    
    def __init__(self, max_versions_per_item: int = 50):
        self.max_versions_per_item = max_versions_per_item
        self._version_histories: Dict[str, VersionHistory] = {}
    
    async def create_version(self, item_id: str, item_type: str, 
                           data: Dict[str, Any], user_id: str,
                           change_type: ChangeType = ChangeType.CREATE,
                           session_id: Optional[str] = None,
                           reason: Optional[str] = None,
                           automated: bool = False) -> str:
        """Create a new version record."""
        try:
            # Get or create version history
            if item_id not in self._version_histories:
                self._version_histories[item_id] = VersionHistory(
                    item_id=item_id,
                    item_type=item_type,
                    current_version=0,
                    total_versions=0,
                    created_by=user_id
                )
            
            history = self._version_histories[item_id]
            
            # Create new version
            version_number = history.current_version + 1
            version_id = f"{item_id}_v{version_number}"
            
            # Get previous version for comparison
            previous_version_id = None
            fields_changed = []
            if history.versions:
                latest_version = history.get_latest_version()
                if latest_version:
                    previous_version_id = latest_version.version_id
                    fields_changed = self._calculate_changed_fields(latest_version.data_snapshot, data)
            
            version_record = VersionRecord(
                version_id=version_id,
                item_id=item_id,
                version_number=version_number,
                change_type=change_type,
                changed_by=user_id,
                session_id=session_id,
                data_snapshot=data.copy(),
                changes_summary=reason,
                fields_changed=fields_changed,
                previous_version=previous_version_id,
                reason=reason,
                automated=automated,
                size_bytes=len(json.dumps(data, default=str))
            )
            
            # Add to history
            history.versions.append(version_record)
            history.current_version = version_number
            history.total_versions += 1
            history.last_modified = datetime.utcnow()
            history.last_modified_by = user_id
            
            # Cleanup old versions if needed
            await self._cleanup_old_versions(history)
            
            logger.info(f"Created version {version_id} for item {item_id}")
            return version_id
        except Exception as e:
            logger.error(f"Error creating version: {e}")
            raise
    
    def _calculate_changed_fields(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> List[str]:
        """Calculate which fields changed between versions."""
        changed_fields = []
        
        # Check for modified and new fields
        for key, value in new_data.items():
            if key not in old_data or old_data[key] != value:
                changed_fields.append(key)
        
        # Check for removed fields
        for key in old_data:
            if key not in new_data:
                changed_fields.append(f"-{key}")  # Prefix with - to indicate removal
        
        return changed_fields
    
    async def get_version_history(self, item_id: str) -> Optional[VersionHistory]:
        """Get complete version history for an item."""
        return self._version_histories.get(item_id)
    
    async def get_version_data(self, item_id: str, version_number: int) -> Optional[Dict[str, Any]]:
        """Get data for a specific version."""
        history = self._version_histories.get(item_id)
        if not history:
            return None
        
        version = history.get_version(version_number)
        if version:
            return version.data_snapshot.copy()
        
        return None
    
    async def get_latest_version_data(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get data for the latest version."""
        history = self._version_histories.get(item_id)
        if not history:
            return None
        
        latest_version = history.get_latest_version()
        if latest_version:
            return latest_version.data_snapshot.copy()
        
        return None
    
    async def rollback_to_version(self, item_id: str, version_number: int, 
                                user_id: str, reason: Optional[str] = None) -> Optional[str]:
        """Rollback an item to a previous version."""
        try:
            history = self._version_histories.get(item_id)
            if not history:
                logger.warning(f"No version history found for item {item_id}")
                return None
            
            target_version = history.get_version(version_number)
            if not target_version:
                logger.warning(f"Version {version_number} not found for item {item_id}")
                return None
            
            # Create new version with the old data
            rollback_reason = reason or f"Rollback to version {version_number}"
            new_version_id = await self.create_version(
                item_id=item_id,
                item_type=history.item_type,
                data=target_version.data_snapshot,
                user_id=user_id,
                change_type=ChangeType.RESTORE,
                reason=rollback_reason
            )
            
            logger.info(f"Rolled back item {item_id} to version {version_number}, created {new_version_id}")
            return new_version_id
        except Exception as e:
            logger.error(f"Error rolling back version: {e}")
            return None
    
    async def compare_versions(self, item_id: str, version1: int, version2: int) -> Optional[Dict[str, Any]]:
        """Compare two versions and return differences."""
        try:
            history = self._version_histories.get(item_id)
            if not history:
                return None
            
            v1_record = history.get_version(version1)
            v2_record = history.get_version(version2)
            
            if not v1_record or not v2_record:
                return None
            
            v1_data = v1_record.data_snapshot
            v2_data = v2_record.data_snapshot
            
            comparison = {
                "version1": {
                    "number": version1,
                    "timestamp": v1_record.timestamp.isoformat(),
                    "changed_by": v1_record.changed_by
                },
                "version2": {
                    "number": version2,
                    "timestamp": v2_record.timestamp.isoformat(),
                    "changed_by": v2_record.changed_by
                },
                "differences": {
                    "added": {},
                    "modified": {},
                    "removed": {}
                }
            }
            
            # Find added and modified fields
            for key, value in v2_data.items():
                if key not in v1_data:
                    comparison["differences"]["added"][key] = value
                elif v1_data[key] != value:
                    comparison["differences"]["modified"][key] = {
                        "old": v1_data[key],
                        "new": value
                    }
            
            # Find removed fields
            for key, value in v1_data.items():
                if key not in v2_data:
                    comparison["differences"]["removed"][key] = value
            
            return comparison
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return None
    
    async def get_user_changes(self, user_id: str, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> List[VersionRecord]:
        """Get all changes made by a specific user."""
        user_changes = []
        
        for history in self._version_histories.values():
            user_versions = history.get_versions_by_user(user_id)
            
            if start_date or end_date:
                filtered_versions = []
                for version in user_versions:
                    if start_date and version.timestamp < start_date:
                        continue
                    if end_date and version.timestamp > end_date:
                        continue
                    filtered_versions.append(version)
                user_versions = filtered_versions
            
            user_changes.extend(user_versions)
        
        # Sort by timestamp, most recent first
        user_changes.sort(key=lambda v: v.timestamp, reverse=True)
        return user_changes
    
    async def _cleanup_old_versions(self, history: VersionHistory):
        """Clean up old versions based on retention policy."""
        try:
            # Remove versions exceeding max count
            if len(history.versions) > history.max_versions:
                # Sort by version number and keep the most recent ones
                history.versions.sort(key=lambda v: v.version_number)
                versions_to_remove = len(history.versions) - history.max_versions
                history.versions = history.versions[versions_to_remove:]
                logger.info(f"Removed {versions_to_remove} old versions for item {history.item_id}")
            
            # Remove versions exceeding retention period
            if history.retention_days:
                cutoff_date = datetime.utcnow().replace(day=datetime.utcnow().day - history.retention_days)
                original_count = len(history.versions)
                history.versions = [v for v in history.versions if v.timestamp > cutoff_date]
                removed_count = original_count - len(history.versions)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} expired versions for item {history.item_id}")
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {e}")
    
    async def get_version_statistics(self) -> Dict[str, Any]:
        """Get statistics about version tracking."""
        try:
            stats = {
                "total_items": len(self._version_histories),
                "total_versions": 0,
                "items_by_type": {},
                "versions_by_change_type": {},
                "average_versions_per_item": 0,
                "most_versioned_item": None,
                "most_versioned_count": 0
            }
            
            total_versions = 0
            most_versioned_count = 0
            most_versioned_item = None
            
            for item_id, history in self._version_histories.items():
                total_versions += history.total_versions
                
                # Track by item type
                item_type = history.item_type
                stats["items_by_type"][item_type] = stats["items_by_type"].get(item_type, 0) + 1
                
                # Track most versioned item
                if history.total_versions > most_versioned_count:
                    most_versioned_count = history.total_versions
                    most_versioned_item = item_id
                
                # Count by change type
                for version in history.versions:
                    change_type = version.change_type.value
                    stats["versions_by_change_type"][change_type] = stats["versions_by_change_type"].get(change_type, 0) + 1
            
            stats["total_versions"] = total_versions
            stats["average_versions_per_item"] = total_versions / len(self._version_histories) if self._version_histories else 0
            stats["most_versioned_item"] = most_versioned_item
            stats["most_versioned_count"] = most_versioned_count
            
            return stats
        except Exception as e:
            logger.error(f"Error getting version statistics: {e}")
            return {}
    
    async def export_version_history(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Export complete version history for an item."""
        try:
            history = self._version_histories.get(item_id)
            if not history:
                return None
            
            export_data = {
                "item_id": item_id,
                "item_type": history.item_type,
                "current_version": history.current_version,
                "total_versions": history.total_versions,
                "created_at": history.created_at.isoformat(),
                "last_modified": history.last_modified.isoformat(),
                "created_by": history.created_by,
                "last_modified_by": history.last_modified_by,
                "versions": []
            }
            
            for version in history.versions:
                version_data = {
                    "version_id": version.version_id,
                    "version_number": version.version_number,
                    "change_type": version.change_type.value,
                    "timestamp": version.timestamp.isoformat(),
                    "changed_by": version.changed_by,
                    "data_hash": version.data_hash,
                    "changes_summary": version.changes_summary,
                    "fields_changed": version.fields_changed,
                    "size_bytes": version.size_bytes,
                    "reason": version.reason,
                    "automated": version.automated
                }
                export_data["versions"].append(version_data)
            
            return export_data
        except Exception as e:
            logger.error(f"Error exporting version history: {e}")
            return None
