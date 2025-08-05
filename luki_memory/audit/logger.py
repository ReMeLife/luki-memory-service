"""
Audit logging for LUKi Memory Service.

Tracks all data access, modifications, and security events for compliance and monitoring.
"""

import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    # Data operations
    DATA_ACCESS = "data_access"
    DATA_CREATE = "data_create"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    
    # Search operations
    SEARCH_QUERY = "search_query"
    SEARCH_RESULT = "search_result"
    
    # Authentication and authorization
    AUTH_LOGIN = "auth_login"
    AUTH_LOGOUT = "auth_logout"
    AUTH_FAILURE = "auth_failure"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    
    # Consent operations
    CONSENT_GRANTED = "consent_granted"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    CONSENT_EXPIRED = "consent_expired"
    
    # System operations
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_ERROR = "system_error"
    
    # Privacy operations
    DATA_ANONYMIZED = "data_anonymized"
    DATA_PSEUDONYMIZED = "data_pseudonymized"
    PII_DETECTED = "pii_detected"
    PII_REDACTED = "pii_redacted"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Individual audit event record."""
    
    # Core identification
    event_id: str
    event_type: AuditEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # User and session context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_owner_id: Optional[str] = None
    
    # Operation details
    operation: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    # Data context
    data_categories: List[str] = field(default_factory=list)
    consent_level: Optional[str] = None
    sensitivity_level: Optional[str] = None
    
    # Security context
    severity: AuditSeverity = AuditSeverity.LOW
    risk_score: Optional[float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Compliance fields
    legal_basis: Optional[str] = None
    retention_period: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for serialization."""
        event_dict = asdict(self)
        event_dict['timestamp'] = self.timestamp.isoformat()
        event_dict['event_type'] = self.event_type.value
        event_dict['severity'] = self.severity.value
        return event_dict
    
    def to_json(self) -> str:
        """Convert audit event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """Audit logging system for compliance and security monitoring."""
    
    def __init__(self, log_file_path: Optional[str] = None, 
                 structured_logging: bool = True,
                 retention_days: int = 2555):  # 7 years default for compliance
        self.log_file_path = log_file_path or "audit.log"
        self.structured_logging = structured_logging
        self.retention_days = retention_days
        
        # In-memory cache for recent events (for performance monitoring)
        self._recent_events: List[AuditEvent] = []
        self._max_recent_events = 1000
        
        # Setup file logging
        self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Setup file-based audit logging."""
        try:
            # Create audit log directory if it doesn't exist
            log_path = Path(self.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Setup structured logging handler
            audit_logger = logging.getLogger("audit")
            audit_logger.setLevel(logging.INFO)
            
            # Create file handler with rotation
            from logging.handlers import RotatingFileHandler
            handler = RotatingFileHandler(
                self.log_file_path,
                maxBytes=100*1024*1024,  # 100MB
                backupCount=10
            )
            
            if self.structured_logging:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            else:
                formatter = logging.Formatter('%(message)s')
            
            handler.setFormatter(formatter)
            audit_logger.addHandler(handler)
            
            self._audit_logger = audit_logger
            
        except Exception as e:
            logger.error(f"Failed to setup audit logging: {e}")
            self._audit_logger = None
    
    async def log_event(self, event: AuditEvent) -> bool:
        """Log an audit event."""
        try:
            # Add to recent events cache
            self._recent_events.append(event)
            if len(self._recent_events) > self._max_recent_events:
                self._recent_events.pop(0)
            
            # Log to file
            if self._audit_logger:
                if self.structured_logging:
                    self._audit_logger.info(event.to_json())
                else:
                    self._audit_logger.info(f"{event.timestamp.isoformat()} | {event.event_type.value} | {event.user_id} | {event.operation} | {event.success}")
            
            # Log to console for development
            if event.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
                logger.warning(f"High severity audit event: {event.event_type.value} - {event.operation}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return False
    
    async def log_operation(self, user_id: str, operation: str, 
                          resource_type: Optional[str] = None,
                          resource_id: Optional[str] = None,
                          success: bool = True,
                          error_message: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          severity: AuditSeverity = AuditSeverity.LOW) -> bool:
        """Log a general operation."""
        from ..utils.ids import generate_audit_id
        
        event = AuditEvent(
            event_id=generate_audit_id(),
            event_type=AuditEventType.DATA_ACCESS if success else AuditEventType.SYSTEM_ERROR,
            user_id=user_id,
            operation=operation,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
            error_message=error_message,
            metadata=metadata or {},
            severity=severity
        )
        
        return await self.log_event(event)
    
    async def log_data_access(self, user_id: str, resource_type: str, resource_id: str,
                            resource_owner_id: Optional[str] = None,
                            consent_level: Optional[str] = None,
                            operation: str = "read",
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Log data access event."""
        from ..utils.ids import generate_audit_id
        
        event = AuditEvent(
            event_id=generate_audit_id(),
            event_type=AuditEventType.DATA_ACCESS,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_owner_id=resource_owner_id,
            operation=operation,
            consent_level=consent_level,
            metadata=metadata or {},
            severity=AuditSeverity.LOW if user_id == resource_owner_id else AuditSeverity.MEDIUM
        )
        
        return await self.log_event(event)
    
    async def log_search_query(self, user_id: str, query: str, 
                             results_count: int,
                             filters: Optional[Dict[str, Any]] = None,
                             execution_time_ms: Optional[float] = None) -> bool:
        """Log search query event."""
        from ..utils.ids import generate_audit_id
        
        event = AuditEvent(
            event_id=generate_audit_id(),
            event_type=AuditEventType.SEARCH_QUERY,
            user_id=user_id,
            operation="search",
            metadata={
                "query": query,
                "results_count": results_count,
                "filters": filters or {},
                "execution_time_ms": execution_time_ms
            },
            severity=AuditSeverity.LOW
        )
        
        return await self.log_event(event)
    
    async def log_consent_event(self, user_id: str, action: str, 
                              consent_level: str,
                              event_type: AuditEventType = AuditEventType.CONSENT_GRANTED) -> bool:
        """Log consent-related event."""
        from ..utils.ids import generate_audit_id
        
        event = AuditEvent(
            event_id=generate_audit_id(),
            event_type=event_type,
            user_id=user_id,
            operation=action,
            consent_level=consent_level,
            metadata={"action": action},
            severity=AuditSeverity.MEDIUM,
            legal_basis="consent"
        )
        
        return await self.log_event(event)
    
    async def log_permission_event(self, user_id: str, permission: str, 
                                 resource_id: Optional[str] = None,
                                 granted: bool = True) -> bool:
        """Log permission check event."""
        from ..utils.ids import generate_audit_id
        
        event = AuditEvent(
            event_id=generate_audit_id(),
            event_type=AuditEventType.PERMISSION_GRANTED if granted else AuditEventType.PERMISSION_DENIED,
            user_id=user_id,
            resource_id=resource_id,
            operation=f"check_permission:{permission}",
            success=granted,
            metadata={"permission": permission},
            severity=AuditSeverity.LOW if granted else AuditSeverity.MEDIUM
        )
        
        return await self.log_event(event)
    
    async def log_pii_event(self, user_id: str, operation: str,
                          pii_types: List[str],
                          resource_id: Optional[str] = None) -> bool:
        """Log PII detection/redaction event."""
        from ..utils.ids import generate_audit_id
        
        event = AuditEvent(
            event_id=generate_audit_id(),
            event_type=AuditEventType.PII_DETECTED if operation == "detect" else AuditEventType.PII_REDACTED,
            user_id=user_id,
            resource_id=resource_id,
            operation=operation,
            data_categories=pii_types,
            metadata={"pii_types": pii_types},
            severity=AuditSeverity.HIGH,
            legal_basis="data_protection"
        )
        
        return await self.log_event(event)
    
    async def get_user_audit_trail(self, user_id: str, 
                                 event_types: Optional[List[AuditEventType]] = None,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None,
                                 limit: int = 100) -> List[AuditEvent]:
        """Get audit trail for a specific user."""
        try:
            filtered_events = []
            
            for event in reversed(self._recent_events):  # Most recent first
                # Filter by user
                if event.user_id != user_id and event.resource_owner_id != user_id:
                    continue
                
                # Filter by event types
                if event_types and event.event_type not in event_types:
                    continue
                
                # Filter by date range
                if start_date and event.timestamp < start_date:
                    continue
                if end_date and event.timestamp > end_date:
                    continue
                
                filtered_events.append(event)
                
                if len(filtered_events) >= limit:
                    break
            
            return filtered_events
        except Exception as e:
            logger.error(f"Error getting user audit trail: {e}")
            return []
    
    async def get_security_events(self, severity: Optional[AuditSeverity] = None,
                                limit: int = 100) -> List[AuditEvent]:
        """Get security-related audit events."""
        try:
            security_event_types = {
                AuditEventType.AUTH_FAILURE,
                AuditEventType.PERMISSION_DENIED,
                AuditEventType.PII_DETECTED,
                AuditEventType.SYSTEM_ERROR
            }
            
            filtered_events = []
            
            for event in reversed(self._recent_events):
                # Filter by security event types
                if event.event_type not in security_event_types:
                    continue
                
                # Filter by severity
                if severity and event.severity != severity:
                    continue
                
                filtered_events.append(event)
                
                if len(filtered_events) >= limit:
                    break
            
            return filtered_events
        except Exception as e:
            logger.error(f"Error getting security events: {e}")
            return []
    
    async def get_audit_statistics(self, start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get audit statistics for a time period."""
        try:
            stats = {
                "total_events": 0,
                "events_by_type": {},
                "events_by_severity": {},
                "unique_users": set(),
                "failed_operations": 0,
                "security_events": 0
            }
            
            for event in self._recent_events:
                # Filter by date range
                if start_date and event.timestamp < start_date:
                    continue
                if end_date and event.timestamp > end_date:
                    continue
                
                stats["total_events"] += 1
                
                # Count by type
                event_type = event.event_type.value
                stats["events_by_type"][event_type] = stats["events_by_type"].get(event_type, 0) + 1
                
                # Count by severity
                severity = event.severity.value
                stats["events_by_severity"][severity] = stats["events_by_severity"].get(severity, 0) + 1
                
                # Track unique users
                if event.user_id:
                    stats["unique_users"].add(event.user_id)
                
                # Count failures
                if not event.success:
                    stats["failed_operations"] += 1
                
                # Count security events
                if event.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
                    stats["security_events"] += 1
            
            # Convert set to count
            stats["unique_users"] = len(stats["unique_users"])
            
            return stats
        except Exception as e:
            logger.error(f"Error getting audit statistics: {e}")
            return {}
    
    async def cleanup_old_events(self) -> int:
        """Clean up old events from memory cache."""
        try:
            cutoff_date = datetime.utcnow().replace(day=datetime.utcnow().day - 1)  # Keep 1 day in memory
            
            original_count = len(self._recent_events)
            self._recent_events = [
                event for event in self._recent_events 
                if event.timestamp > cutoff_date
            ]
            
            cleaned_count = original_count - len(self._recent_events)
            logger.info(f"Cleaned up {cleaned_count} old audit events from memory")
            return cleaned_count
        except Exception as e:
            logger.error(f"Error cleaning up old events: {e}")
            return 0
