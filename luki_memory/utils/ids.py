"""
ID generation and hashing utilities for LUKi Memory Service.

Provides functions for generating unique identifiers, hashing data, and managing ID formats.
"""

import uuid
import hashlib
import secrets
import time
from typing import Optional, Union, Any
from datetime import datetime


def generate_uuid() -> str:
    """Generate a standard UUID4."""
    return str(uuid.uuid4())


def generate_short_id(length: int = 8) -> str:
    """Generate a short random ID using URL-safe characters."""
    return secrets.token_urlsafe(length)[:length]


def generate_elr_item_id(user_id: str, content_type: str = "memory") -> str:
    """Generate an ELR item ID with user and type prefix."""
    timestamp = int(time.time())
    short_id = generate_short_id(6)
    return f"elr_{user_id}_{content_type}_{timestamp}_{short_id}"


def generate_chunk_id(item_id: str, chunk_index: int) -> str:
    """Generate a chunk ID based on parent item and index."""
    return f"{item_id}_chunk_{chunk_index:04d}"


def generate_session_id() -> str:
    """Generate a session ID."""
    timestamp = int(time.time())
    random_part = generate_short_id(12)
    return f"sess_{timestamp}_{random_part}"


def generate_conversation_id(user_id: str) -> str:
    """Generate a conversation ID."""
    timestamp = int(time.time())
    short_id = generate_short_id(8)
    return f"conv_{user_id}_{timestamp}_{short_id}"


def generate_audit_id() -> str:
    """Generate an audit event ID."""
    timestamp = int(time.time() * 1000)  # Include milliseconds
    short_id = generate_short_id(6)
    return f"audit_{timestamp}_{short_id}"


def generate_batch_id() -> str:
    """Generate a batch operation ID."""
    timestamp = int(time.time())
    short_id = generate_short_id(8)
    return f"batch_{timestamp}_{short_id}"


def hash_string(data: str, algorithm: str = "sha256") -> str:
    """Hash a string using the specified algorithm."""
    if algorithm == "md5":
        return hashlib.md5(data.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(data.encode()).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data.encode()).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(data.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def hash_data(data: Any, algorithm: str = "sha256") -> str:
    """Hash any data by converting to string first."""
    import json
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hash_string(data_str, algorithm)


def generate_content_hash(content: str) -> str:
    """Generate a content hash for deduplication."""
    # Normalize content for consistent hashing
    normalized = content.strip().lower()
    return hash_string(normalized)


def generate_user_hash(user_id: str, salt: Optional[str] = None) -> str:
    """Generate a hash for user ID (for anonymization)."""
    if salt is None:
        salt = "luki_default_salt"  # In production, use a proper salt
    
    combined = f"{user_id}:{salt}"
    return hash_string(combined)


def is_valid_uuid(id_string: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(id_string)
        return True
    except ValueError:
        return False


def extract_timestamp_from_id(id_string: str) -> Optional[datetime]:
    """Extract timestamp from IDs that contain timestamps."""
    try:
        # Handle different ID formats
        parts = id_string.split('_')
        
        if len(parts) >= 3:
            # Look for timestamp part (usually numeric)
            for part in parts[1:]:  # Skip the prefix
                if part.isdigit():
                    timestamp = int(part)
                    # Check if it's a reasonable timestamp (after 2020)
                    if timestamp > 1577836800:  # Jan 1, 2020
                        if timestamp > 1e12:  # Milliseconds
                            timestamp = timestamp / 1000
                        return datetime.fromtimestamp(timestamp)
        
        return None
    except (ValueError, OSError):
        return None


def generate_deterministic_id(seed_data: str, prefix: str = "") -> str:
    """Generate a deterministic ID based on seed data."""
    hash_value = hash_string(seed_data)
    short_hash = hash_value[:16]  # Use first 16 characters
    
    if prefix:
        return f"{prefix}_{short_hash}"
    return short_hash


def anonymize_id(original_id: str, salt: str = "luki_anon_salt") -> str:
    """Anonymize an ID for research/analytics purposes."""
    combined = f"{original_id}:{salt}"
    hash_value = hash_string(combined)
    return f"anon_{hash_value[:12]}"


def validate_id_format(id_string: str, expected_prefix: Optional[str] = None) -> bool:
    """Validate ID format and optionally check prefix."""
    if not id_string or not isinstance(id_string, str):
        return False
    
    # Check minimum length
    if len(id_string) < 3:
        return False
    
    # Check prefix if specified
    if expected_prefix:
        if not id_string.startswith(f"{expected_prefix}_"):
            return False
    
    # Check for valid characters (alphanumeric, underscore, dash)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', id_string):
        return False
    
    return True


def generate_checksum(data: Union[str, bytes], algorithm: str = "crc32") -> str:
    """Generate a checksum for data integrity verification."""
    if isinstance(data, str):
        data = data.encode()
    
    if algorithm == "crc32":
        import zlib
        checksum = zlib.crc32(data) & 0xffffffff
        return f"{checksum:08x}"
    elif algorithm in ["md5", "sha1", "sha256", "sha512"]:
        return hashlib.new(algorithm, data).hexdigest()
    else:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")


def split_compound_id(compound_id: str, separator: str = "_") -> list:
    """Split a compound ID into its components."""
    return compound_id.split(separator)


def join_id_parts(*parts: str, separator: str = "_") -> str:
    """Join ID parts into a compound ID."""
    return separator.join(str(part) for part in parts if part)


def generate_hierarchical_id(parent_id: str, child_type: str, child_index: Optional[int] = None) -> str:
    """Generate a hierarchical ID for nested resources."""
    timestamp = int(time.time())
    
    if child_index is not None:
        return f"{parent_id}.{child_type}_{child_index:04d}_{timestamp}"
    else:
        short_id = generate_short_id(4)
        return f"{parent_id}.{child_type}_{timestamp}_{short_id}"


def mask_sensitive_id(id_string: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """Mask sensitive parts of an ID for logging/display."""
    if len(id_string) <= visible_chars * 2:
        return mask_char * len(id_string)
    
    start_chars = id_string[:visible_chars]
    end_chars = id_string[-visible_chars:]
    masked_middle = mask_char * (len(id_string) - visible_chars * 2)
    
    return f"{start_chars}{masked_middle}{end_chars}"


class IDGenerator:
    """ID generator with configurable prefixes and formats."""
    
    def __init__(self, prefix: str = "", use_timestamp: bool = True, use_random: bool = True):
        self.prefix = prefix
        self.use_timestamp = use_timestamp
        self.use_random = use_random
        self._counter = 0
    
    def generate(self, suffix: str = "") -> str:
        """Generate an ID with the configured format."""
        parts = []
        
        if self.prefix:
            parts.append(self.prefix)
        
        if self.use_timestamp:
            parts.append(str(int(time.time())))
        
        if self.use_random:
            parts.append(generate_short_id(6))
        else:
            # Use counter instead of random
            self._counter += 1
            parts.append(f"{self._counter:06d}")
        
        if suffix:
            parts.append(suffix)
        
        return "_".join(parts)
    
    def generate_batch(self, count: int, suffix_template: str = "item_{:04d}") -> list:
        """Generate a batch of IDs."""
        return [self.generate(suffix_template.format(i)) for i in range(count)]


# Pre-configured generators for common use cases
elr_id_generator = IDGenerator("elr", use_timestamp=True, use_random=True)
chunk_id_generator = IDGenerator("chunk", use_timestamp=False, use_random=False)
session_id_generator = IDGenerator("sess", use_timestamp=True, use_random=True)
audit_id_generator = IDGenerator("audit", use_timestamp=True, use_random=True)
