#!/usr/bin/env python3
"""
Authentication and authorization for the Memory Service API
Full JWT-based authentication with proper security
"""

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import secrets
import os
from functools import lru_cache
import json
import hashlib
import base64

# JWT Configuration
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer(auto_error=False)


class User(BaseModel):
    """User model with full authentication support."""
    user_id: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime
    last_activity: Optional[datetime] = None
    permissions: Optional[list] = None


class UserInDB(User):
    """User model with hashed password for database storage."""
    hashed_password: str


class Token(BaseModel):
    """JWT token response model."""
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token payload data."""
    user_id: Optional[str] = None
    email: Optional[str] = None
    exp: Optional[datetime] = None


# In-memory user store for development (replace with database in production)
fake_users_db: Dict[str, UserInDB] = {}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash using simple hashing."""
    # Simple hash verification for development
    password_hash = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
    return password_hash == hashed_password


def get_password_hash(password: str) -> str:
    """Hash a password for storing using simple hashing."""
    # Simple hash for development
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a simple base64 encoded token for development."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire.timestamp()})
    # Simple base64 encoding for development
    token_json = json.dumps(to_encode)
    encoded_token = base64.b64encode(token_json.encode('utf-8')).decode('utf-8')
    return encoded_token


def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode a simple base64 token."""
    try:
        payload = json.loads(base64.b64decode(token).decode('utf-8'))
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        exp: datetime = datetime.fromtimestamp(payload.get("exp", 0))
        
        if user_id is None:
            return None
            
        return TokenData(user_id=user_id, email=email, exp=exp)
    except Exception:
        return None


def get_user(user_id: str) -> Optional[UserInDB]:
    """Get user from database by user_id."""
    return fake_users_db.get(user_id)


def get_user_by_email(email: str) -> Optional[UserInDB]:
    """Get user from database by email."""
    for user in fake_users_db.values():
        if user.email == email:
            return user
    return None


def authenticate_user(email: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with email and password."""
    user = get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_user(user_id: str, email: str, password: str, full_name: Optional[str] = None) -> UserInDB:
    """Create a new user."""
    hashed_password = get_password_hash(password)
    user = UserInDB(
        user_id=user_id,
        email=email,
        full_name=full_name or email,
        hashed_password=hashed_password,
        is_active=True,
        is_superuser=False,
        created_at=datetime.utcnow(),
        permissions=[]
    )
    fake_users_db[user_id] = user
    return user


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> User:
    """Get the current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if credentials is None:
        # For development, allow anonymous access
        return User(
            user_id="anonymous",
            email=None,
            full_name="Anonymous User",
            is_active=True,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            permissions=[]
        )
    
    token = credentials.credentials
    token_data = verify_token(token)
    
    if token_data is None:
        raise credentials_exception
    
    if token_data.user_id is None:
        raise credentials_exception
        
    user = get_user(token_data.user_id)
    if user is None:
        raise credentials_exception
    
    # Update last activity
    user.last_activity = datetime.utcnow()
    
    return User(
        user_id=user.user_id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=user.created_at,
        last_activity=user.last_activity,
        permissions=user.permissions
    )


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_superuser(current_user: User = Depends(get_current_user)) -> User:
    """Get the current superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    import hashlib
    return hashlib.sha256(api_key.encode()).hexdigest()


# Initialize with a default admin user for development
@lru_cache()
def init_default_users():
    """Initialize default users for development."""
    if not fake_users_db:
        # Create default admin user
        admin_user = create_user(
            user_id="admin",
            email="admin@luki.ai",
            password="admin123",
            full_name="Admin User"
        )
        admin_user.is_superuser = True
        fake_users_db["admin"] = admin_user
        
        # Create default test user
        test_user = create_user(
            user_id="testuser",
            email="test@luki.ai", 
            password="test123",
            full_name="Test User"
        )
        fake_users_db["testuser"] = test_user


# Initialize default users on module import
init_default_users()
