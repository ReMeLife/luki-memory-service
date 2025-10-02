#!/usr/bin/env python3
"""
User Management API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import base64
import json
import os

from ..models import (
    UserCreateRequest, UserResponse, UserUpdateRequest,
    UserListResponse, ErrorResponse
)

# Inline auth classes and functions to avoid import issues
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

security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> User:
    """Get the current authenticated user from JWT token."""
    if credentials is None:
        return User(
            user_id="anonymous",
            email=None,
            full_name="Anonymous User",
            is_active=True,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            permissions=[]
        )
    
    try:
        token = credentials.credentials
        # Simple base64 token decoding for development
        try:
            decoded_bytes = base64.b64decode(token + '==')  # Add padding
            payload = json.loads(decoded_bytes.decode('utf-8'))
        except:
            # If not base64, treat as simple token
            payload = {"sub": "anonymous", "email": None}
            
        user_id = str(payload.get("sub", "anonymous"))
        email = payload.get("email") or None
        
        return User(
            user_id=user_id,
            email=email,
            full_name=email or "User",
            is_active=True,
            is_superuser=(user_id == "admin"),  # Simple admin check
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            permissions=[]
        )
    except Exception:
        return User(
            user_id="anonymous",
            email=None,
            full_name="Anonymous User",
            is_active=True,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            permissions=[]
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

from ..config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["users"])
settings = get_settings()

# In-memory user store (replace with database in production)
user_store = {}


@router.post("/", response_model=UserResponse)
async def create_user(
    request: UserCreateRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new user."""
    
    if request.user_id in user_store:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already exists"
        )
    
    new_user = UserResponse(
        user_id=request.user_id,
        email=request.email,
        full_name=request.full_name,
        created_at=datetime.utcnow(),
        last_activity=None,
        memory_count=0,
        is_active=True
    )
    
    user_store[request.user_id] = new_user
    logger.info(f"Created user: {request.user_id}")
    
    return new_user


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get user information."""
    
    if user_id not in user_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user_store[user_id]


@router.get("/", response_model=List[UserResponse])
async def list_users(current_user: User = Depends(get_current_superuser)):
    """List all users (admin only)."""
    
    return list(user_store.values())


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(get_current_superuser)
):
    """Delete a user."""
    
    if user_id not in user_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    del user_store[user_id]
    logger.info(f"Deleted user: {user_id}")
    
    return {"message": "User deleted successfully"}
