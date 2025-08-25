#!/usr/bin/env python3
"""
User Management API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional
import logging
from datetime import datetime

from ..models import UserCreateRequest, UserResponse, ErrorResponse
from ..auth import get_current_active_user, User
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
    # Only service users can create other users
    if current_user.user_id != "api_service":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to create users"
        )
    
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
    # Users can only access their own info, except service users
    if current_user.user_id != user_id and current_user.user_id != "api_service":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this user's information"
        )
    
    if user_id not in user_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user_store[user_id]


@router.get("/", response_model=List[UserResponse])
async def list_users(
    current_user: User = Depends(get_current_active_user)
):
    """List all users (admin only)."""
    # Only service users can list all users
    if current_user.user_id != "api_service":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to list users"
        )
    
    return list(user_store.values())


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a user."""
    # Only service users can delete users
    if current_user.user_id != "api_service":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete users"
        )
    
    if user_id not in user_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    del user_store[user_id]
    logger.info(f"Deleted user: {user_id}")
    
    return {"message": "User deleted successfully"}
