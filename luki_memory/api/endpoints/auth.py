#!/usr/bin/env python3
"""
Authentication API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import base64
import json
import hashlib
import os

from ..models import (
    LoginRequest, LoginResponse, UserResponse,
    TokenResponse, ErrorResponse
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

class UserInDB(User):
    """User model with hashed password for database storage."""
    hashed_password: str

# JWT Configuration
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer(auto_error=False)

# In-memory user store for development
fake_users_db: Dict[str, UserInDB] = {
    "admin": UserInDB(
        user_id="admin",
        email="admin@luki.ai",
        full_name="Admin User",
        hashed_password=hashlib.sha256("admin123".encode('utf-8')).hexdigest(),
        is_active=True,
        is_superuser=True,
        created_at=datetime.utcnow(),
        permissions=[]
    )
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash using simple hashing."""
    password_hash = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
    return password_hash == hashed_password

def authenticate_user(email: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with email and password."""
    for user in fake_users_db.values():
        if user.email == email:
            if verify_password(password, user.hashed_password):
                return user
    return None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a simple base64 encoded token for development."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire.timestamp()})
    token_json = json.dumps(to_encode)
    encoded_token = base64.b64encode(token_json.encode('utf-8')).decode('utf-8')
    return encoded_token

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
            
        user_id: str = payload.get("sub", "anonymous")
        email: str = payload.get("email")
        
        return User(
            user_id=user_id,
            email=email,
            full_name=email or "User",
            is_active=True,
            is_superuser=(user_id == "admin"),
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

from ..config import get_settings

router = APIRouter(prefix="/auth", tags=["authentication"])
settings = get_settings()

class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    token_type: str
    expires_in: int


class DevTokenRequest(BaseModel):
    """Development token request model."""
    user_id: str
    email: Optional[str] = None
    full_name: Optional[str] = None


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and return JWT token.
    """
    if not request.username or not request.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password are required"
        )
    
    # Authenticate user with email/password
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create token data
    token_data = {
        "sub": user.user_id,
        "email": user.email,
        "full_name": user.full_name
    }
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data=token_data,
        expires_delta=access_token_expires
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60
    )


@router.post("/dev-token", response_model=LoginResponse)
async def create_dev_token(request: DevTokenRequest):
    """
    Create a development JWT token for testing.
    
    This endpoint is for development/testing only and should be
    disabled in production environments.
    """
    if not settings.debug_mode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Development endpoints not available"
        )
    
    # Create token data
    token_data = {
        "sub": request.user_id,
        "email": request.email or f"{request.user_id}@dev.local",
        "full_name": request.full_name or f"Dev User {request.user_id}"
    }
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data=token_data,
        expires_delta=access_token_expires
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60
    )


@router.post("/service-token", response_model=LoginResponse)
async def create_service_token():
    """
    Create a service token for API-to-API communication.
    
    This creates a token with 'api_service' user_id that can
    access resources for any user.
    """
    token_data = {
        "sub": "api_service",
        "email": "service@luki.local",
        "full_name": "API Service User"
    }
    
    # Service tokens have longer expiration
    access_token_expires = timedelta(hours=24)
    access_token = create_access_token(
        data=token_data,
        expires_delta=access_token_expires
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=24 * 60 * 60  # 24 hours in seconds
    )
