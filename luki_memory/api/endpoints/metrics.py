"""Metrics endpoints for LUKi Memory Service.

These endpoints expose high-level activity, mood, and engagement metrics
for use by downstream services like the reporting module. Metrics are
derived from ELR chunks stored in the ELR store (ChromaDB) using only
metadata and timestamps. If no relevant memories exist for the requested
date range, the responses contain empty lists so callers can fall back to
their own demo or default data.
"""

from datetime import date, datetime
from typing import Any, Dict, List
from enum import Enum
import base64
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from ..policy_client import enforce_policy_scopes
from ...metrics.elr_metrics import (
    build_activity_logs_from_elr,
    build_mood_entries_from_elr,
    build_engagement_metrics_from_elr,
)


router = APIRouter(prefix="/v1/metrics", tags=["metrics"])

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


class User(BaseModel):
    user_id: str
    email: str | None = None
    full_name: str | None = None
    is_active: bool = True
    is_superuser: bool = False


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> User:
    """Decode a simple base64 bearer token into a User model.

    Mirrors the lightweight development auth model used in other endpoints
    so that service tokens from /auth/service-token (sub "api_service") work
    for module-to-module calls, while anonymous access remains possible for
    local testing.
    """
    if credentials is None:
        return User(
            user_id="anonymous",
            email=None,
            full_name="Anonymous User",
            is_active=True,
            is_superuser=False,
        )

    try:
        token = credentials.credentials
        try:
            decoded_bytes = base64.b64decode(token + "==")
            payload = json.loads(decoded_bytes.decode("utf-8"))
        except Exception:
            payload = {"sub": "anonymous", "email": None}

        user_id = str(payload.get("sub", "anonymous"))
        email = payload.get("email") or None
        is_superuser = user_id in {"admin", "api_service"}

        return User(
            user_id=user_id,
            email=email,
            full_name=email or "User",
            is_active=True,
            is_superuser=is_superuser,
        )
    except Exception:
        return User(
            user_id="anonymous",
            email=None,
            full_name="Anonymous User",
            is_active=True,
            is_superuser=False,
        )


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Ensure the current user is active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    return current_user


def _authorize_metrics_access(requested_user_id: str, current_user: User) -> None:
    """Enforce simple user isolation for metrics.

    The "api_service" and "admin" identities are allowed to read metrics for
    any user to support backend modules. Regular users may only access their
    own metrics.
    """
    if current_user.user_id in {requested_user_id, "admin", "api_service"}:
        return

    logger.warning(
        "Metrics access denied",
        extra={"requested_user_id": requested_user_id, "current_user_id": current_user.user_id},
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not authorized to access metrics for this user",
    )


class MoodLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    NEUTRAL = "neutral"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ActivityType(str, Enum):
    PHYSICAL = "physical"
    COGNITIVE = "cognitive"
    SOCIAL = "social"
    CREATIVE = "creative"
    RECREATIONAL = "recreational"
    THERAPEUTIC = "therapeutic"
    DAILY_LIVING = "daily_living"


class EngagementLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ActivityLogDTO(BaseModel):
    id: str
    user_id: str
    timestamp: datetime
    activity_type: ActivityType
    activity_name: str
    duration_minutes: int
    engagement_level: EngagementLevel
    completion_rate: float
    notes: str | None = None
    carer_present: bool = False


class MoodEntryDTO(BaseModel):
    id: str
    user_id: str
    timestamp: datetime
    mood_level: MoodLevel
    energy_level: int
    anxiety_level: int
    pain_level: int
    sleep_quality: int
    notes: str | None = None
    source: str = "memory_service_demo"


class EngagementMetricDTO(BaseModel):
    id: str
    user_id: str
    date: date
    total_activities: int
    total_duration_minutes: int
    avg_engagement_score: float
    social_interactions: int
    family_engagement_minutes: int
    cognitive_activities: int
    physical_activities: int
    mood_entries: int
    avg_mood_score: float | None = None


class ActivitiesResponse(BaseModel):
    activities: List[ActivityLogDTO] = Field(default_factory=list)


class MoodResponse(BaseModel):
    mood_entries: List[MoodEntryDTO] = Field(default_factory=list)


class EngagementResponse(BaseModel):
    metrics: List[EngagementMetricDTO] = Field(default_factory=list)


@router.get("/activities", response_model=ActivitiesResponse)
async def get_activity_metrics(
    user_id: str,
    start_date: date,
    end_date: date,
    current_user: User = Depends(get_current_active_user),
) -> ActivitiesResponse:
    """Return activity metrics for a user over a date range.

    Activity logs are derived from ELR metadata for the given user and
    date range. If no memories match, an empty list is returned.
    """
    _authorize_metrics_access(user_id, current_user)

    if end_date < start_date:
        return ActivitiesResponse(activities=[])

    policy_result = await enforce_policy_scopes(
        user_id=user_id,
        requested_scopes=["analytics"],
        requester_role="memory_service",
        context={"operation": "get_activity_metrics"},
    )
    if not policy_result.get("allowed", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient consent to access analytics for this user",
        )

    try:
        records = build_activity_logs_from_elr(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Failed to build activity metrics from ELR: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute activity metrics",
        ) from exc

    activities = [ActivityLogDTO(**record) for record in records]

    return ActivitiesResponse(activities=activities)


@router.get("/mood", response_model=MoodResponse)
async def get_mood_metrics(
    user_id: str,
    start_date: date,
    end_date: date,
    current_user: User = Depends(get_current_active_user),
) -> MoodResponse:
    """Return mood metrics for a user over a date range.

    Mood entries are inferred from ELR metadata (e.g. health or mood
    related sections) for the given user and date range. If no
    mood-related memories are available, an empty list is returned.
    """
    _authorize_metrics_access(user_id, current_user)

    if end_date < start_date:
        return MoodResponse(mood_entries=[])

    policy_result = await enforce_policy_scopes(
        user_id=user_id,
        requested_scopes=["analytics"],
        requester_role="memory_service",
        context={"operation": "get_mood_metrics"},
    )
    if not policy_result.get("allowed", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient consent to access analytics for this user",
        )

    try:
        records = build_mood_entries_from_elr(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Failed to build mood metrics from ELR: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute mood metrics",
        ) from exc

    entries = [MoodEntryDTO(**record) for record in records]

    return MoodResponse(mood_entries=entries)


@router.get("/engagement", response_model=EngagementResponse)
async def get_engagement_metrics(
    user_id: str,
    start_date: date,
    end_date: date,
    current_user: User = Depends(get_current_active_user),
) -> EngagementResponse:
    """Return engagement metrics for a user over a date range.

    Engagement metrics are aggregated per day from ELR-derived activity
    and mood records for the given user. If no underlying activity exists
    in the requested window, an empty metrics list is returned.
    """
    _authorize_metrics_access(user_id, current_user)

    if end_date < start_date:
        return EngagementResponse(metrics=[])

    policy_result = await enforce_policy_scopes(
        user_id=user_id,
        requested_scopes=["analytics"],
        requester_role="memory_service",
        context={"operation": "get_engagement_metrics"},
    )
    if not policy_result.get("allowed", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient consent to access analytics for this user",
        )

    try:
        activity_records = build_activity_logs_from_elr(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        mood_records = build_mood_entries_from_elr(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        records = build_engagement_metrics_from_elr(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            activity_records=activity_records,
            mood_records=mood_records,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Failed to build engagement metrics from ELR: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute engagement metrics",
        ) from exc

    metrics = [EngagementMetricDTO(**record) for record in records]

    return EngagementResponse(metrics=metrics)
