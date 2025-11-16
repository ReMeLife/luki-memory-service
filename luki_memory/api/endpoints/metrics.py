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

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ...metrics.elr_metrics import (
    build_activity_logs_from_elr,
    build_mood_entries_from_elr,
    build_engagement_metrics_from_elr,
)


router = APIRouter(prefix="/v1/metrics", tags=["metrics"])


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
async def get_activity_metrics(user_id: str, start_date: date, end_date: date) -> ActivitiesResponse:
    """Return activity metrics for a user over a date range.

    Activity logs are derived from ELR metadata for the given user and
    date range. If no memories match, an empty list is returned.
    """
    if end_date < start_date:
        return ActivitiesResponse(activities=[])

    records = build_activity_logs_from_elr(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )
    activities = [ActivityLogDTO(**record) for record in records]

    return ActivitiesResponse(activities=activities)


@router.get("/mood", response_model=MoodResponse)
async def get_mood_metrics(user_id: str, start_date: date, end_date: date) -> MoodResponse:
    """Return mood metrics for a user over a date range.

    Mood entries are inferred from ELR metadata (e.g. health or mood
    related sections) for the given user and date range. If no
    mood-related memories are available, an empty list is returned.
    """
    if end_date < start_date:
        return MoodResponse(mood_entries=[])

    records = build_mood_entries_from_elr(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )
    entries = [MoodEntryDTO(**record) for record in records]

    return MoodResponse(mood_entries=entries)


@router.get("/engagement", response_model=EngagementResponse)
async def get_engagement_metrics(user_id: str, start_date: date, end_date: date) -> EngagementResponse:
    """Return engagement metrics for a user over a date range.

    Engagement metrics are aggregated per day from ELR-derived activity
    and mood records for the given user. If no underlying activity exists
    in the requested window, an empty metrics list is returned.
    """
    if end_date < start_date:
        return EngagementResponse(metrics=[])

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
    metrics = [EngagementMetricDTO(**record) for record in records]

    return EngagementResponse(metrics=metrics)
