"""Metrics endpoints for LUKi Memory Service.

These endpoints expose high-level activity, mood, and engagement metrics
for use by downstream services like the reporting module. For now they
return empty datasets so callers can safely fall back to demo data.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List
from enum import Enum
from uuid import uuid4
import random

from fastapi import APIRouter
from pydantic import BaseModel, Field


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

    Currently returns an empty list so downstream callers can fall back
    to demo data while the real metrics pipeline is being implemented.
    """
    if end_date < start_date:
        return ActivitiesResponse(activities=[])

    activities: List[ActivityLogDTO] = []
    # Simple demo activity templates reused across days
    activity_templates = [
        ("Morning Walk", ActivityType.PHYSICAL),
        ("Crossword Puzzle", ActivityType.COGNITIVE),
        ("Family Video Call", ActivityType.SOCIAL),
    ]

    current = start_date
    while current <= end_date:
        for idx, (name, activity_type) in enumerate(activity_templates):
            hour = 9 + idx * 3
            ts = datetime.combine(current, datetime.min.time()).replace(hour=hour)
            duration = 30 + idx * 15
            engagement = random.choice(list(EngagementLevel))
            completion = round(0.7 + 0.1 * idx, 2)
            activities.append(
                ActivityLogDTO(
                    id=str(uuid4()),
                    user_id=user_id,
                    timestamp=ts,
                    activity_type=activity_type,
                    activity_name=name,
                    duration_minutes=duration,
                    engagement_level=engagement,
                    completion_rate=min(completion, 1.0),
                    notes="Generated demo activity log from memory service metrics endpoint",
                    carer_present=(idx == 0),
                )
            )
        current += timedelta(days=1)

    return ActivitiesResponse(activities=activities)


@router.get("/mood", response_model=MoodResponse)
async def get_mood_metrics(user_id: str, start_date: date, end_date: date) -> MoodResponse:
    """Return mood metrics for a user over a date range.

    Currently returns an empty list so downstream callers can fall back
    to demo data while the real metrics pipeline is being implemented.
    """
    if end_date < start_date:
        return MoodResponse(mood_entries=[])

    entries: List[MoodEntryDTO] = []
    current = start_date
    while current <= end_date:
        for hour, label in ((9, "morning"), (18, "evening")):
            ts = datetime.combine(current, datetime.min.time()).replace(hour=hour)
            entries.append(
                MoodEntryDTO(
                    id=str(uuid4()),
                    user_id=user_id,
                    timestamp=ts,
                    mood_level=random.choice(list(MoodLevel)),
                    energy_level=random.randint(3, 8),
                    anxiety_level=random.randint(2, 6),
                    pain_level=random.randint(1, 4),
                    sleep_quality=random.randint(5, 9),
                    notes=f"Generated demo mood entry ({label})",
                )
            )
        current += timedelta(days=1)

    return MoodResponse(mood_entries=entries)


@router.get("/engagement", response_model=EngagementResponse)
async def get_engagement_metrics(user_id: str, start_date: date, end_date: date) -> EngagementResponse:
    """Return engagement metrics for a user over a date range.

    Currently returns an empty list so downstream callers can fall back
    to demo data while the real metrics pipeline is being implemented.
    """
    if end_date < start_date:
        return EngagementResponse(metrics=[])

    metrics: List[EngagementMetricDTO] = []
    current = start_date
    while current <= end_date:
        total_activities = random.randint(2, 6)
        total_duration = random.randint(60, 240)
        social_interactions = random.randint(0, 4)
        family_minutes = random.randint(0, max(30, total_duration // 2))
        cognitive = random.randint(0, total_activities)
        physical = random.randint(0, max(0, total_activities - cognitive))
        mood_count = random.randint(1, 3)
        avg_engagement = round(random.uniform(0.6, 0.95), 3)
        avg_mood_score = round(random.uniform(0.5, 0.9), 3)

        metrics.append(
            EngagementMetricDTO(
                id=str(uuid4()),
                user_id=user_id,
                date=current,
                total_activities=total_activities,
                total_duration_minutes=total_duration,
                avg_engagement_score=avg_engagement,
                social_interactions=social_interactions,
                family_engagement_minutes=family_minutes,
                cognitive_activities=cognitive,
                physical_activities=physical,
                mood_entries=mood_count,
                avg_mood_score=avg_mood_score,
            )
        )

        current += timedelta(days=1)

    return EngagementResponse(metrics=metrics)
