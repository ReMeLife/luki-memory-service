"""Metrics endpoints for LUKi Memory Service.

These endpoints expose high-level activity, mood, and engagement metrics
for use by downstream services like the reporting module. For now they
return empty datasets so callers can safely fall back to demo data.
"""

from datetime import date
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter(prefix="/v1/metrics", tags=["metrics"])


class ActivitiesResponse(BaseModel):
    activities: List[Dict[str, Any]] = Field(default_factory=list)


class MoodResponse(BaseModel):
    mood_entries: List[Dict[str, Any]] = Field(default_factory=list)


class EngagementResponse(BaseModel):
    metrics: List[Dict[str, Any]] = Field(default_factory=list)


@router.get("/activities", response_model=ActivitiesResponse)
async def get_activity_metrics(user_id: str, start_date: date, end_date: date) -> ActivitiesResponse:
    """Return activity metrics for a user over a date range.

    Currently returns an empty list so downstream callers can fall back
    to demo data while the real metrics pipeline is being implemented.
    """
    _ = (user_id, start_date, end_date)
    return ActivitiesResponse()


@router.get("/mood", response_model=MoodResponse)
async def get_mood_metrics(user_id: str, start_date: date, end_date: date) -> MoodResponse:
    """Return mood metrics for a user over a date range.

    Currently returns an empty list so downstream callers can fall back
    to demo data while the real metrics pipeline is being implemented.
    """
    _ = (user_id, start_date, end_date)
    return MoodResponse()


@router.get("/engagement", response_model=EngagementResponse)
async def get_engagement_metrics(user_id: str, start_date: date, end_date: date) -> EngagementResponse:
    """Return engagement metrics for a user over a date range.

    Currently returns an empty list so downstream callers can fall back
    to demo data while the real metrics pipeline is being implemented.
    """
    _ = (user_id, start_date, end_date)
    return EngagementResponse()
