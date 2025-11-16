#!/usr/bin/env python3
"""API tests for ELR-derived metrics endpoints.

These tests focus on authentication/authorization and basic response
shape for the /v1/metrics endpoints, without depending on a real ELR
store. The underlying aggregation helpers are exercised separately
in unit tests.
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from luki_memory.api.main import app
from luki_memory.api.auth import create_access_token
from luki_memory.api.endpoints import metrics as metrics_module


# Ensure metrics router is included on the main app for these tests.
app.include_router(metrics_module.router)
client = TestClient(app)


def _auth_headers(user_id: str) -> dict:
    """Create Authorization headers for a given user_id."""
    token_data = {"sub": user_id, "email": f"{user_id}@example.com"}
    token = create_access_token(token_data)
    return {"Authorization": f"Bearer {token}"}


def test_activity_metrics_requires_authorization_for_other_user() -> None:
    """Requests for another user's metrics should be forbidden."""
    params = {
        "user_id": "target_user_123",
        "start_date": "2025-01-01",
        "end_date": "2025-01-07",
    }

    # Auth token for a different user
    headers = _auth_headers("other_user_456")

    response = client.get("/v1/metrics/activities", params=params, headers=headers)

    assert response.status_code == 403


def test_activity_metrics_allows_same_user_and_returns_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same-user requests should succeed and return DTO-shaped data."""

    def _fake_build_activity_logs_from_elr(user_id: str, start_date, end_date):
        return [
            {
                "id": "chunk_1",
                "user_id": user_id,
                "timestamp": datetime(2025, 1, 3, 10, 0, 0),
                "activity_type": "physical",
                "activity_name": "Walk",
                "duration_minutes": 30,
                "engagement_level": "high",
                "completion_rate": 1.0,
                "notes": None,
                "carer_present": False,
            }
        ]

    monkeypatch.setattr(
        metrics_module,
        "build_activity_logs_from_elr",
        _fake_build_activity_logs_from_elr,
    )

    user_id = "test_user_123"
    params = {
        "user_id": user_id,
        "start_date": "2025-01-01",
        "end_date": "2025-01-07",
    }
    headers = _auth_headers(user_id)

    response = client.get("/v1/metrics/activities", params=params, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert "activities" in data
    assert len(data["activities"]) == 1
    activity = data["activities"][0]
    assert activity["user_id"] == user_id
    assert activity["activity_name"] == "Walk"


def test_activity_metrics_invalid_date_range_returns_empty() -> None:
    """If end_date is before start_date, an empty list is returned."""
    user_id = "test_user_123"
    params = {
        "user_id": user_id,
        "start_date": "2025-01-10",
        "end_date": "2025-01-01",
    }
    headers = _auth_headers(user_id)

    response = client.get("/v1/metrics/activities", params=params, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["activities"] == []
