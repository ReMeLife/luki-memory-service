#!/usr/bin/env python3
"""Unit tests for ELR-derived metrics helpers.

These tests exercise the pure aggregation logic in
`luki_memory.metrics.elr_metrics` using a fake ELR store, without touching
ChromaDB or the API layer.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List

import pytest

from luki_memory.metrics import elr_metrics as em


class FakeELRStore:
    """Simple in-memory stand-in for ELRStore.search_user_memories."""

    def __init__(self, memories: List[Dict[str, Any]]):
        self._memories = memories

    def search_user_memories(self, user_id: str, query: str, k: int = 10000, **_: Any) -> List[Dict[str, Any]]:
        # In tests we ignore user_id/query and just return the configured list.
        return list(self._memories)


def _make_memory(
    chunk_id: str,
    ts: datetime,
    content_type: str = "MEMORY",
    section: str | None = None,
    tags: List[str] | None = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "chunk_id": chunk_id,
        "created_at": ts.isoformat(),
        "content_type": content_type,
    }
    if section is not None:
        meta["section"] = section
    if tags is not None:
        meta["tags"] = tags

    return {
        "id": chunk_id,
        "content": "dummy",
        "metadata": meta,
        "similarity_score": 1.0,
        "distance": 0.0,
    }


def test_build_activity_logs_no_memories(monkeypatch: pytest.MonkeyPatch) -> None:
    """If there are no ELR memories, no activity logs should be returned."""
    store = FakeELRStore(memories=[])
    monkeypatch.setattr(em, "get_elr_store", lambda: store)

    records = em.build_activity_logs_from_elr(
        user_id="user_1",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 7),
    )

    assert records == []


def test_build_activity_logs_basic_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single ELR memory within range should produce a single activity log."""
    ts = datetime(2025, 1, 3, 10, 0, 0)
    mem = _make_memory(
        chunk_id="chunk_1",
        ts=ts,
        content_type="INTEREST",
        section="interests",
        tags=["walk", "family"],
    )
    store = FakeELRStore(memories=[mem])
    monkeypatch.setattr(em, "get_elr_store", lambda: store)

    records = em.build_activity_logs_from_elr(
        user_id="user_1",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 7),
    )

    assert len(records) == 1
    rec = records[0]
    assert rec["id"] == "chunk_1"
    assert rec["user_id"] == "user_1"
    assert rec["timestamp"] == ts
    assert isinstance(rec["activity_name"], str) and rec["activity_name"]
    assert isinstance(rec["duration_minutes"], int) and rec["duration_minutes"] > 0
    assert rec["completion_rate"] == 1.0


def test_build_mood_entries_from_mood_like_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memories with mood-related metadata should yield mood entries."""
    ts = datetime(2025, 1, 2, 9, 0, 0)
    mem = _make_memory(
        chunk_id="mood_1",
        ts=ts,
        content_type="HEALTH_NOTE",
        section="health_notes",
        tags=["mood", "anxiety"],
    )
    store = FakeELRStore(memories=[mem])
    monkeypatch.setattr(em, "get_elr_store", lambda: store)

    records = em.build_mood_entries_from_elr(
        user_id="user_1",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 7),
    )

    assert len(records) == 1
    rec = records[0]
    assert rec["id"].startswith("mood_1") or rec["id"].startswith("mood_user_1")
    assert rec["user_id"] == "user_1"
    assert rec["timestamp"] == ts
    assert rec["mood_level"] == "neutral"
    assert rec["source"] == "memory_service_elr_metrics"


def test_build_mood_entries_neutral_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """If there is activity but no explicit mood data, create neutral daily entries."""
    ts = datetime(2025, 1, 5, 15, 0, 0)
    mem = _make_memory(
        chunk_id="act_1",
        ts=ts,
        content_type="MEMORY",
        section="activities",
        tags=["walk"],
    )
    store = FakeELRStore(memories=[mem])
    monkeypatch.setattr(em, "get_elr_store", lambda: store)

    records = em.build_mood_entries_from_elr(
        user_id="user_1",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 7),
    )

    # There should be at least one neutral mood entry for that activity day
    assert len(records) >= 1
    days = {rec["timestamp"].date() for rec in records}
    assert date(2025, 1, 5) in days


def test_build_engagement_metrics_aggregates() -> None:
    """Engagement aggregation should compute per-day totals and averages."""
    d = date(2025, 1, 10)
    ts1 = datetime(2025, 1, 10, 9, 0, 0)
    ts2 = datetime(2025, 1, 10, 17, 0, 0)

    activity_records = [
        {
            "id": "a1",
            "user_id": "user_1",
            "timestamp": ts1,
            "activity_type": "physical",
            "activity_name": "Walk",
            "duration_minutes": 30,
            "engagement_level": "high",
            "completion_rate": 1.0,
            "notes": None,
            "carer_present": False,
        },
        {
            "id": "a2",
            "user_id": "user_1",
            "timestamp": ts2,
            "activity_type": "cognitive",
            "activity_name": "Puzzle",
            "duration_minutes": 45,
            "engagement_level": "moderate",
            "completion_rate": 1.0,
            "notes": None,
            "carer_present": False,
        },
    ]

    mood_records = [
        {
            "id": "m1",
            "user_id": "user_1",
            "timestamp": datetime(2025, 1, 10, 20, 0, 0),
            "mood_level": "high",
            "energy_level": 7,
            "anxiety_level": 3,
            "pain_level": 2,
            "sleep_quality": 8,
            "notes": None,
            "source": "test",
        }
    ]

    metrics = em.build_engagement_metrics_from_elr(
        user_id="user_1",
        start_date=d,
        end_date=d,
        activity_records=activity_records,
        mood_records=mood_records,
    )

    assert len(metrics) == 1
    m = metrics[0]
    assert m["user_id"] == "user_1"
    assert m["date"] == d
    assert m["total_activities"] == 2
    assert m["total_duration_minutes"] == 75
    assert m["cognitive_activities"] == 1
    assert m["physical_activities"] == 1
    assert m["mood_entries"] == 1
    # Average engagement score and mood score should be within [0, 1]
    assert 0.0 <= m["avg_engagement_score"] <= 1.0
    assert m["avg_mood_score"] is not None
    assert 0.0 <= m["avg_mood_score"] <= 1.0
