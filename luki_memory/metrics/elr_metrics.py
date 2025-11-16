"""ELR-derived metrics aggregation helpers.

These helpers read user ELR chunks from the ELRStore (ChromaDB-backed)
using only metadata (not raw content) and aggregate them into
activity, mood, and engagement metric records. The calling API layer
is responsible for converting these plain records into Pydantic DTOs.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time
from typing import Any, Dict, Iterable, List, Optional

from ..storage.elr_store import get_elr_store


def _parse_metadata_datetime(metadata: Dict[str, Any]) -> Optional[datetime]:
    """Best-effort parsing of a timestamp from ELR metadata.

    Prefers the original ELR "timestamp" if present, otherwise falls
    back to the ingestion "created_at" field. Values are stored as
    ISO 8601 strings in ChromaDB metadata.
    """
    candidates: Iterable[str] = []
    raw_timestamp = metadata.get("timestamp")
    raw_created = metadata.get("created_at")

    values: List[str] = []
    if raw_timestamp:
        values.append(str(raw_timestamp))
    if raw_created:
        values.append(str(raw_created))

    for raw in values:
        value = raw.strip()
        if not value:
            continue
        # Normalise common suffixes like trailing "Z"
        normalised = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalised)
        except Exception:
            # If parsing fails, try dropping timezone information
            try:
                if "+" in normalised:
                    base = normalised.split("+", 1)[0]
                elif "-" in normalised[10:]:
                    # Handle forms like 2024-01-02T12:34:56-05:00
                    base = normalised.split("-", 3)[0]
                else:
                    base = normalised
                return datetime.fromisoformat(base)
            except Exception:
                continue
    return None


def _infer_activity_type(content_type: str, section: str, tags: List[str]) -> str:
    """Infer a coarse activity type label from ELR metadata.

    Returns a string matching ActivityType enum values.
    """
    text = " ".join([content_type, section] + tags).lower()

    if any(key in text for key in ["walk", "exercise", "gym", "yoga", "sport", "physio"]):
        return "physical"
    if any(key in text for key in ["puzzle", "reading", "book", "learn", "study", "game"]):
        return "cognitive"
    if any(key in text for key in ["friend", "family", "visit", "call", "social", "party"]):
        return "social"
    if any(key in text for key in ["art", "music", "painting", "craft", "creative"]):
        return "creative"
    if any(key in text for key in ["hobby", "interests", "fun", "leisure", "recreation"]):
        return "recreational"
    if any(key in text for key in ["therapy", "therapeutic", "health", "medication", "clinic"]):
        return "therapeutic"

    # Default: treat as daily living / general activity
    return "daily_living"


def _infer_engagement_level(content_type: str, section: str, tags: List[str]) -> str:
    """Rough engagement level heuristic based on metadata only.

    Returns a string matching EngagementLevel enum values.
    """
    text = " ".join([content_type, section] + tags).lower()

    if any(key in text for key in ["favorite", "favourite", "love", "enjoy", "passion"]):
        return "very_high"
    if any(key in text for key in ["often", "regular", "routine", "daily"]):
        return "high"
    if any(key in text for key in ["sometimes", "occasional", "now and then"]):
        return "moderate"
    if any(key in text for key in ["rare", "seldom", "hardly"]):
        return "low"
    return "moderate"


def _looks_like_mood_related(content_type: str, section: str, tags: List[str]) -> bool:
    text = " ".join([content_type, section] + tags).lower()
    if any(key in text for key in ["mood", "feeling", "emotion", "wellbeing", "well-being"]):
        return True
    if any(key in text for key in ["health", "mental", "depression", "anxiety", "stress"]):
        return True
    return False


def _default_mood_level() -> str:
    # Without real sentiment, stay conservative and neutral.
    return "neutral"


def _mood_level_to_score(level: str) -> float:
    level = level.lower()
    if level == "very_low":
        return 0.0
    if level == "low":
        return 0.25
    if level == "neutral":
        return 0.5
    if level == "high":
        return 0.75
    if level == "very_high":
        return 1.0
    return 0.5


def _normalise_tags(raw_tags: Any) -> List[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        return [raw_tags]
    if isinstance(raw_tags, (list, tuple, set)):
        return [str(t) for t in raw_tags]
    return [str(raw_tags)]


def _get_memories_in_range(user_id: str, start_dt: datetime, end_dt: datetime, max_items: int = 10000) -> List[Dict[str, Any]]:
    """Fetch all ELR chunks for a user whose metadata timestamp is in range.

    Uses ELRStore.search_user_memories with an empty query to retrieve
    all chunks for the user, then filters by timestamp.
    """
    store = get_elr_store()
    # search_user_memories with empty query returns all user chunks from Chroma
    all_memories = store.search_user_memories(user_id=user_id, query="", k=max_items)

    in_range: List[Dict[str, Any]] = []
    for mem in all_memories:
        meta = mem.get("metadata") or {}
        ts = _parse_metadata_datetime(meta)
        if ts is None:
            continue
        if start_dt <= ts <= end_dt:
            in_range.append(mem)
    return in_range


def build_activity_logs_from_elr(user_id: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
    """Build ActivityLogDTO-compatible records from ELR metadata.

    Returns a list of dicts keyed exactly like ActivityLogDTO fields,
    but with primitive Python types (strings, ints, floats, datetimes).
    """
    if end_date < start_date:
        return []

    start_dt = datetime.combine(start_date, time.min)
    end_dt = datetime.combine(end_date, time.max)

    memories = _get_memories_in_range(user_id, start_dt, end_dt)
    activities: List[Dict[str, Any]] = []

    for mem in memories:
        meta = mem.get("metadata") or {}
        ts = _parse_metadata_datetime(meta)
        if ts is None:
            continue

        content_type = str(meta.get("content_type") or "").lower()
        section = str(meta.get("section") or "").lower()
        tags = _normalise_tags(meta.get("tags"))

        activity_type = _infer_activity_type(content_type, section, tags)
        engagement_level = _infer_engagement_level(content_type, section, tags)

        # Derive a safe, non-PII activity name from metadata only
        title = str(meta.get("title") or "").strip()
        if title:
            activity_name = title
        elif section:
            activity_name = f"ELR {section.replace('_', ' ')} entry"
        elif content_type:
            activity_name = f"ELR {content_type.replace('_', ' ')} entry"
        else:
            activity_name = "ELR memory"

        # Use a fixed nominal duration; can be refined later
        duration_minutes = 30
        completion_rate = 1.0

        activities.append(
            {
                "id": str(meta.get("chunk_id") or f"{user_id}_{ts.isoformat()}"),
                "user_id": user_id,
                "timestamp": ts,
                "activity_type": activity_type,
                "activity_name": activity_name,
                "duration_minutes": duration_minutes,
                "engagement_level": engagement_level,
                "completion_rate": completion_rate,
                "notes": None,
                "carer_present": False,
            }
        )

    # Sort by time for deterministic output
    activities.sort(key=lambda a: a["timestamp"])
    return activities


def build_mood_entries_from_elr(user_id: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
    """Build MoodEntryDTO-compatible records from ELR metadata.

    Currently uses coarse heuristics and defaults to neutral mood
    when no explicit mood information is encoded in metadata.
    """
    if end_date < start_date:
        return []

    start_dt = datetime.combine(start_date, time.min)
    end_dt = datetime.combine(end_date, time.max)

    memories = _get_memories_in_range(user_id, start_dt, end_dt)
    mood_entries: List[Dict[str, Any]] = []

    for mem in memories:
        meta = mem.get("metadata") or {}
        ts = _parse_metadata_datetime(meta)
        if ts is None:
            continue

        content_type = str(meta.get("content_type") or "").lower()
        section = str(meta.get("section") or "").lower()
        tags = _normalise_tags(meta.get("tags"))

        if not _looks_like_mood_related(content_type, section, tags):
            continue

        mood_level = _default_mood_level()

        mood_entries.append(
            {
                "id": str(meta.get("chunk_id") or f"mood_{user_id}_{ts.isoformat()}"),
                "user_id": user_id,
                "timestamp": ts,
                "mood_level": mood_level,
                # Use mid-range defaults until richer sentiment is available
                "energy_level": 5,
                "anxiety_level": 5,
                "pain_level": 5,
                "sleep_quality": 5,
                "notes": None,
                "source": "memory_service_elr_metrics",
            }
        )

    # If no explicit mood-like memories exist but there is ELR activity
    # in the range, emit one neutral mood entry per active day so the
    # reporting module has something real (albeit coarse) to work with.
    if not mood_entries and memories:
        days_with_activity = sorted({
            (_parse_metadata_datetime(mem.get("metadata") or {}) or start_dt).date()
            for mem in memories
        })
        for d in days_with_activity:
            ts = datetime.combine(d, time(hour=9))
            mood_entries.append(
                {
                    "id": f"mood_{user_id}_{d.isoformat()}",
                    "user_id": user_id,
                    "timestamp": ts,
                    "mood_level": _default_mood_level(),
                    "energy_level": 5,
                    "anxiety_level": 5,
                    "pain_level": 5,
                    "sleep_quality": 5,
                    "notes": None,
                    "source": "memory_service_elr_metrics",
                }
            )

    mood_entries.sort(key=lambda m: m["timestamp"])
    return mood_entries


def build_engagement_metrics_from_elr(
    user_id: str,
    start_date: date,
    end_date: date,
    activity_records: Optional[List[Dict[str, Any]]] = None,
    mood_records: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build EngagementMetricDTO-compatible records from ELR metadata.

    Aggregates per-day totals based on activity and mood records.
    If precomputed activity/mood records are not provided, they will
    be generated from ELR first.
    """
    if end_date < start_date:
        return []

    if activity_records is None:
        activity_records = build_activity_logs_from_elr(user_id, start_date, end_date)
    if mood_records is None:
        mood_records = build_mood_entries_from_elr(user_id, start_date, end_date)

    by_date: Dict[date, Dict[str, Any]] = defaultdict(lambda: {
        "total_activities": 0,
        "total_duration_minutes": 0,
        "avg_engagement_score": 0.0,
        "social_interactions": 0,
        "family_engagement_minutes": 0,
        "cognitive_activities": 0,
        "physical_activities": 0,
        "mood_entries": 0,
        "mood_score_sum": 0.0,
    })

    for act in activity_records:
        ts: datetime = act["timestamp"]
        d = ts.date()
        if d < start_date or d > end_date:
            continue
        bucket = by_date[d]
        bucket["total_activities"] += 1
        bucket["total_duration_minutes"] += int(act.get("duration_minutes", 0))

        atype = str(act.get("activity_type") or "").lower()
        if atype == "cognitive":
            bucket["cognitive_activities"] += 1
        if atype == "physical":
            bucket["physical_activities"] += 1

        # Treat social / family oriented activities as social interactions
        if atype in {"social", "recreational", "therapeutic"}:
            bucket["social_interactions"] += 1
            # Nominally attribute some minutes as "family engagement" too
            bucket["family_engagement_minutes"] += int(act.get("duration_minutes", 0)) // 2

        # Map engagement level enum label to a [0, 1] score
        engagement = str(act.get("engagement_level") or "moderate").lower()
        if engagement == "very_high":
            score = 1.0
        elif engagement == "high":
            score = 0.8
        elif engagement == "moderate":
            score = 0.6
        elif engagement == "low":
            score = 0.3
        else:
            score = 0.0

        # Running average: store as summed score then divide later
        current_total = bucket.get("_engagement_score_sum", 0.0)
        sample_count = bucket.get("_engagement_samples", 0)
        bucket["_engagement_score_sum"] = current_total + score
        bucket["_engagement_samples"] = sample_count + 1

    for mood in mood_records:
        ts: datetime = mood["timestamp"]
        d = ts.date()
        if d < start_date or d > end_date:
            continue
        bucket = by_date[d]
        bucket["mood_entries"] += 1
        level = str(mood.get("mood_level") or "neutral")
        bucket["mood_score_sum"] += _mood_level_to_score(level)

    metrics: List[Dict[str, Any]] = []
    for d in sorted(by_date.keys()):
        bucket = by_date[d]
        total_acts = bucket["total_activities"]

        # Finalise engagement score
        if bucket.get("_engagement_samples", 0) > 0:
            avg_engagement = bucket["_engagement_score_sum"] / bucket["_engagement_samples"]
        else:
            # If there were activities but no engagement labels, fall back to mid-range
            avg_engagement = 0.6 if total_acts > 0 else 0.0

        # Finalise mood score
        if bucket["mood_entries"] > 0:
            avg_mood = bucket["mood_score_sum"] / bucket["mood_entries"]
        else:
            avg_mood = None

        metrics.append(
            {
                "id": f"engagement_{user_id}_{d.isoformat()}",
                "user_id": user_id,
                "date": d,
                "total_activities": total_acts,
                "total_duration_minutes": int(bucket["total_duration_minutes"]),
                "avg_engagement_score": float(avg_engagement),
                "social_interactions": int(bucket["social_interactions"]),
                "family_engagement_minutes": int(bucket["family_engagement_minutes"]),
                "cognitive_activities": int(bucket["cognitive_activities"]),
                "physical_activities": int(bucket["physical_activities"]),
                "mood_entries": int(bucket["mood_entries"]),
                "avg_mood_score": float(avg_mood) if avg_mood is not None else None,
            }
        )

    return metrics
