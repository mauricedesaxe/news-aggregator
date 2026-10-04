"""Read durable video state for the periodic incident monitor."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

from romanian_news import BUCHAREST
from romanian_news.catalog.video_digest import read_stalled_fal_queue_incidents
from romanian_news.catalog_transport import catalog_query
from romanian_news.identity import canonical_json, sha256
from romanian_news.video_digest.incidents import VideoIncident, VideoIncidentCategory

MONITOR_LOOKBACK = timedelta(days=7)
PUBLICATION_DELAY = timedelta(minutes=60)


def read_due_video_incidents(now: datetime) -> tuple[VideoIncident, ...]:
    if now.tzinfo is None:
        raise ValueError("Video incident monitor time must have a timezone")
    current = now.astimezone(UTC)
    since = current - MONITOR_LOOKBACK
    failed = catalog_query(
        """SELECT slot_id, failure_reason FROM video_digest_slots
           WHERE stage = 'failed' AND scheduled_at >= %s""",
        [since],
    )
    late = catalog_query(
        """SELECT slot_id FROM video_digest_slots
           WHERE edition_id IS NOT NULL
             AND stage NOT IN ('published', 'skipped', 'failed')
             AND scheduled_at <= %s AND scheduled_at >= %s""",
        [current - PUBLICATION_DELAY, since],
    )
    conflicts = catalog_query(
        """SELECT publication_id FROM video_digest_publication_intents
           WHERE stage = 'conflict' AND updated_at >= %s""",
        [since],
    )
    success_gap = catalog_query(
        """SELECT slot.slot_id
           FROM video_digest_slot_sources source
           JOIN video_digest_slots slot ON slot.slot_id = source.slot_id
           WHERE source.selected_at <= %s
             AND source.selection IS NOT NULL
             AND jsonb_array_length(source.selection->'selected_sections') > 0
             AND slot.stage <> 'skipped'
             AND NOT EXISTS (
                 SELECT 1 FROM video_digest_publication_intents publication
                 WHERE publication.stage = 'published'
                   AND publication.published_at >= source.selected_at
             )
           ORDER BY source.selected_at, slot.slot_id
           LIMIT 1""",
        [current - timedelta(hours=24)],
    )
    day = current.astimezone(BUCHAREST).date()
    budget = catalog_query(
        """SELECT scope_kind, scope_key
           FROM video_digest_generation_reservations
           WHERE (scope_kind = 'bucharest_day' AND scope_key = %s)
              OR (scope_kind = 'calendar_month' AND scope_key = %s)
           GROUP BY scope_kind, scope_key
           HAVING SUM(reserved_usd) >= MIN(limit_usd) * 0.8""",
        [day.isoformat(), day.replace(day=1).isoformat()],
    )
    incidents = [_incident(str(row["failure_reason"]), str(row["slot_id"])) for row in failed]
    incidents.extend(_incident("publication_late", str(row["slot_id"])) for row in late)
    incidents.extend(
        _incident("publication_conflict", str(row["publication_id"])) for row in conflicts
    )
    incidents.extend(_incident("no_success_24_hours", str(row["slot_id"])) for row in success_gap)
    incidents.extend(
        _incident("budget_80_percent", f"{row['scope_kind']}:{row['scope_key']}") for row in budget
    )
    incidents.extend(
        VideoIncident(
            alert_id=candidate.alert_id,
            category="fal_queue_stalled",
            scope_id=candidate.request_id,
        )
        for candidate in read_stalled_fal_queue_incidents(current)
    )
    return tuple(incidents)


def _incident(category: str, scope_id: str) -> VideoIncident:
    if category not in {
        "terminal_failure",
        "deadline",
        "publication_late",
        "budget_80_percent",
        "publication_conflict",
        "no_success_24_hours",
    }:
        raise ValueError(f"Unknown video incident category: {category}")
    return VideoIncident(
        alert_id=sha256(canonical_json({"category": category, "slot_id": scope_id})),
        category=cast(VideoIncidentCategory, category),
        scope_id=scope_id,
    )
