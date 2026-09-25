from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from romanian_news.catalog import video_digest_monitor
from romanian_news.catalog.artifacts import canonical_json, sha256


def test_due_incidents_have_stable_scope_ids_and_bucharest_budget_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries: list[tuple[str, list[object]]] = []

    def query(sql: str, params: list[object]) -> list[dict[str, str]]:
        queries.append((sql, params))
        if "failure_reason FROM video_digest_slots" in sql:
            return [{"slot_id": "a" * 64, "failure_reason": "deadline"}]
        if "edition_id IS NOT NULL" in sql:
            return [{"slot_id": "b" * 64}]
        if "video_digest_slot_sources" in sql:
            return [{"slot_id": "d" * 64}]
        if "video_digest_publication_intents" in sql:
            return [{"publication_id": "c" * 64}]
        return [{"scope_kind": "bucharest_day", "scope_key": "2026-09-25"}]

    monkeypatch.setattr(video_digest_monitor, "catalog_query", query)
    monkeypatch.setattr(
        video_digest_monitor,
        "read_stalled_fal_queue_incidents",
        lambda _now: (SimpleNamespace(alert_id="e" * 64, request_id="f" * 64),),
    )
    incidents = video_digest_monitor.read_due_video_incidents(datetime(2026, 9, 24, 22, tzinfo=UTC))
    assert [(item.category, item.scope_id) for item in incidents] == [
        ("deadline", "a" * 64),
        ("publication_late", "b" * 64),
        ("publication_conflict", "c" * 64),
        ("no_success_24_hours", "d" * 64),
        ("budget_80_percent", "bucharest_day:2026-09-25"),
        ("fal_queue_stalled", "f" * 64),
    ]
    assert incidents[-1].alert_id == "e" * 64
    assert incidents[0].alert_id == sha256(
        canonical_json({"category": "deadline", "slot_id": "a" * 64})
    )
    assert queries[3][1] == [datetime(2026, 9, 23, 22, tzinfo=UTC)]
    assert queries[4][1] == ["2026-09-25", "2026-09-01"]


def test_monitor_requires_aware_time() -> None:
    with pytest.raises(ValueError, match="timezone"):
        video_digest_monitor.read_due_video_incidents(datetime(2026, 9, 25))
