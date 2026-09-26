from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from romanian_news.catalog import video_digest_monitor


def test_stalled_fal_queue_candidates_become_request_scoped_incidents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_digest_monitor, "catalog_query", lambda *_args: [])
    monkeypatch.setattr(
        video_digest_monitor,
        "read_stalled_fal_queue_incidents",
        lambda _now: (SimpleNamespace(alert_id="e" * 64, request_id="f" * 64),),
    )

    incidents = video_digest_monitor.read_due_video_incidents(datetime(2026, 9, 24, 22, tzinfo=UTC))

    assert [
        (incident.category, incident.scope_id, incident.alert_id) for incident in incidents
    ] == [("fal_queue_stalled", "f" * 64, "e" * 64)]


def test_monitor_requires_aware_time() -> None:
    with pytest.raises(ValueError, match="timezone"):
        video_digest_monitor.read_due_video_incidents(datetime(2026, 9, 25))
