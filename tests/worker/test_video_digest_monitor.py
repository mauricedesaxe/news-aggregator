import dagster as dg
import pytest

from romanian_news.video_digest.incidents import VideoIncident
from romanian_news.worker import video_digest_monitor
from romanian_news.worker.definitions import defs


def test_video_incident_monitor_is_registered_but_stopped() -> None:
    schedule = video_digest_monitor.scheduled_video_digest_incident_monitor
    assert schedule.cron_schedule == "*/5 * * * *"
    assert schedule.default_status is dg.DefaultScheduleStatus.STOPPED
    assert defs.get_schedule_def(schedule.name) is not None


def test_monitor_delivers_durable_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    incident = VideoIncident(alert_id="a" * 64, category="publication_late", scope_id="b" * 64)
    sent: list[VideoIncident] = []
    monkeypatch.setattr(video_digest_monitor, "read_due_video_incidents", lambda _now: (incident,))
    monkeypatch.setattr(
        video_digest_monitor, "send_video_incident", lambda item: sent.append(item) or True
    )

    class Context:
        def add_output_metadata(self, metadata: dict[str, object]) -> None:
            assert metadata == {"incident_candidates": 1, "delivered": 1}

    compute_fn = video_digest_monitor.deliver_video_digest_incidents.compute_fn
    decorated_fn = getattr(compute_fn, "decorated_fn", None)
    assert callable(decorated_fn)
    decorated_fn(Context())
    assert sent == [incident]
