from typing import cast

import requests

from romanian_news.video_digest.incidents import VideoIncident, send_video_incident


class _Response:
    def raise_for_status(self) -> None:
        pass


class _Session:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, dict[str, object], int]] = []

    def post(self, url: str, *, json: dict[str, object], timeout: int) -> _Response:
        self.calls.append((url, json, timeout))
        if self.fail:
            raise requests.Timeout("unavailable")
        return _Response()


def test_incident_payload_uses_stable_alert_id_without_sensitive_context() -> None:
    session = _Session()
    incident = VideoIncident(
        alert_id="a" * 64,
        category="publication_conflict",
        scope_id="b" * 64,
    )
    assert send_video_incident(
        incident,
        webhook_url="https://uptime.betterstack.com/example",
        session=cast(requests.Session, cast(object, session)),
    )
    url, payload, timeout = session.calls[0]
    assert url == "https://uptime.betterstack.com/example"
    assert timeout == 5
    assert payload == {
        "incident": {
            "id": "a" * 64,
            "status": "alert",
            "title": "Video digest: publication conflict",
            "description": f"publication_conflict for {'b' * 64}",
            "metadata": {
                "service": "video_digest",
                "category": "publication_conflict",
                "scope_id": "b" * 64,
            },
        }
    }


def test_incident_delivery_failure_does_not_fail_video_work() -> None:
    incident = VideoIncident(alert_id="a" * 64, category="deadline", scope_id="b" * 64)
    session = _Session(fail=True)
    assert not send_video_incident(
        incident,
        webhook_url="https://uptime.betterstack.com/example",
        session=cast(requests.Session, cast(object, session)),
    )
    assert len(session.calls) == 1
