from romanian_news import alerts as news_alerts
from romanian_news.alerts import ping_heartbeat


def test_ping_heartbeat_is_a_no_op_without_a_configured_url(monkeypatch) -> None:
    monkeypatch.setattr(news_alerts, "BETTERSTACK_REPORT_HEARTBEAT_URL", None)
    calls: list[str] = []

    class _Forbidden:
        def get(self, *_args: object, **_kwargs: object) -> None:
            calls.append("get")

    monkeypatch.setattr(news_alerts.requests, "get", _Forbidden().get)
    ping_heartbeat("report")

    assert calls == []


def test_ping_heartbeat_posts_fire_and_forget(monkeypatch) -> None:
    monkeypatch.setattr(
        news_alerts, "BETTERSTACK_RESEARCH_TRIGGER_HEARTBEAT_URL", "https://example.test/ping"
    )
    seen: dict[str, object] = {}

    def _get(url: str, timeout: float) -> object:
        seen["url"] = url
        seen["timeout"] = timeout

        class _Response:
            @staticmethod
            def raise_for_status() -> None:
                return None

        return _Response()

    monkeypatch.setattr(news_alerts.requests, "get", _get)
    ping_heartbeat("research_trigger")
    for thread in news_alerts._heartbeat_threads:
        thread.join(timeout=5)

    assert seen == {"url": "https://example.test/ping", "timeout": 5}


def test_ping_heartbeat_swallows_transport_errors_without_logging_the_url(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(
        news_alerts, "BETTERSTACK_REPORT_HEARTBEAT_URL", "https://example.test/ping"
    )

    def _get(url: str, timeout: float) -> object:
        raise news_alerts.requests.RequestException(f"failed request to {url}")

    monkeypatch.setattr(news_alerts.requests, "get", _get)
    ping_heartbeat("report")
    for thread in news_alerts._heartbeat_threads:
        thread.join(timeout=5)

    assert "Heartbeat ping failed for report" in caplog.text
    assert "https://example.test/ping" not in caplog.text
