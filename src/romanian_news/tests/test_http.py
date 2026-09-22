import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
import requests
import urllib3.util.retry
from requests.adapters import HTTPAdapter

from romanian_news.http import (
    UnsafeNewsRedirect,
    create_news_session,
    get_with_validated_redirects,
)


class _ThrottlingHandler(BaseHTTPRequestHandler):
    failures_left = 1
    retry_after: str | None = "1"
    request_count = 0

    def do_GET(self) -> None:
        type(self).request_count += 1
        if type(self).failures_left > 0:
            type(self).failures_left -= 1
            if type(self).retry_after is not None:
                self.send_header("Retry-After", type(self).retry_after)
            self.send_response(429)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"recovered")

    def log_message(self, *_args: object) -> None:
        pass


def _stop(server: HTTPServer) -> None:
    server.shutdown()
    server.server_close()


def _local_session() -> tuple[requests.Session, str, HTTPServer]:
    session = create_news_session()
    adapter = session.get_adapter("https://")
    assert isinstance(adapter, HTTPAdapter)
    session.mount("http://", adapter)
    server = HTTPServer(("127.0.0.1", 0), _ThrottlingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return session, f"http://127.0.0.1:{server.server_port}/feed.xml", server


def _record_retry_pauses(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, float]]:
    pauses: list[tuple[str, float]] = []

    def sleep_for_retry(self: Any, response: Any) -> bool:
        retry_after = self.get_retry_after(response)
        pauses.append(("retry_after", retry_after if retry_after is not None else -1.0))
        return bool(retry_after)

    def sleep_backoff(self: Any) -> None:
        pauses.append(("backoff", self.get_backoff_time()))

    monkeypatch.setattr(urllib3.util.retry.Retry, "sleep_for_retry", sleep_for_retry)
    monkeypatch.setattr(urllib3.util.retry.Retry, "_sleep_backoff", sleep_backoff)
    return pauses


def test_news_session_retries_a_throttled_get_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _record_retry_pauses(monkeypatch)
    _ThrottlingHandler.failures_left = 1
    _ThrottlingHandler.retry_after = "1"
    _ThrottlingHandler.request_count = 0
    session, url, server = _local_session()
    try:
        response = get_with_validated_redirects(
            session, url, headers={"Accept": "application/rss+xml"}, timeout=10
        )
    finally:
        _stop(server)

    assert response.status_code == 200
    assert response.content == b"recovered"
    assert _ThrottlingHandler.request_count == 2


def test_news_session_stops_retrying_a_sustained_throttle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _record_retry_pauses(monkeypatch)
    _ThrottlingHandler.failures_left = 99
    _ThrottlingHandler.retry_after = None
    _ThrottlingHandler.request_count = 0
    session, url, server = _local_session()
    try:
        with pytest.raises(requests.RequestException):
            get_with_validated_redirects(session, url, headers={}, timeout=10)
    finally:
        _stop(server)

    assert _ThrottlingHandler.request_count == 4


def test_get_validates_redirect_before_sending_the_next_request(monkeypatch) -> None:
    responses = [
        _response(302, location="https://cdn.example/article"),
        _response(200),
    ]
    requested = []
    session = requests.Session()

    def get(url, **kwargs):
        requested.append((url, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(session, "get", get)
    monkeypatch.setattr(
        "romanian_news.http.socket.getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )

    response = get_with_validated_redirects(
        session,
        "https://publisher.example/article",
        headers={"Accept": "text/html"},
        timeout=10,
        stream=True,
    )

    assert response.status_code == 200
    assert [request[0] for request in requested] == [
        "https://publisher.example/article",
        "https://cdn.example/article",
    ]
    assert all(request[1]["allow_redirects"] is False for request in requested)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "::1",
    ],
)
def test_get_rejects_non_public_redirects_before_request(monkeypatch, address: str) -> None:
    redirect = _response(302, location="http://internal.example/latest")
    session = requests.Session()
    requested = []

    def get(url, **kwargs):
        requested.append(url)
        return redirect

    monkeypatch.setattr(session, "get", get)
    monkeypatch.setattr(
        "romanian_news.http.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))],
    )

    with pytest.raises(UnsafeNewsRedirect, match="non-public"):
        get_with_validated_redirects(
            session,
            "https://publisher.example/article",
            headers={},
            timeout=10,
        )

    assert requested == ["https://publisher.example/article"]


def _response(status: int, *, location: str | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = b""
    response.__dict__["_content_consumed"] = True
    if location is not None:
        response.headers["Location"] = location
    return response
