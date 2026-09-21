import socket

import pytest
import requests
from requests.adapters import HTTPAdapter

from romanian_news.http import (
    UnsafeNewsRedirect,
    create_news_session,
    get_with_validated_redirects,
)


def test_news_session_retries_get_status_failures_with_server_aware_backoff() -> None:
    adapter = create_news_session().get_adapter("https://")
    assert isinstance(adapter, HTTPAdapter)
    retry = adapter.max_retries

    assert retry.total == 3
    assert retry.backoff_factor == 10
    assert all(retry.is_retry("GET", status) for status in (429, 500, 502, 503, 504))
    assert not retry.is_retry("POST", 503)
    assert retry.respect_retry_after_header is True


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
