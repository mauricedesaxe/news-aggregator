from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime
from email.utils import format_datetime
from typing import Any

import pytest
import requests
from botocore.exceptions import ClientError
from pydantic import HttpUrl
from urllib3.response import HTTPResponse

from romanian_news import storage
from romanian_news.articles import acquisition as article_acquisition
from romanian_news.catalog.feeds import publish_feed_acquisition
from romanian_news.feeds import acquisition as feed_acquisition
from romanian_news.feeds import recovery
from romanian_news.feeds.acquisition import fetch_feed
from romanian_news.feeds.models import (
    FeedAcquisitionResult,
    FeedCapture,
    FeedRegistry,
    FeedSpec,
    FeedValidator,
    feed_entry_event,
    registry_version_id,
)
from tests.postgres_catalog import postgres_catalog_fixture


class FakeR2Client:
    """In-memory stand-in for the R2 S3 client at the storage network boundary."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {"Metadata": dict(self.metadata[Key])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str]) -> None:
        self.objects[Key] = Body
        self.metadata[Key] = dict(Metadata)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, name: str) -> _FakePaginator:
        return _FakePaginator(self)


class _FakePaginator:
    def __init__(self, client: FakeR2Client) -> None:
        self._client = client

    def paginate(self, *, Bucket: str, Prefix: str) -> Iterator[dict[str, object]]:
        contents = [
            {
                "Key": key,
                "Size": len(self._client.objects[key]),
                "ETag": f'"{hashlib.sha256(self._client.objects[key]).hexdigest()}"',
            }
            for key in sorted(self._client.objects)
            if key.startswith(Prefix)
        ]
        yield {"Contents": contents}


class FakeNewsSession(requests.Session):
    """Serve canned responses per URL at the requests.Session network boundary."""

    def __init__(self) -> None:
        super().__init__()
        self._routes: dict[str, tuple[int, bytes, dict[str, str]] | Exception] = {}

    def serve(
        self,
        url: str,
        content: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._routes[url] = (status, content, dict(headers or {}))

    def fail(self, url: str, error: Exception) -> None:
        self._routes[url] = error

    def get(self, url: str | bytes, **kwargs: Any) -> requests.Response:
        target = str(url)
        try:
            route = self._routes[target]
        except KeyError:
            raise AssertionError(f"No fake HTTP route for {target}") from None
        if isinstance(route, Exception):
            raise route
        status, content, headers = route
        response = requests.Response()
        response.status_code = status
        response.url = target
        response.headers.update(headers)
        response.raw = HTTPResponse(body=io.BytesIO(content), status=status, preload_content=False)
        return response


def feed_rss(items: Iterable[tuple[str, str, datetime, str]]) -> bytes:
    """Build RSS 2.0 bytes from (link, title, published, description) items."""
    chunks = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<rss version="2.0"><channel>',
        "<title>Test feed</title>",
        "<link>https://example.test/</link>",
        "<description>Canal de test</description>",
    ]
    for link, title, published, description in items:
        escaped_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        chunks.extend(
            (
                "<item>",
                f"<title>{escaped_title}</title>",
                f"<link>{link}</link>",
                f"<guid>{link}</guid>",
                f"<pubDate>{format_datetime(published.astimezone(UTC))}</pubDate>",
                f"<description>{description}</description>",
                "</item>",
            )
        )
    chunks.append("</channel></rss>")
    return "".join(chunks).encode()


def article_html(title: str, paragraphs: Iterable[str]) -> bytes:
    body = "\n".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
    return (
        "<!doctype html><html><head>"
        f"<title>{title}</title>"
        "</head><body><article>\n"
        f"<h1>{title}</h1>\n{body}\n"
        "</article></body></html>"
    ).encode()


def make_feed(
    feed_id: str,
    outlet_id: str,
    url: str,
    *hosts: str,
) -> FeedSpec:
    return FeedSpec(
        id=feed_id,
        outlet_id=outlet_id,
        outlet_name=f"Outlet {outlet_id}",
        url=HttpUrl(url),
        category="general",
        article_hosts=hosts,
    )


def make_registry(*feeds_: FeedSpec) -> FeedRegistry:
    ordered = tuple(sorted(feeds_, key=lambda feed: feed.id))
    return FeedRegistry(feeds=ordered, version_id=registry_version_id(ordered))


def seed_feed_observation(
    session: FakeNewsSession,
    registry: FeedRegistry,
    feed: FeedSpec,
    items: tuple[tuple[str, str, datetime, str], ...],
    observed_at: datetime,
) -> FeedCapture:
    """Run the real fetch-parse-publish path for one feed without the dlt lake."""
    session.serve(str(feed.url), feed_rss(items))
    capture = fetch_feed(session, feed, observed_at, FeedValidator())
    publish_feed_acquisition(
        FeedAcquisitionResult(load_ids=(f"seed:{feed.id}",), captures=(capture,)),
        registry,
        "git:test",
    )
    return capture


def seeded_event_ids(capture: FeedCapture, registry: FeedRegistry) -> tuple[str, ...]:
    """Return the durable event IDs cataloged for one seeded capture."""
    return tuple(
        feed_entry_event(entry, capture, registry.version_id).event_id for entry in capture.entries
    )


class WorkerHarness:
    """Builder entry points shared by the worker E2E tests."""

    feed = staticmethod(make_feed)
    registry = staticmethod(make_registry)
    rss = staticmethod(feed_rss)
    article = staticmethod(article_html)
    seed = staticmethod(seed_feed_observation)
    event_ids = staticmethod(seeded_event_ids)


postgres_catalog = postgres_catalog_fixture("news_worker_contract")


@pytest.fixture
def fake_r2(monkeypatch: pytest.MonkeyPatch) -> FakeR2Client:
    client = FakeR2Client()
    monkeypatch.setattr(storage, "_r2_client", lambda: client)
    monkeypatch.setattr(recovery, "_r2_client", lambda: client)
    return client


@pytest.fixture
def fake_http(monkeypatch: pytest.MonkeyPatch) -> FakeNewsSession:
    session = FakeNewsSession()
    monkeypatch.setattr(feed_acquisition, "create_news_session", lambda: session)
    monkeypatch.setattr(article_acquisition, "create_news_session", lambda: session)
    return session


@pytest.fixture
def news_day() -> date:
    return date(2099, 9, 2)


@pytest.fixture
def harness() -> WorkerHarness:
    return WorkerHarness()
