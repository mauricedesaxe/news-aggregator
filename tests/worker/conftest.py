from __future__ import annotations

import hashlib
import io
import re
import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import UTC, date, datetime
from email.utils import format_datetime
from pathlib import Path
from typing import Any, cast

import pytest
import requests
from botocore.exceptions import ClientError
from pydantic import HttpUrl
from urllib3.response import HTTPResponse

from romanian_news import BUCHAREST, storage
from romanian_news.articles import acquisition as article_acquisition
from romanian_news.catalog import (
    analysis,
    analysis_inputs,
    articles,
    artifacts,
    daily,
    feeds,
)
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
from romanian_news.worker import operations

_SLOT_DAY = re.compile(r"\((\w+\.)?scheduled_slot AT TIME ZONE 'Europe/Bucharest'\)::date")
_ANY_PARAMETER = re.compile(r"=\s*ANY\(%s\)", re.IGNORECASE)
_ROW_VALUE_SUBQUERY = re.compile(r"\s*FROM\s+(\w+)\s+WHERE\s+([\w.]+)\s*=\s*([^\s)]+)\s*\)")
_IS_NOT_DISTINCT_FROM = re.compile(r"\bIS\s+NOT\s+DISTINCT\s+FROM\b", re.IGNORECASE)
_IS_DISTINCT_FROM = re.compile(r"\bIS\s+DISTINCT\s+FROM\b", re.IGNORECASE)


def _split_top_level_columns(value: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for character in value:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if character == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    parts.append("".join(current).strip())
    return parts


def _bucharest_day(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(BUCHAREST).date().isoformat()


def _parameter_bindings(sql: str, parameters: Sequence[object]) -> tuple[str, dict[str, object]]:
    """Translate one Postgres-shaped statement to named sqlite parameters.

    Named parameters let a rewritten row-value subquery reference one bound
    parameter from several per-column scalar subqueries.
    """
    rewritten = _SLOT_DAY.sub(r"bucharest_day(\1scheduled_slot)", sql)
    rewritten = _IS_NOT_DISTINCT_FROM.sub("IS", rewritten)
    rewritten = _IS_DISTINCT_FROM.sub("IS NOT", rewritten)
    bindings: dict[str, object] = {}
    out: list[str] = []
    position = 0
    for start, end, replacement in _parameter_sites(rewritten, parameters):
        segment = rewritten[position:start]
        out.append(_render_plain(segment, rewritten[:position].count("%s"), bindings, parameters))
        out.append(replacement(bindings))
        position = end
    out.append(
        _render_plain(rewritten[position:], rewritten[:position].count("%s"), bindings, parameters)
    )
    return "".join(out), bindings


def _render_plain(
    segment: str,
    base: int,
    bindings: dict[str, object],
    parameters: Sequence[object],
) -> str:
    pieces = segment.split("%s")
    for offset in range(1, len(pieces)):
        index = base + offset - 1
        pieces[offset] = f":p{index}" + pieces[offset]
        bindings[f"p{index}"] = parameters[index]
    return "".join(pieces)


def _parameter_sites(
    sql: str, parameters: Sequence[object]
) -> list[tuple[int, int, Callable[[dict[str, object]], str]]]:
    sites: list[tuple[int, int, Callable[[dict[str, object]], str]]] = []
    for match in _ANY_PARAMETER.finditer(sql):
        parameter_index = sql[: match.start()].count("%s")

        def render_any(bindings: dict[str, object], index: int = parameter_index) -> str:
            items = list(cast(Iterable[object], parameters[index]))
            for position, item in enumerate(items):
                bindings[f"p{index}_{position}"] = item
            if not items:
                return "IN (SELECT NULL WHERE 0)"
            placeholders = ", ".join(f":p{index}_{position}" for position in range(len(items)))
            return f"IN ({placeholders})"

        sites.append((match.start(), match.end(), render_any))
    for outer, tail, columns, table, column, value in _row_value_sites(sql):
        parameter_index = sql[: tail.start(3)].count("%s") if value == "%s" else None

        def render_row(
            bindings: dict[str, object],
            columns: list[str] = columns,
            table: str = table,
            column: str = column,
            value: str = value,
            parameter_index: int | None = parameter_index,
        ) -> str:
            bound = value if parameter_index is None else f":p{parameter_index}"
            if parameter_index is not None:
                bindings[f"p{parameter_index}"] = parameters[parameter_index]
            subqueries = ", ".join(
                f"(SELECT {item} FROM {table} WHERE {column} = {bound})" for item in columns
            )
            return f"({subqueries})"

        sites.append((outer, tail.end(), render_row))
    sites.sort(key=lambda site: site[0])
    return sites


def _row_value_sites(sql: str) -> list[tuple[int, re.Match[str], list[str], str, str, str]]:
    marker = "SELECT ("
    sites = []
    position = 0
    while (found := sql.find(marker, position)) != -1:
        outer = found - 1
        while outer >= position and sql[outer].isspace():
            outer -= 1
        columns_start = found + len(marker)
        depth = 1
        index = columns_start
        while depth and index < len(sql):
            if sql[index] == "(":
                depth += 1
            elif sql[index] == ")":
                depth -= 1
            index += 1
        tail = _ROW_VALUE_SUBQUERY.match(sql, index) if not depth else None
        columns = _split_top_level_columns(sql[columns_start : index - 1]) if not depth else []
        if tail is not None and len(columns) > 1 and outer >= position and sql[outer] == "(":
            sites.append((outer, tail, columns, tail.group(1), tail.group(2), tail.group(3)))
        position = found + len(marker)
    return sites


class SqliteCatalog:
    """Serve the Postgres-shaped catalog API from one locked sqlite connection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._lock = threading.Lock()

    def query(
        self, statement: str, parameters: Sequence[object] | None = None
    ) -> list[dict[str, Any]]:
        sql, bindings = _parameter_bindings(statement, parameters or ())
        with self._lock:
            cursor = self._connection.execute(sql, bindings)
            return [dict(row) for row in cursor.fetchall()]

    def batch(
        self,
        statements: Sequence[tuple[str, Sequence[object]]],
        *,
        retry_transient_errors: bool = False,
    ) -> None:
        if not statements:
            return
        with self._lock, self._connection:
            for statement, parameters in statements:
                sql, bindings = _parameter_bindings(statement, parameters)
                self._connection.execute(sql, bindings)


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


@pytest.fixture
def catalog_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.create_collation("C", lambda left, right: (left > right) - (left < right))
    connection.create_function("bucharest_day", 1, _bucharest_day, deterministic=True)
    schema = Path(__file__).parents[1] / "fixtures" / "sqlite_catalog.sql"
    connection.executescript(schema.read_text())
    return connection


@pytest.fixture
def sqlite_catalog(
    monkeypatch: pytest.MonkeyPatch, catalog_connection: sqlite3.Connection
) -> SqliteCatalog:
    catalog = SqliteCatalog(catalog_connection)
    for module in (analysis, analysis_inputs, articles, artifacts, daily, feeds):
        monkeypatch.setattr(module, "catalog_query", catalog.query, raising=False)
        monkeypatch.setattr(module, "catalog_batch", catalog.batch, raising=False)
    monkeypatch.setattr(operations, "ensure_news_catalog_schema", lambda: None)
    return catalog


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
