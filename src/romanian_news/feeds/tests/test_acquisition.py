import base64
import hashlib
import json
import threading
from datetime import datetime
from types import SimpleNamespace

import requests
from pydantic import HttpUrl

from romanian_news.feeds.acquisition import (
    _destination,
    _poll_feeds,
    acquire_feeds,
    feed_entry_event_from_payload,
    fetch_feed,
)
from romanian_news.feeds.models import (
    FeedAcquisitionRequest,
    FeedCapture,
    FeedEntry,
    FeedRegistry,
    FeedValidator,
    feed_schedule_slot,
    registry_version_id,
)
from romanian_news.feeds.registry import feed_registry


def test_r2_destination_does_not_inherit_the_runtime_aws_region(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(
        "romanian_news.feeds.acquisition.r2_s3_config",
        lambda: SimpleNamespace(
            endpoint_url="https://account.r2.cloudflarestorage.com",
            access_key_id="key",
            secret_access_key="secret",
        ),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.acquisition.dlt.destinations.filesystem",
        lambda **kwargs: captured.update(kwargs),
    )

    _destination(
        FeedAcquisitionRequest(
            registry=feed_registry(),
            observed_at=datetime.fromisoformat("2026-09-01T12:00:00+03:00"),
            workspace_dir=tmp_path,
            destination="r2",
        )
    )

    assert captured["credentials"]["region_name"] == "auto"


def test_repeated_fall_hour_has_distinct_feed_slots() -> None:
    first = feed_schedule_slot(datetime.fromisoformat("2026-10-25T03:15:00+03:00"))
    second = feed_schedule_slot(datetime.fromisoformat("2026-10-25T03:15:00+02:00"))

    assert first.isoformat() == "2026-10-25T03:15:00+03:00"
    assert second.isoformat() == "2026-10-25T03:15:00+02:00"
    assert first.isoformat() != second.isoformat()


def test_feed_slot_rounds_down_to_the_quarter_hour() -> None:
    observed_at = datetime.fromisoformat("2026-09-02T12:29:59+03:00")

    assert feed_schedule_slot(observed_at).isoformat() == "2026-09-02T12:15:00+03:00"


def test_feed_entry_identity_depends_only_on_the_entry_payload() -> None:
    entry = FeedEntry(
        feed_id="hotnews",
        source_id="source-1",
        url=HttpUrl("https://hotnews.ro/source-1"),
        title="Titlu",
        summary="Rezumat",
        feed_content="entry text",
        published_at=datetime.fromisoformat("2026-09-01T12:00:00+03:00"),
        source_updated_at=None,
        author=None,
    )
    first = feed_entry_event_from_payload(
        entry,
        "a" * 64,
        "b" * 64,
        datetime.fromisoformat("2026-09-01T13:00:00+03:00"),
    )
    repeated = feed_entry_event_from_payload(
        entry,
        "c" * 64,
        "d" * 64,
        datetime.fromisoformat("2026-09-01T14:00:00+03:00"),
    )
    changed = feed_entry_event_from_payload(
        entry.model_copy(update={"title": "Titlu schimbat"}),
        "c" * 64,
        "d" * 64,
        datetime.fromisoformat("2026-09-01T14:00:00+03:00"),
    )

    assert first.event_id == repeated.event_id
    assert changed.event_id != first.event_id


def test_changed_feed_writes_one_snapshot_event_without_duplicating_entry_bytes(
    monkeypatch,
    tmp_path,
) -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    registry = FeedRegistry(feeds=(feed,), version_id=registry_version_id((feed,)))
    content = b"<rss>exact raw bytes</rss>"
    observed_at = datetime.fromisoformat("2026-09-01T12:00:00+03:00")
    entry = FeedEntry(
        feed_id=feed.id,
        source_id="source-1",
        url=HttpUrl("https://hotnews.ro/source-1"),
        title="Titlu",
        summary="Rezumat",
        feed_content="entry text",
        published_at=observed_at,
        source_updated_at=None,
        author=None,
    )
    capture = FeedCapture(
        feed_id=feed.id,
        observed_at=observed_at,
        scheduled_slot=feed_schedule_slot(observed_at),
        status="ok",
        http_status=200,
        latency_ms=1,
        content=content,
        content_digest=hashlib.sha256(content).hexdigest(),
        entries=(entry,),
        rejected_entries=0,
        validator=FeedValidator(),
        error=None,
    )
    rows = []

    class Pipeline:
        def run(self, values, **_kwargs):
            rows.extend(values)
            return SimpleNamespace(loads_ids=("load-1",))

    monkeypatch.setattr("romanian_news.catalog.feeds.read_durable_feed_ids", lambda: frozenset())
    monkeypatch.setattr(
        "romanian_news.feeds.acquisition.dlt.resource",
        lambda **_kwargs: lambda function: function,
    )
    monkeypatch.setattr("romanian_news.feeds.acquisition.dlt.current.resource_state", lambda: {})
    monkeypatch.setattr("romanian_news.feeds.acquisition.fetch_feed", lambda *_args: capture)
    monkeypatch.setattr(
        "romanian_news.feeds.acquisition.dlt.pipeline", lambda **_kwargs: Pipeline()
    )

    acquire_feeds(
        FeedAcquisitionRequest(
            registry=registry,
            observed_at=observed_at,
            workspace_dir=tmp_path,
            destination="local",
        )
    )

    snapshot_rows = [row for row in rows if row["event_type"] == "feed_snapshot"]
    entry_rows = [row for row in rows if row["event_type"] == "feed_entry"]
    assert len(snapshot_rows) == 1
    snapshot_payload = json.loads(snapshot_rows[0]["payload_json"])
    assert base64.b64decode(snapshot_payload["content_base64"]) == content
    assert snapshot_payload["content_digest"] == capture.content_digest
    assert "content_base64" not in json.loads(entry_rows[0]["payload_json"])


def test_fetch_feed_records_digest_and_stores_conditional_get_validators(monkeypatch) -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    observed_at = datetime.fromisoformat("2026-09-01T12:00:00+03:00")
    content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>HotNews</title>
      <item>
        <guid>article-1</guid><title>Guvernul anunta masuri economice</title>
        <link>https://www.hotnews.ro/stiri/article-1</link>
        <pubDate>Mon, 31 Aug 2026 12:13:04 +0300</pubDate>
        <description><![CDATA[Un rezumat.]]></description>
      </item>
    </channel></rss>"""
    requests_seen = []

    def get(_session, _url, *, headers, timeout):
        requests_seen.append(headers)
        return SimpleNamespace(
            status_code=200,
            headers={"ETag": '"etag-1"', "Last-Modified": "Mon, 31 Aug 2026 09:00:00 GMT"},
            content=content,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("romanian_news.feeds.acquisition.get_with_validated_redirects", get)

    capture = fetch_feed(requests.Session(), feed, observed_at, FeedValidator())

    assert capture.status == "ok"
    assert capture.http_status == 200
    assert capture.content == content
    assert capture.content_digest == hashlib.sha256(content).hexdigest()
    assert [entry.source_id for entry in capture.entries] == ["article-1"]
    assert capture.rejected_entries == 0
    assert capture.validator == FeedValidator(
        etag='"etag-1"',
        last_modified="Mon, 31 Aug 2026 09:00:00 GMT",
    )
    assert "If-None-Match" not in requests_seen[0]
    assert "If-Modified-Since" not in requests_seen[0]


def test_fetch_feed_sends_stored_validators_and_caps_304_without_entry_bytes(monkeypatch) -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    observed_at = datetime.fromisoformat("2026-09-01T12:15:00+03:00")
    validator = FeedValidator(
        etag='"etag-1"',
        last_modified="Mon, 31 Aug 2026 09:00:00 GMT",
    )
    requests_seen = []

    def get(_session, _url, *, headers, timeout):
        requests_seen.append(headers)
        return SimpleNamespace(status_code=304, headers={}, content=b"")

    monkeypatch.setattr("romanian_news.feeds.acquisition.get_with_validated_redirects", get)

    capture = fetch_feed(requests.Session(), feed, observed_at, validator)

    assert requests_seen[0]["If-None-Match"] == '"etag-1"'
    assert requests_seen[0]["If-Modified-Since"] == "Mon, 31 Aug 2026 09:00:00 GMT"
    assert capture.status == "not_modified"
    assert capture.http_status == 304
    assert capture.content is None
    assert capture.content_digest is None
    assert capture.entries == ()
    assert capture.rejected_entries == 0
    assert capture.validator == validator


def test_fetch_feed_captures_network_failure_with_the_error_classified(monkeypatch) -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    observed_at = datetime.fromisoformat("2026-09-01T12:30:00+03:00")
    validator = FeedValidator(etag='"etag-1"')

    def get(_session, _url, **_kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("romanian_news.feeds.acquisition.get_with_validated_redirects", get)

    capture = fetch_feed(requests.Session(), feed, observed_at, validator)

    assert capture.status == "failed"
    assert capture.http_status is None
    assert capture.content is None
    assert capture.content_digest is None
    assert capture.entries == ()
    assert capture.validator == validator
    assert capture.error == "connection refused"


def test_feed_poll_fetches_outlets_concurrently(monkeypatch) -> None:
    feeds = feed_registry().feeds[:2]
    observed_at = datetime.fromisoformat("2026-09-02T12:00:00+03:00")
    barrier = threading.Barrier(2)

    def fetch(_session, feed, _observed_at, validator):
        barrier.wait(timeout=1)
        return FeedCapture(
            feed_id=feed.id,
            observed_at=observed_at,
            scheduled_slot=feed_schedule_slot(observed_at),
            status="not_modified",
            http_status=304,
            latency_ms=1,
            content=None,
            content_digest=None,
            entries=(),
            rejected_entries=0,
            validator=validator,
            error=None,
        )

    monkeypatch.setattr("romanian_news.feeds.acquisition.fetch_feed", fetch)

    captures = _poll_feeds(
        feeds,
        observed_at,
        {feed.id: FeedValidator() for feed in feeds},
    )

    assert [capture.feed_id for capture in captures] == [feed.id for feed in feeds]
