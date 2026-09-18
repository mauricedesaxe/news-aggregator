import hashlib
import sqlite3
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import HttpUrl

from romanian_news.catalog.feeds import (
    catalog_feed_entry_events,
    mark_feed_entry_projection_load,
    publish_feed_acquisition,
    read_cataloged_feed_entry_references,
    read_feed_catalog_load_state,
    read_feed_snapshot_files,
    read_successful_feed_counts,
    read_successful_feed_observation_count,
)
from romanian_news.feeds.acquisition import _entry_record, feed_entry_event_from_payload
from romanian_news.feeds.models import (
    FeedAcquisitionResult,
    FeedCapture,
    FeedEntry,
    FeedEntryEventOccurrence,
    FeedValidator,
    feed_entry_event,
)
from romanian_news.feeds.registry import feed_registry

_CATALOG_DIGEST = "a" * 64
_CATALOG_REGISTRY = "b" * 64
_CATALOG_SNAPSHOT = "c" * 64
_CATALOG_EVENT_IDENTITY = "e" * 64


def test_live_feed_publication_catalogs_the_exact_dlt_event(monkeypatch) -> None:
    registry = feed_registry()
    content = b"<rss/>"
    entry = FeedEntry(
        feed_id="hotnews",
        source_id="source-1",
        url=HttpUrl("https://hotnews.ro/source-1"),
        title="Titlu",
        summary="Rezumat",
        feed_content="full feed content secret",
        published_at=datetime.fromisoformat("2026-09-01T10:00:00+03:00"),
        source_updated_at=None,
        author=None,
    )
    observed_at = datetime.fromisoformat("2026-09-01T11:00:00+03:00")
    capture = FeedCapture(
        feed_id="hotnews",
        observed_at=observed_at,
        scheduled_slot=observed_at.replace(minute=0),
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
    batches = []
    monkeypatch.setattr(
        "romanian_news.catalog.feeds.publish_immutable_r2_objects",
        lambda _objects: SimpleNamespace(uploaded_objects=0, reused_objects=4),
    )
    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_query", lambda *_args: [])
    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_batch", batches.append)

    publication = publish_feed_acquisition(
        FeedAcquisitionResult(load_ids=("load-1",), captures=(capture,)),
        registry,
        "test",
    )

    event = feed_entry_event(entry, capture, registry.version_id)
    dlt_record = _entry_record(event)
    event_statement = next(
        statement for statement in batches[0] if "news_feed_entry_events" in statement[0]
    )
    load_position = next(
        index for index, statement in enumerate(batches[0]) if "news_dlt_loads" in statement[0]
    )
    event_position = batches[0].index(event_statement)
    marker_statement = next(
        statement for statement in batches[0] if "news_feed_entry_projection_loads" in statement[0]
    )
    marker_position = batches[0].index(marker_statement)
    alias_statement = next(
        statement for statement in batches[0] if "news_feed_entry_event_versions" in statement[0]
    )
    assert publication.feed_entry_events == 1
    assert event_statement[1][0] == dlt_record["event_id"]
    assert event_statement[1][1] == "load-1"
    assert event_statement[1][2] == dlt_record["registry_version_id"]
    assert entry.title not in event_statement[1]
    assert entry.summary not in event_statement[1]
    assert entry.feed_content not in event_statement[1]
    assert load_position < event_position < marker_position
    assert marker_statement[1][0] == "load-1"
    assert alias_statement[1] == [event.event_id, event.event_id]


def test_feed_publication_rejects_more_than_one_dlt_load() -> None:
    registry = feed_registry()
    with __import__("pytest").raises(ValueError, match="Expected one dlt load"):
        publish_feed_acquisition(
            FeedAcquisitionResult(load_ids=("load-1", "load-2"), captures=()),
            registry,
            "test",
        )


def test_feed_publication_validates_implementation_ref_before_dlt_load() -> None:
    with pytest.raises(ValueError, match="Implementation reference is required"):
        publish_feed_acquisition(
            FeedAcquisitionResult(load_ids=(), captures=()),
            feed_registry(),
            "",
        )


def test_feed_entry_reference_ranks_before_the_published_window_filter(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "romanian_news.catalog.feeds.catalog_query",
        _sqlite_query(_feed_reference_database()),
    )

    references = read_cataloged_feed_entry_references(
        start_at=datetime.fromisoformat("2026-09-01T12:00:00+03:00"),
        end_at=datetime.fromisoformat("2026-09-01T15:00:00+03:00"),
    )

    assert [reference.event_id for reference in references] == [_CATALOG_EVENT_IDENTITY]
    assert references[0].published_at == datetime.fromisoformat("2026-09-01T10:00:00+00:00")


def _feed_reference_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    schema = Path(__file__).parents[4] / "tests" / "fixtures" / "sqlite_catalog.sql"
    connection.executescript(schema.read_text())
    connection.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "news:feed:hotnews",
            "news_feed_snapshot",
            "Hotnews snapshot",
            "source",
            "current",
            "public",
            None,
            "2026-09-01T07:00:00+00:00",
        ),
    )
    connection.execute(
        "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
        (
            _CATALOG_SNAPSHOT,
            "news:feed:hotnews",
            1,
            _CATALOG_DIGEST,
            None,
            "2026-09-01T07:00:00+00:00",
        ),
    )
    connection.execute(
        "INSERT INTO news_dlt_loads VALUES ('load-old', ?, '2026-09-01T08:00:00+00:00')",
        (_CATALOG_SNAPSHOT,),
    )
    connection.execute(
        "INSERT INTO news_dlt_loads VALUES ('load-new', ?, '2026-09-01T14:00:00+00:00')",
        (_CATALOG_SNAPSHOT,),
    )
    connection.execute(
        "INSERT INTO news_feed_entry_events VALUES (?, 'load-old', ?, ?, 'hotnews', 'source-1', 'https://hotnews.ro/a', '2026-09-01T10:00:00+00:00', NULL, '2026-09-01T08:30:00+00:00')",
        ("event-in-window", _CATALOG_REGISTRY, _CATALOG_SNAPSHOT),
    )
    connection.execute(
        "INSERT INTO news_feed_entry_events VALUES (?, 'load-new', ?, ?, 'hotnews', 'source-1', 'https://hotnews.ro/a', '2026-09-01T18:00:00+00:00', NULL, '2026-09-01T07:30:00+00:00')",
        ("event-out-of-window", "d" * 64, _CATALOG_SNAPSHOT),
    )
    connection.execute(
        "INSERT INTO news_feed_entry_event_versions VALUES (?, ?)",
        ("event-in-window", _CATALOG_EVENT_IDENTITY),
    )
    connection.execute(
        "INSERT INTO news_feed_entry_event_versions VALUES (?, ?)",
        ("event-out-of-window", _CATALOG_EVENT_IDENTITY),
    )
    connection.execute(
        "INSERT INTO news_feed_entry_occurrences VALUES (?, 'load-new', ?, ?, '2026-09-01T07:30:00+00:00')",
        ("event-out-of-window", "d" * 64, _CATALOG_SNAPSHOT),
    )
    connection.execute(
        "INSERT INTO news_feed_entry_occurrences VALUES (?, 'load-old', ?, ?, '2026-09-01T08:30:00+00:00')",
        ("event-in-window", _CATALOG_REGISTRY, _CATALOG_SNAPSHOT),
    )
    connection.commit()
    return connection


def _sqlite_query(connection: sqlite3.Connection):
    def query(sql, params=None):
        return [
            dict(row) for row in connection.execute(sql.replace("%s", "?"), params or []).fetchall()
        ]

    return query


def test_feed_publication_normalizes_mixed_captures_before_cataloging(monkeypatch) -> None:
    registry = feed_registry()
    content = b"<rss>changed</rss>"
    digest = hashlib.sha256(content).hexdigest()
    observed_at = datetime.fromisoformat("2026-09-01T11:00:00+03:00")
    entry = FeedEntry(
        feed_id="hotnews",
        source_id="source-1",
        url=HttpUrl("https://hotnews.ro/source-1"),
        title="Titlu",
        summary="Rezumat",
        feed_content=None,
        published_at=datetime.fromisoformat("2026-09-01T10:00:00+03:00"),
        source_updated_at=None,
        author=None,
    )
    captures = (
        FeedCapture(
            feed_id="hotnews",
            observed_at=observed_at,
            scheduled_slot=observed_at.replace(minute=0),
            status="ok",
            http_status=200,
            latency_ms=1,
            content=content,
            content_digest=digest,
            entries=(entry,),
            rejected_entries=0,
            validator=FeedValidator(),
            error=None,
        ),
        FeedCapture(
            feed_id="digi24",
            observed_at=observed_at,
            scheduled_slot=observed_at.replace(minute=0),
            status="not_modified",
            http_status=304,
            latency_ms=2,
            content=None,
            content_digest=None,
            entries=(),
            rejected_entries=0,
            validator=FeedValidator(),
            error=None,
        ),
        FeedCapture(
            feed_id="g4media",
            observed_at=observed_at,
            scheduled_slot=observed_at.replace(minute=0),
            status="failed",
            http_status=500,
            latency_ms=3,
            content=None,
            content_digest=None,
            entries=(entry,),
            rejected_entries=0,
            validator=FeedValidator(),
            error="upstream failed",
        ),
    )
    current_version = "a" * 64
    call_order = []
    uploaded = []
    batches = []

    def query(sql, _parameters):
        if "current_version_id" in sql:
            return [{"feed_id": "digi24", "current_version_id": current_version}]
        return []

    def upload(objects):
        call_order.append("upload")
        uploaded.extend(objects)
        return SimpleNamespace(uploaded_objects=len(uploaded), reused_objects=0)

    def batch(statements):
        call_order.append("d1")
        batches.append(statements)

    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_query", query)
    monkeypatch.setattr("romanian_news.catalog.feeds.publish_immutable_r2_objects", upload)
    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_batch", batch)

    publication = publish_feed_acquisition(
        FeedAcquisitionResult(load_ids=("load-1",), captures=captures),
        registry,
        "test",
    )

    observation_rows = [
        parameters for sql, parameters in batches[0] if "news_feed_observations" in sql
    ]
    assert call_order == ["upload", "d1"]
    assert len(uploaded) == 6
    assert publication.feed_observations == 3
    assert publication.feed_snapshots == 1
    assert publication.feed_entry_events == 1
    assert publication.feed_snapshot_versions == {
        "hotnews": observation_rows[0][1],
        "digi24": current_version,
    }
    assert [(row[2], row[4], row[6]) for row in observation_rows] == [
        ("hotnews", "ok", 1),
        ("digi24", "not_modified", 0),
        ("g4media", "failed", 1),
    ]


def test_unchanged_feed_requires_a_current_snapshot_before_upload(monkeypatch) -> None:
    observed_at = datetime.fromisoformat("2026-09-01T11:00:00+03:00")
    capture = FeedCapture(
        feed_id="digi24",
        observed_at=observed_at,
        scheduled_slot=observed_at.replace(minute=0),
        status="not_modified",
        http_status=304,
        latency_ms=2,
        content=None,
        content_digest=None,
        entries=(),
        rejected_entries=0,
        validator=FeedValidator(),
        error=None,
    )
    uploads = []
    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_query", lambda *_args: [])
    monkeypatch.setattr(
        "romanian_news.catalog.feeds.publish_immutable_r2_objects",
        lambda objects: uploads.extend(objects),
    )

    with pytest.raises(ValueError, match="Successful feed observation has no snapshot: digi24"):
        publish_feed_acquisition(
            FeedAcquisitionResult(load_ids=("load-1",), captures=(capture,)),
            feed_registry(),
            "test",
        )

    assert uploads == []


def test_successful_feed_requires_a_new_snapshot_before_storage(monkeypatch) -> None:
    capture = _successful_capture(content=None)
    storage_calls = []
    monkeypatch.setattr(
        "romanian_news.catalog.feeds.catalog_query",
        lambda *_args: storage_calls.append("d1")
        or [{"feed_id": "hotnews", "current_version_id": "a" * 64}],
    )
    monkeypatch.setattr(
        "romanian_news.catalog.feeds.publish_immutable_r2_objects",
        lambda _objects: storage_calls.append("upload")
        or SimpleNamespace(uploaded_objects=0, reused_objects=0),
    )
    monkeypatch.setattr(
        "romanian_news.catalog.feeds.catalog_batch",
        lambda _statements: storage_calls.append("d1"),
    )

    with pytest.raises(ValueError, match="Successful feed capture has no new snapshot: hotnews"):
        publish_feed_acquisition(
            FeedAcquisitionResult(load_ids=("load-1",), captures=(capture,)),
            feed_registry(),
            "test",
        )

    assert storage_calls == []


def test_successful_feed_rejects_entries_from_another_feed_before_storage(monkeypatch) -> None:
    capture = _successful_capture(entry_feed_id="digi24")
    storage_calls = []
    monkeypatch.setattr(
        "romanian_news.catalog.feeds.catalog_query",
        lambda *_args: storage_calls.append("d1") or [],
    )
    monkeypatch.setattr(
        "romanian_news.catalog.feeds.publish_immutable_r2_objects",
        lambda _objects: storage_calls.append("upload")
        or SimpleNamespace(uploaded_objects=0, reused_objects=0),
    )
    monkeypatch.setattr(
        "romanian_news.catalog.feeds.catalog_batch",
        lambda _statements: storage_calls.append("d1"),
    )

    with pytest.raises(ValueError, match="Feed capture hotnews contains entries for another feed"):
        publish_feed_acquisition(
            FeedAcquisitionResult(load_ids=("load-1",), captures=(capture,)),
            feed_registry(),
            "test",
        )

    assert storage_calls == []


def _successful_capture(
    *,
    content: bytes | None = b"<rss/>",
    entry_feed_id: str | None = None,
) -> FeedCapture:
    observed_at = datetime.fromisoformat("2026-09-01T11:00:00+03:00")
    entries = ()
    if entry_feed_id is not None:
        entries = (
            FeedEntry(
                feed_id=entry_feed_id,
                source_id="source-1",
                url=HttpUrl("https://hotnews.ro/source-1"),
                title="Titlu",
                summary="Rezumat",
                feed_content=None,
                published_at=datetime.fromisoformat("2026-09-01T10:00:00+03:00"),
                source_updated_at=None,
                author=None,
            ),
        )
    return FeedCapture(
        feed_id="hotnews",
        observed_at=observed_at,
        scheduled_slot=observed_at.replace(minute=0),
        status="ok",
        http_status=200,
        latency_ms=1,
        content=content,
        content_digest=hashlib.sha256(content).hexdigest() if content is not None else None,
        entries=entries,
        rejected_entries=0,
        validator=FeedValidator(),
        error=None,
    )


def test_feed_snapshot_files_parse_catalog_rows(monkeypatch) -> None:
    version_id = "a" * 64
    digest = "b" * 64
    calls = []
    monkeypatch.setattr(
        "romanian_news.catalog.feeds.catalog_query",
        lambda sql, parameters: calls.append((sql, parameters))
        or [
            {
                "version_id": version_id,
                "content_digest": digest,
                "feed_id": "hotnews",
                "r2_key": "news/feeds/hotnews/snapshot.xml",
            }
        ],
    )

    snapshots = read_feed_snapshot_files((version_id,))

    assert calls[0][1] == [version_id]
    assert snapshots[version_id].feed_id == "hotnews"
    assert snapshots[version_id].content_digest == digest


def test_feed_catalog_load_state_parses_registered_times(monkeypatch) -> None:
    registered_at = datetime.fromisoformat("2026-09-01T00:00:00+00:00")

    def query(sql):
        if "projection_loads" in sql:
            return [{"load_id": "load-1"}]
        if "news_dlt_loads" in sql:
            return [{"load_id": "load-1", "registered_at": registered_at.isoformat()}]
        raise AssertionError(sql)

    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_query", query)

    state = read_feed_catalog_load_state()

    assert state.projected_load_ids == frozenset({"load-1"})
    assert state.registered_at_by_load_id == {"load-1": registered_at}


def test_feed_projection_marker_serializes_utc_time(monkeypatch) -> None:
    batches = []
    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_batch", batches.append)

    mark_feed_entry_projection_load(
        "load-1",
        datetime.fromisoformat("2026-09-01T03:00:00+03:00"),
    )

    assert batches[0][0][1] == ["load-1", "2026-09-01T00:00:00+00:00"]


def test_successful_feed_observation_count_uses_the_exact_slot(monkeypatch) -> None:
    queries = []
    monkeypatch.setattr(
        "romanian_news.catalog.feeds.catalog_query",
        lambda sql, parameters: queries.append((sql, parameters)) or [{"feed_count": 3}],
    )
    scheduled_slot = datetime.fromisoformat("2026-09-01T03:00:00+03:00")

    result = read_successful_feed_observation_count(scheduled_slot)

    assert result == 3
    assert queries[0][1] == ["2026-09-01T03:00:00+03:00"]
    assert "status != 'failed'" in queries[0][0]


def test_successful_feed_counts_return_typed_days(monkeypatch) -> None:
    monkeypatch.setattr(
        "romanian_news.catalog.feeds.catalog_query",
        lambda _sql, _parameters: [{"day": "2026-09-01", "successful_feeds": 42}],
    )

    assert read_successful_feed_counts((date(2026, 9, 1),)) == {date(2026, 9, 1): 42}


def test_recovery_resolves_feed_and_digest_before_bounded_catalog_writes(monkeypatch) -> None:
    event = _catalog_event()
    occurrence = FeedEntryEventOccurrence(
        source_event_id=event.event_id,
        dlt_load_id="load-1",
        event=event,
    )
    batches = []

    def query(sql: str, parameters: list[object]):
        if "FROM news_dlt_loads" in sql:
            return [{"load_id": "load-1"}]
        if "JOIN artifacts artifact" in sql:
            assert parameters == ["news:feed:hotnews", _CATALOG_DIGEST]
            return [{"id": _CATALOG_SNAPSHOT}]
        if "FROM news_feed_entry_events" in sql:
            return []
        raise AssertionError(sql)

    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_query", query)
    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_batch", batches.append)

    result = catalog_feed_entry_events((occurrence,), batch_size=1)

    assert result.cataloged_events == 1
    assert batches[0][0][1][1] == "load-1"
    assert batches[0][0][1][3] == _CATALOG_SNAPSHOT


def test_recovery_keeps_event_batches_within_parameter_limit(monkeypatch) -> None:
    events = tuple(
        feed_entry_event_from_payload(
            FeedEntry.model_validate(
                {
                    **_catalog_event().entry.model_dump(),
                    "source_id": f"source-{index}",
                    "url": f"https://hotnews.ro/source-{index}",
                }
            ),
            _CATALOG_DIGEST,
            _CATALOG_REGISTRY,
            datetime.fromisoformat("2026-08-31T13:00:00+03:00"),
        )
        for index in range(11)
    )
    occurrences = tuple(
        FeedEntryEventOccurrence(
            source_event_id=event.event_id,
            dlt_load_id="load-1",
            event=event,
        )
        for event in events
    )
    batches = []

    def query(sql: str, parameters: list[object]):
        if "FROM news_dlt_loads" in sql:
            return [{"load_id": "load-1"}]
        if "JOIN artifacts artifact" in sql:
            return [{"id": _CATALOG_SNAPSHOT}]
        if "FROM news_feed_entry_events" in sql:
            return []
        raise AssertionError(sql)

    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_query", query)
    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_batch", batches.append)

    catalog_feed_entry_events(occurrences)

    statements = [statement for batch in batches for statement in batch]
    assert [len(parameters) for _sql, parameters in statements] == [100, 10, 55, 22]
    assert max(len(parameters) for _sql, parameters in statements) <= 100


def test_recovery_keeps_the_earliest_exact_occurrence(monkeypatch) -> None:
    event = _catalog_event()
    later = FeedEntryEventOccurrence(
        source_event_id=event.event_id,
        dlt_load_id="load-2",
        event=event.model_copy(
            update={"observed_at": datetime.fromisoformat("2026-08-31T14:00:00+03:00")}
        ),
    )
    first = FeedEntryEventOccurrence(
        source_event_id=event.event_id,
        dlt_load_id="load-1",
        event=event,
    )
    batches = []

    def query(sql: str, parameters: list[object]):
        if "FROM news_dlt_loads" in sql:
            return [{"load_id": load_id} for load_id in parameters]
        if "JOIN artifacts artifact" in sql:
            return [{"id": _CATALOG_SNAPSHOT}]
        if "FROM news_feed_entry_events" in sql:
            return []
        raise AssertionError(sql)

    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_query", query)
    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_batch", batches.append)

    result = catalog_feed_entry_events((later, first))

    assert result.cataloged_events == 1
    assert batches[0][0][1][1] == "load-1"
    assert (
        batches[0][0][1][9]
        == event.observed_at.astimezone(__import__("datetime").timezone.utc).isoformat()
    )


def test_recovery_keeps_one_identity_across_feed_snapshots(monkeypatch) -> None:
    first_event = _catalog_event()
    later_event = feed_entry_event_from_payload(
        first_event.entry,
        "e" * 64,
        first_event.registry_version_id,
        datetime.fromisoformat("2026-08-31T14:00:00+03:00"),
    )
    first = FeedEntryEventOccurrence(
        source_event_id=first_event.event_id,
        dlt_load_id="load-1",
        event=first_event,
    )
    later = FeedEntryEventOccurrence(
        source_event_id=later_event.event_id,
        dlt_load_id="load-2",
        event=later_event,
    )
    batches = []

    def query(sql: str, parameters: list[object]):
        if "FROM news_dlt_loads" in sql:
            return [{"load_id": load_id} for load_id in parameters]
        if "JOIN artifacts artifact" in sql:
            return [{"id": _CATALOG_SNAPSHOT if parameters[1] == _CATALOG_DIGEST else "f" * 64}]
        if "FROM news_feed_entry_events" in sql:
            return []
        raise AssertionError(sql)

    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_query", query)
    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_batch", batches.append)

    result = catalog_feed_entry_events((later, first))

    assert first_event.event_id == later_event.event_id
    assert result.cataloged_events == 1
    assert batches[0][0][1][1] == "load-1"
    assert batches[0][0][1][3] == _CATALOG_SNAPSHOT


def test_recovery_accepts_the_same_event_under_a_new_registry(monkeypatch) -> None:
    event = _catalog_event()
    next_registry = FeedEntryEventOccurrence(
        source_event_id=event.event_id,
        dlt_load_id="load-2",
        event=event.model_copy(
            update={
                "registry_version_id": "d" * 64,
                "observed_at": datetime.fromisoformat("2026-08-31T14:00:00+03:00"),
            }
        ),
    )
    batches = []

    def query(sql: str, parameters: list[object]):
        if "FROM news_dlt_loads" in sql:
            return [{"load_id": load_id} for load_id in parameters]
        if "JOIN artifacts artifact" in sql:
            return [{"id": _CATALOG_SNAPSHOT}]
        if "FROM news_feed_entry_events" in sql:
            return []
        raise AssertionError(sql)

    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_query", query)
    monkeypatch.setattr("romanian_news.catalog.feeds.catalog_batch", batches.append)

    result = catalog_feed_entry_events(
        (
            next_registry,
            FeedEntryEventOccurrence(
                source_event_id=event.event_id,
                dlt_load_id="load-1",
                event=event,
            ),
        )
    )

    assert result.cataloged_events == 1
    assert batches[0][0][1][1] == "load-1"


def _catalog_event():
    entry = FeedEntry(
        feed_id="hotnews",
        source_id="source-1",
        url=HttpUrl("https://hotnews.ro/source-1"),
        title="Titlu",
        summary="Rezumat",
        feed_content=None,
        published_at=datetime.fromisoformat("2026-08-31T12:00:00+03:00"),
        source_updated_at=None,
        author=None,
    )
    return feed_entry_event_from_payload(
        entry,
        _CATALOG_DIGEST,
        _CATALOG_REGISTRY,
        datetime.fromisoformat("2026-08-31T13:00:00+03:00"),
    )
