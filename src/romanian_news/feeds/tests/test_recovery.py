import base64
import hashlib
import json
from datetime import datetime

import duckdb
import pytest
from pydantic import HttpUrl

from romanian_news.catalog.feeds import FeedCatalogLoadState, FeedEntryCatalogResult
from romanian_news.feeds.acquisition import (
    _entry_record,
    feed_entry_event_from_payload,
    feed_snapshot_event_from_payload,
)
from romanian_news.feeds.models import (
    DltFeedEvents,
    FeedEntry,
    FeedEntryEventOccurrence,
    FeedSnapshotEventOccurrence,
    legacy_feed_entry_event_id,
)
from romanian_news.feeds.recovery import (
    DltParquetObject,
    _completed_load_id_from_key,
    _register_unregistered_load,
    parse_dlt_feed_event_row,
    read_dlt_feed_events,
    reconcile_feed_entry_projection,
)

_DIGEST = "a" * 64
_REGISTRY = "b" * 64
_SNAPSHOT = "c" * 64


def test_completion_key_exposes_the_exact_dlt_load_id() -> None:
    key = (
        "dlt/romanian-news/romanian_news_intake/_dlt_loads/"
        "chartly_romanian_news_live__1788170888.4308279.jsonl"
    )

    assert _completed_load_id_from_key(key) == "1788170888.4308279"


def test_recovery_parses_the_typed_dlt_payload_and_exact_occurrence() -> None:
    event = _event()
    record = _entry_record(event)

    recovered = parse_dlt_feed_event_row({**record, "_dlt_load_id": "load-1"})

    assert recovered == FeedEntryEventOccurrence(
        source_event_id=event.event_id,
        dlt_load_id="load-1",
        event=event,
    )


def test_recovery_parses_and_verifies_a_legacy_dlt_entry_identity() -> None:
    event = _event()
    record = _entry_record(event)
    record["event_id"] = legacy_feed_entry_event_id(
        event.entry,
        event.feed_content_digest,
    )

    recovered = parse_dlt_feed_event_row({**record, "_dlt_load_id": "load-1"})

    assert recovered == FeedEntryEventOccurrence(
        source_event_id=record["event_id"],
        dlt_load_id="load-1",
        event=event,
    )


def _event():
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
        _DIGEST,
        _REGISTRY,
        datetime.fromisoformat("2026-08-31T13:00:00+03:00"),
    )


def test_reconciliation_ignores_an_incomplete_parquet_upload(monkeypatch) -> None:
    value = _parquet_object("load-incomplete")
    monkeypatch.setattr("romanian_news.feeds.recovery._list_dlt_parquet_objects", lambda: (value,))
    monkeypatch.setattr("romanian_news.feeds.recovery._list_completed_dlt_load_ids", lambda: ())
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.feed_catalog.read_feed_catalog_load_state",
        lambda: _catalog_load_state(),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery._r2_client",
        lambda: (_ for _ in ()).throw(AssertionError("incomplete loads must not download")),
    )

    result = reconcile_feed_entry_projection()

    assert result.listed_loads == 1
    assert result.examined_loads == 0


def test_malformed_dlt_snapshot_digest_fails() -> None:
    payload = {
        "feed_id": "hotnews",
        "content_digest": "a" * 64,
        "content_base64": base64.b64encode(b"different bytes").decode(),
    }

    with pytest.raises(ValueError, match="content digest mismatch"):
        parse_dlt_feed_event_row(
            {
                "event_id": "b" * 64,
                "event_type": "feed_snapshot",
                "registry_version_id": None,
                "payload_json": json.dumps(payload),
                "observed_at": None,
                "_dlt_load_id": "load-1",
            }
        )


def test_recovery_unions_historical_and_snapshot_parquet_schemas(tmp_path) -> None:
    entry_event = _event()
    entry_record = _entry_record(entry_event)
    snapshot_content = b"exact snapshot bytes"
    snapshot_event = feed_snapshot_event_from_payload(
        "hotnews",
        hashlib.sha256(snapshot_content).hexdigest(),
        snapshot_content,
    )
    historical_path = tmp_path / "historical.parquet"
    snapshot_path = tmp_path / "snapshot.parquet"
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE historical (event_id VARCHAR, event_type VARCHAR, "
            "registry_version_id VARCHAR, payload_json VARCHAR, "
            "observed_at TIMESTAMPTZ, _dlt_load_id VARCHAR)"
        )
        connection.execute(
            "INSERT INTO historical VALUES (?, 'feed_entry', ?, ?, ?, 'load-1')",
            [
                entry_record["event_id"],
                entry_record["registry_version_id"],
                entry_record["payload_json"],
                entry_record["observed_at"],
            ],
        )
        connection.execute(f"COPY historical TO '{historical_path}' (FORMAT PARQUET)")
        connection.execute(
            "CREATE TABLE snapshots (event_id VARCHAR, event_type VARCHAR, "
            "payload_json VARCHAR, _dlt_load_id VARCHAR)"
        )
        connection.execute(
            "INSERT INTO snapshots VALUES (?, 'feed_snapshot', ?, 'load-1')",
            [
                snapshot_event.event_id,
                json.dumps(
                    {
                        "feed_id": snapshot_event.feed_id,
                        "content_digest": snapshot_event.content_digest,
                        "content_base64": base64.b64encode(snapshot_content).decode("ascii"),
                    }
                ),
            ],
        )
        connection.execute(f"COPY snapshots TO '{snapshot_path}' (FORMAT PARQUET)")

    recovered = read_dlt_feed_events((historical_path, snapshot_path), ("load-1",))

    assert recovered.entries == (
        FeedEntryEventOccurrence(
            source_event_id=entry_event.event_id,
            dlt_load_id="load-1",
            event=entry_event,
        ),
    )
    assert recovered.snapshots == (
        FeedSnapshotEventOccurrence(dlt_load_id="load-1", event=snapshot_event),
    )


def test_completed_unpublished_load_recovers_snapshot_bytes_and_catalogs_event(
    monkeypatch,
    tmp_path,
) -> None:
    content = b"snapshot bytes committed inside dlt"
    digest = hashlib.sha256(content).hexdigest()
    entry_event = feed_entry_event_from_payload(
        _event().entry,
        digest,
        _REGISTRY,
        datetime.fromisoformat("2026-08-31T13:00:00+03:00"),
    )
    snapshot_event = feed_snapshot_event_from_payload("hotnews", digest, content)
    entry_occurrence = FeedEntryEventOccurrence(
        source_event_id=entry_event.event_id,
        dlt_load_id="load-new",
        event=entry_event,
    )
    snapshot_occurrence = FeedSnapshotEventOccurrence(dlt_load_id="load-new", event=snapshot_event)
    events = DltFeedEvents(entries=(entry_occurrence,), snapshots=(snapshot_occurrence,))
    value = _parquet_object("load-new")
    publications = []
    batches = []
    cataloged = []
    monkeypatch.setattr("romanian_news.feeds.recovery._list_dlt_parquet_objects", lambda: (value,))
    monkeypatch.setattr(
        "romanian_news.feeds.recovery._list_completed_dlt_load_ids", lambda: ("load-new",)
    )
    monkeypatch.setattr("romanian_news.feeds.recovery._r2_client", object)
    monkeypatch.setattr(
        "romanian_news.feeds.recovery._download_load_files",
        lambda _client, _objects, _directory: (tmp_path / "load.parquet",),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.read_dlt_feed_events", lambda _paths, _loads: events
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.read_verified_r2_object",
        lambda *_args: (_ for _ in ()).throw(AssertionError("canonical XML is absent")),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.publish_immutable_r2_objects", publications.append
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.catalog_feed_entry_events",
        lambda entries, batch_size: cataloged.append(entries)
        or FeedEntryCatalogResult(cataloged_events=1, existing_events=0),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.feed_catalog.read_feed_catalog_load_state",
        lambda: _catalog_load_state(),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.feed_catalog.register_recovered_dlt_load",
        lambda *args: batches.append(args),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.feed_catalog.mark_feed_entry_projection_load",
        lambda *args: batches.append(args),
    )

    result = reconcile_feed_entry_projection()

    canonical_key = f"news/feeds/hotnews/{digest}.xml"
    published_objects = tuple(publications[0])
    assert (canonical_key, content) in published_objects
    assert cataloged == [(entry_occurrence,)]
    assert result.registered_loads == 1
    assert result.cataloged_events == 1
    assert batches[0][0] == "load-new"


def test_reconciliation_projects_existing_registered_history(monkeypatch, tmp_path) -> None:
    event = _event()
    occurrence = FeedEntryEventOccurrence(
        source_event_id=event.event_id,
        dlt_load_id="load-1",
        event=event,
    )
    value = _parquet_object("load-1")
    batches = []
    monkeypatch.setattr("romanian_news.feeds.recovery._list_dlt_parquet_objects", lambda: (value,))
    monkeypatch.setattr(
        "romanian_news.feeds.recovery._list_completed_dlt_load_ids",
        lambda: (value.load_id,),
    )
    monkeypatch.setattr("romanian_news.feeds.recovery._r2_client", object)
    monkeypatch.setattr(
        "romanian_news.feeds.recovery._download_load_files",
        lambda _client, _objects, _directory: (tmp_path / "load.parquet",),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.read_dlt_feed_events",
        lambda _paths, _loads: DltFeedEvents(entries=(occurrence,), snapshots=()),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.catalog_feed_entry_events",
        lambda _events, batch_size: FeedEntryCatalogResult(cataloged_events=1, existing_events=0),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery._register_unregistered_load",
        lambda *_args: (_ for _ in ()).throw(AssertionError("already registered")),
    )

    registered_at = datetime.fromisoformat("2026-09-01T00:00:00+00:00")
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.feed_catalog.read_feed_catalog_load_state",
        lambda: _catalog_load_state(registered={"load-1": registered_at}),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.feed_catalog.mark_feed_entry_projection_load",
        lambda *args: batches.append(args),
    )

    result = reconcile_feed_entry_projection()

    assert result.examined_loads == 1
    assert result.registered_loads == 0
    assert result.cataloged_events == 1
    assert batches[-1] == ("load-1", registered_at)


def test_reconciliation_registers_a_completed_unpublished_load(monkeypatch, tmp_path) -> None:
    event = _event()
    occurrence = FeedEntryEventOccurrence(
        source_event_id=event.event_id,
        dlt_load_id="load-2",
        event=event,
    )
    value = _parquet_object("load-2")
    calls = []
    monkeypatch.setattr("romanian_news.feeds.recovery._list_dlt_parquet_objects", lambda: (value,))
    monkeypatch.setattr(
        "romanian_news.feeds.recovery._list_completed_dlt_load_ids",
        lambda: (value.load_id,),
    )
    monkeypatch.setattr("romanian_news.feeds.recovery._r2_client", object)
    monkeypatch.setattr(
        "romanian_news.feeds.recovery._download_load_files",
        lambda _client, _objects, _directory: (tmp_path / "load.parquet",),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.read_dlt_feed_events",
        lambda _paths, _loads: DltFeedEvents(entries=(occurrence,), snapshots=()),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery._register_unregistered_load",
        lambda load_id, objects, events, snapshots: calls.append(
            (load_id, objects, events, snapshots)
        )
        or "2026-09-01T00:00:00+00:00",
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.catalog_feed_entry_events",
        lambda _events, batch_size: FeedEntryCatalogResult(cataloged_events=1, existing_events=0),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.feed_catalog.read_feed_catalog_load_state",
        lambda: _catalog_load_state(),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.feed_catalog.mark_feed_entry_projection_load",
        lambda *_args: None,
    )

    result = reconcile_feed_entry_projection()

    assert result.registered_loads == 1
    assert calls == [("load-2", (value,), (occurrence,), ())]


def test_reconciliation_marks_a_registered_load_with_no_entries(monkeypatch, tmp_path) -> None:
    value = _parquet_object("load-empty")
    batches = []
    monkeypatch.setattr("romanian_news.feeds.recovery._list_dlt_parquet_objects", lambda: (value,))
    monkeypatch.setattr(
        "romanian_news.feeds.recovery._list_completed_dlt_load_ids",
        lambda: (value.load_id,),
    )
    monkeypatch.setattr("romanian_news.feeds.recovery._r2_client", object)
    monkeypatch.setattr(
        "romanian_news.feeds.recovery._download_load_files",
        lambda _client, _objects, _directory: (tmp_path / "load.parquet",),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.read_dlt_feed_events",
        lambda _paths, _loads: DltFeedEvents(entries=(), snapshots=()),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.catalog_feed_entry_events",
        lambda events, batch_size: FeedEntryCatalogResult(
            cataloged_events=0 if events == () else 1,
            existing_events=0,
        ),
    )

    registered_at = datetime.fromisoformat("2026-09-01T00:00:00+00:00")
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.feed_catalog.read_feed_catalog_load_state",
        lambda: _catalog_load_state(registered={"load-empty": registered_at}),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.feed_catalog.mark_feed_entry_projection_load",
        lambda *args: batches.append(args),
    )

    result = reconcile_feed_entry_projection()

    assert result.cataloged_events == 0
    assert batches[-1] == ("load-empty", registered_at)


def test_reconciliation_is_a_listing_only_noop_after_projection(monkeypatch) -> None:
    value = _parquet_object("load-1")
    monkeypatch.setattr("romanian_news.feeds.recovery._list_dlt_parquet_objects", lambda: (value,))
    monkeypatch.setattr(
        "romanian_news.feeds.recovery._list_completed_dlt_load_ids",
        lambda: (value.load_id,),
    )

    registered_at = datetime.fromisoformat("2026-09-01T00:00:00+00:00")
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.feed_catalog.read_feed_catalog_load_state",
        lambda: _catalog_load_state(
            projected={"load-1"},
            registered={"load-1": registered_at},
        ),
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery._r2_client",
        lambda: (_ for _ in ()).throw(AssertionError("must not download projected loads")),
    )

    result = reconcile_feed_entry_projection()

    assert result.examined_loads == 0
    assert result.cataloged_events == 0


def test_historical_load_without_snapshot_event_uses_canonical_fallback(monkeypatch) -> None:
    content = b"exact immutable feed snapshot"
    digest = hashlib.sha256(content).hexdigest()
    entry = _event().entry.model_copy(update={"feed_id": "hotnews"})
    event = feed_entry_event_from_payload(
        entry,
        digest,
        _REGISTRY,
        datetime.fromisoformat("2026-08-31T13:00:00+03:00"),
    )
    occurrence = FeedEntryEventOccurrence(
        source_event_id=event.event_id,
        dlt_load_id="load-3",
        event=event,
    )
    value = _parquet_object("load-3")
    publications = []
    batches = []

    def read(key, expected_digest):
        assert key == f"news/feeds/hotnews/{digest}.xml"
        assert expected_digest == digest
        return content

    monkeypatch.setattr("romanian_news.feeds.recovery.read_verified_r2_object", read)
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.publish_immutable_r2_objects", publications.append
    )
    monkeypatch.setattr(
        "romanian_news.feeds.recovery.feed_catalog.register_recovered_dlt_load",
        lambda *args: batches.append(args),
    )

    _register_unregistered_load("load-3", (value,), (occurrence,), ())

    manifest_objects = tuple(publications[0])
    assert len(manifest_objects) == 1
    assert value.key.encode() in manifest_objects[0][1]
    load_id, snapshots, manifest, registered_at = batches[0]
    assert load_id == "load-3"
    assert len(snapshots) == 1
    assert manifest.artifact_kind == "news_dlt_load"
    assert registered_at.tzinfo is not None


def _catalog_load_state(
    *,
    projected: set[str] | None = None,
    registered: dict[str, datetime] | None = None,
) -> FeedCatalogLoadState:
    return FeedCatalogLoadState(
        projected_load_ids=frozenset(projected or set()),
        registered_at_by_load_id=registered or {},
    )


def _parquet_object(load_id: str) -> DltParquetObject:
    return DltParquetObject(
        load_id=load_id,
        key=f"dlt/romanian-news/romanian_news_intake/rss_events/{load_id}.file.parquet",
        size=123,
        etag='"etag"',
    )
