from __future__ import annotations

import base64
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import boto3
import duckdb
from botocore.client import BaseClient
from pydantic import AliasChoices, Field

from romanian_news import NewsModel, Sha256
from romanian_news.catalog import feeds as feed_catalog
from romanian_news.catalog.artifacts import artifact_file
from romanian_news.catalog.feeds import catalog_feed_entry_events
from romanian_news.config import NEWS_R2_BUCKET
from romanian_news.feeds.acquisition import (
    feed_entry_event_from_payload,
    feed_snapshot_event_from_payload,
)
from romanian_news.feeds.models import (
    DltFeedEvents,
    FeedEntry,
    FeedEntryEventOccurrence,
    FeedSnapshotEventOccurrence,
)
from romanian_news.identity import canonical_json, sha256
from romanian_news.storage import (
    publish_immutable_r2_objects,
    r2_s3_config,
    read_verified_r2_object,
)

_DLT_FEED_ENTRY_PREFIX = "dlt/romanian-news/romanian_news_intake/rss_events/"
_DLT_LOAD_COMPLETION_PREFIX = "dlt/romanian-news/romanian_news_intake/_dlt_loads/"


class DltParquetObject(NewsModel):
    load_id: Annotated[str, Field(min_length=1)]
    key: Annotated[str, Field(min_length=1)]
    size: Annotated[int, Field(ge=0)]
    etag: Annotated[str, Field(min_length=1)]


class FeedEntryProjectionReconciliation(NewsModel):
    listed_loads: Annotated[int, Field(ge=0)]
    examined_loads: Annotated[int, Field(ge=0)]
    registered_loads: Annotated[int, Field(ge=0)]
    cataloged_events: Annotated[int, Field(ge=0)]


class _DltFeedEntryPayload(FeedEntry):
    feed_content_digest: Sha256


class _DltFeedSnapshotPayload(NewsModel):
    feed_id: str
    content_digest: Sha256
    content_base64: str


class _DltEventRow(NewsModel):
    event_id: Sha256
    event_type: str
    registry_version_id: Sha256 | None
    payload_json: str
    observed_at: datetime | None
    dlt_load_id: Annotated[
        str,
        Field(min_length=1, validation_alias=AliasChoices("_dlt_load_id", "dlt_load_id")),
    ]


def parse_dlt_feed_event_row(
    row: Mapping[str, object],
) -> FeedEntryEventOccurrence | FeedSnapshotEventOccurrence:
    """Parse one raw dlt Parquet row into a typed feed event occurrence."""
    parsed = _DltEventRow.model_validate(row)
    if parsed.event_type == "feed_entry":
        if parsed.registry_version_id is None or parsed.observed_at is None:
            raise ValueError("dlt feed entry event has incomplete lineage")
        payload = _DltFeedEntryPayload.model_validate_json(parsed.payload_json)
        entry = FeedEntry.model_validate(payload.model_dump(exclude={"feed_content_digest"}))
        event = feed_entry_event_from_payload(
            entry,
            payload.feed_content_digest,
            parsed.registry_version_id,
            parsed.observed_at,
        )
        try:
            return FeedEntryEventOccurrence(
                source_event_id=parsed.event_id,
                dlt_load_id=parsed.dlt_load_id,
                event=event,
            )
        except ValueError as error:
            raise ValueError(
                f"dlt feed entry event identity mismatch: {parsed.event_id}"
            ) from error
    if parsed.event_type == "feed_snapshot":
        payload = _DltFeedSnapshotPayload.model_validate_json(parsed.payload_json)
        try:
            content = base64.b64decode(payload.content_base64, validate=True)
        except ValueError as error:
            raise ValueError("dlt feed snapshot payload is not valid base64") from error
        event = feed_snapshot_event_from_payload(
            payload.feed_id,
            payload.content_digest,
            content,
        )
        if event.event_id != parsed.event_id:
            raise ValueError(f"dlt feed snapshot event identity mismatch: {parsed.event_id}")
        return FeedSnapshotEventOccurrence(dlt_load_id=parsed.dlt_load_id, event=event)
    raise ValueError(f"Expected a feed dlt event, received {parsed.event_type}")


def read_dlt_feed_events(
    parquet_paths: tuple[Path, ...],
    load_ids: tuple[str, ...],
) -> DltFeedEvents:
    """Read typed feed-entry and feed-snapshot events from selected dlt loads."""
    if not parquet_paths or not load_ids:
        return DltFeedEvents(entries=(), snapshots=())
    placeholders = ", ".join("?" for _ in load_ids)
    rows = duckdb.execute(
        f"SELECT event_id, event_type, registry_version_id, payload_json, "
        f"observed_at, _dlt_load_id FROM read_parquet(?, union_by_name = true) "
        f"WHERE event_type IN ('feed_entry', 'feed_snapshot') "
        f"AND _dlt_load_id IN ({placeholders}) "
        f"ORDER BY _dlt_load_id, event_type, event_id",
        [[str(path) for path in parquet_paths], *load_ids],
    ).fetchall()
    columns = (
        "event_id",
        "event_type",
        "registry_version_id",
        "payload_json",
        "observed_at",
        "_dlt_load_id",
    )
    entries = []
    snapshots = []
    for row in rows:
        event = parse_dlt_feed_event_row(dict(zip(columns, row, strict=True)))
        if isinstance(event, FeedEntryEventOccurrence):
            entries.append(event)
        else:
            snapshots.append(event)
    return DltFeedEvents(entries=tuple(entries), snapshots=tuple(snapshots))


def reconcile_feed_entry_projection(*, batch_size: int = 50) -> FeedEntryProjectionReconciliation:
    """Project only completed dlt loads that have no immutable projection marker."""
    objects = _list_dlt_parquet_objects()
    objects_by_load: dict[str, list[DltParquetObject]] = defaultdict(list)
    for value in objects:
        objects_by_load[value.load_id].append(value)
    completed = set(_list_completed_dlt_load_ids())
    catalog_state = feed_catalog.read_feed_catalog_load_state()
    projected = set(catalog_state.projected_load_ids)
    registered = dict(catalog_state.registered_at_by_load_id)
    missing_files = sorted((set(registered) & completed) - projected - set(objects_by_load))
    if missing_files:
        raise ValueError(
            f"Registered dlt loads have no Parquet objects: {', '.join(missing_files)}"
        )
    pending = tuple(sorted((set(objects_by_load) & completed) - projected))
    registered_count = 0
    cataloged_events = 0
    client: BaseClient | None = None
    for load_id in pending:
        client = client or _r2_client()
        load_objects = tuple(sorted(objects_by_load[load_id], key=lambda value: value.key))
        with tempfile.TemporaryDirectory() as directory:
            paths = _download_load_files(client, load_objects, Path(directory))
            events = read_dlt_feed_events(paths, (load_id,))
        projected_at = registered.get(load_id)
        if projected_at is None:
            projected_at = _register_unregistered_load(
                load_id,
                load_objects,
                events.entries,
                events.snapshots,
            )
            registered[load_id] = projected_at
            registered_count += 1
        result = catalog_feed_entry_events(events.entries, batch_size=batch_size)
        cataloged_events += result.cataloged_events
        feed_catalog.mark_feed_entry_projection_load(load_id, projected_at)
    return FeedEntryProjectionReconciliation(
        listed_loads=len(objects_by_load),
        examined_loads=len(pending),
        registered_loads=registered_count,
        cataloged_events=cataloged_events,
    )


def _register_unregistered_load(
    load_id: str,
    objects: tuple[DltParquetObject, ...],
    occurrences: tuple[FeedEntryEventOccurrence, ...],
    snapshot_occurrences: tuple[FeedSnapshotEventOccurrence, ...],
) -> datetime:
    registered_at = datetime.now(UTC)
    snapshot_content = {}
    snapshot_digests_by_feed: dict[str, set[Sha256]] = defaultdict(set)
    for occurrence in snapshot_occurrences:
        if occurrence.dlt_load_id != load_id:
            raise ValueError(f"Feed snapshot event belongs to another load: {load_id}")
        event = occurrence.event
        key = (event.feed_id, event.content_digest)
        existing = snapshot_content.get(key)
        if existing is not None and existing != event.content:
            raise ValueError(f"Conflicting feed snapshot event bytes: {event.feed_id}")
        snapshot_content[key] = event.content
        snapshot_digests_by_feed[event.feed_id].add(event.content_digest)
    required_snapshots = set(snapshot_content)
    for occurrence in occurrences:
        event = occurrence.event
        key = (event.entry.feed_id, event.feed_content_digest)
        required_snapshots.add(key)
        snapshot_digests_by_feed[event.entry.feed_id].add(event.feed_content_digest)
    conflicts = sorted(
        feed_id for feed_id, digests in snapshot_digests_by_feed.items() if len(digests) > 1
    )
    if conflicts:
        raise ValueError(f"Conflicting feed snapshot identities: {', '.join(conflicts)}")
    snapshots = {}
    canonical_objects = []
    for feed_id, content_digest in sorted(required_snapshots):
        key = f"news/feeds/{feed_id}/{content_digest}.xml"
        content = snapshot_content.get((feed_id, content_digest))
        if content is None:
            content = read_verified_r2_object(key, content_digest)
        else:
            canonical_objects.append((key, content))
        snapshot = artifact_file(
            artifact_id=f"news:feed:{feed_id}",
            artifact_kind="news_feed",
            title=f"RSS feed: {feed_id}",
            content=content,
            r2_key=key,
            media_type="application/xml",
        )
        snapshots[snapshot.version_id] = snapshot
    manifest_content = canonical_json(
        {
            "load_id": load_id,
            "objects": [
                {"key": value.key, "size": value.size, "etag": value.etag} for value in objects
            ],
        }
    )
    manifest_digest = sha256(manifest_content)
    manifest = artifact_file(
        artifact_id=f"news:dlt-recovery-load:{load_id}",
        artifact_kind="news_dlt_load",
        title=f"Recovered Romanian news dlt load {load_id}",
        content=manifest_content,
        r2_key=f"news/dlt-recovery-loads/{load_id}/{manifest_digest}.json",
        media_type="application/json",
    )
    publish_immutable_r2_objects((*canonical_objects, (manifest.r2_key, manifest.content)))
    feed_catalog.register_recovered_dlt_load(
        load_id,
        tuple(snapshots.values()),
        manifest,
        registered_at,
    )
    return registered_at


def _list_dlt_parquet_objects() -> tuple[DltParquetObject, ...]:
    client = _r2_client()
    values = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=NEWS_R2_BUCKET, Prefix=_DLT_FEED_ENTRY_PREFIX):
        for raw in page.get("Contents", []):
            key = str(raw["Key"])
            load_id = _load_id_from_parquet_key(key)
            if load_id is None:
                continue
            values.append(
                DltParquetObject(
                    load_id=load_id,
                    key=key,
                    size=int(raw["Size"]),
                    etag=str(raw["ETag"]),
                )
            )
    return tuple(sorted(values, key=lambda value: value.key))


def _list_completed_dlt_load_ids() -> tuple[str, ...]:
    client = _r2_client()
    load_ids = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=NEWS_R2_BUCKET,
        Prefix=_DLT_LOAD_COMPLETION_PREFIX,
    ):
        for raw in page.get("Contents", []):
            load_id = _completed_load_id_from_key(str(raw["Key"]))
            if load_id is not None:
                load_ids.add(load_id)
    return tuple(sorted(load_ids))


def _completed_load_id_from_key(key: str) -> str | None:
    name = Path(key).name
    if not name.endswith(".jsonl"):
        return None
    stem = name.removesuffix(".jsonl")
    parts = stem.rsplit("__", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[1]


def _load_id_from_parquet_key(key: str) -> str | None:
    name = Path(key).name
    parts = name.rsplit(".", 2)
    if len(parts) != 3 or parts[2] != "parquet" or not parts[0] or not parts[1]:
        return None
    return parts[0]


def _download_load_files(
    client: BaseClient,
    objects: tuple[DltParquetObject, ...],
    directory: Path,
) -> tuple[Path, ...]:
    paths = []
    for index, value in enumerate(objects):
        path = directory / f"{index:06d}.parquet"
        client.download_file(NEWS_R2_BUCKET, value.key, str(path))
        paths.append(path)
    return tuple(paths)


def _r2_client() -> BaseClient:
    config = r2_s3_config()
    return boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        region_name="auto",
    )
