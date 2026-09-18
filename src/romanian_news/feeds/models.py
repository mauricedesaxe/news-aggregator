from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, HttpUrl, model_validator

from romanian_news import BUCHAREST, FeedId, NewsModel, OutletId, Sha256


class FeedSpec(NewsModel):
    id: FeedId
    outlet_id: OutletId
    outlet_name: Annotated[str, Field(min_length=1)]
    url: HttpUrl
    category: Literal["general", "economic", "political", "investigative", "regional"]
    article_hosts: tuple[Annotated[str, Field(min_length=1)], ...]
    article_xpath: str | None = None
    excluded_article_path_prefixes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_article_hosts(self) -> FeedSpec:
        if not self.article_hosts:
            raise ValueError("A feed must allow at least one article host")
        normalized = tuple(host.lower().removeprefix("www.") for host in self.article_hosts)
        if len(set(normalized)) != len(normalized):
            raise ValueError("Article hosts must be unique")
        if any(not prefix.startswith("/") for prefix in self.excluded_article_path_prefixes):
            raise ValueError("Excluded article path prefixes must start with a slash")
        return self

    def accepts_article_url(self, url: str) -> bool:
        path = urlsplit(url).path
        return not any(path.startswith(prefix) for prefix in self.excluded_article_path_prefixes)


class FeedEntry(NewsModel):
    feed_id: FeedId
    source_id: Annotated[str, Field(min_length=1)]
    url: HttpUrl
    title: Annotated[str, Field(min_length=1)]
    summary: str
    feed_content: str | None
    published_at: datetime
    source_updated_at: datetime | None
    author: str | None

    @model_validator(mode="after")
    def validate_datetimes(self) -> FeedEntry:
        if self.published_at.tzinfo is None:
            raise ValueError("Published time must include a UTC offset")
        if self.source_updated_at is not None and self.source_updated_at.tzinfo is None:
            raise ValueError("Updated time must include a UTC offset")
        return self


class FeedEntryVersion(NewsModel):
    version_id: Sha256
    entry: FeedEntry

    @model_validator(mode="after")
    def validate_identity(self) -> FeedEntryVersion:
        if self.version_id != feed_entry_version_id(self.entry):
            raise ValueError("Feed entry version does not match its content")
        return self


class FeedEntryEvent(NewsModel):
    version: FeedEntryVersion
    registry_version_id: Sha256
    feed_content_digest: Sha256
    observed_at: datetime

    @property
    def event_id(self) -> Sha256:
        return self.version.version_id

    @property
    def entry(self) -> FeedEntry:
        return self.version.entry

    @model_validator(mode="after")
    def validate_observed_at(self) -> FeedEntryEvent:
        if self.observed_at.tzinfo is None:
            raise ValueError("Feed entry observation time must include a UTC offset")
        return self


class CatalogedFeedEntryReference(NewsModel):
    event_id: Sha256
    dlt_load_id: Annotated[str, Field(min_length=1)]
    registry_version_id: Sha256
    feed_snapshot_version_id: Sha256
    observed_at: datetime
    feed_id: FeedId
    source_id: Annotated[str, Field(min_length=1)]
    url: HttpUrl
    published_at: datetime
    source_updated_at: datetime | None

    @model_validator(mode="after")
    def validate_datetimes(self) -> CatalogedFeedEntryReference:
        for name, value in (
            ("observation", self.observed_at),
            ("publication", self.published_at),
            ("source update", self.source_updated_at),
        ):
            if value is not None and value.tzinfo is None:
                raise ValueError(f"Cataloged feed entry {name} time must include a UTC offset")
        return self


class CatalogedFeedEntry(CatalogedFeedEntryReference):
    entry: FeedEntry

    @model_validator(mode="after")
    def validate_entry_identity(self) -> CatalogedFeedEntry:
        if (
            self.entry.feed_id != self.feed_id
            or self.entry.source_id != self.source_id
            or self.entry.url != self.url
            or self.entry.published_at != self.published_at
            or self.entry.source_updated_at != self.source_updated_at
        ):
            raise ValueError("Materialized feed entry does not match its catalog reference")
        return self


class FeedSnapshotEvent(NewsModel):
    event_id: Sha256
    feed_id: FeedId
    content_digest: Sha256
    content: bytes


class FeedSnapshotEventOccurrence(NewsModel):
    dlt_load_id: Annotated[str, Field(min_length=1)]
    event: FeedSnapshotEvent


class FeedEntryEventOccurrence(NewsModel):
    source_event_id: Sha256
    dlt_load_id: Annotated[str, Field(min_length=1)]
    event: FeedEntryEvent

    @model_validator(mode="after")
    def validate_source_event_id(self) -> FeedEntryEventOccurrence:
        valid_event_ids = {
            self.event.event_id,
            legacy_feed_entry_event_id(self.event.entry, self.event.feed_content_digest),
        }
        if self.source_event_id not in valid_event_ids:
            raise ValueError("Feed entry occurrence identity does not match its event")
        return self


class DltFeedEvents(NewsModel):
    entries: tuple[FeedEntryEventOccurrence, ...]
    snapshots: tuple[FeedSnapshotEventOccurrence, ...]


class FeedRegistry(NewsModel):
    feeds: tuple[FeedSpec, ...]
    version_id: Sha256

    @model_validator(mode="after")
    def validate_registry(self) -> FeedRegistry:
        ids = [feed.id for feed in self.feeds]
        urls = [str(feed.url) for feed in self.feeds]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("Feeds must have sorted unique IDs")
        if len(urls) != len(set(urls)):
            raise ValueError("Feed URLs must be unique")
        if self.version_id != registry_version_id(self.feeds):
            raise ValueError("Feed registry version does not match its content")
        return self


class FeedValidator(NewsModel):
    etag: str | None = None
    last_modified: str | None = None


class FeedCapture(NewsModel):
    feed_id: str
    observed_at: datetime
    scheduled_slot: datetime
    status: Literal["ok", "not_modified", "failed"]
    http_status: int | None
    latency_ms: Annotated[int, Field(ge=0)]
    content: bytes | None
    content_digest: Sha256 | None
    entries: tuple[FeedEntry, ...]
    rejected_entries: Annotated[int, Field(ge=0)]
    validator: FeedValidator
    error: str | None


class FeedAcquisitionRequest(NewsModel):
    registry: FeedRegistry
    observed_at: datetime
    scheduled_slot: datetime | None = None
    workspace_dir: Path
    destination: Literal["local", "r2"] = "r2"

    @model_validator(mode="after")
    def validate_times(self) -> FeedAcquisitionRequest:
        if self.observed_at.tzinfo is None:
            raise ValueError("Feed observation time must include a UTC offset")
        if self.scheduled_slot is not None and self.scheduled_slot.tzinfo is None:
            raise ValueError("Feed schedule time must include a UTC offset")
        return self


class FeedAcquisitionResult(NewsModel):
    load_ids: tuple[str, ...]
    captures: tuple[FeedCapture, ...]


class CurrentFeedState(NewsModel):
    acquisition: FeedAcquisitionResult
    snapshot_versions: dict[str, Sha256]
    missing_feed_ids: tuple[str, ...] = ()


def feed_entry_event(
    entry: FeedEntry,
    capture: FeedCapture,
    registry_version_id: Sha256,
) -> FeedEntryEvent:
    """Create the durable dlt event for one entry from its feed snapshot."""
    if capture.content_digest is None:
        raise ValueError("A feed entry event requires feed snapshot bytes")
    return FeedEntryEvent(
        version=FeedEntryVersion(
            version_id=feed_entry_version_id(entry),
            entry=entry,
        ),
        registry_version_id=registry_version_id,
        feed_content_digest=capture.content_digest,
        observed_at=capture.observed_at,
    )


def feed_entry_version_id(entry: FeedEntry) -> Sha256:
    return _feed_entry_payload_id(entry.model_dump(mode="json"))


def legacy_feed_entry_event_id(entry: FeedEntry, feed_content_digest: Sha256) -> Sha256:
    """Compute the identity stored by the original dlt feed-entry schema."""
    payload = entry.model_dump(mode="json")
    payload["feed_content_digest"] = feed_content_digest
    return _feed_entry_payload_id(payload)


def _feed_entry_payload_id(payload: dict[str, object]) -> Sha256:
    content = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(content).hexdigest()


def registry_version_id(feeds: tuple[FeedSpec, ...]) -> Sha256:
    content = json.dumps(
        [feed.model_dump(mode="json") for feed in feeds],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(content).hexdigest()


def feed_schedule_slot(observed_at: datetime) -> datetime:
    if observed_at.tzinfo is None:
        raise ValueError("Feed schedule time must include a UTC offset")
    local = observed_at.astimezone(BUCHAREST)
    return local.replace(minute=local.minute // 15 * 15, second=0, microsecond=0)
