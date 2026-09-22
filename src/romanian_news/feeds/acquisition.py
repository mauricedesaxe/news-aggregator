from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Literal

import dlt
import requests

from romanian_news import Sha256
from romanian_news.config import NEWS_R2_BUCKET
from romanian_news.feeds import models as feed_models
from romanian_news.feeds.models import (
    FeedAcquisitionRequest,
    FeedAcquisitionResult,
    FeedCapture,
    FeedEntry,
    FeedEntryEvent,
    FeedEntryVersion,
    FeedSnapshotEvent,
    FeedSpec,
    FeedValidator,
    feed_entry_version_id,
    feed_schedule_slot,
)
from romanian_news.feeds.parsing import parse_feed
from romanian_news.http import create_news_session, get_with_validated_redirects
from romanian_news.identity import canonical_json as _canonical_json
from romanian_news.identity import sha256 as _sha256
from romanian_news.storage import r2_s3_config

_USER_AGENT = "Romanian news aggregator/1.0 (+https://github.com/mauricedesaxe/news-aggregator)"


def acquire_feeds(request: FeedAcquisitionRequest) -> FeedAcquisitionResult:
    """Poll every feed and commit its validators with the completed dlt load."""
    from romanian_news.catalog.feeds import read_durable_feed_ids

    captures: list[FeedCapture] = []
    durable_feeds = read_durable_feed_ids()

    @dlt.resource(name="rss_events", write_disposition="append")
    def rss_events() -> Any:
        state = dlt.current.resource_state()
        raw_validators = state.setdefault("validators", {})
        validators = {
            feed.id: (
                FeedValidator.model_validate(raw_validators.get(feed.id, {}))
                if feed.id in durable_feeds
                else FeedValidator()
            )
            for feed in request.registry.feeds
        }
        polled = _poll_feeds(request.registry.feeds, request.observed_at, validators)
        for feed, capture in zip(request.registry.feeds, polled, strict=True):
            if request.scheduled_slot is not None:
                capture = capture.model_copy(update={"scheduled_slot": request.scheduled_slot})
            captures.append(capture)
            if capture.status != "failed":
                raw_validators[feed.id] = capture.validator.model_dump(mode="json")
            yield _observation_record(capture, request.registry.version_id)
            if capture.status == "ok":
                yield _snapshot_record(feed_snapshot_event(capture))
            for entry in capture.entries:
                yield _entry_record(
                    feed_models.feed_entry_event(entry, capture, request.registry.version_id)
                )

    pipeline = dlt.pipeline(
        pipeline_name="chartly_romanian_news_live",
        pipelines_dir=str(request.workspace_dir / ".dlt"),
        destination=_destination(request),
        dataset_name="romanian_news_intake",
    )
    load_info = pipeline.run(rss_events(), loader_file_format="parquet")
    load_ids = tuple(load_info.loads_ids)
    if len(load_ids) != 1:
        raise ValueError(f"Expected one dlt load for one pipeline run, received {len(load_ids)}")
    return FeedAcquisitionResult(
        load_ids=load_ids,
        captures=tuple(captures),
    )


def _poll_feeds(
    feeds: tuple[FeedSpec, ...],
    observed_at: datetime,
    validators: dict[str, FeedValidator],
) -> tuple[FeedCapture, ...]:
    if not feeds:
        return ()
    with ThreadPoolExecutor(max_workers=len(feeds)) as executor:
        futures = [
            executor.submit(_fetch_feed_with_session, feed, observed_at, validators[feed.id])
            for feed in feeds
        ]
        return tuple(future.result() for future in futures)


def fetch_feed(
    session: requests.Session,
    feed: FeedSpec,
    observed_at: datetime,
    validator: FeedValidator,
) -> FeedCapture:
    """Fetch and parse one feed without letting one source abort the full poll."""
    if observed_at.tzinfo is None:
        raise ValueError("Feed observation time must include a UTC offset")
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
    }
    if validator.etag:
        headers["If-None-Match"] = validator.etag
    if validator.last_modified:
        headers["If-Modified-Since"] = validator.last_modified
    started = time.monotonic()
    try:
        response = get_with_validated_redirects(
            session,
            str(feed.url),
            headers=headers,
            timeout=15,
        )
        latency_ms = round((time.monotonic() - started) * 1000)
        next_validator = FeedValidator(
            etag=response.headers.get("ETag") or validator.etag,
            last_modified=response.headers.get("Last-Modified") or validator.last_modified,
        )
        if response.status_code == 304:
            return _capture(feed, observed_at, "not_modified", 304, latency_ms, next_validator)
        response.raise_for_status()
        entries, rejected = parse_feed(feed, response.content)
        content_digest = _sha256(response.content)
        return FeedCapture(
            feed_id=feed.id,
            observed_at=observed_at.astimezone(UTC),
            scheduled_slot=feed_schedule_slot(observed_at),
            status="ok",
            http_status=response.status_code,
            latency_ms=latency_ms,
            content=response.content,
            content_digest=content_digest,
            entries=entries,
            rejected_entries=rejected,
            validator=next_validator,
            error=None,
        )
    except (requests.RequestException, ValueError) as error:
        return _capture(
            feed,
            observed_at,
            "failed",
            getattr(getattr(error, "response", None), "status_code", None),
            round((time.monotonic() - started) * 1000),
            validator,
            error=str(error),
        )


def _fetch_feed_with_session(
    feed: FeedSpec,
    observed_at: datetime,
    validator: FeedValidator,
) -> FeedCapture:
    session = create_news_session()
    try:
        return fetch_feed(session, feed, observed_at, validator)
    finally:
        session.close()


def _destination(request: FeedAcquisitionRequest) -> Any:
    if request.destination == "local":
        destination_dir = (request.workspace_dir / "dlt-destination").resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        return dlt.destinations.filesystem(bucket_url=destination_dir.as_uri())
    config = r2_s3_config()
    return dlt.destinations.filesystem(
        bucket_url=f"s3://{NEWS_R2_BUCKET}/dlt/romanian-news",
        credentials={
            "aws_access_key_id": config.access_key_id,
            "aws_secret_access_key": config.secret_access_key,
            "endpoint_url": config.endpoint_url,
            "region_name": "auto",
        },
    )


def _capture(
    feed: FeedSpec,
    observed_at: datetime,
    status: Literal["not_modified", "failed"],
    http_status: int | None,
    latency_ms: int,
    validator: FeedValidator,
    error: str | None = None,
) -> FeedCapture:
    return FeedCapture(
        feed_id=feed.id,
        observed_at=observed_at.astimezone(UTC),
        scheduled_slot=feed_schedule_slot(observed_at),
        status=status,
        http_status=http_status,
        latency_ms=latency_ms,
        content=None,
        content_digest=None,
        entries=(),
        rejected_entries=0,
        validator=validator,
        error=error,
    )


def _observation_record(capture: FeedCapture, registry_version_id: str) -> dict[str, object]:
    payload = capture.model_dump(mode="json", exclude={"content", "entries"})
    return {
        "event_id": _sha256(_canonical_json(payload)),
        "event_type": "feed_observation",
        "registry_version_id": registry_version_id,
        "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        "observed_at": capture.observed_at,
    }


def feed_snapshot_event(capture: FeedCapture) -> FeedSnapshotEvent:
    if capture.content is None or capture.content_digest is None:
        raise ValueError("A feed snapshot event requires response bytes")
    return feed_snapshot_event_from_payload(
        capture.feed_id,
        capture.content_digest,
        capture.content,
    )


def feed_snapshot_event_from_payload(
    feed_id: str,
    content_digest: Sha256,
    content: bytes,
) -> FeedSnapshotEvent:
    if _sha256(content) != content_digest:
        raise ValueError(f"Feed snapshot content digest mismatch: {feed_id}")
    payload = _feed_snapshot_payload(feed_id, content_digest, content)
    return FeedSnapshotEvent(
        event_id=_sha256(_canonical_json(payload)),
        feed_id=feed_id,
        content_digest=content_digest,
        content=content,
    )


def _snapshot_record(event: FeedSnapshotEvent) -> dict[str, object]:
    payload = _feed_snapshot_payload(event.feed_id, event.content_digest, event.content)
    return {
        "event_id": event.event_id,
        "event_type": "feed_snapshot",
        "registry_version_id": None,
        "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        "observed_at": None,
    }


def _feed_snapshot_payload(
    feed_id: str,
    content_digest: Sha256,
    content: bytes,
) -> dict[str, object]:
    return {
        "feed_id": feed_id,
        "content_digest": content_digest,
        "content_base64": base64.b64encode(content).decode("ascii"),
    }


def feed_entry_event_from_payload(
    entry: FeedEntry,
    feed_content_digest: Sha256,
    registry_version_id: Sha256,
    observed_at: datetime,
) -> FeedEntryEvent:
    """Rebuild a durable feed-entry event from its stored dlt payload."""
    return FeedEntryEvent(
        version=FeedEntryVersion(
            version_id=feed_entry_version_id(entry),
            entry=entry,
        ),
        registry_version_id=registry_version_id,
        feed_content_digest=feed_content_digest,
        observed_at=observed_at,
    )


def _entry_record(event: FeedEntryEvent) -> dict[str, object]:
    payload = _dlt_feed_entry_payload(event.entry, event.feed_content_digest)
    return {
        "event_id": event.event_id,
        "event_type": "feed_entry",
        "registry_version_id": event.registry_version_id,
        "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        "observed_at": event.observed_at,
    }


def _dlt_feed_entry_payload(
    entry: FeedEntry,
    feed_content_digest: Sha256,
) -> dict[str, object]:
    payload = entry.model_dump(mode="json")
    payload["feed_content_digest"] = feed_content_digest
    return payload
