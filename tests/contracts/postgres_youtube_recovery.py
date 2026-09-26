from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from romanian_news.catalog.youtube import claim_youtube_video, reject_youtube_video
from romanian_news.youtube.errors import YouTubeDeterministicError
from romanian_news.youtube.models import YOUTUBE_SOURCES, YouTubeVideoState
from romanian_news.youtube.recovery import (
    inspect_youtube_quarantine,
    release_quarantined_youtube_video,
)
from tests.postgres_catalog import postgres_catalog_fixture

postgres_catalog = postgres_catalog_fixture("youtube_recovery")
SOURCE = "recorder-youtube"
SOURCE_SPEC = next(source for source in YOUTUBE_SOURCES.sources if source.source_id == SOURCE)
VIDEO = "abcdefghijk"
REQUESTED_AT = datetime(2026, 9, 25, tzinfo=UTC)
POISON = ValueError("deterministic poison payload")


def _seed_quarantine(catalog) -> None:
    catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
        "visibility, created_at) VALUES ('poll', 'youtube_feed_snapshot', 'Poll', "
        "'source', 'current', 'private', %s)",
        (REQUESTED_AT,),
    )
    catalog.execute(
        "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
        "created_at) VALUES ('poll-v1', 'poll', 1, 'digest', %s)",
        (REQUESTED_AT,),
    )
    catalog.execute(
        "INSERT INTO youtube_videos (source_id, video_id, first_poll_version_id, "
        "state, published_at, title, deterministic_failure_fingerprint, "
        "unchanged_deterministic_failures, last_error, quarantine_generation) "
        "VALUES (%s, %s, 'poll-v1', 'quarantined', %s, 'Video', 'fingerprint', 3, "
        "'bad payload', 1)",
        (SOURCE, VIDEO, REQUESTED_AT),
    )


def _release(request_id, *, generation: int = 1):
    return release_quarantined_youtube_video(
        SOURCE,
        VIDEO,
        expected_generation=generation,
        request_id=request_id,
        requested_by="operator",
        reason="Source payload fixed",
        requested_at=REQUESTED_AT,
    )


def _seed_pending_video(catalog) -> None:
    first_poll_version_id = "3" * 64
    catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
        "visibility, created_at) VALUES ('poll', 'youtube_feed_snapshot', 'Poll', "
        "'source', 'current', 'private', %s)",
        (REQUESTED_AT,),
    )
    catalog.execute(
        "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
        "created_at) VALUES (%s, 'poll', 1, 'digest', %s)",
        (first_poll_version_id, REQUESTED_AT),
    )
    catalog.execute(
        "INSERT INTO youtube_videos (source_id, video_id, first_poll_version_id, "
        "state, published_at, title) VALUES (%s, %s, %s, 'pending', %s, 'Video')",
        (SOURCE, VIDEO, first_poll_version_id, REQUESTED_AT),
    )


def test_repeated_rejection_increments_quarantine_generation_across_release_cycles(
    postgres_catalog,
) -> None:
    _seed_pending_video(postgres_catalog)
    now = datetime(2026, 9, 25, 10, tzinfo=UTC)

    for cycle in (1, 2):
        state = None
        for attempt in range(3):
            lease = claim_youtube_video(SOURCE_SPEC, f"owner-{cycle}-{attempt}", now)
            assert lease is not None
            state = reject_youtube_video(lease, POISON, now)
            now = now + timedelta(hours=1)

        assert state == YouTubeVideoState.QUARANTINED
        rows = postgres_catalog.query(
            "SELECT state, quarantine_generation, retry_at FROM youtube_videos "
            "WHERE source_id = %s AND video_id = %s",
            [SOURCE, VIDEO],
        )
        assert rows[0]["state"] == "quarantined"
        assert rows[0]["quarantine_generation"] == cycle
        assert rows[0]["retry_at"] is None

        _release(uuid4(), generation=cycle)

        released = inspect_youtube_quarantine(SOURCE, VIDEO)
        assert released.state == "pending"
        assert released.quarantine_generation == cycle
        assert released.failure_fingerprint is None


def test_release_is_recorded_and_replay_is_exact(postgres_catalog) -> None:
    _seed_quarantine(postgres_catalog)
    request_id = uuid4()
    receipt = _release(request_id)
    assert _release(request_id) == receipt
    status = inspect_youtube_quarantine(SOURCE, VIDEO)
    assert status.state == "pending"
    assert status.quarantine_generation == 1
    assert status.failure_fingerprint is None
    assert status.unchanged_failures == 0
    assert status.last_error is None
    rows = postgres_catalog.query("SELECT * FROM youtube_quarantine_releases")
    assert len(rows) == 1
    assert rows[0]["failure_fingerprint"] == "fingerprint"
    assert rows[0]["last_error"] == "bad payload"
    with pytest.raises(ValueError, match="different details"):
        release_quarantined_youtube_video(
            SOURCE,
            VIDEO,
            expected_generation=1,
            request_id=request_id,
            requested_by="operator",
            reason="Different reason",
            requested_at=REQUESTED_AT,
        )


def test_concurrent_release_only_records_one_transition(postgres_catalog) -> None:
    _seed_quarantine(postgres_catalog)
    request_id = uuid4()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(_release, (request_id, request_id)))
    assert results[0] == results[1]
    assert len(postgres_catalog.query("SELECT request_id FROM youtube_quarantine_releases")) == 1
    with pytest.raises(ValueError, match="changed"):
        _release(uuid4())


def test_stale_release_cannot_reopen_next_quarantine(postgres_catalog) -> None:
    _seed_quarantine(postgres_catalog)
    first = uuid4()
    _release(first)
    postgres_catalog.execute(
        "UPDATE youtube_videos SET state = 'quarantined', quarantine_generation = 2, "
        "deterministic_failure_fingerprint = 'new-fingerprint', "
        "unchanged_deterministic_failures = 3, last_error = 'new failure' "
        "WHERE source_id = %s AND video_id = %s",
        (SOURCE, VIDEO),
    )
    with pytest.raises(ValueError, match="changed"):
        _release(uuid4(), generation=1)
    assert _release(first).quarantine_generation == 1
    assert inspect_youtube_quarantine(SOURCE, VIDEO).state == "quarantined"
    _release(uuid4(), generation=2)
    assert len(postgres_catalog.query("SELECT request_id FROM youtube_quarantine_releases")) == 2


def test_repeated_deterministic_failures_quarantine_with_generation_one(postgres_catalog) -> None:
    postgres_catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
        "visibility, created_at) VALUES ('poll', 'youtube_feed_snapshot', 'Poll', "
        "'source', 'current', 'private', %s)",
        (REQUESTED_AT,),
    )
    poll_version = "5" * 64
    postgres_catalog.execute(
        "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
        "created_at) VALUES (%s, 'poll', 1, 'digest', %s)",
        (poll_version, REQUESTED_AT),
    )
    postgres_catalog.execute(
        "INSERT INTO youtube_videos (source_id, video_id, first_poll_version_id, "
        "state, published_at, title) VALUES (%s, %s, %s, 'pending', %s, 'Video')",
        (SOURCE, VIDEO, poll_version, REQUESTED_AT),
    )
    source = YOUTUBE_SOURCES.source(SOURCE)
    error = YouTubeDeterministicError("payload is unusable")
    now = datetime(2026, 9, 25, 10, tzinfo=UTC)

    states = []
    for attempt in range(3):
        at = now + timedelta(hours=attempt)
        lease = claim_youtube_video(source, "worker-1", at)
        assert lease is not None
        states.append(reject_youtube_video(lease, error, at))

    assert states == ["deferred", "deferred", "quarantined"]
    status = inspect_youtube_quarantine(SOURCE, VIDEO)
    assert status.state == "quarantined"
    assert status.quarantine_generation == 1
    receipt = _release(uuid4())
    assert receipt.quarantine_generation == 1
