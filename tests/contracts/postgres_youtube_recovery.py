from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from romanian_news.youtube.recovery import (
    inspect_youtube_quarantine,
    release_quarantined_youtube_video,
)
from tests.postgres_catalog import postgres_catalog_fixture

postgres_catalog = postgres_catalog_fixture("youtube_recovery")
SOURCE = "recorder-youtube"
VIDEO = "abcdefghijk"
REQUESTED_AT = datetime(2026, 9, 25, tzinfo=UTC)


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
