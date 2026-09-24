from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog_transport import CatalogConnection, catalog_query, catalog_transaction


@dataclass(frozen=True)
class YouTubeQuarantineStatus:
    source_id: str
    video_id: str
    state: str
    quarantine_generation: int
    failure_fingerprint: str | None
    unchanged_failures: int
    last_error: str | None


@dataclass(frozen=True)
class YouTubeQuarantineRelease:
    request_id: UUID
    source_id: str
    video_id: str
    quarantine_generation: int
    requested_by: str
    reason: str
    requested_at: datetime


def inspect_youtube_quarantine(source_id: str, video_id: str) -> YouTubeQuarantineStatus:
    ensure_news_catalog_schema()
    rows = catalog_query(
        "SELECT state, quarantine_generation, deterministic_failure_fingerprint, "
        "unchanged_deterministic_failures, last_error FROM youtube_videos "
        "WHERE source_id = %s AND video_id = %s",
        (source_id, video_id),
    )
    if not rows:
        raise ValueError("YouTube video not found")
    row = rows[0]
    return YouTubeQuarantineStatus(
        source_id=source_id,
        video_id=video_id,
        state=str(row["state"]),
        quarantine_generation=int(row["quarantine_generation"]),
        failure_fingerprint=(
            str(row["deterministic_failure_fingerprint"])
            if row["deterministic_failure_fingerprint"] is not None
            else None
        ),
        unchanged_failures=int(row["unchanged_deterministic_failures"]),
        last_error=str(row["last_error"]) if row["last_error"] is not None else None,
    )


def release_quarantined_youtube_video(
    source_id: str,
    video_id: str,
    *,
    expected_generation: int,
    request_id: UUID,
    requested_by: str,
    reason: str,
    requested_at: datetime,
) -> YouTubeQuarantineRelease:
    if expected_generation <= 0:
        raise ValueError("Expected quarantine generation must be positive")
    if not requested_by.strip() or not reason.strip():
        raise ValueError("Requested by and reason are required")
    if requested_at.tzinfo is None:
        raise ValueError("Requested at must include a timezone")
    requested_at = requested_at.astimezone(UTC)
    ensure_news_catalog_schema()
    release = YouTubeQuarantineRelease(
        request_id, source_id, video_id, expected_generation, requested_by, reason, requested_at
    )

    def commit(connection: CatalogConnection) -> YouTubeQuarantineRelease:
        row = connection.execute(
            "SELECT state, quarantine_generation, deterministic_failure_fingerprint, "
            "unchanged_deterministic_failures, last_error FROM youtube_videos "
            "WHERE source_id = %s AND video_id = %s FOR UPDATE",
            (source_id, video_id),
        ).fetchone()
        if row is None:
            raise ValueError("YouTube video not found")
        existing = connection.execute(
            "SELECT source_id, video_id, quarantine_generation, requested_by, reason, requested_at "
            "FROM youtube_quarantine_releases WHERE request_id = %s",
            (request_id,),
        ).fetchone()
        if existing is not None:
            recorded = YouTubeQuarantineRelease(
                request_id,
                str(existing["source_id"]),
                str(existing["video_id"]),
                int(existing["quarantine_generation"]),
                str(existing["requested_by"]),
                str(existing["reason"]),
                existing["requested_at"],
            )
            if recorded != release:
                raise ValueError("YouTube recovery request ID was used for different details")
            return recorded
        if row["state"] != "quarantined" or row["quarantine_generation"] != expected_generation:
            raise ValueError("YouTube quarantine changed; inspect the current generation")
        if row["deterministic_failure_fingerprint"] is None or row["last_error"] is None:
            raise ValueError("YouTube quarantine has no failure evidence to preserve")
        connection.execute(
            "INSERT INTO youtube_quarantine_releases "
            "(request_id, source_id, video_id, quarantine_generation, failure_fingerprint, "
            "unchanged_failures, last_error, requested_by, reason, requested_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                request_id,
                source_id,
                video_id,
                expected_generation,
                row["deterministic_failure_fingerprint"],
                row["unchanged_deterministic_failures"],
                row["last_error"],
                requested_by,
                reason,
                requested_at,
            ),
        )
        connection.execute(
            "UPDATE youtube_videos SET state = 'pending', retry_at = NULL, "
            "deterministic_failure_fingerprint = NULL, unchanged_deterministic_failures = 0, "
            "last_error = NULL WHERE source_id = %s AND video_id = %s",
            (source_id, video_id),
        )
        return release

    return catalog_transaction(commit)
