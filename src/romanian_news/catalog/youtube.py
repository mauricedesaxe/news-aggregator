from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from romanian_news import BUCHAREST, NewsModel, Sha256
from romanian_news.analysis.attempts import ModelAttempt
from romanian_news.articles.models import ExtractedArticle
from romanian_news.catalog.artifacts import (
    ArtifactFile,
    artifact_file,
    artifact_statements,
    canonical_json,
    sha256,
)
from romanian_news.catalog.model_calls import ModelAttemptRecord, ModelTraceRecord
from romanian_news.catalog_transport import (
    ResearchCatalogError,
    advance_artifact_current_version_from_run_statement,
    advance_artifact_current_version_statement,
    catalog_batch,
    catalog_mutation,
    catalog_query,
)
from romanian_news.storage import publish_immutable_r2_objects, read_verified_r2_object
from romanian_news.youtube.errors import (
    YouTubeEvidenceError,
    YouTubeLeaseLostError,
    YouTubeMergeError,
)
from romanian_news.youtube.models import (
    LIVE_MEDIA_REPLAY_LIMIT,
    MAX_UNCHANGED_DETERMINISTIC_FAILURES,
    YOUTUBE_LEASE_SECONDS,
    YOUTUBE_MODEL,
    AcceptedClipExtraction,
    CandidateEvidence,
    GeminiResponse,
    MeasuredCost,
    SystemApproval,
    UnmeasuredCost,
    YouTubeCandidate,
    YouTubeMergedArticle,
    YouTubeModelReceipt,
    YouTubePollCapture,
    YouTubePollDecision,
    YouTubePollStatus,
    YouTubePublicationResult,
    YouTubeSource,
    YouTubeVideoId,
    YouTubeVideoLease,
    YouTubeVideoMetadata,
    YouTubeVideoState,
)

CLAIM_YOUTUBE_VIDEO_SQL = """
UPDATE youtube_videos
SET state = 'running', owner_token = %s, lease_expires_at = %s, retry_at = NULL
WHERE source_id = %s AND video_id = COALESCE(
    (SELECT video_id FROM youtube_videos WHERE source_id = %s AND state = 'running' AND owner_token = %s LIMIT 1),
    (SELECT video_id FROM youtube_videos
     WHERE source_id = %s AND (
         state = 'pending'
         OR (state = 'deferred' AND retry_at <= %s)
         OR (state = 'running' AND lease_expires_at <= %s)
     )
     ORDER BY CASE state WHEN 'pending' THEN 0 WHEN 'running' THEN 1 WHEN 'deferred' THEN 2 END,
              published_at, video_id
     LIMIT 1)
)
RETURNING video_id, owner_token, lease_expires_at, first_poll_version_id, metadata_version_id,
          deterministic_failure_fingerprint, unchanged_deterministic_failures
"""


class AcceptedClipReference(NewsModel):
    artifact_version_id: Sha256
    start_second: int
    duration_seconds: int


def publish_youtube_poll(capture: YouTubePollCapture) -> YouTubePollDecision:
    from romanian_news.youtube.discovery import decide_youtube_poll

    source = capture.source
    slot = capture.scheduled_at.astimezone(UTC).isoformat()
    recorded = catalog_query(
        "SELECT status FROM youtube_source_polls WHERE source_id = %s AND scheduled_at = %s",
        [source.source_id, slot],
    )
    if recorded:
        return YouTubePollDecision(
            source=source,
            status=YouTubePollStatus(str(recorded[0]["status"])),
            pending_video_ids=(),
        )
    known_rows = catalog_query(
        "SELECT video_id FROM youtube_videos WHERE source_id = %s", [source.source_id]
    )
    channel = catalog_query(
        "SELECT baseline_poll_version_id FROM youtube_source_state WHERE source_id = %s",
        [source.source_id],
    )
    known = frozenset(str(row["video_id"]) for row in known_rows)
    decision = decide_youtube_poll(capture.feed, known, baseline_exists=bool(channel))
    content_digest = sha256(capture.content)
    slot_digest = sha256(f"{source.source_id}\0{slot}".encode())
    value = artifact_file(
        artifact_id=f"news:youtube-poll:{source.source_id}:{slot_digest}",
        artifact_kind="youtube_feed_snapshot",
        title=f"{source.display_name} YouTube feed snapshot {slot}",
        content=capture.content,
        r2_key=f"news/youtube/{source.source_id}/feed/{slot}/{content_digest}.json",
        media_type="application/json",
    )
    publish_immutable_r2_objects(((value.r2_key, value.content),))
    timestamp = capture.observed_at.astimezone(UTC).isoformat()
    statements = artifact_statements(
        value, timestamp, produced_by_run_id=None, authority_class="source"
    )
    statements.append(
        advance_artifact_current_version_statement(value.artifact_id, value.version_id)
    )
    statements.append(
        (
            "INSERT INTO youtube_source_polls "
            "(artifact_version_id, source_id, scheduled_at, observed_at, status, entry_count) VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING",
            [
                value.version_id,
                source.source_id,
                slot,
                timestamp,
                decision.status.value,
                len(capture.feed.entries),
            ],
        )
    )
    if decision.status != YouTubePollStatus.OVERLAP_LOST:
        baseline_version = (
            value.version_id if not channel else channel[0]["baseline_poll_version_id"]
        )
        statements.append(
            (
                "INSERT INTO youtube_source_state VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT(source_id) DO UPDATE SET last_poll_version_id = excluded.last_poll_version_id, updated_at = excluded.updated_at",
                [
                    source.source_id,
                    source.channel_id,
                    baseline_version,
                    value.version_id,
                    timestamp,
                ],
            )
        )
        pending = set(decision.pending_video_ids)
        for entry in capture.feed.entries:
            statements.append(
                (
                    "INSERT INTO youtube_videos "
                    "(source_id, video_id, first_poll_version_id, state, published_at, title) VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT DO NOTHING",
                    [
                        source.source_id,
                        entry.video_id,
                        value.version_id,
                        YouTubeVideoState.PENDING
                        if entry.video_id in pending
                        else YouTubeVideoState.BASELINE,
                        entry.published_at.astimezone(UTC).isoformat(),
                        entry.title,
                    ],
                )
            )
    catalog_batch(statements)
    return decision


def claim_youtube_video(
    source: YouTubeSource, owner_token: str, now: datetime
) -> YouTubeVideoLease | None:
    expires = now.astimezone(UTC) + timedelta(seconds=YOUTUBE_LEASE_SECONDS)
    rows = catalog_mutation(
        CLAIM_YOUTUBE_VIDEO_SQL,
        [
            owner_token,
            expires.isoformat(),
            source.source_id,
            source.source_id,
            owner_token,
            source.source_id,
            now.astimezone(UTC).isoformat(),
            now.astimezone(UTC).isoformat(),
        ],
    )
    if not rows:
        return None
    row = rows[0]
    return YouTubeVideoLease(
        source=source,
        video_id=YouTubeVideoId(str(row["video_id"])),
        owner_token=str(row["owner_token"]),
        lease_expires_at=datetime.fromisoformat(str(row["lease_expires_at"])),
        first_poll_version_id=TypeAdapter(Sha256).validate_python(
            row["first_poll_version_id"], strict=True
        ),
        metadata_version_id=TypeAdapter(Sha256 | None).validate_python(
            row["metadata_version_id"], strict=True
        ),
        deterministic_failure_fingerprint=TypeAdapter(Sha256 | None).validate_python(
            row["deterministic_failure_fingerprint"], strict=True
        ),
        unchanged_deterministic_failures=int(row["unchanged_deterministic_failures"]),
    )


def renew_youtube_lease(lease: YouTubeVideoLease, now: datetime) -> YouTubeVideoLease:
    current = now.astimezone(UTC)
    expires = current + timedelta(seconds=YOUTUBE_LEASE_SECONDS)
    rows = catalog_mutation(
        "UPDATE youtube_videos SET lease_expires_at = %s WHERE source_id = %s AND video_id = %s "
        "AND state = 'running' AND owner_token = %s AND lease_expires_at > %s RETURNING video_id",
        [
            expires.isoformat(),
            lease.source.source_id,
            lease.video_id,
            lease.owner_token,
            current.isoformat(),
        ],
    )
    if not rows:
        raise YouTubeLeaseLostError("YouTube video lease was lost")
    return lease.model_copy(update={"lease_expires_at": expires})


def defer_youtube_video(
    lease: YouTubeVideoLease, error: Exception, now: datetime, *, minutes: int = 5
) -> None:
    retry_at = now.astimezone(UTC) + timedelta(minutes=min(max(minutes, 1), 360))
    _fenced_update(
        "UPDATE youtube_videos SET state = 'deferred', owner_token = NULL, lease_expires_at = NULL, retry_at = %s, last_error = %s "
        "WHERE source_id = %s AND video_id = %s AND state = 'running' AND owner_token = %s RETURNING video_id",
        [
            retry_at.isoformat(),
            _error_text(error),
            lease.source.source_id,
            lease.video_id,
            lease.owner_token,
        ],
    )


def reject_youtube_video(
    lease: YouTubeVideoLease, error: Exception, now: datetime
) -> YouTubeVideoState:
    fingerprint = sha256(_error_text(error).encode())
    unchanged = (
        lease.unchanged_deterministic_failures + 1
        if lease.deterministic_failure_fingerprint == fingerprint
        else 1
    )
    state = (
        YouTubeVideoState.QUARANTINED
        if unchanged >= MAX_UNCHANGED_DETERMINISTIC_FAILURES
        else YouTubeVideoState.DEFERRED
    )
    retry_at = (
        None
        if state == YouTubeVideoState.QUARANTINED
        else now.astimezone(UTC) + timedelta(minutes=min(15 * 2 ** (unchanged - 1), 360))
    )
    _fenced_update(
        "UPDATE youtube_videos SET state = %s, owner_token = NULL, lease_expires_at = NULL, retry_at = %s, deterministic_failure_fingerprint = %s, unchanged_deterministic_failures = %s, last_error = %s, quarantine_generation = quarantine_generation + %s "
        "WHERE source_id = %s AND video_id = %s AND state = 'running' AND owner_token = %s RETURNING video_id",
        [
            state,
            retry_at.isoformat() if retry_at else None,
            fingerprint,
            unchanged,
            _error_text(error),
            1 if state == YouTubeVideoState.QUARANTINED else 0,
            lease.source.source_id,
            lease.video_id,
            lease.owner_token,
        ],
    )
    return state


def publish_youtube_metadata(
    metadata: YouTubeVideoMetadata,
    implementation_ref: str,
    lease: YouTubeVideoLease,
) -> ArtifactFile:
    if metadata.source != lease.source or metadata.video_id != lease.video_id:
        raise YouTubeEvidenceError("YouTube metadata does not match its lease")
    content = canonical_json(metadata.model_dump(mode="json"))
    source_id = metadata.source.source_id
    value = artifact_file(
        artifact_id=f"news:youtube-video:{source_id}:{metadata.video_id}",
        artifact_kind="youtube_video_metadata",
        title=metadata.title,
        content=content,
        r2_key=f"news/youtube/{source_id}/videos/{metadata.video_id}/{sha256(content)}.json",
        media_type="application/json",
    )
    run_id = sha256(
        canonical_json({"metadata": value.version_id, "implementation_ref": implementation_ref})
    )
    timestamp = datetime.now(UTC).isoformat()
    statements = derived_artifact_run_statements(
        value,
        run_id,
        "news.youtube.fetch_metadata",
        implementation_ref,
        ((metadata.feed_snapshot_version_id, "feed_snapshot", None),),
        timestamp,
        executor_kind="youtube-data-api",
    )
    publish_immutable_r2_objects(((value.r2_key, value.content),))
    catalog_batch(statements)
    _fenced_update(
        "UPDATE youtube_videos SET metadata_version_id = %s "
        "WHERE source_id = %s AND video_id = %s AND state = 'running' "
        "AND owner_token = %s AND lease_expires_at > CURRENT_TIMESTAMP "
        "RETURNING video_id",
        [value.version_id, source_id, metadata.video_id, lease.owner_token],
    )
    return value


def read_youtube_metadata(
    lease: YouTubeVideoLease,
) -> tuple[YouTubeVideoMetadata, ArtifactFile]:
    if lease.metadata_version_id is None:
        raise ValueError("YouTube lease has no stored metadata version")
    rows = catalog_query(
        "SELECT artifact.id AS artifact_id, artifact.kind AS artifact_kind, artifact.title, "
        "file.content_digest, file.r2_key, file.media_type "
        "FROM artifact_versions version "
        "JOIN artifacts artifact ON artifact.id = version.artifact_id "
        "JOIN artifact_files file ON file.artifact_version_id = version.id "
        "WHERE version.id = %s",
        [lease.metadata_version_id],
    )
    if len(rows) != 1:
        raise YouTubeEvidenceError("Stored YouTube metadata artifact is unavailable")
    row = rows[0]
    content = read_verified_r2_object(str(row["r2_key"]), str(row["content_digest"]))
    value = artifact_file(
        artifact_id=str(row["artifact_id"]),
        artifact_kind=str(row["artifact_kind"]),
        title=str(row["title"]),
        content=content,
        r2_key=str(row["r2_key"]),
        media_type=str(row["media_type"]),
    )
    if (
        value.version_id != lease.metadata_version_id
        or value.artifact_kind != "youtube_video_metadata"
    ):
        raise YouTubeEvidenceError("Stored YouTube metadata artifact identity is invalid")
    try:
        metadata = YouTubeVideoMetadata.model_validate_json(content, strict=True)
    except ValidationError as error:
        raise YouTubeEvidenceError("Stored YouTube metadata is invalid") from error
    if metadata.source != lease.source or metadata.video_id != lease.video_id:
        raise YouTubeEvidenceError("Stored YouTube metadata belongs to another source or video")
    return metadata, value


def read_accepted_clips(
    metadata: YouTubeVideoMetadata, metadata_version_id: Sha256, model: str, request_digest: Sha256
) -> tuple[AcceptedClipExtraction, ...]:
    rows = catalog_query(
        "SELECT file.r2_key, file.content_digest FROM youtube_clip_analyses clip "
        "JOIN artifact_files file ON file.artifact_version_id = clip.artifact_version_id "
        "WHERE clip.source_id = %s AND clip.video_id = %s AND clip.metadata_version_id = %s "
        "AND clip.model = %s AND clip.request_digest = %s ORDER BY clip.start_second",
        [metadata.source.source_id, metadata.video_id, metadata_version_id, model, request_digest],
    )
    try:
        values = tuple(
            AcceptedClipExtraction.model_validate_json(
                read_verified_r2_object(str(row["r2_key"]), str(row["content_digest"])), strict=True
            )
            for row in rows
        )
    except ValidationError as error:
        raise YouTubeEvidenceError("Stored YouTube clip evidence is invalid") from error
    if any(value.source != metadata.source for value in values):
        raise YouTubeEvidenceError("Stored YouTube clip belongs to another source")
    return values


def publish_receipt(receipt: YouTubeModelReceipt) -> YouTubeModelReceipt:
    """Store the typed response envelope before PostgreSQL; a failed catalog write can leave an immutable R2 orphan."""
    content = canonical_json(receipt.model_dump(mode="json"))
    value = artifact_file(
        artifact_id=f"news:youtube-model-receipt:{receipt.operation_key}:{receipt.request_id}:{receipt.attempt_index}",
        artifact_kind="youtube_model_receipt",
        title=f"YouTube model receipt {receipt.operation_key} {receipt.attempt_index}",
        content=content,
        r2_key=f"news/youtube/receipts/{receipt.request_id}/{receipt.attempt_index}/{sha256(content)}.json",
        media_type="application/json",
    )
    timestamp = receipt.received_at.astimezone(UTC).isoformat()
    statements = artifact_statements(
        value, timestamp, produced_by_run_id=None, authority_class="source"
    )
    statements.append(
        advance_artifact_current_version_statement(value.artifact_id, value.version_id)
    )
    statements.append(
        (
            "INSERT INTO youtube_model_receipts "
            "(request_id, operation_key, attempt_index, artifact_version_id, requested_model, "
            "http_status, status, received_at) VALUES (%s, %s, %s, %s, %s, %s, 'unhandled', %s) "
            "ON CONFLICT DO NOTHING",
            [
                receipt.request_id,
                receipt.operation_key,
                receipt.attempt_index,
                value.version_id,
                receipt.response.requested_model,
                receipt.response.status_code,
                timestamp,
            ],
        )
    )
    publish_immutable_r2_objects(((value.r2_key, value.content),))
    catalog_batch(statements, retry_transient_errors=True)
    return receipt


def read_unhandled_receipt(request_id: Sha256, operation_key: str) -> YouTubeModelReceipt | None:
    rows = catalog_query(
        "SELECT file.r2_key, file.content_digest FROM youtube_model_receipts receipt "
        "JOIN artifact_files file ON file.artifact_version_id = receipt.artifact_version_id "
        "WHERE receipt.request_id = %s AND receipt.operation_key = %s AND receipt.status = 'unhandled' "
        "ORDER BY receipt.attempt_index LIMIT 1",
        [request_id, operation_key],
    )
    if not rows:
        return None
    return YouTubeModelReceipt.model_validate_json(
        read_verified_r2_object(str(rows[0]["r2_key"]), str(rows[0]["content_digest"])), strict=True
    )


def next_receipt_attempt_index(request_id: Sha256, operation_key: str) -> int:
    rows = catalog_query(
        "SELECT COALESCE(MAX(attempt_index), -1) + 1 AS attempt_index FROM youtube_model_receipts "
        "WHERE request_id = %s AND operation_key = %s",
        [request_id, operation_key],
    )
    return int(rows[0]["attempt_index"])


def reject_receipt(
    receipt: YouTubeModelReceipt,
    attempt: ModelAttempt | None,
    handled_at: datetime,
    error: str,
) -> None:
    if attempt is not None:
        _validate_receipt_attempt(receipt, attempt, receipt.operation_key, "rejected")
    catalog_batch(
        _receipt_handled_statements(receipt, attempt, "rejected", handled_at, error=error),
        retry_transient_errors=True,
    )


def publish_accepted_clip(
    metadata: YouTubeVideoMetadata,
    accepted: AcceptedClipExtraction,
    metadata_file: ArtifactFile,
    receipt: YouTubeModelReceipt,
    attempt: ModelAttempt,
    implementation_ref: str,
) -> AcceptedClipExtraction:
    _validate_accepted_clip(metadata, accepted, metadata_file, receipt, attempt)
    content = canonical_json(accepted.model_dump(mode="json"))
    clip_range = accepted.clip_range
    request_id = receipt.request_id
    value = artifact_file(
        artifact_id=(
            f"news:youtube-clip:{accepted.source.source_id}:" f"{request_id}:{sha256(content)}"
        ),
        artifact_kind="youtube_clip_analysis",
        title=f"{metadata_file.title} [{clip_range.start_second}s]",
        content=content,
        r2_key=(
            f"news/youtube/{accepted.source.source_id}/clips/{accepted.video_id}/"
            f"{clip_range.start_second}/{sha256(content)}.json"
        ),
        media_type="application/json",
    )
    timestamp = datetime.now(UTC)
    run_id = sha256(f"{request_id}\0{value.version_id}\0{implementation_ref}".encode())
    statements = _receipt_handled_statements(receipt, attempt, "accepted", timestamp)
    statements.extend(
        derived_artifact_run_statements(
            value,
            run_id,
            "news.youtube.extract_clip",
            implementation_ref,
            ((metadata_file.version_id, "video_metadata", clip_range.model_dump(mode="json")),),
            timestamp.isoformat(),
            executor_kind="gemini",
            model=accepted.model,
        )
    )
    statements.append(
        (
            "INSERT INTO youtube_clip_analyses "
            "(artifact_version_id, source_id, video_id, metadata_version_id, start_second, duration_seconds, "
            "model, request_digest, model_attempt_id, accepted_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING",
            [
                value.version_id,
                accepted.source.source_id,
                accepted.video_id,
                accepted.metadata_version_id,
                clip_range.start_second,
                clip_range.duration_seconds,
                accepted.model,
                accepted.request_digest,
                attempt.attempt_id,
                timestamp.isoformat(),
            ],
        )
    )
    publish_immutable_r2_objects(((value.r2_key, value.content),))
    catalog_batch(statements, retry_transient_errors=True)
    return read_accepted_clip_checkpoint(
        accepted.source.source_id,
        accepted.video_id,
        accepted.metadata_version_id,
        clip_range.start_second,
        clip_range.duration_seconds,
        accepted.model,
        accepted.request_digest,
    )


def read_accepted_clip_checkpoint(
    source_id: str,
    video_id: str,
    metadata_version_id: Sha256,
    start_second: int,
    duration_seconds: int,
    model: str,
    request_digest: Sha256,
) -> AcceptedClipExtraction:
    rows = catalog_query(
        "SELECT file.r2_key, file.content_digest FROM youtube_clip_analyses clip "
        "JOIN artifact_files file ON file.artifact_version_id = clip.artifact_version_id "
        "WHERE clip.source_id = %s AND clip.video_id = %s AND clip.metadata_version_id = %s AND clip.start_second = %s "
        "AND clip.duration_seconds = %s AND clip.model = %s AND clip.request_digest = %s",
        [
            source_id,
            video_id,
            metadata_version_id,
            start_second,
            duration_seconds,
            model,
            request_digest,
        ],
    )
    if len(rows) != 1:
        raise ValueError("Accepted YouTube clip checkpoint is unavailable")
    return AcceptedClipExtraction.model_validate_json(
        read_verified_r2_object(str(rows[0]["r2_key"]), str(rows[0]["content_digest"])),
        strict=True,
    )


def read_accepted_clip_references(
    metadata: YouTubeVideoMetadata,
    metadata_version_id: Sha256,
    model: str,
    request_digest: Sha256,
) -> tuple[AcceptedClipReference, ...]:
    rows = catalog_query(
        "SELECT artifact_version_id, start_second, duration_seconds FROM youtube_clip_analyses "
        "WHERE source_id = %s AND video_id = %s AND metadata_version_id = %s AND model = %s "
        "AND request_digest = %s ORDER BY start_second",
        [
            metadata.source.source_id,
            metadata.video_id,
            metadata_version_id,
            model,
            request_digest,
        ],
    )
    return tuple(AcceptedClipReference.model_validate(row, strict=True) for row in rows)


def read_youtube_merge(
    request_id: Sha256, source: YouTubeSource, title: str
) -> tuple[YouTubeMergedArticle, ArtifactFile] | None:
    rows = catalog_query(
        "SELECT file.r2_key, file.content_digest FROM youtube_merge_analyses merge_analysis "
        "JOIN artifact_files file ON file.artifact_version_id = merge_analysis.artifact_version_id "
        "WHERE merge_analysis.request_id = %s AND merge_analysis.source_id = %s",
        [request_id, source.source_id],
    )
    if not rows:
        return None
    content = read_verified_r2_object(str(rows[0]["r2_key"]), str(rows[0]["content_digest"]))
    try:
        payload = TypeAdapter(dict[str, object]).validate_json(content, strict=True)
        article = YouTubeMergedArticle.model_validate(payload["article"], strict=True)
    except (KeyError, ValidationError, ValueError) as error:
        raise YouTubeMergeError("Stored YouTube merge is invalid") from error
    return article, artifact_file(
        artifact_id=f"news:youtube-merge:{source.source_id}:{request_id}",
        artifact_kind="youtube_video_merge",
        title=title,
        content=content,
        r2_key=str(rows[0]["r2_key"]),
        media_type="application/json",
    )


def publish_accepted_merge(
    metadata: YouTubeVideoMetadata,
    clips: tuple[AcceptedClipExtraction, ...],
    clip_version_ids: tuple[Sha256, ...],
    article: YouTubeMergedArticle,
    request_id: Sha256,
    request_digest: Sha256,
    response: GeminiResponse,
    receipt: YouTubeModelReceipt,
    attempt: ModelAttempt,
    implementation_ref: str,
) -> tuple[YouTubeMergedArticle, ArtifactFile]:
    expected_request_id = sha256(
        canonical_json(
            {
                "source_id": metadata.source.source_id,
                "clip_version_ids": clip_version_ids,
                "request_digest": request_digest,
                "model": response.model,
            }
        )
    )
    if request_id != expected_request_id or receipt.request_id != request_id:
        raise ValueError("YouTube merge request identity does not match its accepted output")
    _validate_receipt_attempt(receipt, attempt, "news.youtube.merge_video", "accepted")
    _validate_gemini_response(receipt, response, attempt)
    if attempt.response_id != response.response_id or attempt.model != response.model:
        raise ValueError("YouTube merge response does not match its model attempt")
    input_payload = _merge_input_payload(clips)
    content = canonical_json(
        {
            "article": article.model_dump(mode="json"),
            "clip_version_ids": clip_version_ids,
            "merge_inputs": input_payload,
            "model": response.model,
            "request_digest": request_digest,
            "provider_response": response.payload,
            "source_replay_limit": LIVE_MEDIA_REPLAY_LIMIT,
        }
    )
    value = artifact_file(
        artifact_id=f"news:youtube-merge:{metadata.source.source_id}:{request_id}",
        artifact_kind="youtube_video_merge",
        title=metadata.title,
        content=content,
        r2_key=(
            f"news/youtube/{metadata.source.source_id}/merges/{metadata.video_id}/"
            f"{sha256(content)}.json"
        ),
        media_type="application/json",
    )
    timestamp = datetime.now(UTC)
    run_id = sha256(f"{request_id}\0{value.version_id}\0{implementation_ref}".encode())
    inputs = tuple(
        (version_id, "accepted_clip", clip.clip_range.model_dump(mode="json"))
        for version_id, clip in zip(clip_version_ids, clips, strict=True)
    )
    statements = _receipt_handled_statements(receipt, attempt, "accepted", timestamp)
    statements.extend(
        derived_artifact_run_statements(
            value,
            run_id,
            "news.youtube.merge_video",
            implementation_ref,
            inputs,
            timestamp.isoformat(),
            executor_kind="gemini",
            model=response.model,
        )
    )
    statements.append(
        (
            "INSERT INTO youtube_merge_analyses "
            "(artifact_version_id, source_id, video_id, request_id, model_attempt_id, accepted_at) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                value.version_id,
                metadata.source.source_id,
                metadata.video_id,
                request_id,
                attempt.attempt_id,
                timestamp.isoformat(),
            ],
        )
    )
    publish_immutable_r2_objects(((value.r2_key, value.content),))
    catalog_batch(statements, retry_transient_errors=True)
    return article, value


def _merge_input_payload(
    clips: tuple[AcceptedClipExtraction, ...],
) -> list[dict[str, object]]:
    return [
        {
            "clip_range": clip.clip_range.model_dump(mode="json"),
            "title": clip.extraction.title,
            "standfirst": clip.extraction.standfirst,
            "narrative": clip.extraction.narrative,
            "evidence": [value.model_dump(mode="json") for value in clip.extraction.evidence],
            "source_stated_uncertainties": clip.extraction.source_stated_uncertainties,
        }
        for clip in clips
    ]


def publish_youtube_candidate(
    metadata: YouTubeVideoMetadata,
    metadata_file: ArtifactFile,
    clips: tuple[AcceptedClipExtraction, ...],
    merge_file: ArtifactFile,
    merged: YouTubeMergedArticle,
    implementation_ref: str,
    lease: YouTubeVideoLease,
) -> Sha256:
    source = metadata.source
    clip_version_ids, receipt_version_ids, cost = _candidate_accounting(
        metadata, metadata_file, clips, merge_file
    )
    evidence = tuple(
        CandidateEvidence(
            clip_version_id=version_id,
            kind=item.kind,
            text=item.text,
            attribution=item.attribution,
            start_second=clip.clip_range.start_second + item.start_second,
            duration_seconds=item.duration_seconds,
        )
        for version_id, clip in zip(clip_version_ids, clips, strict=True)
        for item in clip.extraction.evidence
    )
    candidate = YouTubeCandidate(
        source=source,
        video_id=metadata.video_id,
        video_url=metadata.url,
        video_title=metadata.title,
        published_at=metadata.published_at,
        feed_snapshot_version_id=metadata.feed_snapshot_version_id,
        metadata_version_id=metadata_file.version_id,
        clip_version_ids=clip_version_ids,
        merge_version_id=merge_file.version_id,
        receipt_version_ids=receipt_version_ids,
        article=merged,
        evidence=evidence,
        uncertainties=tuple(
            dict.fromkeys(
                value for clip in clips for value in clip.extraction.source_stated_uncertainties
            )
        ),
        omissions=merged.omitted_clips,
        implementation_ref=implementation_ref,
        model=YOUTUBE_MODEL,
        cost=cost,
    )
    content = canonical_json(candidate.model_dump(mode="json"))
    value = artifact_file(
        artifact_id=f"news:youtube-candidate:{source.source_id}:{metadata.video_id}",
        artifact_kind="youtube_article_candidate",
        title=merged.title,
        content=content,
        r2_key=f"news/youtube/{source.source_id}/candidates/{metadata.video_id}/{sha256(content)}.json",
        media_type="application/json",
    )
    timestamp = datetime.now(UTC).isoformat()
    inputs = (
        (metadata_file.version_id, "video_metadata", None),
        *((version_id, "accepted_clip", None) for version_id in clip_version_ids),
        (merge_file.version_id, "video_merge", None),
        *((version_id, "model_receipt", None) for version_id in receipt_version_ids),
    )
    run_id = sha256(
        canonical_json({"candidate": value.version_id, "implementation_ref": implementation_ref})
    )
    statements = derived_artifact_run_statements(
        value,
        run_id,
        "news.youtube.create_candidate",
        implementation_ref,
        inputs,
        timestamp,
        executor_kind="python",
    )
    statements.extend(
        [
            (
                "INSERT INTO youtube_candidates "
                "(candidate_version_id, source_id, video_id, policy, metadata_version_id, "
                "merge_version_id, owner_token, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                [
                    value.version_id,
                    source.source_id,
                    metadata.video_id,
                    "automatic",
                    metadata_file.version_id,
                    merge_file.version_id,
                    lease.owner_token,
                    timestamp,
                ],
            ),
            (
                "UPDATE youtube_videos SET state = 'candidate', candidate_version_id = %s, owner_token = NULL, lease_expires_at = NULL, retry_at = NULL, last_error = NULL WHERE source_id = %s AND video_id = %s AND state = 'running' AND owner_token = %s",
                [value.version_id, source.source_id, metadata.video_id, lease.owner_token],
            ),
        ]
    )
    publish_immutable_r2_objects(((value.r2_key, value.content),))
    try:
        catalog_batch(statements)
    except ResearchCatalogError as error:
        if _exception_contains(error, "youtube candidate lease was lost"):
            raise YouTubeLeaseLostError("YouTube video lease was lost") from error
        raise
    _record_decision(value.version_id, source)
    return value.version_id


def _exception_contains(error: BaseException, expected: str) -> bool:
    current: BaseException | None = error
    while current is not None:
        if expected in str(current).casefold():
            return True
        current = current.__cause__
    return False


def _candidate_accounting(
    metadata: YouTubeVideoMetadata,
    metadata_file: ArtifactFile,
    clips: tuple[AcceptedClipExtraction, ...],
    merge_file: ArtifactFile,
) -> tuple[tuple[Sha256, ...], tuple[Sha256, ...], MeasuredCost | UnmeasuredCost]:
    if not clips:
        raise YouTubeEvidenceError("YouTube candidate requires accepted clips")
    generation = {(clip.model, clip.request_digest) for clip in clips}
    if (
        len(generation) != 1
        or any(
            clip.source != metadata.source or clip.video_id != metadata.video_id for clip in clips
        )
        or any(clip.metadata_version_id != metadata_file.version_id for clip in clips)
    ):
        raise YouTubeEvidenceError("YouTube candidate clips do not share one analysis generation")
    clip_model, clip_request_digest = next(iter(generation))
    clip_rows = catalog_query(
        "SELECT clip.artifact_version_id, clip.model_attempt_id, receipt.artifact_version_id AS receipt_version_id, "
        "attempt.cost_usd FROM youtube_clip_analyses clip "
        "JOIN news_model_attempts attempt ON attempt.attempt_id = clip.model_attempt_id "
        "JOIN youtube_model_receipts receipt ON receipt.handled_attempt_id = clip.model_attempt_id "
        "WHERE clip.source_id = %s AND clip.video_id = %s AND clip.metadata_version_id = %s "
        "AND clip.model = %s AND clip.request_digest = %s ORDER BY clip.start_second",
        [
            metadata.source.source_id,
            metadata.video_id,
            metadata_file.version_id,
            clip_model,
            clip_request_digest,
        ],
    )
    merge_rows = catalog_query(
        "SELECT merge.model_attempt_id, receipt.artifact_version_id AS receipt_version_id, attempt.cost_usd "
        "FROM youtube_merge_analyses merge "
        "JOIN news_model_attempts attempt ON attempt.attempt_id = merge.model_attempt_id "
        "JOIN youtube_model_receipts receipt ON receipt.handled_attempt_id = merge.model_attempt_id "
        "WHERE merge.artifact_version_id = %s AND merge.source_id = %s AND merge.video_id = %s",
        [merge_file.version_id, metadata.source.source_id, metadata.video_id],
    )
    if len(clip_rows) != len(clips) or len(merge_rows) != 1:
        raise YouTubeEvidenceError("Candidate accounting is incomplete")
    clip_version_ids = tuple(
        TypeAdapter(Sha256).validate_python(row["artifact_version_id"], strict=True)
        for row in clip_rows
    )
    accounting_rows = catalog_query(
        "SELECT receipt.artifact_version_id AS receipt_version_id, attempt.cost_usd "
        "FROM youtube_model_receipts receipt "
        "LEFT JOIN news_model_attempts attempt ON attempt.attempt_id = receipt.handled_attempt_id "
        "WHERE EXISTS (SELECT 1 FROM youtube_clip_analyses clip "
        "JOIN news_model_attempts accepted ON accepted.attempt_id = clip.model_attempt_id "
        "WHERE clip.source_id = %s AND clip.video_id = %s AND clip.metadata_version_id = %s "
        "AND clip.model = %s AND clip.request_digest = %s "
        "AND accepted.request_id = receipt.request_id "
        "AND receipt.operation_key = 'news.youtube.extract_clip') "
        "OR EXISTS (SELECT 1 FROM youtube_merge_analyses merge "
        "JOIN news_model_attempts accepted ON accepted.attempt_id = merge.model_attempt_id "
        "WHERE merge.artifact_version_id = %s AND merge.source_id = %s AND merge.video_id = %s "
        "AND accepted.request_id = receipt.request_id "
        "AND receipt.operation_key = 'news.youtube.merge_video') "
        "ORDER BY receipt.operation_key, receipt.request_id, receipt.attempt_index",
        [
            metadata.source.source_id,
            metadata.video_id,
            metadata_file.version_id,
            clip_model,
            clip_request_digest,
            merge_file.version_id,
            metadata.source.source_id,
            metadata.video_id,
        ],
    )
    if not accounting_rows:
        raise YouTubeEvidenceError("Candidate accounting has no model receipts")
    receipt_version_ids = tuple(
        TypeAdapter(Sha256).validate_python(row["receipt_version_id"], strict=True)
        for row in accounting_rows
    )
    costs = [row["cost_usd"] for row in accounting_rows]
    cost = (
        MeasuredCost(usd=round(sum(float(value) for value in costs), 8))
        if all(value is not None for value in costs)
        else UnmeasuredCost(reason="One or more model attempts have no measured cost")
    )
    return clip_version_ids, receipt_version_ids, cost


def read_youtube_candidate(candidate_version_id: Sha256) -> YouTubeCandidate:
    rows = catalog_query(
        "SELECT file.r2_key, file.content_digest "
        "FROM youtube_candidates candidate JOIN artifact_files file ON file.artifact_version_id = candidate.candidate_version_id "
        "WHERE candidate.candidate_version_id = %s",
        [candidate_version_id],
    )
    if len(rows) != 1:
        raise ValueError("YouTube candidate is unavailable")
    row = rows[0]
    return YouTubeCandidate.model_validate_json(
        read_verified_r2_object(str(row["r2_key"]), str(row["content_digest"])), strict=True
    )


def _record_decision(
    candidate_version_id: Sha256,
    source: YouTubeSource,
) -> SystemApproval:
    existing = _read_decision(candidate_version_id)
    if existing is not None:
        return existing
    decided_at = datetime.now(UTC)
    decision = SystemApproval(
        candidate_version_id=candidate_version_id,
        source_id=source.source_id,
        decided_at=decided_at,
    )
    content = canonical_json(decision.model_dump(mode="json"))
    value = artifact_file(
        artifact_id=f"news:youtube-decision:{candidate_version_id}",
        artifact_kind="youtube_candidate_decision",
        title=f"{source.display_name} candidate decision",
        content=content,
        r2_key=f"news/youtube/{source.source_id}/decisions/{candidate_version_id}/{sha256(content)}.json",
        media_type="application/json",
    )
    timestamp = decided_at.isoformat()
    statements = artifact_statements(
        value, timestamp, produced_by_run_id=None, authority_class="source"
    )
    statements.append(
        advance_artifact_current_version_statement(value.artifact_id, value.version_id)
    )
    statements.append(
        (
            "INSERT INTO youtube_candidate_decisions "
            "(artifact_version_id, candidate_version_id, source_id, kind, actor, note, decided_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [
                value.version_id,
                candidate_version_id,
                source.source_id,
                decision.kind,
                decision.actor,
                None,
                timestamp,
            ],
        )
    )
    publish_immutable_r2_objects(((value.r2_key, value.content),))
    try:
        catalog_batch(statements)
    except ResearchCatalogError:
        persisted = _read_decision(candidate_version_id)
        if persisted is None:
            raise
        return persisted
    return decision


def _read_decision(candidate_version_id: Sha256) -> SystemApproval | None:
    rows = catalog_query(
        "SELECT kind, source_id, actor, decided_at FROM youtube_candidate_decisions WHERE candidate_version_id = %s",
        [candidate_version_id],
    )
    return _decision_from_row(candidate_version_id, rows[0]) if rows else None


def _decision_from_row(candidate_version_id: Sha256, row: dict[str, object]) -> SystemApproval:
    payload: dict[str, object] = {
        "kind": str(row["kind"]),
        "candidate_version_id": candidate_version_id,
        "source_id": str(row["source_id"]),
        "actor": str(row["actor"]),
        "decided_at": datetime.fromisoformat(str(row["decided_at"])),
    }
    return SystemApproval.model_validate(payload, strict=True)


def _reconcile_automatic_approval() -> None:
    rows = catalog_query(
        "SELECT candidate.candidate_version_id FROM youtube_candidates candidate "
        "LEFT JOIN youtube_candidate_decisions decision ON decision.candidate_version_id = candidate.candidate_version_id "
        "WHERE decision.candidate_version_id IS NULL AND candidate.policy = 'automatic' "
        "ORDER BY candidate.created_at LIMIT 1"
    )
    if not rows:
        return
    candidate_version_id = TypeAdapter(Sha256).validate_python(
        rows[0]["candidate_version_id"], strict=True
    )
    candidate = read_youtube_candidate(candidate_version_id)
    _record_decision(candidate_version_id, candidate.source)


def _read_publication_without_relevance() -> YouTubePublicationResult | None:
    rows = catalog_query(
        "SELECT publication.source_id, publication.video_id, publication.candidate_version_id, "
        "publication.article_version_id, article.bucharest_day "
        "FROM youtube_publications publication "
        "JOIN news_article_versions article ON article.artifact_version_id = publication.article_version_id "
        "WHERE NOT EXISTS (SELECT 1 FROM run_inputs input "
        "JOIN runs run ON run.id = input.run_id "
        "JOIN run_outputs output ON output.run_id = run.id "
        "JOIN news_relevance_versions relevance ON relevance.artifact_version_id = output.artifact_version_id "
        "WHERE input.artifact_version_id = publication.article_version_id "
        "AND input.role = 'article' AND run.operation_key IN ('news.relevance', 'news.relevance.v3')) "
        "ORDER BY article.bucharest_day, publication.video_id LIMIT 1"
    )
    if not rows:
        return None
    row = rows[0]
    candidate_version_id = TypeAdapter(Sha256).validate_python(
        row["candidate_version_id"], strict=True
    )
    candidate = read_youtube_candidate(candidate_version_id)
    return YouTubePublicationResult(
        source_id=candidate.source.source_id,
        video_id=YouTubeVideoId(str(row["video_id"])),
        candidate_version_id=candidate_version_id,
        article_version_id=TypeAdapter(Sha256).validate_python(
            row["article_version_id"], strict=True
        ),
        bucharest_day=date.fromisoformat(str(row["bucharest_day"])),
        accepted_clip_count=len(candidate.clip_version_ids),
    )


def publish_next_approved_candidate(implementation_ref: str) -> YouTubePublicationResult | None:
    _reconcile_automatic_approval()
    pending = _read_publication_without_relevance()
    if pending is not None:
        return pending
    rows = catalog_query(
        "SELECT candidate.candidate_version_id, decision.artifact_version_id AS decision_version_id "
        "FROM youtube_candidates candidate "
        "JOIN youtube_candidate_decisions decision ON decision.candidate_version_id = candidate.candidate_version_id "
        "LEFT JOIN youtube_publications publication ON publication.candidate_version_id = candidate.candidate_version_id "
        "WHERE decision.kind = 'automatic_approved' AND publication.candidate_version_id IS NULL "
        "ORDER BY decision.decided_at LIMIT 1"
    )
    if not rows:
        return None
    candidate_version_id = TypeAdapter(Sha256).validate_python(
        rows[0]["candidate_version_id"], strict=True
    )
    decision_version_id = TypeAdapter(Sha256).validate_python(
        rows[0]["decision_version_id"], strict=True
    )
    candidate = read_youtube_candidate(candidate_version_id)
    source = candidate.source
    body = f"{candidate.article.standfirst}\n\n{candidate.article.narrative}"
    article_id = sha256(f"{source.source_id}\0{candidate.video_id}".encode())
    material_digest = sha256(canonical_json({"title": candidate.article.title, "body": body}))
    article = ExtractedArticle(
        article_id=article_id,
        outlet_id=source.outlet_id,
        canonical_url=candidate.video_url,
        title=candidate.article.title,
        body=body,
        author=source.display_name,
        published_at=candidate.published_at,
        source_updated_at=None,
        bucharest_day=candidate.published_at.astimezone(BUCHAREST).date(),
        material_digest=material_digest,
        extraction_digest=candidate.merge_version_id,
    )
    content = canonical_json(article.model_dump(mode="json"))
    value = artifact_file(
        artifact_id=f"news:article:{article_id}",
        artifact_kind="news_article",
        title=article.title,
        content=content,
        r2_key=f"news/articles/{article_id}/{sha256(content)}.json",
        media_type="application/json",
    )
    timestamp = datetime.now(UTC).isoformat()
    run_id = sha256(
        canonical_json(
            {
                "article_version_id": value.version_id,
                "candidate_version_id": candidate_version_id,
                "implementation_ref": implementation_ref,
            }
        )
    )
    statements = derived_artifact_run_statements(
        value,
        run_id,
        "news.youtube.publish_article",
        implementation_ref,
        (
            (candidate_version_id, "approved_candidate", None),
            (decision_version_id, "approval_decision", None),
        ),
        timestamp,
        executor_kind="python",
        authority_class="derived",
    )
    statements.extend(
        [
            (
                "INSERT INTO youtube_publications VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    candidate_version_id,
                    source.source_id,
                    candidate.video_id,
                    value.version_id,
                    timestamp,
                ],
            ),
            (
                "INSERT INTO news_article_aliases VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [f"url:{candidate.video_url}", value.artifact_id, "canonical_url", timestamp],
            ),
            (
                "INSERT INTO news_article_versions "
                "(artifact_version_id, article_artifact_id, outlet_id, canonical_url, "
                "published_at, source_updated_at, bucharest_day, material_digest, "
                "extraction_digest, feed_snapshot_version_id, page_capture_version_id, captured_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                [
                    value.version_id,
                    value.artifact_id,
                    source.outlet_id,
                    str(candidate.video_url),
                    candidate.published_at.astimezone(UTC).isoformat(),
                    None,
                    article.bucharest_day.isoformat(),
                    article.material_digest,
                    article.extraction_digest,
                    candidate.feed_snapshot_version_id,
                    None,
                    timestamp,
                ],
            ),
        ]
    )
    publish_immutable_r2_objects(((value.r2_key, value.content),))
    catalog_batch(statements)
    return YouTubePublicationResult(
        source_id=source.source_id,
        video_id=candidate.video_id,
        candidate_version_id=candidate_version_id,
        article_version_id=value.version_id,
        bucharest_day=article.bucharest_day,
        accepted_clip_count=len(candidate.clip_version_ids),
    )


def derived_artifact_run_statements(
    value: ArtifactFile,
    run_id: Sha256,
    operation_key: str,
    implementation_ref: str,
    inputs: tuple[tuple[str, str, object | None], ...],
    timestamp: str,
    *,
    executor_kind: str,
    authority_class: str = "derived",
    model: str | None = None,
) -> list[tuple[str, list[object]]]:
    statements: list[tuple[str, list[object]]] = [
        (
            "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING",
            [
                run_id,
                operation_key,
                executor_kind,
                implementation_ref,
                canonical_json({"model": model}).decode(),
                "chartly",
                "publishing",
                f"{operation_key}:{run_id}",
                None,
                timestamp,
                None,
            ],
        )
    ]
    for position, (version_id, role, locator) in enumerate(inputs):
        rows = catalog_query(
            "SELECT content_digest FROM artifact_versions WHERE id = %s", [version_id]
        )
        if len(rows) != 1:
            raise ValueError(f"YouTube run input is unavailable: {version_id}")
        statements.append(
            (
                "INSERT INTO run_inputs VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                [
                    run_id,
                    position,
                    version_id,
                    role,
                    canonical_json(locator).decode() if locator is not None else None,
                    str(rows[0]["content_digest"]),
                    "whole_file",
                    None,
                ],
            )
        )
    statements.extend(
        artifact_statements(
            value, timestamp, produced_by_run_id=run_id, authority_class=authority_class
        )
    )
    statements.extend(
        [
            (
                "INSERT INTO run_outputs VALUES (%s, 0, %s, 'output') ON CONFLICT DO NOTHING",
                [run_id, value.version_id],
            ),
            advance_artifact_current_version_from_run_statement(
                value.artifact_id, value.version_id, run_id
            ),
            (
                "UPDATE runs SET status = 'completed', completed_at = %s WHERE id = %s AND status = 'publishing'",
                [timestamp, run_id],
            ),
        ]
    )
    return statements


def _receipt_handled_statements(
    receipt: YouTubeModelReceipt,
    attempt: ModelAttempt | None,
    status: Literal["accepted", "rejected"],
    handled_at: datetime,
    *,
    error: str | None = None,
) -> list[tuple[str, list[object]]]:
    statements = [_model_attempt_statement(attempt)] if attempt is not None else []
    if receipt.trace is not None and attempt is not None:
        statements.append(_model_trace_link_statement(attempt, receipt))
    statements.append(
        (
            "UPDATE youtube_model_receipts SET status = %s, handled_attempt_id = %s, error = %s, handled_at = %s "
            "WHERE request_id = %s AND operation_key = %s AND attempt_index = %s AND status = 'unhandled'",
            [
                status,
                attempt.attempt_id if attempt is not None else None,
                error,
                handled_at.astimezone(UTC).isoformat(),
                receipt.request_id,
                receipt.operation_key,
                receipt.attempt_index,
            ],
        )
    )
    return statements


def _model_attempt_statement(attempt: ModelAttempt) -> tuple[str, list[object]]:
    record = ModelAttemptRecord.model_validate(attempt.model_dump(), strict=True)
    return (
        "INSERT INTO news_model_attempts "
        "(attempt_id, request_id, operation_key, response_id, model, input_tokens, output_tokens, "
        "cost_usd, latency_ms, status, error, observed_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT DO NOTHING",
        [
            record.attempt_id,
            record.request_id,
            record.operation_key,
            record.response_id,
            record.model,
            record.input_tokens,
            record.output_tokens,
            record.cost_usd,
            record.latency_ms,
            record.status,
            record.error,
            record.observed_at.astimezone(UTC).isoformat(),
        ],
    )


def _model_trace_link_statement(
    attempt: ModelAttempt, receipt: YouTubeModelReceipt
) -> tuple[str, list[object]]:
    if receipt.trace is None:
        raise ValueError("YouTube receipt has no model trace")
    trace = ModelTraceRecord(
        provider=receipt.trace.provider,
        trace_id=receipt.trace.trace_id,
        observation_id=receipt.trace.observation_id,
        project_ref=receipt.trace.project_ref,
        recorded_at=receipt.trace.recorded_at,
    )
    return (
        "INSERT INTO news_model_trace_links "
        "(attempt_id, provider, trace_id, observation_id, project_ref, recorded_at) "
        "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
        [
            attempt.attempt_id,
            trace.provider,
            trace.trace_id,
            trace.observation_id,
            trace.project_ref,
            trace.recorded_at.astimezone(UTC).isoformat(),
        ],
    )


def _validate_accepted_clip(
    metadata: YouTubeVideoMetadata,
    accepted: AcceptedClipExtraction,
    metadata_file: ArtifactFile,
    receipt: YouTubeModelReceipt,
    attempt: ModelAttempt,
) -> None:
    if (
        accepted.source != metadata.source
        or accepted.video_id != metadata.video_id
        or accepted.video_url != metadata.url
        or accepted.metadata_version_id != metadata_file.version_id
    ):
        raise ValueError("YouTube clip output does not match its metadata")
    expected_request_id = sha256(
        canonical_json(
            {
                "source_id": metadata.source.source_id,
                "video_id": metadata.video_id,
                "video_url": str(metadata.url),
                "duration_seconds": metadata.duration_seconds,
                "range": accepted.clip_range.model_dump(mode="json"),
                "request_digest": accepted.request_digest,
                "model": accepted.model,
            }
        )
    )
    if receipt.request_id != expected_request_id:
        raise ValueError("YouTube clip request identity does not match its accepted output")
    _validate_receipt_attempt(receipt, attempt, "news.youtube.extract_clip", "accepted")
    response_bytes = receipt.response.body_bytes()
    try:
        response_payload = TypeAdapter(dict[str, object]).validate_json(response_bytes, strict=True)
    except (ValidationError, ValueError) as error:
        raise ValueError("YouTube clip receipt payload is invalid") from error
    response_id = str(response_payload.get("responseId") or sha256(response_bytes))
    if (
        accepted.provider_response != response_payload
        or attempt.response_id != response_id
        or attempt.model != accepted.model
    ):
        raise ValueError("YouTube clip output does not match its model attempt")


def _validate_receipt_attempt(
    receipt: YouTubeModelReceipt,
    attempt: ModelAttempt,
    operation_key: str,
    status: Literal["accepted", "rejected"],
) -> None:
    expected_attempt_id = sha256(
        canonical_json(
            {
                "attempt_index": receipt.attempt_index,
                "operation_key": operation_key,
                "request_id": receipt.request_id,
                "response_id": attempt.response_id,
            }
        )
    )
    if (
        receipt.operation_key != operation_key
        or attempt.operation_key != operation_key
        or attempt.request_id != receipt.request_id
        or attempt.attempt_id != expected_attempt_id
        or attempt.model != receipt.response.requested_model
        or attempt.status != status
    ):
        raise ValueError("YouTube model receipt does not match its model attempt")


def _validate_gemini_response(
    receipt: YouTubeModelReceipt,
    response: GeminiResponse,
    attempt: ModelAttempt,
) -> None:
    try:
        response_payload = TypeAdapter(dict[str, object]).validate_json(
            receipt.response.body_bytes(), strict=True
        )
    except (ValidationError, ValueError) as error:
        raise ValueError("YouTube merge receipt payload is invalid") from error
    if (
        response.payload != response_payload
        or response.model != receipt.response.requested_model
        or response.latency_ms != receipt.response.latency_ms
        or attempt.input_tokens != response.input_tokens
        or attempt.output_tokens != response.output_tokens
        or attempt.cost_usd != response.cost_usd
        or attempt.latency_ms != response.latency_ms
    ):
        raise ValueError("YouTube merge response does not match its receipt")


def _fenced_update(sql: str, params: list[object]) -> None:
    if not catalog_mutation(sql, params):
        raise YouTubeLeaseLostError("YouTube video lease was lost")


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"
