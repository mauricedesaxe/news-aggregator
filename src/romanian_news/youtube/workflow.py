from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from romanian_news import Sha256
from romanian_news.analysis.tracing import flush_langfuse_traces
from romanian_news.catalog.youtube import (
    claim_youtube_video,
    defer_youtube_video,
    publish_next_approved_candidate,
    publish_youtube_candidate,
    publish_youtube_metadata,
    publish_youtube_poll,
    read_accepted_clips,
    read_youtube_metadata,
    reject_youtube_video,
    renew_youtube_lease,
)
from romanian_news.catalog_transport import ResearchCatalogError
from romanian_news.storage import (
    ResearchObjectIntegrityError,
    ResearchObjectUnavailable,
)
from romanian_news.youtube.analysis import (
    analyze_and_publish_clip,
    analyze_and_publish_merge,
    clip_request_digest,
    complete_clip_coverage,
    enforce_video_budget,
    plan_clip_ranges,
)
from romanian_news.youtube.discovery import acquire_youtube_poll, fetch_video_metadata
from romanian_news.youtube.errors import (
    YouTubeDeterministicError,
    YouTubeInfrastructureError,
    YouTubeLeaseLostError,
)
from romanian_news.youtube.models import (
    YOUTUBE_MODEL,
    YouTubePollStatus,
    YouTubePublicationResult,
    YouTubeSource,
    YouTubeSourceResult,
    YouTubeVideoLease,
)


def materialize_youtube_source(
    source: YouTubeSource, scheduled_at: datetime, implementation_ref: str
) -> YouTubeSourceResult:
    try:
        capture = acquire_youtube_poll(source, scheduled_at)
        decision = publish_youtube_poll(capture)
        if decision.status == YouTubePollStatus.OVERLAP_LOST:
            raise RuntimeError(
                f"{source.display_name} YouTube feed lost overlap with known video IDs; refusing backfill"
            )
        lease = claim_youtube_video(source, str(uuid4()), datetime.now(UTC))
        if lease is None:
            return YouTubeSourceResult(
                source_id=source.source_id, poll_status=decision.status, candidate_version_id=None
            )
        candidate_version_id = _process_claimed_video(lease, implementation_ref)
        return YouTubeSourceResult(
            source_id=source.source_id,
            poll_status=decision.status,
            candidate_version_id=candidate_version_id,
        )
    finally:
        flush_langfuse_traces()


def materialize_next_approved_publication(
    implementation_ref: str,
) -> YouTubePublicationResult | None:
    return publish_next_approved_candidate(implementation_ref)


def _process_claimed_video(lease: YouTubeVideoLease, implementation_ref: str) -> Sha256:
    try:
        return _create_claimed_candidate(lease, implementation_ref)
    except YouTubeLeaseLostError:
        raise
    except YouTubeDeterministicError as error:
        reject_youtube_video(lease, error, datetime.now(UTC))
        raise
    except (
        YouTubeInfrastructureError,
        ResearchCatalogError,
        ResearchObjectUnavailable,
        ResearchObjectIntegrityError,
    ) as error:
        defer_youtube_video(lease, error, datetime.now(UTC))
        raise


def _create_claimed_candidate(lease: YouTubeVideoLease, implementation_ref: str) -> Sha256:
    lease = renew_youtube_lease(lease, datetime.now(UTC))
    if lease.metadata_version_id is None:
        metadata = fetch_video_metadata(lease.source, lease.video_id, lease.first_poll_version_id)
        metadata_file = publish_youtube_metadata(metadata, implementation_ref, lease)
    else:
        metadata, metadata_file = read_youtube_metadata(lease)
    enforce_video_budget(metadata.duration_seconds)

    def renew() -> None:
        nonlocal lease
        lease = renew_youtube_lease(lease, datetime.now(UTC))

    planned_ranges = plan_clip_ranges(metadata.duration_seconds)
    accepted = [
        value
        for value in read_accepted_clips(
            metadata, metadata_file.version_id, YOUTUBE_MODEL, clip_request_digest()
        )
        if value.clip_range in planned_ranges
    ]
    accepted_ranges = {
        (value.clip_range.start_second, value.clip_range.duration_seconds) for value in accepted
    }
    for clip_range in planned_ranges:
        if (clip_range.start_second, clip_range.duration_seconds) in accepted_ranges:
            continue
        accepted.append(
            analyze_and_publish_clip(metadata, metadata_file, clip_range, implementation_ref, renew)
        )
    complete = complete_clip_coverage(metadata.duration_seconds, tuple(accepted))
    merge, merge_file = analyze_and_publish_merge(metadata, complete, implementation_ref, renew)
    renew()
    return publish_youtube_candidate(
        metadata, metadata_file, complete, merge_file, merge, implementation_ref, lease
    )
