from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file, canonical_json, sha256
from romanian_news.catalog.video_digest import (
    read_assembly_attempts,
    read_subtitle_attempts,
    record_subtitle_attempt,
    renew_slot,
)
from romanian_news.storage import publish_immutable_r2_objects, read_verified_r2_object
from romanian_news.video_digest.errors import VideoDigestCatalogError
from romanian_news.video_digest.media import (
    AssembledEdition,
    AssemblyManifest,
    SubtitleAttemptFailure,
    SubtitleFailureEvidence,
    SubtitleStrategy,
    SubtitleTimeSpan,
    SubtitleTimingRequest,
    _webvtt,
    subtitle_timing_requests,
)
from romanian_news.video_digest.models import SlotLease
from romanian_news.video_digest.orchestration import (
    LEASE_DURATION,
    ActionAdvanced,
    ActionOutcome,
    SubtitleAction,
    SubtitleAttemptReference,
)
from romanian_news.video_digest.preflight import read_accepted_digest_plan
from romanian_news.video_digest.subtitle_timing import FasterWhisperSubtitleTimingProvider


class TimingProvider(Protocol):
    def timings(
        self,
        strategy: SubtitleStrategy,
        requests: tuple[SubtitleTimingRequest, ...],
        media_path: Path,
    ) -> tuple[tuple[SubtitleTimeSpan, ...], ...]: ...


class ResumableSubtitlePort:
    def __init__(
        self,
        *,
        provider_factory: Callable[[Callable[[], None]], TimingProvider] | None = None,
        renew_lease: Callable[[SlotLease], SlotLease] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._provider_factory = provider_factory or (
            lambda before_transcribe: FasterWhisperSubtitleTimingProvider(
                before_transcribe=before_transcribe
            )
        )
        self._clock = clock or (lambda: datetime.now(UTC))
        self._renew_lease = renew_lease or (
            lambda lease: renew_slot(lease, now=self._clock(), lease_duration=LEASE_DURATION)
        )

    def execute(self, action: SubtitleAction) -> ActionOutcome:
        attempts = read_subtitle_attempts(action.lease.edition_id)
        if len(attempts) > action.attempt_index:
            return ActionAdvanced()
        if len(attempts) != action.attempt_index or any(
            attempt.disposition != "failed" for attempt in attempts
        ):
            raise ValueError("Subtitle action does not match the durable attempt sequence")
        prior_failures = tuple(
            _read_failure(action.lease.edition_id, attempt) for attempt in attempts
        )
        lease = action.lease

        def renew() -> None:
            nonlocal lease
            lease = self._renew_lease(lease)

        renew()
        verified, _plan = read_accepted_digest_plan(lease.edition_id)
        screenplay = verified.plan
        assembled = _read_assembled_edition(lease)
        if screenplay.edition_id != lease.edition_id:
            raise ValueError("Accepted screenplay does not match the subtitle edition")
        if len(assembled.clip_durations_ms) != len(screenplay.stories):
            raise ValueError("Accepted assembly does not cover the screenplay stories")
        strategy = SubtitleStrategy(action.strategy)
        try:
            cue_texts, requests = subtitle_timing_requests(screenplay, assembled)
            with TemporaryDirectory() as directory:
                media_path = Path(directory) / "edition.mp4"
                media_path.write_bytes(assembled.video_file.content)
                provider = self._provider_factory(renew)
                renew()
                timings = provider.timings(strategy, requests, media_path)
            content = _webvtt(screenplay, cue_texts, timings, assembled.duration_ms)
        except VideoDigestCatalogError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failure = SubtitleAttemptFailure(strategy=strategy, code=type(error).__name__)
            evidence_file = _failure_file(
                lease.edition_id, action.attempt_index, (*prior_failures, failure)
            )
            renew()
            publish_immutable_r2_objects(((evidence_file.r2_key, evidence_file.content),))
            record_subtitle_attempt(
                lease,
                action.attempt_index,
                action.strategy,
                "failed",
                evidence_file=evidence_file,
                recorded_at=self._clock(),
            )
            return ActionAdvanced()

        subtitle_file = artifact_file(
            artifact_id=f"{lease.edition_id}:subtitles",
            artifact_kind="video_digest_subtitles",
            title="Video digest subtitles",
            content=content,
            r2_key=(
                f"news/video-digest/{lease.edition_id}/subtitles/"
                f"subtitles-{sha256(content)}.vtt"
            ),
            media_type="text/vtt",
        )
        renew()
        publish_immutable_r2_objects(((subtitle_file.r2_key, subtitle_file.content),))
        record_subtitle_attempt(
            lease,
            action.attempt_index,
            action.strategy,
            "succeeded",
            evidence_file=subtitle_file,
            subtitle_file=subtitle_file,
            recorded_at=self._clock(),
        )
        return ActionAdvanced()


def _read_assembled_edition(lease: SlotLease) -> AssembledEdition:
    attempts = read_assembly_attempts(lease.edition_id)
    if not attempts or attempts[-1].disposition != "succeeded":
        raise ValueError("Subtitle action has no accepted assembled media")
    accepted = attempts[-1]
    if accepted.video is None or accepted.manifest is None:
        raise ValueError("Accepted assembly is missing its video or manifest")
    video_content = read_verified_r2_object(accepted.video.r2_key, accepted.video.content_digest)
    video_file = artifact_file(
        artifact_id=f"{lease.edition_id}:assembled-video",
        artifact_kind="video_digest_assembled_video",
        title="Assembled video digest",
        content=video_content,
        r2_key=accepted.video.r2_key,
        media_type="video/mp4",
    )
    if (
        _reference(video_file) != _reference(accepted.video)
        or len(video_content) != accepted.video.byte_size
        or accepted.video.media_type != "video/mp4"
    ):
        raise ValueError("Accepted assembled video conflicts with its catalog reference")
    manifest_content = read_verified_r2_object(
        accepted.manifest.r2_key, accepted.manifest.content_digest
    )
    manifest_file = artifact_file(
        artifact_id=f"{lease.edition_id}:assembly-manifest",
        artifact_kind="video_digest_assembly_manifest",
        title="Video digest assembly manifest",
        content=manifest_content,
        r2_key=accepted.manifest.r2_key,
        media_type="application/json",
    )
    if _reference(manifest_file) != accepted.manifest:
        raise ValueError("Accepted assembly manifest conflicts with its catalog reference")
    manifest = AssemblyManifest.model_validate_json(manifest_content, strict=True)
    if (
        manifest.edition_id != lease.edition_id
        or manifest.output_content_digest != video_file.content_digest
        or manifest.output_byte_size != len(video_content)
        or tuple(clip.story_position for clip in manifest.clips)
        != tuple(range(len(manifest.clips)))
        or manifest.expected_duration_ms != sum(clip.duration_ms for clip in manifest.clips)
        or abs(manifest.expected_duration_ms - manifest.output_probe.duration_ms) > 100
    ):
        raise ValueError("Accepted assembly manifest does not match its video or edition")
    return AssembledEdition(
        video_file=video_file,
        manifest_file=manifest_file,
        duration_ms=manifest.output_probe.duration_ms,
        clip_durations_ms=tuple(clip.duration_ms for clip in manifest.clips),
    )


def _read_failure(edition_id: str, reference: SubtitleAttemptReference) -> SubtitleAttemptFailure:
    if reference.evidence.artifact_id != f"{edition_id}:{reference.attempt_index}:subtitle-attempt":
        raise ValueError("Stored subtitle failure identity is invalid")
    content = read_verified_r2_object(reference.evidence.r2_key, reference.evidence.content_digest)
    file = _attempt_failure_file(reference.evidence.artifact_id, reference.attempt_index, content)
    if _reference(file) != reference.evidence:
        raise ValueError("Stored subtitle attempt evidence conflicts with its catalog reference")
    failure = SubtitleAttemptFailure.model_validate_json(content, strict=True)
    if (
        canonical_json(failure.model_dump(mode="json")) != content
        or failure.strategy.value != reference.strategy
    ):
        raise ValueError("Stored subtitle failure does not match its attempt")
    return failure


def _attempt_failure_file(artifact_id: str, index: int, content: bytes) -> ArtifactFile:
    if not artifact_id.endswith(f":{index}:subtitle-attempt"):
        raise ValueError("Stored subtitle failure identity is invalid")
    edition_id = artifact_id.removesuffix(f":{index}:subtitle-attempt")
    return artifact_file(
        artifact_id=artifact_id,
        artifact_kind="video_digest_subtitle_attempt",
        title="Video digest subtitle attempt",
        content=content,
        r2_key=(
            f"news/video-digest/{edition_id}/subtitles/" f"attempt-{index}-{sha256(content)}.json"
        ),
        media_type="application/json",
    )


def _failure_file(
    edition_id: str, index: int, failures: tuple[SubtitleAttemptFailure, ...]
) -> ArtifactFile:
    if index == 2:
        content = canonical_json(
            SubtitleFailureEvidence(edition_id=edition_id, attempts=failures).model_dump(
                mode="json"
            )
        )
        return artifact_file(
            artifact_id=f"{edition_id}:subtitle-failure",
            artifact_kind="video_digest_subtitle_failure",
            title="Video digest subtitle failure",
            content=content,
            r2_key=f"news/video-digest/{edition_id}/subtitles/failure-{sha256(content)}.json",
            media_type="application/json",
        )
    content = canonical_json(failures[-1].model_dump(mode="json"))
    return _attempt_failure_file(f"{edition_id}:{index}:subtitle-attempt", index, content)


def _reference(file: ArtifactFile | ArtifactReference) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=file.artifact_id,
        version_id=file.version_id,
        content_digest=file.content_digest,
        r2_key=file.r2_key,
    )
