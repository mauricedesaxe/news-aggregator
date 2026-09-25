from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file, canonical_json, sha256
from romanian_news.catalog.video_digest import (
    GenerationAttemptReference,
    checkpoint_assembled_video,
    checkpoint_assembly_ready,
    checkpoint_generation_acceptance,
    checkpoint_generation_failure,
    checkpoint_subtitles,
    read_generation_attempts,
    record_assembly_attempt,
)
from romanian_news.storage import (
    ResearchObjectIntegrityError,
    publish_immutable_r2_objects,
    publish_private_video_object,
    read_verified_r2_object,
)
from romanian_news.video_digest.generation import (
    CandidateReady,
    GenerationFailed,
    GenerationRetryAvailable,
)
from romanian_news.video_digest.models import (
    AvailableSubtitles,
    DigestPlan,
    FailedSubtitles,
    GenerationStage,
    PlannedStory,
    SlotLease,
    SubtitleOutcome,
)
from romanian_news.video_digest.narration_quality import (
    NarrationCheckError,
    clip_narration_matches,
    clip_narration_tail_cutoff_ms,
)
from romanian_news.video_digest.planning import ScreenplayPlan
from romanian_news.video_digest.subtitle_text import (
    SubtitleAttemptFailure as SubtitleAttemptFailure,
)
from romanian_news.video_digest.subtitle_text import (
    SubtitleFailureEvidence as SubtitleFailureEvidence,
)
from romanian_news.video_digest.subtitle_text import (
    SubtitleStrategy as SubtitleStrategy,
)
from romanian_news.video_digest.subtitle_text import (
    SubtitleTimeSpan as SubtitleTimeSpan,
)
from romanian_news.video_digest.subtitle_text import (
    SubtitleTimingProvider as SubtitleTimingProvider,
)
from romanian_news.video_digest.subtitle_text import (
    SubtitleTimingRequest as SubtitleTimingRequest,
)
from romanian_news.video_digest.subtitle_text import (
    _webvtt as _webvtt,
)
from romanian_news.video_digest.subtitle_text import (
    subtitle_timing_requests as subtitle_timing_requests,
)

MEDIA_POLICY_VERSION = "video-digest-media-v1"
CANDIDATE_MEDIA_POLICY_VERSION = "video-digest-media-v2"
LOUDNESS_POLICY_VERSION = "ebu-r128-minus-16-v1"
FADE_SECONDS = Decimal("0.25")
AAC_FRAME_DURATION_MS = 32
MEDIA_PROCESS_TIMEOUT_SECONDS = 900


class MediaValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class MediaProbe(NewsModel):
    duration_ms: Annotated[int, Field(gt=0)]
    video_duration_ms: Annotated[int, Field(gt=0)]
    audio_duration_ms: Annotated[int, Field(gt=0)]
    video_start_ms: Annotated[int, Field(ge=0)]
    audio_start_ms: Annotated[int, Field(ge=0)]
    video_frames: Annotated[int, Field(gt=0)]
    audio_frames: Annotated[int, Field(gt=0)]
    video_codec: Literal["h264"]
    width: Literal[1344]
    height: Literal[768]
    frame_rate: Literal["24/1"]
    pixel_format: Literal["yuv420p"]
    audio_codec: Literal["aac"]
    channels: Literal[2]
    sample_rate_hz: Literal[32000]


class CandidateValidationEvidence(NewsModel):
    policy_version: Literal["video-digest-media-v1", "video-digest-media-v2"] = (
        CANDIDATE_MEDIA_POLICY_VERSION
    )
    request_id: str
    request_artifact_version_id: Sha256
    response_artifact_version_id: Sha256
    story_id: str
    story_position: Annotated[int, Field(ge=0)]
    attempt_index: Annotated[int, Field(ge=0, le=1)]
    requested_duration_ms: Annotated[int, Field(gt=0)]
    candidate_r2_key: str
    candidate_content_digest: Sha256
    candidate_byte_size: Annotated[int, Field(gt=0)]
    accepted_clip_content_digest: Sha256 | None = None
    accepted_clip_byte_size: Annotated[int, Field(gt=0)] | None = None
    muted_after_ms: Annotated[int, Field(gt=0)] | None = None
    probe: MediaProbe
    full_decode_succeeded: Literal[True] = True

    @model_validator(mode="after")
    def require_accepted_clip_provenance(self) -> CandidateValidationEvidence:
        if self.policy_version == "video-digest-media-v2":
            if self.accepted_clip_content_digest is None or self.accepted_clip_byte_size is None:
                raise ValueError("Media v2 evidence requires accepted clip identity")
            if self.muted_after_ms is None and (
                self.accepted_clip_content_digest != self.candidate_content_digest
                or self.accepted_clip_byte_size != self.candidate_byte_size
            ):
                raise ValueError("Unchanged candidate must retain its original identity")
        return self


class AcceptedCandidate(NewsModel):
    clip_artifact_version_id: Sha256
    validation_artifact_version_id: Sha256


class AcceptedClipManifestEntry(NewsModel):
    story_position: Annotated[int, Field(ge=0)]
    request_id: str
    artifact_version_id: Sha256
    content_digest: Sha256
    byte_size: Annotated[int, Field(gt=0)]
    duration_ms: Annotated[int, Field(gt=0)]


class LoudnessMeasurement(NewsModel):
    input_i: Decimal
    input_lra: Decimal
    input_tp: Decimal
    input_thresh: Decimal
    target_offset: Decimal


class AssemblyManifest(NewsModel):
    media_policy_version: Literal["video-digest-media-v1"] = MEDIA_POLICY_VERSION
    loudness_policy_version: Literal["ebu-r128-minus-16-v1"] = LOUDNESS_POLICY_VERSION
    edition_id: str
    clips: Annotated[tuple[AcceptedClipManifestEntry, ...], Field(min_length=1)]
    fade_seconds: Decimal = FADE_SECONDS
    transition_mode: Literal["non-overlap"] = "non-overlap"
    video_codec: Literal["libx264"] = "libx264"
    video_preset: Literal["medium"] = "medium"
    video_crf: Literal[18] = 18
    video_threads: Literal[1] = 1
    pixel_format: Literal["yuv420p"] = "yuv420p"
    audio_codec: Literal["aac"] = "aac"
    audio_bitrate: Literal["192k"] = "192k"
    audio_channels: Literal[2] = 2
    audio_sample_rate_hz: Literal[32000] = 32000
    metadata_policy: Literal["stripped"] = "stripped"
    container_policy: Literal["faststart"] = "faststart"
    loudness_target_i: Decimal = Decimal("-16")
    loudness_target_lra: Decimal = Decimal("11")
    loudness_target_tp: Decimal = Decimal("-1.5")
    loudness_measurement: LoudnessMeasurement
    expected_duration_ms: Annotated[int, Field(gt=0)]
    output_content_digest: Sha256
    output_byte_size: Annotated[int, Field(gt=0)]
    output_probe: MediaProbe
    full_decode_succeeded: Literal[True] = True


class AssembledEdition(NewsModel):
    video_file: ArtifactFile
    manifest_file: ArtifactFile
    duration_ms: Annotated[int, Field(gt=0)]
    clip_durations_ms: Annotated[tuple[int, ...], Field(min_length=1)]


def ffprobe_command(path: Path) -> tuple[str, ...]:
    return (
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    )


def full_decode_command(path: Path) -> tuple[str, ...]:
    return (
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    )


def assembly_command(
    inputs: Sequence[Path], durations_ms: Sequence[int], output: Path
) -> tuple[str, ...]:
    if not inputs or len(inputs) != len(durations_ms):
        raise ValueError("Assembly requires one duration for every input")
    arguments = ["ffmpeg", "-nostdin", "-v", "error", "-xerror"]
    for path in inputs:
        arguments.extend(("-i", str(path)))
    filters: list[str] = []
    joins: list[str] = []
    for index, duration_ms in enumerate(durations_ms):
        duration = Decimal(duration_ms) / 1000
        if duration <= FADE_SECONDS * 2:
            raise ValueError("Accepted clips must be longer than both fades")
        fade_out = _decimal_argument(duration - FADE_SECONDS)
        filters.extend(
            (
                f"[{index}:v]setpts=PTS-STARTPTS,fade=t=in:st=0:d=0.25,"
                f"fade=t=out:st={fade_out}:d=0.25[v{index}]",
                f"[{index}:a]asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.25,"
                f"afade=t=out:st={fade_out}:d=0.25[a{index}]",
            )
        )
        joins.extend((f"[v{index}]", f"[a{index}]"))
    filters.append("".join(joins) + f"concat=n={len(inputs)}:v=1:a=1[v][a]")
    arguments.extend(
        (
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "24",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ac",
            "2",
            "-ar",
            "32000",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(output),
        )
    )
    return tuple(arguments)


def loudnorm_analysis_command(path: Path) -> tuple[str, ...]:
    return (
        "ffmpeg",
        "-nostdin",
        "-v",
        "info",
        "-xerror",
        "-i",
        str(path),
        "-vn",
        "-af",
        "loudnorm=I=-16:LRA=11:TP=-1.5:print_format=json",
        "-f",
        "null",
        "-",
    )


def loudnorm_render_command(
    source: Path, output: Path, measurement: LoudnessMeasurement
) -> tuple[str, ...]:
    loudnorm = (
        "loudnorm=I=-16:LRA=11:TP=-1.5:"
        f"measured_I={_decimal_argument(measurement.input_i)}:"
        f"measured_LRA={_decimal_argument(measurement.input_lra)}:"
        f"measured_TP={_decimal_argument(measurement.input_tp)}:"
        f"measured_thresh={_decimal_argument(measurement.input_thresh)}:"
        f"offset={_decimal_argument(measurement.target_offset)}:linear=true:print_format=summary"
    )
    return (
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c:v",
        "copy",
        "-af",
        loudnorm,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        "-ar",
        "32000",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(output),
    )


def validate_media_file(
    path: Path,
    *,
    expected_digest: Sha256,
    expected_size: int,
    requested_duration_ms: int,
    before_process: Callable[[], None] | None = None,
) -> MediaProbe:
    content = path.read_bytes()
    if len(content) != expected_size:
        raise MediaValidationError("stored_size_mismatch", "Stored candidate size does not match")
    if sha256(content) != expected_digest:
        raise MediaValidationError(
            "stored_digest_mismatch", "Stored candidate digest does not match"
        )
    try:
        if before_process is not None:
            before_process()
        probe_result = _run(ffprobe_command(path), capture_output=True)
    except subprocess.CalledProcessError as error:
        raise MediaValidationError("probe_failed", "Candidate probe failed") from error
    try:
        payload = cast(dict[str, object], json.loads(probe_result.stdout))
        probe = _parse_probe(payload)
    except (
        InvalidOperation,
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        json.JSONDecodeError,
    ) as error:
        raise MediaValidationError("unusable_probe", "Candidate probe is unusable") from error
    within_duration_tolerance = all(
        abs(duration_ms - requested_duration_ms) <= 100
        for duration_ms in (
            probe.duration_ms,
            probe.video_duration_ms,
            probe.audio_duration_ms,
        )
    )
    aac_tail_ms = probe.audio_duration_ms - probe.video_duration_ms
    within_aac_padding_tolerance = (
        0 < aac_tail_ms <= AAC_FRAME_DURATION_MS
        and abs(probe.video_duration_ms - requested_duration_ms) <= 100
        and abs(probe.duration_ms - aac_tail_ms - requested_duration_ms) <= 100
    )
    if not (within_duration_tolerance or within_aac_padding_tolerance):
        raise MediaValidationError(
            "duration_mismatch", "Candidate duration differs from the requested duration"
        )
    try:
        if before_process is not None:
            before_process()
        _run(full_decode_command(path), capture_output=True)
    except subprocess.CalledProcessError as error:
        raise MediaValidationError("full_decode_failed", "Candidate decode failed") from error
    return probe


def accept_candidate(
    lease: SlotLease,
    candidate: CandidateReady,
    story: PlannedStory,
    *,
    approved_narration: str,
) -> AcceptedCandidate | GenerationRetryAvailable | GenerationFailed:
    attempt = _matching_attempt(lease, candidate)
    if story.position != candidate.story_position or story.story_id != candidate.story_id:
        raise ValueError("Candidate does not match its planned story")
    if attempt.stage is GenerationStage.ACCEPTED:
        return _replay_accepted_candidate(attempt, candidate)
    try:
        content, probe, muted_after_ms = _validated_candidate_media(
            candidate, story, approved_narration
        )
    except (
        MediaValidationError,
        ResearchObjectIntegrityError,
        subprocess.TimeoutExpired,
    ) as error:
        return _reject_candidate(lease, candidate, error)

    evidence = CandidateValidationEvidence(
        request_id=candidate.request_id,
        request_artifact_version_id=attempt.request.request_artifact_version_id,
        response_artifact_version_id=candidate.response_artifact_version_id,
        story_id=candidate.story_id,
        story_position=candidate.story_position,
        attempt_index=candidate.attempt_index,
        requested_duration_ms=story.requested_duration_ms,
        candidate_r2_key=candidate.candidate.r2_key,
        candidate_content_digest=candidate.candidate.content_digest,
        candidate_byte_size=candidate.candidate.byte_size,
        accepted_clip_content_digest=sha256(content),
        accepted_clip_byte_size=len(content),
        muted_after_ms=muted_after_ms,
        probe=probe,
    )
    clip_file = artifact_file(
        artifact_id=f"{story.story_id}:accepted-clip",
        artifact_kind="video_digest_accepted_clip",
        title=f"Accepted video digest clip {story.position}",
        content=content,
        r2_key=(
            f"news/video-digest/{lease.edition_id}/clips/"
            f"{story.position}-{sha256(content)}"
            f"{'-silenced' if muted_after_ms is not None else ''}.mp4"
        ),
        media_type="video/mp4",
    )
    evidence_content = canonical_json(evidence.model_dump(mode="json"))
    validation_file = artifact_file(
        artifact_id=f"{candidate.request_id}:validation",
        artifact_kind="video_digest_candidate_validation",
        title=f"Candidate validation for {candidate.request_id}",
        content=evidence_content,
        r2_key=(
            f"news/video-digest/{lease.edition_id}/generation/"
            f"validation-{candidate.request_id}-{sha256(evidence_content)}.json"
        ),
        media_type="application/json",
    )
    publish_private_video_object(
        clip_file.r2_key,
        clip_file.content,
        retention="permanent",
        source_lineage=clip_file.version_id,
    )
    publish_immutable_r2_objects(((validation_file.r2_key, validation_file.content),))
    checkpoint_generation_acceptance(
        lease,
        candidate.request_id,
        clip_file=clip_file,
        validation_file=validation_file,
        cost=candidate.cost,
        recorded_at=_now(),
    )
    return AcceptedCandidate(
        clip_artifact_version_id=clip_file.version_id,
        validation_artifact_version_id=validation_file.version_id,
    )


def _replay_accepted_candidate(
    attempt: GenerationAttemptReference, candidate: CandidateReady
) -> AcceptedCandidate:
    clip = attempt.accepted_clip
    validation = attempt.validation_evidence
    if clip is None or validation is None:
        raise ValueError("Accepted candidate does not match stored media evidence")
    if clip.r2_key.endswith("-silenced.mp4"):
        recorded = CandidateValidationEvidence.model_validate_json(
            read_verified_r2_object(validation.r2_key, validation.content_digest), strict=True
        )
        if (
            recorded.muted_after_ms is None
            or recorded.request_id != candidate.request_id
            or recorded.candidate_content_digest != candidate.candidate.content_digest
            or recorded.candidate_byte_size != candidate.candidate.byte_size
            or clip.content_digest != recorded.accepted_clip_content_digest
            or clip.byte_size != recorded.accepted_clip_byte_size
        ):
            raise ValueError("Accepted candidate does not match stored media evidence")
    elif (
        clip.content_digest != candidate.candidate.content_digest
        or clip.byte_size != candidate.candidate.byte_size
    ):
        raise ValueError("Accepted candidate does not match stored media evidence")
    return AcceptedCandidate(
        clip_artifact_version_id=clip.version_id,
        validation_artifact_version_id=validation.version_id,
    )


def _validated_candidate_media(
    candidate: CandidateReady, story: PlannedStory, approved_narration: str
) -> tuple[bytes, MediaProbe, int | None]:
    content = read_verified_r2_object(
        candidate.candidate.r2_key, candidate.candidate.content_digest
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate.mp4"
        path.write_bytes(content)
        probe = validate_media_file(
            path,
            expected_digest=candidate.candidate.content_digest,
            expected_size=candidate.candidate.byte_size,
            requested_duration_ms=story.requested_duration_ms,
        )
        if _checked_narration_matches(path, approved_narration):
            return content, probe, None
        try:
            cutoff_ms = clip_narration_tail_cutoff_ms(path, approved_narration)
        except NarrationCheckError as error:
            raise MediaValidationError("narration_check_failed", str(error)) from error
        if cutoff_ms is None:
            raise MediaValidationError(
                "narration_mismatch", "Candidate speech differs from the approved screenplay"
            )
        accepted_path = Path(directory) / "accepted.mp4"
        _silence_unapproved_tail(path, accepted_path, cutoff_ms)
        content = accepted_path.read_bytes()
        probe = validate_media_file(
            accepted_path,
            expected_digest=sha256(content),
            expected_size=len(content),
            requested_duration_ms=story.requested_duration_ms,
        )
        if not _checked_narration_matches(accepted_path, approved_narration):
            raise MediaValidationError(
                "narration_mismatch", "Candidate speech differs from the approved screenplay"
            )
        return content, probe, cutoff_ms


def _checked_narration_matches(path: Path, approved_narration: str) -> bool:
    try:
        return clip_narration_matches(path, approved_narration)
    except NarrationCheckError as error:
        raise MediaValidationError("narration_check_failed", str(error)) from error


def _silence_unapproved_tail(source: Path, destination: Path, cutoff_ms: int) -> None:
    try:
        _run(
            (
                "ffmpeg",
                "-y",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c:v",
                "copy",
                "-af",
                f"volume=enable='gte(t,{cutoff_ms / 1000:.3f})':volume=0",
                "-c:a",
                "aac",
                "-ar",
                "32000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(destination),
            ),
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise MediaValidationError(
            "narration_tail_processing_failed", "Could not silence extra speech"
        ) from error


def assemble_edition(
    lease: SlotLease,
    plan: DigestPlan,
    *,
    attempt_index: int | None = None,
    renew_lease: Callable[[SlotLease], SlotLease] | None = None,
) -> AssembledEdition:
    if lease.edition_id != plan.edition_id:
        raise ValueError("Assembly plan does not match the claimed edition")
    current_lease = lease

    def renew() -> None:
        nonlocal current_lease
        if renew_lease is not None:
            current_lease = renew_lease(current_lease)

    renew()
    attempts = read_generation_attempts(lease.edition_id)
    accepted = _accepted_attempts(plan, attempts)
    checkpoint_assembly_ready(current_lease, recorded_at=_now())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths: list[Path] = []
        entries: list[AcceptedClipManifestEntry] = []
        for story, attempt in zip(plan.stories, accepted, strict=True):
            assert attempt.accepted_clip is not None
            content = read_verified_r2_object(
                attempt.accepted_clip.r2_key, attempt.accepted_clip.content_digest
            )
            path = root / f"clip-{story.position}.mp4"
            path.write_bytes(content)
            probe = validate_media_file(
                path,
                expected_digest=attempt.accepted_clip.content_digest,
                expected_size=attempt.accepted_clip.byte_size,
                requested_duration_ms=story.requested_duration_ms,
                before_process=renew,
            )
            paths.append(path)
            entries.append(
                AcceptedClipManifestEntry(
                    story_position=story.position,
                    request_id=attempt.request.request_id,
                    artifact_version_id=attempt.accepted_clip.version_id,
                    content_digest=attempt.accepted_clip.content_digest,
                    byte_size=attempt.accepted_clip.byte_size,
                    duration_ms=probe.duration_ms,
                )
            )
        intermediate = root / "assembled.mp4"
        final = root / "final.mp4"
        renew()
        _run(
            assembly_command(paths, [entry.duration_ms for entry in entries], intermediate),
            capture_output=True,
        )
        renew()
        analysis = _run(loudnorm_analysis_command(intermediate), capture_output=True)
        measurement = _parse_loudness(analysis.stderr)
        renew()
        _run(loudnorm_render_command(intermediate, final, measurement), capture_output=True)
        output = final.read_bytes()
        expected_duration_ms = sum(entry.duration_ms for entry in entries)
        output_probe = validate_media_file(
            final,
            expected_digest=sha256(output),
            expected_size=len(output),
            requested_duration_ms=expected_duration_ms,
            before_process=renew,
        )

    video_file = artifact_file(
        artifact_id=f"{lease.edition_id}:assembled-video",
        artifact_kind="video_digest_assembled_video",
        title="Assembled video digest",
        content=output,
        r2_key=(f"news/video-digest/{lease.edition_id}/assembled/" f"video-{sha256(output)}.mp4"),
        media_type="video/mp4",
    )
    manifest = AssemblyManifest(
        edition_id=lease.edition_id,
        clips=tuple(entries),
        loudness_measurement=measurement,
        expected_duration_ms=expected_duration_ms,
        output_content_digest=video_file.content_digest,
        output_byte_size=len(output),
        output_probe=output_probe,
    )
    manifest_content = canonical_json(manifest.model_dump(mode="json"))
    manifest_file = artifact_file(
        artifact_id=f"{lease.edition_id}:assembly-manifest",
        artifact_kind="video_digest_assembly_manifest",
        title="Video digest assembly manifest",
        content=manifest_content,
        r2_key=(
            f"news/video-digest/{lease.edition_id}/assembled/"
            f"manifest-{sha256(manifest_content)}.json"
        ),
        media_type="application/json",
    )
    publish_immutable_r2_objects(
        ((video_file.r2_key, video_file.content), (manifest_file.r2_key, manifest_file.content))
    )
    renew()
    if attempt_index is None:
        checkpoint_assembled_video(
            current_lease,
            video_file=video_file,
            manifest_file=manifest_file,
            recorded_at=_now(),
        )
    else:
        record_assembly_attempt(
            current_lease,
            attempt_index,
            "succeeded",
            evidence_file=manifest_file,
            video_file=video_file,
            manifest_file=manifest_file,
            recorded_at=_now(),
        )
    return AssembledEdition(
        video_file=video_file,
        manifest_file=manifest_file,
        duration_ms=output_probe.duration_ms,
        clip_durations_ms=tuple(entry.duration_ms for entry in entries),
    )


def produce_subtitles(
    lease: SlotLease,
    screenplay: ScreenplayPlan,
    assembled: AssembledEdition,
    provider: SubtitleTimingProvider,
) -> SubtitleOutcome:
    if lease.edition_id != screenplay.edition_id:
        raise ValueError("Subtitle screenplay does not match the claimed edition")
    failures: list[SubtitleAttemptFailure] = []
    try:
        cue_texts, requests = subtitle_timing_requests(screenplay, assembled)
    except ValueError as error:
        failures.extend(
            SubtitleAttemptFailure(strategy=strategy, code=type(error).__name__)
            for strategy in SubtitleStrategy
        )
    else:
        with tempfile.TemporaryDirectory() as directory:
            media_path = Path(directory) / "edition.mp4"
            media_path.write_bytes(assembled.video_file.content)
            for strategy in SubtitleStrategy:
                try:
                    timings = provider.timings(strategy, requests, media_path)
                    webvtt = _webvtt(
                        screenplay,
                        cue_texts,
                        timings,
                        assembled.duration_ms,
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    failures.append(
                        SubtitleAttemptFailure(
                            strategy=strategy,
                            code=type(error).__name__,
                        )
                    )
                    continue
                subtitle_file = artifact_file(
                    artifact_id=f"{lease.edition_id}:subtitles",
                    artifact_kind="video_digest_subtitles",
                    title="Video digest subtitles",
                    content=webvtt,
                    r2_key=(
                        f"news/video-digest/{lease.edition_id}/subtitles/"
                        f"subtitles-{sha256(webvtt)}.vtt"
                    ),
                    media_type="text/vtt",
                )
                publish_immutable_r2_objects(((subtitle_file.r2_key, subtitle_file.content),))
                outcome = AvailableSubtitles(artifact_version_id=subtitle_file.version_id)
                return checkpoint_subtitles(
                    lease,
                    outcome,
                    artifact_file=subtitle_file,
                    recorded_at=_now(),
                )

    evidence = SubtitleFailureEvidence(edition_id=lease.edition_id, attempts=tuple(failures))
    content = canonical_json(evidence.model_dump(mode="json"))
    failure_file = artifact_file(
        artifact_id=f"{lease.edition_id}:subtitle-failure",
        artifact_kind="video_digest_subtitle_failure",
        title="Video digest subtitle failure",
        content=content,
        r2_key=(
            f"news/video-digest/{lease.edition_id}/subtitles/" f"failure-{sha256(content)}.json"
        ),
        media_type="application/json",
    )
    publish_immutable_r2_objects(((failure_file.r2_key, failure_file.content),))
    outcome = FailedSubtitles(evidence_artifact_version_id=failure_file.version_id)
    return checkpoint_subtitles(
        lease,
        outcome,
        artifact_file=failure_file,
        recorded_at=_now(),
    )


def _parse_probe(payload: dict[str, object]) -> MediaProbe:
    streams = cast(list[dict[str, object]], payload["streams"])
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(streams) != 2 or len(video) != 1 or len(audio) != 1:
        raise ValueError("Media must contain exactly one video and one audio stream")
    video_stream = video[0]
    audio_stream = audio[0]
    frame_rate = Fraction(str(video_stream["avg_frame_rate"]))
    if frame_rate != 24:
        raise ValueError("Video frame rate must be 24 fps")
    format_data = cast(dict[str, object], payload["format"])
    values = {
        "duration_ms": _milliseconds(format_data["duration"]),
        "video_duration_ms": _milliseconds(video_stream["duration"]),
        "audio_duration_ms": _milliseconds(audio_stream["duration"]),
        "video_start_ms": _milliseconds(video_stream["start_time"], allow_zero=True),
        "audio_start_ms": _milliseconds(audio_stream["start_time"], allow_zero=True),
        "video_frames": int(str(video_stream["nb_read_frames"])),
        "audio_frames": int(str(audio_stream["nb_read_frames"])),
        "video_codec": video_stream["codec_name"],
        "width": video_stream["width"],
        "height": video_stream["height"],
        "frame_rate": "24/1",
        "pixel_format": video_stream["pix_fmt"],
        "audio_codec": audio_stream["codec_name"],
        "channels": audio_stream["channels"],
        "sample_rate_hz": int(str(audio_stream["sample_rate"])),
    }
    return MediaProbe.model_validate(values, strict=True)


def _milliseconds(value: object, *, allow_zero: bool = False) -> int:
    seconds = Decimal(str(value))
    if not seconds.is_finite() or seconds < 0 or (seconds == 0 and not allow_zero):
        raise ValueError("Media timestamp or duration is unusable")
    return int((seconds * 1000).quantize(Decimal("1")))


def _run(arguments: Sequence[str], *, capture_output: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(arguments),
        check=True,
        capture_output=capture_output,
        env={**os.environ, "LC_ALL": "C"},
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=MEDIA_PROCESS_TIMEOUT_SECONDS,
    )


def _matching_attempt(lease: SlotLease, candidate: CandidateReady) -> GenerationAttemptReference:
    matches = tuple(
        attempt
        for attempt in read_generation_attempts(lease.edition_id)
        if attempt.request.request_id == candidate.request_id
    )
    if len(matches) != 1 or matches[0].stage not in {
        GenerationStage.PROCESSING,
        GenerationStage.ACCEPTED,
    }:
        raise ValueError("Candidate does not match one active generation request")
    attempt = matches[0]
    if (
        attempt.request.story_position != candidate.story_position
        or attempt.request.attempt_index != candidate.attempt_index
        or attempt.response_evidence is None
        or attempt.response_evidence.version_id != candidate.response_artifact_version_id
    ):
        raise ValueError("Candidate does not match its exact generation request")
    return attempt


def _reject_candidate(
    lease: SlotLease,
    candidate: CandidateReady,
    error: MediaValidationError | ResearchObjectIntegrityError | subprocess.TimeoutExpired,
) -> GenerationRetryAvailable | GenerationFailed:
    if isinstance(error, MediaValidationError):
        code = error.code
    elif isinstance(error, ResearchObjectIntegrityError):
        code = "stored_digest_mismatch"
    else:
        code = "media_tool_timeout"
    reason = f"Candidate validation failed: {code}"
    content = canonical_json({"request_id": candidate.request_id, "code": code})
    failure_file = artifact_file(
        artifact_id=f"{candidate.request_id}:failure",
        artifact_kind="video_digest_generation_failure",
        title=f"Generation failure for {candidate.request_id}",
        content=content,
        r2_key=(
            f"news/video-digest/{lease.edition_id}/generation/"
            f"failure-{candidate.request_id}-{sha256(content)}.json"
        ),
        media_type="application/json",
    )
    publish_immutable_r2_objects(((failure_file.r2_key, failure_file.content),))
    checkpoint_generation_failure(
        lease,
        candidate.request_id,
        evidence_file=failure_file,
        cost=candidate.cost,
        recorded_at=_now(),
    )
    if candidate.attempt_index == 1:
        return GenerationFailed(edition_id=lease.edition_id, reason=reason)
    return GenerationRetryAvailable(
        edition_id=lease.edition_id,
        story_position=candidate.story_position,
        reason=reason,
    )


def _accepted_attempts(
    plan: DigestPlan,
    attempts: tuple[GenerationAttemptReference, ...],
) -> tuple[GenerationAttemptReference, ...]:
    accepted = tuple(attempt for attempt in attempts if attempt.stage is GenerationStage.ACCEPTED)
    by_position = {attempt.request.story_position: attempt for attempt in accepted}
    expected = tuple(story.position for story in plan.stories if story.mandatory)
    if len(accepted) != len(expected) or tuple(sorted(by_position)) != expected:
        raise ValueError("Assembly requires exactly one accepted clip per mandatory story")
    ordered = tuple(by_position[position] for position in expected)
    if any(
        attempt.accepted_clip is None or attempt.validation_evidence is None for attempt in ordered
    ):
        raise ValueError("Accepted generation is missing media evidence")
    return ordered


def _parse_loudness(stderr: str) -> LoudnessMeasurement:
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start < 0 or end <= start:
        raise MediaValidationError("loudnorm_analysis_failed", "Loudness analysis is missing")
    try:
        payload = cast(dict[str, object], json.loads(stderr[start : end + 1]))
        values = {
            name: _finite_decimal(payload[name])
            for name in ("input_i", "input_lra", "input_tp", "input_thresh", "target_offset")
        }
        return LoudnessMeasurement.model_validate(values)
    except (InvalidOperation, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise MediaValidationError(
            "loudnorm_analysis_failed", "Loudness analysis is unusable"
        ) from error


def _finite_decimal(value: object) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite() or math.isinf(float(result)):
        raise ValueError("Loudness value must be finite")
    return result


def _decimal_argument(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _now() -> datetime:
    return datetime.now(UTC)
