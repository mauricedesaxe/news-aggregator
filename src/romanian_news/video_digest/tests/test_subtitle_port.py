from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file, canonical_json, sha256
from romanian_news.video_digest import subtitle_port
from romanian_news.video_digest.errors import VideoDigestLeaseLostError
from romanian_news.video_digest.media import (
    AcceptedClipManifestEntry,
    AssemblyManifest,
    LoudnessMeasurement,
    MediaProbe,
    SubtitleFailureEvidence,
    SubtitleStrategy,
    SubtitleTimeSpan,
    SubtitleTimingRequest,
)
from romanian_news.video_digest.models import SlotLease
from romanian_news.video_digest.orchestration import (
    ActionAdvanced,
    AssemblyAttemptReference,
    MediaArtifactReference,
    SubtitleAction,
    SubtitleAttemptReference,
)
from romanian_news.video_digest.tests.test_media import _subtitle_inputs


def _reference(file: ArtifactFile) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=file.artifact_id,
        version_id=file.version_id,
        content_digest=file.content_digest,
        r2_key=file.r2_key,
    )


def _assembly(
    edition_id: str, *, clip_count: int = 2
) -> tuple[ArtifactFile, ArtifactFile, AssemblyAttemptReference]:
    video = artifact_file(
        artifact_id=f"{edition_id}:assembled-video",
        artifact_kind="video_digest_assembled_video",
        title="Assembled video digest",
        content=b"accepted assembled video",
        r2_key=f"news/video-digest/{edition_id}/assembled/video.mp4",
        media_type="video/mp4",
    )
    probe = MediaProbe(
        duration_ms=clip_count * 1000,
        video_duration_ms=clip_count * 1000,
        audio_duration_ms=clip_count * 1000,
        video_start_ms=0,
        audio_start_ms=0,
        video_frames=clip_count * 24,
        audio_frames=63,
        video_codec="h264",
        width=1344,
        height=768,
        frame_rate="24/1",
        pixel_format="yuv420p",
        audio_codec="aac",
        channels=2,
        sample_rate_hz=32000,
    )
    clips = tuple(
        AcceptedClipManifestEntry(
            story_position=position,
            request_id=f"request-{position}",
            artifact_version_id="a" * 64,
            content_digest="b" * 64,
            byte_size=100,
            duration_ms=1000,
        )
        for position in range(clip_count)
    )
    manifest = AssemblyManifest(
        edition_id=edition_id,
        clips=clips,
        loudness_measurement=LoudnessMeasurement(
            input_i=Decimal("-20"),
            input_lra=Decimal("6"),
            input_tp=Decimal("-3"),
            input_thresh=Decimal("-30"),
            target_offset=Decimal("4"),
        ),
        expected_duration_ms=clip_count * 1000,
        output_content_digest=video.content_digest,
        output_byte_size=len(video.content),
        output_probe=probe,
    )
    manifest_content = canonical_json(manifest.model_dump(mode="json"))
    manifest_file = artifact_file(
        artifact_id=f"{edition_id}:assembly-manifest",
        artifact_kind="video_digest_assembly_manifest",
        title="Video digest assembly manifest",
        content=manifest_content,
        r2_key=f"news/video-digest/{edition_id}/assembled/manifest.json",
        media_type="application/json",
    )
    attempt = AssemblyAttemptReference(
        attempt_index=0,
        disposition="succeeded",
        evidence=_reference(manifest_file),
        video=MediaArtifactReference(
            **_reference(video).model_dump(),
            byte_size=len(video.content),
            media_type=video.media_type,
        ),
        manifest=_reference(manifest_file),
    )
    return video, manifest_file, attempt


class FakeTimingProvider:
    def __init__(self, before_transcribe: Callable[[], None], *, fail: bool) -> None:
        self._before_transcribe = before_transcribe
        self.fail = fail
        self.strategies: list[SubtitleStrategy] = []

    def timings(
        self,
        strategy: SubtitleStrategy,
        requests: tuple[SubtitleTimingRequest, ...],
        media_path: Path,
    ) -> tuple[tuple[SubtitleTimeSpan, ...], ...]:
        assert media_path.read_bytes() == b"accepted assembled video"
        self._before_transcribe()
        self.strategies.append(strategy)
        if self.fail:
            raise RuntimeError("timing unavailable")
        return tuple(
            tuple(
                SubtitleTimeSpan(
                    start_ms=request.start_ms + index * 100,
                    end_ms=request.start_ms + (index + 1) * 100,
                )
                for index in range(request.cue_count)
            )
            for request in requests
        )


def _setup(
    monkeypatch: pytest.MonkeyPatch, *, clip_count: int = 2
) -> tuple[SlotLease, list[SubtitleAttemptReference], dict[str, bytes]]:
    lease, screenplay, _assembled = _subtitle_inputs()
    video, manifest, assembly_attempt = _assembly(lease.edition_id, clip_count=clip_count)
    objects = {video.r2_key: video.content, manifest.r2_key: manifest.content}
    attempts: list[SubtitleAttemptReference] = []

    def read_object(key: str, digest: str) -> bytes:
        content = objects[key]
        assert sha256(content) == digest
        return content

    def publish(values: tuple[tuple[str, bytes], ...]) -> None:
        objects.update(values)

    def record(
        _lease: SlotLease,
        index: int,
        strategy: Literal["whole-edition-v1", "per-story-v1", "per-story-without-vad-v1"],
        disposition: Literal["failed", "succeeded"],
        *,
        evidence_file: ArtifactFile,
        subtitle_file: ArtifactFile | None = None,
        recorded_at: object,
    ) -> SubtitleAttemptReference:
        assert _lease == lease
        assert index == len(attempts)
        assert recorded_at is not None
        attempt = SubtitleAttemptReference(
            attempt_index=index,
            strategy=strategy,
            disposition=disposition,
            evidence=_reference(evidence_file),
            subtitle=(
                MediaArtifactReference(
                    **_reference(subtitle_file).model_dump(),
                    byte_size=len(subtitle_file.content),
                    media_type=subtitle_file.media_type,
                )
                if subtitle_file is not None
                else None
            ),
        )
        attempts.append(attempt)
        return attempt

    monkeypatch.setattr(subtitle_port, "read_subtitle_attempts", lambda _edition: tuple(attempts))
    monkeypatch.setattr(
        subtitle_port, "read_assembly_attempts", lambda _edition: (assembly_attempt,)
    )
    monkeypatch.setattr(
        subtitle_port,
        "read_accepted_digest_plan",
        lambda _edition: (SimpleNamespace(plan=screenplay), None),
    )
    monkeypatch.setattr(subtitle_port, "read_verified_r2_object", read_object)
    monkeypatch.setattr(subtitle_port, "publish_immutable_r2_objects", publish)
    monkeypatch.setattr(subtitle_port, "record_subtitle_attempt", record)
    return lease, attempts, objects


def test_one_strategy_records_subtitles_and_preserves_approved_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, attempts, objects = _setup(monkeypatch)
    renewals = []
    providers: list[FakeTimingProvider] = []

    def provider_factory(before_transcribe: Callable[[], None]) -> FakeTimingProvider:
        provider = FakeTimingProvider(before_transcribe, fail=False)
        providers.append(provider)
        return provider

    def renew(current: SlotLease) -> SlotLease:
        renewals.append(current)
        return current

    port = subtitle_port.ResumableSubtitlePort(provider_factory=provider_factory, renew_lease=renew)
    action = SubtitleAction(lease=lease, attempt_index=0, strategy="whole-edition-v1")

    assert isinstance(port.execute(action), ActionAdvanced)
    assert len(renewals) == 4
    assert providers[0].strategies == [SubtitleStrategy.WHOLE_EDITION]
    assert attempts[0].disposition == "succeeded"
    assert attempts[0].subtitle is not None
    subtitles = objects[attempts[0].subtitle.r2_key].decode()
    assert subtitles.startswith("WEBVTT\n")
    assert "Acesta este textul aprobat" in subtitles
    assert "Al doilea text aprobat" in subtitles
    assert isinstance(port.execute(action), ActionAdvanced)
    assert len(providers) == 1


def test_three_durable_failures_publish_one_complete_failure_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, attempts, objects = _setup(monkeypatch)
    providers: list[FakeTimingProvider] = []

    def provider_factory(before_transcribe: Callable[[], None]) -> FakeTimingProvider:
        provider = FakeTimingProvider(before_transcribe, fail=True)
        providers.append(provider)
        return provider

    port = subtitle_port.ResumableSubtitlePort(
        provider_factory=provider_factory, renew_lease=lambda current: current
    )
    for index, strategy in enumerate(SubtitleStrategy):
        action = SubtitleAction.model_validate(
            {"lease": lease, "attempt_index": index, "strategy": strategy.value}
        )
        assert isinstance(port.execute(action), ActionAdvanced)
        assert len(attempts) == index + 1
        assert providers[-1].strategies == [strategy]

    assert [attempt.disposition for attempt in attempts] == ["failed"] * 3
    final = SubtitleFailureEvidence.model_validate_json(
        objects[attempts[-1].evidence.r2_key], strict=True
    )
    assert final.edition_id == lease.edition_id
    assert [failure.strategy for failure in final.attempts] == list(SubtitleStrategy)
    assert [failure.code for failure in final.attempts] == ["RuntimeError"] * 3


def test_lost_lease_during_transcription_does_not_record_subtitle_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, attempts, _objects = _setup(monkeypatch)
    renewal_count = 0

    def renew(current: SlotLease) -> SlotLease:
        nonlocal renewal_count
        renewal_count += 1
        if renewal_count == 3:
            raise VideoDigestLeaseLostError("lease lost")
        return current

    port = subtitle_port.ResumableSubtitlePort(
        provider_factory=lambda before: FakeTimingProvider(before, fail=False),
        renew_lease=renew,
    )

    with pytest.raises(VideoDigestLeaseLostError):
        port.execute(SubtitleAction(lease=lease, attempt_index=0, strategy="whole-edition-v1"))
    assert attempts == []


def test_rehydration_rejects_corrupt_accepted_video_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, attempts, objects = _setup(monkeypatch)
    video_key = next(key for key in objects if key.endswith("video.mp4"))
    objects[video_key] = b"different video"
    called = False

    def provider_factory(_before: Callable[[], None]) -> FakeTimingProvider:
        nonlocal called
        called = True
        return FakeTimingProvider(_before, fail=False)

    port = subtitle_port.ResumableSubtitlePort(
        provider_factory=provider_factory, renew_lease=lambda current: current
    )

    with pytest.raises(AssertionError):
        port.execute(SubtitleAction(lease=lease, attempt_index=0, strategy="whole-edition-v1"))
    assert not called
    assert attempts == []


def test_accepted_assembly_must_cover_every_approved_story(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, attempts, _objects = _setup(monkeypatch, clip_count=1)
    called = False

    def provider_factory(before: Callable[[], None]) -> FakeTimingProvider:
        nonlocal called
        called = True
        return FakeTimingProvider(before, fail=False)

    port = subtitle_port.ResumableSubtitlePort(
        provider_factory=provider_factory, renew_lease=lambda current: current
    )

    with pytest.raises(ValueError, match="does not cover the screenplay"):
        port.execute(SubtitleAction(lease=lease, attempt_index=0, strategy="whole-edition-v1"))
    assert not called
    assert attempts == []
