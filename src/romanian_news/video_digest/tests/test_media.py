from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file
from romanian_news.catalog.video_digest import AcceptedClipReference, GenerationAttemptReference
from romanian_news.identity import sha256
from romanian_news.storage import ResearchObjectIntegrityError
from romanian_news.video_digest import media, narration_quality
from romanian_news.video_digest.generation import CandidateReady, CandidateReference
from romanian_news.video_digest.models import (
    AvailableSubtitles,
    DigestPlan,
    EditionId,
    FailedSubtitles,
    GenerationRequestIdentity,
    GenerationStage,
    PlannedStory,
    SlotId,
    SlotLease,
    UnknownAttemptCost,
    edition_id,
    generation_request_id,
    planned_story_id,
)
from romanian_news.video_digest.narration_quality import NarrationCheckError
from romanian_news.video_digest.planning import ScreenplayPlan, ScreenplayStory

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg and ffprobe are required")


def _accept_synthetic_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media, "clip_narration_matches", lambda _path, _text: True)


def _accept_candidate(lease: SlotLease, candidate: CandidateReady, story: PlannedStory):
    return media.accept_candidate(
        lease, candidate, story, approved_narration="Approved English narration"
    )


def _synthetic_clip(
    path: Path, color: str, *, size: str = "1344x768", duration: str = "1"
) -> bytes:
    subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s={size}:r=24:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=32000:duration={duration}",
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-ar",
            "32000",
            "-movflags",
            "+faststart",
            str(path),
        ),
        check=True,
    )
    return path.read_bytes()


def _probe_payload(
    *,
    frame_rate: str = "24/1",
    pixel_format: str = "yuv420p",
    duration: str = "1.0",
    video_duration: str | None = None,
    audio_duration: str | None = None,
) -> dict[str, object]:
    return {
        "streams": [
            {
                "codec_type": "video",
                "avg_frame_rate": frame_rate,
                "duration": video_duration or duration,
                "start_time": "0",
                "nb_read_frames": "24",
                "codec_name": "h264",
                "width": 1344,
                "height": 768,
                "pix_fmt": pixel_format,
            },
            {
                "codec_type": "audio",
                "duration": audio_duration or duration,
                "start_time": "0",
                "nb_read_frames": "32",
                "codec_name": "aac",
                "channels": 2,
                "sample_rate": "32000",
            },
        ],
        "format": {"duration": duration},
    }


def _story(edition: EditionId, position: int) -> PlannedStory:
    subject = str(position + 1) * 64
    return PlannedStory(
        story_id=planned_story_id(edition, position, subject),
        edition_id=edition,
        position=position,
        report_subject_id=subject,
        title=f"Story {position}",
        requested_duration_ms=1000,
    )


def _lease(edition: EditionId) -> SlotLease:
    from datetime import UTC, datetime, timedelta

    return SlotLease(
        slot_id=SlotId("f" * 64),
        edition_id=edition,
        owner_token="media-test",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        claim_count=1,
    )


def _attempt(story: PlannedStory, content: bytes, name: str) -> GenerationAttemptReference:
    request_version = sha256(f"request-{name}".encode())
    request_id = generation_request_id(story.edition_id, story.position, 0, request_version)
    reference = AcceptedClipReference(
        artifact_id=f"{story.story_id}:accepted-clip",
        version_id=sha256(f"clip-version-{name}".encode()),
        content_digest=sha256(content),
        r2_key=f"clips/{name}.mp4",
        byte_size=len(content),
    )
    return GenerationAttemptReference(
        request=GenerationRequestIdentity(
            request_id=request_id,
            edition_id=story.edition_id,
            story_position=story.position,
            attempt_index=0,
            request_artifact_version_id=request_version,
        ),
        stage=GenerationStage.ACCEPTED,
        provider_receipt_id=f"fal-{name}",
        cost=UnknownAttemptCost(reason="test fixture"),
        request_evidence=ArtifactReference(
            artifact_id=f"request-{name}",
            version_id=request_version,
            content_digest=sha256(f"request-content-{name}".encode()),
            r2_key=f"requests/{name}.json",
        ),
        receipt_evidence=None,
        response_evidence=None,
        accepted_clip=reference,
        validation_evidence=ArtifactReference(
            artifact_id=f"{request_id}:validation",
            version_id=sha256(f"validation-version-{name}".encode()),
            content_digest=sha256(f"validation-content-{name}".encode()),
            r2_key=f"validation/{name}.json",
        ),
    )


@requires_ffmpeg
def test_technical_validation_accepts_exact_profile_and_full_decode(tmp_path: Path) -> None:
    path = tmp_path / "candidate.mp4"
    content = _synthetic_clip(path, "red")

    probe = media.validate_media_file(
        path,
        expected_digest=sha256(content),
        expected_size=len(content),
        requested_duration_ms=1000,
    )

    assert probe.video_codec == "h264"
    assert (probe.width, probe.height, probe.frame_rate) == (1344, 768, "24/1")
    assert (probe.audio_codec, probe.channels, probe.sample_rate_hz) == ("aac", 2, 32000)
    assert probe.video_frames > 0
    assert probe.audio_frames > 0


@requires_ffmpeg
def test_silencing_unapproved_tail_preserves_clip_profile(tmp_path: Path) -> None:
    source = tmp_path / "candidate.mp4"
    original = _synthetic_clip(source, "red")
    accepted = tmp_path / "accepted.mp4"

    media._silence_unapproved_tail(source, accepted, 500)

    content = accepted.read_bytes()
    assert content != original
    probe = media.validate_media_file(
        accepted,
        expected_digest=sha256(content),
        expected_size=len(content),
        requested_duration_ms=1000,
    )
    assert probe.audio_codec == "aac"
    assert probe.sample_rate_hz == 32000


@requires_ffmpeg
def test_technical_validation_rejects_profile_duration_digest_and_size(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.mp4"
    valid = _synthetic_clip(valid_path, "red")
    wrong_profile_path = tmp_path / "wrong-profile.mp4"
    wrong_profile = _synthetic_clip(wrong_profile_path, "red", size="1280x720")

    with pytest.raises(media.MediaValidationError, match="unusable"):
        media.validate_media_file(
            wrong_profile_path,
            expected_digest=sha256(wrong_profile),
            expected_size=len(wrong_profile),
            requested_duration_ms=1000,
        )
    with pytest.raises(media.MediaValidationError, match="duration"):
        media.validate_media_file(
            valid_path,
            expected_digest=sha256(valid),
            expected_size=len(valid),
            requested_duration_ms=1200,
        )
    with pytest.raises(media.MediaValidationError, match="digest"):
        media.validate_media_file(
            valid_path,
            expected_digest="0" * 64,
            expected_size=len(valid),
            requested_duration_ms=1000,
        )
    with pytest.raises(media.MediaValidationError, match="size"):
        media.validate_media_file(
            valid_path,
            expected_digest=sha256(valid),
            expected_size=len(valid) + 1,
            requested_duration_ms=1000,
        )


@requires_ffmpeg
@pytest.mark.parametrize("audio_duration", ("15.104", "15.150"))
def test_technical_validation_allows_one_aac_frame_of_tail_padding(
    tmp_path: Path,
    audio_duration: str,
) -> None:
    path = tmp_path / "candidate.mp4"
    subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=1344x768:r=24:d=15.083333",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=32000:duration={audio_duration}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-ar",
            "32000",
            "-movflags",
            "+faststart",
            str(path),
        ),
        check=True,
    )
    content = path.read_bytes()

    if audio_duration == "15.104":
        probe = media.validate_media_file(
            path,
            expected_digest=sha256(content),
            expected_size=len(content),
            requested_duration_ms=15_000,
        )
        assert probe.audio_duration_ms > probe.video_duration_ms
    else:
        with pytest.raises(media.MediaValidationError) as captured:
            media.validate_media_file(
                path,
                expected_digest=sha256(content),
                expected_size=len(content),
                requested_duration_ms=15_000,
            )
        assert captured.value.code == "duration_mismatch"


def test_technical_validation_rejects_indeterminate_frame_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.mp4"
    content = b"candidate"
    path.write_bytes(content)
    monkeypatch.setattr(
        media,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            (), 0, json.dumps(_probe_payload(frame_rate="0/0")), ""
        ),
    )

    with pytest.raises(media.MediaValidationError) as captured:
        media.validate_media_file(
            path,
            expected_digest=sha256(content),
            expected_size=len(content),
            requested_duration_ms=1000,
        )

    assert captured.value.code == "unusable_probe"


def test_technical_validation_rejects_unsupported_pixel_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.mp4"
    content = b"candidate"
    path.write_bytes(content)
    monkeypatch.setattr(
        media,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            (), 0, json.dumps(_probe_payload(pixel_format="yuv444p")), ""
        ),
    )

    with pytest.raises(media.MediaValidationError) as captured:
        media.validate_media_file(
            path,
            expected_digest=sha256(content),
            expected_size=len(content),
            requested_duration_ms=1000,
        )

    assert captured.value.code == "unusable_probe"


def test_technical_validation_rejects_unavailable_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.mp4"
    content = b"candidate"
    path.write_bytes(content)
    monkeypatch.setattr(
        media,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            (), 0, json.dumps(_probe_payload(duration="N/A")), ""
        ),
    )

    with pytest.raises(media.MediaValidationError) as captured:
        media.validate_media_file(
            path,
            expected_digest=sha256(content),
            expected_size=len(content),
            requested_duration_ms=1000,
        )

    assert captured.value.code == "unusable_probe"


@pytest.mark.parametrize(
    ("failing_program", "expected_code"),
    (("ffprobe", "probe_failed"), ("ffmpeg", "full_decode_failed")),
)
def test_technical_validation_identifies_failed_media_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_program: str,
    expected_code: str,
) -> None:
    path = tmp_path / "candidate.mp4"
    content = b"candidate"
    path.write_bytes(content)

    def run(arguments, *, capture_output):
        del capture_output
        if arguments[0] == failing_program:
            raise subprocess.CalledProcessError(1, arguments)
        return subprocess.CompletedProcess((), 0, json.dumps(_probe_payload()), "")

    monkeypatch.setattr(media, "_run", run)

    with pytest.raises(media.MediaValidationError) as captured:
        media.validate_media_file(
            path,
            expected_digest=sha256(content),
            expected_size=len(content),
            requested_duration_ms=1000,
        )

    assert captured.value.code == expected_code


@requires_ffmpeg
def test_candidate_bytes_publish_before_atomic_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.mp4"
    content = _synthetic_clip(path, "red")
    edition = EditionId("e" * 64)
    story = _story(edition, 0)
    attempt = _attempt(story, content, "red")
    assert attempt.response_evidence is None
    response = ArtifactReference(
        artifact_id=f"{attempt.request.request_id}:response",
        version_id=sha256(b"response-version"),
        content_digest=sha256(b"response"),
        r2_key="responses/red.json",
    )
    attempt = attempt.model_copy(
        update={"stage": GenerationStage.PROCESSING, "response_evidence": response}
    )
    candidate = CandidateReady(
        request_id=attempt.request.request_id,
        story_id=story.story_id,
        story_position=0,
        attempt_index=0,
        response_artifact_version_id=response.version_id,
        candidate=CandidateReference(
            r2_key="candidates/red.mp4",
            content_digest=sha256(content),
            byte_size=len(content),
        ),
        cost=UnknownAttemptCost(reason="test fixture"),
    )
    events: list[str] = []
    published: dict[str, bytes] = {}
    _accept_synthetic_audio(monkeypatch)
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: (attempt,))
    monkeypatch.setattr(media, "read_verified_r2_object", lambda _key, _digest: content)

    def publish(objects) -> None:
        events.append("evidence")
        published.update(objects)

    def publish_clip(key: str, content: bytes, *, retention: str, source_lineage: str) -> None:
        events.append("private")
        assert retention == "permanent"
        assert source_lineage == sha256(
            f"{story.story_id}:accepted-clip\0{sha256(content)}".encode()
        )
        published[key] = content

    def checkpoint(*_args, validation_file, **_kwargs) -> None:
        events.append("checkpoint")
        evidence = json.loads(validation_file.content)
        assert evidence["request_id"] == candidate.request_id
        assert (
            evidence["request_artifact_version_id"] == attempt.request.request_artifact_version_id
        )

    monkeypatch.setattr(media, "publish_immutable_r2_objects", publish)
    monkeypatch.setattr(media, "publish_private_video_object", publish_clip)
    monkeypatch.setattr(media, "checkpoint_generation_acceptance", checkpoint)

    result = _accept_candidate(_lease(edition), candidate, story)

    assert isinstance(result, media.AcceptedCandidate)
    assert events == ["private", "evidence", "checkpoint"]
    assert any(key.endswith(".mp4") for key in published)

    accepted_attempt = attempt.model_copy(update={"stage": GenerationStage.ACCEPTED})
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: (accepted_attempt,))
    monkeypatch.setattr(
        media,
        "read_verified_r2_object",
        lambda *_args: pytest.fail("accepted candidates must replay stored evidence"),
    )

    replayed = _accept_candidate(_lease(edition), candidate, story)

    assert accepted_attempt.accepted_clip is not None
    assert accepted_attempt.validation_evidence is not None
    assert replayed == media.AcceptedCandidate(
        clip_artifact_version_id=accepted_attempt.accepted_clip.version_id,
        validation_artifact_version_id=accepted_attempt.validation_evidence.version_id,
    )

    events.clear()

    def corrupt_read(_key: str, _digest: str) -> bytes:
        raise ResearchObjectIntegrityError("digest mismatch")

    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: (attempt,))
    monkeypatch.setattr(media, "read_verified_r2_object", corrupt_read)
    monkeypatch.setattr(
        media,
        "checkpoint_generation_failure",
        lambda *_args, **_kwargs: events.append("failure-checkpoint"),
    )

    rejected = _accept_candidate(_lease(edition), candidate, story)

    assert isinstance(rejected, media.GenerationRetryAvailable)
    assert events == ["evidence", "failure-checkpoint"]


def _processing_attempt(
    story: PlannedStory, content: bytes, name: str
) -> tuple[GenerationAttemptReference, CandidateReady]:
    attempt = _attempt(story, content, name)
    response = ArtifactReference(
        artifact_id=f"{attempt.request.request_id}:response",
        version_id=sha256(f"response-version-{name}".encode()),
        content_digest=sha256(f"response-content-{name}".encode()),
        r2_key=f"responses/{name}.json",
    )
    attempt = attempt.model_copy(
        update={"stage": GenerationStage.PROCESSING, "response_evidence": response}
    )
    candidate = CandidateReady(
        request_id=attempt.request.request_id,
        story_id=story.story_id,
        story_position=story.position,
        attempt_index=0,
        response_artifact_version_id=response.version_id,
        candidate=CandidateReference(
            r2_key=f"candidates/{name}.mp4",
            content_digest=sha256(content),
            byte_size=len(content),
        ),
        cost=UnknownAttemptCost(reason="test fixture"),
    )
    return attempt, candidate


@requires_ffmpeg
def test_candidate_with_prompt_speech_after_narration_is_silenced_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _synthetic_clip(tmp_path / "candidate.mp4", "red")
    edition = EditionId("e" * 64)
    story = _story(edition, 0)
    attempt, candidate = _processing_attempt(story, content, "tail-speech")
    published: dict[str, bytes] = {}
    checkpointed: dict[str, ArtifactFile] = {}
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: (attempt,))
    monkeypatch.setattr(media, "read_verified_r2_object", lambda _key, _digest: content)
    narration_check_paths: list[str] = []

    def check_narration(path: Path, _text: str) -> bool:
        narration_check_paths.append(path.name)
        return path.name == "accepted.mp4"

    monkeypatch.setattr(media, "clip_narration_matches", check_narration)
    monkeypatch.setattr(media, "clip_narration_tail_cutoff_ms", lambda _path, _text: 500)
    monkeypatch.setattr(
        media,
        "publish_private_video_object",
        lambda key, value, **_kwargs: published.update({key: value}),
    )
    monkeypatch.setattr(
        media,
        "publish_immutable_r2_objects",
        lambda objects: published.update(objects),
    )

    def checkpoint(
        *_args: object, clip_file: ArtifactFile, validation_file: ArtifactFile, **_kwargs: object
    ) -> None:
        checkpointed.update(clip=clip_file, validation=validation_file)

    monkeypatch.setattr(media, "checkpoint_generation_acceptance", checkpoint)

    result = _accept_candidate(_lease(edition), candidate, story)

    assert isinstance(result, media.AcceptedCandidate)
    assert narration_check_paths == ["candidate.mp4", "accepted.mp4"]
    clip_key = next(key for key in published if key.endswith("-silenced.mp4"))
    assert published[clip_key] != content
    validation_key = next(key for key in published if "/validation-" in key)
    evidence = media.CandidateValidationEvidence.model_validate_json(
        published[validation_key], strict=True
    )
    assert evidence.muted_after_ms == 500
    assert evidence.accepted_clip_content_digest == sha256(published[clip_key])
    assert evidence.candidate_content_digest == sha256(content)

    clip = checkpointed["clip"]
    validation = checkpointed["validation"]
    accepted_attempt = attempt.model_copy(
        update={
            "stage": GenerationStage.ACCEPTED,
            "accepted_clip": AcceptedClipReference(
                artifact_id=clip.artifact_id,
                version_id=clip.version_id,
                content_digest=clip.content_digest,
                r2_key=clip.r2_key,
                byte_size=len(clip.content),
            ),
            "validation_evidence": ArtifactReference(
                artifact_id=validation.artifact_id,
                version_id=validation.version_id,
                content_digest=validation.content_digest,
                r2_key=validation.r2_key,
            ),
        }
    )
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: (accepted_attempt,))
    monkeypatch.setattr(media, "read_verified_r2_object", lambda key, _digest: published[key])
    tampered = json.loads(published[validation_key])
    tampered["request_id"] = "tampered-request"
    monkeypatch.setattr(
        media,
        "read_verified_r2_object",
        lambda _key, _digest: json.dumps(tampered).encode(),
    )
    with pytest.raises(ValueError, match="does not match stored media evidence"):
        _accept_candidate(_lease(edition), candidate, story)
    monkeypatch.setattr(media, "read_verified_r2_object", lambda key, _digest: published[key])
    assert _accept_candidate(_lease(edition), candidate, story) == result


@requires_ffmpeg
def test_candidate_that_still_mismatches_after_silencing_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _synthetic_clip(tmp_path / "candidate.mp4", "red")
    edition = EditionId("e" * 64)
    story = _story(edition, 0)
    attempt, candidate = _processing_attempt(story, content, "unfixable-speech")
    events: list[str] = []
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: (attempt,))
    monkeypatch.setattr(media, "read_verified_r2_object", lambda _key, _digest: content)
    monkeypatch.setattr(media, "clip_narration_matches", lambda _path, _text: False)
    monkeypatch.setattr(media, "clip_narration_tail_cutoff_ms", lambda _path, _text: 500)
    monkeypatch.setattr(
        media,
        "publish_private_video_object",
        lambda *_args, **_kwargs: pytest.fail("a still-mismatching clip was accepted"),
    )
    monkeypatch.setattr(
        media,
        "publish_immutable_r2_objects",
        lambda _objects: events.append("failure-evidence"),
    )
    monkeypatch.setattr(
        media,
        "checkpoint_generation_failure",
        lambda *_args, **_kwargs: events.append("failure-checkpoint"),
    )

    result = _accept_candidate(_lease(edition), candidate, story)

    assert isinstance(result, media.GenerationRetryAvailable)
    assert "narration_mismatch" in result.reason
    assert events == ["failure-evidence", "failure-checkpoint"]


@requires_ffmpeg
@pytest.mark.parametrize(
    ("check_result", "expected_code"),
    [
        (False, "narration_mismatch"),
        (NarrationCheckError("transcription failed"), "narration_check_failed"),
    ],
)
def test_candidate_narration_failure_is_recorded_before_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    check_result: bool | NarrationCheckError,
    expected_code: str,
) -> None:
    content = _synthetic_clip(tmp_path / "candidate.mp4", "red")
    edition = EditionId("e" * 64)
    story = _story(edition, 0)
    attempt, candidate = _processing_attempt(story, content, "spoken-prompt")
    events: list[str] = []
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: (attempt,))
    monkeypatch.setattr(media, "read_verified_r2_object", lambda _key, _digest: content)

    def check_narration(_path: Path, _text: str) -> bool:
        if isinstance(check_result, NarrationCheckError):
            raise check_result
        return check_result

    monkeypatch.setattr(media, "clip_narration_matches", check_narration)
    monkeypatch.setattr(media, "clip_narration_tail_cutoff_ms", lambda _path, _text: None)
    monkeypatch.setattr(
        media,
        "publish_private_video_object",
        lambda *_args, **_kwargs: pytest.fail("bad narration was accepted"),
    )
    monkeypatch.setattr(
        media,
        "publish_immutable_r2_objects",
        lambda _objects: events.append("failure-evidence"),
    )
    monkeypatch.setattr(
        media,
        "checkpoint_generation_failure",
        lambda *_args, **_kwargs: events.append("failure-checkpoint"),
    )

    result = media.accept_candidate(
        _lease(edition), candidate, story, approved_narration="Approved English line"
    )

    assert isinstance(result, media.GenerationRetryAvailable)
    assert result.reason == f"Candidate validation failed: {expected_code}"
    assert events == ["failure-evidence", "failure-checkpoint"]


def _fake_narration_check_stdout(monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    # the real check shells out to a local Whisper model; fake its single subprocess seam
    monkeypatch.setattr(narration_quality, "_run_narration_check", lambda *_args, **_kwargs: stdout)


def test_tail_cutoff_stdout_mismatch_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_narration_check_stdout(monkeypatch, "mismatch")

    result = narration_quality.clip_narration_tail_cutoff_ms(tmp_path / "clip.mp4", "Approved")

    assert result is None


@pytest.mark.parametrize("stdout", ["banana", "0", "-5"])
def test_tail_cutoff_rejects_non_integer_or_non_positive_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    _fake_narration_check_stdout(monkeypatch, stdout)

    with pytest.raises(NarrationCheckError):
        narration_quality.clip_narration_tail_cutoff_ms(tmp_path / "clip.mp4", "Approved")


def _media_probe() -> media.MediaProbe:
    return media.MediaProbe(
        duration_ms=1000,
        video_duration_ms=1000,
        audio_duration_ms=1000,
        video_start_ms=0,
        audio_start_ms=0,
        video_frames=24,
        audio_frames=32,
        video_codec="h264",
        width=1344,
        height=768,
        frame_rate="24/1",
        pixel_format="yuv420p",
        audio_codec="aac",
        channels=2,
        sample_rate_hz=32000,
    )


def _evidence_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "request_id": "request",
        "request_artifact_version_id": "1" * 64,
        "response_artifact_version_id": "2" * 64,
        "story_id": "story",
        "story_position": 0,
        "attempt_index": 0,
        "requested_duration_ms": 1000,
        "candidate_r2_key": "candidates/clip.mp4",
        "candidate_content_digest": "3" * 64,
        "candidate_byte_size": 100,
        "accepted_clip_content_digest": "3" * 64,
        "accepted_clip_byte_size": 100,
        "probe": _media_probe(),
    }
    values.update(overrides)
    return values


def test_v2_candidate_evidence_requires_accepted_clip_identity() -> None:
    assert media.CandidateValidationEvidence.model_validate(_evidence_values())

    with pytest.raises(ValueError, match="requires accepted clip identity"):
        media.CandidateValidationEvidence.model_validate(
            _evidence_values(accepted_clip_content_digest=None, accepted_clip_byte_size=None)
        )
    with pytest.raises(ValueError, match="requires accepted clip identity"):
        media.CandidateValidationEvidence.model_validate(
            _evidence_values(accepted_clip_byte_size=None)
        )


def test_unchanged_candidate_evidence_must_retain_its_original_identity() -> None:
    with pytest.raises(ValueError, match="must retain its original identity"):
        media.CandidateValidationEvidence.model_validate(
            _evidence_values(accepted_clip_content_digest="4" * 64)
        )

    assert media.CandidateValidationEvidence.model_validate(
        _evidence_values(muted_after_ms=500, accepted_clip_content_digest="4" * 64)
    )


def test_byte_identical_candidates_share_stable_accepted_object_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edition = EditionId("e" * 64)
    story = _story(edition, 0)
    content = b"identical candidate bytes"
    attempts = tuple(_processing_attempt(story, content, name) for name in ("first", "second"))
    active: list[GenerationAttemptReference] = []
    stored: dict[str, tuple[bytes, str]] = {}
    accepted_requests: list[str] = []
    probe = media.MediaProbe(
        duration_ms=1000,
        video_duration_ms=1000,
        audio_duration_ms=1000,
        video_start_ms=0,
        audio_start_ms=0,
        video_frames=24,
        audio_frames=32,
        video_codec="h264",
        width=1344,
        height=768,
        frame_rate="24/1",
        pixel_format="yuv420p",
        audio_codec="aac",
        channels=2,
        sample_rate_hz=32000,
    )
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: tuple(active))
    monkeypatch.setattr(media, "read_verified_r2_object", lambda *_args: content)
    monkeypatch.setattr(media, "validate_media_file", lambda *_args, **_kwargs: probe)
    monkeypatch.setattr(media, "publish_immutable_r2_objects", lambda _objects: None)
    _accept_synthetic_audio(monkeypatch)

    def publish_clip(key: str, value: bytes, *, retention: str, source_lineage: str) -> None:
        assert retention == "permanent"
        existing = stored.setdefault(key, (value, source_lineage))
        if existing != (value, source_lineage):
            raise ResearchObjectIntegrityError("object classification differs")

    def checkpoint(_lease, request_id, *, validation_file, **_kwargs) -> None:
        assert json.loads(validation_file.content)["request_id"] == request_id
        accepted_requests.append(request_id)

    monkeypatch.setattr(media, "publish_private_video_object", publish_clip)
    monkeypatch.setattr(media, "checkpoint_generation_acceptance", checkpoint)

    outcomes = []
    for attempt, candidate in attempts:
        active[:] = [attempt]
        outcomes.append(_accept_candidate(_lease(edition), candidate, story))

    assert all(isinstance(outcome, media.AcceptedCandidate) for outcome in outcomes)
    assert accepted_requests == [candidate.request_id for _, candidate in attempts]
    assert accepted_requests[0] != accepted_requests[1]
    assert len(stored) == 1


def test_acceptance_replay_rejects_a_candidate_that_differs_from_stored_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edition = EditionId("e" * 64)
    story = _story(edition, 0)
    content = b"stored candidate"
    attempt, candidate = _processing_attempt(story, content, "red")
    accepted = attempt.model_copy(update={"stage": GenerationStage.ACCEPTED})
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: (accepted,))
    monkeypatch.setattr(
        media,
        "read_verified_r2_object",
        lambda *_args: pytest.fail("an accepted candidate must replay stored evidence"),
    )
    tampered = candidate.model_copy(
        update={
            "candidate": CandidateReference(
                r2_key=candidate.candidate.r2_key,
                content_digest=sha256(b"different bytes"),
                byte_size=len(b"different bytes"),
            )
        }
    )

    with pytest.raises(ValueError, match="does not match stored media evidence"):
        _accept_candidate(_lease(edition), tampered, story)

    with pytest.raises(ValueError, match="does not match stored media evidence"):
        _accept_candidate(
            _lease(edition),
            candidate.model_copy(
                update={
                    "candidate": CandidateReference(
                        r2_key=candidate.candidate.r2_key,
                        content_digest=candidate.candidate.content_digest,
                        byte_size=candidate.candidate.byte_size + 1,
                    )
                }
            ),
            story,
        )


@pytest.mark.parametrize("stage", [GenerationStage.FAILED, GenerationStage.PENDING])
def test_candidate_matching_rejects_inactive_attempts(
    monkeypatch: pytest.MonkeyPatch, stage: GenerationStage
) -> None:
    edition = EditionId("e" * 64)
    story = _story(edition, 0)
    attempt, candidate = _processing_attempt(story, b"candidate", "red")
    attempt = attempt.model_copy(update={"stage": stage})
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: (attempt,))
    monkeypatch.setattr(
        media,
        "read_verified_r2_object",
        lambda *_args: pytest.fail("an inactive attempt must not be read"),
    )

    with pytest.raises(ValueError, match="does not match one active generation request"):
        _accept_candidate(_lease(edition), candidate, story)


def test_candidate_matching_rejects_a_divergent_response_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edition = EditionId("e" * 64)
    story = _story(edition, 0)
    attempt, candidate = _processing_attempt(story, b"candidate", "red")
    divergent = candidate.model_copy(
        update={"response_artifact_version_id": sha256(b"a different response")}
    )
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: (attempt,))
    monkeypatch.setattr(
        media,
        "read_verified_r2_object",
        lambda *_args: pytest.fail("a divergent candidate must not be read"),
    )

    with pytest.raises(ValueError, match="does not match its exact generation request"):
        _accept_candidate(_lease(edition), divergent, story)


def test_media_tool_timeout_fails_the_attempt_for_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edition = EditionId("e" * 64)
    story = _story(edition, 0)
    content = b"candidate"
    attempt, candidate = _processing_attempt(story, content, "red")
    events: list[str] = []
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: (attempt,))
    monkeypatch.setattr(media, "read_verified_r2_object", lambda _key, _digest: content)
    monkeypatch.setattr(media, "publish_immutable_r2_objects", lambda _objects: None)
    monkeypatch.setattr(
        media,
        "checkpoint_generation_failure",
        lambda *_args, evidence_file, **_kwargs: events.append(
            json.loads(evidence_file.content)["code"]
        ),
    )

    monkeypatch.setattr(media, "ffprobe_command", lambda _path: ("sleep", "30"))
    monkeypatch.setattr(media, "MEDIA_PROCESS_TIMEOUT_SECONDS", 1)

    outcome = _accept_candidate(_lease(edition), candidate, story)

    assert isinstance(outcome, media.GenerationRetryAvailable)
    assert "media_tool_timeout" in outcome.reason
    assert events == ["media_tool_timeout"]


@requires_ffmpeg
def test_assembly_preserves_plan_order_and_deterministic_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    red = _synthetic_clip(tmp_path / "red.mp4", "red")
    blue = _synthetic_clip(tmp_path / "blue.mp4", "blue")
    edition = EditionId("e" * 64)
    stories = (_story(edition, 0), _story(edition, 1))
    plan = DigestPlan(edition_id=edition, artifact_version_id="a" * 64, stories=stories)
    attempts = (_attempt(stories[0], red, "red"), _attempt(stories[1], blue, "blue"))
    objects = {"clips/red.mp4": red, "clips/blue.mp4": blue}
    publications: list[dict[str, bytes]] = []
    recorded: list[tuple[int, str, str]] = []
    renewals: list[SlotLease] = []
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: attempts)
    monkeypatch.setattr(media, "read_verified_r2_object", lambda key, _digest: objects[key])
    monkeypatch.setattr(media, "checkpoint_assembly_ready", lambda *_args, **_kwargs: None)
    legacy_checkpoints: list[tuple[ArtifactFile, ArtifactFile]] = []
    monkeypatch.setattr(
        media,
        "checkpoint_assembled_video",
        lambda _lease, *, video_file, manifest_file, **_kwargs: legacy_checkpoints.append(
            (video_file, manifest_file)
        ),
    )
    monkeypatch.setattr(
        media,
        "record_assembly_attempt",
        lambda _lease, index, disposition, *, evidence_file, **_kwargs: recorded.append(
            (index, disposition, evidence_file.artifact_kind)
        ),
    )
    monkeypatch.setattr(
        media,
        "publish_immutable_r2_objects",
        lambda values: publications.append(dict(values)),
    )

    def renew(lease: SlotLease) -> SlotLease:
        renewals.append(lease)
        return lease

    first = media.assemble_edition(_lease(edition), plan, attempt_index=0, renew_lease=renew)
    second = media.assemble_edition(_lease(edition), plan)

    assert first.video_file.content == second.video_file.content
    assert first.manifest_file.content == second.manifest_file.content
    assert legacy_checkpoints == [(second.video_file, second.manifest_file)]
    manifest = json.loads(first.manifest_file.content)
    assert [entry["story_position"] for entry in manifest["clips"]] == [0, 1]
    assert manifest["expected_duration_ms"] == 2000
    assert manifest["loudness_target_i"] == "-16"
    assert abs(first.duration_ms - 2000) <= 100
    assert publications[0][first.video_file.r2_key] == first.video_file.content
    assert recorded == [(0, "succeeded", "video_digest_assembly_manifest")]
    assert renewals
    output = tmp_path / "result.mp4"
    output.write_bytes(first.video_file.content)
    assert _sample_rgb(output, "0.5") == "red"
    assert _sample_rgb(output, "1.5") == "blue"


def test_media_processes_are_noninteractive_locale_stable_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finished = media._run(("cat",), capture_output=True)
    assert finished.returncode == 0

    locale = media._run(("printenv", "LC_ALL"), capture_output=True)
    assert locale.stdout.strip() == "C"

    monkeypatch.setattr(media, "MEDIA_PROCESS_TIMEOUT_SECONDS", 1)
    with pytest.raises(subprocess.TimeoutExpired):
        media._run(("sleep", "30"), capture_output=True)


class _SuccessfulTimingProvider:
    def __init__(self) -> None:
        self.strategies: list[media.SubtitleStrategy] = []

    def timings(
        self,
        strategy: media.SubtitleStrategy,
        requests: tuple[media.SubtitleTimingRequest, ...],
        media_path: Path,
    ) -> tuple[tuple[media.SubtitleTimeSpan, ...], ...]:
        del media_path
        self.strategies.append(strategy)
        cursor = 0
        result = []
        for request in requests:
            spans = []
            for _ in range(request.cue_count):
                spans.append(media.SubtitleTimeSpan(start_ms=cursor, end_ms=cursor + 100))
                cursor += 100
            result.append(tuple(spans))
        return tuple(result)


class _FailingTimingProvider:
    def __init__(self) -> None:
        self.strategies: list[media.SubtitleStrategy] = []

    def timings(
        self,
        strategy: media.SubtitleStrategy,
        requests: tuple[media.SubtitleTimingRequest, ...],
        media_path: Path,
    ) -> tuple[tuple[media.SubtitleTimeSpan, ...], ...]:
        del requests, media_path
        self.strategies.append(strategy)
        raise RuntimeError("timing unavailable")


def test_webvtt_uses_exact_approved_text_and_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, screenplay, assembled = _subtitle_inputs()
    provider = _SuccessfulTimingProvider()
    published: dict[str, bytes] = {}
    monkeypatch.setattr(
        media,
        "publish_immutable_r2_objects",
        lambda values: published.update(values),
    )
    monkeypatch.setattr(
        media,
        "checkpoint_subtitles",
        lambda _lease, outcome, **_kwargs: outcome,
    )

    outcome = media.produce_subtitles(lease, screenplay, assembled, provider)

    assert isinstance(outcome, AvailableSubtitles)
    assert provider.strategies == [media.SubtitleStrategy.WHOLE_EDITION]
    content = next(iter(published.values())).decode()
    assert content.startswith("WEBVTT\n")
    covered = " ".join(
        line for line in content.splitlines() if line and "-->" not in line and not line.isdigit()
    ).replace("WEBVTT ", "")
    assert covered.split() == " ".join(story.narration for story in screenplay.stories).split()
    cue_blocks = content.strip().split("\n\n")[1:]
    assert all(len(block.splitlines()) <= 4 for block in cue_blocks)
    assert all(len(" ".join(block.splitlines()[2:])) <= 60 for block in cue_blocks)


def test_subtitle_failure_tries_three_strategies_and_keeps_clean_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, screenplay, assembled = _subtitle_inputs()
    original_video = assembled.video_file.content
    provider = _FailingTimingProvider()
    published: dict[str, bytes] = {}
    monkeypatch.setattr(
        media,
        "publish_immutable_r2_objects",
        lambda values: published.update(values),
    )
    monkeypatch.setattr(
        media,
        "checkpoint_subtitles",
        lambda _lease, outcome, **_kwargs: outcome,
    )

    outcome = media.produce_subtitles(lease, screenplay, assembled, provider)

    assert isinstance(outcome, FailedSubtitles)
    assert provider.strategies == list(media.SubtitleStrategy)
    assert assembled.video_file.content == original_video
    assert len(published) == 1
    evidence = json.loads(next(iter(published.values())))
    assert [attempt["strategy"] for attempt in evidence["attempts"]] == [
        strategy.value for strategy in media.SubtitleStrategy
    ]


def test_subtitle_text_bounds_record_failure_and_keep_clean_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, screenplay, assembled = _subtitle_inputs()
    screenplay = screenplay.model_copy(
        update={
            "stories": (
                screenplay.stories[0].model_copy(update={"narration": "x" * 61}),
                screenplay.stories[1],
            )
        }
    )
    provider = _SuccessfulTimingProvider()
    published: dict[str, bytes] = {}
    monkeypatch.setattr(
        media,
        "publish_immutable_r2_objects",
        lambda values: published.update(values),
    )
    monkeypatch.setattr(
        media,
        "checkpoint_subtitles",
        lambda _lease, outcome, **_kwargs: outcome,
    )

    outcome = media.produce_subtitles(lease, screenplay, assembled, provider)

    assert isinstance(outcome, FailedSubtitles)
    assert provider.strategies == []
    assert assembled.video_file.content == b"clean-video"
    evidence = json.loads(next(iter(published.values())))
    assert [attempt["code"] for attempt in evidence["attempts"]] == ["ValueError"] * 3


def _sample_rgb(path: Path, timestamp: str) -> str:
    sample = subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            timestamp,
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            "scale=1:1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ),
        check=True,
        capture_output=True,
    ).stdout
    red, _green, blue = sample
    return "red" if red > blue else "blue"


def _subtitle_inputs() -> tuple[SlotLease, ScreenplayPlan, media.AssembledEdition]:
    report_version = "a" * 64
    policy_version = "b" * 64
    edition = edition_id(report_version, policy_version)
    stories = (
        ScreenplayStory(
            report_subject_id="1" * 64,
            title="Prima poveste",
            citation_article_version_ids=("3" * 64,),
            narration=(
                "Acesta este textul aprobat pentru prima poveste si ramane exact in subtitrare."
            ),
            visual_direction="Cadru unu",
            requested_duration_ms=1000,
        ),
        ScreenplayStory(
            report_subject_id="2" * 64,
            title="A doua poveste",
            citation_article_version_ids=("4" * 64,),
            narration="Al doilea text aprobat urmeaza in ordinea exacta a planului.",
            visual_direction="Cadru doi",
            requested_duration_ms=1000,
        ),
    )
    screenplay = ScreenplayPlan(
        edition_id=edition,
        daily_report_version_id=report_version,
        policy_bundle_version_id=policy_version,
        policy_bundle_digest="c" * 64,
        stories=stories,
    )
    video_file = artifact_file(
        artifact_id=f"{edition}:assembled-video",
        artifact_kind="video_digest_assembled_video",
        title="Assembled video",
        content=b"clean-video",
        r2_key=f"assembled/{edition}.mp4",
        media_type="video/mp4",
    )
    manifest_file = artifact_file(
        artifact_id=f"{edition}:assembly-manifest",
        artifact_kind="video_digest_assembly_manifest",
        title="Assembly manifest",
        content=b"manifest",
        r2_key=f"assembled/{edition}.json",
        media_type="application/json",
    )
    return (
        _lease(edition),
        screenplay,
        media.AssembledEdition(
            video_file=video_file,
            manifest_file=manifest_file,
            duration_ms=2000,
            clip_durations_ms=(1000, 1000),
        ),
    )
