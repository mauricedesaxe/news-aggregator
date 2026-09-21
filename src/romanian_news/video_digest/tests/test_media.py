from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import artifact_file, sha256
from romanian_news.catalog.video_digest import AcceptedClipReference, GenerationAttemptReference
from romanian_news.storage import ResearchObjectIntegrityError
from romanian_news.video_digest import media
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
from romanian_news.video_digest.planning import ScreenplayPlan, ScreenplayStory

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
pytestmark = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg and ffprobe are required")


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


def _probe_payload(*, frame_rate: str = "24/1", pixel_format: str = "yuv420p") -> dict[str, object]:
    return {
        "streams": [
            {
                "codec_type": "video",
                "avg_frame_rate": frame_rate,
                "duration": "1.0",
                "start_time": "0",
                "nb_read_frames": "24",
                "codec_name": "h264",
                "width": 1344,
                "height": 768,
                "pix_fmt": pixel_format,
            },
            {
                "codec_type": "audio",
                "duration": "1.0",
                "start_time": "0",
                "nb_read_frames": "32",
                "codec_name": "aac",
                "channels": 2,
                "sample_rate": "32000",
            },
        ],
        "format": {"duration": "1.0"},
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
    assert "-xerror" in media.full_decode_command(path)


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
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: (attempt,))
    monkeypatch.setattr(media, "read_verified_r2_object", lambda _key, _digest: content)

    def publish(objects) -> None:
        events.append("publish")
        published.update(objects)

    def checkpoint(*_args, validation_file, **_kwargs) -> None:
        events.append("checkpoint")
        evidence = json.loads(validation_file.content)
        assert evidence["request_id"] == candidate.request_id
        assert (
            evidence["request_artifact_version_id"] == attempt.request.request_artifact_version_id
        )

    monkeypatch.setattr(media, "publish_immutable_r2_objects", publish)
    monkeypatch.setattr(media, "checkpoint_generation_acceptance", checkpoint)

    result = media.accept_candidate(_lease(edition), candidate, story)

    assert isinstance(result, media.AcceptedCandidate)
    assert events == ["publish", "checkpoint"]
    assert any(key.endswith(".mp4") for key in published)

    events.clear()

    def corrupt_read(_key: str, _digest: str) -> bytes:
        raise ResearchObjectIntegrityError("digest mismatch")

    monkeypatch.setattr(media, "read_verified_r2_object", corrupt_read)
    monkeypatch.setattr(
        media,
        "checkpoint_generation_failure",
        lambda *_args, **_kwargs: events.append("failure-checkpoint"),
    )

    rejected = media.accept_candidate(_lease(edition), candidate, story)

    assert isinstance(rejected, media.GenerationRetryAvailable)
    assert events == ["publish", "failure-checkpoint"]


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
    monkeypatch.setattr(media, "read_generation_attempts", lambda _edition: attempts)
    monkeypatch.setattr(media, "read_verified_r2_object", lambda key, _digest: objects[key])
    monkeypatch.setattr(media, "checkpoint_assembly_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(media, "checkpoint_assembled_video", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        media,
        "publish_immutable_r2_objects",
        lambda values: publications.append(dict(values)),
    )

    first = media.assemble_edition(_lease(edition), plan)
    second = media.assemble_edition(_lease(edition), plan)

    assert first.manifest_file.content == second.manifest_file.content
    manifest = json.loads(first.manifest_file.content)
    assert [entry["story_position"] for entry in manifest["clips"]] == [0, 1]
    assert manifest["expected_duration_ms"] == 2000
    assert manifest["loudness_target_i"] == "-16"
    assert abs(first.duration_ms - 2000) <= 100
    assert publications[0][first.video_file.r2_key] == first.video_file.content
    output = tmp_path / "result.mp4"
    output.write_bytes(first.video_file.content)
    assert _sample_rgb(output, "0.5") == "red"
    assert _sample_rgb(output, "1.5") == "blue"


def test_commands_encode_fixed_media_policy(tmp_path: Path) -> None:
    inputs = (tmp_path / "0.mp4", tmp_path / "1.mp4")
    command = media.assembly_command(inputs, (1000, 1000), tmp_path / "output.mp4")
    joined = " ".join(command)

    assert "concat=n=2:v=1:a=1" in joined
    assert "fade=t=out:st=0.75:d=0.25" in joined
    assert "-preset medium -crf 18" in joined
    assert "-pix_fmt yuv420p" in joined
    assert "-threads 1" in joined
    assert "-b:a 192k -ac 2 -ar 32000" in joined
    assert "-map_metadata -1 -movflags +faststart" in joined
    assert "-nostdin" in command


def test_media_processes_are_noninteractive_bounded_and_locale_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def run(*_args, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess((), 0, "", "")

    monkeypatch.setattr(media.subprocess, "run", run)

    media._run(("ffmpeg", "-version"), capture_output=True)

    assert calls == [
        {
            "check": True,
            "capture_output": True,
            "env": {**media.os.environ, "LC_ALL": "C"},
            "stdin": subprocess.DEVNULL,
            "text": True,
            "timeout": media.MEDIA_PROCESS_TIMEOUT_SECONDS,
        }
    ]


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
        ),
    )
