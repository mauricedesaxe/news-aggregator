from pathlib import Path
from subprocess import CalledProcessError
from types import SimpleNamespace

import pytest

from romanian_news.video_digest import subtitle_timing
from romanian_news.video_digest.media import SubtitleStrategy, SubtitleTimingRequest


def _requests() -> tuple[SubtitleTimingRequest, ...]:
    return (
        SubtitleTimingRequest(
            story_position=0,
            cue_count=1,
            approved_text="Hello world",
            start_ms=0,
            end_ms=1000,
        ),
        SubtitleTimingRequest(
            story_position=1,
            cue_count=1,
            approved_text="Next story",
            start_ms=1000,
            end_ms=2000,
        ),
    )


class FakeWhisper:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def transcribe(
        self, path: str, *, language: str, word_timestamps: bool, vad_filter: bool
    ) -> tuple[list[SimpleNamespace], None]:
        assert language == "en"
        assert word_timestamps
        self.calls.append((path, vad_filter))
        if path.endswith("story-0.wav"):
            tokens = ("Hello", "world")
            times = ((0.1, 0.2), (0.3, 0.4))
        elif path.endswith("story-1.wav"):
            tokens = ("Next", "story")
            times = ((0.1, 0.2), (0.3, 0.4))
        else:
            tokens = ("Hello", "world", "Next", "story")
            times = ((0.1, 0.2), (0.3, 0.4), (1.1, 1.2), (1.3, 1.4))
        words = [
            SimpleNamespace(word=token, start=start, end=end)
            for token, (start, end) in zip(tokens, times, strict=True)
        ]
        return [SimpleNamespace(words=words)], None


def test_whole_edition_maps_word_timestamps_to_approved_story_cues() -> None:
    whisper = FakeWhisper()
    provider = subtitle_timing.FasterWhisperSubtitleTimingProvider(lambda: whisper)

    result = provider.timings(SubtitleStrategy.WHOLE_EDITION, _requests(), Path("edition.mp4"))

    assert [[(span.start_ms, span.end_ms) for span in story] for story in result] == [
        [(100, 400)],
        [(1100, 1400)],
    ]
    assert whisper.calls == [("edition.mp4", True)]


@pytest.mark.parametrize(
    ("strategy", "vad_filter"),
    [
        (SubtitleStrategy.PER_STORY, True),
        (SubtitleStrategy.PER_STORY_WITHOUT_VAD, False),
    ],
)
def test_story_strategies_extract_exact_windows_and_offset_timestamps(
    strategy: SubtitleStrategy, vad_filter: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], *, capture_output: bool) -> None:
        assert capture_output
        commands.append(command)
        Path(command[-1]).write_bytes(b"audio")

    monkeypatch.setattr(subtitle_timing, "_run", run)
    whisper = FakeWhisper()
    renewals: list[None] = []
    provider = subtitle_timing.FasterWhisperSubtitleTimingProvider(
        lambda: whisper, before_transcribe=lambda: renewals.append(None)
    )

    result = provider.timings(strategy, _requests(), Path("edition.mp4"))

    assert [(span.start_ms, span.end_ms) for story in result for span in story] == [
        (100, 400),
        (1100, 1400),
    ]
    assert [command[command.index("-ss") + 1] for command in commands] == ["0.000", "1.000"]
    assert [command[command.index("-t") + 1] for command in commands] == ["1.000", "1.000"]
    assert [call[1] for call in whisper.calls] == [vad_filter, vad_filter]
    assert len(renewals) == 2


def test_transcript_mismatch_cannot_publish_unsynchronized_cues() -> None:
    provider = subtitle_timing.FasterWhisperSubtitleTimingProvider(
        lambda: SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: (
                [
                    SimpleNamespace(
                        words=[
                            SimpleNamespace(word="different", start=0.1, end=0.2),
                            SimpleNamespace(word="speech", start=0.3, end=0.4),
                        ]
                    )
                ],
                None,
            )
        )
    )

    with pytest.raises(ValueError, match="differs from approved screenplay"):
        provider.timings(SubtitleStrategy.WHOLE_EDITION, _requests(), Path("edition.mp4"))


def test_audio_extraction_failure_is_recoverable_by_subtitle_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_command: tuple[str, ...], *, capture_output: bool) -> None:
        assert capture_output
        raise CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr(subtitle_timing, "_run", fail)
    provider = subtitle_timing.FasterWhisperSubtitleTimingProvider(lambda: FakeWhisper())

    with pytest.raises(RuntimeError, match="audio extraction failed"):
        provider.timings(SubtitleStrategy.PER_STORY, _requests(), Path("edition.mp4"))


def test_zero_duration_word_uses_neighboring_verified_timestamps() -> None:
    provider = subtitle_timing.FasterWhisperSubtitleTimingProvider(
        lambda: SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: (
                [
                    SimpleNamespace(
                        words=[
                            SimpleNamespace(word="Hello", start=0.1, end=0.2),
                            SimpleNamespace(word="little", start=0.3, end=0.3),
                            SimpleNamespace(word="story", start=0.7, end=0.8),
                        ]
                    )
                ],
                None,
            )
        )
    )
    request = SubtitleTimingRequest(
        story_position=0,
        cue_count=1,
        approved_text="Hello little story",
        start_ms=0,
        end_ms=1000,
    )

    result = provider.timings(SubtitleStrategy.WHOLE_EDITION, (request,), Path("edition.mp4"))

    assert result[0][0].start_ms == 100
    assert result[0][0].end_ms == 800


def test_missing_words_interpolate_only_between_verified_anchors() -> None:
    words = (
        subtitle_timing.TimedWord("Hello", 100, 200),
        subtitle_timing.TimedWord("story", 700, 800),
    )
    request = SubtitleTimingRequest(
        story_position=0,
        cue_count=1,
        approved_text="Hello little story",
        start_ms=0,
        end_ms=1000,
    )

    result = subtitle_timing.FasterWhisperSubtitleTimingProvider._align_requests((request,), words)

    assert result[0][0].start_ms == 100
    assert result[0][0].end_ms == 800


def test_interpolation_borrows_one_millisecond_from_an_adjacent_anchor() -> None:
    words = subtitle_timing._interpolate_words(
        3,
        {
            0: subtitle_timing.TimedWord("S", 100, 200),
            2: subtitle_timing.TimedWord("P", 200, 500),
        },
    )

    assert [(word.start_ms, word.end_ms) for word in words] == [
        (100, 200),
        (200, 201),
        (201, 500),
    ]


def test_missing_story_end_cannot_be_extrapolated_to_publish() -> None:
    request = SubtitleTimingRequest(
        story_position=0,
        cue_count=1,
        approved_text="alpha beta gamma delta epsilon",
        start_ms=0,
        end_ms=1000,
    )
    words = (
        subtitle_timing.TimedWord("alpha", 100, 200),
        subtitle_timing.TimedWord("beta", 250, 300),
        subtitle_timing.TimedWord("gamma", 350, 400),
    )

    with pytest.raises(ValueError, match="differs from approved screenplay"):
        subtitle_timing.FasterWhisperSubtitleTimingProvider._align_requests((request,), words)
