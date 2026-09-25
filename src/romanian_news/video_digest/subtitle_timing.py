from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired
from tempfile import TemporaryDirectory
from typing import Any

from romanian_news.video_digest.media import (
    SubtitleStrategy,
    SubtitleTimeSpan,
    SubtitleTimingRequest,
    _run,
)
from romanian_news.video_digest.subtitle_text import _subtitle_cue_texts

MODEL_NAME = "base.en"
MIN_MATCH_FRACTION = 0.6


@dataclass(frozen=True)
class TimedWord:
    text: str
    start_ms: int
    end_ms: int


def _load_model() -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")


class FasterWhisperSubtitleTimingProvider:
    def __init__(
        self,
        model_factory: Callable[[], Any] = _load_model,
        *,
        before_transcribe: Callable[[], None] | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._model: Any | None = None
        self._before_transcribe = before_transcribe

    def timings(
        self,
        strategy: SubtitleStrategy,
        requests: tuple[SubtitleTimingRequest, ...],
        media_path: Path,
    ) -> tuple[tuple[SubtitleTimeSpan, ...], ...]:
        if not requests:
            raise ValueError("Subtitle timing needs at least one story")
        self._validate_windows(requests)
        if strategy is SubtitleStrategy.WHOLE_EDITION:
            words = self._transcribe(media_path, vad_filter=True)
            return self._align_requests(requests, words)
        if strategy not in (SubtitleStrategy.PER_STORY, SubtitleStrategy.PER_STORY_WITHOUT_VAD):
            raise ValueError("Unsupported subtitle timing strategy")
        vad_filter = strategy is SubtitleStrategy.PER_STORY
        with TemporaryDirectory() as directory:
            result = []
            for request in requests:
                audio_path = Path(directory) / f"story-{request.story_position}.wav"
                try:
                    _run(
                        (
                            "ffmpeg",
                            "-nostdin",
                            "-v",
                            "error",
                            "-xerror",
                            "-i",
                            str(media_path),
                            "-ss",
                            f"{request.start_ms / 1000:.3f}",
                            "-t",
                            f"{(request.end_ms - request.start_ms) / 1000:.3f}",
                            "-vn",
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            str(audio_path),
                        ),
                        capture_output=True,
                    )
                except (CalledProcessError, TimeoutExpired) as error:
                    raise RuntimeError("Subtitle audio extraction failed") from error
                words = self._transcribe(audio_path, vad_filter=vad_filter)
                spans = self._align_requests((request,), words, offset_ms=request.start_ms)[0]
                result.append(spans)
        return tuple(result)

    def _transcribe(self, path: Path, *, vad_filter: bool) -> tuple[TimedWord, ...]:
        if self._before_transcribe is not None:
            self._before_transcribe()
        model = self._model
        if model is None:
            model = self._model_factory()
            self._model = model
        segments, _info = model.transcribe(
            str(path), language="en", word_timestamps=True, vad_filter=vad_filter
        )
        words = tuple(
            TimedWord(
                text=word.word,
                start_ms=round(word.start * 1000),
                end_ms=round(word.end * 1000),
            )
            for segment in segments
            for word in segment.words or ()
            if _normalized(word.word) and round(word.end * 1000) > round(word.start * 1000)
        )
        if not words:
            raise ValueError("Whisper returned no usable word timestamps")
        return words

    @staticmethod
    def _validate_windows(requests: tuple[SubtitleTimingRequest, ...]) -> None:
        previous_end = 0
        for position, request in enumerate(requests):
            if request.story_position != position or request.start_ms != previous_end:
                raise ValueError("Subtitle story windows must be contiguous and ordered")
            previous_end = request.end_ms

    @staticmethod
    def _align_requests(
        requests: tuple[SubtitleTimingRequest, ...],
        words: tuple[TimedWord, ...],
        *,
        offset_ms: int = 0,
    ) -> tuple[tuple[SubtitleTimeSpan, ...], ...]:
        cue_groups = tuple(_subtitle_cue_texts(request.approved_text) for request in requests)
        if any(
            len(cues) != request.cue_count
            for cues, request in zip(cue_groups, requests, strict=True)
        ):
            raise ValueError("Subtitle cue count differs from approved text")
        approved = [word for request in requests for word in request.approved_text.split()]
        approved_tokens = [_normalized(word) for word in approved]
        if not all(approved_tokens):
            raise ValueError("Approved subtitle text contains no spoken word")
        transcript_tokens = [_normalized(word.text) for word in words]
        matches = SequenceMatcher(
            None, approved_tokens, transcript_tokens, autojunk=False
        ).get_matching_blocks()
        anchors = {
            approved_index + delta: words[transcript_index + delta]
            for approved_index, transcript_index, length in matches
            for delta in range(length)
        }
        cursor = 0
        for request in requests:
            story_count = len(request.approved_text.split())
            story_anchors = sum(cursor <= index < cursor + story_count for index in anchors)
            if (
                story_anchors < story_count * MIN_MATCH_FRACTION
                or cursor not in anchors
                or cursor + story_count - 1 not in anchors
            ):
                raise ValueError("Whisper transcript differs from approved screenplay")
            cursor += story_count
        timed_words = _interpolate_words(len(approved), anchors)
        result = []
        cursor = 0
        for request, cues in zip(requests, cue_groups, strict=True):
            spans = []
            for cue in cues:
                count = len(cue.split())
                first = timed_words[cursor]
                last = timed_words[cursor + count - 1]
                start_ms = first.start_ms + offset_ms
                end_ms = last.end_ms + offset_ms
                if start_ms < request.start_ms or end_ms > request.end_ms:
                    raise ValueError("Subtitle timestamps exceed the story window")
                spans.append(SubtitleTimeSpan(start_ms=start_ms, end_ms=end_ms))
                cursor += count
            result.append(tuple(spans))
        return tuple(result)


def _normalized(value: str) -> str:
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKC", value)
        if character.isalnum()
    )


def _interpolate_words(count: int, anchors: dict[int, TimedWord]) -> tuple[TimedWord, ...]:
    ordered = sorted(anchors)
    spans: list[TimedWord | None] = [None] * count
    for index, word in anchors.items():
        spans[index] = word
    for left_index, right_index in pairwise(ordered):
        missing = right_index - left_index - 1
        if not missing:
            continue
        left_time = anchors[left_index].end_ms
        right_time = anchors[right_index].start_ms
        if right_time - left_time < missing:
            borrowed = missing - (right_time - left_time)
            right_anchor = anchors[right_index]
            if right_anchor.end_ms - right_anchor.start_ms <= borrowed:
                raise ValueError("Whisper timing has no room for unmatched words")
            spans[right_index] = TimedWord(
                right_anchor.text, right_anchor.start_ms + borrowed, right_anchor.end_ms
            )
            right_time += borrowed
        for step in range(1, missing + 1):
            start = left_time + (right_time - left_time) * (step - 1) // missing
            end = left_time + (right_time - left_time) * step // missing
            spans[left_index + step] = TimedWord("", start, end)
    if any(span is None or span.end_ms <= span.start_ms for span in spans):
        raise ValueError("Whisper timing cannot cover approved screenplay")
    return tuple(span for span in spans if span is not None)
