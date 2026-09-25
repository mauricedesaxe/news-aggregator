from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from romanian_news import NewsModel
from romanian_news.video_digest.planning import ScreenplayPlan

SUBTITLE_POLICY_VERSION = "approved-screenplay-webvtt-v1"


class SubtitleAssembly(Protocol):
    @property
    def duration_ms(self) -> int: ...

    @property
    def clip_durations_ms(self) -> tuple[int, ...]: ...


class SubtitleStrategy(StrEnum):
    WHOLE_EDITION = "whole-edition-v1"
    PER_STORY = "per-story-v1"
    PER_STORY_WITHOUT_VAD = "per-story-without-vad-v1"


class SubtitleTimingRequest(NewsModel):
    story_position: Annotated[int, Field(ge=0)]
    cue_count: Annotated[int, Field(gt=0)]
    approved_text: str
    start_ms: Annotated[int, Field(ge=0)]
    end_ms: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def require_positive_duration(self) -> SubtitleTimingRequest:
        if self.end_ms <= self.start_ms:
            raise ValueError("Subtitle story window must have positive duration")
        return self


class SubtitleTimeSpan(NewsModel):
    start_ms: Annotated[int, Field(ge=0)]
    end_ms: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def require_positive_duration(self) -> SubtitleTimeSpan:
        if self.end_ms <= self.start_ms:
            raise ValueError("Subtitle timing must have positive duration")
        return self


class SubtitleTimingProvider(Protocol):
    def timings(
        self,
        strategy: SubtitleStrategy,
        requests: tuple[SubtitleTimingRequest, ...],
        media_path: Path,
    ) -> tuple[tuple[SubtitleTimeSpan, ...], ...]: ...


class SubtitleAttemptFailure(NewsModel):
    strategy: SubtitleStrategy
    code: str


class SubtitleFailureEvidence(NewsModel):
    policy_version: Literal["approved-screenplay-webvtt-v1"] = SUBTITLE_POLICY_VERSION
    edition_id: str
    attempts: Annotated[tuple[SubtitleAttemptFailure, ...], Field(min_length=3, max_length=3)]


def _subtitle_cue_texts(text: str) -> tuple[str, ...]:
    words = text.split()
    if not words:
        raise ValueError("Approved subtitle text must not be empty")
    cues: list[str] = []
    lines: list[str] = []
    for word in words:
        if len(word) > 60:
            raise ValueError("Approved subtitle text contains a word longer than 60 characters")
        if len(word) > 30:
            if lines:
                cues.append("\n".join(lines))
                lines = []
            cues.append(word)
            continue
        if not lines:
            lines.append(word)
        elif len(lines[-1]) + 1 + len(word) <= 30:
            lines[-1] += f" {word}"
        elif len(lines) == 1:
            lines.append(word)
        else:
            cues.append("\n".join(lines))
            lines = [word]
    if lines:
        cues.append("\n".join(lines))
    if any(len(cue.replace("\n", " ")) > 60 or cue.count("\n") > 1 for cue in cues):
        raise ValueError("Subtitle cue exceeds its line or character bound")
    return tuple(cues)


def subtitle_timing_requests(
    screenplay: ScreenplayPlan, assembled: SubtitleAssembly
) -> tuple[tuple[tuple[str, ...], ...], tuple[SubtitleTimingRequest, ...]]:
    cue_texts = tuple(_subtitle_cue_texts(story.narration) for story in screenplay.stories)
    if len(assembled.clip_durations_ms) != len(screenplay.stories):
        raise ValueError("Subtitle story windows must cover every story")
    cursor = 0
    windows = []
    for duration_ms in assembled.clip_durations_ms:
        if duration_ms <= 0:
            raise ValueError("Subtitle story window must have positive duration")
        windows.append((cursor, cursor + duration_ms))
        cursor += duration_ms
    if abs(cursor - assembled.duration_ms) > 100:
        raise ValueError("Subtitle story windows differ from assembled duration")
    requests = tuple(
        SubtitleTimingRequest(
            story_position=position,
            cue_count=len(texts),
            approved_text=screenplay.stories[position].narration,
            start_ms=windows[position][0],
            end_ms=windows[position][1],
        )
        for position, texts in enumerate(cue_texts)
    )
    return cue_texts, requests


def _webvtt(
    screenplay: ScreenplayPlan,
    cue_texts: tuple[tuple[str, ...], ...],
    timings: tuple[tuple[SubtitleTimeSpan, ...], ...],
    duration_ms: int,
) -> bytes:
    if len(timings) != len(cue_texts):
        raise ValueError("Subtitle timings must cover every story")
    cues: list[tuple[SubtitleTimeSpan, str]] = []
    previous_end = 0
    for texts, spans in zip(cue_texts, timings, strict=True):
        if len(texts) != len(spans):
            raise ValueError("Subtitle timings must cover every approved cue")
        for text, span in zip(texts, spans, strict=True):
            if span.start_ms < previous_end or span.end_ms > duration_ms:
                raise ValueError("Subtitle timings overlap or exceed the edition")
            previous_end = span.end_ms
            cues.append((span, text))
    approved = " ".join(story.narration for story in screenplay.stories).split()
    covered = " ".join(text.replace("\n", " ") for _, text in cues).split()
    if covered != approved:
        raise ValueError("Subtitle cues do not exactly cover the approved screenplay")
    lines = ["WEBVTT", ""]
    for index, (span, text) in enumerate(cues, start=1):
        lines.extend(
            (
                str(index),
                f"{_webvtt_timestamp(span.start_ms)} --> {_webvtt_timestamp(span.end_ms)}",
                text,
                "",
            )
        )
    return "\n".join(lines).encode()


def _webvtt_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"
