from __future__ import annotations

import base64
import binascii
from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, NewType

from pydantic import (
    AwareDatetime,
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
    model_validator,
)

from romanian_news import NewsModel, OutletId, Sha256
from romanian_news.analysis.tracing import ModelTraceReference

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
YouTubeSourceId = NewType("YouTubeSourceId", str)
SourceId = Annotated[YouTubeSourceId, Field(pattern=r"^[a-z0-9-]+$")]
YouTubeVideoId = NewType("YouTubeVideoId", str)
VideoId = Annotated[YouTubeVideoId, Field(pattern=r"^[A-Za-z0-9_-]{11}$")]
YOUTUBE_MODEL = "gemini-3.6-flash"
YOUTUBE_RATE_CARD_VERSION = "gemini-3.6-flash-2026-09-09"
YOUTUBE_CLIP_SECONDS = 300
AUTO_VIDEO_BUDGET_USD = 5.0
ESTIMATED_EXTRACTION_USD_PER_MINUTE = 0.014
MERGE_RESERVE_USD = 0.05
MAX_UNCHANGED_DETERMINISTIC_FAILURES = 3
YOUTUBE_LEASE_SECONDS = 600
LIVE_MEDIA_REPLAY_LIMIT = "Gemini read the live YouTube URL; exact media bytes were not captured."


class YouTubeSource(NewsModel):
    source_id: SourceId
    channel_id: Annotated[str, Field(pattern=r"^UC[A-Za-z0-9_-]{22}$")]
    outlet_id: OutletId
    display_name: NonEmptyText

    @property
    def uploads_playlist_id(self) -> str:
        return f"UU{self.channel_id.removeprefix('UC')}"


class YouTubeSourceRegistry(NewsModel):
    sources: Annotated[tuple[YouTubeSource, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_identity(self) -> YouTubeSourceRegistry:
        for field in ("source_id", "channel_id", "outlet_id"):
            values = tuple(getattr(source, field) for source in self.sources)
            if len(values) != len(set(values)):
                raise ValueError(f"YouTube source registry has duplicate {field}")
        return self

    def source(self, source_id: str) -> YouTubeSource:
        for value in self.sources:
            if value.source_id == source_id:
                return value
        raise ValueError(f"Unknown YouTube source: {source_id}")


YOUTUBE_SOURCES = YouTubeSourceRegistry(
    sources=(
        YouTubeSource(
            source_id=YouTubeSourceId("recorder-youtube"),
            channel_id="UChDQ6nYN6XyRU-8IEgbym1g",
            outlet_id="recorder",
            display_name="Recorder",
        ),
        YouTubeSource(
            source_id=YouTubeSourceId("starea-impostorilor-youtube"),
            channel_id="UCtK5Oe8sHjp6WPcwWuHUVpQ",
            outlet_id="starea-impostorilor",
            display_name="Starea Impostorilor",
        ),
        YouTubeSource(
            source_id=YouTubeSourceId("snoop-youtube"),
            channel_id="UCi7oQBZ4Amu_xPxc78F970A",
            outlet_id="snoop",
            display_name="Snoop",
        ),
        YouTubeSource(
            source_id=YouTubeSourceId("stirile-protv-youtube"),
            channel_id="UCEJf5cGtkBdZS8Jh2uSW9xw",
            outlet_id="stirile-protv",
            display_name="Știrile ProTV",
        ),
    )
)


class GeminiRateCard(NewsModel):
    version: Literal["gemini-3.6-flash-2026-09-09"]
    text_input_usd_per_million: float
    audio_input_usd_per_million: float
    output_usd_per_million: float
    thought_tokens_billed_as_output: bool


YOUTUBE_RATE_CARDS: dict[str, GeminiRateCard] = {
    YOUTUBE_MODEL: GeminiRateCard(
        version=YOUTUBE_RATE_CARD_VERSION,
        text_input_usd_per_million=0.30,
        audio_input_usd_per_million=1.00,
        output_usd_per_million=2.50,
        thought_tokens_billed_as_output=True,
    )
}


class GeminiResponse(NewsModel):
    response_id: NonEmptyText
    model: NonEmptyText
    payload: dict[str, object]
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    thought_tokens: Annotated[int, Field(ge=0)]
    cost_usd: Annotated[float, Field(ge=0)]
    rate_card_version: NonEmptyText
    latency_ms: Annotated[int, Field(ge=0)]

    def attempt_payload(self) -> dict[str, object]:
        return {
            "id": self.response_id,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.input_tokens,
                "completion_tokens": self.output_tokens,
                "thought_tokens": self.thought_tokens,
                "cost": self.cost_usd,
            },
            "latency_ms": self.latency_ms,
            "provider_response": self.payload,
            "rate_card_version": self.rate_card_version,
        }


class GeminiHttpResponse(NewsModel):
    requested_model: NonEmptyText
    status_code: Annotated[int, Field(ge=100, le=599)]
    body_base64: str
    latency_ms: Annotated[int, Field(ge=0)]

    @field_validator("body_base64")
    @classmethod
    def validate_body_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("Gemini HTTP response body is not valid base64") from error
        return value

    def body_bytes(self) -> bytes:
        return base64.b64decode(self.body_base64, validate=True)


class YouTubeModelReceipt(NewsModel):
    request_id: Sha256
    operation_key: NonEmptyText
    attempt_index: Annotated[int, Field(ge=0)]
    response: GeminiHttpResponse
    trace: ModelTraceReference | None
    received_at: AwareDatetime


class YouTubeVideoState(StrEnum):
    BASELINE = "baseline"
    PENDING = "pending"
    RUNNING = "running"
    DEFERRED = "deferred"
    QUARANTINED = "quarantined"
    CANDIDATE = "candidate"


class YouTubeFeedEntry(NewsModel):
    source: YouTubeSource
    video_id: VideoId
    title: NonEmptyText
    published_at: AwareDatetime


class YouTubeFeed(NewsModel):
    source: YouTubeSource
    entries: Annotated[tuple[YouTubeFeedEntry, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_identity(self) -> YouTubeFeed:
        if any(entry.source != self.source for entry in self.entries):
            raise ValueError("YouTube entries do not share the feed source")
        ids = tuple(value.video_id for value in self.entries)
        if len(ids) != len(set(ids)):
            raise ValueError("YouTube feed contains duplicate video IDs")
        return self


class YouTubePollStatus(StrEnum):
    BASELINE = "baseline"
    CURRENT = "current"
    OVERLAP_LOST = "overlap_lost"


class YouTubePollCapture(NewsModel):
    source: YouTubeSource
    scheduled_at: AwareDatetime
    observed_at: AwareDatetime
    content: bytes
    feed: YouTubeFeed

    @model_validator(mode="after")
    def validate_source(self) -> YouTubePollCapture:
        if self.feed.source != self.source:
            raise ValueError("YouTube poll and feed sources differ")
        return self


class YouTubePollDecision(NewsModel):
    source: YouTubeSource
    status: YouTubePollStatus
    pending_video_ids: tuple[VideoId, ...]


class YouTubeVideoLease(NewsModel):
    source: YouTubeSource
    video_id: VideoId
    owner_token: NonEmptyText
    lease_expires_at: AwareDatetime
    first_poll_version_id: Sha256
    metadata_version_id: Sha256 | None = None
    deterministic_failure_fingerprint: Sha256 | None
    unchanged_deterministic_failures: Annotated[int, Field(ge=0)]


class ClipRange(NewsModel):
    start_second: Annotated[int, Field(ge=0)]
    duration_seconds: Annotated[int, Field(gt=0, le=YOUTUBE_CLIP_SECONDS)]

    @property
    def end_second(self) -> int:
        return self.start_second + self.duration_seconds


class YouTubeVideoMetadata(NewsModel):
    source: YouTubeSource
    video_id: VideoId
    url: HttpUrl
    title: NonEmptyText
    published_at: AwareDatetime
    duration_seconds: Annotated[int, Field(gt=0)]
    feed_snapshot_version_id: Sha256
    retrieved_at: AwareDatetime
    provider_response: dict[str, object]
    raw_provider_response: bytes


class ClipEvidence(NewsModel):
    kind: Literal[
        "reported_fact",
        "attributed_claim",
        "visual_observation",
        "on_screen_data",
        "publisher_framing",
    ]
    text: NonEmptyText
    attribution: NonEmptyText
    start_second: Annotated[float, Field(ge=0)]
    duration_seconds: Annotated[float, Field(gt=0)]


class YouTubeClipExtraction(NewsModel):
    title: NonEmptyText
    standfirst: NonEmptyText
    narrative: NonEmptyText
    evidence: Annotated[tuple[ClipEvidence, ...], Field(min_length=1)]
    source_stated_uncertainties: tuple[str, ...]


class AcceptedClipExtraction(NewsModel):
    source: YouTubeSource
    video_id: VideoId
    metadata_version_id: Sha256
    video_url: HttpUrl
    clip_range: ClipRange
    request_digest: Sha256
    model: Literal["gemini-3.6-flash"]
    extraction: YouTubeClipExtraction
    provider_response: dict[str, object]
    source_replay_limit: Literal[
        "Gemini read the live YouTube URL; exact media bytes were not captured."
    ] = LIVE_MEDIA_REPLAY_LIMIT


class OmittedClip(NewsModel):
    clip_start: Annotated[int, Field(ge=0)]
    reason: NonEmptyText


class YouTubeMergedArticle(NewsModel):
    title: NonEmptyText
    standfirst: NonEmptyText
    narrative: NonEmptyText
    covered_clip_starts: tuple[int, ...]
    omitted_clips: tuple[OmittedClip, ...]


class CandidateEvidence(NewsModel):
    clip_version_id: Sha256
    kind: Literal[
        "reported_fact",
        "attributed_claim",
        "visual_observation",
        "on_screen_data",
        "publisher_framing",
    ]
    text: NonEmptyText
    attribution: NonEmptyText
    start_second: Annotated[float, Field(ge=0)]
    duration_seconds: Annotated[float, Field(gt=0)]


class MeasuredCost(NewsModel):
    kind: Literal["measured"] = "measured"
    usd: Annotated[float, Field(ge=0)]


class UnmeasuredCost(NewsModel):
    kind: Literal["unmeasured"] = "unmeasured"
    reason: NonEmptyText


CandidateCost = Annotated[MeasuredCost | UnmeasuredCost, Field(discriminator="kind")]


class YouTubeCandidate(NewsModel):
    source: YouTubeSource
    video_id: VideoId
    video_url: HttpUrl
    video_title: NonEmptyText
    published_at: AwareDatetime
    feed_snapshot_version_id: Sha256
    metadata_version_id: Sha256
    clip_version_ids: Annotated[tuple[Sha256, ...], Field(min_length=1)]
    merge_version_id: Sha256
    receipt_version_ids: Annotated[tuple[Sha256, ...], Field(min_length=1)]
    article: YouTubeMergedArticle
    evidence: Annotated[tuple[CandidateEvidence, ...], Field(min_length=1)]
    uncertainties: tuple[str, ...]
    omissions: tuple[OmittedClip, ...]
    implementation_ref: NonEmptyText
    model: NonEmptyText
    cost: CandidateCost
    replay_limitation: Literal[
        "Gemini read the live YouTube URL; exact media bytes were not captured."
    ] = LIVE_MEDIA_REPLAY_LIMIT


class SystemApproval(NewsModel):
    """Internal decision record proving a candidate entered the pipeline without an owner."""

    kind: Literal["automatic_approved"] = "automatic_approved"
    candidate_version_id: Sha256
    source_id: SourceId
    actor: Literal["system"] = "system"
    decided_at: AwareDatetime


class YouTubeSourceResult(NewsModel):
    source_id: SourceId
    poll_status: YouTubePollStatus
    candidate_version_id: Sha256 | None


class YouTubePublicationResult(NewsModel):
    source_id: SourceId
    video_id: VideoId
    candidate_version_id: Sha256
    article_version_id: Sha256
    bucharest_day: date
    accepted_clip_count: Annotated[int, Field(ge=0)]
