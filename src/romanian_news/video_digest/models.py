from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, NewType

from pydantic import (
    AfterValidator,
    AwareDatetime,
    Field,
    HttpUrl,
    StringConstraints,
    model_validator,
)

from romanian_news import BUCHAREST, NewsModel, Sha256
from romanian_news.identity import canonical_json, sha256

EditionId = NewType("EditionId", str)
SlotId = NewType("SlotId", str)
StoryId = NewType("StoryId", str)
GenerationRequestId = NewType("GenerationRequestId", str)
PublicationId = NewType("PublicationId", str)

PUBLIC_MEDIA_CACHE_CONTROL = "public,max-age=31536000,immutable"
PUBLIC_MEDIA_VISIBILITY = "public"
PUBLIC_MEDIA_RETENTION = "permanent"

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
EditionIdField = Annotated[EditionId, Field(pattern=_SHA256_PATTERN)]
SlotIdField = Annotated[SlotId, Field(pattern=_SHA256_PATTERN)]
StoryIdField = Annotated[StoryId, Field(pattern=_SHA256_PATTERN)]
GenerationRequestIdField = Annotated[GenerationRequestId, Field(pattern=_SHA256_PATTERN)]
PublicationIdField = Annotated[PublicationId, Field(pattern=_SHA256_PATTERN)]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SlotName(StrEnum):
    MORNING = "morning"
    MIDDAY = "midday"
    EVENING = "evening"


class SlotStage(StrEnum):
    SCHEDULED = "scheduled"
    CLAIMED = "claimed"
    PLANNING = "planning"
    GENERATING = "generating"
    ASSEMBLING = "assembling"
    SUBTITLING = "subtitling"
    PUBLISHING = "publishing"
    SKIPPED = "skipped"
    FAILED = "failed"
    PUBLISHED = "published"


class SlotFailureReason(StrEnum):
    TERMINAL_FAILURE = "terminal_failure"
    DEADLINE = "deadline"


class StoryStage(StrEnum):
    PLANNED = "planned"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    GENERATING = "generating"
    ACCEPTED = "accepted"
    FAILED = "failed"


class GenerationStage(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    ACCEPTED = "accepted"
    FAILED = "failed"


class SubtitleState(StrEnum):
    PENDING = "pending"
    AVAILABLE = "available"
    FAILED = "failed"


class PublicationState(StrEnum):
    PENDING = "pending"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    VERIFIED = "verified"
    PUBLISHED = "published"
    CONFLICT = "conflict"
    FAILED = "failed"


class PolicyBundleIdentity(NewsModel):
    version_id: Sha256


def edition_id(
    daily_report_version_id: Sha256,
    policy_bundle_version_id: Sha256,
    selection_digest: Sha256 | None = None,
) -> EditionId:
    inputs = {
        "daily_report_version_id": daily_report_version_id,
        "policy_bundle_version_id": policy_bundle_version_id,
    }
    if selection_digest is not None:
        inputs["selection_digest"] = selection_digest
    return EditionId(sha256(canonical_json(inputs)))


class EditionIdentity(NewsModel):
    edition_id: EditionIdField
    daily_report_version_id: Sha256
    policy_bundle_version_id: Sha256
    selection_digest: Sha256 | None = None

    @model_validator(mode="after")
    def validate_edition_id(self) -> EditionIdentity:
        expected = edition_id(
            self.daily_report_version_id, self.policy_bundle_version_id, self.selection_digest
        )
        if self.edition_id != expected:
            raise ValueError("Edition ID does not match its report and policy bundle versions")
        return self


def scheduled_slot_id(name: SlotName, scheduled_at: datetime) -> SlotId:
    if scheduled_at.tzinfo is None or scheduled_at.utcoffset() is None:
        raise ValueError("Scheduled slot time must include a UTC offset")
    return SlotId(
        sha256(
            canonical_json(
                {
                    "name": name.value,
                    "scheduled_at": scheduled_at.astimezone(UTC).isoformat(),
                }
            )
        )
    )


class ScheduledSlot(NewsModel):
    slot_id: SlotIdField
    name: SlotName
    scheduled_at: AwareDatetime
    bucharest_day: date

    @model_validator(mode="after")
    def validate_slot(self) -> ScheduledSlot:
        if self.bucharest_day != self.scheduled_at.astimezone(BUCHAREST).date():
            raise ValueError("Slot day does not match the Europe/Bucharest calendar day")
        if self.slot_id != scheduled_slot_id(self.name, self.scheduled_at):
            raise ValueError("Slot ID does not match its name and scheduled time")
        return self


class SlotLease(NewsModel):
    slot_id: SlotIdField
    edition_id: EditionIdField
    owner_token: NonEmptyText
    expires_at: AwareDatetime
    claim_count: Annotated[int, Field(gt=0)]


def planned_story_id(
    edition_id_value: EditionId,
    position: int,
    report_subject_id: Sha256,
) -> StoryId:
    if position < 0:
        raise ValueError("Story position must be zero or greater")
    return StoryId(
        sha256(
            canonical_json(
                {
                    "edition_id": edition_id_value,
                    "position": position,
                    "report_subject_id": report_subject_id,
                }
            )
        )
    )


class PlannedStory(NewsModel):
    story_id: StoryIdField
    edition_id: EditionIdField
    position: Annotated[int, Field(ge=0)]
    report_subject_id: Sha256
    title: NonEmptyText
    mandatory: Literal[True] = True
    requested_duration_ms: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_story_id(self) -> PlannedStory:
        expected = planned_story_id(self.edition_id, self.position, self.report_subject_id)
        if self.story_id != expected:
            raise ValueError("Story ID does not match its edition, position, and report subject")
        return self


class DigestPlan(NewsModel):
    edition_id: EditionIdField
    artifact_version_id: Sha256
    stories: Annotated[tuple[PlannedStory, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_stories(self) -> DigestPlan:
        if any(story.edition_id != self.edition_id for story in self.stories):
            raise ValueError("Every story must belong to the plan edition")
        if tuple(story.position for story in self.stories) != tuple(range(len(self.stories))):
            raise ValueError("Story positions must be unique, ordered, and contiguous from zero")
        if len({story.report_subject_id for story in self.stories}) != len(self.stories):
            raise ValueError("Report subjects must be unique within a plan")
        return self


def generation_request_id(
    edition_id_value: EditionId,
    story_position: int,
    attempt_index: int,
    request_artifact_version_id: Sha256,
) -> GenerationRequestId:
    if story_position < 0:
        raise ValueError("Story position must be zero or greater")
    if attempt_index not in (0, 1):
        raise ValueError("Generation attempt index must be 0 or 1")
    return GenerationRequestId(
        sha256(
            canonical_json(
                {
                    "attempt_index": attempt_index,
                    "edition_id": edition_id_value,
                    "request_artifact_version_id": request_artifact_version_id,
                    "story_position": story_position,
                }
            )
        )
    )


class GenerationRequestIdentity(NewsModel):
    request_id: GenerationRequestIdField
    edition_id: EditionIdField
    story_position: Annotated[int, Field(ge=0)]
    attempt_index: Annotated[int, Field(ge=0, le=1)]
    request_artifact_version_id: Sha256

    @model_validator(mode="after")
    def validate_request_id(self) -> GenerationRequestIdentity:
        expected = generation_request_id(
            self.edition_id,
            self.story_position,
            self.attempt_index,
            self.request_artifact_version_id,
        )
        if self.request_id != expected:
            raise ValueError("Generation request ID does not match its inputs")
        return self


class PendingAttemptCost(NewsModel):
    kind: Literal["pending"] = "pending"


class EstimatedAttemptCost(NewsModel):
    kind: Literal["estimated"] = "estimated"
    usd: Annotated[Decimal, Field(ge=0)]


class MeasuredAttemptCost(NewsModel):
    kind: Literal["measured"] = "measured"
    usd: Annotated[Decimal, Field(ge=0)]


class UnknownAttemptCost(NewsModel):
    kind: Literal["unknown"] = "unknown"
    reason: NonEmptyText


AttemptCost = Annotated[
    PendingAttemptCost | EstimatedAttemptCost | MeasuredAttemptCost | UnknownAttemptCost,
    Field(discriminator="kind"),
]


class GenerationRequestState(NewsModel):
    request_id: GenerationRequestIdField
    stage: GenerationStage
    provider_receipt_id: NonEmptyText | None = None
    cost: AttemptCost


class GenerationSpend(NewsModel):
    measured_usd: Annotated[Decimal, Field(ge=0)]
    estimated_usd: Annotated[Decimal, Field(ge=0)]
    pending_requests: Annotated[int, Field(ge=0)]
    unknown_requests: Annotated[int, Field(ge=0)]


class GenerationBudgetLimits(NewsModel):
    story_usd: Annotated[Decimal, Field(gt=0)]
    edition_usd: Annotated[Decimal, Field(gt=0)]
    bucharest_day_usd: Annotated[Decimal, Field(gt=0)]
    calendar_month_usd: Annotated[Decimal, Field(gt=0)]


class GenerationAdmission(NewsModel):
    generation_policy_artifact_version_id: Sha256
    reserved_usd: Annotated[Decimal, Field(gt=0)]
    limits: GenerationBudgetLimits


class GenerationRequestCheckpoint(NewsModel):
    state: GenerationRequestState
    created: bool


class SlotSkipReason(StrEnum):
    SOURCE_MISSING = "source_missing"
    SOURCE_EMPTY = "source_empty"
    UNCHANGED = "unchanged"
    ALREADY_PUBLISHED = "already_published"
    ACTIVE_EDITION = "active_edition"
    OVERLAPPING_RUN = "overlapping_run"


class TerminalSlotState(StrEnum):
    FAILED = "failed"
    PUBLISHED = "published"


class ClaimedSlot(NewsModel):
    kind: Literal["claimed"] = "claimed"
    lease: SlotLease


class BusySlot(NewsModel):
    kind: Literal["busy"] = "busy"
    retry_at: AwareDatetime


class SkippedSlot(NewsModel):
    kind: Literal["skipped"] = "skipped"
    reason: SlotSkipReason


class TerminalSlot(NewsModel):
    kind: Literal["terminal"] = "terminal"
    state: TerminalSlotState


ClaimResult = Annotated[ClaimedSlot | SkippedSlot | TerminalSlot, Field(discriminator="kind")]
ReacquireResult = Annotated[ClaimedSlot | BusySlot | TerminalSlot, Field(discriminator="kind")]


class AvailableSubtitles(NewsModel):
    kind: Literal["available"] = "available"
    artifact_version_id: Sha256


class FailedSubtitles(NewsModel):
    kind: Literal["failed"] = "failed"
    evidence_artifact_version_id: Sha256


SubtitleOutcome = Annotated[
    AvailableSubtitles | FailedSubtitles,
    Field(discriminator="kind"),
]


def _normalized_public_object_key(value: str) -> str:
    if value.startswith("/") or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("Public object key must be a normalized relative path")
    return value


PublicObjectKey = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$",
    ),
    AfterValidator(_normalized_public_object_key),
]


class SubtitleObjectMetadata(NewsModel):
    expected_key: PublicObjectKey
    content_digest: Sha256
    byte_size: Annotated[int, Field(gt=0)]
    media_type: Literal["text/vtt"]


def publication_id(
    *,
    edition_id_value: EditionId,
    expected_video_key: str,
    video_digest: Sha256,
    video_byte_size: int,
    video_media_type: str,
    subtitle: SubtitleObjectMetadata | None,
    source_video_version_id: Sha256,
    source_subtitle_version_id: Sha256 | None,
    cache_control: str = PUBLIC_MEDIA_CACHE_CONTROL,
    visibility: str = PUBLIC_MEDIA_VISIBILITY,
    retention: str = PUBLIC_MEDIA_RETENTION,
) -> PublicationId:
    return PublicationId(
        sha256(
            canonical_json(
                {
                    "edition_id": edition_id_value,
                    "expected_video_key": expected_video_key,
                    "cache_control": cache_control,
                    "retention": retention,
                    "source_subtitle_version_id": source_subtitle_version_id,
                    "source_video_version_id": source_video_version_id,
                    "subtitle": subtitle.model_dump(mode="json") if subtitle is not None else None,
                    "video_byte_size": video_byte_size,
                    "video_digest": video_digest,
                    "video_media_type": video_media_type,
                    "visibility": visibility,
                }
            )
        )
    )


class PublicationIntent(NewsModel):
    publication_id: PublicationIdField
    edition_id: EditionIdField
    expected_video_key: PublicObjectKey
    video_digest: Sha256
    video_byte_size: Annotated[int, Field(gt=0)]
    video_media_type: Literal["video/mp4"]
    subtitle: SubtitleObjectMetadata | None = None
    source_video_version_id: Sha256
    source_subtitle_version_id: Sha256 | None = None
    cache_control: Literal["public,max-age=31536000,immutable"] = PUBLIC_MEDIA_CACHE_CONTROL
    visibility: Literal["public"] = PUBLIC_MEDIA_VISIBILITY
    retention: Literal["permanent"] = PUBLIC_MEDIA_RETENTION

    @model_validator(mode="after")
    def validate_publication(self) -> PublicationIntent:
        if (self.subtitle is None) != (self.source_subtitle_version_id is None):
            raise ValueError(
                "Subtitle object metadata and source subtitle version must be provided together"
            )
        if self.edition_id not in self.expected_video_key or self.video_digest not in (
            self.expected_video_key
        ):
            raise ValueError("Public video key must include the edition ID and content digest")
        if not self.expected_video_key.endswith(".mp4"):
            raise ValueError("Public video key must end in .mp4")
        if self.subtitle is not None and (
            self.edition_id not in self.subtitle.expected_key
            or self.subtitle.content_digest not in self.subtitle.expected_key
            or not self.subtitle.expected_key.endswith(".vtt")
        ):
            raise ValueError(
                "Public subtitle key must include the edition ID and content digest and end in .vtt"
            )
        expected = publication_id(
            edition_id_value=self.edition_id,
            expected_video_key=self.expected_video_key,
            video_digest=self.video_digest,
            video_byte_size=self.video_byte_size,
            video_media_type=self.video_media_type,
            subtitle=self.subtitle,
            source_video_version_id=self.source_video_version_id,
            source_subtitle_version_id=self.source_subtitle_version_id,
            cache_control=self.cache_control,
            visibility=self.visibility,
            retention=self.retention,
        )
        if self.publication_id != expected:
            raise ValueError("Publication ID does not match its intent")
        return self


class AssembledVideo(NewsModel):
    edition_id: EditionIdField
    artifact_version_id: Sha256


class PublicationStatus(NewsModel):
    publication_id: PublicationIdField
    edition_id: EditionIdField
    stage: PublicationState


class PublicationAttemptReady(NewsModel):
    kind: Literal["ready"] = "ready"
    attempt_index: Annotated[int, Field(ge=0, le=4)]


class PublicationAttemptWaiting(NewsModel):
    kind: Literal["waiting"] = "waiting"
    attempt_index: Annotated[int, Field(ge=0, le=3)]
    retry_at: AwareDatetime


class PublicationCheckpointSuperseded(NewsModel):
    kind: Literal["superseded"] = "superseded"
    status: PublicationStatus
    retry_at: AwareDatetime | None = None


PublicationAttempt = Annotated[
    PublicationAttemptReady | PublicationAttemptWaiting | PublicationCheckpointSuperseded,
    Field(discriminator="kind"),
]


class UploadingPublication(NewsModel):
    kind: Literal["uploading"] = "uploading"


class UploadedPublication(NewsModel):
    kind: Literal["uploaded"] = "uploaded"
    evidence_artifact_version_id: Sha256


class VerifiedPublicObject(NewsModel):
    content_digest: Sha256
    byte_size: Annotated[int, Field(gt=0)]
    media_type: NonEmptyText
    source_artifact_version_id: Sha256


class VerifiedPublication(NewsModel):
    kind: Literal["verified"] = "verified"
    evidence_artifact_version_id: Sha256
    video: VerifiedPublicObject
    subtitle: VerifiedPublicObject | None = None


PublicationProgress = Annotated[
    UploadingPublication | UploadedPublication | VerifiedPublication,
    Field(discriminator="kind"),
]


class PublishedPublication(NewsModel):
    publication_id: PublicationIdField
    edition_id: EditionIdField
    published_at: AwareDatetime


class PublishedMedia(NewsModel):
    url: HttpUrl
    content_digest: Sha256
    byte_size: Annotated[int, Field(gt=0)]
    media_type: NonEmptyText


class PublishedSubtitleAvailable(NewsModel):
    kind: Literal["available"] = "available"
    media: PublishedMedia


class PublishedSubtitleFailed(NewsModel):
    kind: Literal["failed"] = "failed"


PublishedSubtitle = Annotated[
    PublishedSubtitleAvailable | PublishedSubtitleFailed,
    Field(discriminator="kind"),
]


class PublishedStory(NewsModel):
    story_id: StoryIdField
    position: Annotated[int, Field(ge=0)]
    report_subject_id: Sha256
    title: NonEmptyText
    requested_duration_ms: Annotated[int, Field(gt=0)]


class PublishedEditionSummary(NewsModel):
    edition_id: EditionIdField
    publication_id: PublicationIdField
    day: date
    slot_name: SlotName
    scheduled_at: AwareDatetime
    published_at: AwareDatetime
    daily_report_version_id: Sha256
    video: PublishedMedia
    subtitle: PublishedSubtitle


class PublishedEdition(PublishedEditionSummary):
    stories: Annotated[tuple[PublishedStory, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_story_order(self) -> PublishedEdition:
        if tuple(story.position for story in self.stories) != tuple(range(len(self.stories))):
            raise ValueError("Published stories must be ordered and contiguous from zero")
        return self
