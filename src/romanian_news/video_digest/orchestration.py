from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import ceil
from typing import Annotated, Literal, Protocol

from pydantic import AwareDatetime, Field

from romanian_news import NewsModel, Sha256
from romanian_news.catalog.artifacts import CatalogArtifactReference, canonical_json, sha256
from romanian_news.video_digest.models import (
    BusySlot,
    ClaimedSlot,
    EditionIdentity,
    ScheduledSlot,
    SkippedSlot,
    SlotFailureReason,
    SlotLease,
    SlotSkipReason,
    TerminalSlot,
)

SLOT_DEADLINE = timedelta(minutes=90)
LEASE_DURATION = timedelta(minutes=10)
MAX_RUNNER_ACTIONS = 64
MAX_RETRY_SECONDS = 300


class MediaArtifactReference(CatalogArtifactReference):
    byte_size: Annotated[int, Field(gt=0)]
    media_type: str


class AssemblyAttemptReference(NewsModel):
    attempt_index: Annotated[int, Field(ge=0, le=2)]
    disposition: Literal["failed", "succeeded"]
    evidence: CatalogArtifactReference
    video: MediaArtifactReference | None = None
    manifest: CatalogArtifactReference | None = None


class SubtitleAttemptReference(NewsModel):
    attempt_index: Annotated[int, Field(ge=0, le=2)]
    strategy: Literal[
        "whole-edition-v1",
        "per-story-v1",
        "per-story-without-vad-v1",
    ]
    disposition: Literal["failed", "succeeded"]
    evidence: CatalogArtifactReference
    subtitle: MediaArtifactReference | None = None


class SourceUnavailable(NewsModel):
    kind: Literal["unavailable"] = "unavailable"
    reason: Literal[
        SlotSkipReason.SOURCE_MISSING,
        SlotSkipReason.SOURCE_EMPTY,
        SlotSkipReason.UNCHANGED,
    ]


class SourceReady(NewsModel):
    kind: Literal["ready"] = "ready"
    edition: EditionIdentity


SlotSource = Annotated[SourceUnavailable | SourceReady, Field(discriminator="kind")]


class VideoDigestRunRequest(NewsModel):
    slot: ScheduledSlot
    owner_token: str
    source: SlotSource


class ScheduledResume(NewsModel):
    kind: Literal["scheduled"] = "scheduled"
    slot: ScheduledSlot


class PlanningResume(NewsModel):
    kind: Literal["planning"] = "planning"
    slot: ScheduledSlot
    lease: SlotLease


class GenerationResume(NewsModel):
    kind: Literal["generation"] = "generation"
    slot: ScheduledSlot
    lease: SlotLease


class AssemblyResume(NewsModel):
    kind: Literal["assembly"] = "assembly"
    slot: ScheduledSlot
    lease: SlotLease
    attempts: tuple[AssemblyAttemptReference, ...]


class SubtitleResume(NewsModel):
    kind: Literal["subtitles"] = "subtitles"
    slot: ScheduledSlot
    lease: SlotLease
    video: MediaArtifactReference
    attempts: tuple[SubtitleAttemptReference, ...]


class AvailablePublicationSubtitles(NewsModel):
    kind: Literal["available"] = "available"
    artifact: MediaArtifactReference


class FailedPublicationSubtitles(NewsModel):
    kind: Literal["failed"] = "failed"


PublicationSubtitles = Annotated[
    AvailablePublicationSubtitles | FailedPublicationSubtitles,
    Field(discriminator="kind"),
]


class PublicationHandoff(NewsModel):
    slot_id: str
    edition_id: str
    slot_name: str
    scheduled_at: AwareDatetime
    video: MediaArtifactReference
    subtitles: PublicationSubtitles


class PublicationResume(NewsModel):
    kind: Literal["publication"] = "publication"
    slot: ScheduledSlot
    lease: SlotLease
    handoff: PublicationHandoff


class SkippedResume(NewsModel):
    kind: Literal["skipped"] = "skipped"
    slot: ScheduledSlot
    reason: SlotSkipReason


class FailedResume(NewsModel):
    kind: Literal["failed"] = "failed"
    slot: ScheduledSlot
    reason: SlotFailureReason


class PublishedResume(NewsModel):
    kind: Literal["published"] = "published"
    slot: ScheduledSlot


SlotResumeState = Annotated[
    ScheduledResume
    | PlanningResume
    | GenerationResume
    | AssemblyResume
    | SubtitleResume
    | PublicationResume
    | SkippedResume
    | FailedResume
    | PublishedResume,
    Field(discriminator="kind"),
]


class PlanAction(NewsModel):
    kind: Literal["plan"] = "plan"
    lease: SlotLease


class GenerateAction(NewsModel):
    kind: Literal["generate"] = "generate"
    lease: SlotLease


class AssembleAction(NewsModel):
    kind: Literal["assemble"] = "assemble"
    lease: SlotLease
    attempt_index: Annotated[int, Field(ge=0, le=2)]


class SubtitleAction(NewsModel):
    kind: Literal["subtitle"] = "subtitle"
    lease: SlotLease
    attempt_index: Annotated[int, Field(ge=0, le=2)]
    strategy: Literal[
        "whole-edition-v1",
        "per-story-v1",
        "per-story-without-vad-v1",
    ]


class PublishAction(NewsModel):
    kind: Literal["publish"] = "publish"
    lease: SlotLease
    handoff: PublicationHandoff


NextAction = PlanAction | GenerateAction | AssembleAction | SubtitleAction | PublishAction


class ActionAdvanced(NewsModel):
    kind: Literal["advanced"] = "advanced"


class ActionWaiting(NewsModel):
    kind: Literal["waiting"] = "waiting"
    reason: str
    retry_after_seconds: Annotated[int, Field(ge=1, le=MAX_RETRY_SECONDS)]


ActionOutcome = ActionAdvanced | ActionWaiting


class RunSkipped(NewsModel):
    kind: Literal["skipped"] = "skipped"
    reason: SlotSkipReason


class RunPublished(NewsModel):
    kind: Literal["published"] = "published"


class RunFailed(NewsModel):
    kind: Literal["failed"] = "failed"
    reason: SlotFailureReason = SlotFailureReason.TERMINAL_FAILURE


class RunDeferred(NewsModel):
    kind: Literal["deferred"] = "deferred"
    reason: str
    retry_after_seconds: Annotated[int, Field(ge=1, le=MAX_RETRY_SECONDS)]


VideoDigestRunOutcome = RunSkipped | RunPublished | RunFailed | RunDeferred


class NoAlert(NewsModel):
    kind: Literal["none"] = "none"


class IncidentAlert(NewsModel):
    kind: Literal["incident"] = "incident"
    alert_id: Sha256
    category: Literal["terminal_failure", "deadline"]
    slot_id: str


AlertDisposition = NoAlert | IncidentAlert


class CatalogPort(Protocol):
    def schedule(self, slot: ScheduledSlot, *, recorded_at: datetime) -> None: ...

    def read(self, slot_id: str) -> SlotResumeState: ...

    def skip(
        self, slot_id: str, reason: SlotSkipReason, *, recorded_at: datetime
    ) -> SkippedSlot: ...

    def claim(
        self,
        slot_id: str,
        edition: EditionIdentity,
        *,
        owner_token: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> ClaimedSlot | SkippedSlot | TerminalSlot: ...

    def reacquire(
        self,
        slot_id: str,
        *,
        owner_token: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> ClaimedSlot | BusySlot | TerminalSlot: ...

    def renew(self, lease: SlotLease, *, now: datetime, lease_duration: timedelta) -> SlotLease: ...

    def fail_deadline(self, lease: SlotLease, *, recorded_at: datetime) -> TerminalSlot: ...


class PlanningPort(Protocol):
    def execute(self, action: PlanAction) -> ActionOutcome: ...


class GenerationPort(Protocol):
    def execute(self, action: GenerateAction) -> ActionOutcome: ...


class AssemblyPort(Protocol):
    def execute(self, action: AssembleAction) -> ActionOutcome: ...


class SubtitlePort(Protocol):
    def execute(self, action: SubtitleAction) -> ActionOutcome: ...


class PublicationPort(Protocol):
    def execute(self, action: PublishAction) -> ActionOutcome: ...


@dataclass(frozen=True)
class DomainPorts:
    planning: PlanningPort
    generation: GenerationPort
    assembly: AssemblyPort
    subtitles: SubtitlePort
    publication: PublicationPort


def next_action(state: SlotResumeState) -> NextAction | None:
    if isinstance(state, PlanningResume):
        return PlanAction(lease=state.lease)
    if isinstance(state, GenerationResume):
        return GenerateAction(lease=state.lease)
    if isinstance(state, AssemblyResume):
        if len(state.attempts) >= 3:
            return None
        return AssembleAction(lease=state.lease, attempt_index=len(state.attempts))
    if isinstance(state, SubtitleResume):
        if len(state.attempts) >= 3:
            return None
        strategies = (
            "whole-edition-v1",
            "per-story-v1",
            "per-story-without-vad-v1",
        )
        index = len(state.attempts)
        return SubtitleAction(
            lease=state.lease,
            attempt_index=index,
            strategy=strategies[index],
        )
    if isinstance(state, PublicationResume):
        return PublishAction(lease=state.lease, handoff=state.handoff)
    return None


def alert_disposition(slot_id: str, outcome: VideoDigestRunOutcome) -> AlertDisposition:
    if not isinstance(outcome, RunFailed):
        return NoAlert()
    category: Literal["terminal_failure", "deadline"] = outcome.reason.value
    return IncidentAlert(
        alert_id=sha256(canonical_json({"category": category, "slot_id": slot_id})),
        category=category,
        slot_id=slot_id,
    )


def run_video_digest(
    request: VideoDigestRunRequest,
    catalog: CatalogPort,
    ports: DomainPorts,
    *,
    now: Callable[[], datetime] | None = None,
    max_actions: int = MAX_RUNNER_ACTIONS,
) -> tuple[VideoDigestRunOutcome, AlertDisposition]:
    clock = now or (lambda: datetime.now(UTC))
    recorded_at = _utc(clock())
    catalog.schedule(request.slot, recorded_at=recorded_at)
    state = catalog.read(request.slot.slot_id)

    for _ in range(max_actions):
        if isinstance(state, ScheduledResume):
            if isinstance(request.source, SourceUnavailable):
                catalog.skip(request.slot.slot_id, request.source.reason, recorded_at=_utc(clock()))
            else:
                catalog.claim(
                    request.slot.slot_id,
                    request.source.edition,
                    owner_token=request.owner_token,
                    now=_utc(clock()),
                    lease_duration=LEASE_DURATION,
                )
            state = catalog.read(request.slot.slot_id)
            continue
        terminal = _terminal_outcome(state)
        if terminal is not None:
            return terminal, alert_disposition(request.slot.slot_id, terminal)

        if not isinstance(
            state,
            PlanningResume | GenerationResume | AssemblyResume | SubtitleResume | PublicationResume,
        ):
            raise AssertionError("Unhandled video digest resume state")
        acquisition = catalog.reacquire(
            request.slot.slot_id,
            owner_token=request.owner_token,
            now=_utc(clock()),
            lease_duration=LEASE_DURATION,
        )
        if isinstance(acquisition, BusySlot):
            retry_after = _retry_after_seconds(_utc(clock()), acquisition.retry_at)
            outcome = RunDeferred(
                reason="video digest slot lease is busy",
                retry_after_seconds=retry_after,
            )
            return outcome, alert_disposition(request.slot.slot_id, outcome)
        state = catalog.read(request.slot.slot_id)
        terminal = _terminal_outcome(state)
        if terminal is not None:
            return terminal, alert_disposition(request.slot.slot_id, terminal)
        if not isinstance(acquisition, ClaimedSlot):
            raise AssertionError("Terminal reacquisition did not reload a terminal slot")
        if not isinstance(
            state,
            PlanningResume | GenerationResume | AssemblyResume | SubtitleResume | PublicationResume,
        ):
            raise AssertionError("Reacquired video digest slot is not active")
        if _utc(clock()) >= request.slot.scheduled_at.astimezone(UTC) + SLOT_DEADLINE:
            catalog.fail_deadline(acquisition.lease, recorded_at=_utc(clock()))
            outcome = RunFailed(reason=SlotFailureReason.DEADLINE)
            return outcome, alert_disposition(request.slot.slot_id, outcome)
        renewed = catalog.renew(
            acquisition.lease,
            now=_utc(clock()),
            lease_duration=LEASE_DURATION,
        )
        state = state.model_copy(update={"lease": renewed})

        action = next_action(state)
        if action is None:
            outcome = RunFailed()
            return outcome, alert_disposition(request.slot.slot_id, outcome)
        execution = _execute(action, ports)
        if isinstance(execution, ActionWaiting):
            outcome = RunDeferred(
                reason=execution.reason,
                retry_after_seconds=execution.retry_after_seconds,
            )
            return outcome, alert_disposition(request.slot.slot_id, outcome)
        reloaded = catalog.read(request.slot.slot_id)
        if reloaded == state:
            outcome = RunDeferred(
                reason="durable state did not advance",
                retry_after_seconds=MAX_RETRY_SECONDS,
            )
            return outcome, alert_disposition(request.slot.slot_id, outcome)
        state = reloaded

    outcome = RunDeferred(
        reason="runner action bound reached",
        retry_after_seconds=MAX_RETRY_SECONDS,
    )
    return outcome, alert_disposition(request.slot.slot_id, outcome)


def _execute(action: NextAction, ports: DomainPorts) -> ActionOutcome:
    if isinstance(action, PlanAction):
        return ports.planning.execute(action)
    if isinstance(action, GenerateAction):
        return ports.generation.execute(action)
    if isinstance(action, AssembleAction):
        return ports.assembly.execute(action)
    if isinstance(action, SubtitleAction):
        return ports.subtitles.execute(action)
    return ports.publication.execute(action)


def _terminal_outcome(state: SlotResumeState) -> VideoDigestRunOutcome | None:
    if isinstance(state, SkippedResume):
        return RunSkipped(reason=state.reason)
    if isinstance(state, FailedResume):
        return RunFailed(reason=state.reason)
    if isinstance(state, PublishedResume):
        return RunPublished()
    return None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Runner time must include a UTC offset")
    return value.astimezone(UTC)


def _retry_after_seconds(now: datetime, retry_at: datetime) -> int:
    remaining = ceil((_utc(retry_at) - now).total_seconds())
    return min(MAX_RETRY_SECONDS, max(1, remaining))
