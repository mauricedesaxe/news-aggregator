from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import cast

import pytest

from romanian_news.video_digest.errors import VideoDigestCheckpointConflictError
from romanian_news.video_digest.models import (
    BusySlot,
    ClaimedSlot,
    EditionIdentity,
    ScheduledSlot,
    SkippedSlot,
    SlotFailureReason,
    SlotLease,
    SlotName,
    SlotSkipReason,
    TerminalSlot,
    TerminalSlotState,
    edition_id,
    scheduled_slot_id,
)
from romanian_news.video_digest.orchestration import (
    ActionAdvanced,
    ActionWaiting,
    AssembleAction,
    AssemblyAttemptReference,
    AssemblyResume,
    CatalogPort,
    DomainPorts,
    FailedPublicationSubtitles,
    FailedResume,
    GenerateAction,
    GenerationResume,
    IncidentAlert,
    MediaArtifactReference,
    NextAction,
    NoAlert,
    PlanningResume,
    PublicationHandoff,
    PublicationResume,
    PublishAction,
    PublishedResume,
    RunDeferred,
    RunFailed,
    RunPublished,
    RunSkipped,
    ScheduledResume,
    SkippedResume,
    SlotResumeState,
    SourceReady,
    SourceUnavailable,
    SubtitleAction,
    SubtitleAttemptReference,
    SubtitleResume,
    VideoDigestRunRequest,
    alert_disposition,
    next_action,
    run_video_digest,
)

NOW = datetime(2026, 9, 21, 5, tzinfo=UTC)
SLOT = ScheduledSlot(
    slot_id=scheduled_slot_id(SlotName.MORNING, NOW),
    name=SlotName.MORNING,
    scheduled_at=NOW,
    bucharest_day=date(2026, 9, 21),
)
EDITION = EditionIdentity(
    edition_id=edition_id("1" * 64, "2" * 64),
    daily_report_version_id="1" * 64,
    policy_bundle_version_id="2" * 64,
)
LEASE = SlotLease(
    slot_id=SLOT.slot_id,
    edition_id=EDITION.edition_id,
    owner_token="owner",
    expires_at=NOW + timedelta(minutes=10),
    claim_count=1,
)
VIDEO = MediaArtifactReference(
    artifact_id=f"{EDITION.edition_id}:assembled-video",
    version_id="3" * 64,
    content_digest="4" * 64,
    r2_key="video-digest/video.mp4",
    byte_size=100,
    media_type="video/mp4",
)
HANDOFF = PublicationHandoff(
    slot_id=SLOT.slot_id,
    edition_id=EDITION.edition_id,
    slot_name=SLOT.name.value,
    scheduled_at=SLOT.scheduled_at,
    video=VIDEO,
    subtitles=FailedPublicationSubtitles(),
)


class _Catalog:
    def __init__(self, state: SlotResumeState) -> None:
        self.state: SlotResumeState = state
        self.renewed = 0
        self.skips: list[SlotSkipReason] = []
        self.deadline_failed = False
        self.events: list[str] = []
        self.reacquire_result: ClaimedSlot | BusySlot | TerminalSlot | None = None

    def schedule(self, slot: ScheduledSlot, *, recorded_at: datetime) -> None:
        assert slot == SLOT
        assert recorded_at.tzinfo is UTC

    def read(self, slot_id: str) -> SlotResumeState:
        assert slot_id == SLOT.slot_id
        self.events.append("read")
        return self.state

    def skip(self, slot_id: str, reason: SlotSkipReason, *, recorded_at: datetime) -> SkippedSlot:
        assert slot_id == SLOT.slot_id
        assert recorded_at.tzinfo is UTC
        self.skips.append(reason)
        self.state = SkippedResume(slot=SLOT, reason=reason)
        return SkippedSlot(reason=reason)

    def claim(
        self,
        slot_id: str,
        edition: EditionIdentity,
        *,
        owner_token: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> ClaimedSlot | SkippedSlot | TerminalSlot:
        assert slot_id == SLOT.slot_id
        assert edition == EDITION
        assert owner_token == "owner"
        assert now.tzinfo is UTC
        assert lease_duration == timedelta(minutes=10)
        self.state = PlanningResume(slot=SLOT, lease=LEASE)
        return ClaimedSlot(lease=LEASE)

    def renew(self, lease: SlotLease, *, now: datetime, lease_duration: timedelta) -> SlotLease:
        assert lease.slot_id == SLOT.slot_id
        assert now.tzinfo is UTC
        assert lease_duration == timedelta(minutes=10)
        self.events.append("renew")
        self.renewed += 1
        renewed = lease.model_copy(update={"expires_at": now + lease_duration})
        if isinstance(
            self.state,
            PlanningResume | GenerationResume | AssemblyResume | SubtitleResume | PublicationResume,
        ):
            self.state = self.state.model_copy(update={"lease": renewed})
        return renewed

    def reacquire(
        self,
        slot_id: str,
        *,
        owner_token: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> ClaimedSlot | BusySlot | TerminalSlot:
        assert slot_id == SLOT.slot_id
        assert owner_token == "owner"
        assert now.tzinfo is UTC
        assert lease_duration == timedelta(minutes=10)
        self.events.append("reacquire")
        return self.reacquire_result or ClaimedSlot(lease=LEASE)

    def fail_deadline(self, lease: SlotLease, *, recorded_at: datetime) -> TerminalSlot:
        assert lease.slot_id == SLOT.slot_id
        assert recorded_at.tzinfo is UTC
        self.deadline_failed = True
        self.state = FailedResume(slot=SLOT, reason=SlotFailureReason.DEADLINE)
        return TerminalSlot(state=TerminalSlotState.FAILED)


class _Port:
    def __init__(self, catalog: _Catalog, replacement: object | None = None) -> None:
        self.catalog = catalog
        self.replacement = cast(SlotResumeState | None, replacement)
        self.calls: list[NextAction] = []

    def execute(self, action: NextAction) -> ActionAdvanced:
        self.calls.append(action)
        if self.replacement is not None:
            self.catalog.state = self.replacement
        return ActionAdvanced()


class _FailingAssemblyPort(_Port):
    def execute(self, action: NextAction) -> ActionAdvanced:
        self.calls.append(action)
        current = cast(AssemblyResume, self.catalog.state)
        attempt_index = len(current.attempts)
        evidence = VIDEO.model_copy(
            update={
                "artifact_id": f"assembly-failure-{attempt_index}",
                "version_id": str(attempt_index + 5) * 64,
            }
        )
        attempts = (
            *current.attempts,
            AssemblyAttemptReference(
                attempt_index=attempt_index,
                disposition="failed",
                evidence=evidence,
            ),
        )
        self.catalog.state = (
            FailedResume(slot=SLOT, reason=SlotFailureReason.TERMINAL_FAILURE)
            if len(attempts) == 3
            else current.model_copy(update={"attempts": attempts})
        )
        return ActionAdvanced()


class _FailingSubtitlePort(_Port):
    def execute(self, action: NextAction) -> ActionAdvanced:
        self.calls.append(action)
        current = cast(SubtitleResume, self.catalog.state)
        attempt_index = len(current.attempts)
        strategies = (
            "whole-edition-v1",
            "per-story-v1",
            "per-story-without-vad-v1",
        )
        evidence = VIDEO.model_copy(
            update={
                "artifact_id": f"subtitle-failure-{attempt_index}",
                "version_id": str(attempt_index + 8) * 64,
            }
        )
        attempts = (
            *current.attempts,
            SubtitleAttemptReference(
                attempt_index=attempt_index,
                strategy=strategies[attempt_index],
                disposition="failed",
                evidence=evidence,
            ),
        )
        self.catalog.state = (
            PublicationResume(slot=SLOT, lease=current.lease, handoff=HANDOFF)
            if len(attempts) == 3
            else current.model_copy(update={"attempts": attempts})
        )
        return ActionAdvanced()


class _WaitingPort:
    def __init__(self) -> None:
        self.calls: list[NextAction] = []

    def execute(self, action: NextAction) -> ActionWaiting:
        self.calls.append(action)
        return ActionWaiting(reason="provider is processing", retry_after_seconds=45)


def _ports(catalog: _Catalog, replacements: dict[str, object] | None = None) -> DomainPorts:
    values = replacements or {}
    return DomainPorts(
        planning=_Port(catalog, values.get("planning")),
        generation=_Port(catalog, values.get("generation")),
        assembly=_Port(catalog, values.get("assembly")),
        subtitles=_Port(catalog, values.get("subtitles")),
        publication=_Port(catalog, values.get("publication")),
    )


@pytest.mark.parametrize(
    "reason",
    [
        SlotSkipReason.SOURCE_MISSING,
        SlotSkipReason.SOURCE_EMPTY,
        SlotSkipReason.UNCHANGED,
    ],
)
def test_runner_durably_records_source_skips_without_alert(reason: SlotSkipReason) -> None:
    catalog = _Catalog(ScheduledResume(slot=SLOT))
    request = VideoDigestRunRequest(
        slot=SLOT,
        owner_token="owner",
        source=SourceUnavailable.model_validate({"reason": reason}),
    )

    outcome, alert = run_video_digest(
        request, cast(CatalogPort, catalog), _ports(catalog), now=lambda: NOW
    )

    assert outcome == RunSkipped(reason=reason)
    assert alert == NoAlert()
    assert catalog.skips == [reason]
    assert catalog.renewed == 0


@pytest.mark.parametrize("reason", tuple(SlotSkipReason))
def test_every_skip_reason_has_no_incident(reason: SlotSkipReason) -> None:
    outcome = RunSkipped(reason=reason)
    assert alert_disposition(SLOT.slot_id, outcome) == NoAlert()


def test_reducer_selects_three_assembly_attempts_and_fixed_subtitle_strategies() -> None:
    evidence = VIDEO.model_copy(update={"artifact_id": "evidence"})
    for index in range(3):
        assembly = AssemblyResume(
            slot=SLOT,
            lease=LEASE,
            attempts=tuple(
                AssemblyAttemptReference(
                    attempt_index=value,
                    disposition="failed",
                    evidence=evidence,
                )
                for value in range(index)
            ),
        )
        action = next_action(assembly)
        assert action is not None
        assert action.kind == "assemble"
        assert action.attempt_index == index

    strategies = (
        "whole-edition-v1",
        "per-story-v1",
        "per-story-without-vad-v1",
    )
    for index, strategy in enumerate(strategies):
        subtitles = SubtitleResume(
            slot=SLOT,
            lease=LEASE,
            video=VIDEO,
            attempts=tuple(
                SubtitleAttemptReference(
                    attempt_index=value,
                    strategy=strategies[value],
                    disposition="failed",
                    evidence=evidence,
                )
                for value in range(index)
            ),
        )
        action = next_action(subtitles)
        assert action is not None
        assert action.kind == "subtitle"
        assert (action.attempt_index, action.strategy) == (index, strategy)


@pytest.mark.parametrize(
    ("state", "port_name", "action_kind"),
    [
        (PlanningResume(slot=SLOT, lease=LEASE), "planning", "plan"),
        (GenerationResume(slot=SLOT, lease=LEASE), "generation", "generate"),
        (AssemblyResume(slot=SLOT, lease=LEASE, attempts=()), "assembly", "assemble"),
        (
            SubtitleResume(slot=SLOT, lease=LEASE, video=VIDEO, attempts=()),
            "subtitles",
            "subtitle",
        ),
        (
            PublicationResume(slot=SLOT, lease=LEASE, handoff=HANDOFF),
            "publication",
            "publish",
        ),
    ],
)
def test_runner_resumes_each_durable_stage_without_repeating_side_effects(
    state: SlotResumeState, port_name: str, action_kind: str
) -> None:
    catalog = _Catalog(state)
    ports = _ports(catalog, {port_name: PublishedResume(slot=SLOT)})
    request = VideoDigestRunRequest(
        slot=SLOT, owner_token="owner", source=SourceReady(edition=EDITION)
    )

    outcome, alert = run_video_digest(
        request,
        cast(CatalogPort, catalog),
        ports,
        now=lambda: NOW + timedelta(minutes=1),
    )

    assert outcome == RunPublished()
    assert alert == NoAlert()
    selected = cast(_Port, getattr(ports, port_name))
    assert len(selected.calls) == 1
    assert selected.calls[0].kind == action_kind
    assert catalog.renewed == 1
    assert catalog.events[:4] == ["read", "reacquire", "read", "renew"]


def test_runner_converges_across_process_restarts_without_replaying_actions() -> None:
    catalog = _Catalog(ScheduledResume(slot=SLOT))
    ports = _ports(
        catalog,
        {
            "planning": GenerationResume(slot=SLOT, lease=LEASE),
            "generation": AssemblyResume(slot=SLOT, lease=LEASE, attempts=()),
            "assembly": SubtitleResume(slot=SLOT, lease=LEASE, video=VIDEO, attempts=()),
            "subtitles": PublicationResume(slot=SLOT, lease=LEASE, handoff=HANDOFF),
            "publication": PublishedResume(slot=SLOT),
        },
    )
    request = VideoDigestRunRequest(
        slot=SLOT, owner_token="owner", source=SourceReady(edition=EDITION)
    )

    outcomes = [
        run_video_digest(
            request,
            cast(CatalogPort, catalog),
            ports,
            now=lambda: NOW + timedelta(minutes=1),
            max_actions=action_budget,
        )[0]
        for action_budget in (2, 1, 1, 1, 1, 1)
    ]

    assert (
        outcomes[:5]
        == [RunDeferred(reason="runner action bound reached", retry_after_seconds=300)] * 5
    )
    assert outcomes[5] == RunPublished()
    assert [
        len(cast(_Port, getattr(ports, name)).calls)
        for name in ("planning", "generation", "assembly", "subtitles", "publication")
    ] == [1, 1, 1, 1, 1]


def test_deadline_reacquires_before_terminalizing_without_renewal() -> None:
    catalog = _Catalog(PlanningResume(slot=SLOT, lease=LEASE))
    request = VideoDigestRunRequest(
        slot=SLOT, owner_token="owner", source=SourceReady(edition=EDITION)
    )

    outcome, alert = run_video_digest(
        request,
        cast(CatalogPort, catalog),
        _ports(catalog),
        now=lambda: NOW + timedelta(minutes=90),
    )

    assert outcome == RunFailed(reason=SlotFailureReason.DEADLINE)
    assert catalog.renewed == 0
    assert catalog.deadline_failed is True
    assert catalog.events == ["read", "reacquire", "read"]
    assert isinstance(alert, IncidentAlert)
    assert alert == alert_disposition(SLOT.slot_id, outcome)
    assert alert.category == "deadline"


def test_busy_reacquisition_returns_bounded_retry_without_renewal_or_action() -> None:
    catalog = _Catalog(PlanningResume(slot=SLOT, lease=LEASE))
    catalog.reacquire_result = BusySlot(retry_at=NOW + timedelta(minutes=10))
    ports = _ports(catalog)
    request = VideoDigestRunRequest(
        slot=SLOT, owner_token="owner", source=SourceReady(edition=EDITION)
    )

    outcome, alert = run_video_digest(request, cast(CatalogPort, catalog), ports, now=lambda: NOW)

    assert outcome == RunDeferred(
        reason="video digest slot lease is busy",
        retry_after_seconds=300,
    )
    assert alert == NoAlert()
    assert catalog.renewed == 0
    assert catalog.events == ["read", "reacquire"]
    assert cast(_Port, ports.planning).calls == []


def test_waiting_action_remains_nonterminal_with_retry_timing() -> None:
    catalog = _Catalog(PlanningResume(slot=SLOT, lease=LEASE))
    waiting = _WaitingPort()
    ports = _ports(catalog)
    ports = DomainPorts(
        planning=waiting,
        generation=ports.generation,
        assembly=ports.assembly,
        subtitles=ports.subtitles,
        publication=ports.publication,
    )
    request = VideoDigestRunRequest(
        slot=SLOT, owner_token="owner", source=SourceReady(edition=EDITION)
    )

    outcome, alert = run_video_digest(
        request, cast(CatalogPort, catalog), ports, now=lambda: NOW + timedelta(minutes=1)
    )

    assert outcome == RunDeferred(reason="provider is processing", retry_after_seconds=45)
    assert alert == NoAlert()
    assert len(waiting.calls) == 1


def test_action_bound_remains_nonterminal_with_bounded_retry_timing() -> None:
    catalog = _Catalog(PlanningResume(slot=SLOT, lease=LEASE))
    request = VideoDigestRunRequest(
        slot=SLOT, owner_token="owner", source=SourceReady(edition=EDITION)
    )

    outcome, alert = run_video_digest(
        request,
        cast(CatalogPort, catalog),
        _ports(catalog),
        now=lambda: NOW,
        max_actions=0,
    )

    assert outcome == RunDeferred(
        reason="runner action bound reached",
        retry_after_seconds=300,
    )
    assert alert == NoAlert()


def test_generation_progress_continues_when_slot_projection_is_unchanged() -> None:
    catalog = _Catalog(GenerationResume(slot=SLOT, lease=LEASE))

    class ProgressingGenerationPort:
        calls = 0

        def execute(self, action: GenerateAction) -> ActionAdvanced:
            assert action.kind == "generate"
            self.calls += 1
            if self.calls == 2:
                catalog.state = PublishedResume(slot=SLOT)
            return ActionAdvanced(durable_progress=True)

    generation = ProgressingGenerationPort()
    default_ports = _ports(catalog)
    ports = DomainPorts(
        planning=default_ports.planning,
        generation=generation,
        assembly=default_ports.assembly,
        subtitles=default_ports.subtitles,
        publication=default_ports.publication,
    )
    request = VideoDigestRunRequest(
        slot=SLOT, owner_token="owner", source=SourceReady(edition=EDITION)
    )

    outcome, alert = run_video_digest(request, cast(CatalogPort, catalog), ports, now=lambda: NOW)

    assert outcome == RunPublished()
    assert alert == NoAlert()
    assert generation.calls == 2


def test_active_state_without_a_next_action_raises_a_checkpoint_conflict() -> None:
    attempts = tuple(
        AssemblyAttemptReference(
            attempt_index=index,
            disposition="failed",
            evidence=VIDEO.model_copy(
                update={
                    "artifact_id": f"assembly-failure-{index}",
                    "version_id": str(index + 5) * 64,
                }
            ),
        )
        for index in range(3)
    )
    catalog = _Catalog(AssemblyResume(slot=SLOT, lease=LEASE, attempts=attempts))
    request = VideoDigestRunRequest(
        slot=SLOT, owner_token="owner", source=SourceReady(edition=EDITION)
    )

    with pytest.raises(
        VideoDigestCheckpointConflictError,
        match="Active video digest slot has no durable next action",
    ):
        run_video_digest(
            request,
            cast(CatalogPort, catalog),
            _ports(catalog),
            now=lambda: NOW + timedelta(minutes=1),
        )


def test_publication_port_receives_the_exact_typed_handoff() -> None:
    catalog = _Catalog(PublicationResume(slot=SLOT, lease=LEASE, handoff=HANDOFF))
    ports = _ports(catalog, {"publication": PublishedResume(slot=SLOT)})
    request = VideoDigestRunRequest(
        slot=SLOT, owner_token="owner", source=SourceReady(edition=EDITION)
    )

    outcome, _alert = run_video_digest(
        request,
        cast(CatalogPort, catalog),
        ports,
        now=lambda: NOW + timedelta(minutes=1),
    )

    assert outcome == RunPublished()
    calls = cast(_Port, ports.publication).calls
    assert len(calls) == 1
    assert cast(PublishAction, calls[0]).handoff == HANDOFF


@pytest.mark.parametrize(
    "subtitles",
    [
        {"kind": "available"},
        {"kind": "failed", "artifact": VIDEO.model_dump()},
    ],
)
def test_publication_handoff_rejects_invalid_subtitle_artifact_shapes(
    subtitles: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        PublicationHandoff.model_validate(
            {
                **HANDOFF.model_dump(),
                "subtitles": subtitles,
            }
        )


def test_terminal_failure_alerts_but_recovered_work_does_not() -> None:
    incident = alert_disposition(SLOT.slot_id, RunFailed())
    assert isinstance(incident, IncidentAlert)
    assert incident.category == "terminal_failure"


def test_resumed_deadline_failure_keeps_the_deadline_alert_category() -> None:
    catalog = _Catalog(FailedResume(slot=SLOT, reason=SlotFailureReason.DEADLINE))
    request = VideoDigestRunRequest(
        slot=SLOT, owner_token="owner", source=SourceReady(edition=EDITION)
    )

    outcome, alert = run_video_digest(
        request, cast(CatalogPort, catalog), _ports(catalog), now=lambda: NOW
    )

    assert outcome == RunFailed(reason=SlotFailureReason.DEADLINE)
    assert isinstance(alert, IncidentAlert)
    assert alert.category == "deadline"
    assert catalog.events == ["read"]


def test_three_failed_assembly_attempts_become_terminal_failure() -> None:
    catalog = _Catalog(AssemblyResume(slot=SLOT, lease=LEASE, attempts=()))
    assembly = _FailingAssemblyPort(catalog)
    ports = _ports(catalog)
    ports = DomainPorts(
        planning=ports.planning,
        generation=ports.generation,
        assembly=assembly,
        subtitles=ports.subtitles,
        publication=ports.publication,
    )
    request = VideoDigestRunRequest(
        slot=SLOT, owner_token="owner", source=SourceReady(edition=EDITION)
    )

    outcome, alert = run_video_digest(
        request,
        cast(CatalogPort, catalog),
        ports,
        now=lambda: NOW + timedelta(minutes=1),
    )

    assert outcome == RunFailed()
    assert isinstance(alert, IncidentAlert)
    assert [cast(AssembleAction, call).attempt_index for call in assembly.calls] == [0, 1, 2]


def test_three_failed_subtitle_strategies_publish_clean_video_without_alert() -> None:
    catalog = _Catalog(SubtitleResume(slot=SLOT, lease=LEASE, video=VIDEO, attempts=()))
    subtitles = _FailingSubtitlePort(catalog)
    publication = _Port(catalog, PublishedResume(slot=SLOT))
    ports = _ports(catalog)
    ports = DomainPorts(
        planning=ports.planning,
        generation=ports.generation,
        assembly=ports.assembly,
        subtitles=subtitles,
        publication=publication,
    )
    request = VideoDigestRunRequest(
        slot=SLOT, owner_token="owner", source=SourceReady(edition=EDITION)
    )

    outcome, alert = run_video_digest(
        request,
        cast(CatalogPort, catalog),
        ports,
        now=lambda: NOW + timedelta(minutes=1),
    )

    assert outcome == RunPublished()
    assert alert == NoAlert()
    assert [cast(SubtitleAction, call).strategy for call in subtitles.calls] == [
        "whole-edition-v1",
        "per-story-v1",
        "per-story-without-vad-v1",
    ]
    assert cast(PublishAction, publication.calls[0]).handoff == HANDOFF
