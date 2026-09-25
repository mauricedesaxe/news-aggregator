from datetime import UTC, datetime
from typing import cast

import pytest

from romanian_news.catalog import report_inputs, video_digest
from romanian_news.video_digest.models import (
    EditionId,
    ScheduledSlot,
    SlotName,
    SlotSkipReason,
    scheduled_slot_id,
)
from romanian_news.video_digest.orchestration import (
    ActionAdvanced,
    AssemblyPort,
    CatalogPort,
    DomainPorts,
    GenerationPort,
    NoAlert,
    PlanningPort,
    PublicationPort,
    RunSkipped,
    ScheduledResume,
    SkippedResume,
    SubtitlePort,
    VideoDigestRunRequest,
)
from romanian_news.worker import video_digest_runtime

AT = datetime(2026, 9, 25, 5, tzinfo=UTC)
SLOT = ScheduledSlot(
    slot_id=scheduled_slot_id(SlotName.MORNING, AT),
    name=SlotName.MORNING,
    scheduled_at=AT,
    bucharest_day=AT.date(),
)


class _Port:
    def execute(self, _action: object) -> ActionAdvanced:
        return ActionAdvanced()


class _Catalog:
    def __init__(self, state: ScheduledResume | SkippedResume) -> None:
        self.state = state

    def read(self, _slot_id: str) -> ScheduledResume | SkippedResume:
        return self.state


def test_resumed_slot_does_not_recheck_current_report(monkeypatch: pytest.MonkeyPatch) -> None:
    state = SkippedResume(slot=SLOT, reason=SlotSkipReason.UNCHANGED)
    catalog = _Catalog(state)
    port = _Port()
    ports = DomainPorts(
        planning=cast(PlanningPort, cast(object, port)),
        generation=cast(GenerationPort, cast(object, port)),
        assembly=cast(AssemblyPort, cast(object, port)),
        subtitles=cast(SubtitlePort, cast(object, port)),
        publication=cast(PublicationPort, cast(object, port)),
    )
    captured: list[VideoDigestRunRequest] = []

    def resolve(*_args: object, **_kwargs: object) -> VideoDigestRunRequest:
        raise AssertionError("A resumed slot must not recheck the current report")

    def run(request: VideoDigestRunRequest, *_args: object) -> tuple[RunSkipped, NoAlert]:
        captured.append(request)
        return RunSkipped(reason=SlotSkipReason.UNCHANGED), NoAlert()

    monkeypatch.setattr(video_digest_runtime, "resolve_video_digest_run_request", resolve)
    monkeypatch.setattr(video_digest_runtime, "run_video_digest", run)
    runtime = video_digest_runtime.ProductionVideoDigestRuntime(
        generation=cast(GenerationPort, cast(object, port)),
        catalog=cast(CatalogPort, cast(object, catalog)),
        ports=ports,
    )
    outcome, _ = runtime.run(SLOT.slot_id, owner_token="run-1")
    assert isinstance(outcome, RunSkipped)
    assert outcome.reason == SlotSkipReason.UNCHANGED
    assert captured[0].slot == SLOT


def test_exact_daily_report_file_rejects_missing_or_duplicate_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report_inputs, "catalog_query", lambda *_args: [])
    with pytest.raises(ValueError, match="Exact daily report file is unavailable"):
        report_inputs.read_exact_daily_report_file("1" * 64)


def test_edition_identity_rejects_missing_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_digest, "catalog_query", lambda *_args: [])
    with pytest.raises(ValueError, match="Video digest edition is unavailable"):
        video_digest.read_edition_identity(EditionId("1" * 64))
