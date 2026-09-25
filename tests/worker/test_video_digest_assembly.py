from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from romanian_news.catalog.artifacts import ArtifactFile
from romanian_news.video_digest.models import (
    DigestPlan,
    EditionId,
    PlannedStory,
    SlotId,
    SlotLease,
    planned_story_id,
)
from romanian_news.video_digest.orchestration import ActionAdvanced, AssembleAction
from romanian_news.worker import video_digest_assembly as assembly


def _action(index: int) -> AssembleAction:
    return AssembleAction(
        lease=SlotLease(
            slot_id=SlotId("a" * 64),
            edition_id=EditionId("b" * 64),
            owner_token="run-1",
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            claim_count=1,
        ),
        attempt_index=index,
    )


def _plan(lease: SlotLease) -> DigestPlan:
    subject = "d" * 64
    return DigestPlan(
        edition_id=lease.edition_id,
        artifact_version_id="c" * 64,
        stories=(
            PlannedStory(
                story_id=planned_story_id(lease.edition_id, 0, subject),
                edition_id=lease.edition_id,
                position=0,
                report_subject_id=subject,
                title="Test story",
                requested_duration_ms=1000,
            ),
        ),
    )


def test_assembly_port_uses_accepted_plan_and_renews_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = _action(1)
    plan = _plan(action.lease)
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        assembly,
        "read_accepted_digest_plan",
        lambda edition_id: (calls.append(("read", edition_id)), (None, plan))[1],
    )

    def renew(lease: SlotLease, **_kwargs: object) -> SlotLease:
        calls.append(("renew", lease.slot_id))
        return lease

    def assemble(
        lease: SlotLease,
        accepted: DigestPlan,
        *,
        attempt_index: int,
        renew_lease: Callable[[SlotLease], SlotLease],
    ) -> None:
        assert lease == action.lease
        assert accepted == plan
        assert attempt_index == 1
        renew_lease(lease)
        calls.append(("assemble", accepted.edition_id))

    monkeypatch.setattr(assembly, "renew_slot", renew)
    monkeypatch.setattr(assembly, "assemble_edition", assemble)

    assert assembly.ProductionAssemblyPort().execute(action) == ActionAdvanced()
    assert calls == [
        ("read", action.lease.edition_id),
        ("renew", action.lease.slot_id),
        ("assemble", action.lease.edition_id),
    ]


@pytest.mark.parametrize("index", [0, 2])
def test_assembly_port_records_retry_or_terminal_failure(
    monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    action = _action(index)
    plan = _plan(action.lease)
    monkeypatch.setattr(assembly, "read_accepted_digest_plan", lambda _edition: (None, plan))
    monkeypatch.setattr(assembly, "renew_slot", lambda lease, **_kwargs: lease)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("ffmpeg", 900)

    monkeypatch.setattr(assembly, "assemble_edition", fail)
    published: list[tuple[str, bytes]] = []
    recorded: list[tuple[int, str, ArtifactFile]] = []
    monkeypatch.setattr(
        assembly,
        "publish_immutable_r2_objects",
        lambda objects: published.extend(objects),
    )
    monkeypatch.setattr(
        assembly,
        "record_assembly_attempt",
        lambda _lease, attempt_index, disposition, *, evidence_file, **_kwargs: recorded.append(
            (attempt_index, disposition, evidence_file)
        ),
    )

    assert assembly.ProductionAssemblyPort().execute(action) == ActionAdvanced()
    assert len(recorded) == 1
    assert recorded[0][:2] == (index, "failed")
    evidence = recorded[0][2]
    assert evidence.artifact_kind == (
        "video_digest_failure" if index == 2 else "video_digest_assembly_attempt"
    )
    assert json.loads(evidence.content)["code"] == "media_tool_timeout"
    assert published == [(evidence.r2_key, evidence.content)]
