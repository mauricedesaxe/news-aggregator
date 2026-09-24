from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import ArtifactFile
from romanian_news.catalog.video_digest import PlanningAttemptReference
from romanian_news.video_digest.models import EditionId, SlotId, SlotLease
from romanian_news.video_digest.orchestration import ActionAdvanced, PlanAction
from romanian_news.video_digest.preflight import (
    PRODUCTION_POLICY,
    PlanningExhaustedError,
    PlanningExhaustion,
    PlanningReport,
)
from romanian_news.worker import video_digest_planning as planning

NOW = datetime(2026, 9, 25, 5, tzinfo=UTC)
LEASE = SlotLease(
    slot_id=SlotId("1" * 64),
    edition_id=EditionId("2" * 64),
    owner_token="dagster-run",
    expires_at=NOW + timedelta(minutes=10),
    claim_count=1,
)


def _attempt(index: int) -> PlanningAttemptReference:
    return PlanningAttemptReference(
        attempt_index=index,
        disposition="rejected",
        evidence=ArtifactReference(
            artifact_id=f"attempt-{index}",
            version_id=str(index + 3) * 64,
            content_digest="a" * 64,
            r2_key=f"attempt-{index}.json",
        ),
        accepted_plan_artifact_version_id=None,
    )


def test_planning_exhaustion_persists_one_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = cast(PlanningReport, object())
    policy = PRODUCTION_POLICY
    renewed = LEASE.model_copy(update={"expires_at": LEASE.expires_at + timedelta(minutes=1)})
    monkeypatch.setattr(planning, "renew_slot", lambda *_args, **_kwargs: renewed)

    def exhausted(
        lease: SlotLease,
        value: PlanningReport,
        used_policy: object,
        *,
        renew_lease,
    ) -> None:
        assert (lease, value, used_policy) == (LEASE, report, policy)
        assert renew_lease(lease) == renewed
        raise PlanningExhaustedError(
            PlanningExhaustion.model_construct(edition_id=LEASE.edition_id, attempts=())
        )

    monkeypatch.setattr(planning, "prepare_paid_generation", exhausted)
    monkeypatch.setattr(
        planning, "read_planning_attempts", lambda *_args: tuple(map(_attempt, range(3)))
    )
    published: list[tuple[str, bytes]] = []
    monkeypatch.setattr(planning, "publish_immutable_r2_objects", published.extend)
    failures: list[tuple[SlotLease, ArtifactFile]] = []
    monkeypatch.setattr(
        planning,
        "fail_slot",
        lambda lease, *, evidence_file, recorded_at: failures.append((lease, evidence_file)),
    )

    assert planning.ProductionPlanningPort(report, policy).execute(PlanAction(lease=LEASE)) == (
        ActionAdvanced()
    )
    assert len(published) == len(failures) == 1
    lease, evidence = failures[0]
    assert lease == renewed
    assert evidence.artifact_id == f"{LEASE.slot_id}:failure"
    assert evidence.artifact_kind == "video_digest_failure"
    assert evidence.content == published[0][1]
    assert evidence.r2_key == published[0][0]
    assert b'"attempt_version_ids"' in evidence.content


def test_partial_planning_failure_cannot_be_terminalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planning,
        "prepare_paid_generation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PlanningExhaustedError(
                PlanningExhaustion.model_construct(edition_id=LEASE.edition_id, attempts=())
            )
        ),
    )
    monkeypatch.setattr(planning, "read_planning_attempts", lambda *_args: (_attempt(0),))
    monkeypatch.setattr(
        planning,
        "publish_immutable_r2_objects",
        lambda *_args: pytest.fail("partial attempts published terminal failure"),
    )
    with pytest.raises(ValueError, match="persisted attempts"):
        planning.ProductionPlanningPort(cast(PlanningReport, object())).execute(
            PlanAction(lease=LEASE)
        )
