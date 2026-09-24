from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from romanian_news.catalog.artifacts import artifact_file
from romanian_news.catalog.video_digest import fail_slot, read_planning_attempts, renew_slot
from romanian_news.identity import canonical_json, sha256
from romanian_news.storage import publish_immutable_r2_objects
from romanian_news.video_digest.models import SlotLease
from romanian_news.video_digest.orchestration import LEASE_DURATION, ActionAdvanced, PlanAction
from romanian_news.video_digest.preflight import (
    PRODUCTION_POLICY,
    PlanningExhaustedError,
    PlanningPolicy,
    PlanningReport,
    prepare_paid_generation,
)


@dataclass(frozen=True)
class ProductionPlanningPort:
    report: PlanningReport
    policy: PlanningPolicy = PRODUCTION_POLICY

    def execute(self, action: PlanAction) -> ActionAdvanced:
        current_lease = action.lease

        def renew(lease: SlotLease) -> SlotLease:
            nonlocal current_lease
            current_lease = renew_slot(
                lease,
                now=datetime.now(UTC),
                lease_duration=LEASE_DURATION,
            )
            return current_lease

        try:
            prepare_paid_generation(
                action.lease,
                self.report,
                self.policy,
                renew_lease=renew,
            )
        except PlanningExhaustedError:
            attempts = read_planning_attempts(action.lease.edition_id)
            if len(attempts) != 3 or any(item.disposition != "rejected" for item in attempts):
                raise ValueError("Planning exhaustion does not match persisted attempts") from None
            content = canonical_json(
                {
                    "kind": "planning_exhausted",
                    "edition_id": action.lease.edition_id,
                    "attempt_version_ids": [item.evidence.version_id for item in attempts],
                }
            )
            evidence = artifact_file(
                artifact_id=f"{action.lease.slot_id}:failure",
                artifact_kind="video_digest_failure",
                title="Video digest planning exhausted",
                content=content,
                r2_key=(
                    f"news/video-digest/{action.lease.edition_id}/failure/"
                    f"planning-{sha256(content)}.json"
                ),
                media_type="application/json",
            )
            publish_immutable_r2_objects(((evidence.r2_key, evidence.content),))
            fail_slot(current_lease, evidence_file=evidence, recorded_at=datetime.now(UTC))
        return ActionAdvanced()
