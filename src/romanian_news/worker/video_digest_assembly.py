from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from romanian_news.catalog.artifacts import artifact_file
from romanian_news.catalog.video_digest import record_assembly_attempt, renew_slot
from romanian_news.identity import canonical_json, sha256
from romanian_news.storage import publish_immutable_r2_objects
from romanian_news.video_digest.media import MediaValidationError, assemble_edition
from romanian_news.video_digest.models import SlotLease
from romanian_news.video_digest.orchestration import ActionAdvanced, AssembleAction
from romanian_news.video_digest.preflight import read_accepted_digest_plan

ASSEMBLY_LEASE_DURATION = timedelta(minutes=20)


@dataclass(frozen=True)
class ProductionAssemblyPort:
    def execute(self, action: AssembleAction) -> ActionAdvanced:
        _, plan = read_accepted_digest_plan(action.lease.edition_id)
        current_lease = action.lease

        def renew(lease: SlotLease) -> SlotLease:
            nonlocal current_lease
            current_lease = renew_slot(
                lease,
                now=datetime.now(UTC),
                lease_duration=ASSEMBLY_LEASE_DURATION,
            )
            return current_lease

        try:
            _ = assemble_edition(
                current_lease,
                plan,
                attempt_index=action.attempt_index,
                renew_lease=renew,
            )
        except (
            MediaValidationError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as error:
            code = _failure_code(error)
            final = action.attempt_index == 2
            content = canonical_json(
                {
                    "kind": "assembly_attempt_failure",
                    "edition_id": current_lease.edition_id,
                    "slot_id": current_lease.slot_id,
                    "attempt_index": action.attempt_index,
                    "code": code,
                }
            )
            evidence = artifact_file(
                artifact_id=(
                    f"{current_lease.slot_id}:failure"
                    if final
                    else f"{current_lease.edition_id}:{action.attempt_index}:assembly-attempt"
                ),
                artifact_kind=(
                    "video_digest_failure" if final else "video_digest_assembly_attempt"
                ),
                title="Video digest assembly failed",
                content=content,
                r2_key=(
                    f"news/video-digest/{current_lease.edition_id}/assembly/"
                    f"attempt-{action.attempt_index}-{sha256(content)}.json"
                ),
                media_type="application/json",
            )
            _ = publish_immutable_r2_objects(((evidence.r2_key, evidence.content),))
            _ = renew(current_lease)
            _ = record_assembly_attempt(
                current_lease,
                action.attempt_index,
                "failed",
                evidence_file=evidence,
                recorded_at=datetime.now(UTC),
            )
        return ActionAdvanced()


def _failure_code(error: Exception) -> str:
    if isinstance(error, MediaValidationError):
        return error.code
    if isinstance(error, subprocess.TimeoutExpired):
        return "media_tool_timeout"
    if isinstance(error, FileNotFoundError):
        return "media_tool_missing"
    return "media_tool_failed"
