from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, LiteralString, TypeVar, cast

from pydantic import ValidationError

from romanian_news.catalog.artifacts import ArtifactFile, artifact_statements
from romanian_news.catalog_transport import (
    CatalogConnection,
    ResearchCatalogError,
    advance_artifact_current_version_statement,
    catalog_integrity_identity,
    catalog_query,
    catalog_transaction,
)
from romanian_news.video_digest.errors import (
    VideoDigestCheckpointConflictError,
    VideoDigestLeaseLostError,
)
from romanian_news.video_digest.models import (
    ClaimedSlot,
    ClaimResult,
    DigestPlan,
    EditionId,
    EditionIdentity,
    EstimatedAttemptCost,
    GenerationRequestId,
    GenerationRequestIdentity,
    GenerationRequestState,
    GenerationSpend,
    GenerationStage,
    MeasuredAttemptCost,
    PendingAttemptCost,
    PlannedStory,
    ScheduledSlot,
    SkippedSlot,
    SlotId,
    SlotLease,
    SlotName,
    SlotSkipReason,
    SlotStage,
    StoryId,
    TerminalSlot,
    TerminalSlotState,
    UnknownAttemptCost,
)

_ACTIVE_STAGES = frozenset(
    {
        SlotStage.CLAIMED,
        SlotStage.PLANNING,
        SlotStage.GENERATING,
        SlotStage.ASSEMBLING,
        SlotStage.SUBTITLING,
        SlotStage.PUBLISHING,
    }
)
_TERMINAL_STAGES = frozenset({SlotStage.SKIPPED, SlotStage.FAILED, SlotStage.PUBLISHED})
_Result = TypeVar("_Result")


def schedule_slot(slot: ScheduledSlot, *, recorded_at: datetime) -> ScheduledSlot:
    scheduled_at = _utc(slot.scheduled_at, "scheduled_at")
    timestamp = _utc(recorded_at, "recorded_at")

    def schedule(connection: CatalogConnection) -> ScheduledSlot:
        connection.execute(
            """
            INSERT INTO video_digest_slots
                (slot_id, name, scheduled_at, bucharest_day, stage, claim_count,
                 created_at, updated_at)
            VALUES (%s, %s, %s, %s, 'scheduled', 0, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                slot.slot_id,
                slot.name.value,
                scheduled_at,
                slot.bucharest_day,
                timestamp,
                timestamp,
            ),
        )
        row = _lock_slot(connection, slot.slot_id)
        try:
            stored = _scheduled_slot_from_row(row)
        except (KeyError, ValidationError, ValueError) as error:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot identity conflicts with the schedule request"
            ) from error
        expected = slot.model_copy(update={"scheduled_at": scheduled_at})
        if stored != expected:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot identity conflicts with the schedule request"
            )
        return stored

    return _checkpoint_transaction(schedule)


def skip_slot(
    slot_id: SlotId,
    reason: SlotSkipReason,
    *,
    recorded_at: datetime,
) -> SkippedSlot:
    timestamp = _utc(recorded_at, "recorded_at")

    def skip(connection: CatalogConnection) -> SkippedSlot:
        row = _lock_slot(connection, slot_id)
        locked_slot_id = SlotId(str(row["slot_id"]))
        stage = SlotStage(str(row["stage"]))
        if stage == SlotStage.SCHEDULED:
            _persist_skip(connection, locked_slot_id, reason, timestamp)
            return SkippedSlot(reason=reason)
        if stage == SlotStage.SKIPPED and str(row["skip_reason"]) == reason.value:
            return SkippedSlot(reason=reason)
        raise VideoDigestCheckpointConflictError(
            "Stored video digest slot conflicts with the skip request"
        )

    return _checkpoint_transaction(skip)


def claim_slot(
    slot_id: SlotId,
    edition: EditionIdentity,
    *,
    owner_token: str,
    now: datetime,
    lease_duration: timedelta,
) -> ClaimResult:
    _utc(now, "now")
    duration = _positive_duration(lease_duration)
    owner = _owner_token(owner_token)

    def claim(connection: CatalogConnection) -> ClaimResult:
        row = _lock_slot(connection, slot_id)
        current = _database_now(connection)
        expires_at = current + duration
        stage = SlotStage(str(row["stage"]))
        if stage in _TERMINAL_STAGES:
            return TerminalSlot(state=TerminalSlotState(stage.value))
        if stage in _ACTIVE_STAGES:
            return _claim_active_slot(
                connection,
                row,
                edition,
                owner=owner,
                now=current,
                expires_at=expires_at,
            )
        if stage != SlotStage.SCHEDULED:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot has an unsupported stage"
            )
        return _claim_scheduled_slot(
            connection,
            SlotId(str(row["slot_id"])),
            edition,
            owner=owner,
            expires_at=expires_at,
            recorded_at=current,
        )

    return _checkpoint_transaction(claim)


def renew_slot(
    lease: SlotLease,
    *,
    now: datetime,
    lease_duration: timedelta,
) -> SlotLease:
    _utc(now, "now")
    duration = _positive_duration(lease_duration)
    _utc(lease.expires_at, "lease.expires_at")

    def renew(connection: CatalogConnection) -> SlotLease:
        row, current = _lock_slot_for_lease(connection, lease)
        expires_at = current + duration
        locked_slot_id = SlotId(str(row["slot_id"]))
        connection.execute(
            """
            UPDATE video_digest_slots
            SET lease_expires_at = %s, updated_at = %s
            WHERE slot_id = %s
            """,
            (expires_at, current, locked_slot_id),
        )
        return lease.model_copy(update={"expires_at": expires_at})

    return _checkpoint_transaction(renew)


def checkpoint_plan(
    lease: SlotLease,
    plan: DigestPlan,
    *,
    plan_file: ArtifactFile,
    recorded_at: datetime,
) -> DigestPlan:
    _utc(recorded_at, "recorded_at")
    if plan.edition_id != lease.edition_id:
        raise ValueError("Digest plan edition does not match the slot lease")
    if plan_file.version_id != plan.artifact_version_id:
        raise ValueError("Digest plan artifact does not match the plan version")
    if plan_file.artifact_id != plan.edition_id or plan_file.artifact_kind != "video_digest_plan":
        raise ValueError("Digest plan artifact identity is invalid")

    def checkpoint(connection: CatalogConnection) -> DigestPlan:
        row, current = _lock_slot_for_lease(connection, lease)
        try:
            stage = SlotStage(str(row["stage"]))
        except ValueError as error:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot stage conflicts with the plan request"
            ) from error
        if stage in {
            SlotStage.GENERATING,
            SlotStage.ASSEMBLING,
            SlotStage.SUBTITLING,
            SlotStage.PUBLISHING,
        }:
            stored = _stored_plan(connection, lease.edition_id)
            if stored == plan and _stored_artifact_matches(connection, plan_file):
                return stored
            raise VideoDigestCheckpointConflictError(
                "Stored video digest plan conflicts with the plan request"
            )
        if stage != SlotStage.CLAIMED:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot stage conflicts with the plan request"
            )

        _register_artifact(connection, plan_file, current)
        _execute_returning(
            connection,
            """
            UPDATE video_digest_slots
            SET stage = 'planning', updated_at = %s
            WHERE slot_id = %s AND lease_owner_token = %s AND claim_count = %s
              AND stage = 'claimed'
            RETURNING slot_id
            """,
            (current, lease.slot_id, lease.owner_token, lease.claim_count),
            VideoDigestLeaseLostError("Video digest slot lease was lost"),
        )
        for story in plan.stories:
            connection.execute(
                """
                INSERT INTO video_digest_stories
                    (story_id, edition_id, position, report_subject_id, title,
                     mandatory, requested_duration_ms, stage, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'planned', %s, %s)
                """,
                (
                    story.story_id,
                    story.edition_id,
                    story.position,
                    story.report_subject_id,
                    story.title,
                    story.mandatory,
                    story.requested_duration_ms,
                    current,
                    current,
                ),
            )
        _execute_returning(
            connection,
            """
            UPDATE video_digest_editions
            SET plan_artifact_version_id = %s, updated_at = %s
            WHERE edition_id = %s AND plan_artifact_version_id IS NULL
            RETURNING edition_id
            """,
            (plan.artifact_version_id, current, lease.edition_id),
            VideoDigestCheckpointConflictError(
                "Stored video digest edition conflicts with the plan request"
            ),
        )
        _execute_returning(
            connection,
            """
            UPDATE video_digest_slots
            SET stage = 'generating', updated_at = %s
            WHERE slot_id = %s AND lease_owner_token = %s AND claim_count = %s
              AND stage = 'planning'
            RETURNING slot_id
            """,
            (current, lease.slot_id, lease.owner_token, lease.claim_count),
            VideoDigestLeaseLostError("Video digest slot lease was lost"),
        )
        return plan

    return _checkpoint_transaction(checkpoint)


def checkpoint_story_verification(
    lease: SlotLease,
    story_id: StoryId,
    *,
    evidence_file: ArtifactFile,
    recorded_at: datetime,
) -> None:
    _utc(recorded_at, "recorded_at")
    if evidence_file.artifact_id != story_id:
        raise ValueError("Verification evidence artifact does not match the story ID")
    if evidence_file.artifact_kind != "video_digest_story_verification":
        raise ValueError("Verification evidence artifact kind is invalid")

    def checkpoint(connection: CatalogConnection) -> None:
        slot_row, current = _lock_slot_for_lease(connection, lease)
        slot_stage = str(slot_row["stage"])
        story_row = connection.execute(
            """
            SELECT story_id, edition_id, stage,
                   verification_evidence_artifact_version_id,
                   accepted_clip_artifact_version_id,
                   failure_evidence_artifact_version_id
            FROM video_digest_stories
            WHERE story_id = %s AND edition_id = %s
            FOR UPDATE
            """,
            (story_id, lease.edition_id),
        ).fetchone()
        if story_row is None or (str(story_row["story_id"]), str(story_row["edition_id"])) != (
            story_id,
            lease.edition_id,
        ):
            raise VideoDigestCheckpointConflictError(
                "Stored video digest story identity conflicts with the verification request"
            )

        stage = str(story_row["stage"])
        stored_evidence = story_row["verification_evidence_artifact_version_id"]
        if stage in {"verified", "generating", "accepted"} and stored_evidence == (
            evidence_file.version_id
        ):
            if _stored_artifact_matches(connection, evidence_file):
                return
            raise VideoDigestCheckpointConflictError(
                "Stored video digest verification artifact conflicts with the request"
            )
        if (
            slot_stage != SlotStage.GENERATING.value
            or stage != "planned"
            or any(
                story_row[field] is not None
                for field in (
                    "verification_evidence_artifact_version_id",
                    "accepted_clip_artifact_version_id",
                    "failure_evidence_artifact_version_id",
                )
            )
        ):
            raise VideoDigestCheckpointConflictError(
                "Stored video digest story conflicts with the verification request"
            )

        _register_artifact(connection, evidence_file, current)
        identity = (story_id, lease.edition_id)
        conflict = VideoDigestCheckpointConflictError(
            "Stored video digest story conflicts with the verification request"
        )
        _execute_returning(
            connection,
            """
            UPDATE video_digest_stories
            SET stage = 'verifying', updated_at = %s
            WHERE story_id = %s AND edition_id = %s AND stage = 'planned'
              AND verification_evidence_artifact_version_id IS NULL
              AND accepted_clip_artifact_version_id IS NULL
              AND failure_evidence_artifact_version_id IS NULL
            RETURNING story_id
            """,
            (current, *identity),
            conflict,
        )
        _execute_returning(
            connection,
            """
            UPDATE video_digest_stories
            SET stage = 'verified', verification_evidence_artifact_version_id = %s,
                updated_at = %s
            WHERE story_id = %s AND edition_id = %s AND stage = 'verifying'
              AND verification_evidence_artifact_version_id IS NULL
              AND accepted_clip_artifact_version_id IS NULL
              AND failure_evidence_artifact_version_id IS NULL
            RETURNING story_id
            """,
            (evidence_file.version_id, current, *identity),
            conflict,
        )
        _execute_returning(
            connection,
            """
            UPDATE video_digest_stories
            SET stage = 'generating', updated_at = %s
            WHERE story_id = %s AND edition_id = %s AND stage = 'verified'
              AND verification_evidence_artifact_version_id = %s
              AND accepted_clip_artifact_version_id IS NULL
              AND failure_evidence_artifact_version_id IS NULL
            RETURNING story_id
            """,
            (current, *identity, evidence_file.version_id),
            conflict,
        )

    _checkpoint_transaction(checkpoint)


def checkpoint_generation_request(
    lease: SlotLease,
    request: GenerationRequestIdentity,
    *,
    request_file: ArtifactFile,
    recorded_at: datetime,
) -> GenerationRequestState:
    _utc(recorded_at, "recorded_at")
    if request.edition_id != lease.edition_id:
        raise ValueError("Generation request edition does not match the slot lease")
    if request_file.version_id != request.request_artifact_version_id:
        raise ValueError("Generation request artifact does not match the request version")
    if (
        request_file.artifact_id
        != _generation_request_artifact_id(
            request.edition_id, request.story_position, request.attempt_index
        )
        or request_file.artifact_kind != "video_digest_generation_request"
    ):
        raise ValueError("Generation request artifact identity is invalid")

    def checkpoint(connection: CatalogConnection) -> GenerationRequestState:
        slot_row, current = _lock_slot_for_lease(connection, lease)
        stored = _lock_generation_request(connection, request.request_id)
        if stored is not None:
            _require_generation_identity(stored, request)
            if not _stored_artifact_matches(connection, request_file):
                raise VideoDigestCheckpointConflictError(
                    "Stored generation request artifact conflicts with the request"
                )
            return _generation_state(stored)
        if str(slot_row["stage"]) != SlotStage.GENERATING.value:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot conflicts with the generation request"
            )
        story_row = _lock_story_at_position(connection, request.edition_id, request.story_position)
        if str(story_row["stage"]) != "generating":
            raise VideoDigestCheckpointConflictError(
                "Stored video digest story conflicts with the generation request"
            )

        occupied = connection.execute(
            """
            SELECT request_id
            FROM video_digest_generation_requests
            WHERE edition_id = %s AND story_position = %s AND attempt_index = %s
            FOR UPDATE
            """,
            (request.edition_id, request.story_position, request.attempt_index),
        ).fetchone()
        if occupied is not None:
            raise VideoDigestCheckpointConflictError(
                "Stored generation attempt conflicts with the request"
            )
        if request.attempt_index == 1:
            predecessor = connection.execute(
                """
                SELECT stage
                FROM video_digest_generation_requests
                WHERE edition_id = %s AND story_position = %s AND attempt_index = 0
                FOR UPDATE
                """,
                (request.edition_id, request.story_position),
            ).fetchone()
            if predecessor is None or str(predecessor["stage"]) != GenerationStage.FAILED.value:
                raise VideoDigestCheckpointConflictError(
                    "Second generation attempt requires a failed first attempt"
                )
        active = connection.execute(
            """
            SELECT request_id
            FROM video_digest_generation_requests
            WHERE edition_id = %s AND stage IN ('pending', 'submitted', 'processing')
            LIMIT 1
            FOR UPDATE
            """,
            (request.edition_id,),
        ).fetchone()
        if active is not None:
            raise VideoDigestCheckpointConflictError(
                "Another video digest generation attempt is active"
            )

        _register_artifact(connection, request_file, current)
        connection.execute(
            """
            INSERT INTO video_digest_generation_requests
                (request_id, edition_id, story_position, attempt_index,
                 request_artifact_version_id, stage, cost_kind, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, 'pending', 'pending', %s, %s)
            """,
            (
                request.request_id,
                request.edition_id,
                request.story_position,
                request.attempt_index,
                request.request_artifact_version_id,
                current,
                current,
            ),
        )
        return GenerationRequestState(
            request_id=request.request_id,
            stage=GenerationStage.PENDING,
            cost=PendingAttemptCost(),
        )

    return _checkpoint_transaction(checkpoint)


def checkpoint_generation_submission(
    lease: SlotLease,
    request_id: GenerationRequestId,
    *,
    provider_receipt_id: str,
    receipt_file: ArtifactFile,
    cost: EstimatedAttemptCost,
    recorded_at: datetime,
) -> GenerationRequestState:
    _utc(recorded_at, "recorded_at")
    receipt_id = provider_receipt_id.strip()
    if not receipt_id:
        raise ValueError("provider_receipt_id must not be empty")
    if (
        receipt_file.artifact_id != receipt_id
        or receipt_file.artifact_kind != "video_digest_provider_receipt"
    ):
        raise ValueError("Provider receipt artifact identity is invalid")

    def checkpoint(connection: CatalogConnection) -> GenerationRequestState:
        slot_row, current = _lock_slot_for_lease(connection, lease)
        row = _required_generation_request(connection, request_id, lease.edition_id)
        stage = GenerationStage(str(row["stage"]))
        stored_receipt = row["provider_receipt_id"]
        if stage in {
            GenerationStage.SUBMITTED,
            GenerationStage.PROCESSING,
            GenerationStage.ACCEPTED,
            GenerationStage.FAILED,
        }:
            active_cost_matches = (
                stage
                not in {
                    GenerationStage.SUBMITTED,
                    GenerationStage.PROCESSING,
                }
                or _cost_from_row(row) == cost
            )
            if (
                str(stored_receipt) == receipt_id
                and active_cost_matches
                and _stored_artifact_matches(connection, receipt_file)
            ):
                return _generation_state(row)
            raise VideoDigestCheckpointConflictError(
                "Stored provider receipt conflicts with the generation submission"
            )
        if (
            str(slot_row["stage"]) != SlotStage.GENERATING.value
            or stage != GenerationStage.PENDING
            or stored_receipt is not None
        ):
            raise VideoDigestCheckpointConflictError(
                "Stored generation request conflicts with the generation submission"
            )

        _register_artifact(connection, receipt_file, current)
        _execute_returning(
            connection,
            """
            UPDATE video_digest_generation_requests
            SET stage = 'submitted', provider_receipt_id = %s,
                cost_kind = 'estimated', cost_usd = %s, updated_at = %s
            WHERE request_id = %s AND edition_id = %s AND stage = 'pending'
              AND provider_receipt_id IS NULL
            RETURNING request_id
            """,
            (receipt_id, cost.usd, current, request_id, lease.edition_id),
            VideoDigestCheckpointConflictError(
                "Stored generation request conflicts with the generation submission"
            ),
        )
        return GenerationRequestState(
            request_id=request_id,
            stage=GenerationStage.SUBMITTED,
            provider_receipt_id=receipt_id,
            cost=cost,
        )

    return _checkpoint_transaction(checkpoint)


def checkpoint_generation_response(
    lease: SlotLease,
    request_id: GenerationRequestId,
    *,
    response_file: ArtifactFile,
    recorded_at: datetime,
) -> GenerationRequestState:
    _utc(recorded_at, "recorded_at")
    if (
        response_file.artifact_id != f"{request_id}:response"
        or response_file.artifact_kind != "video_digest_generation_response"
    ):
        raise ValueError("Generation response artifact kind is invalid")

    def checkpoint(connection: CatalogConnection) -> GenerationRequestState:
        slot_row, current = _lock_slot_for_lease(connection, lease)
        row = _required_generation_request(connection, request_id, lease.edition_id)
        stage = GenerationStage(str(row["stage"]))
        stored_response = row["response_artifact_version_id"]
        if stage in {GenerationStage.PROCESSING, GenerationStage.ACCEPTED, GenerationStage.FAILED}:
            if stored_response == response_file.version_id and _stored_artifact_matches(
                connection, response_file
            ):
                return _generation_state(row)
            raise VideoDigestCheckpointConflictError(
                "Stored generation response conflicts with the request"
            )
        if (
            str(slot_row["stage"]) != SlotStage.GENERATING.value
            or stage != GenerationStage.SUBMITTED
            or row["provider_receipt_id"] is None
        ):
            raise VideoDigestCheckpointConflictError(
                "Stored generation request conflicts with the response"
            )

        _register_artifact(connection, response_file, current)
        _execute_returning(
            connection,
            """
            UPDATE video_digest_generation_requests
            SET stage = 'processing', response_artifact_version_id = %s, updated_at = %s
            WHERE request_id = %s AND edition_id = %s AND stage = 'submitted'
              AND provider_receipt_id IS NOT NULL
              AND response_artifact_version_id IS NULL
            RETURNING request_id
            """,
            (response_file.version_id, current, request_id, lease.edition_id),
            VideoDigestCheckpointConflictError(
                "Stored generation request conflicts with the response"
            ),
        )
        return _generation_state(
            {
                **row,
                "stage": GenerationStage.PROCESSING.value,
                "response_artifact_version_id": response_file.version_id,
            }
        )

    return _checkpoint_transaction(checkpoint)


def checkpoint_generation_acceptance(
    lease: SlotLease,
    request_id: GenerationRequestId,
    *,
    clip_file: ArtifactFile,
    cost: MeasuredAttemptCost | UnknownAttemptCost,
    recorded_at: datetime,
) -> GenerationRequestState:
    _utc(recorded_at, "recorded_at")
    if clip_file.artifact_kind != "video_digest_accepted_clip":
        raise ValueError("Accepted clip artifact kind is invalid")

    def checkpoint(connection: CatalogConnection) -> GenerationRequestState:
        slot_row, current = _lock_slot_for_lease(connection, lease)
        row = _required_generation_request(connection, request_id, lease.edition_id)
        story = _lock_story_at_position(connection, lease.edition_id, int(row["story_position"]))
        if clip_file.artifact_id != f"{story['story_id']}:accepted-clip":
            raise ValueError("Accepted clip artifact identity is invalid")
        if str(row["stage"]) == GenerationStage.ACCEPTED.value:
            if (
                row["accepted_clip_artifact_version_id"] == clip_file.version_id
                and story["accepted_clip_artifact_version_id"] == clip_file.version_id
                and _cost_from_row(row) == cost
                and _stored_artifact_matches(connection, clip_file)
            ):
                return _generation_state(row)
            raise VideoDigestCheckpointConflictError(
                "Stored generation acceptance conflicts with the request"
            )
        if (
            str(slot_row["stage"]) != SlotStage.GENERATING.value
            or str(row["stage"]) != GenerationStage.PROCESSING.value
            or row["response_artifact_version_id"] is None
            or str(story["stage"]) != "generating"
        ):
            raise VideoDigestCheckpointConflictError(
                "Stored generation request conflicts with acceptance"
            )

        _register_artifact(connection, clip_file, current)
        cost_kind, cost_usd, unknown_reason = _cost_columns(cost)
        _execute_returning(
            connection,
            """
            UPDATE video_digest_generation_requests
            SET stage = 'accepted', accepted_clip_artifact_version_id = %s,
                cost_kind = %s, cost_usd = %s, cost_unknown_reason = %s, updated_at = %s
            WHERE request_id = %s AND edition_id = %s AND stage = 'processing'
              AND response_artifact_version_id IS NOT NULL
              AND accepted_clip_artifact_version_id IS NULL
              AND failure_evidence_artifact_version_id IS NULL
            RETURNING request_id
            """,
            (
                clip_file.version_id,
                cost_kind,
                cost_usd,
                unknown_reason,
                current,
                request_id,
                lease.edition_id,
            ),
            VideoDigestCheckpointConflictError(
                "Stored generation request conflicts with acceptance"
            ),
        )
        _execute_returning(
            connection,
            """
            UPDATE video_digest_stories
            SET stage = 'accepted', accepted_clip_artifact_version_id = %s, updated_at = %s
            WHERE edition_id = %s AND position = %s AND stage = 'generating'
              AND accepted_clip_artifact_version_id IS NULL
              AND failure_evidence_artifact_version_id IS NULL
            RETURNING story_id
            """,
            (
                clip_file.version_id,
                current,
                lease.edition_id,
                int(row["story_position"]),
            ),
            VideoDigestCheckpointConflictError(
                "Stored video digest story conflicts with generation acceptance"
            ),
        )
        return GenerationRequestState(
            request_id=request_id,
            stage=GenerationStage.ACCEPTED,
            provider_receipt_id=str(row["provider_receipt_id"]),
            cost=cost,
        )

    return _checkpoint_transaction(checkpoint)


def checkpoint_generation_failure(
    lease: SlotLease,
    request_id: GenerationRequestId,
    *,
    evidence_file: ArtifactFile,
    cost: MeasuredAttemptCost | UnknownAttemptCost,
    recorded_at: datetime,
) -> GenerationRequestState:
    _utc(recorded_at, "recorded_at")
    if (
        evidence_file.artifact_id != f"{request_id}:failure"
        or evidence_file.artifact_kind != "video_digest_generation_failure"
    ):
        raise ValueError("Generation failure artifact kind is invalid")

    def checkpoint(connection: CatalogConnection) -> GenerationRequestState:
        slot_row = _lock_slot(connection, lease.slot_id)
        if str(slot_row["stage"]) == SlotStage.FAILED.value:
            if not _terminal_fence_matches(slot_row, lease):
                raise VideoDigestLeaseLostError("Video digest slot lease was lost")
            row = _required_generation_request(connection, request_id, lease.edition_id)
            if (
                str(row["stage"]) == GenerationStage.FAILED.value
                and row["failure_evidence_artifact_version_id"] == evidence_file.version_id
                and _cost_from_row(row) == cost
                and _stored_artifact_matches(connection, evidence_file)
            ):
                return _generation_state(row)
            raise VideoDigestCheckpointConflictError(
                "Stored generation failure conflicts with the request"
            )

        current = _validate_locked_lease(connection, slot_row, lease)
        row = _required_generation_request(connection, request_id, lease.edition_id)
        if str(row["stage"]) == GenerationStage.FAILED.value:
            if (
                row["failure_evidence_artifact_version_id"] == evidence_file.version_id
                and _cost_from_row(row) == cost
                and _stored_artifact_matches(connection, evidence_file)
            ):
                return _generation_state(row)
            raise VideoDigestCheckpointConflictError(
                "Stored generation failure conflicts with the request"
            )

        if str(slot_row["stage"]) != SlotStage.GENERATING.value or str(row["stage"]) not in {
            GenerationStage.PENDING.value,
            GenerationStage.SUBMITTED.value,
            GenerationStage.PROCESSING.value,
        }:
            raise VideoDigestCheckpointConflictError(
                "Stored generation request conflicts with failure"
            )
        story = _lock_story_at_position(connection, lease.edition_id, int(row["story_position"]))
        if str(story["stage"]) != "generating":
            raise VideoDigestCheckpointConflictError(
                "Stored video digest story conflicts with generation failure"
            )

        _register_artifact(connection, evidence_file, current)
        cost_kind, cost_usd, unknown_reason = _cost_columns(cost)
        _execute_returning(
            connection,
            """
            UPDATE video_digest_generation_requests
            SET stage = 'failed', failure_evidence_artifact_version_id = %s,
                cost_kind = %s, cost_usd = %s, cost_unknown_reason = %s, updated_at = %s
            WHERE request_id = %s AND edition_id = %s
              AND stage IN ('pending', 'submitted', 'processing')
              AND accepted_clip_artifact_version_id IS NULL
              AND failure_evidence_artifact_version_id IS NULL
            RETURNING request_id
            """,
            (
                evidence_file.version_id,
                cost_kind,
                cost_usd,
                unknown_reason,
                current,
                request_id,
                lease.edition_id,
            ),
            VideoDigestCheckpointConflictError("Stored generation request conflicts with failure"),
        )
        if int(row["attempt_index"]) == 1:
            _execute_returning(
                connection,
                """
                UPDATE video_digest_stories
                SET stage = 'failed', failure_evidence_artifact_version_id = %s,
                    updated_at = %s
                WHERE edition_id = %s AND position = %s AND stage = 'generating'
                  AND accepted_clip_artifact_version_id IS NULL
                  AND failure_evidence_artifact_version_id IS NULL
                RETURNING story_id
                """,
                (
                    evidence_file.version_id,
                    current,
                    lease.edition_id,
                    int(row["story_position"]),
                ),
                VideoDigestCheckpointConflictError(
                    "Stored video digest story conflicts with generation failure"
                ),
            )
            _execute_returning(
                connection,
                """
                UPDATE video_digest_slots
                SET stage = 'failed', lease_owner_token = NULL, lease_expires_at = NULL,
                    terminal_lease_owner_token = %s, terminal_lease_expires_at = %s,
                    terminal_claim_count = %s,
                    failure_evidence_artifact_version_id = %s, updated_at = %s
                WHERE slot_id = %s AND lease_owner_token = %s AND claim_count = %s
                  AND stage = 'generating'
                RETURNING slot_id
                """,
                (
                    lease.owner_token,
                    lease.expires_at,
                    lease.claim_count,
                    evidence_file.version_id,
                    current,
                    lease.slot_id,
                    lease.owner_token,
                    lease.claim_count,
                ),
                VideoDigestLeaseLostError("Video digest slot lease was lost"),
            )
        return GenerationRequestState(
            request_id=request_id,
            stage=GenerationStage.FAILED,
            provider_receipt_id=(
                str(row["provider_receipt_id"]) if row["provider_receipt_id"] is not None else None
            ),
            cost=cost,
        )

    return _checkpoint_transaction(checkpoint)


def checkpoint_assembly_ready(lease: SlotLease, *, recorded_at: datetime) -> None:
    _utc(recorded_at, "recorded_at")

    def checkpoint(connection: CatalogConnection) -> None:
        slot_row, current = _lock_slot_for_lease(connection, lease)
        stage = SlotStage(str(slot_row["stage"]))
        if stage in {SlotStage.ASSEMBLING, SlotStage.SUBTITLING, SlotStage.PUBLISHING}:
            return
        if stage != SlotStage.GENERATING:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot conflicts with assembly readiness"
            )
        incomplete = connection.execute(
            """
            SELECT story_id
            FROM video_digest_stories
            WHERE edition_id = %s AND (stage <> 'accepted' OR accepted_clip_artifact_version_id IS NULL)
            LIMIT 1
            """,
            (lease.edition_id,),
        ).fetchone()
        if incomplete is not None:
            raise VideoDigestCheckpointConflictError(
                "Video digest edition has incomplete mandatory stories"
            )
        _execute_returning(
            connection,
            """
            UPDATE video_digest_slots
            SET stage = 'assembling', updated_at = %s
            WHERE slot_id = %s AND lease_owner_token = %s AND claim_count = %s
              AND stage = 'generating'
            RETURNING slot_id
            """,
            (current, lease.slot_id, lease.owner_token, lease.claim_count),
            VideoDigestLeaseLostError("Video digest slot lease was lost"),
        )

    _checkpoint_transaction(checkpoint)


def read_generation_spend(edition_id: EditionId) -> GenerationSpend:
    rows = catalog_query(
        """
        SELECT
            COALESCE(sum(cost_usd) FILTER (WHERE cost_kind = 'measured'), 0) AS measured_usd,
            COALESCE(sum(cost_usd) FILTER (WHERE cost_kind = 'estimated'), 0) AS estimated_usd,
            count(*) FILTER (WHERE cost_kind = 'pending') AS pending_requests,
            count(*) FILTER (WHERE cost_kind = 'unknown') AS unknown_requests
        FROM video_digest_generation_requests
        WHERE edition_id = %s
        """,
        [edition_id],
    )
    if len(rows) != 1:
        raise ResearchCatalogError("PostgreSQL did not return video digest spend")
    return GenerationSpend.model_validate(rows[0])


def _claim_active_slot(
    connection: CatalogConnection,
    row: Mapping[str, Any],
    edition: EditionIdentity,
    *,
    owner: str,
    now: datetime,
    expires_at: datetime,
) -> ClaimResult:
    stored_edition_id = str(row["edition_id"])
    stored_owner = str(row["lease_owner_token"])
    stored_expiry = _datetime(row["lease_expires_at"])
    claim_count = int(row["claim_count"])
    slot_id = SlotId(str(row["slot_id"]))

    if stored_edition_id != edition.edition_id:
        raise VideoDigestCheckpointConflictError(
            "Active video digest lease belongs to another edition"
        )
    if stored_expiry > now:
        if stored_owner != owner:
            return SkippedSlot(reason=SlotSkipReason.OVERLAPPING_RUN)
        return ClaimedSlot(
            lease=SlotLease(
                slot_id=slot_id,
                edition_id=edition.edition_id,
                owner_token=owner,
                expires_at=stored_expiry,
                claim_count=claim_count,
            )
        )

    if stored_owner == owner:
        raise VideoDigestCheckpointConflictError(
            "Expired video digest lease recovery requires a fresh owner token"
        )

    next_claim_count = claim_count + 1
    connection.execute(
        """
        UPDATE video_digest_slots
        SET lease_owner_token = %s, lease_expires_at = %s,
            claim_count = %s, updated_at = %s
        WHERE slot_id = %s
        """,
        (owner, expires_at, next_claim_count, now, slot_id),
    )
    return ClaimedSlot(
        lease=SlotLease(
            slot_id=slot_id,
            edition_id=edition.edition_id,
            owner_token=owner,
            expires_at=expires_at,
            claim_count=next_claim_count,
        )
    )


def _claim_scheduled_slot(
    connection: CatalogConnection,
    slot_id: SlotId,
    edition: EditionIdentity,
    *,
    owner: str,
    expires_at: datetime,
    recorded_at: datetime,
) -> ClaimResult:
    connection.execute(
        """
        INSERT INTO video_digest_editions
            (edition_id, daily_report_version_id, policy_bundle_version_id,
             subtitle_state, created_at, updated_at)
        VALUES (%s, %s, %s, 'pending', %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (
            edition.edition_id,
            edition.daily_report_version_id,
            edition.policy_bundle_version_id,
            recorded_at,
            recorded_at,
        ),
    )
    edition_row = connection.execute(
        """
        SELECT edition_id, daily_report_version_id, policy_bundle_version_id
        FROM video_digest_editions
        WHERE edition_id = %s
        FOR UPDATE
        """,
        (edition.edition_id,),
    ).fetchone()
    if edition_row is None or (
        str(edition_row["edition_id"]),
        str(edition_row["daily_report_version_id"]),
        str(edition_row["policy_bundle_version_id"]),
    ) != (
        edition.edition_id,
        edition.daily_report_version_id,
        edition.policy_bundle_version_id,
    ):
        raise VideoDigestCheckpointConflictError(
            "Stored video digest edition identity conflicts with the claim request"
        )

    published = connection.execute(
        """
        SELECT publication_id
        FROM video_digest_publication_intents
        WHERE edition_id = %s AND stage = 'published'
        """,
        (edition.edition_id,),
    ).fetchone()
    if published is not None:
        _persist_skip(
            connection,
            slot_id,
            SlotSkipReason.ALREADY_PUBLISHED,
            recorded_at,
            edition_id=edition.edition_id,
        )
        return SkippedSlot(reason=SlotSkipReason.ALREADY_PUBLISHED)

    active = connection.execute(
        """
        SELECT slot_id
        FROM video_digest_slots
        WHERE edition_id = %s AND slot_id <> %s
          AND stage IN ('claimed', 'planning', 'generating', 'assembling',
                        'subtitling', 'publishing')
        """,
        (edition.edition_id, slot_id),
    ).fetchone()
    if active is not None:
        _persist_skip(
            connection,
            slot_id,
            SlotSkipReason.ACTIVE_EDITION,
            recorded_at,
            edition_id=edition.edition_id,
        )
        return SkippedSlot(reason=SlotSkipReason.ACTIVE_EDITION)

    connection.execute(
        """
        UPDATE video_digest_slots
        SET stage = 'claimed', edition_id = %s, lease_owner_token = %s,
            lease_expires_at = %s, claim_count = 1, updated_at = %s
        WHERE slot_id = %s
        """,
        (edition.edition_id, owner, expires_at, recorded_at, slot_id),
    )
    return ClaimedSlot(
        lease=SlotLease(
            slot_id=slot_id,
            edition_id=edition.edition_id,
            owner_token=owner,
            expires_at=expires_at,
            claim_count=1,
        )
    )


def _lock_slot(connection: CatalogConnection, slot_id: SlotId) -> Mapping[str, Any]:
    row = connection.execute(
        """
        SELECT slot_id, name, scheduled_at, bucharest_day, stage, edition_id,
               lease_owner_token, lease_expires_at, claim_count, skip_reason,
               terminal_lease_owner_token, terminal_lease_expires_at,
               terminal_claim_count
        FROM video_digest_slots
        WHERE slot_id = %s
        FOR UPDATE
        """,
        (slot_id,),
    ).fetchone()
    if row is None:
        raise VideoDigestCheckpointConflictError("Video digest slot is unavailable")
    return row


def _lock_slot_for_lease(
    connection: CatalogConnection,
    lease: SlotLease,
) -> tuple[Mapping[str, Any], datetime]:
    row = _lock_slot(connection, lease.slot_id)
    return row, _validate_locked_lease(connection, row, lease)


def _validate_locked_lease(
    connection: CatalogConnection,
    row: Mapping[str, Any],
    lease: SlotLease,
) -> datetime:
    now = _database_now(connection)
    stored_expiry = _optional_datetime(row["lease_expires_at"])
    if (
        str(row["slot_id"]) != lease.slot_id
        or str(row["edition_id"]) != lease.edition_id
        or str(row["lease_owner_token"]) != lease.owner_token
        or int(row["claim_count"]) != lease.claim_count
        or stored_expiry != _utc(lease.expires_at, "lease.expires_at")
        or stored_expiry <= now
    ):
        raise VideoDigestLeaseLostError("Video digest slot lease was lost")
    return now


def _terminal_fence_matches(row: Mapping[str, Any], lease: SlotLease) -> bool:
    stored_owner = row["terminal_lease_owner_token"]
    stored_expiry = row["terminal_lease_expires_at"]
    stored_claim_count = row["terminal_claim_count"]
    if stored_owner is None or stored_expiry is None or stored_claim_count is None:
        return False
    return (
        str(row["slot_id"]) == lease.slot_id
        and str(row["edition_id"]) == lease.edition_id
        and str(stored_owner) == lease.owner_token
        and int(stored_claim_count) == lease.claim_count
        and _datetime(stored_expiry) == _utc(lease.expires_at, "lease.expires_at")
    )


def _database_now(connection: CatalogConnection) -> datetime:
    row = connection.execute("SELECT clock_timestamp() AS database_now").fetchone()
    if row is None:
        raise VideoDigestLeaseLostError("PostgreSQL did not report the current time")
    return _datetime(row["database_now"])


def _persist_skip(
    connection: CatalogConnection,
    slot_id: SlotId,
    reason: SlotSkipReason,
    recorded_at: datetime,
    *,
    edition_id: str | None = None,
) -> None:
    connection.execute(
        """
        UPDATE video_digest_slots
        SET stage = 'skipped', edition_id = COALESCE(%s, edition_id),
            skip_reason = %s, updated_at = %s
        WHERE slot_id = %s
        """,
        (edition_id, reason.value, recorded_at, slot_id),
    )


def _scheduled_slot_from_row(row: Mapping[str, Any]) -> ScheduledSlot:
    day_value = row["bucharest_day"]
    return ScheduledSlot(
        slot_id=SlotId(str(row["slot_id"])),
        name=SlotName(str(row["name"])),
        scheduled_at=_datetime(row["scheduled_at"]),
        bucharest_day=(
            day_value if isinstance(day_value, date) else date.fromisoformat(str(day_value))
        ),
    )


def _stored_plan(connection: CatalogConnection, edition_id: EditionId) -> DigestPlan:
    edition_row = connection.execute(
        """
        SELECT plan_artifact_version_id
        FROM video_digest_editions
        WHERE edition_id = %s
        """,
        (edition_id,),
    ).fetchone()
    story_rows = connection.execute(
        """
        SELECT story_id, edition_id, position, report_subject_id, title,
               mandatory, requested_duration_ms
        FROM video_digest_stories
        WHERE edition_id = %s
        ORDER BY position
        """,
        (edition_id,),
    ).fetchall()
    try:
        if edition_row is None:
            raise ValueError("Edition is unavailable")
        story_fields = (
            "story_id",
            "edition_id",
            "position",
            "report_subject_id",
            "title",
            "mandatory",
            "requested_duration_ms",
        )
        stories = tuple(
            PlannedStory.model_validate({field: row[field] for field in story_fields})
            for row in story_rows
        )
        return DigestPlan(
            edition_id=edition_id,
            artifact_version_id=edition_row["plan_artifact_version_id"],
            stories=stories,
        )
    except (KeyError, TypeError, ValidationError, ValueError) as error:
        raise VideoDigestCheckpointConflictError(
            "Stored video digest plan conflicts with the plan request"
        ) from error


def _lock_story_at_position(
    connection: CatalogConnection,
    edition_id: EditionId,
    position: int,
) -> Mapping[str, Any]:
    row = connection.execute(
        """
        SELECT story_id, edition_id, position, stage,
               verification_evidence_artifact_version_id,
               accepted_clip_artifact_version_id,
               failure_evidence_artifact_version_id
        FROM video_digest_stories
        WHERE edition_id = %s AND position = %s
        FOR UPDATE
        """,
        (edition_id, position),
    ).fetchone()
    if row is None:
        raise VideoDigestCheckpointConflictError("Video digest story is unavailable")
    return row


def _lock_generation_request(
    connection: CatalogConnection,
    request_id: GenerationRequestId,
) -> Mapping[str, Any] | None:
    return connection.execute(
        """
        SELECT request_id, edition_id, story_position, attempt_index,
               request_artifact_version_id, stage, provider_receipt_id,
               response_artifact_version_id, accepted_clip_artifact_version_id,
               failure_evidence_artifact_version_id, cost_kind, cost_usd,
               cost_unknown_reason
        FROM video_digest_generation_requests
        WHERE request_id = %s
        FOR UPDATE
        """,
        (request_id,),
    ).fetchone()


def _required_generation_request(
    connection: CatalogConnection,
    request_id: GenerationRequestId,
    edition_id: EditionId,
) -> Mapping[str, Any]:
    row = _lock_generation_request(connection, request_id)
    if row is None or str(row["edition_id"]) != edition_id:
        raise VideoDigestCheckpointConflictError("Video digest generation request is unavailable")
    return row


def _require_generation_identity(
    row: Mapping[str, Any],
    request: GenerationRequestIdentity,
) -> None:
    if (
        str(row["request_id"]),
        str(row["edition_id"]),
        int(row["story_position"]),
        int(row["attempt_index"]),
        str(row["request_artifact_version_id"]),
    ) != (
        request.request_id,
        request.edition_id,
        request.story_position,
        request.attempt_index,
        request.request_artifact_version_id,
    ):
        raise VideoDigestCheckpointConflictError(
            "Stored generation request identity conflicts with the request"
        )


def _generation_request_artifact_id(
    edition_id: EditionId,
    story_position: int,
    attempt_index: int,
) -> str:
    return f"{edition_id}:{story_position}:{attempt_index}:generation-request"


def _generation_state(row: Mapping[str, Any]) -> GenerationRequestState:
    receipt = row["provider_receipt_id"]
    return GenerationRequestState(
        request_id=GenerationRequestId(str(row["request_id"])),
        stage=GenerationStage(str(row["stage"])),
        provider_receipt_id=str(receipt) if receipt is not None else None,
        cost=_cost_from_row(row),
    )


def _cost_from_row(
    row: Mapping[str, Any],
) -> PendingAttemptCost | EstimatedAttemptCost | MeasuredAttemptCost | UnknownAttemptCost:
    kind = str(row["cost_kind"])
    if kind == "pending":
        return PendingAttemptCost()
    if kind == "estimated":
        return EstimatedAttemptCost(usd=Decimal(str(row["cost_usd"])))
    if kind == "measured":
        return MeasuredAttemptCost(usd=Decimal(str(row["cost_usd"])))
    if kind == "unknown":
        return UnknownAttemptCost(reason=str(row["cost_unknown_reason"]))
    raise VideoDigestCheckpointConflictError("Stored generation cost kind is invalid")


def _cost_columns(
    cost: MeasuredAttemptCost | UnknownAttemptCost,
) -> tuple[str, Decimal | None, str | None]:
    if isinstance(cost, MeasuredAttemptCost):
        return cost.kind, cost.usd, None
    return cost.kind, None, cost.reason


def _register_artifact(
    connection: CatalogConnection,
    file: ArtifactFile,
    created_at: datetime,
) -> None:
    for statement, parameters in artifact_statements(
        file,
        created_at.isoformat(),
        produced_by_run_id=None,
    ):
        connection.execute(cast(LiteralString, statement), parameters)
    statement, parameters = advance_artifact_current_version_statement(
        file.artifact_id, file.version_id
    )
    connection.execute(cast(LiteralString, statement), parameters)


def _stored_artifact_matches(connection: CatalogConnection, file: ArtifactFile) -> bool:
    row = connection.execute(
        """
        SELECT artifact.id AS artifact_id, artifact.kind AS artifact_kind,
               artifact.title, version.id AS version_id, version.content_digest,
               stored_file.r2_key, stored_file.media_type, stored_file.byte_size
        FROM artifacts AS artifact
        JOIN artifact_versions AS version ON version.artifact_id = artifact.id
        JOIN artifact_files AS stored_file ON stored_file.artifact_version_id = version.id
        WHERE version.id = %s
        """,
        (file.version_id,),
    ).fetchone()
    if row is None:
        return False
    return (
        str(row["artifact_id"]),
        str(row["artifact_kind"]),
        str(row["title"]),
        str(row["version_id"]),
        str(row["content_digest"]),
        str(row["r2_key"]),
        str(row["media_type"]),
        int(row["byte_size"]),
    ) == (
        file.artifact_id,
        file.artifact_kind,
        file.title,
        file.version_id,
        file.content_digest,
        file.r2_key,
        file.media_type,
        len(file.content),
    )


def _execute_returning(
    connection: CatalogConnection,
    statement: LiteralString,
    parameters: Sequence[object],
    missing_error: Exception,
) -> Mapping[str, Any]:
    row = connection.execute(statement, parameters).fetchone()
    if row is None:
        raise missing_error
    return row


def _checkpoint_transaction(operation: Callable[[CatalogConnection], _Result]) -> _Result:
    try:
        return catalog_transaction(operation, retry_transient_errors=False)
    except ResearchCatalogError as error:
        identity = catalog_integrity_identity(error)
        if identity is not None and identity.split(":", 2)[1].startswith("23"):
            raise VideoDigestCheckpointConflictError(
                "Stored video digest identity conflicts with the request"
            ) from error
        raise


def _owner_token(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("owner_token must not be empty")
    return normalized


def _positive_duration(value: timedelta) -> timedelta:
    if value <= timedelta(0):
        raise ValueError("lease_duration must be positive")
    return value


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset")
    return value.astimezone(UTC)


def _datetime(value: object) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    return _utc(parsed, "PostgreSQL datetime")


def _optional_datetime(value: object) -> datetime:
    if value is None:
        raise VideoDigestLeaseLostError("Video digest slot lease was lost")
    return _datetime(value)
