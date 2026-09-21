from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Any, Literal, LiteralString, TypeVar, cast
from urllib.parse import urlsplit

from pydantic import Field, ValidationError

from romanian_news import NewsModel, Sha256
from romanian_news.catalog.artifacts import (
    ArtifactFile,
    CatalogArtifactReference,
    artifact_statements,
)
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
    AssembledVideo,
    AttemptCost,
    AvailableSubtitles,
    ClaimedSlot,
    ClaimResult,
    DigestPlan,
    EditionId,
    EditionIdentity,
    EstimatedAttemptCost,
    FailedSubtitles,
    GenerationAdmission,
    GenerationRequestCheckpoint,
    GenerationRequestId,
    GenerationRequestIdentity,
    GenerationRequestState,
    GenerationSpend,
    GenerationStage,
    MeasuredAttemptCost,
    PendingAttemptCost,
    PlannedStory,
    PublicationId,
    PublicationIntent,
    PublicationProgress,
    PublicationState,
    PublicationStatus,
    PublishedEdition,
    PublishedEditionSummary,
    PublishedMedia,
    PublishedPublication,
    PublishedStory,
    PublishedSubtitleAvailable,
    PublishedSubtitleFailed,
    ScheduledSlot,
    SkippedSlot,
    SlotId,
    SlotLease,
    SlotName,
    SlotSkipReason,
    SlotStage,
    StoryId,
    SubtitleOutcome,
    TerminalSlot,
    TerminalSlotState,
    UnknownAttemptCost,
    UploadedPublication,
    UploadingPublication,
    VerifiedPublication,
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


class PlanningAttemptReference(NewsModel):
    attempt_index: Annotated[int, Field(ge=0, le=2)]
    disposition: Literal["rejected", "accepted"]
    evidence: CatalogArtifactReference
    accepted_plan_artifact_version_id: Sha256 | None


class AcceptedClipReference(CatalogArtifactReference):
    byte_size: Annotated[int, Field(gt=0)]


class GenerationAttemptReference(NewsModel):
    request: GenerationRequestIdentity
    stage: GenerationStage
    provider_receipt_id: str | None
    cost: AttemptCost
    request_evidence: CatalogArtifactReference
    receipt_evidence: CatalogArtifactReference | None
    response_evidence: CatalogArtifactReference | None
    accepted_clip: AcceptedClipReference | None = None
    validation_evidence: CatalogArtifactReference | None = None


def read_generation_attempts(edition_id: EditionId) -> tuple[GenerationAttemptReference, ...]:
    rows = catalog_query(
        """
        SELECT request.request_id, request.edition_id, request.story_position,
               request.attempt_index, request.request_artifact_version_id,
               request.stage, request.provider_receipt_id, request.cost_kind,
               request.cost_usd, request.cost_unknown_reason,
               request_artifact.id AS request_artifact_id,
               request_file.content_digest AS request_content_digest,
               request_file.r2_key AS request_r2_key,
               receipt_version.id AS receipt_artifact_version_id,
               receipt_file.content_digest AS receipt_content_digest,
               receipt_file.r2_key AS receipt_r2_key,
                response_artifact.id AS response_artifact_id,
                response_file.content_digest AS response_content_digest,
                response_file.r2_key AS response_r2_key,
                request.response_artifact_version_id,
                clip_artifact.id AS clip_artifact_id,
                clip_file.content_digest AS clip_content_digest,
                clip_file.r2_key AS clip_r2_key,
                clip_file.byte_size AS clip_byte_size,
                validation_artifact.id AS validation_artifact_id,
                validation_file.content_digest AS validation_content_digest,
                validation_file.r2_key AS validation_r2_key,
                request.accepted_clip_artifact_version_id,
                request.validation_evidence_artifact_version_id
        FROM video_digest_generation_requests AS request
        JOIN artifact_versions AS request_version
          ON request_version.id = request.request_artifact_version_id
        JOIN artifacts AS request_artifact ON request_artifact.id = request_version.artifact_id
        JOIN artifact_files AS request_file
          ON request_file.artifact_version_id = request_version.id
        LEFT JOIN artifacts AS receipt_artifact
          ON receipt_artifact.id = request.provider_receipt_id
        LEFT JOIN artifact_versions AS receipt_version
          ON receipt_version.id = receipt_artifact.current_version_id
        LEFT JOIN artifact_files AS receipt_file
          ON receipt_file.artifact_version_id = receipt_version.id
        LEFT JOIN artifact_versions AS response_version
          ON response_version.id = request.response_artifact_version_id
        LEFT JOIN artifacts AS response_artifact
          ON response_artifact.id = response_version.artifact_id
        LEFT JOIN artifact_files AS response_file
          ON response_file.artifact_version_id = response_version.id
        LEFT JOIN artifact_versions AS clip_version
          ON clip_version.id = request.accepted_clip_artifact_version_id
        LEFT JOIN artifacts AS clip_artifact ON clip_artifact.id = clip_version.artifact_id
        LEFT JOIN artifact_files AS clip_file
          ON clip_file.artifact_version_id = clip_version.id
        LEFT JOIN artifact_versions AS validation_version
          ON validation_version.id = request.validation_evidence_artifact_version_id
        LEFT JOIN artifacts AS validation_artifact
          ON validation_artifact.id = validation_version.artifact_id
        LEFT JOIN artifact_files AS validation_file
          ON validation_file.artifact_version_id = validation_version.id
        WHERE request.edition_id = %s
        ORDER BY request.story_position, request.attempt_index
        """,
        [edition_id],
    )
    return tuple(_generation_attempt_reference(row) for row in rows)


def read_generation_deadline(slot_id: SlotId) -> datetime:
    rows = catalog_query(
        """
        SELECT scheduled_at + INTERVAL '90 minutes' AS deadline_at
        FROM video_digest_slots
        WHERE slot_id = %s
        """,
        [slot_id],
    )
    if len(rows) != 1:
        raise ResearchCatalogError("PostgreSQL did not return the video digest deadline")
    return _utc(cast(datetime, rows[0]["deadline_at"]), "deadline_at")


def record_policy_bundle(file: ArtifactFile, *, recorded_at: datetime) -> Sha256:
    timestamp = _utc(recorded_at, "recorded_at")
    if (
        not file.artifact_id.startswith("video-digest-policy:")
        or file.artifact_kind != "video_digest_policy"
        or file.media_type != "application/json"
    ):
        raise ValueError("Video digest policy artifact identity is invalid")

    def record(connection: CatalogConnection) -> Sha256:
        _register_artifact(connection, file, timestamp)
        if not _stored_artifact_matches(connection, file):
            raise VideoDigestCheckpointConflictError(
                "Stored video digest policy conflicts with the request"
            )
        return file.version_id

    return _checkpoint_transaction(record)


def record_generation_policy(file: ArtifactFile, *, recorded_at: datetime) -> Sha256:
    timestamp = _utc(recorded_at, "recorded_at")
    if (
        not file.artifact_id.startswith("video-digest-generation-policy:")
        or file.artifact_kind != "video_digest_generation_policy"
        or file.media_type != "application/json"
    ):
        raise ValueError("Video digest generation policy artifact identity is invalid")

    def record(connection: CatalogConnection) -> Sha256:
        _register_artifact(connection, file, timestamp)
        if not _stored_artifact_matches(connection, file):
            raise VideoDigestCheckpointConflictError(
                "Stored video digest generation policy conflicts with the request"
            )
        return file.version_id

    return _checkpoint_transaction(record)


def read_planning_attempts(edition_id: EditionId) -> tuple[PlanningAttemptReference, ...]:
    rows = catalog_query(
        """
        SELECT attempt.attempt_index, attempt.disposition,
               artifact.id AS artifact_id, version.id AS version_id,
               file.content_digest, file.r2_key,
               attempt.accepted_plan_artifact_version_id
        FROM video_digest_planning_attempts attempt
        JOIN artifact_versions version
          ON version.id = attempt.attempt_evidence_artifact_version_id
        JOIN artifacts artifact ON artifact.id = version.artifact_id
        JOIN artifact_files file ON file.artifact_version_id = version.id
        WHERE attempt.edition_id = %s
        ORDER BY attempt.attempt_index
        """,
        [edition_id],
    )
    attempts = tuple(
        PlanningAttemptReference(
            attempt_index=row["attempt_index"],
            disposition=row["disposition"],
            evidence=CatalogArtifactReference.model_validate(
                {
                    field: row[field]
                    for field in ("artifact_id", "version_id", "content_digest", "r2_key")
                },
                strict=True,
            ),
            accepted_plan_artifact_version_id=row["accepted_plan_artifact_version_id"],
        )
        for row in rows
    )
    if tuple(item.attempt_index for item in attempts) != tuple(range(len(attempts))):
        raise VideoDigestCheckpointConflictError("Stored planning attempts are not contiguous")
    if any(item.disposition == "accepted" for item in attempts[:-1]):
        raise VideoDigestCheckpointConflictError("Accepted planning attempt must be final")
    return attempts


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
        if stage == SlotStage.SKIPPED:
            return SkippedSlot(reason=SlotSkipReason(str(row["skip_reason"])))
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


def checkpoint_planning_attempt(
    lease: SlotLease,
    attempt_index: int,
    disposition: Literal["rejected", "accepted"],
    *,
    evidence_file: ArtifactFile,
    accepted_plan: DigestPlan | None = None,
    plan_file: ArtifactFile | None = None,
    recorded_at: datetime,
) -> DigestPlan | None:
    _utc(recorded_at, "recorded_at")
    if attempt_index not in (0, 1, 2):
        raise ValueError("Planning attempt index must be 0, 1, or 2")
    if disposition not in {"rejected", "accepted"}:
        raise ValueError("Planning attempt disposition must be rejected or accepted")
    if (
        evidence_file.artifact_id != f"{lease.edition_id}:{attempt_index}:planning-attempt"
        or evidence_file.artifact_kind != "video_digest_planning_attempt"
    ):
        raise ValueError("Planning attempt evidence artifact identity is invalid")
    if disposition == "accepted":
        if accepted_plan is None or plan_file is None:
            raise ValueError("Accepted planning attempt requires its canonical plan")
        _validate_plan_checkpoint(lease, accepted_plan, plan_file)
    elif accepted_plan is not None or plan_file is not None:
        raise ValueError("Rejected planning attempt cannot register a canonical plan")

    def checkpoint(connection: CatalogConnection) -> DigestPlan | None:
        slot_row, current = _lock_slot_for_lease(connection, lease)
        row = connection.execute(
            """
            SELECT edition_id, attempt_index, disposition,
                   attempt_evidence_artifact_version_id,
                   accepted_plan_artifact_version_id
            FROM video_digest_planning_attempts
            WHERE edition_id = %s AND attempt_index = %s
            FOR UPDATE
            """,
            (lease.edition_id, attempt_index),
        ).fetchone()
        accepted_version = accepted_plan.artifact_version_id if accepted_plan is not None else None
        if row is not None:
            if (
                str(row["edition_id"]),
                int(row["attempt_index"]),
                str(row["disposition"]),
                str(row["attempt_evidence_artifact_version_id"]),
                (
                    str(row["accepted_plan_artifact_version_id"])
                    if row["accepted_plan_artifact_version_id"] is not None
                    else None
                ),
            ) != (
                lease.edition_id,
                attempt_index,
                disposition,
                evidence_file.version_id,
                accepted_version,
            ) or not _stored_artifact_matches(connection, evidence_file):
                raise VideoDigestCheckpointConflictError(
                    "Stored planning attempt conflicts with the request"
                )
            if accepted_plan is None or plan_file is None:
                return None
            return _checkpoint_plan_locked(
                connection,
                lease,
                accepted_plan,
                plan_file,
                slot_row,
                current,
            )
        if str(slot_row["stage"]) != SlotStage.CLAIMED.value:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot conflicts with the planning attempt"
            )

        _register_artifact(connection, evidence_file, current)
        if plan_file is not None:
            _register_artifact(connection, plan_file, current)
        connection.execute(
            """
            INSERT INTO video_digest_planning_attempts
                (edition_id, attempt_index, disposition,
                 attempt_evidence_artifact_version_id,
                 accepted_plan_artifact_version_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                lease.edition_id,
                attempt_index,
                disposition,
                evidence_file.version_id,
                accepted_version,
                current,
            ),
        )
        if accepted_plan is None or plan_file is None:
            return None
        return _checkpoint_plan_locked(
            connection,
            lease,
            accepted_plan,
            plan_file,
            slot_row,
            current,
            plan_artifact_registered=True,
        )

    return _checkpoint_transaction(checkpoint)


def checkpoint_plan(
    lease: SlotLease,
    plan: DigestPlan,
    *,
    plan_file: ArtifactFile,
    recorded_at: datetime,
) -> DigestPlan:
    _utc(recorded_at, "recorded_at")
    _validate_plan_checkpoint(lease, plan, plan_file)

    def checkpoint(connection: CatalogConnection) -> DigestPlan:
        row, current = _lock_slot_for_lease(connection, lease)
        return _checkpoint_plan_locked(connection, lease, plan, plan_file, row, current)

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


def checkpoint_edition_verification(
    lease: SlotLease,
    *,
    manifest_file: ArtifactFile,
    recorded_at: datetime,
) -> None:
    _utc(recorded_at, "recorded_at")
    if (
        manifest_file.artifact_id != f"{lease.edition_id}:verification-manifest"
        or manifest_file.artifact_kind != "video_digest_verification_manifest"
    ):
        raise ValueError("Edition verification manifest artifact identity is invalid")

    def checkpoint(connection: CatalogConnection) -> None:
        slot_row, current = _lock_slot_for_lease(connection, lease)
        edition_row = connection.execute(
            """
            SELECT plan_artifact_version_id, verification_manifest_artifact_version_id
            FROM video_digest_editions
            WHERE edition_id = %s
            FOR UPDATE
            """,
            (lease.edition_id,),
        ).fetchone()
        if edition_row is None:
            raise VideoDigestCheckpointConflictError("Video digest edition is unavailable")
        if edition_row["plan_artifact_version_id"] is None:
            raise VideoDigestCheckpointConflictError(
                "Edition verification requires an accepted digest plan"
            )
        stored_version = edition_row["verification_manifest_artifact_version_id"]
        if stored_version is not None:
            if stored_version == manifest_file.version_id and _stored_artifact_matches(
                connection, manifest_file
            ):
                return
            raise VideoDigestCheckpointConflictError(
                "Stored edition verification manifest conflicts with the request"
            )
        if str(slot_row["stage"]) != SlotStage.GENERATING.value:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot conflicts with edition verification"
            )
        incomplete = connection.execute(
            """
            SELECT story_id
            FROM video_digest_stories
            WHERE edition_id = %s AND mandatory
              AND (verification_evidence_artifact_version_id IS NULL
                   OR stage NOT IN ('generating', 'accepted'))
            LIMIT 1
            FOR UPDATE
            """,
            (lease.edition_id,),
        ).fetchone()
        if incomplete is not None:
            raise VideoDigestCheckpointConflictError(
                "Edition verification requires every mandatory story to be generation-ready"
            )

        _register_artifact(connection, manifest_file, current)
        _execute_returning(
            connection,
            """
            UPDATE video_digest_editions
            SET verification_manifest_artifact_version_id = %s, updated_at = %s
            WHERE edition_id = %s
              AND verification_manifest_artifact_version_id IS NULL
            RETURNING edition_id
            """,
            (manifest_file.version_id, current, lease.edition_id),
            VideoDigestCheckpointConflictError(
                "Stored video digest edition conflicts with edition verification"
            ),
        )

    _checkpoint_transaction(checkpoint)


def checkpoint_generation_request(
    lease: SlotLease,
    request: GenerationRequestIdentity,
    *,
    request_file: ArtifactFile,
    admission: GenerationAdmission,
    recorded_at: datetime,
) -> GenerationRequestCheckpoint:
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

    def checkpoint(connection: CatalogConnection) -> GenerationRequestCheckpoint:
        slot_row, current = _lock_slot_for_lease(connection, lease)
        stored = _lock_generation_request(connection, request.request_id)
        if stored is not None:
            _require_generation_identity(stored, request)
            if not _stored_artifact_matches(
                connection, request_file
            ) or not _stored_generation_admission_matches(
                connection, request.request_id, admission
            ):
                raise VideoDigestCheckpointConflictError(
                    "Stored generation request admission conflicts with the request"
                )
            return GenerationRequestCheckpoint(state=_generation_state(stored), created=False)
        if str(slot_row["stage"]) != SlotStage.GENERATING.value:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot conflicts with the generation request"
            )
        edition_row = connection.execute(
            """
            SELECT plan_artifact_version_id, verification_manifest_artifact_version_id
            FROM video_digest_editions
            WHERE edition_id = %s
            FOR UPDATE
            """,
            (request.edition_id,),
        ).fetchone()
        if edition_row is None or edition_row["plan_artifact_version_id"] is None:
            raise VideoDigestCheckpointConflictError(
                "Generation request requires an accepted digest plan"
            )
        if edition_row["verification_manifest_artifact_version_id"] is None:
            raise VideoDigestCheckpointConflictError(
                "Generation request requires an edition verification manifest"
            )
        incomplete = connection.execute(
            """
            SELECT story_id
            FROM video_digest_stories
            WHERE edition_id = %s AND mandatory
              AND (verification_evidence_artifact_version_id IS NULL
                   OR stage NOT IN ('generating', 'accepted'))
            LIMIT 1
            FOR UPDATE
            """,
            (request.edition_id,),
        ).fetchone()
        if incomplete is not None:
            raise VideoDigestCheckpointConflictError(
                "Generation request requires every mandatory story to be generation-ready"
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

        scheduled_at = _utc(cast(datetime, slot_row["scheduled_at"]), "scheduled_at")
        deadline_at = scheduled_at + timedelta(minutes=90)
        _register_artifact(connection, request_file, current)
        connection.execute(
            """
            INSERT INTO video_digest_generation_requests
                (request_id, edition_id, story_position, attempt_index,
                 request_artifact_version_id, generation_policy_artifact_version_id,
                 reserved_cost_usd, deadline_at, stage, cost_kind, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', 'pending', %s, %s)
            """,
            (
                request.request_id,
                request.edition_id,
                request.story_position,
                request.attempt_index,
                request.request_artifact_version_id,
                admission.generation_policy_artifact_version_id,
                admission.reserved_usd,
                deadline_at,
                current,
                current,
            ),
        )
        bucharest_day = cast(date, slot_row["bucharest_day"])
        scopes = (
            ("story", str(story_row["story_id"]), admission.limits.story_usd),
            ("edition", str(request.edition_id), admission.limits.edition_usd),
            ("bucharest_day", bucharest_day.isoformat(), admission.limits.bucharest_day_usd),
            (
                "calendar_month",
                bucharest_day.replace(day=1).isoformat(),
                admission.limits.calendar_month_usd,
            ),
        )
        connection.execute(
            """
            INSERT INTO video_digest_generation_reservations
                (request_id, scope_kind, scope_key, limit_usd, reserved_usd,
                 generation_policy_artifact_version_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s),
                   (%s, %s, %s, %s, %s, %s, %s),
                   (%s, %s, %s, %s, %s, %s, %s),
                   (%s, %s, %s, %s, %s, %s, %s)
            """,
            tuple(
                value
                for scope_kind, scope_key, limit_usd in scopes
                for value in (
                    request.request_id,
                    scope_kind,
                    scope_key,
                    limit_usd,
                    admission.reserved_usd,
                    admission.generation_policy_artifact_version_id,
                    current,
                )
            ),
        )
        return GenerationRequestCheckpoint(
            state=GenerationRequestState(
                request_id=request.request_id,
                stage=GenerationStage.PENDING,
                cost=PendingAttemptCost(),
            ),
            created=True,
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
    validation_file: ArtifactFile,
    cost: MeasuredAttemptCost | UnknownAttemptCost,
    recorded_at: datetime,
) -> GenerationRequestState:
    _utc(recorded_at, "recorded_at")
    if clip_file.artifact_kind != "video_digest_accepted_clip":
        raise ValueError("Accepted clip artifact kind is invalid")
    if (
        validation_file.artifact_id != f"{request_id}:validation"
        or validation_file.artifact_kind != "video_digest_candidate_validation"
    ):
        raise ValueError("Candidate validation artifact identity is invalid")

    def checkpoint(connection: CatalogConnection) -> GenerationRequestState:
        slot_row, current = _lock_slot_for_lease(connection, lease)
        row = _required_generation_request(connection, request_id, lease.edition_id)
        story = _lock_story_at_position(connection, lease.edition_id, int(row["story_position"]))
        if clip_file.artifact_id != f"{story['story_id']}:accepted-clip":
            raise ValueError("Accepted clip artifact identity is invalid")
        if str(row["stage"]) == GenerationStage.ACCEPTED.value:
            if (
                row["accepted_clip_artifact_version_id"] == clip_file.version_id
                and row["validation_evidence_artifact_version_id"] == validation_file.version_id
                and story["accepted_clip_artifact_version_id"] == clip_file.version_id
                and _cost_from_row(row) == cost
                and _stored_artifact_matches(connection, clip_file)
                and _stored_artifact_matches(connection, validation_file)
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
        _register_artifact(connection, validation_file, current)
        cost_kind, cost_usd, unknown_reason = _cost_columns(cost)
        _execute_returning(
            connection,
            """
            UPDATE video_digest_generation_requests
            SET stage = 'accepted', accepted_clip_artifact_version_id = %s,
                validation_evidence_artifact_version_id = %s,
                cost_kind = %s, cost_usd = %s, cost_unknown_reason = %s, updated_at = %s
            WHERE request_id = %s AND edition_id = %s AND stage = 'processing'
              AND response_artifact_version_id IS NOT NULL
              AND accepted_clip_artifact_version_id IS NULL
              AND failure_evidence_artifact_version_id IS NULL
            RETURNING request_id
            """,
            (
                clip_file.version_id,
                validation_file.version_id,
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


def checkpoint_assembled_video(
    lease: SlotLease,
    *,
    video_file: ArtifactFile,
    manifest_file: ArtifactFile,
    recorded_at: datetime,
) -> AssembledVideo:
    _utc(recorded_at, "recorded_at")
    if (
        video_file.artifact_id != f"{lease.edition_id}:assembled-video"
        or video_file.artifact_kind != "video_digest_assembled_video"
    ):
        raise ValueError("Assembled video artifact identity is invalid")
    if (
        manifest_file.artifact_id != f"{lease.edition_id}:assembly-manifest"
        or manifest_file.artifact_kind != "video_digest_assembly_manifest"
    ):
        raise ValueError("Assembly manifest artifact identity is invalid")

    def checkpoint(connection: CatalogConnection) -> AssembledVideo:
        slot_row, current = _lock_slot_for_lease(connection, lease)
        edition_row = _lock_edition_outputs(connection, lease.edition_id)
        stored_version = edition_row["assembled_video_artifact_version_id"]
        stored_manifest = edition_row["assembly_manifest_artifact_version_id"]
        if stored_version is not None:
            if (
                stored_version == video_file.version_id
                and stored_manifest == manifest_file.version_id
                and _stored_artifact_matches(connection, video_file)
                and _stored_artifact_matches(connection, manifest_file)
            ):
                return AssembledVideo(
                    edition_id=lease.edition_id,
                    artifact_version_id=video_file.version_id,
                )
            raise VideoDigestCheckpointConflictError(
                "Stored assembled video conflicts with the request"
            )
        if str(slot_row["stage"]) != SlotStage.ASSEMBLING.value:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot conflicts with assembly completion"
            )
        incomplete = connection.execute(
            """
            SELECT story_id
            FROM video_digest_stories
            WHERE edition_id = %s
              AND (stage <> 'accepted' OR accepted_clip_artifact_version_id IS NULL)
            LIMIT 1
            """,
            (lease.edition_id,),
        ).fetchone()
        if incomplete is not None:
            raise VideoDigestCheckpointConflictError(
                "Video digest edition has incomplete mandatory stories"
            )

        _register_artifact(connection, video_file, current)
        _register_artifact(connection, manifest_file, current)
        _execute_returning(
            connection,
            """
            UPDATE video_digest_editions
            SET assembled_video_artifact_version_id = %s,
                assembly_manifest_artifact_version_id = %s, updated_at = %s
            WHERE edition_id = %s AND assembled_video_artifact_version_id IS NULL
            RETURNING edition_id
            """,
            (video_file.version_id, manifest_file.version_id, current, lease.edition_id),
            VideoDigestCheckpointConflictError(
                "Stored video digest edition conflicts with assembly completion"
            ),
        )
        _advance_slot(
            connection,
            lease,
            current,
            from_stage=SlotStage.ASSEMBLING,
            to_stage=SlotStage.SUBTITLING,
        )
        return AssembledVideo(
            edition_id=lease.edition_id,
            artifact_version_id=video_file.version_id,
        )

    return _checkpoint_transaction(checkpoint)


def checkpoint_subtitles(
    lease: SlotLease,
    outcome: SubtitleOutcome,
    *,
    artifact_file: ArtifactFile,
    recorded_at: datetime,
) -> SubtitleOutcome:
    _utc(recorded_at, "recorded_at")
    expected_id = (
        f"{lease.edition_id}:subtitles"
        if isinstance(outcome, AvailableSubtitles)
        else f"{lease.edition_id}:subtitle-failure"
    )
    expected_kind = (
        "video_digest_subtitles"
        if isinstance(outcome, AvailableSubtitles)
        else "video_digest_subtitle_failure"
    )
    outcome_version = (
        outcome.artifact_version_id
        if isinstance(outcome, AvailableSubtitles)
        else outcome.evidence_artifact_version_id
    )
    if (
        artifact_file.artifact_id != expected_id
        or artifact_file.artifact_kind != expected_kind
        or artifact_file.version_id != outcome_version
    ):
        raise ValueError("Subtitle artifact identity is invalid")

    def checkpoint(connection: CatalogConnection) -> SubtitleOutcome:
        slot_row, current = _lock_slot_for_lease(connection, lease)
        edition_row = _lock_edition_outputs(connection, lease.edition_id)
        stored = _subtitle_outcome_from_row(edition_row)
        if stored is not None:
            if stored == outcome and _stored_artifact_matches(connection, artifact_file):
                return stored
            raise VideoDigestCheckpointConflictError(
                "Stored subtitle outcome conflicts with the request"
            )
        if str(slot_row["stage"]) != SlotStage.SUBTITLING.value:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot conflicts with subtitle completion"
            )

        _register_artifact(connection, artifact_file, current)
        if isinstance(outcome, AvailableSubtitles):
            subtitle_state = "available"
            subtitle_version = outcome.artifact_version_id
            failure_version = None
        else:
            subtitle_state = "failed"
            subtitle_version = None
            failure_version = outcome.evidence_artifact_version_id
        _execute_returning(
            connection,
            """
            UPDATE video_digest_editions
            SET subtitle_state = %s, subtitle_artifact_version_id = %s,
                subtitle_failure_evidence_artifact_version_id = %s, updated_at = %s
            WHERE edition_id = %s AND subtitle_state = 'pending'
              AND subtitle_artifact_version_id IS NULL
              AND subtitle_failure_evidence_artifact_version_id IS NULL
            RETURNING edition_id
            """,
            (
                subtitle_state,
                subtitle_version,
                failure_version,
                current,
                lease.edition_id,
            ),
            VideoDigestCheckpointConflictError(
                "Stored video digest edition conflicts with subtitle completion"
            ),
        )
        _advance_slot(
            connection,
            lease,
            current,
            from_stage=SlotStage.SUBTITLING,
            to_stage=SlotStage.PUBLISHING,
        )
        return outcome

    return _checkpoint_transaction(checkpoint)


def record_publication_intent(
    lease: SlotLease,
    intent: PublicationIntent,
    *,
    recorded_at: datetime,
) -> PublicationStatus:
    _utc(recorded_at, "recorded_at")
    if intent.edition_id != lease.edition_id:
        raise ValueError("Publication intent edition does not match the slot lease")

    def checkpoint(connection: CatalogConnection) -> PublicationStatus:
        slot_row = _lock_slot(connection, lease.slot_id)
        current: datetime | None = None
        terminal = str(slot_row["stage"]) in {
            SlotStage.FAILED.value,
            SlotStage.PUBLISHED.value,
        }
        if terminal:
            if not _terminal_fence_matches(slot_row, lease):
                raise VideoDigestLeaseLostError("Video digest slot lease was lost")
        else:
            current = _validate_locked_lease(connection, slot_row, lease)
            if str(slot_row["stage"]) != SlotStage.PUBLISHING.value:
                raise VideoDigestCheckpointConflictError(
                    "Stored video digest slot conflicts with the publication intent"
                )

        edition_row = _lock_edition_outputs(connection, lease.edition_id)
        _validate_intent_sources(connection, intent, edition_row)
        if not terminal:
            assert current is not None
            subtitle = intent.subtitle
            connection.execute(
                """
                INSERT INTO video_digest_publication_intents
                    (publication_id, edition_id, expected_video_key, video_digest,
                     video_byte_size, video_media_type, subtitle_expected_key,
                     subtitle_digest, subtitle_byte_size, subtitle_media_type,
                     source_video_artifact_version_id,
                     source_subtitle_artifact_version_id, stage, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        'pending', %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    intent.publication_id,
                    intent.edition_id,
                    intent.expected_video_key,
                    intent.video_digest,
                    intent.video_byte_size,
                    intent.video_media_type,
                    subtitle.expected_key if subtitle is not None else None,
                    subtitle.content_digest if subtitle is not None else None,
                    subtitle.byte_size if subtitle is not None else None,
                    subtitle.media_type if subtitle is not None else None,
                    intent.source_video_version_id,
                    intent.source_subtitle_version_id,
                    current,
                    current,
                ),
            )
        row = _lock_publication_for_edition(connection, lease.edition_id)
        if row is None or not _publication_intent_matches(row, intent):
            raise VideoDigestCheckpointConflictError(
                "Stored publication intent conflicts with the request"
            )
        return _publication_status(row)

    return _checkpoint_transaction(checkpoint)


def checkpoint_publication_progress(
    lease: SlotLease,
    publication_id: PublicationId,
    progress: PublicationProgress,
    *,
    evidence_file: ArtifactFile | None = None,
    recorded_at: datetime,
) -> PublicationStatus:
    _utc(recorded_at, "recorded_at")
    target = PublicationState(progress.kind)
    if isinstance(progress, UploadingPublication):
        if evidence_file is not None:
            raise ValueError("Uploading publication progress cannot include evidence")
    else:
        expected_kind = (
            "video_digest_publication_upload"
            if isinstance(progress, UploadedPublication)
            else "video_digest_publication_verification"
        )
        expected_suffix = "upload" if isinstance(progress, UploadedPublication) else "verification"
        if (
            evidence_file is None
            or evidence_file.artifact_id != f"{publication_id}:{expected_suffix}"
            or evidence_file.artifact_kind != expected_kind
            or evidence_file.version_id != progress.evidence_artifact_version_id
        ):
            raise ValueError("Publication progress evidence identity is invalid")

    def checkpoint(connection: CatalogConnection) -> PublicationStatus:
        slot_row = _lock_slot(connection, lease.slot_id)
        slot_stage = SlotStage(str(slot_row["stage"]))
        terminal_replay = slot_stage == SlotStage.PUBLISHED
        if terminal_replay:
            if not _terminal_fence_matches(slot_row, lease):
                raise VideoDigestLeaseLostError("Video digest slot lease was lost")
            current = None
        else:
            current = _validate_locked_lease(connection, slot_row, lease)
        if slot_stage not in {SlotStage.PUBLISHING, SlotStage.PUBLISHED}:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot conflicts with publication progress"
            )
        row = _required_publication(connection, publication_id, lease.edition_id)
        if isinstance(progress, VerifiedPublication):
            _validate_verified_publication(progress, row)
        current_stage = PublicationState(str(row["stage"]))
        if terminal_replay:
            if current_stage != PublicationState.PUBLISHED:
                raise VideoDigestCheckpointConflictError(
                    "Stored publication completion conflicts with progress"
                )
            _require_progress_evidence(connection, row, progress, evidence_file)
            return _publication_status(row)
        ranks = {
            PublicationState.PENDING: 0,
            PublicationState.UPLOADING: 1,
            PublicationState.UPLOADED: 2,
            PublicationState.VERIFIED: 3,
        }
        if current_stage not in ranks or target not in ranks:
            raise VideoDigestCheckpointConflictError(
                "Stored publication is terminal or incompatible with progress"
            )
        if ranks[current_stage] >= ranks[target]:
            _require_progress_evidence(connection, row, progress, evidence_file)
            return _publication_status(row)
        if ranks[target] != ranks[current_stage] + 1:
            raise VideoDigestCheckpointConflictError(
                "Publication progress cannot skip a checkpoint"
            )

        assert current is not None
        if evidence_file is not None:
            _register_artifact(connection, evidence_file, current)
        upload_version = (
            progress.evidence_artifact_version_id
            if isinstance(progress, UploadedPublication)
            else None
        )
        verification_version = (
            progress.evidence_artifact_version_id
            if isinstance(progress, VerifiedPublication)
            else None
        )
        _execute_returning(
            connection,
            """
            UPDATE video_digest_publication_intents
            SET stage = %s,
                upload_evidence_artifact_version_id = COALESCE(%s, upload_evidence_artifact_version_id),
                verification_evidence_artifact_version_id = COALESCE(
                    %s, verification_evidence_artifact_version_id
                ),
                updated_at = %s
            WHERE publication_id = %s AND edition_id = %s AND stage = %s
            RETURNING publication_id
            """,
            (
                target.value,
                upload_version,
                verification_version,
                current,
                publication_id,
                lease.edition_id,
                current_stage.value,
            ),
            VideoDigestCheckpointConflictError("Stored publication conflicts with progress"),
        )
        return PublicationStatus(
            publication_id=publication_id,
            edition_id=lease.edition_id,
            stage=target,
        )

    return _checkpoint_transaction(checkpoint)


def complete_publication(
    lease: SlotLease,
    publication_id: PublicationId,
    *,
    recorded_at: datetime,
) -> PublishedPublication:
    _utc(recorded_at, "recorded_at")

    def checkpoint(connection: CatalogConnection) -> PublishedPublication:
        slot_row = _lock_slot(connection, lease.slot_id)
        if str(slot_row["stage"]) == SlotStage.PUBLISHED.value:
            if not _terminal_fence_matches(slot_row, lease):
                raise VideoDigestLeaseLostError("Video digest slot lease was lost")
            row = _required_publication(connection, publication_id, lease.edition_id)
            if str(row["stage"]) != PublicationState.PUBLISHED.value or row["published_at"] is None:
                raise VideoDigestCheckpointConflictError(
                    "Stored publication completion conflicts with the request"
                )
            return PublishedPublication(
                publication_id=publication_id,
                edition_id=lease.edition_id,
                published_at=_datetime(row["published_at"]),
            )

        current = _validate_locked_lease(connection, slot_row, lease)
        if str(slot_row["stage"]) != SlotStage.PUBLISHING.value:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot conflicts with publication completion"
            )
        row = _required_publication(connection, publication_id, lease.edition_id)
        if str(row["stage"]) != PublicationState.VERIFIED.value or (
            bool(row["evidence_required"])
            and (
                row["upload_evidence_artifact_version_id"] is None
                or row["verification_evidence_artifact_version_id"] is None
            )
        ):
            raise VideoDigestCheckpointConflictError("Stored publication is not verified")
        _execute_returning(
            connection,
            """
            UPDATE video_digest_publication_intents
            SET stage = 'published', published_at = %s, updated_at = %s
            WHERE publication_id = %s AND edition_id = %s AND stage = 'verified'
              AND (
                  NOT evidence_required
                  OR (upload_evidence_artifact_version_id IS NOT NULL
                      AND verification_evidence_artifact_version_id IS NOT NULL)
              )
            RETURNING publication_id
            """,
            (current, current, publication_id, lease.edition_id),
            VideoDigestCheckpointConflictError("Stored publication conflicts with completion"),
        )
        _terminalize_slot(
            connection,
            lease,
            current,
            stage=SlotStage.PUBLISHED,
            failure_version=None,
        )
        return PublishedPublication(
            publication_id=publication_id,
            edition_id=lease.edition_id,
            published_at=current,
        )

    return _checkpoint_transaction(checkpoint)


def fail_slot(
    lease: SlotLease,
    *,
    evidence_file: ArtifactFile,
    recorded_at: datetime,
) -> TerminalSlot:
    _validate_failure_file(lease.slot_id, evidence_file)
    _utc(recorded_at, "recorded_at")

    def checkpoint(connection: CatalogConnection) -> TerminalSlot:
        slot_row = _lock_slot(connection, lease.slot_id)
        if str(slot_row["stage"]) == SlotStage.FAILED.value:
            _require_terminal_failure_replay(connection, slot_row, lease, evidence_file)
            if _lock_publication_for_edition(connection, lease.edition_id) is not None:
                raise VideoDigestCheckpointConflictError(
                    "A publication intent requires publication failure"
                )
            return TerminalSlot(state=TerminalSlotState.FAILED)
        current = _validate_locked_lease(connection, slot_row, lease)
        if _lock_publication_for_edition(connection, lease.edition_id) is not None:
            raise VideoDigestCheckpointConflictError(
                "A publication intent requires publication failure"
            )
        active_request = connection.execute(
            """
            SELECT request_id
            FROM video_digest_generation_requests
            WHERE edition_id = %s AND stage IN ('pending', 'submitted', 'processing')
            LIMIT 1
            FOR UPDATE
            """,
            (lease.edition_id,),
        ).fetchone()
        if active_request is not None:
            raise VideoDigestCheckpointConflictError(
                "Active generation must be failed through its request checkpoint"
            )
        _register_artifact(connection, evidence_file, current)
        connection.execute(
            """
            UPDATE video_digest_stories
            SET stage = 'failed', failure_evidence_artifact_version_id = %s,
                updated_at = %s
            WHERE edition_id = %s
              AND stage IN ('planned', 'verifying', 'verified', 'generating')
              AND accepted_clip_artifact_version_id IS NULL
              AND failure_evidence_artifact_version_id IS NULL
            """,
            (evidence_file.version_id, current, lease.edition_id),
        )
        _terminalize_slot(
            connection,
            lease,
            current,
            stage=SlotStage.FAILED,
            failure_version=evidence_file.version_id,
        )
        return TerminalSlot(state=TerminalSlotState.FAILED)

    return _checkpoint_transaction(checkpoint)


def fail_publication(
    lease: SlotLease,
    publication_id: PublicationId,
    *,
    state: Literal[PublicationState.CONFLICT, PublicationState.FAILED],
    evidence_file: ArtifactFile,
    recorded_at: datetime,
) -> TerminalSlot:
    _utc(recorded_at, "recorded_at")
    if (
        evidence_file.artifact_id != f"{publication_id}:failure"
        or evidence_file.artifact_kind != "video_digest_publication_failure"
    ):
        raise ValueError("Publication failure artifact identity is invalid")

    def checkpoint(connection: CatalogConnection) -> TerminalSlot:
        slot_row = _lock_slot(connection, lease.slot_id)
        if str(slot_row["stage"]) == SlotStage.FAILED.value:
            _require_terminal_failure_replay(connection, slot_row, lease, evidence_file)
            row = _required_publication(connection, publication_id, lease.edition_id)
            if (
                str(row["stage"]) != state.value
                or row["failure_evidence_artifact_version_id"] != evidence_file.version_id
            ):
                raise VideoDigestCheckpointConflictError(
                    "Stored publication failure conflicts with the request"
                )
            return TerminalSlot(state=TerminalSlotState.FAILED)

        current = _validate_locked_lease(connection, slot_row, lease)
        if str(slot_row["stage"]) != SlotStage.PUBLISHING.value:
            raise VideoDigestCheckpointConflictError(
                "Stored video digest slot conflicts with publication failure"
            )
        row = _required_publication(connection, publication_id, lease.edition_id)
        if str(row["stage"]) not in {
            PublicationState.PENDING.value,
            PublicationState.UPLOADING.value,
            PublicationState.UPLOADED.value,
            PublicationState.VERIFIED.value,
        }:
            raise VideoDigestCheckpointConflictError("Stored publication conflicts with failure")
        _register_artifact(connection, evidence_file, current)
        _execute_returning(
            connection,
            """
            UPDATE video_digest_publication_intents
            SET stage = %s, failure_evidence_artifact_version_id = %s, updated_at = %s
            WHERE publication_id = %s AND edition_id = %s
              AND stage IN ('pending', 'uploading', 'uploaded', 'verified')
            RETURNING publication_id
            """,
            (
                state.value,
                evidence_file.version_id,
                current,
                publication_id,
                lease.edition_id,
            ),
            VideoDigestCheckpointConflictError("Stored publication conflicts with failure"),
        )
        _terminalize_slot(
            connection,
            lease,
            current,
            stage=SlotStage.FAILED,
            failure_version=evidence_file.version_id,
        )
        return TerminalSlot(state=TerminalSlotState.FAILED)

    return _checkpoint_transaction(checkpoint)


def list_published_editions(
    day: date,
    *,
    public_media_base_url: str,
    limit: int = 20,
) -> tuple[PublishedEditionSummary, ...]:
    base_url = _public_media_origin(public_media_base_url)
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    rows = catalog_query(
        """
        SELECT publication.publication_id, publication.edition_id,
               publication.expected_video_key, publication.video_digest,
               publication.video_byte_size, publication.video_media_type,
               publication.subtitle_expected_key, publication.subtitle_digest,
               publication.subtitle_byte_size, publication.subtitle_media_type,
               publication.published_at, edition.daily_report_version_id,
               edition.subtitle_state, slot.name AS slot_name,
               slot.scheduled_at, slot.bucharest_day AS day
        FROM video_digest_publication_intents AS publication
        JOIN video_digest_editions AS edition ON edition.edition_id = publication.edition_id
        JOIN video_digest_slots AS slot ON slot.edition_id = publication.edition_id
        WHERE publication.stage = 'published' AND slot.stage = 'published'
          AND slot.bucharest_day = %s
        ORDER BY publication.published_at DESC, publication.publication_id DESC
        LIMIT %s
        """,
        [day, limit],
    )
    return tuple(_published_summary(row, base_url) for row in rows)


def read_published_edition(
    edition_id: EditionId,
    *,
    public_media_base_url: str,
) -> PublishedEdition | None:
    base_url = _public_media_origin(public_media_base_url)
    rows = catalog_query(
        """
        SELECT publication.publication_id, publication.edition_id,
               publication.expected_video_key, publication.video_digest,
               publication.video_byte_size, publication.video_media_type,
               publication.subtitle_expected_key, publication.subtitle_digest,
               publication.subtitle_byte_size, publication.subtitle_media_type,
               publication.published_at, edition.daily_report_version_id,
               edition.subtitle_state, slot.name AS slot_name,
               slot.scheduled_at, slot.bucharest_day AS day
        FROM video_digest_publication_intents AS publication
        JOIN video_digest_editions AS edition ON edition.edition_id = publication.edition_id
        JOIN video_digest_slots AS slot ON slot.edition_id = publication.edition_id
        WHERE publication.edition_id = %s
          AND publication.stage = 'published' AND slot.stage = 'published'
        """,
        [edition_id],
    )
    if not rows:
        return None
    if len(rows) != 1:
        raise ResearchCatalogError("PostgreSQL returned duplicate published video digests")
    story_rows = catalog_query(
        """
        SELECT story_id, position, report_subject_id, title, requested_duration_ms
        FROM video_digest_stories
        WHERE edition_id = %s
        ORDER BY position ASC, story_id ASC
        """,
        [edition_id],
    )
    summary = _published_summary(rows[0], base_url)
    return PublishedEdition(
        **summary.model_dump(),
        stories=tuple(PublishedStory.model_validate(row) for row in story_rows),
    )


def _lock_edition_outputs(
    connection: CatalogConnection,
    edition_id: EditionId,
) -> Mapping[str, Any]:
    row = connection.execute(
        """
        SELECT edition_id, assembled_video_artifact_version_id,
               assembly_manifest_artifact_version_id, subtitle_state,
               subtitle_artifact_version_id,
               subtitle_failure_evidence_artifact_version_id
        FROM video_digest_editions
        WHERE edition_id = %s
        FOR UPDATE
        """,
        (edition_id,),
    ).fetchone()
    if row is None:
        raise VideoDigestCheckpointConflictError("Video digest edition is unavailable")
    return row


def _subtitle_outcome_from_row(row: Mapping[str, Any]) -> SubtitleOutcome | None:
    state = str(row["subtitle_state"])
    if state == "pending":
        return None
    if state == "available":
        return AvailableSubtitles(artifact_version_id=row["subtitle_artifact_version_id"])
    if state == "failed":
        return FailedSubtitles(
            evidence_artifact_version_id=row["subtitle_failure_evidence_artifact_version_id"]
        )
    raise VideoDigestCheckpointConflictError("Stored subtitle state is invalid")


def _advance_slot(
    connection: CatalogConnection,
    lease: SlotLease,
    current: datetime,
    *,
    from_stage: SlotStage,
    to_stage: SlotStage,
) -> None:
    _execute_returning(
        connection,
        """
        UPDATE video_digest_slots
        SET stage = %s, updated_at = %s
        WHERE slot_id = %s AND edition_id = %s AND lease_owner_token = %s
          AND lease_expires_at = %s AND claim_count = %s AND stage = %s
        RETURNING slot_id
        """,
        (
            to_stage.value,
            current,
            lease.slot_id,
            lease.edition_id,
            lease.owner_token,
            lease.expires_at,
            lease.claim_count,
            from_stage.value,
        ),
        VideoDigestLeaseLostError("Video digest slot lease was lost"),
    )


def _lock_publication_for_edition(
    connection: CatalogConnection,
    edition_id: EditionId,
) -> Mapping[str, Any] | None:
    return connection.execute(
        """
        SELECT publication_id, edition_id, expected_video_key, video_digest,
               video_byte_size, video_media_type, subtitle_expected_key,
               subtitle_digest, subtitle_byte_size, subtitle_media_type,
               source_video_artifact_version_id, source_subtitle_artifact_version_id,
               stage, evidence_required, upload_evidence_artifact_version_id,
               verification_evidence_artifact_version_id,
               failure_evidence_artifact_version_id, published_at
        FROM video_digest_publication_intents
        WHERE edition_id = %s
        FOR UPDATE
        """,
        (edition_id,),
    ).fetchone()


def _required_publication(
    connection: CatalogConnection,
    publication_id: PublicationId,
    edition_id: EditionId,
) -> Mapping[str, Any]:
    row = _lock_publication_for_edition(connection, edition_id)
    if row is None or str(row["publication_id"]) != publication_id:
        raise VideoDigestCheckpointConflictError("Video digest publication is unavailable")
    return row


def _publication_status(row: Mapping[str, Any]) -> PublicationStatus:
    return PublicationStatus(
        publication_id=PublicationId(str(row["publication_id"])),
        edition_id=EditionId(str(row["edition_id"])),
        stage=PublicationState(str(row["stage"])),
    )


def _publication_intent_matches(row: Mapping[str, Any], intent: PublicationIntent) -> bool:
    subtitle = intent.subtitle
    return (
        str(row["publication_id"]),
        str(row["edition_id"]),
        str(row["expected_video_key"]),
        str(row["video_digest"]),
        int(row["video_byte_size"]),
        str(row["video_media_type"]),
        str(row["subtitle_expected_key"]) if row["subtitle_expected_key"] is not None else None,
        str(row["subtitle_digest"]) if row["subtitle_digest"] is not None else None,
        int(row["subtitle_byte_size"]) if row["subtitle_byte_size"] is not None else None,
        str(row["subtitle_media_type"]) if row["subtitle_media_type"] is not None else None,
        str(row["source_video_artifact_version_id"]),
        (
            str(row["source_subtitle_artifact_version_id"])
            if row["source_subtitle_artifact_version_id"] is not None
            else None
        ),
    ) == (
        intent.publication_id,
        intent.edition_id,
        intent.expected_video_key,
        intent.video_digest,
        intent.video_byte_size,
        intent.video_media_type,
        subtitle.expected_key if subtitle is not None else None,
        subtitle.content_digest if subtitle is not None else None,
        subtitle.byte_size if subtitle is not None else None,
        subtitle.media_type if subtitle is not None else None,
        intent.source_video_version_id,
        intent.source_subtitle_version_id,
    )


def _validate_intent_sources(
    connection: CatalogConnection,
    intent: PublicationIntent,
    edition_row: Mapping[str, Any],
) -> None:
    subtitle_state = str(edition_row["subtitle_state"])
    expected_subtitle = (
        str(edition_row["subtitle_artifact_version_id"])
        if edition_row["subtitle_artifact_version_id"] is not None
        else None
    )
    if (
        str(edition_row["assembled_video_artifact_version_id"]) != intent.source_video_version_id
        or expected_subtitle != intent.source_subtitle_version_id
        or (subtitle_state == "available") != (intent.subtitle is not None)
        or subtitle_state not in {"available", "failed"}
    ):
        raise VideoDigestCheckpointConflictError(
            "Publication intent sources conflict with the edition"
        )
    video_metadata = _artifact_metadata(connection, intent.source_video_version_id)
    if video_metadata != (
        intent.video_digest,
        intent.video_byte_size,
        intent.video_media_type,
    ):
        raise VideoDigestCheckpointConflictError(
            "Publication video metadata conflicts with its source artifact"
        )
    if intent.subtitle is not None and intent.source_subtitle_version_id is not None:
        subtitle_metadata = _artifact_metadata(connection, intent.source_subtitle_version_id)
        if subtitle_metadata != (
            intent.subtitle.content_digest,
            intent.subtitle.byte_size,
            intent.subtitle.media_type,
        ):
            raise VideoDigestCheckpointConflictError(
                "Publication subtitle metadata conflicts with its source artifact"
            )


def _artifact_metadata(
    connection: CatalogConnection,
    version_id: str,
) -> tuple[str, int, str]:
    row = connection.execute(
        """
        SELECT version.content_digest, file.byte_size, file.media_type
        FROM artifact_versions AS version
        JOIN artifact_files AS file ON file.artifact_version_id = version.id
        WHERE version.id = %s
        """,
        (version_id,),
    ).fetchone()
    if row is None:
        raise VideoDigestCheckpointConflictError(
            "Publication source artifact metadata is unavailable"
        )
    return (
        str(row["content_digest"]),
        int(row["byte_size"]),
        str(row["media_type"]),
    )


def _validate_verified_publication(
    progress: VerifiedPublication,
    row: Mapping[str, Any],
) -> None:
    video = progress.video
    if (
        video.content_digest,
        video.byte_size,
        video.media_type,
        video.source_artifact_version_id,
    ) != (
        str(row["video_digest"]),
        int(row["video_byte_size"]),
        str(row["video_media_type"]),
        str(row["source_video_artifact_version_id"]),
    ):
        raise VideoDigestCheckpointConflictError(
            "Verified public video conflicts with the publication intent"
        )
    subtitle = progress.subtitle
    if row["subtitle_expected_key"] is None:
        if subtitle is not None:
            raise VideoDigestCheckpointConflictError(
                "Verified subtitle conflicts with the publication intent"
            )
        return
    if subtitle is None or (
        subtitle.content_digest,
        subtitle.byte_size,
        subtitle.media_type,
        subtitle.source_artifact_version_id,
    ) != (
        str(row["subtitle_digest"]),
        int(row["subtitle_byte_size"]),
        str(row["subtitle_media_type"]),
        str(row["source_subtitle_artifact_version_id"]),
    ):
        raise VideoDigestCheckpointConflictError(
            "Verified subtitle conflicts with the publication intent"
        )


def _require_progress_evidence(
    connection: CatalogConnection,
    row: Mapping[str, Any],
    progress: PublicationProgress,
    evidence_file: ArtifactFile | None,
) -> None:
    if not bool(row["evidence_required"]):
        return
    if isinstance(progress, UploadingPublication):
        return
    field = (
        "upload_evidence_artifact_version_id"
        if isinstance(progress, UploadedPublication)
        else "verification_evidence_artifact_version_id"
    )
    if (
        evidence_file is None
        or row[field] != progress.evidence_artifact_version_id
        or not _stored_artifact_matches(connection, evidence_file)
    ):
        raise VideoDigestCheckpointConflictError(
            "Stored publication evidence conflicts with progress"
        )


def _terminalize_slot(
    connection: CatalogConnection,
    lease: SlotLease,
    current: datetime,
    *,
    stage: Literal[SlotStage.FAILED, SlotStage.PUBLISHED],
    failure_version: str | None,
) -> None:
    _execute_returning(
        connection,
        """
        UPDATE video_digest_slots
        SET stage = %s, lease_owner_token = NULL, lease_expires_at = NULL,
            terminal_lease_owner_token = %s, terminal_lease_expires_at = %s,
            terminal_claim_count = %s,
            failure_evidence_artifact_version_id = %s, updated_at = %s
        WHERE slot_id = %s AND edition_id = %s AND lease_owner_token = %s
          AND lease_expires_at = %s AND claim_count = %s
          AND stage IN ('claimed', 'planning', 'generating', 'assembling',
                        'subtitling', 'publishing')
        RETURNING slot_id
        """,
        (
            stage.value,
            lease.owner_token,
            lease.expires_at,
            lease.claim_count,
            failure_version,
            current,
            lease.slot_id,
            lease.edition_id,
            lease.owner_token,
            lease.expires_at,
            lease.claim_count,
        ),
        VideoDigestLeaseLostError("Video digest slot lease was lost"),
    )


def _validate_failure_file(slot_id: SlotId, evidence_file: ArtifactFile) -> None:
    if (
        evidence_file.artifact_id != f"{slot_id}:failure"
        or evidence_file.artifact_kind != "video_digest_failure"
    ):
        raise ValueError("Video digest failure artifact identity is invalid")


def _require_terminal_failure_replay(
    connection: CatalogConnection,
    slot_row: Mapping[str, Any],
    lease: SlotLease,
    evidence_file: ArtifactFile,
) -> None:
    if (
        not _terminal_fence_matches(slot_row, lease)
        or slot_row["failure_evidence_artifact_version_id"] != evidence_file.version_id
    ):
        raise VideoDigestLeaseLostError("Video digest slot lease was lost")
    if not _stored_artifact_matches(connection, evidence_file):
        raise VideoDigestCheckpointConflictError(
            "Stored video digest failure conflicts with the request"
        )


def _public_media_origin(value: str) -> str:
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("public_media_base_url must be an HTTPS origin")
    return f"https://{parsed.netloc}"


def _published_summary(row: Mapping[str, Any], base_url: str) -> PublishedEditionSummary:
    video = PublishedMedia.model_validate(
        {
            "url": _public_media_url(base_url, row["expected_video_key"]),
            "content_digest": row["video_digest"],
            "byte_size": row["video_byte_size"],
            "media_type": row["video_media_type"],
        }
    )
    subtitle_state = str(row["subtitle_state"])
    if subtitle_state == "available":
        subtitle = PublishedSubtitleAvailable(
            media=PublishedMedia.model_validate(
                {
                    "url": _public_media_url(base_url, row["subtitle_expected_key"]),
                    "content_digest": row["subtitle_digest"],
                    "byte_size": row["subtitle_byte_size"],
                    "media_type": row["subtitle_media_type"],
                }
            )
        )
    elif subtitle_state == "failed":
        subtitle = PublishedSubtitleFailed()
    else:
        raise ResearchCatalogError("Published video digest has unresolved subtitles")
    day_value = row["day"]
    return PublishedEditionSummary(
        edition_id=row["edition_id"],
        publication_id=row["publication_id"],
        day=day_value if isinstance(day_value, date) else date.fromisoformat(str(day_value)),
        slot_name=SlotName(str(row["slot_name"])),
        scheduled_at=_datetime(row["scheduled_at"]),
        published_at=_datetime(row["published_at"]),
        daily_report_version_id=row["daily_report_version_id"],
        video=video,
        subtitle=subtitle,
    )


def _public_media_url(base_url: str, raw_key: object) -> str:
    key = str(raw_key)
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", key) is None
        or key.endswith("/")
        or "//" in key
        or any(segment in {".", ".."} for segment in key.split("/"))
    ):
        raise ResearchCatalogError("Published video digest has an invalid object key")
    return f"{base_url}/{key}"


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


def _validate_plan_checkpoint(
    lease: SlotLease,
    plan: DigestPlan,
    plan_file: ArtifactFile,
) -> None:
    if plan.edition_id != lease.edition_id:
        raise ValueError("Digest plan edition does not match the slot lease")
    if plan_file.version_id != plan.artifact_version_id:
        raise ValueError("Digest plan artifact does not match the plan version")
    if plan_file.artifact_id != plan.edition_id or plan_file.artifact_kind != "video_digest_plan":
        raise ValueError("Digest plan artifact identity is invalid")


def _checkpoint_plan_locked(
    connection: CatalogConnection,
    lease: SlotLease,
    plan: DigestPlan,
    plan_file: ArtifactFile,
    slot_row: Mapping[str, Any],
    current: datetime,
    *,
    plan_artifact_registered: bool = False,
) -> DigestPlan:
    try:
        stage = SlotStage(str(slot_row["stage"]))
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

    if not plan_artifact_registered:
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
               validation_evidence_artifact_version_id,
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


def _stored_generation_admission_matches(
    connection: CatalogConnection,
    request_id: GenerationRequestId,
    admission: GenerationAdmission,
) -> bool:
    rows = connection.execute(
        """
        SELECT scope_kind, limit_usd, reserved_usd,
               generation_policy_artifact_version_id
        FROM video_digest_generation_reservations
        WHERE request_id = %s
        ORDER BY scope_kind
        """,
        (request_id,),
    ).fetchall()
    expected_limits = {
        "story": admission.limits.story_usd,
        "edition": admission.limits.edition_usd,
        "bucharest_day": admission.limits.bucharest_day_usd,
        "calendar_month": admission.limits.calendar_month_usd,
    }
    return len(rows) == 4 and all(
        str(row["scope_kind"]) in expected_limits
        and Decimal(str(row["limit_usd"])) == expected_limits[str(row["scope_kind"])]
        and Decimal(str(row["reserved_usd"])) == admission.reserved_usd
        and str(row["generation_policy_artifact_version_id"])
        == admission.generation_policy_artifact_version_id
        for row in rows
    )


def _generation_state(row: Mapping[str, Any]) -> GenerationRequestState:
    receipt = row["provider_receipt_id"]
    return GenerationRequestState(
        request_id=GenerationRequestId(str(row["request_id"])),
        stage=GenerationStage(str(row["stage"])),
        provider_receipt_id=str(receipt) if receipt is not None else None,
        cost=_cost_from_row(row),
    )


def _generation_attempt_reference(row: Mapping[str, Any]) -> GenerationAttemptReference:
    receipt_version = row["receipt_artifact_version_id"]
    receipt = None
    if receipt_version is not None:
        receipt = CatalogArtifactReference(
            artifact_id=str(row["provider_receipt_id"]),
            version_id=str(receipt_version),
            content_digest=str(row["receipt_content_digest"]),
            r2_key=str(row["receipt_r2_key"]),
        )
    response_version = row["response_artifact_version_id"]
    response = None
    if response_version is not None:
        response = CatalogArtifactReference(
            artifact_id=str(row["response_artifact_id"]),
            version_id=str(response_version),
            content_digest=str(row["response_content_digest"]),
            r2_key=str(row["response_r2_key"]),
        )
    clip_version = row["accepted_clip_artifact_version_id"]
    clip = None
    if clip_version is not None:
        clip = AcceptedClipReference(
            artifact_id=str(row["clip_artifact_id"]),
            version_id=str(clip_version),
            content_digest=str(row["clip_content_digest"]),
            r2_key=str(row["clip_r2_key"]),
            byte_size=int(row["clip_byte_size"]),
        )
    validation_version = row["validation_evidence_artifact_version_id"]
    validation = None
    if validation_version is not None:
        validation = CatalogArtifactReference(
            artifact_id=str(row["validation_artifact_id"]),
            version_id=str(validation_version),
            content_digest=str(row["validation_content_digest"]),
            r2_key=str(row["validation_r2_key"]),
        )
    return GenerationAttemptReference(
        request=GenerationRequestIdentity(
            request_id=GenerationRequestId(str(row["request_id"])),
            edition_id=EditionId(str(row["edition_id"])),
            story_position=int(row["story_position"]),
            attempt_index=int(row["attempt_index"]),
            request_artifact_version_id=str(row["request_artifact_version_id"]),
        ),
        stage=GenerationStage(str(row["stage"])),
        provider_receipt_id=(
            str(row["provider_receipt_id"]) if row["provider_receipt_id"] is not None else None
        ),
        cost=_cost_from_row(row),
        request_evidence=CatalogArtifactReference(
            artifact_id=str(row["request_artifact_id"]),
            version_id=str(row["request_artifact_version_id"]),
            content_digest=str(row["request_content_digest"]),
            r2_key=str(row["request_r2_key"]),
        ),
        receipt_evidence=receipt,
        response_evidence=response,
        accepted_clip=clip,
        validation_evidence=validation,
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
