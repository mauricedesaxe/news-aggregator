from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, TypeVar, cast

import pytest

from romanian_news import Sha256
from romanian_news.catalog import video_digest
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file
from romanian_news.catalog_transport import CatalogConnection, ResearchCatalogError
from romanian_news.video_digest.errors import (
    VideoDigestCheckpointConflictError,
    VideoDigestLeaseLostError,
)
from romanian_news.video_digest.models import (
    ClaimedSlot,
    DigestPlan,
    EditionIdentity,
    EstimatedAttemptCost,
    FailedSubtitles,
    GenerationAdmission,
    GenerationBudgetLimits,
    GenerationRequestIdentity,
    GenerationStage,
    MeasuredAttemptCost,
    PlannedStory,
    PublicationIntent,
    PublicationState,
    PublishedEdition,
    PublishedStory,
    ScheduledSlot,
    SkippedSlot,
    SlotId,
    SlotLease,
    SlotName,
    SlotSkipReason,
    TerminalSlot,
    TerminalSlotState,
    UnknownAttemptCost,
    UploadedPublication,
    UploadingPublication,
    VerifiedPublication,
    VerifiedPublicObject,
    edition_id,
    generation_request_id,
    planned_story_id,
    publication_id,
    scheduled_slot_id,
)

_Result = TypeVar("_Result")
NOW = datetime(2026, 9, 20, 9, tzinfo=UTC)
SCHEDULED_AT = datetime(2026, 9, 20, 6, tzinfo=UTC)
REPORT_ID: Sha256 = "1" * 64
POLICY_ID: Sha256 = "2" * 64
SUBJECT_ID: Sha256 = "3" * 64
EDITION = EditionIdentity(
    edition_id=edition_id(REPORT_ID, POLICY_ID),
    daily_report_version_id=REPORT_ID,
    policy_bundle_version_id=POLICY_ID,
)
SLOT = ScheduledSlot(
    slot_id=scheduled_slot_id(SlotName.MORNING, SCHEDULED_AT),
    name=SlotName.MORNING,
    scheduled_at=SCHEDULED_AT,
    bucharest_day=date(2026, 9, 20),
)
PLAN_FILE = artifact_file(
    artifact_id=EDITION.edition_id,
    artifact_kind="video_digest_plan",
    title="Video digest plan",
    content=b"plan",
    r2_key="video-digest/plans/plan.json",
    media_type="application/json",
)
STORY = PlannedStory(
    story_id=planned_story_id(EDITION.edition_id, 0, SUBJECT_ID),
    edition_id=EDITION.edition_id,
    position=0,
    report_subject_id=SUBJECT_ID,
    title="Story title",
    requested_duration_ms=15_000,
)
PLAN = DigestPlan(
    edition_id=EDITION.edition_id,
    artifact_version_id=PLAN_FILE.version_id,
    stories=(STORY,),
)
PLANNING_ATTEMPT_FILE = artifact_file(
    artifact_id=f"{EDITION.edition_id}:0:planning-attempt",
    artifact_kind="video_digest_planning_attempt",
    title="Planning attempt",
    content=b"planning evidence",
    r2_key="video-digest/planning/attempt-0.json",
    media_type="application/json",
)
EVIDENCE_FILE = artifact_file(
    artifact_id=STORY.story_id,
    artifact_kind="video_digest_story_verification",
    title="Story verification",
    content=b"verified",
    r2_key="video-digest/verification/evidence.json",
    media_type="application/json",
)
MANIFEST_FILE = artifact_file(
    artifact_id=f"{EDITION.edition_id}:verification-manifest",
    artifact_kind="video_digest_verification_manifest",
    title="Edition verification manifest",
    content=b"manifest",
    r2_key="video-digest/verification/manifest.json",
    media_type="application/json",
)
_request_payload_file = artifact_file(
    artifact_id=f"{EDITION.edition_id}:0:0:generation-request",
    artifact_kind="video_digest_generation_request",
    title="Generation request",
    content=b"request",
    r2_key="video-digest/generation/request.json",
    media_type="application/json",
)
REQUEST = GenerationRequestIdentity(
    request_id=generation_request_id(EDITION.edition_id, 0, 0, _request_payload_file.version_id),
    edition_id=EDITION.edition_id,
    story_position=0,
    attempt_index=0,
    request_artifact_version_id=_request_payload_file.version_id,
)
GENERATION_ADMISSION = GenerationAdmission(
    generation_policy_artifact_version_id=POLICY_ID,
    reserved_usd=Decimal("3.25632"),
    limits=GenerationBudgetLimits(
        story_usd=Decimal("7"),
        edition_usd=Decimal("7"),
        bucharest_day_usd=Decimal("150"),
        calendar_month_usd=Decimal("1000"),
    ),
)
REQUEST_FILE = _request_payload_file
RECEIPT_FILE = artifact_file(
    artifact_id="fal-receipt-1",
    artifact_kind="video_digest_provider_receipt",
    title="Fal provider receipt",
    content=b"receipt",
    r2_key="video-digest/generation/receipt.json",
    media_type="application/json",
)
RESPONSE_FILE = artifact_file(
    artifact_id=f"{REQUEST.request_id}:response",
    artifact_kind="video_digest_generation_response",
    title="Fal generation response",
    content=b"response",
    r2_key="video-digest/generation/response.json",
    media_type="application/json",
)
CLIP_FILE = artifact_file(
    artifact_id=f"{STORY.story_id}:accepted-clip",
    artifact_kind="video_digest_accepted_clip",
    title="Accepted story clip",
    content=b"clip",
    r2_key="video-digest/clips/accepted.mp4",
    media_type="video/mp4",
)
VALIDATION_FILE = artifact_file(
    artifact_id=f"{REQUEST.request_id}:validation",
    artifact_kind="video_digest_candidate_validation",
    title="Candidate validation",
    content=b"validation",
    r2_key="video-digest/generation/validation.json",
    media_type="application/json",
)
FAILURE_FILE = artifact_file(
    artifact_id=f"{REQUEST.request_id}:failure",
    artifact_kind="video_digest_generation_failure",
    title="Generation failure",
    content=b"failure",
    r2_key="video-digest/generation/failure.json",
    media_type="application/json",
)
ASSEMBLED_FILE = artifact_file(
    artifact_id=f"{EDITION.edition_id}:assembled-video",
    artifact_kind="video_digest_assembled_video",
    title="Assembled video digest",
    content=b"assembled video",
    r2_key="video-digest/assembled/video.mp4",
    media_type="video/mp4",
)
ASSEMBLY_MANIFEST_FILE = artifact_file(
    artifact_id=f"{EDITION.edition_id}:assembly-manifest",
    artifact_kind="video_digest_assembly_manifest",
    title="Assembly manifest",
    content=b"assembly manifest",
    r2_key="video-digest/assembled/manifest.json",
    media_type="application/json",
)
SUBTITLE_FAILURE_FILE = artifact_file(
    artifact_id=f"{EDITION.edition_id}:subtitle-failure",
    artifact_kind="video_digest_subtitle_failure",
    title="Subtitle failure",
    content=b"subtitle failure",
    r2_key="video-digest/subtitles/failure.json",
    media_type="application/json",
)
SUBTITLE_FAILURE = FailedSubtitles(evidence_artifact_version_id=SUBTITLE_FAILURE_FILE.version_id)
PUBLICATION_ID = publication_id(
    edition_id_value=EDITION.edition_id,
    expected_video_key="video-digests/edition.mp4",
    video_digest=ASSEMBLED_FILE.content_digest,
    video_byte_size=len(ASSEMBLED_FILE.content),
    video_media_type=ASSEMBLED_FILE.media_type,
    subtitle=None,
    source_video_version_id=ASSEMBLED_FILE.version_id,
    source_subtitle_version_id=None,
)
PUBLICATION = PublicationIntent(
    publication_id=PUBLICATION_ID,
    edition_id=EDITION.edition_id,
    expected_video_key="video-digests/edition.mp4",
    video_digest=ASSEMBLED_FILE.content_digest,
    video_byte_size=len(ASSEMBLED_FILE.content),
    video_media_type=ASSEMBLED_FILE.media_type,
    source_video_version_id=ASSEMBLED_FILE.version_id,
)
UPLOAD_FILE = artifact_file(
    artifact_id=f"{PUBLICATION_ID}:upload",
    artifact_kind="video_digest_publication_upload",
    title="Publication upload evidence",
    content=b"uploaded",
    r2_key="video-digest/publication/upload.json",
    media_type="application/json",
)
VERIFICATION_FILE = artifact_file(
    artifact_id=f"{PUBLICATION_ID}:verification",
    artifact_kind="video_digest_publication_verification",
    title="Publication verification evidence",
    content=b"verified public object",
    r2_key="video-digest/publication/verification.json",
    media_type="application/json",
)
PUBLICATION_FAILURE_FILE = artifact_file(
    artifact_id=f"{PUBLICATION_ID}:failure",
    artifact_kind="video_digest_publication_failure",
    title="Publication failure evidence",
    content=b"publication failure",
    r2_key="video-digest/publication/failure.json",
    media_type="application/json",
)
SLOT_FAILURE_FILE = artifact_file(
    artifact_id=f"{SLOT.slot_id}:failure",
    artifact_kind="video_digest_failure",
    title="Slot failure evidence",
    content=b"slot failure",
    r2_key="video-digest/failures/slot.json",
    media_type="application/json",
)

_Rows = Mapping[str, object] | Sequence[Mapping[str, object]] | None


class _Cursor:
    def __init__(self, rows: _Rows) -> None:
        self._rows = [rows] if isinstance(rows, Mapping) else list(rows or ())

    def fetchone(self) -> Mapping[str, object] | None:
        return self._rows.pop(0) if self._rows else None

    def fetchall(self) -> list[Mapping[str, object]]:
        rows = self._rows
        self._rows = []
        return rows


class _ScriptedConnection:
    def __init__(
        self,
        steps: Sequence[tuple[str, _Rows]],
    ) -> None:
        self.steps = list(steps)
        self.statements: list[str] = []
        self.parameters: list[tuple[object, ...]] = []
        self.transaction_count = 0

    def execute(
        self,
        statement: str,
        parameters: Sequence[object] | None = None,
    ) -> _Cursor:
        normalized = " ".join(statement.split())
        if "SELECT clock_timestamp() AS database_now" in normalized:
            self.statements.append(normalized)
            self.parameters.append(tuple(parameters or ()))
            return _Cursor({"database_now": NOW})
        assert self.steps, f"Unexpected statement: {statement}"
        expected, row = self.steps.pop(0)
        assert expected in normalized
        self.statements.append(normalized)
        self.parameters.append(tuple(parameters or ()))
        return _Cursor(row)


def _use_connection(
    monkeypatch: pytest.MonkeyPatch,
    steps: Sequence[tuple[str, _Rows]],
) -> _ScriptedConnection:
    connection = _ScriptedConnection(steps)

    def transaction(
        operation: Callable[[CatalogConnection], _Result],
        *,
        retry_transient_errors: bool = False,
    ) -> _Result:
        assert retry_transient_errors is False
        connection.transaction_count += 1
        return operation(cast(CatalogConnection, cast(Any, connection)))

    monkeypatch.setattr(video_digest, "catalog_transaction", transaction)
    return connection


def _slot_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "slot_id": SLOT.slot_id,
        "name": SLOT.name.value,
        "scheduled_at": SLOT.scheduled_at,
        "bucharest_day": SLOT.bucharest_day,
        "stage": "scheduled",
        "edition_id": None,
        "lease_owner_token": None,
        "lease_expires_at": None,
        "claim_count": 0,
        "skip_reason": None,
        "terminal_lease_owner_token": None,
        "terminal_lease_expires_at": None,
        "terminal_claim_count": None,
    }
    row.update(updates)
    return row


def _active_row(**updates: object) -> dict[str, object]:
    row = _slot_row(
        stage="claimed",
        edition_id=EDITION.edition_id,
        lease_owner_token="owner-a",
        lease_expires_at=NOW + timedelta(hours=1),
        claim_count=1,
    )
    row.update(updates)
    return row


def _edition_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "edition_id": EDITION.edition_id,
        "daily_report_version_id": EDITION.daily_report_version_id,
        "policy_bundle_version_id": EDITION.policy_bundle_version_id,
    }
    row.update(updates)
    return row


def _planning_attempt_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "edition_id": EDITION.edition_id,
        "attempt_index": 0,
        "disposition": "accepted",
        "attempt_evidence_artifact_version_id": PLANNING_ATTEMPT_FILE.version_id,
        "accepted_plan_artifact_version_id": PLAN_FILE.version_id,
    }
    row.update(updates)
    return row


def _lease(**updates: object) -> SlotLease:
    values: dict[str, object] = {
        "slot_id": SLOT.slot_id,
        "edition_id": EDITION.edition_id,
        "owner_token": "owner-a",
        "expires_at": NOW + timedelta(hours=1),
        "claim_count": 1,
    }
    values.update(updates)
    return SlotLease.model_validate(values)


def _scheduled_claim_steps(
    *,
    publication: Mapping[str, object] | None = None,
    active: Mapping[str, object] | None = None,
    include_update: bool = True,
) -> list[tuple[str, _Rows]]:
    steps: list[tuple[str, _Rows]] = [
        ("FROM video_digest_slots", _slot_row()),
        ("INSERT INTO video_digest_editions", None),
        ("FROM video_digest_editions", _edition_row()),
        ("FROM video_digest_publication_intents", publication),
    ]
    if publication is None:
        steps.append(("FROM video_digest_slots", active))
    if include_update:
        steps.append(("UPDATE video_digest_slots", None))
    return steps


def _story_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        **STORY.model_dump(),
        "stage": "planned",
        "verification_evidence_artifact_version_id": None,
        "accepted_clip_artifact_version_id": None,
        "validation_evidence_artifact_version_id": None,
        "failure_evidence_artifact_version_id": None,
    }
    row.update(updates)
    return row


def _generation_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        **REQUEST.model_dump(),
        "stage": "pending",
        "provider_receipt_id": None,
        "response_artifact_version_id": None,
        "accepted_clip_artifact_version_id": None,
        "failure_evidence_artifact_version_id": None,
        "cost_kind": "pending",
        "cost_usd": None,
        "cost_unknown_reason": None,
    }
    row.update(updates)
    return row


def _edition_outputs(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "edition_id": EDITION.edition_id,
        "assembled_video_artifact_version_id": ASSEMBLED_FILE.version_id,
        "assembly_manifest_artifact_version_id": ASSEMBLY_MANIFEST_FILE.version_id,
        "subtitle_state": "failed",
        "subtitle_artifact_version_id": None,
        "subtitle_failure_evidence_artifact_version_id": SUBTITLE_FAILURE_FILE.version_id,
    }
    row.update(updates)
    return row


def _publication_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "publication_id": PUBLICATION.publication_id,
        "edition_id": PUBLICATION.edition_id,
        "expected_video_key": PUBLICATION.expected_video_key,
        "video_digest": PUBLICATION.video_digest,
        "video_byte_size": PUBLICATION.video_byte_size,
        "video_media_type": PUBLICATION.video_media_type,
        "subtitle_expected_key": None,
        "subtitle_digest": None,
        "subtitle_byte_size": None,
        "subtitle_media_type": None,
        "source_video_artifact_version_id": PUBLICATION.source_video_version_id,
        "source_subtitle_artifact_version_id": None,
        "stage": "pending",
        "evidence_required": True,
        "upload_evidence_artifact_version_id": None,
        "verification_evidence_artifact_version_id": None,
        "failure_evidence_artifact_version_id": None,
        "published_at": None,
    }
    row.update(updates)
    return row


def _artifact_steps() -> list[tuple[str, _Rows]]:
    return [
        ("INSERT INTO artifacts", None),
        ("INSERT INTO artifact_versions", None),
        ("INSERT INTO artifact_files", None),
        ("UPDATE artifacts", None),
    ]


def _artifact_row(file: ArtifactFile = PLAN_FILE) -> dict[str, object]:
    return {
        "artifact_id": file.artifact_id,
        "artifact_kind": file.artifact_kind,
        "title": file.title,
        "version_id": file.version_id,
        "content_digest": file.content_digest,
        "r2_key": file.r2_key,
        "media_type": file.media_type,
        "byte_size": len(file.content),
    }


def _artifact_metadata(file: ArtifactFile) -> dict[str, object]:
    return {
        "content_digest": file.content_digest,
        "byte_size": len(file.content),
        "media_type": file.media_type,
    }


def test_schedule_slot_replays_exact_identity_and_returns_utc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offset_slot = SLOT.model_copy(
        update={"scheduled_at": SCHEDULED_AT.astimezone(timezone(timedelta(hours=3)))}
    )
    connection = _use_connection(
        monkeypatch,
        [("INSERT INTO video_digest_slots", None), ("FROM video_digest_slots", _slot_row())],
    )

    result = video_digest.schedule_slot(
        offset_slot,
        recorded_at=NOW.astimezone(timezone(timedelta(hours=3))),
    )

    assert result == SLOT
    assert result.scheduled_at.tzinfo is UTC
    assert connection.transaction_count == 1
    assert connection.steps == []


def test_schedule_slot_rejects_conflicting_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("INSERT INTO video_digest_slots", None),
            ("FROM video_digest_slots", _slot_row(name="midday")),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="identity conflicts"):
        video_digest.schedule_slot(SLOT, recorded_at=NOW)

    assert connection.transaction_count == 1


def test_schedule_slot_rejects_natural_identity_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _use_connection(
        monkeypatch,
        [("INSERT INTO video_digest_slots", None), ("FROM video_digest_slots", None)],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="unavailable"):
        video_digest.schedule_slot(SLOT, recorded_at=NOW)

    assert connection.transaction_count == 1


def test_skip_slot_records_and_replays_exact_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _use_connection(
        monkeypatch,
        [("FROM video_digest_slots", _slot_row()), ("UPDATE video_digest_slots", None)],
    )
    reason = SlotSkipReason.SOURCE_EMPTY

    assert video_digest.skip_slot(SLOT.slot_id, reason, recorded_at=NOW) == SkippedSlot(
        reason=reason
    )
    assert first.transaction_count == 1
    assert first.parameters[-1][-1] == SLOT.slot_id

    replay = _use_connection(
        monkeypatch,
        [("FROM video_digest_slots", _slot_row(stage="skipped", skip_reason=reason.value))],
    )
    assert video_digest.skip_slot(SLOT.slot_id, reason, recorded_at=NOW) == SkippedSlot(
        reason=reason
    )
    assert replay.transaction_count == 1


def test_skip_slot_rejects_conflicting_state(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            (
                "FROM video_digest_slots",
                _slot_row(stage="skipped", skip_reason=SlotSkipReason.UNCHANGED.value),
            )
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="skip request"):
        video_digest.skip_slot(SLOT.slot_id, SlotSkipReason.SOURCE_EMPTY, recorded_at=NOW)

    assert connection.transaction_count == 1


def test_claim_slot_returns_persisted_skip_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    reason = SlotSkipReason.UNCHANGED
    connection = _use_connection(
        monkeypatch,
        [("FROM video_digest_slots", _slot_row(stage="skipped", skip_reason=reason.value))],
    )

    result = video_digest.claim_slot(
        SLOT.slot_id,
        EDITION,
        owner_token="owner-a",
        now=NOW,
        lease_duration=timedelta(minutes=10),
    )

    assert result == SkippedSlot(reason=reason)
    assert connection.transaction_count == 1


@pytest.mark.parametrize("stage", ["failed", "published"])
def test_claim_slot_returns_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [("FROM video_digest_slots", _slot_row(stage=stage))],
    )

    result = video_digest.claim_slot(
        SLOT.slot_id,
        EDITION,
        owner_token="owner-a",
        now=NOW,
        lease_duration=timedelta(minutes=10),
    )

    assert result == TerminalSlot(state=TerminalSlotState(stage))
    assert connection.transaction_count == 1


def test_claim_slot_replays_an_unexpired_same_owner_lease_in_utc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_expiry = (NOW + timedelta(hours=1)).astimezone(timezone(timedelta(hours=3)))
    connection = _use_connection(
        monkeypatch,
        [("FROM video_digest_slots", _active_row(lease_expires_at=stored_expiry))],
    )

    result = video_digest.claim_slot(
        SLOT.slot_id,
        EDITION,
        owner_token=" owner-a ",
        now=NOW,
        lease_duration=timedelta(minutes=10),
    )

    assert result == ClaimedSlot(lease=_lease())
    assert isinstance(result, ClaimedSlot)
    assert result.lease.expires_at.tzinfo is UTC
    assert connection.transaction_count == 1


def test_claim_slot_skips_an_unexpired_other_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _use_connection(
        monkeypatch,
        [("FROM video_digest_slots", _active_row())],
    )

    result = video_digest.claim_slot(
        SLOT.slot_id,
        EDITION,
        owner_token="owner-b",
        now=NOW,
        lease_duration=timedelta(minutes=10),
    )

    assert result == SkippedSlot(reason=SlotSkipReason.OVERLAPPING_RUN)
    assert connection.transaction_count == 1


def test_claim_slot_rejects_an_active_different_edition_before_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = EditionIdentity(
        edition_id=edition_id("3" * 64, POLICY_ID),
        daily_report_version_id="3" * 64,
        policy_bundle_version_id=POLICY_ID,
    )
    connection = _use_connection(
        monkeypatch,
        [("FROM video_digest_slots", _active_row())],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="another edition"):
        video_digest.claim_slot(
            SLOT.slot_id,
            other,
            owner_token="owner-b",
            now=NOW,
            lease_duration=timedelta(minutes=10),
        )

    assert connection.transaction_count == 1


def test_claim_slot_rejects_same_owner_expired_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [("FROM video_digest_slots", _active_row(lease_expires_at=NOW))],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="fresh owner"):
        video_digest.claim_slot(
            SLOT.slot_id,
            EDITION,
            owner_token="owner-a",
            now=NOW,
            lease_duration=timedelta(minutes=10),
        )

    assert connection.transaction_count == 1


def test_claim_slot_recovers_expired_lease_with_incremented_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(lease_expires_at=NOW, claim_count=3)),
            ("UPDATE video_digest_slots", None),
        ],
    )

    result = video_digest.claim_slot(
        SLOT.slot_id,
        EDITION,
        owner_token="owner-b",
        now=NOW,
        lease_duration=timedelta(minutes=10),
    )

    assert result == ClaimedSlot(
        lease=_lease(
            owner_token="owner-b",
            expires_at=NOW + timedelta(minutes=10),
            claim_count=4,
        )
    )
    assert connection.parameters[-1] == (
        "owner-b",
        NOW + timedelta(minutes=10),
        4,
        NOW,
        SLOT.slot_id,
    )
    assert connection.transaction_count == 1


def test_claim_slot_claims_a_scheduled_slot_after_locking_slot_then_edition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(monkeypatch, _scheduled_claim_steps())

    result = video_digest.claim_slot(
        SLOT.slot_id,
        EDITION,
        owner_token="owner-a",
        now=NOW,
        lease_duration=timedelta(minutes=10),
    )

    assert result == ClaimedSlot(lease=_lease(expires_at=NOW + timedelta(minutes=10)))
    slot_lock = next(
        index
        for index, statement in enumerate(connection.statements)
        if "FROM video_digest_slots" in statement and "FOR UPDATE" in statement
    )
    edition_lock = next(
        index
        for index, statement in enumerate(connection.statements)
        if "FROM video_digest_editions" in statement and "FOR UPDATE" in statement
    )
    assert slot_lock < edition_lock
    assert connection.parameters[-1][-1] == SLOT.slot_id
    assert connection.transaction_count == 1


@pytest.mark.parametrize(
    ("steps", "reason"),
    [
        (
            _scheduled_claim_steps(
                publication={"publication_id": "publication"},
            ),
            SlotSkipReason.ALREADY_PUBLISHED,
        ),
        (
            _scheduled_claim_steps(active={"slot_id": "other-slot"}),
            SlotSkipReason.ACTIVE_EDITION,
        ),
    ],
)
def test_claim_slot_skips_published_or_active_edition(
    monkeypatch: pytest.MonkeyPatch,
    steps: Sequence[tuple[str, Mapping[str, object] | None]],
    reason: SlotSkipReason,
) -> None:
    connection = _use_connection(monkeypatch, steps)

    result = video_digest.claim_slot(
        SLOT.slot_id,
        EDITION,
        owner_token="owner-a",
        now=NOW,
        lease_duration=timedelta(minutes=10),
    )

    assert result == SkippedSlot(reason=reason)
    assert connection.parameters[-1] == (EDITION.edition_id, reason.value, NOW, SLOT.slot_id)
    assert connection.transaction_count == 1


def test_claim_slot_rejects_conflicting_edition_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    steps = [
        ("FROM video_digest_slots", _slot_row()),
        ("INSERT INTO video_digest_editions", None),
        ("FROM video_digest_editions", _edition_row(policy_bundle_version_id="3" * 64)),
    ]
    connection = _use_connection(monkeypatch, steps)

    with pytest.raises(VideoDigestCheckpointConflictError, match="edition identity"):
        video_digest.claim_slot(
            SLOT.slot_id,
            EDITION,
            owner_token="owner-a",
            now=NOW,
            lease_duration=timedelta(minutes=10),
        )

    assert connection.transaction_count == 1


def test_renew_slot_extends_an_exact_unexpired_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    lease = _lease()
    connection = _use_connection(
        monkeypatch,
        [("FROM video_digest_slots", _active_row()), ("UPDATE video_digest_slots", None)],
    )

    result = video_digest.renew_slot(
        lease,
        now=NOW,
        lease_duration=timedelta(hours=2),
    )

    assert result == lease.model_copy(update={"expires_at": NOW + timedelta(hours=2)})
    assert result.expires_at.tzinfo is UTC
    assert connection.parameters[-1] == (NOW + timedelta(hours=2), NOW, SLOT.slot_id)
    assert connection.transaction_count == 1
    slot_lock = next(
        index
        for index, statement in enumerate(connection.statements)
        if "FROM video_digest_slots" in statement
    )
    wall_clock = next(
        index
        for index, statement in enumerate(connection.statements)
        if "clock_timestamp()" in statement
    )
    assert slot_lock < wall_clock


@pytest.mark.parametrize(
    "row",
    [
        _active_row(slot_id=SlotId("f" * 64)),
        _active_row(edition_id="3" * 64),
        _active_row(lease_owner_token="owner-b"),
        _active_row(claim_count=2),
        _active_row(lease_expires_at=NOW + timedelta(hours=2)),
        _active_row(lease_expires_at=NOW),
    ],
    ids=["slot", "edition", "owner", "count", "stored-expiry", "expired"],
)
def test_renew_slot_rejects_stale_or_expired_fence(
    monkeypatch: pytest.MonkeyPatch,
    row: Mapping[str, object],
) -> None:
    connection = _use_connection(monkeypatch, [("FROM video_digest_slots", row)])

    with pytest.raises(VideoDigestLeaseLostError, match="lease was lost"):
        video_digest.renew_slot(
            _lease(),
            now=NOW,
            lease_duration=timedelta(minutes=10),
        )

    assert connection.transaction_count == 1


def test_rejected_planning_attempt_records_only_immutable_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row()),
            ("FROM video_digest_planning_attempts", None),
            *_artifact_steps(),
            ("INSERT INTO video_digest_planning_attempts", None),
        ],
    )

    assert (
        video_digest.checkpoint_planning_attempt(
            _lease(),
            0,
            "rejected",
            evidence_file=PLANNING_ATTEMPT_FILE,
            recorded_at=NOW,
        )
        is None
    )
    assert not any("video_digest_stories" in statement for statement in connection.statements)
    assert connection.steps == []


def test_read_planning_attempts_returns_ordered_immutable_artifact_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "attempt_index": 0,
            "disposition": "rejected",
            "artifact_id": PLANNING_ATTEMPT_FILE.artifact_id,
            "version_id": PLANNING_ATTEMPT_FILE.version_id,
            "content_digest": PLANNING_ATTEMPT_FILE.content_digest,
            "r2_key": PLANNING_ATTEMPT_FILE.r2_key,
            "accepted_plan_artifact_version_id": None,
        },
        {
            "attempt_index": 1,
            "disposition": "accepted",
            "artifact_id": f"{EDITION.edition_id}:1:planning-attempt",
            "version_id": "8" * 64,
            "content_digest": "9" * 64,
            "r2_key": "video-digest/planning/attempt-1.json",
            "accepted_plan_artifact_version_id": PLAN_FILE.version_id,
        },
    ]
    monkeypatch.setattr(video_digest, "catalog_query", lambda _query, _values: rows)

    attempts = video_digest.read_planning_attempts(EDITION.edition_id)

    assert tuple(item.attempt_index for item in attempts) == (0, 1)
    assert attempts[0].evidence.version_id == PLANNING_ATTEMPT_FILE.version_id
    assert attempts[1].accepted_plan_artifact_version_id == PLAN_FILE.version_id


def test_record_policy_bundle_validates_and_registers_exact_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_file = artifact_file(
        artifact_id="video-digest-policy:production-v1",
        artifact_kind="video_digest_policy",
        title="Video digest policy",
        content=b"{}",
        r2_key="video-digest/policies/production-v1.json",
        media_type="application/json",
    )
    connection = _use_connection(
        monkeypatch,
        [*_artifact_steps(), ("FROM artifacts AS artifact", _artifact_row(policy_file))],
    )

    assert video_digest.record_policy_bundle(policy_file, recorded_at=NOW) == policy_file.version_id
    assert connection.steps == []


def test_accepted_planning_attempt_registers_the_canonical_plan_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row()),
            ("FROM video_digest_planning_attempts", None),
            *_artifact_steps(),
            *_artifact_steps(),
            ("INSERT INTO video_digest_planning_attempts", None),
            ("UPDATE video_digest_slots", {"slot_id": SLOT.slot_id}),
            ("INSERT INTO video_digest_stories", None),
            ("UPDATE video_digest_editions", {"edition_id": EDITION.edition_id}),
            ("UPDATE video_digest_slots", {"slot_id": SLOT.slot_id}),
        ],
    )

    assert (
        video_digest.checkpoint_planning_attempt(
            _lease(),
            0,
            "accepted",
            evidence_file=PLANNING_ATTEMPT_FILE,
            accepted_plan=PLAN,
            plan_file=PLAN_FILE,
            recorded_at=NOW,
        )
        == PLAN
    )
    writes = [
        statement
        for statement in connection.statements
        if statement.startswith(("INSERT", "UPDATE"))
    ]
    attempt_write = next(
        index
        for index, statement in enumerate(writes)
        if "video_digest_planning_attempts" in statement
    )
    story_write = next(
        index for index, statement in enumerate(writes) if "video_digest_stories" in statement
    )
    assert attempt_write < story_write
    assert connection.steps == []


def test_planning_attempt_replays_only_exact_evidence_and_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            ("FROM video_digest_planning_attempts", _planning_attempt_row()),
            ("FROM artifacts AS artifact", _artifact_row(PLANNING_ATTEMPT_FILE)),
            ("FROM video_digest_editions", {"plan_artifact_version_id": PLAN_FILE.version_id}),
            ("FROM video_digest_stories", [_story_row()]),
            ("FROM artifacts AS artifact", _artifact_row(PLAN_FILE)),
        ],
    )

    assert (
        video_digest.checkpoint_planning_attempt(
            _lease(),
            0,
            "accepted",
            evidence_file=PLANNING_ATTEMPT_FILE,
            accepted_plan=PLAN,
            plan_file=PLAN_FILE,
            recorded_at=NOW,
        )
        == PLAN
    )
    assert all(
        not statement.startswith(("INSERT", "UPDATE")) for statement in connection.statements
    )


def test_planning_attempt_rejects_conflicting_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row()),
            (
                "FROM video_digest_planning_attempts",
                _planning_attempt_row(
                    disposition="rejected", accepted_plan_artifact_version_id=None
                ),
            ),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="planning attempt conflicts"):
        video_digest.checkpoint_planning_attempt(
            _lease(),
            0,
            "accepted",
            evidence_file=PLANNING_ATTEMPT_FILE,
            accepted_plan=PLAN,
            plan_file=PLAN_FILE,
            recorded_at=NOW,
        )
    assert connection.steps == []


def test_planning_attempt_validates_disposition_shape_before_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        video_digest,
        "catalog_transaction",
        lambda *_args, **_kwargs: pytest.fail("validation opened a transaction"),
    )

    with pytest.raises(ValueError, match="index"):
        video_digest.checkpoint_planning_attempt(
            _lease(), 3, "rejected", evidence_file=PLANNING_ATTEMPT_FILE, recorded_at=NOW
        )
    with pytest.raises(ValueError, match="requires its canonical plan"):
        video_digest.checkpoint_planning_attempt(
            _lease(), 0, "accepted", evidence_file=PLANNING_ATTEMPT_FILE, recorded_at=NOW
        )
    with pytest.raises(ValueError, match="cannot register"):
        video_digest.checkpoint_planning_attempt(
            _lease(),
            0,
            "rejected",
            evidence_file=PLANNING_ATTEMPT_FILE,
            accepted_plan=PLAN,
            plan_file=PLAN_FILE,
            recorded_at=NOW,
        )


def test_checkpoint_plan_validates_boundary_identity_before_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_edition = EditionIdentity(
        edition_id=edition_id("4" * 64, POLICY_ID),
        daily_report_version_id="4" * 64,
        policy_bundle_version_id=POLICY_ID,
    )
    other_story = PlannedStory(
        story_id=planned_story_id(other_edition.edition_id, 0, SUBJECT_ID),
        edition_id=other_edition.edition_id,
        position=0,
        report_subject_id=SUBJECT_ID,
        title="Other story",
        requested_duration_ms=15_000,
    )
    other_plan = PLAN.model_copy(
        update={"edition_id": other_edition.edition_id, "stories": (other_story,)}
    )
    mismatched_file = PLAN_FILE.model_copy(update={"version_id": "5" * 64})
    invalid_identity = PLAN_FILE.model_copy(update={"artifact_id": "other-edition"})
    monkeypatch.setattr(
        video_digest,
        "catalog_transaction",
        lambda *_args, **_kwargs: pytest.fail("validation opened a transaction"),
    )

    with pytest.raises(ValueError, match="plan edition"):
        video_digest.checkpoint_plan(_lease(), other_plan, plan_file=PLAN_FILE, recorded_at=NOW)
    with pytest.raises(ValueError, match="plan artifact"):
        video_digest.checkpoint_plan(_lease(), PLAN, plan_file=mismatched_file, recorded_at=NOW)
    with pytest.raises(ValueError, match="artifact identity"):
        video_digest.checkpoint_plan(_lease(), PLAN, plan_file=invalid_identity, recorded_at=NOW)
    with pytest.raises(ValueError, match="recorded_at must include"):
        video_digest.checkpoint_plan(
            _lease(), PLAN, plan_file=PLAN_FILE, recorded_at=NOW.replace(tzinfo=None)
        )


def test_checkpoint_plan_writes_one_atomic_ordered_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row()),
            *_artifact_steps(),
            ("UPDATE video_digest_slots", {"slot_id": SLOT.slot_id}),
            ("INSERT INTO video_digest_stories", None),
            ("UPDATE video_digest_editions", {"edition_id": EDITION.edition_id}),
            ("UPDATE video_digest_slots", {"slot_id": SLOT.slot_id}),
        ],
    )

    assert (
        video_digest.checkpoint_plan(
            _lease(), PLAN, plan_file=PLAN_FILE, recorded_at=NOW - timedelta(days=1)
        )
        == PLAN
    )

    writes = [
        statement for statement in connection.statements if not statement.startswith("SELECT")
    ]
    assert [statement.split()[0:3] for statement in writes] == [
        ["INSERT", "INTO", "artifacts"],
        ["INSERT", "INTO", "artifact_versions"],
        ["INSERT", "INTO", "artifact_files"],
        ["UPDATE", "artifacts", "SET"],
        ["UPDATE", "video_digest_slots", "SET"],
        ["INSERT", "INTO", "video_digest_stories"],
        ["UPDATE", "video_digest_editions", "SET"],
        ["UPDATE", "video_digest_slots", "SET"],
    ]
    assert connection.parameters[2][-1] == NOW.isoformat()
    assert any("owner-a" in parameters for parameters in connection.parameters)
    assert connection.transaction_count == 1
    assert connection.steps == []


def test_checkpoint_plan_replays_only_an_exact_stored_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="assembling")),
            (
                "FROM video_digest_editions",
                {"plan_artifact_version_id": PLAN.artifact_version_id},
            ),
            ("FROM video_digest_stories", [_story_row(stage="generating")]),
            ("FROM artifacts AS artifact", _artifact_row()),
        ],
    )

    assert (
        video_digest.checkpoint_plan(_lease(), PLAN, plan_file=PLAN_FILE, recorded_at=NOW) == PLAN
    )
    story_query = next(
        statement for statement in connection.statements if "ORDER BY position" in statement
    )
    assert "ORDER BY position" in story_query
    assert all(
        not statement.startswith(("INSERT", "UPDATE")) for statement in connection.statements
    )


def test_checkpoint_plan_rejects_stored_plan_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="assembling")),
            (
                "FROM video_digest_editions",
                {"plan_artifact_version_id": PLAN.artifact_version_id},
            ),
            ("FROM video_digest_stories", [_story_row(title="Changed title")]),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="plan request"):
        video_digest.checkpoint_plan(_lease(), PLAN, plan_file=PLAN_FILE, recorded_at=NOW)

    assert connection.transaction_count == 1


def test_checkpoint_plan_rejects_invalid_slot_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _use_connection(
        monkeypatch,
        [("FROM video_digest_slots", _active_row(stage="planning"))],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="slot stage"):
        video_digest.checkpoint_plan(_lease(), PLAN, plan_file=PLAN_FILE, recorded_at=NOW)

    assert connection.transaction_count == 1


def test_checkpoint_plan_propagates_stale_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _use_connection(
        monkeypatch,
        [("FROM video_digest_slots", _active_row(lease_owner_token="owner-b"))],
    )

    with pytest.raises(VideoDigestLeaseLostError, match="lease was lost"):
        video_digest.checkpoint_plan(_lease(), PLAN, plan_file=PLAN_FILE, recorded_at=NOW)

    assert connection.transaction_count == 1


def test_checkpoint_story_verification_validates_evidence_identity_before_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_evidence = EVIDENCE_FILE.model_copy(update={"artifact_id": "other-story"})
    wrong_kind = EVIDENCE_FILE.model_copy(update={"artifact_kind": "other_kind"})
    monkeypatch.setattr(
        video_digest,
        "catalog_transaction",
        lambda *_args, **_kwargs: pytest.fail("validation opened a transaction"),
    )

    with pytest.raises(ValueError, match="evidence artifact"):
        video_digest.checkpoint_story_verification(
            _lease(), STORY.story_id, evidence_file=other_evidence, recorded_at=NOW
        )
    with pytest.raises(ValueError, match="artifact kind"):
        video_digest.checkpoint_story_verification(
            _lease(), STORY.story_id, evidence_file=wrong_kind, recorded_at=NOW
        )
    with pytest.raises(ValueError, match="recorded_at must include"):
        video_digest.checkpoint_story_verification(
            _lease(),
            STORY.story_id,
            evidence_file=EVIDENCE_FILE,
            recorded_at=NOW.replace(tzinfo=None),
        )


def test_checkpoint_story_verification_records_each_transition_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            ("FROM video_digest_stories", _story_row()),
            *_artifact_steps(),
            ("UPDATE video_digest_stories", {"story_id": STORY.story_id}),
            ("UPDATE video_digest_stories", {"story_id": STORY.story_id}),
            ("UPDATE video_digest_stories", {"story_id": STORY.story_id}),
        ],
    )

    assert (
        video_digest.checkpoint_story_verification(
            _lease(), STORY.story_id, evidence_file=EVIDENCE_FILE, recorded_at=NOW
        )
        is None
    )

    story_updates = [
        statement
        for statement in connection.statements
        if statement.startswith("UPDATE video_digest_stories")
    ]
    assert "stage = 'verifying'" in story_updates[0]
    assert "stage = 'verified'" in story_updates[1]
    assert "stage = 'generating'" in story_updates[2]
    assert connection.parameters[3][-1] == NOW.isoformat()
    assert all(STORY.story_id in parameters for parameters in connection.parameters[-3:])
    assert connection.transaction_count == 1


def test_checkpoint_story_verification_replays_exact_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="assembling")),
            (
                "FROM video_digest_stories",
                _story_row(
                    stage="accepted",
                    verification_evidence_artifact_version_id=EVIDENCE_FILE.version_id,
                ),
            ),
            ("FROM artifacts AS artifact", _artifact_row(EVIDENCE_FILE)),
        ],
    )

    assert (
        video_digest.checkpoint_story_verification(
            _lease(), STORY.story_id, evidence_file=EVIDENCE_FILE, recorded_at=NOW
        )
        is None
    )
    assert all(
        not statement.startswith(("INSERT", "UPDATE")) for statement in connection.statements
    )


def test_checkpoint_story_verification_rejects_different_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            (
                "FROM video_digest_stories",
                _story_row(
                    stage="generating",
                    verification_evidence_artifact_version_id="6" * 64,
                ),
            ),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="verification request"):
        video_digest.checkpoint_story_verification(
            _lease(), STORY.story_id, evidence_file=EVIDENCE_FILE, recorded_at=NOW
        )

    assert connection.transaction_count == 1


def test_checkpoint_story_verification_rejects_story_identity_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            ("FROM video_digest_stories", _story_row(edition_id="7" * 64)),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="story identity"):
        video_digest.checkpoint_story_verification(
            _lease(), STORY.story_id, evidence_file=EVIDENCE_FILE, recorded_at=NOW
        )

    assert connection.transaction_count == 1


def test_edition_verification_records_and_replays_an_exact_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            (
                "FROM video_digest_editions",
                {
                    "plan_artifact_version_id": PLAN_FILE.version_id,
                    "verification_manifest_artifact_version_id": None,
                },
            ),
            ("FROM video_digest_stories", None),
            *_artifact_steps(),
            ("UPDATE video_digest_editions", {"edition_id": EDITION.edition_id}),
        ],
    )
    assert (
        video_digest.checkpoint_edition_verification(
            _lease(), manifest_file=MANIFEST_FILE, recorded_at=NOW
        )
        is None
    )
    assert first.steps == []

    replay = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="assembling")),
            (
                "FROM video_digest_editions",
                {
                    "plan_artifact_version_id": PLAN_FILE.version_id,
                    "verification_manifest_artifact_version_id": MANIFEST_FILE.version_id,
                },
            ),
            ("FROM artifacts AS artifact", _artifact_row(MANIFEST_FILE)),
        ],
    )
    assert (
        video_digest.checkpoint_edition_verification(
            _lease(), manifest_file=MANIFEST_FILE, recorded_at=NOW
        )
        is None
    )
    assert all(not statement.startswith(("INSERT", "UPDATE")) for statement in replay.statements)


def test_edition_verification_requires_every_mandatory_story(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            (
                "FROM video_digest_editions",
                {
                    "plan_artifact_version_id": PLAN_FILE.version_id,
                    "verification_manifest_artifact_version_id": None,
                },
            ),
            ("FROM video_digest_stories", {"story_id": STORY.story_id}),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="every mandatory story"):
        video_digest.checkpoint_edition_verification(
            _lease(), manifest_file=MANIFEST_FILE, recorded_at=NOW
        )
    assert connection.steps == []


def test_generation_request_requires_manifest_before_artifact_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            ("FROM video_digest_generation_requests", None),
            (
                "FROM video_digest_editions",
                {
                    "plan_artifact_version_id": PLAN_FILE.version_id,
                    "verification_manifest_artifact_version_id": None,
                },
            ),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="verification manifest"):
        video_digest.checkpoint_generation_request(
            _lease(),
            REQUEST,
            request_file=REQUEST_FILE,
            admission=GENERATION_ADMISSION,
            recorded_at=NOW,
        )
    assert not any("INSERT INTO artifacts" in statement for statement in connection.statements)
    assert connection.steps == []


def test_generation_request_requires_all_mandatory_stories_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            ("FROM video_digest_generation_requests", None),
            (
                "FROM video_digest_editions",
                {
                    "plan_artifact_version_id": PLAN_FILE.version_id,
                    "verification_manifest_artifact_version_id": MANIFEST_FILE.version_id,
                },
            ),
            ("FROM video_digest_stories", {"story_id": STORY.story_id}),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="every mandatory story"):
        video_digest.checkpoint_generation_request(
            _lease(),
            REQUEST,
            request_file=REQUEST_FILE,
            admission=GENERATION_ADMISSION,
            recorded_at=NOW,
        )
    assert connection.steps == []


def test_generation_request_checkpoint_starts_pending_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            ("FROM video_digest_generation_requests", None),
            (
                "FROM video_digest_editions",
                {
                    "plan_artifact_version_id": PLAN_FILE.version_id,
                    "verification_manifest_artifact_version_id": MANIFEST_FILE.version_id,
                },
            ),
            ("FROM video_digest_stories", None),
            ("FROM video_digest_stories", _story_row(stage="generating")),
            ("FROM video_digest_generation_requests", None),
            ("FROM video_digest_generation_requests", None),
            *_artifact_steps(),
            ("INSERT INTO video_digest_generation_requests", None),
            ("INSERT INTO video_digest_generation_reservations", None),
        ],
    )

    state = video_digest.checkpoint_generation_request(
        _lease(),
        REQUEST,
        request_file=REQUEST_FILE,
        admission=GENERATION_ADMISSION,
        recorded_at=NOW,
    )

    assert state.created is True
    assert state.state.request_id == REQUEST.request_id
    assert state.state.stage is GenerationStage.PENDING
    assert state.state.cost.kind == "pending"
    assert connection.steps == []


def test_generation_request_replay_requires_the_exact_budget_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservations = [
        {
            "scope_kind": scope,
            "limit_usd": limit,
            "reserved_usd": GENERATION_ADMISSION.reserved_usd,
            "generation_policy_artifact_version_id": POLICY_ID,
        }
        for scope, limit in (
            ("bucharest_day", Decimal("150")),
            ("calendar_month", Decimal("1000")),
            ("edition", Decimal("7")),
            ("story", Decimal("7")),
        )
    ]
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            ("FROM video_digest_generation_requests", _generation_row()),
            ("FROM artifacts AS artifact", _artifact_row(REQUEST_FILE)),
            ("FROM video_digest_generation_reservations", reservations),
        ],
    )

    checkpoint = video_digest.checkpoint_generation_request(
        _lease(),
        REQUEST,
        request_file=REQUEST_FILE,
        admission=GENERATION_ADMISSION,
        recorded_at=NOW,
    )

    assert checkpoint.created is False
    assert checkpoint.state.stage is GenerationStage.PENDING
    assert connection.steps == []


def test_read_generation_attempts_returns_generation_and_accepted_clip_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_file = artifact_file(
        artifact_id=f"{REQUEST.request_id}:response",
        artifact_kind="video_digest_generation_response",
        title="Generation response",
        content=b"response",
        r2_key="video-digest/generation/response.json",
        media_type="application/json",
    )
    monkeypatch.setattr(
        video_digest,
        "catalog_query",
        lambda _query, _values: [
            {
                **_generation_row(
                    stage="accepted",
                    provider_receipt_id="fal-receipt-1",
                    response_artifact_version_id=response_file.version_id,
                    accepted_clip_artifact_version_id=CLIP_FILE.version_id,
                    validation_evidence_artifact_version_id=VALIDATION_FILE.version_id,
                    cost_kind="measured",
                    cost_usd=Decimal("3.25"),
                ),
                "request_artifact_id": REQUEST_FILE.artifact_id,
                "request_content_digest": REQUEST_FILE.content_digest,
                "request_r2_key": REQUEST_FILE.r2_key,
                "receipt_artifact_version_id": RECEIPT_FILE.version_id,
                "receipt_content_digest": RECEIPT_FILE.content_digest,
                "receipt_r2_key": RECEIPT_FILE.r2_key,
                "response_artifact_id": response_file.artifact_id,
                "response_content_digest": response_file.content_digest,
                "response_r2_key": response_file.r2_key,
                "clip_artifact_id": CLIP_FILE.artifact_id,
                "clip_content_digest": CLIP_FILE.content_digest,
                "clip_r2_key": CLIP_FILE.r2_key,
                "clip_byte_size": len(CLIP_FILE.content),
                "validation_artifact_id": VALIDATION_FILE.artifact_id,
                "validation_content_digest": VALIDATION_FILE.content_digest,
                "validation_r2_key": VALIDATION_FILE.r2_key,
            }
        ],
    )

    attempts = video_digest.read_generation_attempts(EDITION.edition_id)

    assert len(attempts) == 1
    assert attempts[0].request == REQUEST
    assert attempts[0].receipt_evidence is not None
    assert attempts[0].receipt_evidence.version_id == RECEIPT_FILE.version_id
    assert attempts[0].response_evidence is not None
    assert attempts[0].response_evidence.version_id == response_file.version_id
    assert attempts[0].accepted_clip is not None
    assert attempts[0].accepted_clip.version_id == CLIP_FILE.version_id
    assert attempts[0].accepted_clip.byte_size == len(CLIP_FILE.content)
    assert attempts[0].validation_evidence is not None
    assert attempts[0].validation_evidence.version_id == VALIDATION_FILE.version_id


def test_second_generation_request_requires_failed_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second_file = artifact_file(
        artifact_id=f"{EDITION.edition_id}:0:1:generation-request",
        artifact_kind="video_digest_generation_request",
        title="Second generation request",
        content=b"second request",
        r2_key="video-digest/generation/request-1.json",
        media_type="application/json",
    )
    second_request = GenerationRequestIdentity(
        request_id=generation_request_id(EDITION.edition_id, 0, 1, second_file.version_id),
        edition_id=EDITION.edition_id,
        story_position=0,
        attempt_index=1,
        request_artifact_version_id=second_file.version_id,
    )
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            ("FROM video_digest_generation_requests", None),
            (
                "FROM video_digest_editions",
                {
                    "plan_artifact_version_id": PLAN_FILE.version_id,
                    "verification_manifest_artifact_version_id": MANIFEST_FILE.version_id,
                },
            ),
            ("FROM video_digest_stories", None),
            ("FROM video_digest_stories", _story_row(stage="generating")),
            ("FROM video_digest_generation_requests", None),
            ("FROM video_digest_generation_requests", {"stage": "submitted"}),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="failed first attempt"):
        video_digest.checkpoint_generation_request(
            _lease(),
            second_request,
            request_file=second_file,
            admission=GENERATION_ADMISSION,
            recorded_at=NOW,
        )
    assert connection.steps == []


def test_generation_request_rejects_parallel_paid_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            ("FROM video_digest_generation_requests", None),
            (
                "FROM video_digest_editions",
                {
                    "plan_artifact_version_id": PLAN_FILE.version_id,
                    "verification_manifest_artifact_version_id": MANIFEST_FILE.version_id,
                },
            ),
            ("FROM video_digest_stories", None),
            ("FROM video_digest_stories", _story_row(stage="generating")),
            ("FROM video_digest_generation_requests", None),
            ("FROM video_digest_generation_requests", {"request_id": "other"}),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="attempt is active"):
        video_digest.checkpoint_generation_request(
            _lease(),
            REQUEST,
            request_file=REQUEST_FILE,
            admission=GENERATION_ADMISSION,
            recorded_at=NOW,
        )
    assert connection.steps == []


def test_generation_submission_persists_receipt_before_returning_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            ("FROM video_digest_generation_requests", _generation_row()),
            *_artifact_steps(),
            ("UPDATE video_digest_generation_requests", {"request_id": REQUEST.request_id}),
        ],
    )

    state = video_digest.checkpoint_generation_submission(
        _lease(),
        REQUEST.request_id,
        provider_receipt_id="fal-receipt-1",
        receipt_file=RECEIPT_FILE,
        cost=EstimatedAttemptCost(usd=Decimal("1.2500")),
        recorded_at=NOW,
    )

    assert state.provider_receipt_id == "fal-receipt-1"
    assert state.cost == EstimatedAttemptCost(usd=Decimal("1.2500"))
    writes = [
        statement
        for statement in connection.statements
        if statement.startswith(("INSERT", "UPDATE"))
    ]
    assert "INSERT INTO artifact_files" in writes[2]
    assert "UPDATE video_digest_generation_requests" in writes[-1]


def test_generation_submission_reuses_stored_receipt_after_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="assembling")),
            (
                "FROM video_digest_generation_requests",
                _generation_row(
                    stage="processing",
                    provider_receipt_id="fal-receipt-1",
                    response_artifact_version_id=RESPONSE_FILE.version_id,
                    cost_kind="estimated",
                    cost_usd=Decimal("1.25"),
                ),
            ),
            ("FROM artifacts AS artifact", _artifact_row(RECEIPT_FILE)),
        ],
    )

    state = video_digest.checkpoint_generation_submission(
        _lease(),
        REQUEST.request_id,
        provider_receipt_id="fal-receipt-1",
        receipt_file=RECEIPT_FILE,
        cost=EstimatedAttemptCost(usd=Decimal("1.25")),
        recorded_at=NOW,
    )

    assert state.stage is GenerationStage.PROCESSING
    assert state.provider_receipt_id == "fal-receipt-1"
    assert all(
        not statement.startswith(("INSERT", "UPDATE")) for statement in connection.statements
    )


def test_generation_submission_rejects_a_changed_active_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            (
                "FROM video_digest_generation_requests",
                _generation_row(
                    stage="submitted",
                    provider_receipt_id="fal-receipt-1",
                    cost_kind="estimated",
                    cost_usd=Decimal("1.25"),
                ),
            ),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="receipt conflicts"):
        video_digest.checkpoint_generation_submission(
            _lease(),
            REQUEST.request_id,
            provider_receipt_id="fal-receipt-1",
            receipt_file=RECEIPT_FILE,
            cost=EstimatedAttemptCost(usd=Decimal("2.00")),
            recorded_at=NOW,
        )
    assert connection.steps == []


def test_generation_response_is_durable_before_media_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            (
                "FROM video_digest_generation_requests",
                _generation_row(
                    stage="submitted",
                    provider_receipt_id="fal-receipt-1",
                    cost_kind="estimated",
                    cost_usd=Decimal("1.25"),
                ),
            ),
            *_artifact_steps(),
            ("UPDATE video_digest_generation_requests", {"request_id": REQUEST.request_id}),
        ],
    )

    state = video_digest.checkpoint_generation_response(
        _lease(), REQUEST.request_id, response_file=RESPONSE_FILE, recorded_at=NOW
    )

    assert state.stage is GenerationStage.PROCESSING
    assert connection.steps == []


def test_generation_response_rejects_wrong_artifact_identity_before_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_response = artifact_file(
        artifact_id="other-response",
        artifact_kind="video_digest_generation_response",
        title="Wrong response",
        content=b"response",
        r2_key="video-digest/generation/wrong-response.json",
        media_type="application/json",
    )
    monkeypatch.setattr(
        video_digest,
        "catalog_transaction",
        lambda *_args, **_kwargs: pytest.fail("validation opened a transaction"),
    )

    with pytest.raises(ValueError, match="artifact kind"):
        video_digest.checkpoint_generation_response(
            _lease(), REQUEST.request_id, response_file=wrong_response, recorded_at=NOW
        )


def test_generation_acceptance_updates_request_before_story(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            (
                "FROM video_digest_generation_requests",
                _generation_row(
                    stage="processing",
                    provider_receipt_id="fal-receipt-1",
                    response_artifact_version_id=RESPONSE_FILE.version_id,
                    cost_kind="estimated",
                    cost_usd=Decimal("1.25"),
                ),
            ),
            ("FROM video_digest_stories", _story_row(stage="generating")),
            *_artifact_steps(),
            *_artifact_steps(),
            ("UPDATE video_digest_generation_requests", {"request_id": REQUEST.request_id}),
            ("UPDATE video_digest_stories", {"story_id": STORY.story_id}),
        ],
    )

    state = video_digest.checkpoint_generation_acceptance(
        _lease(),
        REQUEST.request_id,
        clip_file=CLIP_FILE,
        validation_file=VALIDATION_FILE,
        cost=MeasuredAttemptCost(usd=Decimal("1.10")),
        recorded_at=NOW,
    )

    writes = [statement for statement in connection.statements if statement.startswith("UPDATE")]
    assert "video_digest_generation_requests" in writes[-2]
    assert "video_digest_stories" in writes[-1]
    assert state.cost == MeasuredAttemptCost(usd=Decimal("1.10"))


def test_second_generation_failure_terminates_story_and_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second_request_id = generation_request_id(EDITION.edition_id, 0, 1, REQUEST_FILE.version_id)
    second_failure_file = artifact_file(
        artifact_id=f"{second_request_id}:failure",
        artifact_kind="video_digest_generation_failure",
        title="Second generation failure",
        content=b"failure",
        r2_key="video-digest/generation/failure-1.json",
        media_type="application/json",
    )
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            (
                "FROM video_digest_generation_requests",
                _generation_row(
                    request_id=second_request_id,
                    attempt_index=1,
                    stage="processing",
                    provider_receipt_id="fal-receipt-2",
                    response_artifact_version_id=RESPONSE_FILE.version_id,
                    cost_kind="estimated",
                    cost_usd=Decimal("1.25"),
                ),
            ),
            ("FROM video_digest_stories", _story_row(stage="generating")),
            *_artifact_steps(),
            ("UPDATE video_digest_generation_requests", {"request_id": second_request_id}),
            ("UPDATE video_digest_stories", {"story_id": STORY.story_id}),
            ("UPDATE video_digest_slots", {"slot_id": SLOT.slot_id}),
        ],
    )

    state = video_digest.checkpoint_generation_failure(
        _lease(),
        second_request_id,
        evidence_file=second_failure_file,
        cost=UnknownAttemptCost(reason="Provider omitted billing data"),
        recorded_at=NOW,
    )

    assert state.stage is GenerationStage.FAILED
    assert state.cost == UnknownAttemptCost(reason="Provider omitted billing data")
    assert "lease_owner_token = NULL" in connection.statements[-1]


def test_generation_failure_replay_does_not_bypass_a_stale_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [("FROM video_digest_slots", _active_row(lease_owner_token="owner-b"))],
    )

    with pytest.raises(VideoDigestLeaseLostError, match="lease was lost"):
        video_digest.checkpoint_generation_failure(
            _lease(),
            REQUEST.request_id,
            evidence_file=FAILURE_FILE,
            cost=MeasuredAttemptCost(usd=Decimal("1.10")),
            recorded_at=NOW,
        )
    assert connection.steps == []


def test_terminal_generation_failure_replays_only_its_final_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            (
                "FROM video_digest_slots",
                _slot_row(
                    stage="failed",
                    edition_id=EDITION.edition_id,
                    claim_count=1,
                    terminal_lease_owner_token="owner-a",
                    terminal_lease_expires_at=NOW + timedelta(hours=1),
                    terminal_claim_count=1,
                ),
            ),
            (
                "FROM video_digest_generation_requests",
                _generation_row(
                    stage="failed",
                    failure_evidence_artifact_version_id=FAILURE_FILE.version_id,
                    cost_kind="measured",
                    cost_usd=Decimal("1.10"),
                ),
            ),
            ("FROM artifacts AS artifact", _artifact_row(FAILURE_FILE)),
        ],
    )

    state = video_digest.checkpoint_generation_failure(
        _lease(),
        REQUEST.request_id,
        evidence_file=FAILURE_FILE,
        cost=MeasuredAttemptCost(usd=Decimal("1.10")),
        recorded_at=NOW,
    )

    assert state.stage is GenerationStage.FAILED
    assert connection.steps == []


def test_assembly_checkpoint_requires_every_story_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            ("FROM video_digest_stories", {"story_id": STORY.story_id}),
        ],
    )
    with pytest.raises(VideoDigestCheckpointConflictError, match="incomplete mandatory"):
        video_digest.checkpoint_assembly_ready(_lease(), recorded_at=NOW)
    assert blocked.steps == []

    ready = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            ("FROM video_digest_stories", None),
            ("UPDATE video_digest_slots", {"slot_id": SLOT.slot_id}),
        ],
    )
    video_digest.checkpoint_assembly_ready(_lease(), recorded_at=NOW)
    assert ready.steps == []


def test_generation_spend_preserves_decimals_and_incomplete_cost_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        video_digest,
        "catalog_query",
        lambda *_args, **_kwargs: [
            {
                "measured_usd": Decimal("1.10"),
                "estimated_usd": Decimal("2.2500"),
                "pending_requests": 1,
                "unknown_requests": 2,
            }
        ],
    )

    spend = video_digest.read_generation_spend(EDITION.edition_id)

    assert spend.measured_usd == Decimal("1.10")
    assert spend.estimated_usd == Decimal("2.2500")
    assert spend.pending_requests == 1
    assert spend.unknown_requests == 2


def test_assembled_video_checkpoint_records_output_before_advancing_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="assembling")),
            (
                "FROM video_digest_editions",
                _edition_outputs(
                    assembled_video_artifact_version_id=None,
                    subtitle_state="pending",
                    subtitle_failure_evidence_artifact_version_id=None,
                ),
            ),
            ("FROM video_digest_stories", None),
            *_artifact_steps(),
            *_artifact_steps(),
            ("UPDATE video_digest_editions", {"edition_id": EDITION.edition_id}),
            ("UPDATE video_digest_slots", {"slot_id": SLOT.slot_id}),
        ],
    )

    assembled = video_digest.checkpoint_assembled_video(
        _lease(),
        video_file=ASSEMBLED_FILE,
        manifest_file=ASSEMBLY_MANIFEST_FILE,
        recorded_at=NOW,
    )

    assert assembled.artifact_version_id == ASSEMBLED_FILE.version_id
    writes = [statement for statement in connection.statements if statement.startswith("UPDATE")]
    assert "video_digest_editions" in writes[-2]
    assert "video_digest_slots" in writes[-1]


def test_assembled_video_checkpoint_requires_accepted_stories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="assembling")),
            (
                "FROM video_digest_editions",
                _edition_outputs(
                    assembled_video_artifact_version_id=None,
                    subtitle_state="pending",
                    subtitle_failure_evidence_artifact_version_id=None,
                ),
            ),
            ("FROM video_digest_stories", {"story_id": STORY.story_id}),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="incomplete mandatory stories"):
        video_digest.checkpoint_assembled_video(
            _lease(),
            video_file=ASSEMBLED_FILE,
            manifest_file=ASSEMBLY_MANIFEST_FILE,
            recorded_at=NOW,
        )

    assert connection.steps == []
    assert all(
        not statement.startswith(("INSERT", "UPDATE")) for statement in connection.statements
    )


def test_assembled_video_checkpoint_replays_after_slot_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="publishing")),
            ("FROM video_digest_editions", _edition_outputs()),
            ("FROM artifacts AS artifact", _artifact_row(ASSEMBLED_FILE)),
            ("FROM artifacts AS artifact", _artifact_row(ASSEMBLY_MANIFEST_FILE)),
        ],
    )

    result = video_digest.checkpoint_assembled_video(
        _lease(),
        video_file=ASSEMBLED_FILE,
        manifest_file=ASSEMBLY_MANIFEST_FILE,
        recorded_at=NOW,
    )

    assert result.artifact_version_id == ASSEMBLED_FILE.version_id
    assert all(
        not statement.startswith(("INSERT", "UPDATE")) for statement in connection.statements
    )


def test_subtitle_failure_is_publishable_and_replay_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="subtitling")),
            (
                "FROM video_digest_editions",
                _edition_outputs(
                    subtitle_state="pending",
                    subtitle_failure_evidence_artifact_version_id=None,
                ),
            ),
            *_artifact_steps(),
            ("UPDATE video_digest_editions", {"edition_id": EDITION.edition_id}),
            ("UPDATE video_digest_slots", {"slot_id": SLOT.slot_id}),
        ],
    )

    result = video_digest.checkpoint_subtitles(
        _lease(),
        SUBTITLE_FAILURE,
        artifact_file=SUBTITLE_FAILURE_FILE,
        recorded_at=NOW,
    )

    assert result == SUBTITLE_FAILURE
    assert "stage = %s" in connection.statements[-1]
    assert "publishing" in connection.parameters[-1]


def test_publication_intent_is_inserted_once_and_compared_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="publishing")),
            ("FROM video_digest_editions", _edition_outputs()),
            ("FROM artifact_versions AS version", _artifact_metadata(ASSEMBLED_FILE)),
            ("INSERT INTO video_digest_publication_intents", None),
            ("FROM video_digest_publication_intents", _publication_row()),
        ],
    )

    status = video_digest.record_publication_intent(_lease(), PUBLICATION, recorded_at=NOW)

    assert status.stage is PublicationState.PENDING
    assert status.publication_id == PUBLICATION_ID
    assert connection.steps == []


def test_publication_intent_rejects_wrong_edition_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="publishing")),
            (
                "FROM video_digest_editions",
                _edition_outputs(assembled_video_artifact_version_id="9" * 64),
            ),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="sources conflict"):
        video_digest.record_publication_intent(_lease(), PUBLICATION, recorded_at=NOW)
    assert connection.steps == []


def test_publication_progress_records_ordered_durable_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploading_connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="publishing")),
            ("FROM video_digest_publication_intents", _publication_row()),
            (
                "UPDATE video_digest_publication_intents",
                {"publication_id": PUBLICATION_ID},
            ),
        ],
    )
    uploading = video_digest.checkpoint_publication_progress(
        _lease(), PUBLICATION_ID, UploadingPublication(), recorded_at=NOW
    )
    assert uploading.stage is PublicationState.UPLOADING
    assert uploading_connection.steps == []

    uploaded_connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="publishing")),
            (
                "FROM video_digest_publication_intents",
                _publication_row(stage="uploading"),
            ),
            *_artifact_steps(),
            (
                "UPDATE video_digest_publication_intents",
                {"publication_id": PUBLICATION_ID},
            ),
        ],
    )
    uploaded = video_digest.checkpoint_publication_progress(
        _lease(),
        PUBLICATION_ID,
        UploadedPublication(evidence_artifact_version_id=UPLOAD_FILE.version_id),
        evidence_file=UPLOAD_FILE,
        recorded_at=NOW,
    )
    assert uploaded.stage is PublicationState.UPLOADED
    assert uploaded_connection.steps == []

    verified_object = VerifiedPublicObject(
        content_digest=PUBLICATION.video_digest,
        byte_size=PUBLICATION.video_byte_size,
        media_type=PUBLICATION.video_media_type,
        source_artifact_version_id=PUBLICATION.source_video_version_id,
    )
    verified_connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="publishing")),
            (
                "FROM video_digest_publication_intents",
                _publication_row(
                    stage="uploaded",
                    upload_evidence_artifact_version_id=UPLOAD_FILE.version_id,
                ),
            ),
            *_artifact_steps(),
            (
                "UPDATE video_digest_publication_intents",
                {"publication_id": PUBLICATION_ID},
            ),
        ],
    )
    verified = video_digest.checkpoint_publication_progress(
        _lease(),
        PUBLICATION_ID,
        VerifiedPublication(
            evidence_artifact_version_id=VERIFICATION_FILE.version_id,
            video=verified_object,
        ),
        evidence_file=VERIFICATION_FILE,
        recorded_at=NOW,
    )
    assert verified.stage is PublicationState.VERIFIED
    assert verified_connection.steps == []


def test_publication_progress_rejects_skipped_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="publishing")),
            ("FROM video_digest_publication_intents", _publication_row()),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="skip"):
        video_digest.checkpoint_publication_progress(
            _lease(),
            PUBLICATION_ID,
            UploadedPublication(evidence_artifact_version_id=UPLOAD_FILE.version_id),
            evidence_file=UPLOAD_FILE,
            recorded_at=NOW,
        )
    assert connection.steps == []


def test_publication_progress_rejects_mismatched_replay_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="publishing")),
            (
                "FROM video_digest_publication_intents",
                _publication_row(
                    stage="verified",
                    verification_evidence_artifact_version_id="0" * 64,
                ),
            ),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="evidence conflicts"):
        video_digest.checkpoint_publication_progress(
            _lease(),
            PUBLICATION_ID,
            VerifiedPublication(
                evidence_artifact_version_id=VERIFICATION_FILE.version_id,
                video=VerifiedPublicObject(
                    content_digest=PUBLICATION.video_digest,
                    byte_size=PUBLICATION.video_byte_size,
                    media_type=PUBLICATION.video_media_type,
                    source_artifact_version_id=PUBLICATION.source_video_version_id,
                ),
            ),
            evidence_file=VERIFICATION_FILE,
            recorded_at=NOW,
        )

    assert connection.steps == []


def test_publication_completion_updates_intent_before_terminal_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="publishing")),
            (
                "FROM video_digest_publication_intents",
                _publication_row(
                    stage="verified",
                    upload_evidence_artifact_version_id=UPLOAD_FILE.version_id,
                    verification_evidence_artifact_version_id=VERIFICATION_FILE.version_id,
                ),
            ),
            (
                "UPDATE video_digest_publication_intents",
                {"publication_id": PUBLICATION_ID},
            ),
            ("UPDATE video_digest_slots", {"slot_id": SLOT.slot_id}),
        ],
    )

    published = video_digest.complete_publication(
        _lease(), PUBLICATION_ID, recorded_at=NOW - timedelta(days=1)
    )

    assert published.published_at == NOW
    assert "video_digest_publication_intents" in connection.statements[-2]
    assert "video_digest_slots" in connection.statements[-1]
    assert "owner-a" in connection.parameters[-1]
    assert connection.steps == []


def test_publication_completion_replays_original_terminal_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published_at = NOW - timedelta(minutes=1)
    connection = _use_connection(
        monkeypatch,
        [
            (
                "FROM video_digest_slots",
                _slot_row(
                    stage="published",
                    edition_id=EDITION.edition_id,
                    claim_count=1,
                    terminal_lease_owner_token="owner-a",
                    terminal_lease_expires_at=NOW + timedelta(hours=1),
                    terminal_claim_count=1,
                ),
            ),
            (
                "FROM video_digest_publication_intents",
                _publication_row(stage="published", published_at=published_at),
            ),
        ],
    )

    published = video_digest.complete_publication(_lease(), PUBLICATION_ID, recorded_at=NOW)

    assert published.published_at == published_at
    assert connection.steps == []


def test_publication_progress_replays_verified_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = VerifiedPublication(
        evidence_artifact_version_id=VERIFICATION_FILE.version_id,
        video=VerifiedPublicObject(
            content_digest=PUBLICATION.video_digest,
            byte_size=PUBLICATION.video_byte_size,
            media_type=PUBLICATION.video_media_type,
            source_artifact_version_id=PUBLICATION.source_video_version_id,
        ),
    )
    connection = _use_connection(
        monkeypatch,
        [
            (
                "FROM video_digest_slots",
                _slot_row(
                    stage="published",
                    edition_id=EDITION.edition_id,
                    claim_count=1,
                    terminal_lease_owner_token="owner-a",
                    terminal_lease_expires_at=NOW + timedelta(hours=1),
                    terminal_claim_count=1,
                ),
            ),
            (
                "FROM video_digest_publication_intents",
                _publication_row(
                    stage="published",
                    upload_evidence_artifact_version_id=UPLOAD_FILE.version_id,
                    verification_evidence_artifact_version_id=VERIFICATION_FILE.version_id,
                    published_at=NOW,
                ),
            ),
            ("FROM artifacts AS artifact", _artifact_row(VERIFICATION_FILE)),
        ],
    )

    status = video_digest.checkpoint_publication_progress(
        _lease(),
        PUBLICATION_ID,
        verified,
        evidence_file=VERIFICATION_FILE,
        recorded_at=NOW,
    )

    assert status.stage is PublicationState.PUBLISHED
    assert connection.steps == []


def test_publication_failure_terminalizes_intent_before_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="publishing")),
            (
                "FROM video_digest_publication_intents",
                _publication_row(stage="uploading"),
            ),
            *_artifact_steps(),
            (
                "UPDATE video_digest_publication_intents",
                {"publication_id": PUBLICATION_ID},
            ),
            ("UPDATE video_digest_slots", {"slot_id": SLOT.slot_id}),
        ],
    )

    terminal = video_digest.fail_publication(
        _lease(),
        PUBLICATION_ID,
        state=PublicationState.CONFLICT,
        evidence_file=PUBLICATION_FAILURE_FILE,
        recorded_at=NOW,
    )

    assert terminal == TerminalSlot(state=TerminalSlotState.FAILED)
    assert "video_digest_publication_intents" in connection.statements[-2]
    assert "video_digest_slots" in connection.statements[-1]


def test_publication_failure_replays_matching_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            (
                "FROM video_digest_slots",
                _slot_row(
                    stage="failed",
                    edition_id=EDITION.edition_id,
                    claim_count=1,
                    terminal_lease_owner_token="owner-a",
                    terminal_lease_expires_at=NOW + timedelta(hours=1),
                    terminal_claim_count=1,
                    failure_evidence_artifact_version_id=PUBLICATION_FAILURE_FILE.version_id,
                ),
            ),
            ("FROM artifacts AS artifact", _artifact_row(PUBLICATION_FAILURE_FILE)),
            (
                "FROM video_digest_publication_intents",
                _publication_row(
                    stage="conflict",
                    failure_evidence_artifact_version_id=PUBLICATION_FAILURE_FILE.version_id,
                ),
            ),
        ],
    )

    terminal = video_digest.fail_publication(
        _lease(),
        PUBLICATION_ID,
        state=PublicationState.CONFLICT,
        evidence_file=PUBLICATION_FAILURE_FILE,
        recorded_at=NOW,
    )

    assert terminal == TerminalSlot(state=TerminalSlotState.FAILED)
    assert connection.steps == []


def test_generic_slot_failure_rejects_active_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="generating")),
            ("FROM video_digest_publication_intents", None),
            ("FROM video_digest_generation_requests", {"request_id": REQUEST.request_id}),
        ],
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="Active generation"):
        video_digest.fail_slot(_lease(), evidence_file=SLOT_FAILURE_FILE, recorded_at=NOW)
    assert connection.steps == []


def test_generic_slot_failure_fails_remaining_stories_before_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _use_connection(
        monkeypatch,
        [
            ("FROM video_digest_slots", _active_row(stage="assembling")),
            ("FROM video_digest_publication_intents", None),
            ("FROM video_digest_generation_requests", None),
            *_artifact_steps(),
            ("UPDATE video_digest_stories", None),
            ("UPDATE video_digest_slots", {"slot_id": SLOT.slot_id}),
        ],
    )

    terminal = video_digest.fail_slot(_lease(), evidence_file=SLOT_FAILURE_FILE, recorded_at=NOW)

    assert terminal == TerminalSlot(state=TerminalSlotState.FAILED)
    assert "video_digest_stories" in connection.statements[-2]
    assert "video_digest_slots" in connection.statements[-1]


def test_published_reads_hide_keys_and_preserve_edition_and_story_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    later = NOW + timedelta(minutes=1)
    summary_rows: list[dict[str, object]] = [
        {
            "publication_id": PUBLICATION_ID,
            "edition_id": EDITION.edition_id,
            "expected_video_key": PUBLICATION.expected_video_key,
            "video_digest": PUBLICATION.video_digest,
            "video_byte_size": PUBLICATION.video_byte_size,
            "video_media_type": PUBLICATION.video_media_type,
            "subtitle_expected_key": None,
            "subtitle_digest": None,
            "subtitle_byte_size": None,
            "subtitle_media_type": None,
            "published_at": later,
            "daily_report_version_id": REPORT_ID,
            "subtitle_state": "failed",
            "slot_name": "morning",
            "scheduled_at": SCHEDULED_AT,
            "day": SLOT.bucharest_day,
        }
    ]
    queries: list[str] = []

    def query(statement: str, _parameters: object) -> list[dict[str, object]]:
        queries.append(" ".join(statement.split()))
        if "FROM video_digest_stories" in statement:
            return [
                {
                    "story_id": STORY.story_id,
                    "position": 0,
                    "report_subject_id": STORY.report_subject_id,
                    "title": STORY.title,
                    "requested_duration_ms": STORY.requested_duration_ms,
                }
            ]
        return summary_rows

    monkeypatch.setattr(video_digest, "catalog_query", query)

    summaries = video_digest.list_published_editions(
        SLOT.bucharest_day,
        public_media_base_url="https://media.example.com",
    )
    edition = video_digest.read_published_edition(
        EDITION.edition_id,
        public_media_base_url="https://media.example.com/",
    )

    assert str(summaries[0].video.url) == (
        f"https://media.example.com/{PUBLICATION.expected_video_key}"
    )
    assert "expected_video_key" not in summaries[0].model_dump()
    assert isinstance(edition, PublishedEdition)
    assert edition.stories == (
        PublishedStory(
            story_id=STORY.story_id,
            position=0,
            report_subject_id=STORY.report_subject_id,
            title=STORY.title,
            requested_duration_ms=STORY.requested_duration_ms,
        ),
    )
    assert "published_at DESC, publication.publication_id DESC" in queries[0]
    assert "ORDER BY position ASC, story_id ASC" in queries[-1]


def test_published_reads_reject_non_origin_media_url_before_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        video_digest,
        "catalog_query",
        lambda *_args, **_kwargs: pytest.fail("invalid origin opened a query"),
    )

    with pytest.raises(ValueError, match="HTTPS origin"):
        video_digest.list_published_editions(
            SLOT.bucharest_day,
            public_media_base_url="https://media.example.com/path",
        )


def test_published_reads_reject_malformed_legacy_object_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        video_digest,
        "catalog_query",
        lambda *_args, **_kwargs: [
            {
                "publication_id": PUBLICATION_ID,
                "edition_id": EDITION.edition_id,
                "expected_video_key": "legacy/video.mp4?download=1",
                "video_digest": PUBLICATION.video_digest,
                "video_byte_size": PUBLICATION.video_byte_size,
                "video_media_type": PUBLICATION.video_media_type,
                "subtitle_expected_key": None,
                "subtitle_digest": None,
                "subtitle_byte_size": None,
                "subtitle_media_type": None,
                "published_at": NOW,
                "daily_report_version_id": REPORT_ID,
                "subtitle_state": "failed",
                "slot_name": "morning",
                "scheduled_at": SCHEDULED_AT,
                "day": SLOT.bucharest_day,
            }
        ],
    )

    with pytest.raises(ResearchCatalogError, match="object key"):
        video_digest.list_published_editions(
            SLOT.bucharest_day,
            public_media_base_url="https://media.example.com",
        )


def test_read_published_edition_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_digest, "catalog_query", lambda *_args, **_kwargs: [])

    assert (
        video_digest.read_published_edition(
            EDITION.edition_id,
            public_media_base_url="https://media.example.com",
        )
        is None
    )


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: video_digest.schedule_slot(SLOT, recorded_at=NOW.replace(tzinfo=None)),
            "recorded_at must include",
        ),
        (
            lambda: video_digest.schedule_slot(
                SLOT.model_copy(update={"scheduled_at": SCHEDULED_AT.replace(tzinfo=None)}),
                recorded_at=NOW,
            ),
            "scheduled_at must include",
        ),
        (
            lambda: video_digest.skip_slot(
                SLOT.slot_id,
                SlotSkipReason.UNCHANGED,
                recorded_at=NOW.replace(tzinfo=None),
            ),
            "recorded_at must include",
        ),
        (
            lambda: video_digest.claim_slot(
                SLOT.slot_id,
                EDITION,
                owner_token="owner",
                now=NOW.replace(tzinfo=None),
                lease_duration=timedelta(minutes=1),
            ),
            "now must include",
        ),
        (
            lambda: video_digest.claim_slot(
                SLOT.slot_id,
                EDITION,
                owner_token="  ",
                now=NOW,
                lease_duration=timedelta(minutes=1),
            ),
            "owner_token must not be empty",
        ),
        (
            lambda: video_digest.claim_slot(
                SLOT.slot_id,
                EDITION,
                owner_token="owner",
                now=NOW,
                lease_duration=timedelta(0),
            ),
            "lease_duration must be positive",
        ),
        (
            lambda: video_digest.renew_slot(
                _lease(),
                now=NOW.replace(tzinfo=None),
                lease_duration=timedelta(minutes=1),
            ),
            "now must include",
        ),
        (
            lambda: video_digest.renew_slot(
                _lease().model_copy(update={"expires_at": NOW.replace(tzinfo=None)}),
                now=NOW,
                lease_duration=timedelta(minutes=1),
            ),
            "lease.expires_at must include",
        ),
        (
            lambda: video_digest.renew_slot(
                _lease(),
                now=NOW,
                lease_duration=-timedelta(seconds=1),
            ),
            "lease_duration must be positive",
        ),
    ],
)
def test_public_operations_validate_before_opening_a_transaction(
    monkeypatch: pytest.MonkeyPatch,
    operation: Callable[[], object],
    message: str,
) -> None:
    monkeypatch.setattr(
        video_digest,
        "catalog_transaction",
        lambda *_args, **_kwargs: pytest.fail("validation opened a transaction"),
    )

    with pytest.raises(ValueError, match=message):
        operation()


def test_unique_catalog_conflict_becomes_checkpoint_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ResearchCatalogError("PostgreSQL catalog request failed")

    def fail_transaction(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(video_digest, "catalog_transaction", fail_transaction)
    monkeypatch.setattr(
        video_digest,
        "catalog_integrity_identity",
        lambda caught: "UniqueViolation:23505:video_digest_slots_one_active_edition"
        if caught is error
        else None,
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="identity conflicts"):
        video_digest.skip_slot(SLOT.slot_id, SlotSkipReason.UNCHANGED, recorded_at=NOW)
