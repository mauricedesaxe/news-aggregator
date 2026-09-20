from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any, TypeVar, cast

import pytest

from romanian_news import Sha256
from romanian_news.catalog import video_digest
from romanian_news.catalog_transport import CatalogConnection, ResearchCatalogError
from romanian_news.video_digest.errors import (
    VideoDigestCheckpointConflictError,
    VideoDigestLeaseLostError,
)
from romanian_news.video_digest.models import (
    ClaimedSlot,
    EditionIdentity,
    ScheduledSlot,
    SkippedSlot,
    SlotId,
    SlotLease,
    SlotName,
    SlotSkipReason,
    TerminalSlot,
    TerminalSlotState,
    edition_id,
    scheduled_slot_id,
)

_Result = TypeVar("_Result")
NOW = datetime(2026, 9, 20, 9, tzinfo=UTC)
SCHEDULED_AT = datetime(2026, 9, 20, 6, tzinfo=UTC)
REPORT_ID: Sha256 = "1" * 64
POLICY_ID: Sha256 = "2" * 64
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


class _Cursor:
    def __init__(self, row: Mapping[str, object] | None) -> None:
        self._row = row

    def fetchone(self) -> Mapping[str, object] | None:
        return self._row


class _ScriptedConnection:
    def __init__(
        self,
        steps: Sequence[tuple[str, Mapping[str, object] | None]],
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
    steps: Sequence[tuple[str, Mapping[str, object] | None]],
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
) -> list[tuple[str, Mapping[str, object] | None]]:
    steps: list[tuple[str, Mapping[str, object] | None]] = [
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
