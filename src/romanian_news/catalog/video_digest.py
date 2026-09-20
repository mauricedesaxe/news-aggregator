from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any, TypeVar

from pydantic import ValidationError

from romanian_news.catalog_transport import (
    CatalogConnection,
    ResearchCatalogError,
    catalog_integrity_identity,
    catalog_transaction,
)
from romanian_news.video_digest.errors import (
    VideoDigestCheckpointConflictError,
    VideoDigestLeaseLostError,
)
from romanian_news.video_digest.models import (
    ClaimedSlot,
    ClaimResult,
    EditionIdentity,
    ScheduledSlot,
    SkippedSlot,
    SlotId,
    SlotLease,
    SlotName,
    SlotSkipReason,
    SlotStage,
    TerminalSlot,
    TerminalSlotState,
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
               lease_owner_token, lease_expires_at, claim_count, skip_reason
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
    return row, now


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


def _checkpoint_transaction(operation: Callable[[CatalogConnection], _Result]) -> _Result:
    try:
        return catalog_transaction(operation, retry_transient_errors=False)
    except ResearchCatalogError as error:
        identity = catalog_integrity_identity(error)
        if identity is not None and ":23505:" in identity:
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
