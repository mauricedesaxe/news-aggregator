"""Durable admission and accounting for paid historical replay calls."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from romanian_news.catalog_transport import catalog_query, catalog_transaction


class ArchiveSpendLimitReached(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchiveSpend:
    spent_usd: Decimal
    held_usd: Decimal
    calls: int


@dataclass(frozen=True)
class ArchiveSpendHold:
    reservation_id: UUID
    created_at: datetime
    age_seconds: int
    operation_key: str
    request_id: str
    held_usd: Decimal


@dataclass(frozen=True)
class ArchiveSpendReconciliation:
    reservation_id: UUID
    actual_usd: Decimal
    operator: str
    reason: str
    billing_reference: str
    reconciled_at: datetime


def read_archive_spend(day: date) -> ArchiveSpend:
    row = catalog_query(
        "SELECT COALESCE(sum(actual_usd), 0) AS spent, "
        "COALESCE(sum(reserved_usd) FILTER (WHERE actual_usd IS NULL), 0) AS held, "
        "count(*) AS calls FROM news_archive_model_reservations WHERE day = %s",
        (day,),
    )[0]
    return ArchiveSpend(Decimal(str(row["spent"])), Decimal(str(row["held"])), int(row["calls"]))


def read_archive_spend_holds(day: date) -> tuple[ArchiveSpendHold, ...]:
    rows = catalog_query(
        "SELECT reservation_id, created_at, "
        "greatest(0, floor(extract(epoch FROM now() - created_at)))::bigint AS age_seconds, "
        "operation_key, request_id, reserved_usd "
        "FROM news_archive_model_reservations "
        "WHERE day = %s AND actual_usd IS NULL ORDER BY created_at, reservation_id",
        (day,),
    )
    return tuple(
        ArchiveSpendHold(
            reservation_id=UUID(str(row["reservation_id"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            age_seconds=int(row["age_seconds"]),
            operation_key=str(row["operation_key"]),
            request_id=str(row["request_id"]),
            held_usd=Decimal(str(row["reserved_usd"])),
        )
        for row in rows
    )


def reserve_archive_spend(
    day: date,
    reservation_id: UUID,
    operation_key: str,
    request_id: str,
    *,
    limit_usd: Decimal,
) -> None:
    if limit_usd <= 0:
        raise ValueError("Archive spend limit must be positive")

    def admit(connection) -> None:
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"archive-spend:{day}",)
        )
        row = connection.execute(
            "SELECT COALESCE(sum(COALESCE(actual_usd, reserved_usd)), 0) AS committed "
            "FROM news_archive_model_reservations WHERE day = %s",
            (day,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Archive spend ledger did not return a total")
        committed = Decimal(str(row["committed"]))
        if committed >= limit_usd:
            raise ArchiveSpendLimitReached(
                f"Archive day {day} spend admission is full "
                f"(${committed} committed, ${limit_usd} limit)"
            )
        connection.execute(
            "INSERT INTO news_archive_model_reservations "
            "(reservation_id, day, operation_key, request_id, reserved_usd) "
            "VALUES (%s, %s, %s, %s, %s)",
            (reservation_id, day, operation_key, request_id, limit_usd - committed),
        )

    catalog_transaction(admit)


def settle_archive_spend(reservation_id: UUID, actual_usd: Decimal) -> None:
    if actual_usd < 0:
        raise ValueError("Archive model cost cannot be negative")

    def settle(connection) -> None:
        row = connection.execute(
            "UPDATE news_archive_model_reservations SET actual_usd = %s, settled_at = now() "
            "WHERE reservation_id = %s AND actual_usd IS NULL RETURNING reservation_id",
            (actual_usd, reservation_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("Archive spend reservation is missing or already settled")

    catalog_transaction(settle)


def reconcile_archive_spend(
    reservation_id: UUID,
    actual_usd: Decimal,
    *,
    operator: str,
    reason: str,
    billing_reference: str,
) -> ArchiveSpendReconciliation:
    if not actual_usd.is_finite() or actual_usd < 0:
        raise ValueError("Measured archive model cost must be finite and nonnegative")
    exponent = actual_usd.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -10:
        raise ValueError("Measured archive model cost exceeds ledger precision")
    operator, reason, billing_reference = (
        operator.strip(),
        reason.strip(),
        billing_reference.strip(),
    )
    if not all((operator, reason, billing_reference)):
        raise ValueError("Operator, reason, and billing reference are required")

    def reconcile(connection) -> ArchiveSpendReconciliation:
        row = connection.execute(
            "SELECT actual_usd, settled_at, reconciled_at, reconciled_by, "
            "reconciliation_reason, billing_reference "
            "FROM news_archive_model_reservations WHERE reservation_id = %s FOR UPDATE",
            (reservation_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Archive spend reservation {reservation_id} does not exist")
        if row["actual_usd"] is not None:
            if (
                row["reconciled_at"] is not None
                and Decimal(str(row["actual_usd"])) == actual_usd
                and row["reconciled_by"] == operator
                and row["reconciliation_reason"] == reason
                and row["billing_reference"] == billing_reference
            ):
                return ArchiveSpendReconciliation(
                    reservation_id,
                    actual_usd,
                    operator,
                    reason,
                    billing_reference,
                    row["reconciled_at"],
                )
            raise ValueError("Reservation was already settled with different evidence")
        updated = connection.execute(
            "UPDATE news_archive_model_reservations SET actual_usd = %s, "
            "settled_at = now(), reconciled_at = now(), reconciled_by = %s, "
            "reconciliation_reason = %s, billing_reference = %s "
            "WHERE reservation_id = %s RETURNING reconciled_at",
            (actual_usd, operator, reason, billing_reference, reservation_id),
        ).fetchone()
        if updated is None:
            raise RuntimeError("Archive spend reconciliation did not update its reservation")
        return ArchiveSpendReconciliation(
            reservation_id,
            actual_usd,
            operator,
            reason,
            billing_reference,
            updated["reconciled_at"],
        )

    return catalog_transaction(reconcile)
