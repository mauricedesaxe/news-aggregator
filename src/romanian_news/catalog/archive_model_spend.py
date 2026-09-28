"""Durable admission and accounting for paid historical replay calls."""

from dataclasses import dataclass
from datetime import date
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


def read_archive_spend(day: date) -> ArchiveSpend:
    row = catalog_query(
        "SELECT COALESCE(sum(actual_usd), 0) AS spent, "
        "COALESCE(sum(reserved_usd) FILTER (WHERE actual_usd IS NULL), 0) AS held, "
        "count(*) AS calls FROM news_archive_model_reservations WHERE day = %s",
        (day,),
    )[0]
    return ArchiveSpend(Decimal(str(row["spent"])), Decimal(str(row["held"])), int(row["calls"]))


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
