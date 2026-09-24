from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from uuid import UUID, uuid4

from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog_transport import CatalogConnection, catalog_transaction


@dataclass(frozen=True)
class RepairReservation:
    day: date
    request_id: UUID
    run_id: str | None
    owns_launch: bool


def reserve_repair(day: date) -> RepairReservation:
    ensure_news_catalog_schema()
    request_id = uuid4()

    def reserve(connection: CatalogConnection) -> RepairReservation:
        connection.execute(
            "INSERT INTO daily_report_repair_requests (day, request_id) "
            "VALUES (%s, %s) ON CONFLICT(day) DO NOTHING",
            (day, request_id),
        )
        row = connection.execute(
            "SELECT request_id, run_id, launch_state FROM daily_report_repair_requests "
            "WHERE day = %s FOR UPDATE",
            (day,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Daily report repair reservation disappeared")
        return RepairReservation(
            day=day,
            request_id=UUID(str(row["request_id"])),
            run_id=str(row["run_id"]) if row["run_id"] is not None else None,
            owns_launch=row["run_id"] is None and row["launch_state"] == "reserved",
        )

    return catalog_transaction(reserve)


def replace_terminal_repair(reservation: RepairReservation) -> RepairReservation:
    replacement_id = uuid4()

    def replace(connection: CatalogConnection) -> RepairReservation:
        row = connection.execute(
            "UPDATE daily_report_repair_requests SET request_id = %s, run_id = NULL, "
            "launch_state = 'reserved' "
            "WHERE day = %s AND request_id = %s AND run_id = %s RETURNING request_id",
            (
                replacement_id,
                reservation.day,
                reservation.request_id,
                reservation.run_id,
            ),
        ).fetchone()
        if row is not None:
            return RepairReservation(reservation.day, replacement_id, None, True)
        current = connection.execute(
            "SELECT request_id, run_id " "FROM daily_report_repair_requests WHERE day = %s",
            (reservation.day,),
        ).fetchone()
        if current is None:
            raise RuntimeError("Daily report repair reservation disappeared")
        return RepairReservation(
            day=reservation.day,
            request_id=UUID(str(current["request_id"])),
            run_id=str(current["run_id"]) if current["run_id"] is not None else None,
            owns_launch=False,
        )

    return catalog_transaction(replace)


def mark_repair_launch_started(reservation: RepairReservation) -> bool:
    def mark(connection: CatalogConnection) -> bool:
        row = connection.execute(
            "UPDATE daily_report_repair_requests SET launch_state = 'launch_started' "
            "WHERE day = %s AND request_id = %s AND run_id IS NULL "
            "AND launch_state = 'reserved' RETURNING request_id",
            (reservation.day, reservation.request_id),
        ).fetchone()
        return row is not None

    return catalog_transaction(mark)


def bind_repair_run(reservation: RepairReservation, run_id: str) -> None:
    def bind(connection: CatalogConnection) -> None:
        row = connection.execute(
            "UPDATE daily_report_repair_requests SET run_id = %s "
            "WHERE day = %s AND request_id = %s AND run_id IS NULL RETURNING run_id",
            (run_id, reservation.day, reservation.request_id),
        ).fetchone()
        if row is not None:
            return
        current = connection.execute(
            "SELECT run_id FROM daily_report_repair_requests " "WHERE day = %s AND request_id = %s",
            (reservation.day, reservation.request_id),
        ).fetchone()
        if current is None or current["run_id"] != run_id:
            raise RuntimeError("Daily report repair run binding changed")

    catalog_transaction(bind)


def release_unlaunched_repair(reservation: RepairReservation) -> None:
    def release(connection: CatalogConnection) -> None:
        connection.execute(
            "DELETE FROM daily_report_repair_requests "
            "WHERE day = %s AND request_id = %s AND run_id IS NULL "
            "AND launch_state = 'reserved'",
            (reservation.day, reservation.request_id),
        )

    catalog_transaction(release)


def release_rejected_repair(reservation: RepairReservation) -> None:
    def release(connection: CatalogConnection) -> None:
        connection.execute(
            "DELETE FROM daily_report_repair_requests "
            "WHERE day = %s AND request_id = %s AND run_id IS NULL "
            "AND launch_state = 'launch_started'",
            (reservation.day, reservation.request_id),
        )

    catalog_transaction(release)
