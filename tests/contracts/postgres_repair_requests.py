from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date

from romanian_news.reader.repair_requests import (
    bind_repair_run,
    mark_repair_launch_started,
    release_rejected_repair,
    release_unlaunched_repair,
    replace_terminal_repair,
    reserve_repair,
)
from tests.postgres_catalog import postgres_catalog_fixture

postgres_catalog = postgres_catalog_fixture("repair_requests")
DAY = date(2026, 9, 15)


def test_concurrent_repair_reservations_have_one_launch_owner(postgres_catalog) -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        reservations = tuple(executor.map(reserve_repair, (DAY, DAY)))

    assert {reservation.request_id for reservation in reservations} == {reservations[0].request_id}
    assert all(reservation.owns_launch for reservation in reservations)
    assert sum(mark_repair_launch_started(reservation) for reservation in reservations) == 1
    rows = postgres_catalog.query(
        "SELECT request_id, run_id FROM daily_report_repair_requests WHERE day = %s", [DAY]
    )
    assert len(rows) == 1
    assert rows[0]["run_id"] is None


def test_bound_repair_is_reused_and_terminal_rotation_is_compare_and_swap(
    postgres_catalog,
) -> None:
    first = reserve_repair(DAY)
    assert mark_repair_launch_started(first)
    bind_repair_run(first, "run-1")
    bind_repair_run(first, "run-1")
    observed = reserve_repair(DAY)
    assert observed.run_id == "run-1"
    assert not observed.owns_launch

    replacement = replace_terminal_repair(observed)
    losing_caller = replace_terminal_repair(observed)

    assert replacement.owns_launch
    assert replacement.run_id is None
    assert replacement.request_id != first.request_id
    assert not losing_caller.owns_launch
    assert losing_caller.request_id == replacement.request_id
    assert losing_caller.run_id is None


def test_prelaunch_failure_releases_only_its_unbound_reservation(postgres_catalog) -> None:
    first = reserve_repair(DAY)
    release_unlaunched_repair(first)
    retry = reserve_repair(DAY)

    assert retry.owns_launch
    assert retry.request_id != first.request_id


def test_definitive_launch_rejection_releases_started_reservation(postgres_catalog) -> None:
    first = reserve_repair(DAY)
    assert mark_repair_launch_started(first)
    assert not reserve_repair(DAY).owns_launch

    release_rejected_repair(first)
    retry = reserve_repair(DAY)

    assert retry.owns_launch
    assert retry.request_id != first.request_id
