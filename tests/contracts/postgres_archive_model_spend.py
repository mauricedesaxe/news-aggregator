"""PostgreSQL admission contract for historical model calls."""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest
from psycopg.errors import RaiseException

from romanian_news.catalog.archive_model_spend import (
    ArchiveSpendLimitReached,
    read_archive_spend,
    read_archive_spend_holds,
    reconcile_archive_spend,
    reserve_archive_spend,
    settle_archive_spend,
)
from tests.postgres_catalog import postgres_catalog_fixture

pytestmark = pytest.mark.skipif(
    not os.getenv("NEWS_TEST_POSTGRES_DSN"), reason="NEWS_TEST_POSTGRES_DSN is required"
)
postgres_catalog = postgres_catalog_fixture("archive_model_spend")
DAY = date(2025, 9, 27)


def test_reservation_survives_replay_and_settlement_frees_allowance(postgres_catalog) -> None:
    first = uuid4()
    reserve_archive_spend(DAY, first, "news.relevance", "request-1", limit_usd=Decimal("1"))
    assert read_archive_spend(DAY).held_usd == Decimal("1")

    with pytest.raises(ArchiveSpendLimitReached):
        reserve_archive_spend(DAY, uuid4(), "news.relevance", "request-2", limit_usd=Decimal("1"))

    settle_archive_spend(first, Decimal("0.003"))
    assert read_archive_spend(DAY).spent_usd == Decimal("0.003")
    second = uuid4()
    reserve_archive_spend(DAY, second, "news.embed", "request-2", limit_usd=Decimal("1"))
    assert read_archive_spend(DAY).held_usd == Decimal("0.997")


def test_concurrent_admission_allows_only_one_call(postgres_catalog) -> None:
    def admit(index: int) -> str:
        try:
            reserve_archive_spend(
                DAY, uuid4(), "news.relevance", f"request-{index}", limit_usd=Decimal("1")
            )
        except ArchiveSpendLimitReached:
            return "blocked"
        return "admitted"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(admit, (1, 2)))
    assert sorted(outcomes) == ["admitted", "blocked"]


def test_unknown_cost_hold_is_visible_and_reconciliation_is_idempotent(postgres_catalog) -> None:
    reservation_id = uuid4()
    reserve_archive_spend(
        DAY, reservation_id, "news.relevance", "request-1", limit_usd=Decimal("1")
    )
    holds = read_archive_spend_holds(DAY)
    assert len(holds) == 1
    assert holds[0].reservation_id == reservation_id
    assert holds[0].operation_key == "news.relevance"
    assert holds[0].request_id == "request-1"
    assert holds[0].held_usd == Decimal("1")
    assert holds[0].age_seconds >= 0
    with pytest.raises(ArchiveSpendLimitReached):
        reserve_archive_spend(DAY, uuid4(), "news.embed", "request-2", limit_usd=Decimal("1"))

    evidence = dict(
        operator="operator", reason="Provider invoice checked", billing_reference="bill-1"
    )
    first = reconcile_archive_spend(reservation_id, Decimal("0.125"), **evidence)
    again = reconcile_archive_spend(reservation_id, Decimal("0.125"), **evidence)
    assert again == first
    assert read_archive_spend_holds(DAY) == ()
    assert read_archive_spend(DAY).spent_usd == Decimal("0.125")
    row = postgres_catalog.execute(
        "SELECT actual_usd, settled_at, reconciled_at, reconciled_by, "
        "reconciliation_reason, billing_reference "
        "FROM news_archive_model_reservations WHERE reservation_id = %s",
        (reservation_id,),
    ).fetchone()
    assert row == {
        "actual_usd": Decimal("0.1250000000"),
        "settled_at": first.reconciled_at,
        "reconciled_at": first.reconciled_at,
        "reconciled_by": "operator",
        "reconciliation_reason": "Provider invoice checked",
        "billing_reference": "bill-1",
    }
    reserve_archive_spend(DAY, uuid4(), "news.embed", "request-2", limit_usd=Decimal("1"))
    assert read_archive_spend(DAY).held_usd == Decimal("0.875")


def test_reconciliation_rejects_changes_and_automatic_settlement(postgres_catalog) -> None:
    reservation_id = uuid4()
    reserve_archive_spend(
        DAY, reservation_id, "news.relevance", "request-1", limit_usd=Decimal("1")
    )
    evidence = dict(
        operator="operator", reason="Provider invoice checked", billing_reference="bill-1"
    )
    reconcile_archive_spend(reservation_id, Decimal("0"), **evidence)
    with pytest.raises(ValueError, match="different evidence"):
        reconcile_archive_spend(reservation_id, Decimal("0.01"), **evidence)
    with pytest.raises(ValueError, match="different evidence"):
        reconcile_archive_spend(
            reservation_id, Decimal("0"), **{**evidence, "billing_reference": "bill-2"}
        )
    with pytest.raises(RaiseException, match="immutable"):
        postgres_catalog.execute(
            "UPDATE news_archive_model_reservations SET actual_usd = 1 WHERE reservation_id = %s",
            (reservation_id,),
        )

    automatic = uuid4()
    reserve_archive_spend(DAY, automatic, "news.embed", "request-2", limit_usd=Decimal("1"))
    settle_archive_spend(automatic, Decimal("0.01"))
    with pytest.raises(ValueError, match="different evidence"):
        reconcile_archive_spend(automatic, Decimal("0.01"), **evidence)


def test_reconciliation_rejects_invalid_evidence_before_writing(postgres_catalog) -> None:
    reservation_id = uuid4()
    reserve_archive_spend(
        DAY, reservation_id, "news.relevance", "request-1", limit_usd=Decimal("1")
    )
    evidence = dict(
        operator="operator", reason="Provider invoice checked", billing_reference="bill-1"
    )
    for cost in (Decimal("NaN"), Decimal("Infinity"), Decimal("-0.01"), Decimal("0.00000000001")):
        with pytest.raises(ValueError):
            reconcile_archive_spend(reservation_id, cost, **evidence)
    with pytest.raises(ValueError):
        reconcile_archive_spend(reservation_id, Decimal("0"), **{**evidence, "reason": " "})
    assert read_archive_spend_holds(DAY)[0].reservation_id == reservation_id


def test_reserve_archive_spend_rejects_non_positive_limit(postgres_catalog) -> None:
    with pytest.raises(ValueError, match="Archive spend limit must be positive"):
        reserve_archive_spend(DAY, uuid4(), "news.relevance", "request-1", limit_usd=Decimal("0"))
    assert read_archive_spend(DAY).calls == 0


def test_settle_archive_spend_rejects_negative_cost(postgres_catalog) -> None:
    reservation_id = uuid4()
    reserve_archive_spend(
        DAY, reservation_id, "news.relevance", "request-1", limit_usd=Decimal("1")
    )
    with pytest.raises(ValueError, match="Archive model cost cannot be negative"):
        settle_archive_spend(reservation_id, Decimal("-0.01"))
    spend = read_archive_spend(DAY)
    assert spend.spent_usd == Decimal("0")
    assert spend.held_usd == Decimal("1")


def test_settle_archive_spend_rejects_unknown_and_repeat_settlement(postgres_catalog) -> None:
    with pytest.raises(RuntimeError, match="missing or already settled"):
        settle_archive_spend(uuid4(), Decimal("0.01"))

    reservation_id = uuid4()
    reserve_archive_spend(
        DAY, reservation_id, "news.relevance", "request-1", limit_usd=Decimal("1")
    )
    settle_archive_spend(reservation_id, Decimal("0.003"))
    settled = postgres_catalog.execute(
        "SELECT actual_usd, settled_at FROM news_archive_model_reservations "
        "WHERE reservation_id = %s",
        (reservation_id,),
    ).fetchone()
    assert settled is not None
    with pytest.raises(RuntimeError, match="missing or already settled"):
        settle_archive_spend(reservation_id, Decimal("0.5"))
    assert (
        postgres_catalog.execute(
            "SELECT actual_usd, settled_at FROM news_archive_model_reservations "
            "WHERE reservation_id = %s",
            (reservation_id,),
        ).fetchone()
        == settled
    )


def test_replay_preflight_reports_a_saturated_day_and_rejects_out_of_window_days(
    postgres_catalog,
) -> None:
    from romanian_news.archive.replay_preflight import read_replay_preflight

    reserve_archive_spend(
        DAY, uuid4(), "news.relevance", "request-saturated", limit_usd=Decimal("10")
    )
    preflight = read_replay_preflight(DAY)
    assert preflight.available_usd == Decimal("0")
    assert preflight.paid_work_admissible is False
    with pytest.raises(ArchiveSpendLimitReached):
        reserve_archive_spend(DAY, uuid4(), "news.embed", "request-next", limit_usd=Decimal("10"))
    with pytest.raises(ValueError, match="Replay day is outside the one-year archive"):
        read_replay_preflight(date(2027, 1, 1))
