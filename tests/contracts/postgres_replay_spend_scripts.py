"""CLI contracts for the archive spend reconciliation and replay preflight scripts."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from romanian_news.catalog.archive_model_spend import reserve_archive_spend
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog_transport import catalog_query
from scripts import preflight_historical_replay, reconcile_historical_spend

pytestmark = pytest.mark.skipif(
    not os.getenv("NEWS_TEST_POSTGRES_DSN"), reason="NEWS_TEST_POSTGRES_DSN is required"
)

DAY = date(2025, 9, 27)


def _reconcile_argv(reservation_id: UUID, amount: str, billing_reference: str) -> list[str]:
    return [
        "reconcile_historical_spend",
        str(reservation_id),
        amount,
        "--operator",
        "ops@example.com",
        "--reason",
        "provider outage",
        "--billing-reference",
        billing_reference,
    ]


def test_reconcile_historical_spend_reports_and_persists_the_audit_record(
    postgres_news_schema: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    reservation_id = uuid4()
    reserve_archive_spend(
        DAY, reservation_id, "news.relevance", "request-1", limit_usd=Decimal("1")
    )
    monkeypatch.setattr(sys, "argv", _reconcile_argv(reservation_id, "0.125", "INV-9"))
    reconcile_historical_spend.main()
    reported = json.loads(capsys.readouterr().out)
    assert reported["reservation_id"] == str(reservation_id)
    assert reported["actual_usd"] == "0.125"
    assert reported["operator"] == "ops@example.com"
    assert reported["reason"] == "provider outage"
    assert reported["billing_reference"] == "INV-9"
    row = catalog_query(
        "SELECT actual_usd, reconciled_at, reconciled_by, reconciliation_reason, "
        "billing_reference FROM news_archive_model_reservations "
        "WHERE reservation_id = %s",
        (reservation_id,),
    )[0]
    assert row["actual_usd"] == Decimal("0.125")
    assert row["reconciled_at"] is not None
    assert row["reconciled_by"] == "ops@example.com"
    assert row["reconciliation_reason"] == "provider outage"
    assert row["billing_reference"] == "INV-9"

    with pytest.raises(SystemExit) as exited:
        monkeypatch.setattr(sys, "argv", _reconcile_argv(reservation_id, "not-a-number", "INV-9"))
        reconcile_historical_spend.main()
    assert exited.value.code != 0
    assert "Cost must be a decimal amount in USD" in capsys.readouterr().err

    settled = catalog_query(
        "SELECT actual_usd, reconciled_at, reconciled_by, reconciliation_reason, "
        "billing_reference FROM news_archive_model_reservations "
        "WHERE reservation_id = %s",
        (reservation_id,),
    )[0]
    # The script lets domain errors surface as tracebacks instead of catching
    # them, so an in-process main() raises where the CLI process exits nonzero.
    with pytest.raises(ValueError, match="different evidence"):
        monkeypatch.setattr(sys, "argv", _reconcile_argv(reservation_id, "0.5", "INV-10"))
        reconcile_historical_spend.main()
    assert (
        catalog_query(
            "SELECT actual_usd, reconciled_at, reconciled_by, reconciliation_reason, "
            "billing_reference FROM news_archive_model_reservations "
            "WHERE reservation_id = %s",
            (reservation_id,),
        )[0]
        == settled
    )


def test_preflight_historical_replay_prints_a_read_only_preview(
    postgres_news_schema: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    reservations_before = catalog_query(
        "SELECT count(*) AS reservations FROM news_archive_model_reservations"
    )[0]["reservations"]
    monkeypatch.setattr(sys, "argv", ["preflight_historical_replay", DAY.isoformat()])
    preflight_historical_replay.main()
    preview = json.loads(capsys.readouterr().out)
    assert preview["day"] == "2025-09-27"
    assert preview["verified_pages"] == 0
    assert preview["captured_articles"] == 0
    assert preview["pending_relevance"] == 0
    assert preview["pending_embeddings"] == 0
    assert preview["limit_usd"] == "10"
    assert preview["available_usd"] == "10"
    assert preview["paid_work_admissible"] is True
    assert (
        catalog_query("SELECT count(*) AS reservations FROM news_archive_model_reservations")[0][
            "reservations"
        ]
        == reservations_before
    )

    with pytest.raises(ValueError, match="Replay day is outside the one-year archive"):
        monkeypatch.setattr(sys, "argv", ["preflight_historical_replay", "2027-01-01"])
        preflight_historical_replay.main()
    assert (
        catalog_query("SELECT count(*) AS reservations FROM news_archive_model_reservations")[0][
            "reservations"
        ]
        == reservations_before
    )
