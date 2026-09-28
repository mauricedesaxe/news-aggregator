"""PostgreSQL admission contract for historical model calls."""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from romanian_news.catalog.archive_model_spend import (
    ArchiveSpendLimitReached,
    read_archive_spend,
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
