from datetime import date

from romanian_news.worker.catalog_status import (
    DayCatalogStatus,
    format_catalog_status,
    format_day_catalog_status,
    recent_catalog_days,
)


def test_recent_catalog_days_are_inclusive_and_oldest_first() -> None:
    assert recent_catalog_days(date(2026, 9, 23), count=4) == (
        date(2026, 9, 20),
        date(2026, 9, 21),
        date(2026, 9, 22),
        date(2026, 9, 23),
    )


def test_format_day_catalog_status_truncates_report_id() -> None:
    row = DayCatalogStatus(
        day=date(2026, 9, 23),
        successful_feeds=12,
        expected_feeds=12,
        freshness="missing",
        report_version_id="a" * 64,
    )

    assert format_day_catalog_status(row) == "2026-09-23 feeds=12/12 freshness=missing report=" + (
        "a" * 12
    )


def test_format_catalog_status_includes_expected_feeds_and_missing_report() -> None:
    rows = (
        DayCatalogStatus(
            day=date(2026, 9, 22),
            successful_feeds=0,
            expected_feeds=12,
            freshness="inputs_not_ready",
            report_version_id=None,
        ),
        DayCatalogStatus(
            day=date(2026, 9, 23),
            successful_feeds=12,
            expected_feeds=12,
            freshness="fresh",
            report_version_id="b" * 64,
        ),
    )

    assert format_catalog_status(rows) == "\n".join(
        (
            "expected_feeds=12",
            "2026-09-22 feeds=0/12 freshness=inputs_not_ready report=-",
            "2026-09-23 feeds=12/12 freshness=fresh report=" + ("b" * 12),
        )
    )
