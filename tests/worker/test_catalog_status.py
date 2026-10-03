from datetime import date

from romanian_news.worker.catalog_status import recent_catalog_days


def test_recent_catalog_days_are_inclusive_and_oldest_first() -> None:
    assert recent_catalog_days(date(2026, 9, 23), count=4) == (
        date(2026, 9, 20),
        date(2026, 9, 21),
        date(2026, 9, 22),
        date(2026, 9, 23),
    )
