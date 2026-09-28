from datetime import date

from romanian_news.archive.backfill import ARCHIVE_END, ARCHIVE_START
from romanian_news.archive.windows import month_windows


def test_month_windows_keep_partial_first_and_last_months() -> None:
    assert month_windows(date(2025, 9, 27), date(2025, 11, 2)) == (
        (date(2025, 9, 27), date(2025, 9, 30)),
        (date(2025, 10, 1), date(2025, 10, 31)),
        (date(2025, 11, 1), date(2025, 11, 2)),
    )


def test_production_archive_spans_one_full_year_of_month_windows() -> None:
    windows = month_windows(ARCHIVE_START, ARCHIVE_END)

    assert len(windows) == 13
    assert windows[0][0] == ARCHIVE_START
    assert windows[0][1] == date(2025, 9, 30)
    assert windows[-1][1] == ARCHIVE_END
