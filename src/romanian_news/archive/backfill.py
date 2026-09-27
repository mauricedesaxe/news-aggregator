"""Choose and advance bounded publisher archive page-check windows."""

from collections import Counter
from datetime import date

from romanian_news.archive.page_checks import check_archive_pages, pending_page_candidates
from romanian_news.archive.windows import month_windows

ARCHIVE_START = date(2025, 9, 27)
ARCHIVE_END = date(2026, 9, 26)
ARCHIVE_OUTLETS = ("hotnews", "digi24")


def next_page_window(outlet_id: str, start: date, end: date) -> tuple[date, date] | None:
    for month_start, month_end in month_windows(start, end):
        if pending_page_candidates(outlet_id, month_start, month_end, 1):
            return month_start, month_end
    return None


def advance_page_checks(
    outlet_id: str, start: date, end: date, limit: int = 100
) -> dict[str, object]:
    window = next_page_window(outlet_id, start, end)
    if window is None:
        return {"outlet": outlet_id, "checked": 0, "complete": True}
    month_start, month_end = window
    checks = check_archive_pages(outlet_id, month_start, month_end, limit=limit)
    return {
        "outlet": outlet_id,
        "start": month_start.isoformat(),
        "end": month_end.isoformat(),
        "checked": len(checks),
        "statuses": dict(Counter(check.status for check in checks)),
    }
