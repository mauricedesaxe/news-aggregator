"""Advance one publisher's oldest unchecked month in the one-year archive."""

import argparse
import calendar
import json
import sys
from collections import Counter
from datetime import date, timedelta

from romanian_news.archive.page_checks import check_archive_pages, pending_page_candidates


def month_windows(start: date, end: date) -> tuple[tuple[date, date], ...]:
    if start > end:
        raise ValueError("Archive window is reversed")
    windows = []
    cursor = start
    while cursor <= end:
        final_day = calendar.monthrange(cursor.year, cursor.month)[1]
        month_end = min(end, date(cursor.year, cursor.month, final_day))
        windows.append((cursor, month_end))
        cursor = month_end + timedelta(days=1)
    return tuple(windows)


def advance_page_checks(
    outlet_id: str, start: date, end: date, limit: int = 100
) -> dict[str, object]:
    for month_start, month_end in month_windows(start, end):
        if not pending_page_candidates(outlet_id, month_start, month_end, 1):
            continue
        checks = check_archive_pages(outlet_id, month_start, month_end, limit=limit)
        return {
            "outlet": outlet_id,
            "start": month_start.isoformat(),
            "end": month_end.isoformat(),
            "checked": len(checks),
            "statuses": dict(Counter(check.status for check in checks)),
        }
    return {"outlet": outlet_id, "checked": 0, "complete": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outlet", choices=("hotnews", "digi24"), required=True)
    args = parser.parse_args()
    result = advance_page_checks(
        args.outlet,
        date(2025, 9, 27),
        date(2026, 9, 26),
    )
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
