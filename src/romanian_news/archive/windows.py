"""Bounded calendar windows for historical publisher work."""

import calendar
from datetime import date, timedelta


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
