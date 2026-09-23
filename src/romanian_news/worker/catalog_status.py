from datetime import date, datetime, timedelta

from romanian_news import BUCHAREST, NewsModel
from romanian_news.catalog.feeds import read_successful_feed_counts
from romanian_news.current_report import (
    read_current_daily_report_head,
    read_daily_report_freshness,
)
from romanian_news.feeds.registry import feed_registry


class DayCatalogStatus(NewsModel):
    day: date
    successful_feeds: int
    expected_feeds: int
    freshness: str
    report_version_id: str | None


def recent_catalog_days(today: date | None = None, *, count: int = 4) -> tuple[date, ...]:
    current = today or datetime.now(BUCHAREST).date()
    return tuple(current - timedelta(days=offset) for offset in range(count - 1, -1, -1))


def format_catalog_status(rows: tuple[DayCatalogStatus, ...]) -> str:
    if not rows:
        return "catalog_status: no days"
    expected = rows[0].expected_feeds
    lines = [f"expected_feeds={expected}"]
    lines.extend(format_day_catalog_status(row) for row in rows)
    return "\n".join(lines)


def format_day_catalog_status(row: DayCatalogStatus) -> str:
    report = row.report_version_id[:12] if row.report_version_id else "-"
    return (
        f"{row.day.isoformat()} feeds={row.successful_feeds}/{row.expected_feeds} "
        f"freshness={row.freshness} report={report}"
    )


def read_day_catalog_status(
    day: date, expected_feeds: int, successful_feeds: int
) -> DayCatalogStatus:
    try:
        freshness = read_daily_report_freshness(day).kind
    except (ValueError, RuntimeError) as error:
        freshness = f"error:{error}"
    head = read_current_daily_report_head(day)
    return DayCatalogStatus(
        day=day,
        successful_feeds=successful_feeds,
        expected_feeds=expected_feeds,
        freshness=freshness,
        report_version_id=None if head is None else head.version_id,
    )


def print_catalog_status(days: tuple[date, ...] | None = None) -> None:
    selected = days or recent_catalog_days()
    expected = len(feed_registry().feeds)
    counts = read_successful_feed_counts(selected)
    rows = tuple(read_day_catalog_status(day, expected, counts.get(day, 0)) for day in selected)
    print(format_catalog_status(rows))
