"""Publish an explicitly sourced historical daily report from ready analysis inputs."""

from datetime import date

from romanian_news.catalog.archive_report_coverage import read_retrospective_coverage
from romanian_news.catalog.reports import NewsDailyReportPublication, publish_daily_report
from romanian_news.reports import (
    build_daily_report_from_input,
    build_retrospective_daily_report,
    read_daily_report_input_cached,
)


def publish_retrospective_daily_report(
    day: date, implementation_ref: str
) -> NewsDailyReportPublication:
    coverage = read_retrospective_coverage(day)
    if coverage is None:
        raise ValueError(f"No captured publisher articles exist for {day.isoformat()}")
    output = build_daily_report_from_input(read_daily_report_input_cached(day))
    return publish_daily_report(
        build_retrospective_daily_report(output, coverage), implementation_ref
    )
