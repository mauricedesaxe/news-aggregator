"""Read and build the current daily report projection."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import Field, ValidationError

from romanian_news import NewsModel, Sha256
from romanian_news.catalog.reports import publish_daily_report
from romanian_news.reports import (
    DailyReportDocument,
    DailyReportInput,
    build_daily_report_from_input,
    read_daily_report_input_cached,
)


class CurrentDailyReportHead(NewsModel):
    """Catalog metadata for one captured current daily report head."""

    day: date
    version_id: Sha256
    run_id: Sha256
    content_digest: Sha256
    r2_key: str
    input_time: datetime


class CurrentDailyReport(NewsModel):
    """One immutable published report and its captured catalog head."""

    head: CurrentDailyReportHead
    report: DailyReportDocument


class ReportInputsNotReady(NewsModel):
    kind: Literal["inputs_not_ready"] = "inputs_not_ready"
    day: date


class ReportMissing(NewsModel):
    kind: Literal["missing"] = "missing"
    day: date


class ReportFresh(NewsModel):
    kind: Literal["fresh"] = "fresh"
    head: CurrentDailyReportHead


class ReportStale(NewsModel):
    kind: Literal["stale"] = "stale"
    head: CurrentDailyReportHead


DailyReportFreshness = Annotated[
    ReportInputsNotReady | ReportMissing | ReportFresh | ReportStale,
    Field(discriminator="kind"),
]


def read_current_daily_report_head(day: date) -> CurrentDailyReportHead | None:
    """Capture the current report head and its recorded input timestamp."""
    from romanian_news.catalog.report_inputs import read_current_daily_report_record

    record = read_current_daily_report_record(day)
    if record is None:
        return None
    return CurrentDailyReportHead.model_validate(record.model_dump(), strict=True)


def read_current_daily_report(day: date) -> CurrentDailyReport | None:
    """Read the exact current report without constructing or publishing it."""
    from romanian_news.reports import parse_daily_report
    from romanian_news.storage import ResearchObjectIntegrityError, read_verified_r2_object

    head = read_current_daily_report_head(day)
    if head is None:
        return None
    content = read_verified_r2_object(head.r2_key, head.content_digest)
    try:
        report = parse_daily_report(content)
    except (ValidationError, ValueError) as error:
        raise ResearchObjectIntegrityError(
            f"Daily report payload is invalid: {head.version_id}"
        ) from error
    if report.day != day:
        raise ResearchObjectIntegrityError("Current daily report belongs to another day")
    return CurrentDailyReport(head=head, report=report)


def read_daily_report_freshness(day: date) -> DailyReportFreshness:
    """Compare one captured report run with current catalog-only input versions."""
    from romanian_news.catalog.report_inputs import (
        read_current_daily_report_input_versions,
        read_daily_report_run_input_versions,
    )

    head = read_current_daily_report_head(day)
    current_inputs = read_current_daily_report_input_versions(day)
    if current_inputs is None:
        return ReportInputsNotReady(day=day)
    if head is None:
        return ReportMissing(day=day)
    recorded_inputs = read_daily_report_run_input_versions(day, head.run_id)
    if recorded_inputs is None:
        raise ValueError(f"Daily report run has no exact inputs: {head.run_id}")
    if recorded_inputs == current_inputs:
        return ReportFresh(head=head)
    return ReportStale(head=head)


def build_and_publish_current_daily_report(
    day: date,
    implementation_ref: str,
) -> CurrentDailyReport:
    """Assemble the day's current inputs into a report and publish it if changed."""
    input_value = read_daily_report_input_cached(day)
    output = build_daily_report_from_input(input_value)
    publication = publish_daily_report(output, implementation_ref)
    return CurrentDailyReport(
        head=CurrentDailyReportHead(
            day=day,
            version_id=publication.version_id,
            run_id=publication.run_id,
            content_digest=output.content_digest,
            r2_key=(f"news/reports/daily/{day.isoformat()}/{output.content_digest}.json"),
            input_time=read_daily_report_input_time(input_value),
        ),
        report=output.report,
    )


def read_daily_report_input_time(value: DailyReportInput) -> datetime:
    """Return the newest current-version timestamp among the exact report inputs."""
    from romanian_news.catalog.artifacts import current_artifact_version_times

    artifact_ids = (
        value.themes.artifact_id,
        value.assessments.artifact_id,
        value.cluster_set.artifact_id,
        *(reference.artifact_id for reference in value.summaries),
        *(reference.artifact_id for reference in value.sentiments),
    )
    times = current_artifact_version_times(artifact_ids)
    if not times:
        raise ValueError(f"No current input versions exist for {value.day.isoformat()}")
    return max(times)
