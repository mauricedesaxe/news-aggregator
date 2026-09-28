"""Run one bounded historical analysis pilot and publish its sourced report."""

from datetime import date, timedelta

import dagster as dg

from romanian_news.analysis.relevance import read_pending_relevance_references
from romanian_news.analysis.relevance_v3 import production_relevance_v3_request_id
from romanian_news.analysis.tracing import archive_model_day
from romanian_news.archive.backfill import next_page_window
from romanian_news.archive.campaign import (
    ARCHIVE_END,
    ARCHIVE_OUTLETS,
    ARCHIVE_START,
)
from romanian_news.archive.capture_batch import next_capture_window
from romanian_news.archive.windows import month_windows
from romanian_news.catalog.archive_model_spend import ArchiveSpendLimitReached, read_archive_spend
from romanian_news.catalog.archive_progress import list_archive_daily_reports
from romanian_news.catalog.archive_report_coverage import read_retrospective_coverage
from romanian_news.config import ARCHIVE_DAY_SPEND_LIMIT_USD, IMPLEMENTATION_REF
from romanian_news.daily import read_daily_article_references
from romanian_news.feedback import read_daily_report_version
from romanian_news.reports import RetrospectiveDailyReport
from romanian_news.retrospective_report import publish_retrospective_daily_report
from romanian_news.worker.operations import (
    materialize_clusters,
    materialize_daily_themes,
    materialize_embeddings,
    materialize_group_sentiment,
    materialize_group_summaries,
    materialize_relevance,
    materialize_subject_assessments,
)

_ARTICLE_LIMIT = 200
_ACTIVE_STATUSES = (
    dg.DagsterRunStatus.QUEUED,
    dg.DagsterRunStatus.NOT_STARTED,
    dg.DagsterRunStatus.MANAGED,
    dg.DagsterRunStatus.STARTING,
    dg.DagsterRunStatus.STARTED,
    dg.DagsterRunStatus.CANCELING,
)


class RetrospectiveAnalysisConfig(dg.Config):
    day: str


def analyze_retrospective_day(day: date, implementation_ref: str) -> str | None:
    if not ARCHIVE_START <= day <= ARCHIVE_END:
        raise ValueError("Retrospective analysis day is outside the one-year archive")
    coverage = read_retrospective_coverage(day)
    if coverage is None or len(coverage.included_outlets) < 2:
        raise ValueError("Retrospective pilot needs captured articles from two outlets")
    article_count = len(read_daily_article_references(day).values)
    if article_count < 1:
        raise ValueError("Retrospective analysis requires published articles")
    if article_count > coverage.captured_article_count:
        raise ValueError("Published articles exceed recorded archive captures")
    spend = read_archive_spend(day)
    if spend.spent_usd + spend.held_usd > ARCHIVE_DAY_SPEND_LIMIT_USD:
        raise ArchiveSpendLimitReached(f"Archive day {day} is over its model spend limit")
    with archive_model_day(day):
        materialize_relevance(day, implementation_ref, limit=_ARTICLE_LIMIT)
        if read_pending_relevance_references(
            day=day, request_id_for_article=production_relevance_v3_request_id
        ):
            return None
        for stage in (
            materialize_embeddings,
            materialize_clusters,
            materialize_group_summaries,
            materialize_group_sentiment,
            materialize_daily_themes,
            materialize_subject_assessments,
        ):
            stage(day, implementation_ref)
    return publish_retrospective_daily_report(day, implementation_ref).version_id


@dg.op
def retrospective_analysis(
    context: dg.OpExecutionContext, config: RetrospectiveAnalysisConfig
) -> None:
    day = date.fromisoformat(config.day)
    version_id = analyze_retrospective_day(day, IMPLEMENTATION_REF)
    if version_id is None:
        context.log.info(
            "Retrospective relevance batch for %s completed; more articles remain", day
        )
    else:
        context.log.info("Retrospective report for %s: version=%s", day, version_id)


@dg.job(tags={"dagster/max_runtime": "21600", "dagster/max_retries": "0"})
def retrospective_analysis_pilot() -> None:
    retrospective_analysis()


def next_automated_day() -> date | None:
    pending_months = []
    for outlet in ARCHIVE_OUTLETS:
        for next_window in (next_page_window, next_capture_window):
            window = next_window(outlet, ARCHIVE_START, ARCHIVE_END)
            if window is not None:
                pending_months.append(window[0])
    reports = {
        report.day: report for report in list_archive_daily_reports(ARCHIVE_START, ARCHIVE_END)
    }
    for month_start, month_end in month_windows(ARCHIVE_START, ARCHIVE_END):
        month_complete = not any(pending <= month_end for pending in pending_months)
        day = month_start
        while day <= month_end:
            published = reports.get(day)
            if published is not None and not month_complete:
                day += timedelta(days=1)
                continue
            coverage = read_retrospective_coverage(day)
            if coverage is not None and len(coverage.included_outlets) >= 2:
                article_count = len(read_daily_article_references(day).values)
                if article_count >= 1:
                    spend = read_archive_spend(day)
                    if spend.spent_usd + spend.held_usd >= ARCHIVE_DAY_SPEND_LIMIT_USD:
                        day += timedelta(days=1)
                        continue
                    if published is None:
                        return day
                    current = read_daily_report_version(published.version_id)
                    if (
                        isinstance(current, RetrospectiveDailyReport)
                        and current.retrospective.captured_article_count
                        < coverage.captured_article_count
                    ):
                        return day
            day += timedelta(days=1)
    return None


@dg.schedule(
    job=retrospective_analysis_pilot,
    cron_schedule="5,35 * * * *",
    execution_timezone="UTC",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def scheduled_retrospective_analysis(
    context: dg.ScheduleEvaluationContext,
) -> dg.RunRequest | dg.SkipReason:
    scheduled_at = context.scheduled_execution_time
    if scheduled_at is None:
        raise ValueError("Retrospective analysis schedule time is required")
    if context.instance.get_runs(
        filters=dg.RunsFilter(
            job_name=retrospective_analysis_pilot.name,
            statuses=_ACTIVE_STATUSES,
        ),
        limit=1,
    ):
        return dg.SkipReason("A retrospective report is already running.")
    day = next_automated_day()
    if day is None:
        return dg.SkipReason("No archive day needs a report in the one-year window.")
    return dg.RunRequest(
        run_key=f"retrospective:{day.isoformat()}:{scheduled_at.isoformat()}",
        run_config={"ops": {"retrospective_analysis": {"config": {"day": day.isoformat()}}}},
        tags={"news/archive_day": day.isoformat()},
    )
