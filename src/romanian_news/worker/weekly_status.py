from __future__ import annotations

from datetime import date, datetime, timedelta

import dagster as dg

from romanian_news import BUCHAREST
from romanian_news.archive.campaign import ARCHIVE_END, ARCHIVE_START
from romanian_news.catalog.weekly_status import (
    WeeklyStatusNotFound,
    publish_weekly_status,
    read_weekly_status,
)
from romanian_news.config import IMPLEMENTATION_REF
from romanian_news.weekly_status import MIN_REPORT_DAYS
from romanian_news.weekly_status_generation import (
    build_weekly_status,
    load_week_reports,
    read_week_input,
)


def refresh_weekly_status(week_start: date, implementation_ref: str) -> str:
    inputs = read_week_input(week_start)
    if inputs.available_days < MIN_REPORT_DAYS:
        return "insufficient"
    try:
        _version_id, current = read_weekly_status(week_start)
    except WeeklyStatusNotFound:
        current = None
    if current is not None and current.policy == inputs.policy and current.days == inputs.days:
        return "current"
    reports = load_week_reports(inputs)
    output = build_weekly_status(inputs, reports)
    publication = publish_weekly_status(output, implementation_ref)
    return publication.status


@dg.op(config_schema={"week_starts": [str]})
def refresh_weekly_statuses(context) -> None:
    for raw_day in context.op_config["week_starts"]:
        week_start = date.fromisoformat(raw_day)
        result = refresh_weekly_status(week_start, IMPLEMENTATION_REF)
        context.log.info("Weekly status %s: %s", week_start.isoformat(), result)


@dg.job
def weekly_status_refresh() -> None:
    refresh_weekly_statuses()


def recent_completed_week_starts(today: date, count: int = 4) -> tuple[date, ...]:
    current_monday = today - timedelta(days=today.weekday())
    return tuple(current_monday - timedelta(weeks=offset) for offset in range(1, count + 1))


def next_historical_week(today: date) -> date | None:
    recent = recent_completed_week_starts(today)
    last_week = min(
        ARCHIVE_END - timedelta(days=ARCHIVE_END.weekday()),
        min(recent) - timedelta(weeks=1),
    )
    week = ARCHIVE_START - timedelta(days=ARCHIVE_START.weekday())
    while week <= last_week:
        inputs = read_week_input(week)
        if inputs.available_days >= MIN_REPORT_DAYS:
            try:
                _version_id, current = read_weekly_status(week)
            except WeeklyStatusNotFound:
                return week
            if current.policy != inputs.policy or current.days != inputs.days:
                return week
        week += timedelta(weeks=1)
    return None


@dg.schedule(
    job=weekly_status_refresh,
    cron_schedule="20 */2 * * *",
    execution_timezone="UTC",
    default_status=dg.DefaultScheduleStatus.STOPPED,
)
def scheduled_historical_weekly_status(
    context: dg.ScheduleEvaluationContext,
) -> dg.RunRequest | dg.SkipReason:
    scheduled_at = context.scheduled_execution_time
    if scheduled_at is None:
        raise ValueError("Historical weekly status schedule time is required")
    if context.instance.get_runs(
        filters=dg.RunsFilter(
            job_name=weekly_status_refresh.name,
            statuses=(
                dg.DagsterRunStatus.QUEUED,
                dg.DagsterRunStatus.NOT_STARTED,
                dg.DagsterRunStatus.MANAGED,
                dg.DagsterRunStatus.STARTING,
                dg.DagsterRunStatus.STARTED,
                dg.DagsterRunStatus.CANCELING,
            ),
        ),
        limit=1,
    ):
        return dg.SkipReason("A weekly status refresh is already running.")
    week = next_historical_week(scheduled_at.date())
    if week is None:
        return dg.SkipReason("No historical week has new published daily reports.")
    return dg.RunRequest(
        run_key=f"archive-weekly:{week.isoformat()}:{scheduled_at.isoformat()}",
        run_config={
            "ops": {"refresh_weekly_statuses": {"config": {"week_starts": [week.isoformat()]}}}
        },
        tags={"news/archive_week": week.isoformat()},
    )


@dg.schedule(
    job=weekly_status_refresh,
    cron_schedule="0 11 * * *",
    execution_timezone="Europe/Bucharest",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def scheduled_weekly_status(
    context: dg.ScheduleEvaluationContext,
) -> dg.RunRequest:
    scheduled_at = context.scheduled_execution_time or datetime.now(BUCHAREST)
    week_starts = recent_completed_week_starts(scheduled_at.date())
    return dg.RunRequest(
        run_key=f"weekly-status:{scheduled_at.date().isoformat()}",
        run_config={
            "ops": {
                "refresh_weekly_statuses": {
                    "config": {"week_starts": [day.isoformat() for day in week_starts]}
                }
            }
        },
    )
