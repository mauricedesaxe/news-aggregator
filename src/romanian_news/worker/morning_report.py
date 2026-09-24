from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import dagster as dg
from dagster import OpExecutionContext

from romanian_news.alerts import ping_heartbeat
from romanian_news.config import IMPLEMENTATION_REF
from romanian_news.current_report import build_and_publish_current_daily_report
from romanian_news.daily import read_daily_report_reference
from romanian_news.worker.assets import daily_reports

BUCHAREST_TIMEZONE = "Europe/Bucharest"


@dg.op(pool="news_catalog")
def morning_report_check_op(context: OpExecutionContext) -> str:
    day = _bucharest_today()
    current = build_and_publish_current_daily_report(day, IMPLEMENTATION_REF)
    context.log_event(
        dg.AssetMaterialization(
            asset_key=daily_reports.key,
            partition=day.isoformat(),
            metadata={"report_version_id": current.head.version_id},
        )
    )
    read_daily_report_reference(_previous_day(day))
    ping_heartbeat("morning_report")
    metadata = {
        "day": day.isoformat(),
        "report_version_id": current.head.version_id,
        "as_of": current.head.input_time.isoformat(),
    }
    context.add_output_metadata(metadata)
    return day.isoformat()


@dg.job
def morning_report_check() -> None:
    morning_report_check_op()


@dg.schedule(
    name="daily_morning_report_check",
    job=morning_report_check,
    cron_schedule="35 9 * * *",
    execution_timezone=BUCHAREST_TIMEZONE,
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def daily_morning_report_check(
    context: dg.ScheduleEvaluationContext,
) -> dg.RunRequest | dg.SkipReason:
    scheduled_at = _scheduled_time(context)
    active = context.instance.get_runs(
        filters=dg.RunsFilter(
            job_name=morning_report_check.name,
            statuses=_NONTERMINAL_RUN_STATUSES,
        ),
        limit=1,
    )
    if active:
        return dg.SkipReason("Skipping because morning_report_check already has an active run.")
    return dg.RunRequest(run_key=f"morning-report-check:{scheduled_at.date().isoformat()}")


_NONTERMINAL_RUN_STATUSES: tuple[dg.DagsterRunStatus, ...] = (
    dg.DagsterRunStatus.QUEUED,
    dg.DagsterRunStatus.NOT_STARTED,
    dg.DagsterRunStatus.MANAGED,
    dg.DagsterRunStatus.STARTING,
    dg.DagsterRunStatus.STARTED,
    dg.DagsterRunStatus.CANCELING,
)


def _bucharest_today() -> date:
    return _now().date()


def _previous_day(day: date) -> date:
    return day - timedelta(days=1)


def _now() -> datetime:
    return datetime.now(ZoneInfo(BUCHAREST_TIMEZONE))


def _scheduled_time(context: dg.ScheduleEvaluationContext) -> datetime:
    if context.scheduled_execution_time is None:
        raise ValueError("Schedule execution time is required")
    return context.scheduled_execution_time
