"""Schedule bounded publisher page checks in the running Dagster deployment."""

import dagster as dg

from romanian_news.archive.backfill import (
    ARCHIVE_END,
    ARCHIVE_OUTLETS,
    ARCHIVE_START,
    advance_page_checks,
    next_page_window,
)

_ACTIVE_STATUSES = (
    dg.DagsterRunStatus.QUEUED,
    dg.DagsterRunStatus.NOT_STARTED,
    dg.DagsterRunStatus.MANAGED,
    dg.DagsterRunStatus.STARTING,
    dg.DagsterRunStatus.STARTED,
    dg.DagsterRunStatus.CANCELING,
)


class ArchivePageBackfillConfig(dg.Config):
    outlet: str


@dg.op
def archive_page_backfill(
    context: dg.OpExecutionContext, config: ArchivePageBackfillConfig
) -> None:
    result = advance_page_checks(config.outlet, ARCHIVE_START, ARCHIVE_END)
    context.log.info("Archive page backfill: %s", result)


@dg.job(tags={"dagster/max_runtime": "900", "dagster/max_retries": "0"})
def archive_page_backfill_job() -> None:
    archive_page_backfill()


@dg.schedule(
    job=archive_page_backfill_job,
    cron_schedule="3,18,33,48 * * * *",
    execution_timezone="UTC",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def scheduled_archive_page_backfill(
    context: dg.ScheduleEvaluationContext,
) -> list[dg.RunRequest] | dg.SkipReason:
    scheduled_at = context.scheduled_execution_time
    if scheduled_at is None:
        raise ValueError("Archive page schedule time is required")
    requests = []
    for outlet in ARCHIVE_OUTLETS:
        if next_page_window(outlet, ARCHIVE_START, ARCHIVE_END) is None:
            continue
        active = context.instance.get_runs(
            filters=dg.RunsFilter(
                job_name=archive_page_backfill_job.name,
                statuses=_ACTIVE_STATUSES,
                tags={"news/archive_outlet": outlet},
            ),
            limit=1,
        )
        if active:
            continue
        requests.append(
            dg.RunRequest(
                run_key=f"archive-page:{outlet}:{scheduled_at.isoformat()}",
                run_config={"ops": {"archive_page_backfill": {"config": {"outlet": outlet}}}},
                tags={"news/archive_outlet": outlet},
            )
        )
    return requests or dg.SkipReason("No outlet has unchecked archive pages ready.")
