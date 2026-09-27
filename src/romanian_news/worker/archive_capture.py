"""Manual production job for bounded historical article capture."""

from datetime import date

import dagster as dg

from romanian_news.archive.capture_batch import capture_archive_batch, next_capture_window
from romanian_news.config import IMPLEMENTATION_REF

_ARCHIVE_START = date(2025, 9, 27)
_ARCHIVE_END = date(2026, 9, 26)
_ARCHIVE_OUTLETS = ("hotnews", "digi24")
_ACTIVE_STATUSES = (
    dg.DagsterRunStatus.QUEUED,
    dg.DagsterRunStatus.NOT_STARTED,
    dg.DagsterRunStatus.MANAGED,
    dg.DagsterRunStatus.STARTING,
    dg.DagsterRunStatus.STARTED,
    dg.DagsterRunStatus.CANCELING,
)


class ArchiveCaptureConfig(dg.Config):
    outlet: str
    start: str
    end: str
    limit: int = 10


@dg.op
def archive_article_capture(context: dg.OpExecutionContext, config: ArchiveCaptureConfig) -> None:
    result = capture_archive_batch(
        config.outlet,
        date.fromisoformat(config.start),
        date.fromisoformat(config.end),
        limit=config.limit,
        implementation_ref=IMPLEMENTATION_REF,
    )
    context.log.info(
        "Archive article capture: selected=%s published=%s unchanged=%s failed=%s",
        result.selected,
        result.published,
        result.unchanged,
        len(result.failed),
    )
    for url, reason in result.failed:
        context.log.warning("Archive article skipped: %s (%s)", url, reason)
    if result.failed:
        raise RuntimeError(f"Archive capture failed for {len(result.failed)} selected pages")


@dg.job(tags={"dagster/max_runtime": "1800", "dagster/max_retries": "0"})
def archive_article_capture_batch() -> None:
    archive_article_capture()


@dg.schedule(
    job=archive_article_capture_batch,
    cron_schedule="10,25,40,55 * * * *",
    execution_timezone="UTC",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def scheduled_archive_article_capture(
    context: dg.ScheduleEvaluationContext,
) -> list[dg.RunRequest] | dg.SkipReason:
    scheduled_at = context.scheduled_execution_time
    if scheduled_at is None:
        raise ValueError("Archive capture schedule time is required")
    requests = []
    for outlet in _ARCHIVE_OUTLETS:
        window = next_capture_window(outlet, _ARCHIVE_START, _ARCHIVE_END)
        if window is None:
            continue
        active = context.instance.get_runs(
            filters=dg.RunsFilter(
                job_name=archive_article_capture_batch.name,
                statuses=_ACTIVE_STATUSES,
                tags={"news/archive_outlet": outlet},
            ),
            limit=1,
        )
        if active:
            continue
        start, end = window
        requests.append(
            dg.RunRequest(
                run_key=f"archive-article:{outlet}:{scheduled_at.isoformat()}",
                run_config={
                    "ops": {
                        "archive_article_capture": {
                            "config": {
                                "outlet": outlet,
                                "start": start.isoformat(),
                                "end": end.isoformat(),
                                "limit": 50,
                            }
                        }
                    }
                },
                tags={"news/archive_outlet": outlet},
            )
        )
    return requests or dg.SkipReason("No outlet has uncaptured verified articles ready.")
