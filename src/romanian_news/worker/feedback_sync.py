import dagster as dg
from dagster import OpExecutionContext

from romanian_news.analysis.feedback_sync import NewsFeedbackSyncResult, sync_news_feedback
from romanian_news.catalog.evaluations import load_news_evaluation_release
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.evaluation import PIN_PATH
from romanian_news.worker.run_overlap import OVERLAP_BLOCKING_RUN_STATUSES


@dg.op(pool="news_feedback_network")
def news_feedback_sync_op(context: OpExecutionContext) -> NewsFeedbackSyncResult:
    ensure_news_catalog_schema()
    release = load_news_evaluation_release(PIN_PATH.read_bytes())
    result = sync_news_feedback(release.manifest_reference.version_id)
    metadata = result.model_dump()
    context.add_output_metadata(metadata)
    if result.failed:
        raise dg.Failure(
            description=f"{result.failed} feedback sync attempts failed",
            metadata=metadata,
        )
    return result


@dg.job
def news_feedback_sync() -> None:
    news_feedback_sync_op()


@dg.schedule(
    name="quarter_hourly_news_feedback_sync",
    job=news_feedback_sync,
    cron_schedule="*/15 * * * *",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def quarter_hourly_news_feedback_sync(
    context: dg.ScheduleEvaluationContext,
) -> dg.RunRequest | dg.SkipReason:
    existing_runs = context.instance.get_runs(
        filters=dg.RunsFilter(
            job_name=news_feedback_sync.name,
            statuses=OVERLAP_BLOCKING_RUN_STATUSES,
        ),
        limit=1,
    )
    if existing_runs:
        return dg.SkipReason(
            "Skipping because news_feedback_sync already has a queued or active run."
        )
    return dg.RunRequest()
