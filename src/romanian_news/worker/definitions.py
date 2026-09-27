import hashlib
import json
from datetime import UTC, date, datetime, timedelta

import dagster as dg

from romanian_news import BUCHAREST
from romanian_news.articles.acquisition import plan_article_work
from romanian_news.articles.recovery import (
    read_article_attempt_states,
    read_article_candidate_days,
)
from romanian_news.config import IMPLEMENTATION_REF
from romanian_news.daily import bucharest_day_window, read_daily_article_references
from romanian_news.feeds.registry import feed_registry
from romanian_news.worker.archive_capture import archive_article_capture_batch
from romanian_news.worker.assets import (
    BUCHAREST_TIMEZONE,
    DAILY_PARTITIONS,
    articles,
    articles_have_no_quarantined_inputs,
    daily_clusters,
    daily_report_complete,
    daily_report_exact_inputs,
    daily_reports,
    daily_subject_assessment_exact_inputs,
    daily_subject_assessments,
    daily_theme_exact_inputs,
    daily_themes,
    embeddings,
    feed_intake,
    group_sentiment,
    group_summaries,
    hourly_partition_key,
    jev_relevance_shadow,
    relevance,
    weekly_report_exact_inputs,
    weekly_reports,
    youtube_publication,
    youtube_source,
)
from romanian_news.worker.catalog_schema import catalog_schema_activation
from romanian_news.worker.feedback_sync import (
    news_feedback_sync,
    quarter_hourly_news_feedback_sync,
)
from romanian_news.worker.jev_relevance_evaluation import jev_relevance_evaluation
from romanian_news.worker.morning_report import (
    daily_morning_report_check,
    morning_report_check,
)
from romanian_news.worker.relevance_comparison import fresh_relevance_comparison
from romanian_news.worker.relevance_v3_evaluation import relevance_v3_evaluation
from romanian_news.worker.theme_comparison import fresh_theme_comparison
from romanian_news.worker.video_digest import scheduled_video_digest, video_digest_job
from romanian_news.worker.video_digest_monitor import (
    scheduled_video_digest_incident_monitor,
    video_digest_incident_monitor_job,
)
from romanian_news.worker.weekly_status import (
    scheduled_weekly_status,
    weekly_status_refresh,
)
from romanian_news.youtube.models import YOUTUBE_SOURCES

WEEKLY_ASSETS = dg.AssetSelection.groups("romanian_news_weekly")
_daily_start = DAILY_PARTITIONS.get_first_partition_key()
if _daily_start is None:
    raise ValueError("News partitions must define a start date")
DAILY_ASSETS_START = date.fromisoformat(_daily_start)
feed_poll_job = dg.define_asset_job("feed_poll", selection=dg.AssetSelection.assets(feed_intake))
youtube_source_job = dg.define_asset_job(
    "youtube_source_job", selection=dg.AssetSelection.assets(youtube_source)
)
youtube_publication_job = dg.define_asset_job(
    "youtube_publication_job", selection=dg.AssetSelection.assets(youtube_publication)
)
youtube_relevance_job = dg.define_asset_job(
    "youtube_relevance", selection=dg.AssetSelection.assets(relevance)
)
weekly_report_job = dg.define_asset_job(
    "weekly_report",
    selection=dg.AssetSelection.assets(weekly_reports),
)
daily_report_repair = dg.define_asset_job(
    "daily_report_repair",
    selection=dg.AssetSelection.assets(daily_reports),
)
article_batch_job = dg.define_asset_job(
    "article_batch",
    selection=dg.AssetSelection.assets(articles),
    run_tags={
        "dagster/max_runtime": "600",
        "dagster/max_retries": "1",
        "dagster/retry_on_asset_or_op_failure": "false",
    },
)
weekly_backfill_job = dg.define_asset_job("weekly_backfill", selection=WEEKLY_ASSETS)
NONTERMINAL_RUN_STATUSES = (
    dg.DagsterRunStatus.QUEUED,
    dg.DagsterRunStatus.NOT_STARTED,
    dg.DagsterRunStatus.MANAGED,
    dg.DagsterRunStatus.STARTING,
    dg.DagsterRunStatus.STARTED,
    dg.DagsterRunStatus.CANCELING,
)


@dg.schedule(
    job=feed_poll_job,
    cron_schedule="0 * * * *",
    execution_timezone=BUCHAREST_TIMEZONE,
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def hourly_registered_feed_poll(context: dg.ScheduleEvaluationContext) -> dg.RunRequest:
    scheduled_at = _scheduled_time(context)
    return dg.RunRequest(
        run_key=f"feed-poll:{scheduled_at.isoformat()}",
        partition_key=hourly_partition_key(scheduled_at),
        tags={"news/scheduled_at": scheduled_at.isoformat()},
    )


@dg.schedule(
    job=youtube_source_job,
    cron_schedule="*/10 * * * *",
    execution_timezone=BUCHAREST_TIMEZONE,
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def youtube_source_poll(
    context: dg.ScheduleEvaluationContext,
) -> list[dg.RunRequest] | dg.SkipReason:
    scheduled_at = _scheduled_time(context)
    requests = []
    for source in YOUTUBE_SOURCES.sources:
        active = context.instance.get_runs(
            filters=dg.RunsFilter(
                job_name=youtube_source_job.name,
                statuses=NONTERMINAL_RUN_STATUSES,
                tags={"news/youtube_source_id": source.source_id},
            ),
            limit=1,
        )
        if active:
            continue
        requests.append(
            dg.RunRequest(
                run_key=f"youtube-source:{source.source_id}:{scheduled_at.isoformat()}",
                tags={
                    "news/youtube_source_id": source.source_id,
                    "news/scheduled_at": scheduled_at.isoformat(),
                },
            )
        )
    return requests or dg.SkipReason("Every YouTube source already has a queued or active run.")


@dg.schedule(
    job=youtube_publication_job,
    cron_schedule="*/10 * * * *",
    execution_timezone=BUCHAREST_TIMEZONE,
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def youtube_approved_publication(context: dg.ScheduleEvaluationContext) -> dg.RunRequest:
    scheduled_at = _scheduled_time(context)
    return dg.RunRequest(run_key=f"youtube-publication:{scheduled_at.isoformat()}")


@dg.asset_sensor(
    asset_key=youtube_publication.key,
    job=youtube_relevance_job,
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def youtube_relevance_controller(
    context: dg.SensorEvaluationContext,
    asset_event: dg.EventLogEntry,
) -> dg.RunRequest | dg.SkipReason:
    publication_day, article_version_id = _youtube_publication(asset_event)
    tags = {"news/youtube_article_version_id": article_version_id}
    active = context.instance.get_runs(
        filters=dg.RunsFilter(
            job_name=youtube_relevance_job.name,
            statuses=NONTERMINAL_RUN_STATUSES,
            tags=tags,
        ),
        limit=1,
    )
    if active:
        return dg.SkipReason("Relevance is already queued or active for this YouTube article.")
    storage_id = _asset_event_storage_id(context, asset_event)
    return dg.RunRequest(
        run_key=f"youtube-relevance:{article_version_id}:{storage_id}",
        partition_key=publication_day,
        tags=tags,
    )


def _asset_event_storage_id(
    context: dg.SensorEvaluationContext, asset_event: dg.EventLogEntry
) -> int:
    after_storage_id = int(context.cursor) if context.cursor is not None else None
    records = context.instance.fetch_materializations(
        dg.AssetRecordsFilter(
            asset_key=youtube_publication.key,
            after_storage_id=after_storage_id,
        ),
        limit=100,
    ).records
    matching = [record.storage_id for record in records if record.event_log_entry == asset_event]
    if len(matching) != 1:
        raise ValueError("YouTube publication materialization storage ID is unavailable")
    return matching[0]


def _youtube_publication_day(asset_event: dg.EventLogEntry) -> str:
    return _youtube_publication(asset_event)[0]


def _youtube_publication(asset_event: dg.EventLogEntry) -> tuple[str, str]:
    materialization = asset_event.asset_materialization
    if materialization is None:
        raise ValueError("YouTube relevance requires an asset materialization")
    value = materialization.metadata.get("bucharest_day")
    if not isinstance(value, dg.TextMetadataValue) or value.text is None:
        raise ValueError("YouTube materialization requires Bucharest publication day metadata")
    article = materialization.metadata.get("article_version_id")
    if not isinstance(article, dg.TextMetadataValue) or article.text is None:
        raise ValueError("YouTube materialization requires article version metadata")
    return date.fromisoformat(value.text).isoformat(), article.text


ARTICLE_BATCH_SIZE = 10
ARTICLE_ASSET_KEY = dg.AssetKey("articles")


@dg.sensor(
    job=article_batch_job,
    minimum_interval_seconds=60,
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def article_batch_controller(
    context: dg.SensorEvaluationContext,
) -> dg.RunRequest | dg.SkipReason:
    active = _active_article_batch_work(context.instance)
    if active is None:
        return dg.SkipReason("Legacy article automation is queued or active.")
    now = _controller_time()
    today = now.astimezone(BUCHAREST).date()
    historical_newest_first = tuple(
        day
        for day in reversed(read_article_candidate_days(DAILY_ASSETS_START, today))
        if day != today
    )
    current_streak = _current_batch_streak(context.cursor)
    day_groups = (
        ((today,), historical_newest_first)
        if current_streak < 4
        else (
            historical_newest_first,
            (today,),
        )
    )
    for days in day_groups:
        for day in days:
            if day.isoformat() in active:
                continue
            request = _article_batch_request(context, day, now)
            if request is None:
                continue
            next_streak = min(4, current_streak + 1) if day == today else 0
            if next_streak != current_streak:
                context.update_cursor(str(next_streak))
            return request
    return dg.SkipReason("No article work is ready.")


def _active_article_batch_work(instance: dg.DagsterInstance) -> set[str] | None:
    """Active batch partitions, or None when unscoped automation blocks every day.

    Day windows partition feed events by publication time, so batches for
    different days claim disjoint events and may run concurrently.
    """
    partitions: set[str] = set()
    for run in instance.get_runs(
        filters=dg.RunsFilter(job_name="article_batch", statuses=NONTERMINAL_RUN_STATUSES),
    ):
        partition = run.tags.get("dagster/partition")
        if partition is None:
            return None
        partitions.add(partition)
    legacy = instance.get_runs(
        filters=dg.RunsFilter(
            job_name="__ASSET_JOB",
            statuses=NONTERMINAL_RUN_STATUSES,
            tags={"dagster/auto_materialize": "true"},
        ),
    )
    if any(
        run.asset_selection is None or ARTICLE_ASSET_KEY in run.asset_selection for run in legacy
    ):
        return None
    return partitions


def _article_batch_request(
    context: dg.SensorEvaluationContext,
    day: date,
    now: datetime,
) -> dg.RunRequest | None:
    start, end = bucharest_day_window(day)
    plan = plan_article_work(
        feed_registry(),
        implementation_ref=IMPLEMENTATION_REF,
        now=now,
        start_at=start,
        end_at=end,
        limit=ARTICLE_BATCH_SIZE,
        revalidate_before=(
            now - timedelta(hours=24) if day == now.astimezone(BUCHAREST).date() else None
        ),
    )
    event_ids = tuple(work.source.event_id for work in plan.selected)
    references = read_daily_article_references(day)
    attempts = read_article_attempt_states(
        {work.source.event_id: work.work_generation for work in plan.selected},
    )
    state = json.dumps(
        {
            "implementation_ref": IMPLEMENTATION_REF,
            "day": day.isoformat(),
            "event_ids": event_ids,
            "attempts": tuple(
                (
                    event_id,
                    attempt.deterministic_fingerprint,
                    attempt.unchanged_deterministic_attempts,
                    attempt.retry_at.isoformat(),
                )
                for event_id, attempt in sorted(attempts.items())
            ),
            "deferred": plan.deferred_event_ids,
            "quarantined": plan.quarantined_event_ids,
            "versions": tuple(reference.version_id for reference in references.values),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    batch_key = hashlib.sha256(state.encode()).hexdigest()
    reported_statuses = [dg.DagsterRunStatus.SUCCESS]
    if not event_ids and plan.quarantined_event_ids:
        reported_statuses.append(dg.DagsterRunStatus.FAILURE)
    reported = context.instance.get_runs(
        filters=dg.RunsFilter(
            job_name="article_batch",
            statuses=reported_statuses,
            tags={"news/article_batch_key": batch_key},
        ),
        limit=1,
    )
    today = now.astimezone(BUCHAREST).date()
    waiting_for_coverage = day not in plan.source_covered_days and day >= today
    if reported or (not event_ids and (plan.remaining_entries or waiting_for_coverage)):
        return None
    recovery_slot = now.replace(second=0, microsecond=0).isoformat()
    return dg.RunRequest(
        run_key=f"article-batch:{batch_key}:{recovery_slot}",
        partition_key=day.isoformat(),
        tags={
            "news/article_batch_key": batch_key,
            "news/article_event_ids": json.dumps(event_ids, separators=(",", ":")),
        },
    )


def _current_batch_streak(cursor: str | None) -> int:
    try:
        value = int(cursor) if cursor is not None else 0
    except ValueError:
        return 0
    return value if 0 <= value <= 4 else 0


def _controller_time() -> datetime:
    return datetime.now(UTC)


news_automation = dg.AutomationConditionSensorDefinition(
    "romanian_news_automation",
    target=dg.AssetSelection.assets(
        relevance,
        jev_relevance_shadow,
        embeddings,
        daily_clusters,
        group_summaries,
        group_sentiment,
        daily_themes,
        daily_subject_assessments,
        daily_reports,
    ),
    default_status=dg.DefaultSensorStatus.RUNNING,
)

weekly_freshness = dg.AutomationConditionSensorDefinition(
    "weekly_report_freshness",
    target=dg.AssetSelection.assets(weekly_reports),
    default_status=dg.DefaultSensorStatus.RUNNING,
    minimum_interval_seconds=3600,
)


def _scheduled_time(context: dg.ScheduleEvaluationContext) -> datetime:
    if context.scheduled_execution_time is None:
        raise ValueError("Schedule execution time is required")
    return context.scheduled_execution_time


defs = dg.Definitions(
    assets=[
        feed_intake,
        youtube_source,
        youtube_publication,
        articles,
        relevance,
        jev_relevance_shadow,
        embeddings,
        daily_clusters,
        group_summaries,
        group_sentiment,
        daily_themes,
        daily_subject_assessments,
        daily_reports,
        weekly_reports,
    ],
    asset_checks=[
        daily_theme_exact_inputs,
        daily_subject_assessment_exact_inputs,
        articles_have_no_quarantined_inputs,
        daily_report_complete,
        daily_report_exact_inputs,
        weekly_report_exact_inputs,
    ],
    jobs=[
        feed_poll_job,
        youtube_source_job,
        youtube_publication_job,
        youtube_relevance_job,
        weekly_report_job,
        daily_report_repair,
        article_batch_job,
        weekly_backfill_job,
        morning_report_check,
        news_feedback_sync,
        fresh_relevance_comparison,
        fresh_theme_comparison,
        relevance_v3_evaluation,
        video_digest_job,
        video_digest_incident_monitor_job,
        jev_relevance_evaluation,
        weekly_status_refresh,
        catalog_schema_activation,
        archive_article_capture_batch,
    ],
    schedules=[
        hourly_registered_feed_poll,
        youtube_source_poll,
        youtube_approved_publication,
        daily_morning_report_check,
        quarter_hourly_news_feedback_sync,
        scheduled_video_digest,
        scheduled_video_digest_incident_monitor,
        scheduled_weekly_status,
    ],
    sensors=[
        article_batch_controller,
        youtube_relevance_controller,
        news_automation,
        weekly_freshness,
    ],
)
