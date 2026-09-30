"""Bounded repair of daily asset partitions outside live automation's window."""

import hashlib
from datetime import UTC, date, datetime, timedelta

import dagster as dg

from romanian_news import BUCHAREST
from romanian_news.current_report import read_daily_report_freshness
from romanian_news.worker.assets import (
    articles,
    daily_clusters,
    daily_reports,
    daily_subject_assessments,
    daily_themes,
    embeddings,
    group_sentiment,
    group_summaries,
    relevance,
)

STAGES = (
    (relevance.key, (articles.key,)),
    (embeddings.key, (relevance.key,)),
    (daily_clusters.key, (embeddings.key,)),
    (group_summaries.key, (daily_clusters.key,)),
    (group_sentiment.key, (daily_clusters.key,)),
    (daily_themes.key, (daily_clusters.key, group_summaries.key)),
    (daily_subject_assessments.key, (daily_themes.key, relevance.key)),
    (daily_reports.key, (daily_subject_assessments.key, group_sentiment.key)),
)
MAX_DAYS_PER_TICK = 5
ACTIVE_STATUSES = (
    dg.DagsterRunStatus.QUEUED,
    dg.DagsterRunStatus.NOT_STARTED,
    dg.DagsterRunStatus.MANAGED,
    dg.DagsterRunStatus.STARTING,
    dg.DagsterRunStatus.STARTED,
    dg.DagsterRunStatus.CANCELING,
)

historical_daily_recovery_job = dg.define_asset_job(
    "historical_daily_recovery_job",
    selection=dg.AssetSelection.assets(*(stage[0] for stage in STAGES)),
)


@dg.sensor(
    job=historical_daily_recovery_job,
    minimum_interval_seconds=60,
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def historical_daily_recovery(
    context: dg.SensorEvaluationContext,
) -> dg.RunRequest | dg.SkipReason:
    if context.instance.get_runs(
        filters=dg.RunsFilter(
            job_name=historical_daily_recovery_job.name,
            statuses=ACTIVE_STATUSES,
        ),
        limit=1,
    ):
        return dg.SkipReason("A historical daily repair is queued or active.")
    now = _recovery_time()
    today = now.astimezone(BUCHAREST).date()
    recent_start = today - timedelta(days=2)
    source_days = context.instance.get_materialized_partitions(articles.key)
    days = sorted(day for day in source_days if date.fromisoformat(day) < recent_start)
    if not days:
        return dg.SkipReason("No historical article partitions need inspection.")

    cursor = context.cursor or ""
    after = [day for day in days if day > cursor]
    before = [day for day in days if day <= cursor]
    for day in (after + before)[:MAX_DAYS_PER_TICK]:
        context.update_cursor(day)
        if _day_has_active_run(context.instance, day):
            continue
        request = _repair_request(context.instance, day, now)
        if request is not None:
            return request
    return dg.SkipReason("No historical daily stage is ready in this scan.")


def _recovery_time() -> datetime:
    return datetime.now(UTC)


def _day_has_active_run(instance: dg.DagsterInstance, day: str) -> bool:
    for job in (historical_daily_recovery_job.name, "__ASSET_JOB", "article_batch"):
        if instance.get_runs(
            filters=dg.RunsFilter(
                job_name=job,
                statuses=ACTIVE_STATUSES,
                tags={"dagster/partition": day},
            ),
            limit=1,
        ):
            return True
    return False


def _repair_request(instance: dg.DagsterInstance, day: str, now: datetime) -> dg.RunRequest | None:
    materializations: dict[dg.AssetKey, dg.EventLogRecord | None] = {}

    def latest(key: dg.AssetKey) -> dg.EventLogRecord | None:
        if key not in materializations:
            records = instance.fetch_materializations(
                dg.AssetRecordsFilter(asset_key=key, asset_partitions=[day]),
                limit=1,
            ).records
            materializations[key] = records[0] if records else None
        return materializations[key]

    for key, parents in STAGES:
        parent_records = {parent: latest(parent) for parent in parents}
        if any(record is None for record in parent_records.values()):
            return None
        child = latest(key)
        if key == daily_reports.key:
            try:
                freshness = read_daily_report_freshness(date.fromisoformat(day)).kind
            except ValueError:
                freshness = "stale"
            if freshness == "inputs_not_ready":
                return None
            if (
                child is not None
                and freshness == "fresh"
                and _matches_inputs(child, parent_records, instance)
            ):
                continue
        elif child is not None and _matches_inputs(child, parent_records, instance):
            continue
        generation = _generation(day, key, parent_records)
        prior = instance.get_run_records(
            filters=dg.RunsFilter(
                job_name=historical_daily_recovery_job.name,
                tags={"news/recovery_generation": generation},
            ),
            limit=32,
        )
        if prior:
            latest_run = prior[0]
            if latest_run.dagster_run.status in ACTIVE_STATUSES:
                return None
            finished = latest_run.end_time or latest_run.create_timestamp.timestamp()
            delay = min(21600, 900 * 2 ** min(len(prior) - 1, 5))
            if now.timestamp() - finished < delay:
                return None
        attempt = prior[0].dagster_run.run_id if prior else "initial"
        return dg.RunRequest(
            run_key=f"historical-daily:{day}:{key.to_user_string()}:{generation}:{attempt}",
            partition_key=day,
            asset_selection=[key],
            tags={
                "dagster/priority": "-10",
                "news/recovery_day": day,
                "news/recovery_stage": key.to_user_string(),
                "news/recovery_generation": generation,
            },
        )
    return None


def _matches_inputs(
    child: dg.EventLogRecord,
    parents: dict[dg.AssetKey, dg.EventLogRecord | None],
    instance: dg.DagsterInstance,
) -> bool:
    tags = child.event_log_entry.tags or {}
    pointers = {
        parent: tags.get(f"dagster/input_event_pointer/{parent.to_user_string()}")
        for parent in parents
    }
    if all(pointer is not None for pointer in pointers.values()):
        return all(
            pointer == str(record.storage_id)
            for parent, pointer in pointers.items()
            if (record := parents[parent]) is not None
        )

    run = instance.get_run_record_by_id(child.event_log_entry.run_id)
    if run is None or run.start_time is None:
        return False
    return all(
        record.event_log_entry.timestamp < run.start_time
        for record in parents.values()
        if record is not None
    )


def _generation(
    day: str, key: dg.AssetKey, parents: dict[dg.AssetKey, dg.EventLogRecord | None]
) -> str:
    state = f"{day}:{key.to_user_string()}:" + ":".join(
        f"{parent.to_user_string()}={record.storage_id}"
        for parent, record in parents.items()
        if record is not None
    )
    return hashlib.sha256(state.encode()).hexdigest()[:20]
