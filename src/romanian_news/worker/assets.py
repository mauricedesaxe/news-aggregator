import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta

import dagster as dg
from pydantic import TypeAdapter

from romanian_news import BUCHAREST, Sha256
from romanian_news.articles.acquisition import plan_article_work
from romanian_news.artifacts import ArtifactReference
from romanian_news.config import (
    IMPLEMENTATION_REF,
    NEWS_JEV_RELEVANCE_SHADOW_ENABLED,
    NEWS_WORKSPACE,
)
from romanian_news.daily import (
    check_current_artifact_inputs,
)
from romanian_news.feeds.registry import feed_registry
from romanian_news.groups import parse_daily_cluster_set
from romanian_news.jev_relevance_shadow import materialize_jev_relevance_shadow
from romanian_news.reports import (
    read_daily_report_input,
    read_recorded_daily_report_input,
    read_weekly_report_input,
)
from romanian_news.storage import read_verified_r2_object
from romanian_news.subject_assessments import read_daily_subject_assessment_input
from romanian_news.themes import read_daily_theme_input, read_recorded_daily_theme_input
from romanian_news.worker.operations import (
    ArticleBatchResult,
    materialize_articles,
    materialize_clusters,
    materialize_daily_report,
    materialize_daily_research_triggers,
    materialize_daily_themes,
    materialize_embeddings,
    materialize_feed_intake,
    materialize_group_sentiment,
    materialize_group_summaries,
    materialize_next_youtube_publication,
    materialize_relevance,
    materialize_subject_assessments,
    materialize_weekly_report,
    materialize_youtube_source,
)

BUCHAREST_TIMEZONE = "Europe/Bucharest"
DAILY_PARTITIONS = dg.DailyPartitionsDefinition(
    start_date="2026-08-01",
    timezone=BUCHAREST_TIMEZONE,
    end_offset=1,
)
WEEKLY_PARTITIONS = dg.WeeklyPartitionsDefinition(
    start_date="2026-08-02",
    day_offset=1,
    timezone=BUCHAREST_TIMEZONE,
)
EAGER = dg.AutomationCondition.eager()
DAILY_FRESHNESS = (
    (
        dg.AutomationCondition.any_deps_updated()
        | (dg.AutomationCondition.initial_evaluation() & dg.AutomationCondition.missing())
        | (
            (dg.AutomationCondition.missing() | dg.AutomationCondition.execution_failed())
            & dg.AutomationCondition.cron_tick_passed("0 * * * *", BUCHAREST_TIMEZONE)
        )
    )
    & ~dg.AutomationCondition.any_deps_missing()
    & ~dg.AutomationCondition.any_deps_in_progress()
    & ~dg.AutomationCondition.in_progress()
)
SHADOW_FRESHNESS = (
    dg.AutomationCondition.in_latest_time_window(timedelta(days=7))
    & (
        dg.AutomationCondition.initial_evaluation()
        | dg.AutomationCondition.missing()
        | dg.AutomationCondition.any_deps_updated()
        | dg.AutomationCondition.code_version_changed()
        | dg.AutomationCondition.execution_failed()
    )
    & ~dg.AutomationCondition.any_deps_missing()
    & ~dg.AutomationCondition.any_deps_in_progress()
    & ~dg.AutomationCondition.in_progress()
)
WEEKLY_FRESHNESS = (
    (
        dg.AutomationCondition.initial_evaluation()
        | dg.AutomationCondition.missing()
        | dg.AutomationCondition.any_deps_updated()
        | dg.AutomationCondition.execution_failed()
        | dg.AutomationCondition.code_version_changed()
    )
    & ~dg.AutomationCondition.any_deps_missing()
    & ~dg.AutomationCondition.any_deps_in_progress()
    & ~dg.AutomationCondition.in_progress()
)


@dg.asset(
    partitions_def=DAILY_PARTITIONS,
    group_name="romanian_news_daily",
    pool="news_feed_network",
)
def feed_intake(context: dg.AssetExecutionContext) -> dg.MaterializeResult[object]:
    day = _partition_day(context)
    scheduled_at = _scheduled_at(context, day)
    values = materialize_feed_intake(
        day,
        scheduled_at,
        NEWS_WORKSPACE,
        IMPLEMENTATION_REF,
    )
    return _result(values.values)


@dg.asset(
    deps=[feed_intake],
    partitions_def=DAILY_PARTITIONS,
    group_name="romanian_news_daily",
    pool="news_article_network",
    output_required=False,
)
def articles(
    context: dg.AssetExecutionContext,
) -> Iterator[dg.MaterializeResult[object] | dg.AssetObservation]:
    day = _partition_day(context)
    event_ids = _article_event_ids(context)
    result = materialize_articles(
        day,
        event_ids,
        IMPLEMENTATION_REF,
        run_id=context.run.run_id,
        retry_number=context.retry_number,
    )
    metadata = _article_batch_metadata(result)
    if not result.complete or result.has_infrastructure_failures or result.quarantined_event_ids:
        yield dg.AssetObservation(
            asset_key="articles",
            partition=context.partition_key,
            metadata=metadata,
        )
    first_snapshot = bool(result.references.values) and not _article_snapshot_exists(context)
    if (
        result.acquired_event_ids
        or first_snapshot
        or (
            result.complete
            and not result.has_infrastructure_failures
            and not result.quarantined_event_ids
        )
    ):
        yield _result(result.references.values, metadata)
    if result.has_infrastructure_failures:
        failures = [
            failure.message for failure in result.failures if failure.kind.value == "infrastructure"
        ]
        raise RuntimeError("; ".join(failures))
    if result.quarantined_event_ids:
        raise dg.Failure(
            f"Article batch has {len(result.quarantined_event_ids)} quarantined input(s)",
            allow_retries=False,
        )


def _article_snapshot_exists(context: dg.AssetExecutionContext) -> bool:
    records = context.instance.fetch_materializations(
        dg.AssetRecordsFilter(
            asset_key=dg.AssetKey("articles"),
            asset_partitions=[context.partition_key],
        ),
        limit=1,
    ).records
    return bool(records)


@dg.asset(group_name="romanian_news_youtube", pool="news_youtube_network", output_required=False)
def youtube_source(
    context: dg.AssetExecutionContext,
) -> Iterator[dg.MaterializeResult[object] | dg.AssetObservation]:
    source_id = context.run.tags.get("news/youtube_source_id")
    scheduled = context.run.tags.get("news/scheduled_at")
    if source_id is None or scheduled is None:
        raise ValueError("YouTube source runs require source and schedule tags")
    result = materialize_youtube_source(
        source_id, datetime.fromisoformat(scheduled), IMPLEMENTATION_REF
    )
    metadata: dict[str, str | int | bool] = {
        "source_id": result.source_id,
        "poll_status": result.poll_status.value,
        "candidate_created": result.candidate_version_id is not None,
    }
    if result.candidate_version_id is not None:
        metadata["candidate_version_id"] = result.candidate_version_id
        yield _youtube_result(metadata)
    else:
        yield dg.AssetObservation(asset_key="youtube_source", metadata=metadata)


@dg.asset(group_name="romanian_news_youtube", pool="news_catalog", output_required=False)
def youtube_publication() -> Iterator[dg.MaterializeResult[object] | dg.AssetObservation]:
    publication = materialize_next_youtube_publication(IMPLEMENTATION_REF)
    if publication is None:
        yield dg.AssetObservation(asset_key="youtube_publication", metadata={"published": False})
        return
    metadata: dict[str, str | int | bool] = {
        "published": True,
        "source_id": publication.source_id,
        "video_id": publication.video_id,
        "candidate_version_id": publication.candidate_version_id,
        "article_version_id": publication.article_version_id,
        "bucharest_day": publication.bucharest_day.isoformat(),
        "accepted_clip_count": publication.accepted_clip_count,
    }
    yield _youtube_result(metadata)


def _youtube_result(metadata: dict[str, str | int | bool]) -> dg.MaterializeResult[object]:
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return dg.MaterializeResult(metadata=metadata, data_version=dg.DataVersion(digest))


@dg.asset(
    deps=[articles],
    partitions_def=DAILY_PARTITIONS,
    group_name="romanian_news_daily",
    pool="news_model",
    automation_condition=DAILY_FRESHNESS,
)
def relevance(context: dg.AssetExecutionContext) -> dg.MaterializeResult[object]:
    return _result(materialize_relevance(_partition_day(context), IMPLEMENTATION_REF).values)


@dg.asset(
    deps=[relevance],
    partitions_def=DAILY_PARTITIONS,
    group_name="romanian_news_daily",
    pool="news_model",
    code_version=f"jev-shadow-{'on' if NEWS_JEV_RELEVANCE_SHADOW_ENABLED else 'off'}",
    automation_condition=SHADOW_FRESHNESS,
    check_specs=[dg.AssetCheckSpec(name="paired_accounting", asset="jev_relevance_shadow")],
)
def jev_relevance_shadow(
    context: dg.AssetExecutionContext,
) -> Iterator[dg.MaterializeResult[object] | dg.AssetCheckResult]:
    result = materialize_jev_relevance_shadow(_partition_day(context))
    metadata = result.model_dump(mode="json")
    yield dg.MaterializeResult(metadata=metadata)
    yield dg.AssetCheckResult(
        passed=not (result.failed or result.unresolved or result.missing_incumbent),
        check_name="paired_accounting",
        metadata=metadata,
    )


@dg.asset(
    deps=[relevance],
    partitions_def=DAILY_PARTITIONS,
    group_name="romanian_news_daily",
    pool="news_model",
    automation_condition=DAILY_FRESHNESS,
)
def embeddings(context: dg.AssetExecutionContext) -> dg.MaterializeResult[object]:
    return _result(materialize_embeddings(_partition_day(context), IMPLEMENTATION_REF).values)


@dg.asset(
    deps=[embeddings],
    partitions_def=DAILY_PARTITIONS,
    group_name="romanian_news_daily",
    pool="news_catalog",
    automation_condition=DAILY_FRESHNESS,
)
def daily_clusters(context: dg.AssetExecutionContext) -> dg.MaterializeResult[object]:
    return _result(materialize_clusters(_partition_day(context), IMPLEMENTATION_REF).values)


@dg.asset(
    deps=[daily_clusters],
    partitions_def=DAILY_PARTITIONS,
    group_name="romanian_news_daily",
    pool="news_model",
    automation_condition=DAILY_FRESHNESS,
)
def group_summaries(context: dg.AssetExecutionContext) -> dg.MaterializeResult[object]:
    return _result(materialize_group_summaries(_partition_day(context), IMPLEMENTATION_REF).values)


@dg.asset(
    deps=[daily_clusters],
    partitions_def=DAILY_PARTITIONS,
    group_name="romanian_news_daily",
    pool="news_model",
    automation_condition=DAILY_FRESHNESS,
)
def group_sentiment(context: dg.AssetExecutionContext) -> dg.MaterializeResult[object]:
    return _result(materialize_group_sentiment(_partition_day(context), IMPLEMENTATION_REF).values)


@dg.asset(
    deps=[daily_clusters, group_summaries],
    partitions_def=DAILY_PARTITIONS,
    group_name="romanian_news_daily",
    pool="news_model",
    automation_condition=DAILY_FRESHNESS,
)
def daily_themes(context: dg.AssetExecutionContext) -> dg.MaterializeResult[object]:
    return _result(materialize_daily_themes(_partition_day(context), IMPLEMENTATION_REF).values)


@dg.asset(
    deps=[daily_themes, relevance],
    partitions_def=DAILY_PARTITIONS,
    group_name="romanian_news_daily",
    pool="news_model",
    automation_condition=DAILY_FRESHNESS,
)
def daily_subject_assessments(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult[object]:
    return _result(
        materialize_subject_assessments(_partition_day(context), IMPLEMENTATION_REF).values
    )


@dg.asset(
    deps=[daily_subject_assessments, group_sentiment],
    partitions_def=DAILY_PARTITIONS,
    group_name="romanian_news_daily",
    pool="news_catalog",
    automation_condition=DAILY_FRESHNESS,
)
def daily_reports(context: dg.AssetExecutionContext) -> dg.MaterializeResult[object]:
    return _result(materialize_daily_report(_partition_day(context), IMPLEMENTATION_REF).values)


@dg.asset(
    deps=[daily_reports],
    partitions_def=DAILY_PARTITIONS,
    group_name="romanian_news_daily",
    pool="news_model",
    automation_condition=EAGER,
)
def daily_research_triggers(context: dg.AssetExecutionContext) -> dg.MaterializeResult[object]:
    return _result(
        materialize_daily_research_triggers(_partition_day(context), IMPLEMENTATION_REF).values
    )


@dg.asset(
    deps=[
        dg.AssetDep(
            daily_reports,
            partition_mapping=dg.TimeWindowPartitionMapping(),
        )
    ],
    partitions_def=WEEKLY_PARTITIONS,
    group_name="romanian_news_weekly",
    pool="news_catalog",
    code_version=IMPLEMENTATION_REF,
    automation_condition=WEEKLY_FRESHNESS,
)
def weekly_reports(context: dg.AssetExecutionContext) -> dg.MaterializeResult[object]:
    return _result(materialize_weekly_report(_partition_day(context), IMPLEMENTATION_REF))


@dg.asset_check(asset=articles, name="no_quarantined_inputs")
def articles_have_no_quarantined_inputs(
    context: dg.AssetCheckExecutionContext,
) -> dg.AssetCheckResult:
    day = _partition_day(context)
    start, end = DAILY_PARTITIONS.time_window_for_partition_key(context.partition_key)
    now = datetime.now(UTC)
    plan = plan_article_work(
        feed_registry(),
        implementation_ref=IMPLEMENTATION_REF,
        now=now,
        start_at=start,
        end_at=end,
        limit=10,
        revalidate_before=(
            now - timedelta(hours=24) if day == now.astimezone(BUCHAREST).date() else None
        ),
    )
    return dg.AssetCheckResult(
        passed=not plan.quarantined_event_ids,
        metadata={
            "quarantined_count": len(plan.quarantined_event_ids),
            "quarantined_event_ids": dg.MetadataValue.json(plan.quarantined_event_ids),
        },
    )


@dg.asset_check(asset=daily_themes, name="exact_inputs", blocking=True)
def daily_theme_exact_inputs(context: dg.AssetCheckExecutionContext) -> dg.AssetCheckResult:
    day = date.fromisoformat(context.partition_key)
    try:
        value = read_daily_theme_input(day)
        expected = (value.cluster_set, *(item.summary for item in value.groups))
        check = check_current_artifact_inputs(
            f"news:themes:{day.isoformat()}",
            expected,
        )
        return dg.AssetCheckResult(
            passed=check.passed,
            metadata={
                "expected_version_ids": dg.MetadataValue.json(check.expected_version_ids),
                "recorded_version_ids": dg.MetadataValue.json(check.recorded_version_ids),
            },
        )
    except ValueError as error:
        return dg.AssetCheckResult(passed=False, metadata={"error": str(error)})


@dg.asset_check(asset=daily_subject_assessments, name="exact_inputs", blocking=True)
def daily_subject_assessment_exact_inputs(
    context: dg.AssetCheckExecutionContext,
) -> dg.AssetCheckResult:
    day = date.fromisoformat(context.partition_key)
    try:
        value = read_daily_subject_assessment_input(day)
        expected = (
            value.themes,
            *(item.reference for item in value.summaries),
            *value.relevance,
        )
        check = check_current_artifact_inputs(
            f"news:subject-assessments:{day.isoformat()}",
            expected,
        )
        return dg.AssetCheckResult(
            passed=check.passed,
            metadata={
                "expected_version_ids": dg.MetadataValue.json(check.expected_version_ids),
                "recorded_version_ids": dg.MetadataValue.json(check.recorded_version_ids),
            },
        )
    except ValueError as error:
        return dg.AssetCheckResult(passed=False, metadata={"error": str(error)})


def daily_report_completeness_result(partition_key: str) -> dg.AssetCheckResult:
    day = date.fromisoformat(partition_key)
    try:
        report_input = read_recorded_daily_report_input(day)
        theme_input = read_recorded_daily_theme_input(day)
        cluster_reference = report_input.cluster_set
        cluster_set = parse_daily_cluster_set(
            read_verified_r2_object(
                cluster_reference.r2_key,
                cluster_reference.content_digest,
            )
        )
        summaries = report_input.summaries
        sentiments = report_input.sentiments
        grouped_article_ids = tuple(
            article_id for group in cluster_set.groups for article_id in group.article_version_ids
        )
        passed = (
            len(cluster_set.article_version_ids)
            == len(cluster_set.relevance_version_ids)
            == len(cluster_set.embedding_version_ids)
            and len(cluster_set.article_version_ids) == len(set(cluster_set.article_version_ids))
            and len(cluster_set.relevance_version_ids)
            == len(set(cluster_set.relevance_version_ids))
            and len(cluster_set.embedding_version_ids)
            == len(set(cluster_set.embedding_version_ids))
            and sorted(grouped_article_ids) == sorted(cluster_set.article_version_ids)
            and theme_input.cluster_set == report_input.cluster_set
            and tuple(item.summary for item in theme_input.groups) == summaries
            and len(summaries) == len(cluster_set.groups)
            and len(sentiments) == len(cluster_set.groups)
        )
        return dg.AssetCheckResult(
            passed=passed,
            metadata={
                "articles": len(cluster_set.article_version_ids),
                "relevance": len(cluster_set.relevance_version_ids),
                "embeddings": len(cluster_set.embedding_version_ids),
                "groups": len(cluster_set.groups),
                "summaries": len(summaries),
                "sentiments": len(sentiments),
            },
        )
    except ValueError as error:
        return dg.AssetCheckResult(passed=False, metadata={"error": str(error)})


@dg.asset_check(asset=daily_reports, name="complete", blocking=True)
def daily_report_complete(context: dg.AssetCheckExecutionContext) -> dg.AssetCheckResult:
    return daily_report_completeness_result(context.partition_key)


@dg.asset_check(asset=daily_reports, name="exact_inputs", blocking=True)
def daily_report_exact_inputs(context: dg.AssetCheckExecutionContext) -> dg.AssetCheckResult:
    day = date.fromisoformat(context.partition_key)
    try:
        value = read_daily_report_input(day)
        expected = [value.themes, value.assessments, value.cluster_set]
        for summary, sentiment in zip(value.summaries, value.sentiments, strict=True):
            expected.extend((summary, sentiment))
        check = check_current_artifact_inputs(
            f"news:daily:{day.isoformat()}",
            tuple(expected),
        )
        return dg.AssetCheckResult(
            passed=check.passed,
            metadata={
                "expected_version_ids": dg.MetadataValue.json(check.expected_version_ids),
                "recorded_version_ids": dg.MetadataValue.json(check.recorded_version_ids),
            },
        )
    except ValueError as error:
        return dg.AssetCheckResult(passed=False, metadata={"error": str(error)})


@dg.asset_check(asset=weekly_reports, name="exact_inputs", blocking=True)
def weekly_report_exact_inputs(context: dg.AssetCheckExecutionContext) -> dg.AssetCheckResult:
    week_start = date.fromisoformat(context.partition_key)
    try:
        value = read_weekly_report_input(week_start)
        check = check_current_artifact_inputs(
            f"news:weekly:{week_start.isoformat()}",
            value.daily_reports,
        )
        return dg.AssetCheckResult(
            passed=check.passed,
            metadata={
                "expected_version_ids": dg.MetadataValue.json(check.expected_version_ids),
                "recorded_version_ids": dg.MetadataValue.json(check.recorded_version_ids),
            },
        )
    except ValueError as error:
        return dg.AssetCheckResult(passed=False, metadata={"error": str(error)})


def _partition_day(
    context: dg.AssetExecutionContext | dg.AssetCheckExecutionContext,
) -> date:
    return date.fromisoformat(context.partition_key)


def _scheduled_at(context: dg.AssetExecutionContext, day: date) -> datetime:
    value = context.run.tags.get("news/scheduled_at")
    if value is not None:
        scheduled_at = datetime.fromisoformat(value).astimezone(BUCHAREST)
        if scheduled_at.date() != day:
            raise ValueError("Feed schedule time does not match the daily partition")
        return scheduled_at
    return datetime.combine(day, time.min, tzinfo=BUCHAREST)


def _article_event_ids(context: dg.AssetExecutionContext) -> tuple[Sha256, ...]:
    raw = context.run.tags.get("news/article_event_ids")
    if raw is None:
        raise ValueError("Article runs require exact event IDs")
    event_ids = TypeAdapter(tuple[Sha256, ...]).validate_json(raw)
    if len(event_ids) > 10 or len(event_ids) != len(set(event_ids)):
        raise ValueError("Article runs require at most 10 unique event IDs")
    return event_ids


def _article_batch_metadata(
    result: ArticleBatchResult,
) -> dict[str, dg.MetadataValue | int]:
    return {
        "requested_count": len(result.requested_event_ids),
        "acquired_count": len(result.acquired_event_ids),
        "skipped_completed_count": len(result.skipped_event_ids),
        "failure_count": len(result.failures),
        "failures": dg.MetadataValue.json(
            tuple(failure.model_dump(mode="json") for failure in result.failures)
        ),
        "remaining_retryable_count": result.remaining_entries,
        "deferred_count": len(result.deferred_event_ids),
        "quarantined_count": len(result.quarantined_event_ids),
        "source_covered": result.source_covered,
        "quarantined_event_ids": dg.MetadataValue.json(result.quarantined_event_ids),
    }


def _result(
    references: tuple[ArtifactReference, ...],
    extra_metadata: dict[str, dg.MetadataValue | int] | None = None,
) -> dg.MaterializeResult[object]:
    version_ids = tuple(value.version_id for value in references)
    digest = hashlib.sha256(json.dumps(version_ids, separators=(",", ":")).encode()).hexdigest()
    metadata: dict[str, dg.MetadataValue | int] = {
        "artifact_count": len(references),
        "version_ids": dg.MetadataValue.json(version_ids),
    }
    metadata.update(extra_metadata or {})
    return dg.MaterializeResult[object](
        metadata=metadata,
        data_version=dg.DataVersion(digest),
    )


def hourly_partition_key(scheduled_at: datetime) -> str:
    return scheduled_at.astimezone(BUCHAREST).date().isoformat()
