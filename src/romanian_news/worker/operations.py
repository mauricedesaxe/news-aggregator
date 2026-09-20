from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from romanian_news import BUCHAREST, Sha256
from romanian_news.alerts import ping_heartbeat
from romanian_news.analysis.embeddings import (
    embed_article,
    load_embedding_input,
    read_pending_embedding_references,
)
from romanian_news.analysis.groups.pending import (
    load_group_analysis_input,
    read_pending_group_analysis_references,
)
from romanian_news.analysis.groups.workflow import analyze_group
from romanian_news.analysis.relevance import (
    load_article_analysis_input,
    read_pending_relevance_references,
)
from romanian_news.analysis.relevance_v3 import (
    analyze_relevance_v3,
    production_relevance_v3_request_id,
)
from romanian_news.analysis.tracing import flush_langfuse_traces
from romanian_news.articles.acquisition import (
    acquire_article_batch_item,
    load_exact_article_work,
    read_article_work_status,
)
from romanian_news.articles.extraction import normalize_article_url
from romanian_news.articles.models import (
    ArticleAcquisitionFailure,
    ArticleAcquisitionResult,
    ArticleBatchCapture,
    ArticleBatchSkip,
    ArticleFailureKind,
    ArticleWorkItem,
)
from romanian_news.articles.recovery import (
    article_work_generation,
    record_article_failure_attempts,
)
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.analysis import (
    publish_embedding_outputs,
    publish_group_analysis_outputs,
    publish_relevance_outputs,
)
from romanian_news.catalog.articles import publish_articles
from romanian_news.catalog.clusters import publish_daily_clusters
from romanian_news.catalog.feeds import (
    publish_feed_acquisition,
    read_successful_feed_observation_count,
)
from romanian_news.catalog.reports import publish_weekly_report
from romanian_news.catalog.research_triggers import (
    publish_daily_research_triggers,
    read_completed_research_trigger,
    read_daily_research_trigger_reference,
)
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog.subject_assessments import (
    publish_daily_subject_assessments,
    read_completed_subject_assessment,
    read_daily_subject_assessment_reference,
)
from romanian_news.catalog.themes import publish_daily_themes, read_daily_theme_reference
from romanian_news.catalog_transport import catalog_integrity_identity
from romanian_news.current_report import build_and_publish_current_daily_report
from romanian_news.daily import (
    DailyArtifactReferences,
    bucharest_day_window,
    read_daily_article_references,
    read_daily_cluster_reference,
    read_daily_embedding_references,
    read_daily_feed_observation_references,
    read_daily_group_sentiment_references,
    read_daily_group_summary_references,
    read_daily_relevance_references,
    read_daily_report_reference,
    read_weekly_report_reference,
)
from romanian_news.feeds.acquisition import acquire_feeds
from romanian_news.feeds.models import FeedAcquisitionRequest, FeedRegistry
from romanian_news.feeds.recovery import reconcile_feed_entry_projection
from romanian_news.feeds.registry import feed_registry
from romanian_news.groups import cluster_articles, read_embedded_articles
from romanian_news.reports import (
    build_weekly_report_from_input,
    read_weekly_report_input,
)
from romanian_news.research_triggers import (
    construct_daily_research_triggers,
    read_daily_research_trigger_input,
    research_trigger_request_id,
    research_trigger_run_id,
)
from romanian_news.subject_assessments import (
    construct_daily_subject_assessments,
    read_daily_subject_assessment_input,
    subject_assessment_request_id,
    subject_assessment_run_id,
)
from romanian_news.themes import construct_daily_themes, read_daily_theme_input
from romanian_news.youtube.models import (
    YOUTUBE_SOURCES,
    YouTubePublicationResult,
    YouTubeSourceResult,
)
from romanian_news.youtube.workflow import (
    materialize_next_approved_publication as run_next_youtube_publication,
)
from romanian_news.youtube.workflow import (
    materialize_youtube_source as run_youtube_source,
)


@dataclass(frozen=True)
class ArticleBatchResult:
    references: DailyArtifactReferences
    requested_event_ids: tuple[Sha256, ...]
    acquired_event_ids: tuple[Sha256, ...]
    skipped_event_ids: tuple[Sha256, ...]
    failures: tuple[ArticleAcquisitionFailure, ...]
    remaining_entries: int
    deferred_event_ids: tuple[Sha256, ...]
    quarantined_event_ids: tuple[Sha256, ...]
    source_covered: bool

    @property
    def has_infrastructure_failures(self) -> bool:
        return any(failure.kind == ArticleFailureKind.INFRASTRUCTURE for failure in self.failures)

    @property
    def complete(self) -> bool:
        return self.source_covered and self.remaining_entries == 0


ARTICLE_ITEM_WORKERS = 4


def materialize_feed_intake(
    day: date,
    scheduled_at: datetime,
    workspace_dir: Path,
    implementation_ref: str,
) -> DailyArtifactReferences:
    ensure_news_catalog_schema()
    reconcile_feed_entry_projection()
    registry = feed_registry()
    existing = read_daily_feed_observation_references(day)
    if day < datetime.now(BUCHAREST).date():
        if not existing.values:
            raise ValueError(f"No recorded feed intake exists for {day.isoformat()}")
        return existing
    if read_successful_feed_observation_count(scheduled_at) < len(registry.feeds):
        result = acquire_feeds(
            FeedAcquisitionRequest(
                registry=registry,
                observed_at=datetime.now(UTC),
                scheduled_slot=scheduled_at,
                workspace_dir=workspace_dir,
                destination="r2",
            )
        )
        publication = publish_feed_acquisition(result, registry, implementation_ref)
        failed = tuple(capture.feed_id for capture in result.captures if capture.status == "failed")
        if failed:
            raise RuntimeError(f"Feed poll failed for: {', '.join(failed)}")
        if publication.feed_observations != len(registry.feeds):
            raise RuntimeError("Feed poll did not publish every registered feed")
    return read_daily_feed_observation_references(day)


def materialize_articles(
    day: date,
    event_ids: tuple[Sha256, ...],
    implementation_ref: str,
    *,
    run_id: str,
    retry_number: int,
    now: datetime | None = None,
) -> ArticleBatchResult:
    if len(event_ids) > 10 or len(event_ids) != len(set(event_ids)):
        raise ValueError("Article batches require at most 10 unique event IDs")
    ensure_news_catalog_schema()
    registry = feed_registry()
    start, end = bucharest_day_window(day)
    attempted_at = now or datetime.now(UTC)
    revalidate_before = (
        attempted_at - timedelta(hours=24)
        if day == attempted_at.astimezone(BUCHAREST).date()
        else None
    )
    work_items = load_exact_article_work(
        event_ids,
        registry,
        implementation_ref=implementation_ref,
        now=attempted_at,
        start_at=start,
        end_at=end,
        revalidate_before=revalidate_before,
    )
    batch_items, dedupe_skips = _dedupe_batch_aliases(work_items, registry)
    skipped_event_ids = list(dedupe_skips)
    captures = []
    failures = []

    def _record_attempt(failure, work):
        record_article_failure_attempts(
            (failure,),
            work_generations={
                work.source.event_id: article_work_generation(
                    work.source.event_id,
                    work.last_captured_at,
                )
            },
            run_id=run_id,
            retry_number=retry_number,
            implementation_ref=implementation_ref,
            attempted_at=attempted_at,
        )

    def process(work):
        outcome = acquire_article_batch_item(work, registry)
        if isinstance(outcome, ArticleBatchCapture):
            try:
                publish_articles(
                    ArticleAcquisitionResult(captures=(outcome.capture,), skipped_entries=0),
                    implementation_ref,
                )
            except Exception as error:
                failure = _publish_failure(outcome.capture.source.event_id, error)
                _record_attempt(failure, work)
                return failure
            return outcome
        if isinstance(outcome, ArticleBatchSkip):
            return outcome
        _record_attempt(outcome, work)
        return outcome

    with ThreadPoolExecutor(max_workers=ARTICLE_ITEM_WORKERS) as executor:
        outcomes = list(executor.map(process, batch_items))
    for outcome in outcomes:
        if isinstance(outcome, ArticleBatchCapture):
            captures.append(outcome.capture)
        elif isinstance(outcome, ArticleBatchSkip):
            skipped_event_ids.append(outcome.event_id)
        else:
            failures.append(outcome)
    final_status = read_article_work_status(
        registry,
        implementation_ref=implementation_ref,
        now=attempted_at,
        start_at=start,
        end_at=end,
        revalidate_before=revalidate_before,
    )
    return ArticleBatchResult(
        references=read_daily_article_references(day),
        requested_event_ids=event_ids,
        acquired_event_ids=tuple(capture.source.event_id for capture in captures),
        skipped_event_ids=tuple(skipped_event_ids),
        failures=tuple(failures),
        remaining_entries=final_status.retryable_entries,
        deferred_event_ids=final_status.deferred_event_ids,
        quarantined_event_ids=final_status.quarantined_event_ids,
        source_covered=day in final_status.source_covered_days,
    )


def _dedupe_batch_aliases(
    work_items: tuple[ArticleWorkItem, ...],
    registry: FeedRegistry,
) -> tuple[tuple[ArticleWorkItem, ...], tuple[Sha256, ...]]:
    """Keep one work item per article URL so parallel items cannot race a publish."""
    feeds = {feed.id: feed for feed in registry.feeds}
    seen_aliases: set[str] = set()
    unique = []
    duplicates = []
    for work in work_items:
        feed = feeds.get(work.source.feed_id)
        if feed is not None:
            alias = f"url:{normalize_article_url(str(work.source.url), feed.article_hosts)}"
            if alias in seen_aliases:
                duplicates.append(work.source.event_id)
                continue
            seen_aliases.add(alias)
        unique.append(work)
    return tuple(unique), tuple(duplicates)


def _publish_failure(event_id: Sha256, error: Exception) -> ArticleAcquisitionFailure:
    identity = catalog_integrity_identity(error)
    if identity is not None:
        kind = ArticleFailureKind.DETERMINISTIC
        fingerprint = hashlib.sha256(f"publish:{identity}".encode()).hexdigest()
    else:
        kind = ArticleFailureKind.INFRASTRUCTURE
        fingerprint = hashlib.sha256(
            f"publish:{type(error).__module__}.{type(error).__qualname__}".encode()
        ).hexdigest()
    cause: BaseException | None = error
    while cause.__cause__ is not None:
        cause = cause.__cause__
    return ArticleAcquisitionFailure(
        event_id=event_id,
        kind=kind,
        fingerprint=fingerprint,
        message=f"{type(cause).__name__}: {cause}"[:500],
    )


def materialize_youtube_source(
    source_id: str,
    scheduled_at: datetime,
    implementation_ref: str,
) -> YouTubeSourceResult:
    ensure_news_catalog_schema()
    return run_youtube_source(YOUTUBE_SOURCES.source(source_id), scheduled_at, implementation_ref)


def materialize_next_youtube_publication(
    implementation_ref: str,
) -> YouTubePublicationResult | None:
    ensure_news_catalog_schema()
    return run_next_youtube_publication(implementation_ref)


def materialize_relevance(day: date, implementation_ref: str) -> DailyArtifactReferences:
    try:
        pending = read_pending_relevance_references(
            day=day,
            request_id_for_article=production_relevance_v3_request_id,
        )
        for reference in pending:
            output = analyze_relevance_v3(
                load_article_analysis_input(reference),
                mode="production_early_exit",
            )
            publish_relevance_outputs((output,), implementation_ref)
        return read_daily_relevance_references(day)
    finally:
        flush_langfuse_traces()


def materialize_embeddings(day: date, implementation_ref: str) -> DailyArtifactReferences:
    try:
        for reference in read_pending_embedding_references(day=day):
            output = embed_article(load_embedding_input(reference))
            publish_embedding_outputs((output,), implementation_ref)
        return read_daily_embedding_references(day)
    finally:
        flush_langfuse_traces()


def materialize_clusters(day: date, implementation_ref: str) -> DailyArtifactReferences:
    publication = publish_daily_clusters(
        cluster_articles(day, read_embedded_articles(day)),
        implementation_ref,
    )
    reference = read_daily_cluster_reference(day)
    if reference.version_id != publication.version_id:
        raise ValueError("Published cluster version does not match the current artifact")
    return DailyArtifactReferences(day=day, values=(reference,))


def materialize_group_summaries(day: date, implementation_ref: str) -> DailyArtifactReferences:
    _materialize_group_analysis(day, implementation_ref, summary=True)
    return read_daily_group_summary_references(day)


def materialize_group_sentiment(day: date, implementation_ref: str) -> DailyArtifactReferences:
    _materialize_group_analysis(day, implementation_ref, summary=False)
    return read_daily_group_sentiment_references(day)


def materialize_daily_themes(day: date, implementation_ref: str) -> DailyArtifactReferences:
    try:
        publication = publish_daily_themes(
            construct_daily_themes(read_daily_theme_input(day)),
            implementation_ref,
        )
        reference = read_daily_theme_reference(day)
        if reference != publication.reference:
            raise ValueError("Published daily theme version does not match the current artifact")
        return DailyArtifactReferences(day=day, values=(reference,))
    finally:
        flush_langfuse_traces()


def materialize_subject_assessments(day: date, implementation_ref: str) -> DailyArtifactReferences:
    value = read_daily_subject_assessment_input(day)
    run_id = subject_assessment_run_id(subject_assessment_request_id(value), implementation_ref)
    completed = read_completed_subject_assessment(run_id)
    if completed is not None:
        return DailyArtifactReferences(day=day, values=(completed,))
    try:
        publication = publish_daily_subject_assessments(
            construct_daily_subject_assessments(value),
            implementation_ref,
        )
        reference = read_daily_subject_assessment_reference(day)
        if reference != publication.reference:
            raise ValueError(
                "Published subject assessment version does not match the current artifact"
            )
        return DailyArtifactReferences(day=day, values=(reference,))
    finally:
        flush_langfuse_traces()


def materialize_daily_research_triggers(
    day: date, implementation_ref: str
) -> DailyArtifactReferences:
    from romanian_news.research_triggers import (
        PRODUCTION_RESEARCH_TRIGGER_POLICY as policy,
    )

    value = read_daily_research_trigger_input(day)
    run_id = research_trigger_run_id(research_trigger_request_id(value, policy), implementation_ref)
    completed = read_completed_research_trigger(run_id)
    if completed is not None:
        ping_heartbeat("research_trigger")
        return DailyArtifactReferences(day=day, values=(completed,))
    try:
        publication = publish_daily_research_triggers(
            construct_daily_research_triggers(value),
            implementation_ref,
        )
        reference = read_daily_research_trigger_reference(day)
        if reference != publication.reference:
            raise ValueError(
                "Published research trigger version does not match the current artifact"
            )
        ping_heartbeat("research_trigger")
        return DailyArtifactReferences(day=day, values=(reference,))
    finally:
        flush_langfuse_traces()


def materialize_daily_report(day: date, implementation_ref: str) -> DailyArtifactReferences:
    result = build_and_publish_current_daily_report(day, implementation_ref)
    reference = read_daily_report_reference(day)
    if reference.version_id != result.head.version_id:
        raise ValueError("Published daily report version does not match the current artifact")
    ping_heartbeat("report")
    return DailyArtifactReferences(day=day, values=(reference,))


def materialize_weekly_report(
    week_start: date, implementation_ref: str
) -> tuple[ArtifactReference, ...]:
    output = build_weekly_report_from_input(read_weekly_report_input(week_start))
    publication = publish_weekly_report(output, implementation_ref)
    reference = read_weekly_report_reference(week_start)
    if reference.version_id != publication.version_id:
        raise ValueError("Published weekly report version does not match the current artifact")
    return (reference,)


def _materialize_group_analysis(day: date, implementation_ref: str, *, summary: bool) -> None:
    try:
        for pending in read_pending_group_analysis_references({day}):
            if summary and not pending.summary_needed:
                continue
            if not summary and not pending.sentiment_needed:
                continue
            selected = pending.model_copy(
                update={"summary_needed": summary, "sentiment_needed": not summary}
            )
            output = analyze_group(load_group_analysis_input(selected))
            if output.errors:
                raise RuntimeError("; ".join(output.errors))
            values = tuple(
                value for value in (output.summary, output.sentiment) if value is not None
            )
            publish_group_analysis_outputs(values, implementation_ref)
    finally:
        flush_langfuse_traces()
