from datetime import UTC, date, datetime
from types import SimpleNamespace

import psycopg.errors
import pytest

from romanian_news import BUCHAREST
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.articles.models import (
    ArticleAcquisitionFailure,
    ArticleBatchSkip,
    ArticleFailureKind,
)
from romanian_news.catalog_transport import ResearchCatalogError
from romanian_news.daily import DailyArtifactReferences
from romanian_news.worker import operations

DAY = date(2099, 9, 2)


def test_feed_intake_prepares_schema_and_projection_before_feed_reads(
    monkeypatch, tmp_path
) -> None:
    events = []
    references = DailyArtifactReferences(day=DAY, values=())
    monkeypatch.setattr(
        operations, "ensure_news_catalog_schema", lambda: events.append("schema"), raising=False
    )
    monkeypatch.setattr(
        operations,
        "reconcile_feed_entry_projection",
        lambda: events.append("projection"),
        raising=False,
    )
    monkeypatch.setattr(
        operations,
        "feed_registry",
        lambda: events.append("registry") or SimpleNamespace(feeds=()),
    )
    monkeypatch.setattr(
        operations,
        "read_daily_feed_observation_references",
        lambda _day: events.append("feed-state") or references,
    )
    monkeypatch.setattr(
        operations,
        "read_successful_feed_observation_count",
        lambda *_args: events.append("feed-query") or 0,
    )

    result = operations.materialize_feed_intake(
        DAY,
        datetime(2099, 9, 2, tzinfo=UTC),
        tmp_path,
        "git:test",
    )

    assert result == references
    assert events == ["schema", "projection", "registry", "feed-state", "feed-query", "feed-state"]


def test_article_materializer_runs_one_exact_batch_and_rechecks_state(monkeypatch) -> None:
    planned = []
    events = []
    today = datetime.now(BUCHAREST).date()
    event_ids = ("a" * 64, "b" * 64)
    references = DailyArtifactReferences(day=today, values=())
    monkeypatch.setattr(operations, "ARTICLE_ITEM_WORKERS", 1)
    monkeypatch.setattr(operations, "ensure_news_catalog_schema", lambda: events.append("schema"))
    monkeypatch.setattr(operations, "feed_registry", lambda: SimpleNamespace(feeds=()))
    work_items = tuple(
        SimpleNamespace(
            source=SimpleNamespace(
                event_id=event_id, feed_id="feed-1", url="https://example.com/a"
            ),
            last_captured_at=None,
        )
        for event_id in event_ids
    )
    monkeypatch.setattr(
        operations,
        "load_exact_article_work",
        lambda ids, *_args, **_kwargs: events.append(("load", ids)) or work_items,
    )
    monkeypatch.setattr(
        operations,
        "acquire_article_batch_item",
        lambda work, _registry: events.append(("acquire", work.source.event_id))
        or ArticleBatchSkip(event_id=work.source.event_id),
    )
    monkeypatch.setattr(
        operations,
        "record_article_failure_attempts",
        lambda *_args, **_kwargs: events.append("record"),
    )
    monkeypatch.setattr(
        operations,
        "read_article_work_status",
        lambda *_args, **kwargs: planned.append(kwargs)
        or SimpleNamespace(
            retryable_entries=0,
            deferred_event_ids=(),
            quarantined_event_ids=(),
            source_covered_days=(today,),
        ),
    )
    monkeypatch.setattr(operations, "read_daily_article_references", lambda _day: references)

    result = operations.materialize_articles(
        today,
        event_ids,
        "git:test",
        run_id="run-1",
        retry_number=0,
    )

    assert result.complete
    assert result.skipped_event_ids == event_ids
    assert events == [
        "schema",
        ("load", event_ids),
        ("acquire", event_ids[0]),
        ("acquire", event_ids[1]),
    ]
    assert len(planned) == 1
    assert planned[0]["now"] - planned[0]["revalidate_before"] == operations.timedelta(hours=24)


def test_article_materializer_records_item_failure_before_the_next_item(monkeypatch) -> None:
    first_id = "a" * 64
    second_id = "b" * 64
    work_items = tuple(
        SimpleNamespace(
            source=SimpleNamespace(
                event_id=event_id, feed_id="feed-1", url="https://example.com/a"
            ),
            last_captured_at=None,
        )
        for event_id in (first_id, second_id)
    )
    failure = ArticleAcquisitionFailure(
        event_id=first_id,
        kind=ArticleFailureKind.DETERMINISTIC,
        fingerprint="c" * 64,
        message="invalid snapshot",
    )
    events = []

    def acquire(work, _registry):
        events.append(("acquire", work.source.event_id))
        return failure if work.source.event_id == first_id else ArticleBatchSkip(event_id=second_id)

    monkeypatch.setattr(operations, "ARTICLE_ITEM_WORKERS", 1)
    monkeypatch.setattr(operations, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(operations, "feed_registry", lambda: SimpleNamespace(feeds=()))
    monkeypatch.setattr(operations, "load_exact_article_work", lambda *_args, **_kwargs: work_items)
    monkeypatch.setattr(operations, "acquire_article_batch_item", acquire)
    monkeypatch.setattr(
        operations,
        "record_article_failure_attempts",
        lambda failures, **_kwargs: events.append(("record", failures[0].event_id)),
    )
    monkeypatch.setattr(
        operations,
        "read_article_work_status",
        lambda *_args, **_kwargs: SimpleNamespace(
            retryable_entries=1,
            deferred_event_ids=(first_id,),
            quarantined_event_ids=(),
            source_covered_days=(),
        ),
    )
    monkeypatch.setattr(
        operations,
        "read_daily_article_references",
        lambda day: DailyArtifactReferences(day=day, values=()),
    )

    operations.materialize_articles(
        DAY,
        (first_id, second_id),
        "git:test",
        run_id="run-1",
        retry_number=0,
    )

    assert events == [
        ("acquire", first_id),
        ("record", first_id),
        ("acquire", second_id),
    ]


def test_article_materializer_publishes_each_success_before_the_next_item(monkeypatch) -> None:
    first_id = "a" * 64
    second_id = "b" * 64
    work_items = tuple(
        SimpleNamespace(
            source=SimpleNamespace(
                event_id=event_id, feed_id="feed-1", url="https://example.com/a"
            ),
            last_captured_at=None,
        )
        for event_id in (first_id, second_id)
    )
    events = []

    class CaptureOutcome:
        def __init__(self, event_id):
            self.capture = SimpleNamespace(
                source=SimpleNamespace(
                    event_id=event_id, feed_id="feed-1", url="https://example.com/a"
                )
            )

    def acquire(work, _registry):
        events.append(("acquire", work.source.event_id))
        if work.source.event_id == second_id:
            raise RuntimeError("peer hung")
        return CaptureOutcome(work.source.event_id)

    monkeypatch.setattr(operations, "ARTICLE_ITEM_WORKERS", 1)
    monkeypatch.setattr(operations, "ArticleBatchCapture", CaptureOutcome)
    monkeypatch.setattr(operations, "ArticleAcquisitionResult", lambda **kwargs: kwargs)
    monkeypatch.setattr(operations, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(operations, "feed_registry", lambda: SimpleNamespace(feeds=()))
    monkeypatch.setattr(operations, "load_exact_article_work", lambda *_args, **_kwargs: work_items)
    monkeypatch.setattr(operations, "acquire_article_batch_item", acquire)
    monkeypatch.setattr(
        operations,
        "publish_articles",
        lambda result, _implementation_ref: events.append(
            ("publish", result["captures"][0].source.event_id)
        ),
    )

    with pytest.raises(RuntimeError, match="peer hung"):
        operations.materialize_articles(
            DAY,
            (first_id, second_id),
            "git:test",
            run_id="run-1",
            retry_number=0,
        )

    assert events == [
        ("acquire", first_id),
        ("publish", first_id),
        ("acquire", second_id),
    ]


def test_article_materializer_records_publish_conflicts_as_deterministic(
    monkeypatch,
) -> None:
    first_id = "a" * 64
    work_items = (
        SimpleNamespace(
            source=SimpleNamespace(
                event_id=first_id, feed_id="feed-1", url="https://example.com/a"
            ),
            last_captured_at=None,
        ),
    )
    recorded = []

    class CaptureOutcome:
        capture = SimpleNamespace(
            source=SimpleNamespace(event_id=first_id, feed_id="feed-1", url="https://example.com/a")
        )

    conflict = psycopg.errors.IntegrityConstraintViolation("news_article_aliases identity conflict")
    publish_error = ResearchCatalogError("PostgreSQL catalog request failed")
    publish_error.__cause__ = conflict

    def publish(_result, _implementation_ref):
        raise publish_error

    monkeypatch.setattr(operations, "ArticleBatchCapture", CaptureOutcome)
    monkeypatch.setattr(operations, "ArticleAcquisitionResult", lambda **kwargs: kwargs)
    monkeypatch.setattr(operations, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(operations, "feed_registry", lambda: SimpleNamespace(feeds=()))
    monkeypatch.setattr(operations, "load_exact_article_work", lambda *_args, **_kwargs: work_items)
    monkeypatch.setattr(
        operations, "acquire_article_batch_item", lambda work, _registry: CaptureOutcome()
    )
    monkeypatch.setattr(operations, "publish_articles", publish)
    monkeypatch.setattr(
        operations,
        "record_article_failure_attempts",
        lambda failures, **_kwargs: recorded.append(failures[0]),
    )
    monkeypatch.setattr(
        operations,
        "read_article_work_status",
        lambda *_args, **_kwargs: SimpleNamespace(
            retryable_entries=0,
            deferred_event_ids=(),
            quarantined_event_ids=(),
            source_covered_days=(DAY,),
        ),
    )
    monkeypatch.setattr(
        operations,
        "read_daily_article_references",
        lambda day: DailyArtifactReferences(day=day, values=()),
    )

    result = operations.materialize_articles(
        DAY,
        (first_id,),
        "git:test",
        run_id="run-1",
        retry_number=0,
    )
    second = operations._publish_failure(first_id, publish_error)

    assert result.complete
    assert tuple(recorded) == result.failures
    assert recorded[0].kind == ArticleFailureKind.DETERMINISTIC
    assert recorded[0].message.endswith("identity conflict")
    assert second.fingerprint == recorded[0].fingerprint


def test_article_materializer_isolates_a_publish_conflict_across_parallel_items(
    monkeypatch,
) -> None:
    first_id = "a" * 64
    second_id = "b" * 64
    work_items = tuple(
        SimpleNamespace(
            source=SimpleNamespace(
                event_id=event_id, feed_id="feed-1", url="https://example.com/a"
            ),
            last_captured_at=None,
        )
        for event_id in (first_id, second_id)
    )
    recorded = []
    published = []
    conflict = psycopg.errors.IntegrityConstraintViolation("news_article_aliases identity conflict")
    publish_error = ResearchCatalogError("PostgreSQL catalog request failed")
    publish_error.__cause__ = conflict

    class CaptureOutcome:
        def __init__(self, event_id):
            self.capture = SimpleNamespace(
                source=SimpleNamespace(
                    event_id=event_id, feed_id="feed-1", url="https://example.com/a"
                )
            )

    def acquire(work, _registry):
        return CaptureOutcome(work.source.event_id)

    def publish(result, _implementation_ref):
        capture = result["captures"][0]
        if capture.source.event_id == first_id:
            published.append(capture.source.event_id)
            return
        raise publish_error

    monkeypatch.setattr(operations, "ArticleBatchCapture", CaptureOutcome)
    monkeypatch.setattr(operations, "ArticleAcquisitionResult", lambda **kwargs: kwargs)
    monkeypatch.setattr(operations, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(operations, "feed_registry", lambda: SimpleNamespace(feeds=()))
    monkeypatch.setattr(operations, "load_exact_article_work", lambda *_args, **_kwargs: work_items)
    monkeypatch.setattr(operations, "acquire_article_batch_item", acquire)
    monkeypatch.setattr(operations, "publish_articles", publish)
    monkeypatch.setattr(
        operations,
        "record_article_failure_attempts",
        lambda failures, **_kwargs: recorded.append(failures[0]),
    )
    monkeypatch.setattr(
        operations,
        "read_article_work_status",
        lambda *_args, **_kwargs: SimpleNamespace(
            retryable_entries=0,
            deferred_event_ids=(),
            quarantined_event_ids=(),
            source_covered_days=(DAY,),
        ),
    )
    monkeypatch.setattr(
        operations,
        "read_daily_article_references",
        lambda day: DailyArtifactReferences(day=day, values=()),
    )

    result = operations.materialize_articles(
        DAY,
        (first_id, second_id),
        "git:test",
        run_id="run-1",
        retry_number=0,
    )

    assert published == [first_id]
    assert len(result.failures) == 1
    assert result.failures[0].event_id == second_id
    assert result.failures[0].kind == ArticleFailureKind.DETERMINISTIC
    assert tuple(recorded) == result.failures
    assert result.acquired_event_ids == (first_id,)


def test_article_materializer_dedupes_alias_sharing_events(monkeypatch) -> None:
    first_id = "a" * 64
    second_id = "b" * 64
    feed = SimpleNamespace(id="feed-1", article_hosts=("example.com",))
    url = "https://example.com/news/story?utm_source=feed"
    work_items = tuple(
        SimpleNamespace(
            source=SimpleNamespace(event_id=event_id, feed_id="feed-1", url=url),
            last_captured_at=None,
        )
        for event_id in (first_id, second_id)
    )
    acquired = []

    def acquire(work, _registry):
        acquired.append(work.source.event_id)
        return ArticleBatchSkip(event_id=work.source.event_id)

    monkeypatch.setattr(operations, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(operations, "feed_registry", lambda: SimpleNamespace(feeds=(feed,)))
    monkeypatch.setattr(operations, "load_exact_article_work", lambda *_args, **_kwargs: work_items)
    monkeypatch.setattr(operations, "acquire_article_batch_item", acquire)
    monkeypatch.setattr(
        operations,
        "read_article_work_status",
        lambda *_args, **_kwargs: SimpleNamespace(
            retryable_entries=0,
            deferred_event_ids=(),
            quarantined_event_ids=(),
            source_covered_days=(DAY,),
        ),
    )
    monkeypatch.setattr(
        operations,
        "read_daily_article_references",
        lambda day: DailyArtifactReferences(day=day, values=()),
    )

    result = operations.materialize_articles(
        DAY,
        (first_id, second_id),
        "git:test",
        run_id="run-1",
        retry_number=0,
    )

    assert acquired == [first_id]
    assert set(result.skipped_event_ids) == {first_id, second_id}


def test_article_materializer_rejects_more_than_ten_event_ids() -> None:
    with pytest.raises(ValueError, match="at most 10"):
        operations.materialize_articles(
            DAY,
            tuple(f"{index:064x}" for index in range(11)),
            "git:test",
            run_id="run-1",
            retry_number=0,
        )


@pytest.mark.parametrize(
    ("run", "patches"),
    [
        (
            lambda: operations.materialize_relevance(DAY, "git:test"),
            (
                ("read_pending_relevance_references", lambda **_kwargs: (object(),)),
                ("load_article_analysis_input", lambda _reference: object()),
                (
                    "analyze_relevance_v3",
                    lambda _value, **_kwargs: _raise_model_failure(),
                ),
            ),
        ),
        (
            lambda: operations.materialize_embeddings(DAY, "git:test"),
            (
                ("read_pending_embedding_references", lambda **_kwargs: (object(),)),
                ("load_embedding_input", lambda _reference: object()),
                ("embed_article", lambda _value: _raise_model_failure()),
            ),
        ),
        (
            lambda: operations._materialize_group_analysis(DAY, "git:test", summary=True),
            (
                (
                    "read_pending_group_analysis_references",
                    lambda _days: (
                        SimpleNamespace(
                            summary_needed=True,
                            sentiment_needed=False,
                            model_copy=lambda **_kwargs: object(),
                        ),
                    ),
                ),
                ("load_group_analysis_input", lambda _pending: object()),
                ("analyze_group", lambda _value: _raise_model_failure()),
            ),
        ),
    ],
)
def test_model_materializers_flush_traces_without_hiding_model_failure(
    monkeypatch,
    run,
    patches,
) -> None:
    events = []
    for name, value in patches:
        monkeypatch.setattr(operations, name, value)
    monkeypatch.setattr(
        operations,
        "flush_langfuse_traces",
        lambda: events.append("flush"),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="model failed"):
        run()

    assert events == ["flush"]


def test_relevance_materializer_uses_production_v3(monkeypatch) -> None:
    events = []
    references = DailyArtifactReferences(day=DAY, values=())
    reference = object()
    analysis_input = object()
    output = object()

    def pending(**kwargs):
        events.append(("pending", kwargs))
        return (reference,)

    monkeypatch.setattr(operations, "read_pending_relevance_references", pending)
    monkeypatch.setattr(
        operations,
        "load_article_analysis_input",
        lambda value: events.append(("load", value)) or analysis_input,
    )
    monkeypatch.setattr(
        operations,
        "analyze_relevance_v3",
        lambda value, **kwargs: events.append(("analyze", value, kwargs)) or output,
    )
    monkeypatch.setattr(
        operations,
        "publish_relevance_outputs",
        lambda values, implementation_ref: events.append(("publish", values, implementation_ref)),
    )
    monkeypatch.setattr(operations, "read_daily_relevance_references", lambda _day: references)
    monkeypatch.setattr(operations, "flush_langfuse_traces", lambda: events.append(("flush",)))

    assert operations.materialize_relevance(DAY, "git:test") == references
    assert events == [
        (
            "pending",
            {
                "day": DAY,
                "request_id_for_article": operations.production_relevance_v3_request_id,
            },
        ),
        ("load", reference),
        ("analyze", analysis_input, {"mode": "production_early_exit"}),
        ("publish", (output,), "git:test"),
        ("flush",),
    ]


def _raise_model_failure() -> None:
    raise RuntimeError("model failed")


def test_subject_assessment_materializer_reuses_completed_run_before_inference(
    monkeypatch,
) -> None:
    value = object()
    reference = ArtifactReference(
        artifact_id="news:subject-assessments:test",
        version_id="a" * 64,
        content_digest="d" * 64,
        r2_key="news/subject-assessments/test.json",
    )
    events = []
    monkeypatch.setattr(
        operations,
        "read_daily_subject_assessment_input",
        lambda day: events.append(("input", day)) or value,
    )
    monkeypatch.setattr(
        operations,
        "subject_assessment_request_id",
        lambda item: events.append(("request", item)) or "b" * 64,
    )
    monkeypatch.setattr(
        operations,
        "subject_assessment_run_id",
        lambda request_id, implementation_ref: events.append(
            ("run", request_id, implementation_ref)
        )
        or "c" * 64,
    )
    monkeypatch.setattr(
        operations,
        "read_completed_subject_assessment",
        lambda run_id: events.append(("completed", run_id)) or reference,
    )
    monkeypatch.setattr(
        operations,
        "construct_daily_subject_assessments",
        lambda _value: (_ for _ in ()).throw(AssertionError("unexpected inference")),
    )

    result = operations.materialize_subject_assessments(DAY, "git:test")

    assert result.values == (reference,)
    assert events == [
        ("input", DAY),
        ("request", value),
        ("run", "b" * 64, "git:test"),
        ("completed", "c" * 64),
    ]
