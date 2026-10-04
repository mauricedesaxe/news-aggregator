from datetime import date
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from romanian_news.analysis.tracing import archive_model_day
from romanian_news.worker import operations

DAY = date(2026, 9, 29)


def test_group_summaries_run_four_model_calls_concurrently(monkeypatch) -> None:
    pending = tuple(SimpleNamespace(index=index, summary_needed=True) for index in range(4))
    barrier = Barrier(4)
    published = []
    expected = object()
    monkeypatch.setattr(operations, "read_pending_group_analysis_references", lambda _days: pending)
    monkeypatch.setattr(operations, "load_group_analysis_input", lambda reference: reference)
    monkeypatch.setattr(operations, "archive_model_day_active", lambda: False)
    monkeypatch.setattr(operations, "flush_langfuse_traces", lambda: None)
    monkeypatch.setattr(operations, "read_daily_group_summary_references", lambda _day: expected)
    monkeypatch.setattr(
        operations,
        "publish_group_analysis_outputs",
        lambda outputs, _implementation_ref: published.append(outputs[0].index),
    )

    def summarize(value):
        barrier.wait(timeout=5)
        return value

    monkeypatch.setattr(operations, "summarize_group", summarize)

    assert operations.materialize_group_summaries(DAY, "git:test") is expected
    assert sorted(published) == [0, 1, 2, 3]


def test_group_summaries_keep_workers_busy_when_one_group_is_slow(monkeypatch) -> None:
    pending = tuple(SimpleNamespace(index=index, summary_needed=True) for index in range(5))
    fifth_started = Event()
    published = []
    monkeypatch.setattr(operations, "read_pending_group_analysis_references", lambda _days: pending)
    monkeypatch.setattr(operations, "load_group_analysis_input", lambda reference: reference)
    monkeypatch.setattr(operations, "archive_model_day_active", lambda: False)
    monkeypatch.setattr(operations, "flush_langfuse_traces", lambda: None)
    monkeypatch.setattr(operations, "read_daily_group_summary_references", lambda _day: ())
    monkeypatch.setattr(
        operations,
        "publish_group_analysis_outputs",
        lambda outputs, _implementation_ref: published.append(outputs[0].index),
    )

    def summarize(value):
        if value.index == 0:
            assert fifth_started.wait(timeout=5)
        if value.index == 4:
            fifth_started.set()
        return value

    monkeypatch.setattr(operations, "summarize_group", summarize)

    operations.materialize_group_summaries(DAY, "git:test")

    assert sorted(published) == [0, 1, 2, 3, 4]


def test_group_sentiment_preserves_concurrent_successes_when_one_model_call_fails(
    monkeypatch,
) -> None:
    pending = tuple(SimpleNamespace(index=index, sentiment_needed=True) for index in range(4))
    published = []
    monkeypatch.setattr(operations, "read_pending_group_analysis_references", lambda _days: pending)
    monkeypatch.setattr(operations, "load_group_analysis_input", lambda reference: reference)
    monkeypatch.setattr(operations, "archive_model_day_active", lambda: False)
    monkeypatch.setattr(operations, "flush_langfuse_traces", lambda: None)
    monkeypatch.setattr(
        operations,
        "publish_group_analysis_outputs",
        lambda outputs, _implementation_ref: published.append(outputs[0].index),
    )

    def score(value):
        if value.index == 1:
            raise ValueError("invalid model response")
        return value

    monkeypatch.setattr(operations, "score_group_sentiment", score)

    with pytest.raises(RuntimeError, match="sentiment: invalid model response"):
        operations.materialize_group_sentiment(DAY, "git:test")

    assert sorted(published) == [0, 2, 3]


def test_group_summaries_run_with_the_archive_budget_context_when_archiving(monkeypatch) -> None:
    pending = tuple(SimpleNamespace(index=index, summary_needed=True) for index in range(3))
    published = []
    monkeypatch.setattr(operations, "read_pending_group_analysis_references", lambda _days: pending)
    monkeypatch.setattr(operations, "load_group_analysis_input", lambda reference: reference)
    monkeypatch.setattr(operations, "flush_langfuse_traces", lambda: None)
    monkeypatch.setattr(operations, "read_daily_group_summary_references", lambda _day: ())
    monkeypatch.setattr(
        operations,
        "publish_group_analysis_outputs",
        lambda outputs, _implementation_ref: published.append(outputs[0].index),
    )

    def summarize(value):
        assert operations.archive_model_day_active(), "model call missed the archive budget context"
        return value

    monkeypatch.setattr(operations, "summarize_group", summarize)

    with archive_model_day(DAY):
        operations.materialize_group_summaries(DAY, "git:test")

    assert sorted(published) == [0, 1, 2]
