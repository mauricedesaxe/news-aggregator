from datetime import date
from threading import Barrier, Lock
from time import sleep

import pytest

from romanian_news.analysis.tracing import archive_model_day
from romanian_news.worker import operations

DAY = date(2026, 9, 29)


def _stub_embedding_io(monkeypatch, pending, published):
    monkeypatch.setattr(operations, "read_pending_embedding_references", lambda **_kwargs: pending)
    monkeypatch.setattr(operations, "load_embedding_input", lambda reference: reference)
    monkeypatch.setattr(
        operations,
        "publish_embedding_outputs",
        lambda outputs, _ref: published.extend(outputs),
    )
    monkeypatch.setattr(
        operations, "read_daily_embedding_references", lambda _day: tuple(published)
    )
    monkeypatch.setattr(operations, "flush_langfuse_traces", lambda: None)


def test_daily_embeddings_run_four_at_a_time_and_publish_each_result(monkeypatch) -> None:
    published = []
    _stub_embedding_io(monkeypatch, tuple(range(8)), published)
    monkeypatch.setattr(operations, "archive_model_day_active", lambda: False)
    first_wave = Barrier(4)
    lock = Lock()
    active = 0
    peak = 0

    def embed(reference):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        if reference < 4:
            first_wave.wait(timeout=2)
        sleep(0.01)
        with lock:
            active -= 1
        return reference

    monkeypatch.setattr(operations, "embed_article", embed)

    operations.materialize_embeddings(DAY, "git:test")

    assert peak == 4
    assert sorted(published) == list(range(8))


def test_embedding_failure_drains_and_publishes_started_work(monkeypatch) -> None:
    published = []
    started = []
    _stub_embedding_io(monkeypatch, tuple(range(6)), published)
    monkeypatch.setattr(operations, "archive_model_day_active", lambda: False)
    first_wave = Barrier(4)

    def embed(reference):
        started.append(reference)
        first_wave.wait(timeout=2)
        if reference == 0:
            raise RuntimeError("model failed")
        sleep(0.02)
        return reference

    monkeypatch.setattr(operations, "embed_article", embed)

    with pytest.raises(RuntimeError, match="model failed"):
        operations.materialize_embeddings(DAY, "git:test")

    assert sorted(started) == list(range(4))
    assert sorted(published) == [1, 2, 3]


def test_archived_embeddings_run_with_the_archive_budget_context_active(monkeypatch) -> None:
    published = []
    _stub_embedding_io(monkeypatch, (0, 1, 2), published)

    def embed(reference):
        assert operations.archive_model_day_active(), "model call missed the archive budget context"
        return reference

    monkeypatch.setattr(operations, "embed_article", embed)

    with archive_model_day(DAY):
        operations.materialize_embeddings(DAY, "git:test")

    assert published == [0, 1, 2]
