import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from romanian_news.analysis.attempts import model_attempt_from_payload, record_model_attempt
from romanian_news.analysis.tracing import ModelTraceReference


class _Response:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        return {
            "id": "response-1",
            "model": "test/model",
            "created": "2026-09-01T09:00:00+00:00",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01},
        }


def test_embedding_attempt_records_null_completion_tokens_as_zero() -> None:
    attempt = model_attempt_from_payload(
        {
            "id": "response-1",
            "model": "test/model",
            "usage": {"prompt_tokens": 10, "completion_tokens": None, "cost": 0.01},
        },
        request_id="a" * 64,
        operation_key="news.embed",
        attempt_index=0,
        latency_ms=120,
        status="accepted",
        error=None,
        observed_at=datetime(2026, 9, 6, tzinfo=UTC),
    )

    assert attempt.output_tokens == 0


def test_attempt_and_trace_link_persist_in_one_catalog_batch(monkeypatch) -> None:
    connection = _attempts_database()
    monkeypatch.setattr(
        "romanian_news.catalog.model_calls.catalog_batch", _sqlite_batch(connection)
    )
    trace = ModelTraceReference(
        provider="langfuse",
        trace_id="trace-1",
        observation_id="observation-1",
        project_ref="chartly-romanian-news",
        recorded_at=datetime(2026, 9, 1, 9, 0, tzinfo=UTC),
    )

    attempt = record_model_attempt(
        _Response(),
        request_id="a" * 64,
        operation_key="news.relevance",
        attempt_index=0,
        latency_ms=120,
        status="accepted",
        error=None,
        fallback_response_id="fallback-response",
        trace=trace,
    )

    attempts = connection.execute("SELECT * FROM news_model_attempts").fetchall()
    links = connection.execute("SELECT * FROM news_model_trace_links").fetchall()
    assert [row["attempt_id"] for row in attempts] == [attempt.attempt_id]
    assert attempts[0]["input_tokens"] == 10
    assert attempts[0]["output_tokens"] == 5
    assert links[0]["attempt_id"] == attempt.attempt_id
    assert links[0]["trace_id"] == "trace-1"
    assert links[0]["observation_id"] == "observation-1"


def _attempts_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    schema = Path(__file__).parents[4] / "tests" / "fixtures" / "sqlite_catalog.sql"
    connection.executescript(schema.read_text())
    return connection


def _sqlite_batch(connection: sqlite3.Connection):
    def batch(statements, *, retry_transient_errors=False):
        for sql, params in statements:
            connection.execute(sql.replace("%s", "?"), params)
        connection.commit()

    return batch
