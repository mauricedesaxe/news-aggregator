import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

from romanian_news.analysis.attempt_backfill import register_existing_model_calls
from romanian_news.analysis.attempts import read_model_usage

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64


class _ProviderUsagePayload(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    cost: float


class _ProviderPayload(TypedDict):
    id: str
    model: str
    created: int
    usage: _ProviderUsagePayload


class _HistoricalPayload(TypedDict):
    request_id: str
    call: dict[str, object]
    provider_responses: list[_ProviderPayload]


def test_historical_registration_registers_every_version_once(monkeypatch) -> None:
    payloads = {
        "relevance": _historical_payload(
            _A,
            [_provider_payload("historical-relevance", 10, 5, 0.1)],
        ),
        "sentiment": _historical_payload(
            _B,
            [
                _provider_payload("historical-rejected", 20, 8, 0.2),
                _provider_payload("historical-accepted", 30, 12, 0.3),
            ],
        ),
    }
    connection = _backfill_database()
    monkeypatch.setattr(
        "romanian_news.catalog.model_calls.catalog_query", _sqlite_query(connection)
    )
    monkeypatch.setattr(
        "romanian_news.catalog.model_calls.catalog_batch", _sqlite_batch(connection)
    )
    monkeypatch.setattr(
        "romanian_news.analysis.attempt_backfill.read_verified_r2_object",
        lambda key, _digest: json.dumps(payloads[key]).encode(),
    )

    assert register_existing_model_calls() == 5

    attempts = connection.execute(
        "SELECT * FROM news_model_attempts ORDER BY request_id, status"
    ).fetchall()
    calls = connection.execute("SELECT * FROM news_model_calls").fetchall()
    assert len(attempts) == 3
    assert len(calls) == 2
    rejected = [row for row in attempts if row["status"] == "rejected"]
    assert [row["error"] for row in rejected] == [
        "Historical correction response inferred as rejected"
    ]
    assert rejected[0]["latency_ms"] == 0
    accepted = [row for row in attempts if row["status"] == "accepted"]
    assert [row["latency_ms"] for row in accepted] == [120, 120]
    assert read_model_usage() == (60, 25, 0.6)

    assert register_existing_model_calls() == 0
    assert _attempts_count(connection) == 3
    assert _calls_count(connection) == 2


def test_historical_registration_backfills_only_missing_rows(monkeypatch) -> None:
    payloads = {
        "relevance": _historical_payload(
            _A,
            [_provider_payload("historical-relevance", 10, 5, 0.1)],
        ),
        "sentiment": _historical_payload(
            _B,
            [
                _provider_payload("historical-rejected", 20, 8, 0.2),
                _provider_payload("historical-accepted", 30, 12, 0.3),
            ],
        ),
    }
    complete = _patched_connection(monkeypatch, payloads)
    register_existing_model_calls()
    sentiment_accepted = complete.execute(
        "SELECT * FROM news_model_attempts WHERE request_id = ? AND status = 'accepted'", (_B,)
    ).fetchone()
    sentiment_call = complete.execute(
        "SELECT * FROM news_model_calls WHERE artifact_version_id = ?", (_D,)
    ).fetchone()

    partial = _patched_connection(monkeypatch, payloads)
    for attempt in complete.execute("SELECT * FROM news_model_attempts").fetchall():
        if attempt["attempt_id"] != sentiment_accepted["attempt_id"]:
            partial.execute(
                "INSERT INTO news_model_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(attempt),
            )
    for call in complete.execute("SELECT * FROM news_model_calls").fetchall():
        if call["artifact_version_id"] != sentiment_call["artifact_version_id"]:
            partial.execute(
                "INSERT INTO news_model_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(call),
            )
    partial.commit()

    assert register_existing_model_calls() == 2

    assert _attempts_count(partial) == 3
    assert _calls_count(partial) == 2
    assert read_model_usage() == (60, 25, 0.6)


def _patched_connection(
    monkeypatch, payloads: Mapping[str, _HistoricalPayload]
) -> sqlite3.Connection:
    connection = _backfill_database()
    monkeypatch.setattr(
        "romanian_news.catalog.model_calls.catalog_query", _sqlite_query(connection)
    )
    monkeypatch.setattr(
        "romanian_news.catalog.model_calls.catalog_batch", _sqlite_batch(connection)
    )
    monkeypatch.setattr(
        "romanian_news.analysis.attempt_backfill.read_verified_r2_object",
        lambda key, _digest: json.dumps(payloads[key]).encode(),
    )
    return connection


def _backfill_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    schema = Path(__file__).parents[4] / "tests" / "fixtures" / "sqlite_catalog.sql"
    connection.executescript(schema.read_text())
    for artifact_id, kind, title in (
        ("news:relevance:backfill", "news_relevance", "Relevance"),
        ("news:sentiment:backfill", "news_sentiment", "Sentiment"),
    ):
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                artifact_id,
                kind,
                title,
                "derived",
                "current",
                "private",
                None,
                "2026-08-30T09:00:00+00:00",
            ),
        )
    connection.execute(
        "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
        (_C, "news:relevance:backfill", 1, _C, None, "2026-08-30T09:00:00+00:00"),
    )
    connection.execute(
        "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
        (_D, "news:sentiment:backfill", 1, _D, None, "2026-08-31T09:00:00+00:00"),
    )
    connection.execute(
        "INSERT INTO artifact_files VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("file:relevance", _C, "relevance", "application/json", _C, 10, None, None),
    )
    connection.execute(
        "INSERT INTO artifact_files VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("file:sentiment", _D, "sentiment", "application/json", _D, 10, None, None),
    )
    connection.commit()
    return connection


def _sqlite_query(connection: sqlite3.Connection):
    def query(sql, params=None):
        return [
            dict(row) for row in connection.execute(sql.replace("%s", "?"), params or []).fetchall()
        ]

    return query


def _sqlite_batch(connection: sqlite3.Connection):
    def batch(statements, *, retry_transient_errors=False):
        for sql, params in statements:
            connection.execute(sql.replace("%s", "?"), params)
        connection.commit()

    return batch


def _attempts_count(connection: sqlite3.Connection) -> int:
    return connection.execute("SELECT count() FROM news_model_attempts").fetchone()[0]


def _calls_count(connection: sqlite3.Connection) -> int:
    return connection.execute("SELECT count() FROM news_model_calls").fetchone()[0]


def _provider_payload(
    response_id: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
) -> _ProviderPayload:
    return {
        "id": response_id,
        "model": "test/model",
        "created": 1_788_172_800,
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "cost": cost,
        },
    }


def _historical_payload(request_id: str, responses: list[_ProviderPayload]) -> _HistoricalPayload:
    return {
        "request_id": request_id,
        "call": {
            "response_id": responses[-1]["id"],
            "model": "test/model",
            "input_tokens": sum(response["usage"]["prompt_tokens"] for response in responses),
            "output_tokens": sum(response["usage"]["completion_tokens"] for response in responses),
            "latency_ms": 120,
        },
        "provider_responses": responses,
    }
