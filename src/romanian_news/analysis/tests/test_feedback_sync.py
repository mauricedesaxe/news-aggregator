import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from romanian_news.analysis import feedback_sync

MANIFEST_ID = "a" * 64
REPORT_ID = "b" * 64
ASSIGNMENT_OUTPUT_ID = "c" * 64
PROSE_OUTPUT_ID = "d" * 64
ASSIGNMENT_ATTEMPT_ID = "e" * 64
PROSE_ATTEMPT_ID = "f" * 64
FEEDBACK_ID = "00000000-0000-4000-8000-000000000001"
SCHEMA_PATH = Path(__file__).parents[4] / "tests" / "fixtures" / "sqlite_catalog.sql"


class FakeLangfuseClient:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[dict[str, object]] = []
        self.api = type(
            "Api", (), {"scores": type("Scores", (), {"create": self.create_score})()}
        )()

    def create_score(self, **arguments: object):
        self.calls.append(arguments)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("Langfuse unavailable")
        return type("Score", (), {"id": arguments["id"]})()


def test_projects_distinct_concern_polarities_to_exact_attempts(monkeypatch) -> None:
    with closing(_database()) as connection:
        client = _patch_boundaries(monkeypatch, connection)
        _insert_feedback(connection, note="Limba este greșită, dar gruparea este bună.")
        _insert_route(
            connection,
            position=0,
            score_id="language-score",
            concern="language",
            output_id=PROSE_OUTPUT_ID,
            attempt_id=PROSE_ATTEMPT_ID,
            polarity="negative",
        )
        _insert_route(
            connection,
            position=1,
            score_id="grouping-score",
            concern="grouping",
            output_id=ASSIGNMENT_OUTPUT_ID,
            attempt_id=ASSIGNMENT_ATTEMPT_ID,
            polarity="positive",
        )
        _insert_trace(connection, ASSIGNMENT_ATTEMPT_ID, "assignment")
        _insert_trace(connection, PROSE_ATTEMPT_ID, "prose")

        result = feedback_sync.sync_news_feedback(MANIFEST_ID)

        assert result.model_dump() == {
            "selected_feedback": 1,
            "matched_attempts": 2,
            "completed": 2,
            "failed": 0,
            "unresolved": 0,
        }
        assert {(call["name"], call["value"], call["observation_id"]) for call in client.calls} == {
            ("news_reader_language_feedback", 0, "observation-prose"),
            ("news_reader_grouping_feedback", 1, "observation-assignment"),
        }
        for call in client.calls:
            assert call["comment"] == "Limba este greșită, dar gruparea este bună."
            metadata = call["metadata"]
            assert isinstance(metadata, dict)
            assert metadata["manifest_artifact_version_id"] == MANIFEST_ID


def test_excluded_contradictory_and_presentation_decisions_create_no_scores(
    monkeypatch,
) -> None:
    with closing(_database()) as connection:
        client = _patch_boundaries(monkeypatch, connection)
        _insert_feedback(connection)
        _insert_exclusion(connection, 0, "presentation")
        _insert_feedback(
            connection,
            feedback_id="00000000-0000-4000-8000-000000000002",
        )
        _insert_exclusion(
            connection,
            1,
            "contradictory",
            feedback_id="00000000-0000-4000-8000-000000000002",
        )
        combined_feedback_id = "00000000-0000-4000-8000-000000000003"
        _insert_feedback(connection, feedback_id=combined_feedback_id)
        _insert_concern_exclusion(connection, 2, combined_feedback_id)

        result = feedback_sync.sync_news_feedback(MANIFEST_ID)

        assert result.selected_feedback == 0
        assert client.calls == []


def test_failed_route_retries_with_same_score_id(monkeypatch) -> None:
    with closing(_database()) as connection:
        client = _patch_boundaries(monkeypatch, connection, failures=1)
        _insert_feedback(connection)
        _insert_route(connection, score_id="stable-score")
        _insert_trace(connection, ASSIGNMENT_ATTEMPT_ID, "assignment")

        first = feedback_sync.sync_news_feedback(MANIFEST_ID)
        second = feedback_sync.sync_news_feedback(MANIFEST_ID)

        assert first.failed == 1
        assert second.completed == 1
        assert [call["id"] for call in client.calls] == ["stable-score", "stable-score"]
        rows = connection.execute(
            "SELECT sync_attempt_id, score_id, status FROM "
            "news_feedback_concern_score_sync_attempts ORDER BY attempted_at"
        ).fetchall()
        assert rows[0][0] != rows[1][0]
        assert [tuple(row)[1:] for row in rows] == [
            ("stable-score", "failed"),
            ("stable-score", "completed"),
        ]
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE news_feedback_concern_score_sync_attempts SET status = 'failed'"
            )


def test_old_target_receipt_does_not_suppress_concern_score(monkeypatch) -> None:
    with closing(_database()) as connection:
        client = _patch_boundaries(monkeypatch, connection)
        _insert_feedback(connection)
        _insert_route(connection, score_id="concern-score")
        _insert_trace(connection, ASSIGNMENT_ATTEMPT_ID, "assignment")
        connection.execute(
            "INSERT INTO news_feedback_sync_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "old-sync",
                FEEDBACK_ID,
                ASSIGNMENT_ATTEMPT_ID,
                "langfuse",
                "old-target-score",
                "completed",
                None,
                "2026-09-09T00:00:00+00:00",
            ),
        )
        connection.commit()

        result = feedback_sync.sync_news_feedback(MANIFEST_ID)

        assert result.completed == 1
        assert [call["id"] for call in client.calls] == ["concern-score"]


def test_completed_concern_route_is_not_sent_again(monkeypatch) -> None:
    with closing(_database()) as connection:
        client = _patch_boundaries(monkeypatch, connection)
        _insert_feedback(connection)
        _insert_route(connection, score_id="concern-score")
        _insert_trace(connection, ASSIGNMENT_ATTEMPT_ID, "assignment")

        first = feedback_sync.sync_news_feedback(MANIFEST_ID)
        second = feedback_sync.sync_news_feedback(MANIFEST_ID)

        assert first.completed == 1
        assert second.selected_feedback == 0
        assert len(client.calls) == 1


def test_missing_trace_remains_retryable(monkeypatch) -> None:
    with closing(_database()) as connection:
        client = _patch_boundaries(monkeypatch, connection)
        _insert_feedback(connection)
        _insert_route(connection)

        first = feedback_sync.sync_news_feedback(MANIFEST_ID)
        second = feedback_sync.sync_news_feedback(MANIFEST_ID)

        assert first.unresolved == second.unresolved == 1
        assert first.matched_attempts == second.matched_attempts == 1
        assert client.calls == []
        assert (
            connection.execute(
                "SELECT count(*) FROM news_feedback_concern_score_dispositions"
            ).fetchone()[0]
            == 0
        )


def test_langsmith_route_gets_terminal_provider_migration_disposition(monkeypatch) -> None:
    with closing(_database()) as connection:
        client = _patch_boundaries(monkeypatch, connection)
        _insert_feedback(connection)
        _insert_route(connection, score_id="legacy-score")
        _insert_trace(connection, ASSIGNMENT_ATTEMPT_ID, "assignment", provider="langsmith")

        first = feedback_sync.sync_news_feedback(MANIFEST_ID)
        second = feedback_sync.sync_news_feedback(MANIFEST_ID)

        assert first.unresolved == 1
        assert second.selected_feedback == 0
        assert client.calls == []
        assert tuple(
            connection.execute(
                "SELECT score_id, reason, source_provider FROM "
                "news_feedback_concern_score_dispositions"
            ).fetchone()
        ) == ("legacy-score", "provider_migration", "langsmith")


def test_without_api_key_returns_before_catalog_query(monkeypatch) -> None:
    monkeypatch.setattr(feedback_sync, "LANGFUSE_PUBLIC_KEY", None)
    monkeypatch.setattr(feedback_sync, "LANGFUSE_SECRET_KEY", None)
    monkeypatch.setattr(
        feedback_sync.feedback_catalog,
        "read_pending_feedback_score_routes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("catalog was queried")),
    )

    assert feedback_sync.sync_news_feedback(MANIFEST_ID) == feedback_sync.NewsFeedbackSyncResult()


def _database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text())
    for position, version_id in enumerate(
        (MANIFEST_ID, REPORT_ID, ASSIGNMENT_OUTPUT_ID, PROSE_OUTPUT_ID)
    ):
        artifact_id = f"news:test:{position}"
        connection.execute(
            "INSERT INTO artifacts VALUES (?, 'test', ?, 'derived', 'current', "
            "'private', NULL, '2026-09-09T00:00:00+00:00', NULL)",
            (artifact_id, artifact_id),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, 1, ?, NULL, ?)",
            (version_id, artifact_id, version_id, "2026-09-09T00:00:00+00:00"),
        )
    for attempt_id in (ASSIGNMENT_ATTEMPT_ID, PROSE_ATTEMPT_ID):
        connection.execute(
            "INSERT INTO news_model_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                attempt_id,
                attempt_id,
                "news.test",
                f"response-{attempt_id[0]}",
                "test/model",
                1,
                1,
                0.0,
                1,
                "accepted",
                None,
                "2026-09-09T00:00:00+00:00",
            ),
        )
    connection.commit()
    return connection


def _patch_boundaries(
    monkeypatch,
    connection: sqlite3.Connection,
    failures: int = 0,
) -> FakeLangfuseClient:
    monkeypatch.setattr(feedback_sync, "LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setattr(feedback_sync, "LANGFUSE_SECRET_KEY", "secret")
    monkeypatch.setattr(
        feedback_sync.feedback_catalog,
        "catalog_query",
        lambda sql, params=None: [
            dict(row) for row in connection.execute(sql.replace("%s", "?"), params or []).fetchall()
        ],
    )

    def batch(statements) -> None:
        with connection:
            for sql, params in statements:
                connection.execute(sql.replace("%s", "?"), params)

    monkeypatch.setattr(feedback_sync.feedback_catalog, "catalog_batch", batch)
    client = FakeLangfuseClient(failures)
    monkeypatch.setattr(feedback_sync, "_langfuse_client", lambda: client)
    return client


def _insert_feedback(
    connection: sqlite3.Connection,
    feedback_id: str = FEEDBACK_ID,
    note: str | None = None,
) -> None:
    connection.execute(
        "INSERT INTO news_feedback VALUES (?, ?, 'theme', ?, NULL, NULL, 'negative', ?, "
        "'owner', '2026-09-09T00:00:00+00:00')",
        (feedback_id, REPORT_ID, "1" * 64, note),
    )
    connection.commit()


def _insert_route(
    connection: sqlite3.Connection,
    *,
    position: int = 0,
    score_id: str = "score-id",
    concern: str = "grouping",
    output_id: str = ASSIGNMENT_OUTPUT_ID,
    attempt_id: str = ASSIGNMENT_ATTEMPT_ID,
    polarity: str = "negative",
    feedback_id: str = FEEDBACK_ID,
) -> None:
    connection.execute(
        "INSERT INTO news_feedback_score_curation VALUES "
        "(?, 'v-next', ?, ?, ?, NULL, 'projected', ?, ?, ?, ?, "
        "'Reviewed decision.', ?)",
        (
            MANIFEST_ID,
            position,
            feedback_id,
            concern,
            REPORT_ID,
            output_id,
            attempt_id,
            polarity,
            score_id,
        ),
    )
    connection.commit()


def _insert_exclusion(
    connection: sqlite3.Connection,
    position: int,
    concern: str,
    feedback_id: str = FEEDBACK_ID,
) -> None:
    connection.execute(
        "INSERT INTO news_feedback_score_curation VALUES "
        "(?, 'v-next', ?, ?, NULL, ?, 'excluded', ?, NULL, NULL, NULL, "
        "'Excluded.', NULL)",
        (MANIFEST_ID, position, feedback_id, concern, REPORT_ID),
    )
    connection.commit()


def _insert_concern_exclusion(
    connection: sqlite3.Connection,
    position: int,
    feedback_id: str,
) -> None:
    connection.execute(
        "INSERT INTO news_feedback_score_curation VALUES "
        "(?, 'v-next', ?, ?, 'language', 'no_concern_specific_observation', "
        "'excluded', ?, NULL, NULL, 'negative', 'Combined model output.', NULL)",
        (MANIFEST_ID, position, feedback_id, REPORT_ID),
    )
    connection.commit()


def _insert_trace(
    connection: sqlite3.Connection,
    attempt_id: str,
    suffix: str,
    provider: str = "langfuse",
) -> None:
    connection.execute(
        "INSERT INTO news_model_trace_links VALUES (?, ?, ?, ?, 'test', ?)",
        (
            attempt_id,
            provider,
            f"trace-{suffix}",
            f"observation-{suffix}",
            "2026-09-09T00:00:00+00:00",
        ),
    )
    connection.commit()
