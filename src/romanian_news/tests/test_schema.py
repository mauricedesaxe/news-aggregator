import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

SCHEMA_PATH = Path(__file__).parents[3] / "tests" / "fixtures" / "sqlite_catalog.sql"
MIGRATION_PATH = Path(__file__).parents[1] / "catalog" / "migrations" / "0001_initial.sql"
RECOVERY_MIGRATION_PATH = (
    Path(__file__).parents[1] / "catalog" / "migrations" / "0011_article_recovery_overrides.sql"
)
REPAIR_MIGRATION_PATH = (
    Path(__file__).parents[1] / "catalog" / "migrations" / "0012_daily_report_repair_requests.sql"
)
YOUTUBE_RECOVERY_MIGRATION_PATH = (
    Path(__file__).parents[1] / "catalog" / "migrations" / "0013_youtube_quarantine_releases.sql"
)
JEV_SHADOW_MIGRATION_PATH = (
    Path(__file__).parents[1] / "catalog" / "migrations" / "0014_jev_relevance_shadow.sql"
)
H3_REFERENCE_MIGRATION_PATH = (
    Path(__file__).parents[1] / "catalog" / "migrations" / "0015_video_digest_h3_references.sql"
)


def test_sqlite_catalog_covers_the_production_catalog_tables() -> None:
    production_tables = set(
        re.findall(
            r"CREATE TABLE\s+(\w+)",
            MIGRATION_PATH.read_text()
            + RECOVERY_MIGRATION_PATH.read_text()
            + REPAIR_MIGRATION_PATH.read_text()
            + YOUTUBE_RECOVERY_MIGRATION_PATH.read_text()
            + JEV_SHADOW_MIGRATION_PATH.read_text()
            + H3_REFERENCE_MIGRATION_PATH.read_text(),
            flags=re.IGNORECASE,
        )
    ) - {"news_schema_migrations"}
    fixture_tables = set(
        re.findall(
            r"CREATE TABLE(?: IF NOT EXISTS)?\s+(\w+)",
            SCHEMA_PATH.read_text(),
            flags=re.IGNORECASE,
        )
    )

    assert fixture_tables == production_tables


def test_youtube_clip_checkpoints_are_source_scoped_unique_and_immutable() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA_PATH.read_text())
        for artifact_id, version_id in (
            ("poll", "poll-v1"),
            ("clip", "clip-v1"),
            ("other", "clip-v2"),
        ):
            _insert_youtube_schema_artifact(connection, artifact_id, version_id)
        connection.execute(
            "INSERT INTO youtube_videos (source_id, video_id, first_poll_version_id, state, published_at, title) VALUES ('recorder-youtube', 'abcdefghijk', 'poll-v1', 'pending', 'now', 'Recorder')"
        )
        connection.execute(
            "INSERT INTO news_model_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "attempt-1",
                "request-1",
                "news.youtube.extract_clip",
                "response-1",
                "gemini-3.6-flash",
                10,
                5,
                0.01,
                100,
                "accepted",
                None,
                "now",
            ),
        )
        values = (
            "clip-v1",
            "recorder-youtube",
            "abcdefghijk",
            "poll-v1",
            0,
            300,
            "gemini-3.6-flash",
            "a" * 64,
            "attempt-1",
            "now",
        )
        connection.execute(
            "INSERT INTO youtube_clip_analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            connection.execute(
                "INSERT INTO youtube_clip_analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("clip-v2", *values[1:]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE youtube_clip_analyses SET duration_seconds = 299")


def test_news_aliases_and_article_metadata_are_immutable() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:article:a",
                "news_article",
                "Article",
                "source",
                "current",
                "private",
                None,
                "2026-08-31T09:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            ("article-v1", "news:article:a", 1, "digest-v1", None, "2026-08-31T09:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO news_article_aliases VALUES (?, ?, ?, ?)",
            (
                "url:https://example.test/a",
                "news:article:a",
                "canonical_url",
                "2026-08-31T09:00:00+00:00",
            ),
        )

        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT INTO news_article_aliases VALUES (?, ?, ?, ?)",
                (
                    "url:https://example.test/a",
                    "news:article:b",
                    "canonical_url",
                    "2026-08-31T09:00:00+00:00",
                ),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE news_article_aliases SET first_seen_at = ?",
                ("2026-09-01T09:00:00+00:00",),
            )


def test_news_article_failure_attempts_are_append_only() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts "
            "(id, kind, title, authority_class, lifecycle_state, visibility, "
            "current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:feed:hotnews",
                "news_feed",
                "HotNews",
                "source",
                "current",
                "private",
                None,
                "2026-09-09T08:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            (
                "feed-v1",
                "news:feed:hotnews",
                1,
                "feed-digest",
                None,
                "2026-09-09T08:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO news_dlt_loads VALUES (?, ?, ?)",
            ("load-1", "feed-v1", "2026-09-09T08:00:00+00:00"),
        )
        event_id = "a" * 64
        connection.execute(
            "INSERT INTO news_feed_entry_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                "load-1",
                "registry-v1",
                "feed-v1",
                "hotnews",
                "source-1",
                "https://hotnews.ro/source-1",
                "2026-09-09T07:00:00+00:00",
                None,
                "2026-09-09T08:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO news_feed_entry_event_versions VALUES (?, ?)",
            (event_id, event_id),
        )
        values = (
            "attempt-1",
            event_id,
            "git:test",
            "c" * 64,
            "run-1",
            0,
            "deterministic",
            "b" * 64,
            "invalid article",
            "2026-09-09T09:00:00+00:00",
            "2026-09-09T09:15:00+00:00",
        )
        insert = (
            "INSERT OR IGNORE INTO news_article_failure_attempts VALUES "
            + "("
            + ", ".join("?" for _ in values)
            + ")"
        )
        connection.execute(insert, values)
        connection.execute(insert, values)

        assert connection.execute(
            "SELECT count(*) FROM news_article_failure_attempts"
        ).fetchone() == (1,)
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(insert, (*values[:8], "different error", *values[9:]))
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(insert, ("different-id", *values[1:]))
        connection.execute(insert, ("attempt-2", *values[1:5], 1, *values[6:]))
        assert connection.execute(
            "SELECT retry_number FROM news_article_failure_attempts ORDER BY retry_number"
        ).fetchall() == [(0,), (1,)]
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(insert, ("attempt-3", "f" * 64, *values[2:5], 2, *values[6:]))
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE news_article_failure_attempts SET retry_at = ?",
                ("2026-09-09T10:00:00+00:00",),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM news_article_failure_attempts")


def test_article_recovery_overrides_are_append_only() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts "
            "(id, kind, title, authority_class, lifecycle_state, visibility, "
            "current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:feed:hotnews",
                "news_feed",
                "HotNews",
                "source",
                "current",
                "private",
                None,
                "2026-09-09T08:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            ("feed-v1", "news:feed:hotnews", 1, "feed-digest", None, "2026-09-09T08:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO news_dlt_loads VALUES (?, ?, ?)",
            ("load-1", "feed-v1", "2026-09-09T08:00:00+00:00"),
        )
        event_id = "a" * 64
        connection.execute(
            "INSERT INTO news_feed_entry_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                "load-1",
                "registry-v1",
                "feed-v1",
                "hotnews",
                "source-1",
                "https://hotnews.ro/source-1",
                "2026-09-09T07:00:00+00:00",
                None,
                "2026-09-09T08:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO news_feed_entry_event_versions VALUES (?, ?)", (event_id, event_id)
        )
        values = (
            "b" * 64,
            event_id,
            "c" * 64,
            "d" * 64,
            "operator",
            "Parser updated",
            "2026-09-09T10:00:00+00:00",
        )
        insert = (
            "INSERT OR IGNORE INTO news_article_recovery_overrides "
            "(recovery_id, event_id, base_work_generation, expected_work_generation, "
            "requested_by, reason, requested_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        connection.execute(insert, values)
        connection.execute(insert, values)
        assert connection.execute(
            "SELECT count(*) FROM news_article_recovery_overrides"
        ).fetchone() == (1,)
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(insert, (*values[:5], "different reason", values[6]))
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(insert, ("e" * 64, *values[1:]))
        connection.execute(insert, ("e" * 64, *values[1:3], "f" * 64, *values[4:]))
        sequences = connection.execute(
            "SELECT recovery_sequence FROM news_article_recovery_overrides "
            "ORDER BY recovery_sequence"
        ).fetchall()
        assert len(sequences) == 2
        assert sequences[0][0] < sequences[1][0]
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(insert, ("f" * 64, "e" * 64, *values[2:]))
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE news_article_recovery_overrides SET reason = 'other'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM news_article_recovery_overrides")


def test_news_feed_observations_cannot_be_rewritten() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:feed:digi24",
                "news_feed",
                "Digi24",
                "source",
                "current",
                "private",
                None,
                "2026-08-31T09:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            ("feed-v1", "news:feed:digi24", 1, "digest-v1", None, "2026-08-31T09:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO news_feed_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "feed-v1",
                "feed-v1",
                "digi24",
                "2026-08-31T12:00:00+03:00",
                "ok",
                200,
                177,
                "2026-08-31T09:00:00+00:00",
                120,
                None,
            ),
        )

        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "DELETE FROM news_feed_observations WHERE artifact_version_id = 'feed-v1'"
            )


def test_news_model_attempts_are_idempotent_and_immutable() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        values = (
            "attempt-1",
            "request-1",
            "news.relevance",
            "response-1",
            "test/model",
            10,
            5,
            0.01,
            120,
            "accepted",
            None,
            "2026-08-31T09:00:00+00:00",
        )
        connection.execute(
            "INSERT INTO news_model_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values
        )
        connection.execute(
            "INSERT OR IGNORE INTO news_model_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )

        assert connection.execute("SELECT count(*) FROM news_model_attempts").fetchone() == (1,)
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO news_model_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*values[:9], "rejected", "invalid evidence", values[11]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE news_model_attempts SET latency_ms = 121 WHERE attempt_id = 'attempt-1'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM news_model_attempts WHERE attempt_id = 'attempt-1'")


def test_youtube_receipts_allow_rejection_before_usage_is_known() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts "
            "(id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) "
            "VALUES ('receipt', 'youtube_model_receipt', 'Receipt', 'source', 'current', "
            "'private', NULL, '2026-09-09T09:00:00+00:00')"
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES "
            "('receipt-v1', 'receipt', 1, 'digest', NULL, '2026-09-09T09:00:00+00:00')"
        )
        connection.execute(
            "INSERT INTO youtube_model_receipts "
            "(request_id, operation_key, attempt_index, artifact_version_id, requested_model, "
            "http_status, status, received_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "request-1",
                "news.youtube.extract_clip",
                0,
                "receipt-v1",
                "gemini-test",
                200,
                "unhandled",
                "2026-09-09T09:00:00+00:00",
            ),
        )

        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO youtube_model_receipts "
                "(request_id, operation_key, attempt_index, artifact_version_id, requested_model, "
                "http_status, status, received_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "request-1",
                    "news.youtube.extract_clip",
                    0,
                    "receipt-v1",
                    "gemini-test",
                    201,
                    "unhandled",
                    "2026-09-09T09:00:00+00:00",
                ),
            )

        connection.execute(
            "UPDATE youtube_model_receipts SET status = 'rejected', error = 'invalid JSON', "
            "handled_at = '2026-09-09T09:01:00+00:00' WHERE request_id = 'request-1'"
        )
        assert connection.execute(
            "SELECT status, handled_attempt_id, error FROM youtube_model_receipts"
        ).fetchone() == ("rejected", None, "invalid JSON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE youtube_model_receipts SET status = 'accepted', error = NULL "
                "WHERE request_id = 'request-1'"
            )


def test_news_feedback_enforces_identity_locators_and_immutability() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:daily:2026-08-31",
                "news_daily_report",
                "Daily report",
                "derived",
                "current",
                "private",
                None,
                "2026-08-31T09:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            (
                "report-v1",
                "news:daily:2026-08-31",
                1,
                "report-digest",
                None,
                "2026-08-31T09:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:article:a",
                "news_article",
                "Article",
                "source",
                "current",
                "private",
                None,
                "2026-08-31T09:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            (
                "article-v1",
                "news:article:a",
                1,
                "article-digest",
                None,
                "2026-08-31T09:00:00+00:00",
            ),
        )
        values = (
            "00000000-0000-4000-8000-000000000001",
            "report-v1",
            "article",
            None,
            "group-1",
            "article-v1",
            "positive",
            None,
            "owner",
            "2026-08-31T10:00:00+00:00",
        )
        connection.execute(
            "INSERT INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values
        )
        connection.execute(
            "INSERT OR IGNORE INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*values[:9], "2026-08-31T11:00:00+00:00"),
        )

        assert connection.execute(
            "SELECT count(*), created_at FROM news_feedback WHERE feedback_id = ?",
            (values[0],),
        ).fetchone() == (1, "2026-08-31T10:00:00+00:00")
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*values[:6], "negative", *values[7:]),
            )
        theme_feedback = (
            "00000000-0000-4000-8000-000000000002",
            "report-v1",
            "theme",
            "theme-1",
            None,
            None,
            "positive",
            None,
            "owner",
            "2026-08-31T10:00:00+00:00",
        )
        connection.execute(
            "INSERT INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            theme_feedback,
        )
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*theme_feedback[:3], "theme-2", *theme_feedback[4:]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE news_feedback SET rating = 'negative' WHERE feedback_id = ?",
                (values[0],),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM news_feedback WHERE feedback_id = ?", (values[0],))

        invalid_values = (
            ("report", None, "group-1", None, "positive", None, "owner"),
            ("theme", None, None, None, "positive", None, "owner"),
            ("group", "theme-1", "group-1", None, "positive", None, "owner"),
            ("article", None, "group-1", None, "positive", None, "owner"),
            ("report", None, None, None, "neutral", None, "owner"),
            ("report", None, None, None, "positive", "x" * 2001, "owner"),
            ("report", None, None, None, "positive", None, "someone-else"),
        )
        for position, (kind, theme_id, group_id, article_id, rating, note, actor) in enumerate(
            invalid_values
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"invalid-{position}",
                        "report-v1",
                        kind,
                        theme_id,
                        group_id,
                        article_id,
                        rating,
                        note,
                        actor,
                        "2026-08-31T10:00:00+00:00",
                    ),
                )


def test_news_feedback_requires_a_rating_or_non_empty_note() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:daily:test",
                "news_daily_report",
                "Report",
                "derived",
                "current",
                "private",
                None,
                "2026-09-01",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            ("report-v1", "news:daily:test", 1, "digest", None, "2026-09-01"),
        )
        note_only = (
            "feedback-note",
            "report-v1",
            "report",
            None,
            None,
            None,
            None,
            "Needs a source link.",
            "owner",
            "2026-09-01T10:00:00+00:00",
        )

        connection.execute(
            "INSERT INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            note_only,
        )
        connection.execute(
            "INSERT OR IGNORE INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*note_only[:9], "2026-09-01T11:00:00+00:00"),
        )

        assert connection.execute(
            "SELECT rating, note FROM news_feedback WHERE feedback_id = 'feedback-note'"
        ).fetchone() == (None, "Needs a source link.")
        for feedback_id, note in (("feedback-empty", None), ("feedback-space", "   ")):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        feedback_id,
                        "report-v1",
                        "report",
                        None,
                        None,
                        None,
                        None,
                        note,
                        "owner",
                        "2026-09-01T10:00:00+00:00",
                    ),
                )
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*note_only[:7], "Different note.", *note_only[8:]),
            )


def test_news_evaluation_feedback_sources_are_idempotent_and_immutable() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:evaluation:test",
                "news_evaluation_manifest",
                "Evaluation",
                "derived",
                "current",
                "private",
                None,
                "2026-09-04",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            ("evaluation-v1", "news:evaluation:test", 1, "digest", None, "2026-09-04"),
        )
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:daily:test",
                "news_daily_report",
                "Report",
                "derived",
                "current",
                "private",
                None,
                "2026-09-04",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            ("report-v1", "news:daily:test", 1, "report-digest", None, "2026-09-04"),
        )
        connection.execute(
            "INSERT INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "feedback-1",
                "report-v1",
                "report",
                None,
                None,
                None,
                "negative",
                None,
                "owner",
                "2026-09-04",
            ),
        )
        values = ("evaluation-v1", 0, "feedback-1", "represented")
        connection.execute(
            "INSERT INTO news_evaluation_feedback_sources VALUES (?, ?, ?, ?)", values
        )
        connection.execute(
            "INSERT OR IGNORE INTO news_evaluation_feedback_sources VALUES (?, ?, ?, ?)", values
        )

        assert connection.execute(
            "SELECT count(*) FROM news_evaluation_feedback_sources"
        ).fetchone() == (1,)
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO news_evaluation_feedback_sources VALUES (?, ?, ?, ?)",
                ("evaluation-v1", 0, "feedback-1", "excluded"),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE news_evaluation_feedback_sources SET disposition = 'excluded'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM news_evaluation_feedback_sources")


def test_news_evaluation_projections_are_idempotent_and_immutable() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:evaluation:test",
                "news_evaluation_manifest",
                "Evaluation",
                "derived",
                "current",
                "private",
                None,
                "2026-09-04",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            ("evaluation-v1", "news:evaluation:test", 1, "digest", None, "2026-09-04"),
        )
        dataset = (
            "langfuse",
            "evaluation-v1",
            "dataset",
            "00000000-0000-4000-8000-000000000001",
            "chartly-news-evaluation-v1",
            None,
            None,
            None,
            "",
            "2026-09-04T10:00:00+00:00",
        )
        connection.execute(
            "INSERT INTO news_evaluation_projections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            dataset,
        )
        connection.execute(
            "INSERT OR IGNORE INTO news_evaluation_projections "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*dataset[:9], "2026-09-04T11:00:00+00:00"),
        )

        assert connection.execute(
            "SELECT count(*) FROM news_evaluation_projections"
        ).fetchone() == (1,)
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO news_evaluation_projections "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*dataset[:4], "another-name", *dataset[5:]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE news_evaluation_projections " "SET dataset_name = 'changed'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM news_evaluation_projections")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO news_evaluation_projections " "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "langfuse",
                    "evaluation-v1",
                    "experiment",
                    "00000000-0000-4000-8000-000000000001",
                    "chartly-news-evaluation-v1",
                    None,
                    None,
                    None,
                    "git:test",
                    "2026-09-04T10:00:00+00:00",
                ),
            )


def test_news_evaluation_projection_experiments_are_keyed_by_implementation() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:evaluation:test",
                "news_evaluation_manifest",
                "Evaluation",
                "derived",
                "current",
                "private",
                None,
                "2026-09-04",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            ("evaluation-v1", "news:evaluation:test", 1, "digest", None, "2026-09-04"),
        )
        for position, implementation_ref in enumerate(("git:first", "git:second"), start=1):
            connection.execute(
                "INSERT INTO news_evaluation_projections " "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "langfuse",
                    "evaluation-v1",
                    "experiment",
                    "00000000-0000-4000-8000-000000000001",
                    "chartly-news-evaluation-v1",
                    f"00000000-0000-4000-9000-{position:012d}",
                    f"experiment-{position}",
                    None,
                    implementation_ref,
                    "2026-09-04T10:00:00+00:00",
                ),
            )

        assert connection.execute(
            "SELECT implementation_ref FROM news_evaluation_projections "
            "ORDER BY implementation_ref"
        ).fetchall() == [("git:first",), ("git:second",)]
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO news_evaluation_projections "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "langfuse",
                    "evaluation-v1",
                    "experiment",
                    "00000000-0000-4000-8000-000000000001",
                    "chartly-news-evaluation-v1",
                    "00000000-0000-4000-9000-000000000099",
                    "changed",
                    None,
                    "git:first",
                    "2026-09-04T11:00:00+00:00",
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO news_evaluation_projections " "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "langfuse",
                    "evaluation-v1",
                    "experiment",
                    "00000000-0000-4000-8000-000000000001",
                    "chartly-news-evaluation-v1",
                    "00000000-0000-4000-9000-000000000100",
                    "empty-implementation",
                    None,
                    "",
                    "2026-09-04T11:00:00+00:00",
                ),
            )


def test_news_relevance_experiments_are_policy_specific_and_immutable() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:evaluation:test",
                "news_evaluation_manifest",
                "Evaluation",
                "derived",
                "current",
                "private",
                None,
                "2026-09-04",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            ("evaluation-v1", "news:evaluation:test", 1, "digest", None, "2026-09-04"),
        )
        first = (
            "langfuse",
            "evaluation-v1",
            "1" * 64,
            "git:same",
            "relevance-v1",
            "dataset-id",
            "dataset-name",
            "experiment-id-1",
            "experiment-name-1",
            "https://example.test/experiments/1",
            "2026-09-04T10:00:00+00:00",
        )
        second = (
            "langfuse",
            "evaluation-v1",
            "2" * 64,
            "git:same",
            "relevance-v2",
            "dataset-id",
            "dataset-name",
            "experiment-id-2",
            "experiment-name-2",
            None,
            "2026-09-04T10:00:00+00:00",
        )
        connection.executemany(
            "INSERT INTO news_relevance_experiments " "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (first, second),
        )
        connection.execute(
            "INSERT OR IGNORE INTO news_relevance_experiments "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*first[:10], "2026-09-04T11:00:00+00:00"),
        )

        assert connection.execute(
            "SELECT policy_digest FROM news_relevance_experiments " "ORDER BY policy_digest"
        ).fetchall() == [("1" * 64,), ("2" * 64,)]
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO news_relevance_experiments "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*first[:4], "changed-policy", *first[5:]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE news_relevance_experiments " "SET dataset_name = 'changed'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM news_relevance_experiments")


def test_news_model_trace_links_are_idempotent_and_immutable() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        attempt = (
            "attempt-1",
            "request-1",
            "news.relevance",
            "response-1",
            "test/model",
            10,
            5,
            0.01,
            120,
            "accepted",
            None,
            "2026-09-01T09:00:00+00:00",
        )
        connection.execute(
            "INSERT INTO news_model_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            attempt,
        )
        link = (
            "attempt-1",
            "langfuse",
            "018f0000-0000-7000-8000-000000000001",
            "018f0000-0000-7000-8000-000000000002",
            "chartly-romanian-news",
            "2026-09-01T09:00:01+00:00",
        )
        connection.execute("INSERT INTO news_model_trace_links VALUES (?, ?, ?, ?, ?, ?)", link)
        connection.execute(
            "INSERT OR IGNORE INTO news_model_trace_links VALUES (?, ?, ?, ?, ?, ?)", link
        )

        assert connection.execute("SELECT count(*) FROM news_model_trace_links").fetchone() == (1,)
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO news_model_trace_links VALUES (?, ?, ?, ?, ?, ?)",
                (*link[:3], "other-observation", *link[4:]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE news_model_trace_links SET project_ref = 'other' "
                "WHERE attempt_id = 'attempt-1'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM news_model_trace_links WHERE attempt_id = 'attempt-1'")


def test_news_feedback_sync_attempts_are_append_only_and_constrained() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:daily:test",
                "news_daily_report",
                "Report",
                "derived",
                "current",
                "private",
                None,
                "2026-09-01",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            ("report-v1", "news:daily:test", 1, "digest", None, "2026-09-01"),
        )
        connection.execute(
            "INSERT INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "feedback-1",
                "report-v1",
                "report",
                None,
                None,
                None,
                "positive",
                None,
                "owner",
                "2026-09-01",
            ),
        )
        connection.execute(
            "INSERT INTO news_model_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "attempt-1",
                "request-1",
                "news.summary",
                "response-1",
                "test/model",
                1,
                1,
                0,
                1,
                "accepted",
                None,
                "2026-09-01",
            ),
        )
        values = (
            "sync-1",
            "feedback-1",
            "attempt-1",
            "langfuse",
            "00000000-0000-4000-8000-000000000001",
            "failed",
            "RuntimeError: unavailable",
            "2026-09-01T10:00:00+00:00",
        )
        connection.execute(
            "INSERT INTO news_feedback_sync_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", values
        )
        connection.execute(
            "INSERT OR IGNORE INTO news_feedback_sync_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )

        assert connection.execute(
            "SELECT count(*) FROM news_feedback_sync_attempts"
        ).fetchone() == (1,)
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO news_feedback_sync_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (*values[:5], "completed", None, values[7]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE news_feedback_sync_attempts SET error = 'changed' WHERE sync_attempt_id = 'sync-1'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "DELETE FROM news_feedback_sync_attempts WHERE sync_attempt_id = 'sync-1'"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO news_feedback_sync_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "sync-2",
                    "feedback-1",
                    "attempt-1",
                    "langfuse",
                    values[4],
                    "completed",
                    "error",
                    values[7],
                ),
            )


def test_news_feedback_sync_dispositions_are_immutable_and_constrained() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:daily:test",
                "news_daily_report",
                "Report",
                "derived",
                "current",
                "private",
                None,
                "2026-09-01",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            ("report-v1", "news:daily:test", 1, "digest", None, "2026-09-01"),
        )
        connection.execute(
            "INSERT INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "feedback-1",
                "report-v1",
                "report",
                None,
                None,
                None,
                "positive",
                None,
                "owner",
                "2026-09-01",
            ),
        )
        values = (
            "feedback-1",
            "langfuse",
            "permanently_unresolved",
            "provider_migration",
            "langsmith",
            "2026-09-01T10:00:00+00:00",
        )
        connection.execute(
            "INSERT INTO news_feedback_sync_dispositions VALUES (?, ?, ?, ?, ?, ?)", values
        )
        connection.execute(
            "INSERT OR IGNORE INTO news_feedback_sync_dispositions VALUES (?, ?, ?, ?, ?, ?)",
            values,
        )

        assert connection.execute(
            "SELECT count(*) FROM news_feedback_sync_dispositions"
        ).fetchone() == (1,)
        connection.execute(
            "INSERT OR IGNORE INTO news_feedback_sync_dispositions VALUES (?, ?, ?, ?, ?, ?)",
            (*values[:5], "2026-09-01T11:00:00+00:00"),
        )
        assert connection.execute(
            "SELECT recorded_at FROM news_feedback_sync_dispositions"
        ).fetchone() == (values[5],)
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO news_feedback_sync_dispositions VALUES (?, ?, ?, ?, ?, ?)",
                (*values[:3], "another_reason", *values[4:]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE news_feedback_sync_dispositions SET recorded_at = 'changed' "
                "WHERE feedback_id = 'feedback-1'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "DELETE FROM news_feedback_sync_dispositions WHERE feedback_id = 'feedback-1'"
            )
        for feedback_number in range(2, 7):
            connection.execute(
                "INSERT INTO news_feedback SELECT ?, report_version_id, target_kind, theme_id, "
                "group_id, article_version_id, rating, note, actor, created_at "
                "FROM news_feedback WHERE feedback_id = 'feedback-1'",
                [f"feedback-{feedback_number}"],
            )
        invalid_values = (
            ("feedback-2", "langsmith", *values[2:]),
            ("feedback-3", "langfuse", "retryable", *values[3:]),
            ("feedback-4", "langfuse", values[2], "missing_trace", *values[4:]),
            ("feedback-5", "langfuse", *values[2:4], "langfuse", values[5]),
            ("feedback-6", *values[1:5], "  "),
        )
        for invalid in invalid_values:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO news_feedback_sync_dispositions VALUES (?, ?, ?, ?, ?, ?)",
                    invalid,
                )
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                "INSERT INTO news_feedback_sync_dispositions VALUES (?, ?, ?, ?, ?, ?)",
                ("missing-feedback", *values[1:]),
            )


def test_news_feed_entry_events_retain_one_occurrence_and_reject_identity_changes() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:feed:hotnews",
                "news_feed",
                "HotNews",
                "source",
                "current",
                "private",
                None,
                "2026-08-31T09:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            ("feed-v1", "news:feed:hotnews", 1, "digest-v1", None, "2026-08-31T09:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            ("feed-v2", "news:feed:hotnews", 2, "digest-v2", None, "2026-08-31T10:00:00+00:00"),
        )
        for load_id in ("load-1", "load-2"):
            connection.execute(
                "INSERT INTO news_dlt_loads VALUES (?, ?, ?)",
                (load_id, "feed-v1", "2026-08-31T09:00:00+00:00"),
            )
        values = (
            "event-1",
            "load-1",
            "registry-v1",
            "feed-v1",
            "hotnews",
            "source-1",
            "https://hotnews.ro/source-1",
            "2026-08-31T08:00:00+00:00",
            None,
            "2026-08-31T09:00:00+00:00",
        )
        connection.execute(
            "INSERT INTO news_feed_entry_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        connection.execute(
            "INSERT OR IGNORE INTO news_feed_entry_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                values[0],
                "load-2",
                "registry-v2",
                "feed-v2",
                *values[4:9],
                "2026-08-31T10:00:00+00:00",
            ),
        )

        assert connection.execute(
            "SELECT dlt_load_id, observed_at FROM news_feed_entry_events"
        ).fetchone() == ("load-1", "2026-08-31T09:00:00+00:00")
        connection.execute(
            "INSERT OR IGNORE INTO news_feed_entry_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*values[:2], "registry-v2", *values[3:]),
        )
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO news_feed_entry_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*values[:5], "source-2", *values[6:]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE news_feed_entry_events SET source_id = 'changed' WHERE event_id = 'event-1'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM news_feed_entry_events WHERE event_id = 'event-1'")


def test_news_feed_entry_projection_loads_are_idempotent_and_immutable() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "news:dlt-load:test",
                "news_dlt_load",
                "Load",
                "derived",
                "current",
                "private",
                None,
                "2026-09-01T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            (
                "load-version",
                "news:dlt-load:test",
                1,
                "digest",
                None,
                "2026-09-01T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO news_dlt_loads VALUES (?, ?, ?)",
            ("load-1", "load-version", "2026-09-01T00:00:00+00:00"),
        )
        values = ("load-1", "2026-09-01T00:01:00+00:00")
        connection.execute("INSERT INTO news_feed_entry_projection_loads VALUES (?, ?)", values)
        connection.execute(
            "INSERT OR IGNORE INTO news_feed_entry_projection_loads VALUES (?, ?)", values
        )

        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO news_feed_entry_projection_loads VALUES (?, ?)",
                ("load-1", "2026-09-01T00:02:00+00:00"),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE news_feed_entry_projection_loads SET projected_at = ?",
                ("2026-09-01T00:03:00+00:00",),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM news_feed_entry_projection_loads")


def test_youtube_video_state_constraints_reject_invalid_combinations() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        _insert_youtube_schema_artifact(connection, "poll", "poll-v1")
        _insert_youtube_schema_artifact(connection, "candidate", "candidate-v1")
        invalid = (
            ("aaaaaaaaaaa", "running", None, None, None, 0),
            ("bbbbbbbbbbb", "pending", "owner", "later", None, 0),
            ("ccccccccccc", "deferred", None, None, None, 0),
            ("ddddddddddd", "quarantined", None, None, None, 2),
            ("eeeeeeeeeee", "candidate", None, None, None, 0),
            ("fffffffffff", "pending", None, None, "candidate-v1", 0),
        )
        for video in invalid:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO youtube_videos (source_id, video_id, first_poll_version_id, state, published_at, title, owner_token, lease_expires_at, candidate_version_id, unchanged_deterministic_failures) VALUES ('recorder-youtube', ?, 'poll-v1', ?, 'now', 'Video', ?, ?, ?, ?)",
                    video,
                )


def test_youtube_decisions_and_publications_are_immutable_and_conflict_safe() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        for artifact_id, version_id in (
            ("poll", "poll-v1"),
            ("candidate", "candidate-v1"),
            ("decision", "decision-v1"),
            ("other-decision", "decision-v2"),
            ("article", "article-v1"),
        ):
            _insert_youtube_schema_artifact(connection, artifact_id, version_id)
        connection.execute(
            "INSERT INTO youtube_videos (source_id, video_id, first_poll_version_id, state, published_at, title, owner_token, lease_expires_at) VALUES ('starea-impostorilor-youtube', 'abcdefghijk', 'poll-v1', 'running', 'now', 'Starea', 'owner', '2099-01-01T00:00:00+00:00')"
        )
        with pytest.raises(sqlite3.IntegrityError, match="candidate lease was lost"):
            connection.execute(
                "INSERT INTO youtube_candidates VALUES ('candidate-v1', 'starea-impostorilor-youtube', 'abcdefghijk', 'owner_review', 'poll-v1', 'poll-v1', 'stale-owner', '2026-01-01T00:00:00+00:00')"
            )
        connection.execute(
            "INSERT INTO youtube_candidates VALUES ('candidate-v1', 'starea-impostorilor-youtube', 'abcdefghijk', 'owner_review', 'poll-v1', 'poll-v1', 'owner', '2026-01-01T00:00:00+00:00')"
        )
        connection.execute(
            "UPDATE youtube_videos SET state = 'candidate', candidate_version_id = 'candidate-v1', owner_token = NULL, lease_expires_at = NULL"
        )
        with pytest.raises(sqlite3.IntegrityError, match="requires exact approval"):
            connection.execute(
                "INSERT INTO youtube_publications VALUES ('candidate-v1', 'starea-impostorilor-youtube', 'abcdefghijk', 'article-v1', 'now')"
            )
        with pytest.raises(sqlite3.IntegrityError, match="violates source policy"):
            connection.execute(
                "INSERT INTO youtube_candidate_decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "decision-v2",
                    "candidate-v1",
                    "starea-impostorilor-youtube",
                    "automatic_approved",
                    "system",
                    None,
                    "now",
                ),
            )
        decision = (
            "decision-v1",
            "candidate-v1",
            "starea-impostorilor-youtube",
            "owner_approved",
            "owner",
            "Useful context",
            "now",
        )
        connection.execute(
            "INSERT INTO youtube_candidate_decisions VALUES (?, ?, ?, ?, ?, ?, ?)", decision
        )
        connection.execute(
            "INSERT OR IGNORE INTO youtube_candidate_decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
            decision,
        )
        with pytest.raises(sqlite3.IntegrityError, match="decision conflict"):
            connection.execute(
                "INSERT OR IGNORE INTO youtube_candidate_decisions VALUES ('decision-v2', 'candidate-v1', 'starea-impostorilor-youtube', 'owner_rejected', 'owner', 'Useful context', 'later')"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE youtube_candidate_decisions SET kind = 'owner_rejected'")
        connection.execute(
            "INSERT INTO youtube_publications VALUES ('candidate-v1', 'starea-impostorilor-youtube', 'abcdefghijk', 'article-v1', 'later')"
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM youtube_publications")


def _insert_youtube_schema_artifact(
    connection: sqlite3.Connection, artifact_id: str, version_id: str
) -> None:
    connection.execute(
        "INSERT INTO artifacts "
        "(id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
        "VALUES (?, 'test', ?, 'derived', 'current', 'private', 'now')",
        (artifact_id, artifact_id),
    )
    connection.execute(
        "INSERT INTO artifact_versions VALUES (?, ?, 1, ?, NULL, 'now')",
        (version_id, artifact_id, f"digest:{version_id}"),
    )
