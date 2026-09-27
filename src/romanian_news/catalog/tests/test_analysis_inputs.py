import sqlite3
from datetime import date
from pathlib import Path

from romanian_news.catalog.analysis_inputs import read_current_article_analysis_references

_DIGEST = "d" * 64


def test_current_article_references_read_recent_report_days_first(monkeypatch) -> None:
    connection = _analysis_inputs_database()
    monkeypatch.setattr(
        "romanian_news.catalog.analysis_inputs.catalog_query", _sqlite_query(connection)
    )

    references = read_current_article_analysis_references()

    assert [value.bucharest_day for value in references] == [
        date(2026, 9, 1),
        date(2026, 9, 1),
        date(2026, 8, 31),
    ]
    assert [value.reference.artifact_id for value in references] == [
        "news:article:sep-first",
        "news:article:sep-second",
        "news:article:aug-last",
    ]


def _analysis_inputs_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    schema = Path(__file__).parents[4] / "tests" / "fixtures" / "sqlite_catalog.sql"
    connection.executescript(schema.read_text())
    connection.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "news:feed:hotnews",
            "news_feed_snapshot",
            "Hotnews",
            "source",
            "current",
            "public",
            "0" * 64,
            "2026-08-30T09:00:00+00:00",
        ),
    )
    connection.execute(
        "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
        ("0" * 64, "news:feed:hotnews", 1, _DIGEST, None, "2026-08-30T09:00:00+00:00"),
    )
    for name, day, version_id, created_at in (
        ("sep-first", "2026-09-01", "1" * 64, "2026-09-01T09:00:00+00:00"),
        ("sep-second", "2026-09-01", "2" * 64, "2026-09-01T10:00:00+00:00"),
        ("aug-last", "2026-08-31", "3" * 64, "2026-08-31T09:00:00+00:00"),
    ):
        artifact_id = f"news:article:{name}"
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                artifact_id,
                "news_article",
                name,
                "source",
                "current",
                "private",
                version_id,
                created_at,
            ),
        )
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            (version_id, artifact_id, 1, _DIGEST, None, created_at),
        )
        connection.execute(
            "INSERT INTO artifact_files VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"file:{version_id}",
                version_id,
                f"{name}.json",
                "application/json",
                _DIGEST,
                10,
                None,
                None,
            ),
        )
        connection.execute(
            "INSERT INTO news_article_versions "
            "(artifact_version_id, article_artifact_id, outlet_id, canonical_url, "
            "published_at, source_updated_at, bucharest_day, material_digest, "
            "extraction_digest, feed_snapshot_version_id, page_capture_version_id, captured_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, NULL, ?)",
            (
                version_id,
                artifact_id,
                "hotnews",
                f"https://hotnews.ro/{name}",
                f"{day}T08:00:00+00:00",
                day,
                f"digest:{version_id}",
                f"digest:{version_id}",
                "0" * 64,
                f"{day}T07:00:00+00:00",
            ),
        )
    connection.commit()
    return connection


def _sqlite_query(connection: sqlite3.Connection):
    def query(sql, params=None):
        return [
            dict(row) for row in connection.execute(sql.replace("%s", "?"), params or []).fetchall()
        ]

    return query
