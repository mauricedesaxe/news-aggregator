from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest

import romanian_news.catalog.schema as news_schema
import romanian_news.catalog_transport as catalog_transport
from romanian_news.catalog.schema import ensure_news_catalog_schema


def test_catalog_transaction_commits_the_callback_result(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    recorded_at = datetime.now(UTC)

    def record_artifact(connection: catalog_transport.CatalogConnection) -> str:
        connection.execute(
            "INSERT INTO artifacts "
            "(id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
            "VALUES (%s, 'test', 'Committed', 'test', 'active', 'private', %s)",
            ("artifact-commit", recorded_at),
        )
        return "recorded"

    assert catalog_transport.catalog_transaction(record_artifact) == "recorded"
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        total = connection.execute(
            "SELECT count(*) FROM artifacts WHERE id = 'artifact-commit'"
        ).fetchone()
    assert total == (1,)


def test_catalog_transaction_rolls_back_when_the_callback_fails(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    recorded_at = datetime.now(UTC)

    def record_then_fail(connection: catalog_transport.CatalogConnection) -> None:
        connection.execute(
            "INSERT INTO artifacts "
            "(id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
            "VALUES (%s, 'test', 'Rolled back', 'test', 'active', 'private', %s)",
            ("artifact-rollback", recorded_at),
        )
        raise ValueError("invalid checkpoint")

    with pytest.raises(ValueError, match="invalid checkpoint"):
        catalog_transport.catalog_transaction(record_then_fail)

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        total = connection.execute("SELECT count(*) FROM artifacts").fetchone()
    assert total is not None
    assert total[0] == 0


def test_catalog_query_serves_rows_with_utc_datetimes(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "INSERT INTO artifacts "
            "(id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
            "VALUES ('artifact-query', 'test', 'Queried', 'test', 'active', 'private', "
            "TIMESTAMPTZ '2026-09-21T08:00:00+02:00')"
        )

    rows = catalog_transport.catalog_query(
        "SELECT created_at FROM artifacts WHERE id = %s", ["artifact-query"]
    )

    assert len(rows) == 1
    assert rows[0]["created_at"] == "2026-09-21T06:00:00+00:00"
