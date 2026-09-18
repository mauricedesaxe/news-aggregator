from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

import romanian_news.catalog.schema as news_schema
from romanian_news.catalog.schema import NewsCatalogSchemaError, ensure_news_catalog_schema

TEST_POSTGRES_DSN = os.getenv("NEWS_TEST_POSTGRES_DSN")


@pytest.fixture
def postgres_news_schema(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    if TEST_POSTGRES_DSN is None:
        raise RuntimeError("NEWS_TEST_POSTGRES_DSN is required")
    schema = f"news_schema_contract_{uuid4().hex}"
    with psycopg.connect(TEST_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    monkeypatch.setattr(
        news_schema,
        "NEWS_POSTGRES_DSN",
        make_conninfo(TEST_POSTGRES_DSN, options=f"-csearch_path={schema}"),
    )
    try:
        yield schema
    finally:
        with psycopg.connect(TEST_POSTGRES_DSN, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_news_schema_installs_and_verifies_again(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = current_schema()"
            ).fetchall()
        }
        migrations = connection.execute(
            "SELECT version, name, sha256 FROM news_schema_migrations"
        ).fetchall()

    assert "debt_transcript_projection_items" not in tables
    assert len(tables) == 41
    assert migrations == [
        (
            1,
            "initial",
            news_schema.NEWS_CATALOG_MIGRATIONS[0].sha256,
        )
    ]


def test_news_schema_rejects_changed_migration_digest(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        connection.execute("UPDATE news_schema_migrations SET sha256 = %s", ("0" * 64,))

    with pytest.raises(NewsCatalogSchemaError, match="migration 1 differs"):
        ensure_news_catalog_schema()


def test_news_schema_rejects_disabled_triggers(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        connection.execute("ALTER TABLE runs DISABLE TRIGGER runs_protect_identity")

    with pytest.raises(NewsCatalogSchemaError, match="disabled triggers: runs_protect_identity"):
        ensure_news_catalog_schema()
