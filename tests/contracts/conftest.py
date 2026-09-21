from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

import romanian_news.catalog.schema as news_schema
import romanian_news.catalog_transport as catalog_transport

TEST_POSTGRES_DSN = os.getenv("NEWS_TEST_POSTGRES_DSN")


@pytest.fixture
def postgres_news_schema(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    if TEST_POSTGRES_DSN is None:
        raise RuntimeError("NEWS_TEST_POSTGRES_DSN is required")
    schema = f"news_schema_contract_{uuid4().hex}"
    with psycopg.connect(TEST_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    fixture_dsn = make_conninfo(TEST_POSTGRES_DSN, options=f"-csearch_path={schema}")
    monkeypatch.setattr(news_schema, "NEWS_POSTGRES_DSN", fixture_dsn)
    monkeypatch.setattr(catalog_transport, "NEWS_POSTGRES_DSN", fixture_dsn)
    try:
        yield schema
    finally:
        with psycopg.connect(TEST_POSTGRES_DSN, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
