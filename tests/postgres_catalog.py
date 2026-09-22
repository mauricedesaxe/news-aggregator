from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, LiteralString, cast
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row

import romanian_news.catalog.schema as news_schema
import romanian_news.catalog_transport as catalog_transport
from romanian_news.catalog.schema import ensure_news_catalog_schema

TEST_POSTGRES_DSN = os.getenv("NEWS_TEST_POSTGRES_DSN")


class CatalogRows:
    """Cursor-like access to the rows one executed statement returned."""

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self._rows = list(rows)

    def fetchall(self) -> list[dict[str, Any]]:
        return self._rows

    def fetchone(self) -> dict[str, Any] | None:
        return self._rows.pop(0) if self._rows else None


class PostgresCatalog:
    """Seed and assert against one isolated per-test catalog schema."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def execute(self, statement: str, parameters: Sequence[object] = ()) -> CatalogRows:
        with psycopg.connect(self._dsn, row_factory=cast(Any, dict_row)) as connection:
            cursor = connection.execute(cast(LiteralString, statement), parameters)
            rows = [dict(row) for row in cursor.fetchall()] if cursor.description else []
        return CatalogRows(rows)

    def query(
        self, statement: str, parameters: Sequence[object] | None = None
    ) -> list[dict[str, Any]]:
        return catalog_transport.catalog_query(statement, parameters)

    def batch(
        self,
        statements: Sequence[tuple[str, Sequence[object]]],
    ) -> None:
        catalog_transport.catalog_batch(statements)


def postgres_catalog_fixture(schema_prefix: str) -> Any:
    @pytest.fixture
    def postgres_catalog(monkeypatch: pytest.MonkeyPatch) -> Iterator[PostgresCatalog]:
        with isolated_postgres_schema(monkeypatch, schema_prefix) as (_, fixture_dsn):
            ensure_news_catalog_schema()
            yield PostgresCatalog(fixture_dsn)

    return postgres_catalog


@contextmanager
def isolated_postgres_schema(
    monkeypatch: pytest.MonkeyPatch, schema_prefix: str
) -> Iterator[tuple[str, str]]:
    if TEST_POSTGRES_DSN is None:
        raise RuntimeError("NEWS_TEST_POSTGRES_DSN is required")
    schema = f"{schema_prefix}_{uuid4().hex}"
    with psycopg.connect(TEST_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    fixture_dsn = make_conninfo(TEST_POSTGRES_DSN, options=f"-csearch_path={schema}")
    monkeypatch.setattr(news_schema, "NEWS_POSTGRES_DSN", fixture_dsn)
    monkeypatch.setattr(catalog_transport, "NEWS_POSTGRES_DSN", fixture_dsn)
    try:
        yield schema, fixture_dsn
    finally:
        with psycopg.connect(TEST_POSTGRES_DSN, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
