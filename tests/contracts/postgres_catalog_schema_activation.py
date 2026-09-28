"""A schema owner can install the spend ledger for a restricted runtime role."""

import os
import subprocess
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import LiteralString, cast
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from romanian_news.catalog.schema import NEWS_CATALOG_MIGRATIONS
from tests.postgres_catalog import TEST_POSTGRES_DSN, isolated_postgres_schema

pytestmark = pytest.mark.skipif(not TEST_POSTGRES_DSN, reason="NEWS_TEST_POSTGRES_DSN is required")


def test_owner_activation_grants_runtime_spend_access(monkeypatch: pytest.MonkeyPatch) -> None:
    assert TEST_POSTGRES_DSN is not None
    role = f"archive_runtime_{uuid4().hex}"
    role_identifier = sql.Identifier(role)
    with isolated_postgres_schema(monkeypatch, "archive_activation") as (schema, owner_dsn):
        schema_identifier = sql.Identifier(schema)
        with psycopg.connect(owner_dsn, autocommit=True) as connection:
            for migration in NEWS_CATALOG_MIGRATIONS[:21]:
                connection.execute(
                    sql.SQL(cast(LiteralString, migration.path.read_text())), prepare=False
                )
                connection.execute(
                    "INSERT INTO news_schema_migrations (version, name, sha256) "
                    "VALUES (%s, %s, %s)",
                    (migration.version, migration.name, migration.sha256),
                )
            connection.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'archive-test'").format(role_identifier)
            )
            connection.execute(
                sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(schema_identifier, role_identifier)
            )
            connection.execute(
                sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(
                    schema_identifier, role_identifier
                )
            )

        try:
            runtime_dsn = make_conninfo(
                TEST_POSTGRES_DSN,
                user=role,
                password="archive-test",
                options=f"-csearch_path={schema}",
            )
            script = Path(__file__).resolve().parents[2] / "scripts/activate_news_catalog_schema.py"
            environment = {
                **os.environ,
                "NEWS_POSTGRES_DSN": owner_dsn,
                "NEWS_POSTGRES_RUNTIME_DSN": runtime_dsn,
            }
            for _ in range(2):
                subprocess.run(
                    [sys.executable, str(script)],
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            with psycopg.connect(runtime_dsn) as connection:
                reservation_id = uuid4()
                connection.execute(
                    "INSERT INTO news_archive_model_reservations "
                    "(reservation_id, day, operation_key, request_id, reserved_usd) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (reservation_id, date(2025, 10, 5), "news.relevance", "request", Decimal("1")),
                )
                connection.execute(
                    "UPDATE news_archive_model_reservations "
                    "SET actual_usd = %s, settled_at = now() WHERE reservation_id = %s",
                    (Decimal("0.01"), reservation_id),
                )
                row = connection.execute(
                    "SELECT actual_usd FROM news_archive_model_reservations "
                    "WHERE reservation_id = %s",
                    (reservation_id,),
                ).fetchone()
                assert row == (Decimal("0.0100000000"),)
        finally:
            with psycopg.connect(owner_dsn, autocommit=True) as connection:
                connection.execute(
                    sql.SQL("REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA {} FROM {}").format(
                        schema_identifier, role_identifier
                    )
                )
                connection.execute(
                    sql.SQL("REVOKE ALL PRIVILEGES ON SCHEMA {} FROM {}").format(
                        schema_identifier, role_identifier
                    )
                )
                connection.execute(sql.SQL("DROP ROLE {}").format(role_identifier))
