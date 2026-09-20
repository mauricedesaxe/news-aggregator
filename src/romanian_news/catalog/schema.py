from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, LiteralString, cast

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from romanian_news.config import NEWS_POSTGRES_DSN

MIGRATIONS_PATH = Path(__file__).with_name("migrations")
_MIGRATION_LOCK_ID = 7_337_801_455_016_519_295


class NewsCatalogSchemaError(RuntimeError):
    """The Romanian news catalog schema could not be installed or verified."""


@dataclass(frozen=True)
class NewsCatalogMigration:
    version: int
    name: str
    path: Path

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


NEWS_CATALOG_MIGRATIONS = (
    NewsCatalogMigration(1, "initial", MIGRATIONS_PATH / "0001_initial.sql"),
    NewsCatalogMigration(2, "video_digest", MIGRATIONS_PATH / "0002_video_digest.sql"),
    NewsCatalogMigration(
        3,
        "video_digest_generation_fences",
        MIGRATIONS_PATH / "0003_video_digest_generation_fences.sql",
    ),
    NewsCatalogMigration(
        4,
        "video_digest_publication_evidence",
        MIGRATIONS_PATH / "0004_video_digest_publication_evidence.sql",
    ),
)


def ensure_news_catalog_schema() -> None:
    """Apply and verify the append-only Romanian news catalog migrations."""
    if NEWS_POSTGRES_DSN is None:
        raise NewsCatalogSchemaError("NEWS_POSTGRES_DSN is required")
    try:
        with psycopg.connect(NEWS_POSTGRES_DSN, row_factory=cast(Any, dict_row)) as connection:
            catalog = cast(psycopg.Connection[dict[str, Any]], connection)
            catalog.execute("SET LOCAL statement_timeout = '60s'")
            catalog.execute("SET LOCAL TIME ZONE 'UTC'")
            catalog.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_ID,))
            _apply_migrations(catalog)
            _verify_schema(catalog)
    except psycopg.Error as error:
        raise NewsCatalogSchemaError("PostgreSQL news schema check failed") from error


def _apply_migrations(connection: psycopg.Connection[dict[str, Any]]) -> None:
    ledger_exists = connection.execute(
        "SELECT to_regclass('news_schema_migrations') IS NOT NULL AS present"
    ).fetchone()
    if ledger_exists is None:
        raise NewsCatalogSchemaError("PostgreSQL did not report the migration ledger state")
    applied: dict[int, tuple[str, str]] = {}
    if bool(ledger_exists["present"]):
        rows = connection.execute(
            "SELECT version, name, sha256 FROM news_schema_migrations ORDER BY version"
        ).fetchall()
        applied = {int(row["version"]): (str(row["name"]), str(row["sha256"])) for row in rows}
    else:
        expected_tables, _ = _expected_schema_objects()
        existing = _current_schema_objects(connection, "tables")
        partial = sorted(expected_tables & existing)
        if partial:
            raise NewsCatalogSchemaError(
                "PostgreSQL news schema is unversioned; existing tables: " + ", ".join(partial)
            )

    expected_versions = {migration.version for migration in NEWS_CATALOG_MIGRATIONS}
    unexpected = sorted(set(applied) - expected_versions)
    if unexpected:
        raise NewsCatalogSchemaError(
            "PostgreSQL news schema has unknown migrations: "
            + ", ".join(str(version) for version in unexpected)
        )

    for migration in NEWS_CATALOG_MIGRATIONS:
        recorded = applied.get(migration.version)
        if recorded is not None:
            if recorded != (migration.name, migration.sha256):
                raise NewsCatalogSchemaError(
                    f"PostgreSQL news migration {migration.version} differs from the runtime"
                )
            continue
        migration_sql = migration.path.read_text()
        connection.execute(sql.SQL(cast(LiteralString, migration_sql)), prepare=False)
        connection.execute(
            "INSERT INTO news_schema_migrations (version, name, sha256) VALUES (%s, %s, %s)",
            (migration.version, migration.name, migration.sha256),
        )


def _verify_schema(connection: psycopg.Connection[dict[str, Any]]) -> None:
    expected_tables, expected_triggers = _expected_schema_objects()
    missing_tables = expected_tables - _current_schema_objects(connection, "tables")
    if missing_tables:
        raise NewsCatalogSchemaError(
            "PostgreSQL news schema is missing tables: " + ", ".join(sorted(missing_tables))
        )
    missing_triggers = expected_triggers - _current_schema_objects(connection, "triggers")
    if missing_triggers:
        raise NewsCatalogSchemaError(
            "PostgreSQL news schema is missing triggers: " + ", ".join(sorted(missing_triggers))
        )
    disabled = connection.execute(
        "SELECT catalog_trigger.tgname "
        "FROM pg_trigger catalog_trigger "
        "JOIN pg_class catalog_table ON catalog_table.oid = catalog_trigger.tgrelid "
        "JOIN pg_namespace catalog_namespace ON catalog_namespace.oid = catalog_table.relnamespace "
        "WHERE catalog_namespace.nspname = current_schema() "
        "AND NOT catalog_trigger.tgisinternal AND catalog_trigger.tgenabled != 'O'"
    ).fetchall()
    if disabled:
        names = sorted(str(row["tgname"]) for row in disabled)
        raise NewsCatalogSchemaError(
            "PostgreSQL news schema has disabled triggers: " + ", ".join(names)
        )


def _expected_schema_objects() -> tuple[set[str], set[str]]:
    tables: set[str] = set()
    triggers: set[str] = set()
    for migration in NEWS_CATALOG_MIGRATIONS:
        migration_sql = migration.path.read_text()
        tables.update(re.findall(r"^CREATE TABLE (\w+)", migration_sql, re.MULTILINE))
        triggers.update(re.findall(r"^CREATE TRIGGER (\w+)", migration_sql, re.MULTILINE))
    return tables, triggers


def _current_schema_objects(connection: psycopg.Connection[dict[str, Any]], kind: str) -> set[str]:
    if kind == "tables":
        rows = connection.execute(
            "SELECT table_name AS name FROM information_schema.tables "
            "WHERE table_schema = current_schema()"
        ).fetchall()
    elif kind == "triggers":
        rows = connection.execute(
            "SELECT trigger_name AS name FROM information_schema.triggers "
            "WHERE trigger_schema = current_schema()"
        ).fetchall()
    else:
        raise ValueError(f"Unknown PostgreSQL schema object kind: {kind}")
    return {str(row["name"]) for row in rows}
