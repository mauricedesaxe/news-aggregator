"""Apply catalog migrations as the schema owner, then verify runtime ledger access."""

import os
import sys

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from romanian_news.catalog.schema import NEWS_CATALOG_MIGRATIONS, ensure_news_catalog_schema


def _identity(dsn: str) -> dict[str, str]:
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        row = connection.execute(
            "SELECT current_database() AS database, current_schema() AS schema, "
            "current_user AS role"
        ).fetchone()
        if row is None or row["schema"] is None:
            raise RuntimeError("Catalog connection has no active schema")
        return {key: str(value) for key, value in row.items()}


def main() -> None:
    owner_dsn = os.environ.get("NEWS_POSTGRES_DSN")
    runtime_dsn = os.environ.get("NEWS_POSTGRES_RUNTIME_DSN")
    if not owner_dsn or not runtime_dsn:
        raise RuntimeError("Schema owner and runtime PostgreSQL DSNs are required")
    owner = _identity(owner_dsn)
    runtime = _identity(runtime_dsn)
    if (owner["database"], owner["schema"]) != (runtime["database"], runtime["schema"]):
        raise RuntimeError("Schema owner and runtime DSNs target different catalog locations")

    ensure_news_catalog_schema()
    latest = NEWS_CATALOG_MIGRATIONS[-1].version
    with psycopg.connect(owner_dsn, row_factory=dict_row) as connection:
        row = connection.execute(
            "SELECT 1 FROM news_schema_migrations WHERE version = %s", (latest,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Catalog migration {latest} was not installed")
        connection.execute(
            sql.SQL("GRANT SELECT, INSERT, UPDATE ON TABLE {}.{} TO {}").format(
                sql.Identifier(owner["schema"]),
                sql.Identifier("news_archive_model_reservations"),
                sql.Identifier(runtime["role"]),
            )
        )

    with psycopg.connect(runtime_dsn, row_factory=dict_row) as connection:
        permissions = connection.execute(
            "SELECT has_table_privilege(current_user, "
            "'news_archive_model_reservations', 'SELECT') AS can_read, "
            "has_table_privilege(current_user, "
            "'news_archive_model_reservations', 'INSERT') AS can_insert, "
            "has_table_privilege(current_user, "
            "'news_archive_model_reservations', 'UPDATE') AS can_update"
        ).fetchone()
        if permissions is None or not all(permissions.values()):
            raise RuntimeError("Runtime role lacks spend ledger read/write access")
        connection.execute("SELECT 1 FROM news_archive_model_reservations LIMIT 1")
    sys.stdout.write(
        f"Catalog migration {latest} installed; runtime spend ledger access verified\n"
    )


if __name__ == "__main__":
    main()
