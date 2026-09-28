from __future__ import annotations

import psycopg
import pytest

from romanian_news.catalog.schema import NEWS_CATALOG_MIGRATIONS
from romanian_news.catalog_transport import ResearchCatalogError, catalog_query
from romanian_news.worker import catalog_schema
from tests.postgres_catalog import isolated_postgres_schema


def _ledger_versions() -> set[int]:
    return {
        int(row["version"]) for row in catalog_query("SELECT version FROM news_schema_migrations")
    }


def test_catalog_schema_activation_installs_every_migration_on_an_empty_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with isolated_postgres_schema(monkeypatch, "news_catalog_schema_job"):
        with pytest.raises(ResearchCatalogError) as failure:
            catalog_query("SELECT version FROM news_schema_migrations")
        assert isinstance(failure.value.__cause__, psycopg.errors.UndefinedTable)

        result = catalog_schema.catalog_schema_activation.execute_in_process()

        assert result.success
        assert _ledger_versions() == {migration.version for migration in NEWS_CATALOG_MIGRATIONS}


def test_catalog_schema_activation_succeeds_on_an_already_migrated_catalog(
    postgres_catalog,
) -> None:
    result = catalog_schema.catalog_schema_activation.execute_in_process()

    assert result.success
    assert _ledger_versions() == {migration.version for migration in NEWS_CATALOG_MIGRATIONS}
    assert (
        len(
            catalog_query(
                "SELECT version FROM news_schema_migrations WHERE version = %s",
                [NEWS_CATALOG_MIGRATIONS[-1].version],
            )
        )
        == 1
    )

    repeat = catalog_schema.catalog_schema_activation.execute_in_process()

    assert repeat.success
    assert _ledger_versions() == {migration.version for migration in NEWS_CATALOG_MIGRATIONS}
