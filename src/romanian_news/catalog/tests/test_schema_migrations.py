import re

from romanian_news.catalog.schema import NEWS_CATALOG_MIGRATIONS, _expected_schema_objects


def test_news_schema_contains_its_catalog_boundaries() -> None:
    news_tables, news_triggers = _expected_schema_objects()

    assert {"artifacts", "artifact_versions", "runs", "news_schema_migrations"} <= news_tables
    assert {"artifacts_protect_identity", "runs_protect_identity"} <= news_triggers


def test_news_schema_contains_no_debt_objects() -> None:
    migration_sql = "\n".join(migration.path.read_text() for migration in NEWS_CATALOG_MIGRATIONS)

    assert "debt_" not in migration_sql
    assert "_chartly_catalog_schema" not in migration_sql


def test_news_migrations_are_ordered_and_immutable_by_identity() -> None:
    assert tuple(migration.version for migration in NEWS_CATALOG_MIGRATIONS) == tuple(
        range(1, len(NEWS_CATALOG_MIGRATIONS) + 1)
    )
    assert len({migration.name for migration in NEWS_CATALOG_MIGRATIONS}) == len(
        NEWS_CATALOG_MIGRATIONS
    )
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", migration.sha256) for migration in NEWS_CATALOG_MIGRATIONS
    )
