"""Apply catalog migrations in the Dagster runtime."""

import dagster as dg

from romanian_news.catalog.schema import (
    NEWS_CATALOG_MIGRATIONS,
    NewsCatalogSchemaError,
    ensure_news_catalog_schema,
)
from romanian_news.catalog_transport import catalog_query


@dg.op
def apply_catalog_schema() -> None:
    ensure_news_catalog_schema()
    rows = catalog_query(
        "SELECT version FROM news_schema_migrations WHERE version = %s",
        [NEWS_CATALOG_MIGRATIONS[-1].version],
    )
    if len(rows) != 1:
        raise NewsCatalogSchemaError("Latest catalog migration is not installed")


@dg.job
def catalog_schema_activation() -> None:
    apply_catalog_schema()
