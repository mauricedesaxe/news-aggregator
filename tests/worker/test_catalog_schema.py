import pytest

from romanian_news.catalog.schema import NewsCatalogSchemaError
from romanian_news.worker import catalog_schema


def test_schema_activation_fails_when_runtime_cannot_apply_latest_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(catalog_schema, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(catalog_schema, "catalog_query", lambda _query, _params: [])

    with pytest.raises(NewsCatalogSchemaError, match="Latest catalog migration"):
        catalog_schema.catalog_schema_activation.execute_in_process()
