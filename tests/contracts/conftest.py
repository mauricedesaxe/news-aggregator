from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.postgres_catalog import isolated_postgres_schema


@pytest.fixture
def postgres_news_schema(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    with isolated_postgres_schema(monkeypatch, "news_schema_contract") as (schema, _):
        yield schema
