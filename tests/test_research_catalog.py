from __future__ import annotations

from typing import Any, cast

import psycopg
import pytest

from romanian_news import catalog_transport as research_catalog
from romanian_news.catalog_transport import CatalogConnection, ResearchCatalogError


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.exit_error: type[BaseException] | None = None

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: object,
    ) -> None:
        self.exit_error = error_type

    def execute(self, statement: str) -> None:
        self.statements.append(statement)


def test_catalog_operation_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes: list[Exception | str] = [
        psycopg.OperationalError("connection lost"),
        psycopg.OperationalError("connection lost"),
        "complete",
    ]
    delays = []
    monkeypatch.setattr(research_catalog.time, "sleep", delays.append)

    def operation() -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert (
        research_catalog._run_catalog_operation(operation, retry_transient_errors=True)
        == "complete"
    )
    assert len(delays) == 2


def test_catalog_operation_does_not_retry_database_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays = []
    monkeypatch.setattr(research_catalog.time, "sleep", delays.append)

    with pytest.raises(ResearchCatalogError) as failure:
        research_catalog._run_catalog_operation(
            lambda: (_ for _ in ()).throw(psycopg.IntegrityError("conflict")),
            retry_transient_errors=True,
        )

    assert isinstance(failure.value.__cause__, psycopg.IntegrityError)
    assert delays == []


def test_catalog_query_rejects_mutations() -> None:
    with pytest.raises(ValueError, match="catalog_query does not accept UPDATE statements"):
        research_catalog.catalog_query("UPDATE runs SET status = 'failed'")
    with pytest.raises(ValueError, match="catalog_query does not accept WITH statements"):
        research_catalog.catalog_query(
            "WITH removed AS (DELETE FROM runs RETURNING id) SELECT id FROM removed"
        )


def test_catalog_transaction_returns_callback_result(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection()
    monkeypatch.setattr(
        research_catalog,
        "_connect",
        lambda: cast(CatalogConnection, cast(Any, connection)),
    )

    result = research_catalog.catalog_transaction(lambda catalog: catalog is connection)

    assert result is True
    assert connection.statements == [
        "SET LOCAL statement_timeout = '60s'",
        "SET LOCAL TIME ZONE 'UTC'",
    ]
    assert connection.exit_error is None


def test_catalog_transaction_rolls_back_callback_error(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection()
    monkeypatch.setattr(
        research_catalog,
        "_connect",
        lambda: cast(CatalogConnection, cast(Any, connection)),
    )

    with pytest.raises(ValueError, match="invalid checkpoint"):
        research_catalog.catalog_transaction(
            lambda _: (_ for _ in ()).throw(ValueError("invalid checkpoint"))
        )

    assert connection.exit_error is ValueError
