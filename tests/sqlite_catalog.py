from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

_SCHEMA_PATH = Path(__file__).parent / "fixtures" / "sqlite_catalog.sql"


class _CatalogQuery(Protocol):
    def __call__(
        self,
        statement: str,
        parameters: Sequence[object] | None = None,
    ) -> list[dict[str, Any]]: ...


def open_sqlite_catalog() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(_SCHEMA_PATH.read_text())
    return connection


def sqlite_catalog_query(connection: sqlite3.Connection) -> _CatalogQuery:
    def query(
        statement: str,
        parameters: Sequence[object] | None = None,
    ) -> list[dict[str, Any]]:
        rows = connection.execute(statement.replace("%s", "?"), parameters or ()).fetchall()
        return [dict(row) for row in rows]

    return query
