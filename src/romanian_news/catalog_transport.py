from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, LiteralString, TypeVar, cast

import psycopg
from psycopg.rows import dict_row

from romanian_news.config import NEWS_POSTGRES_DSN

Statement = tuple[str, list[object]]
Result = TypeVar("Result")
CatalogConnection = psycopg.Connection[dict[str, Any]]


class ResearchCatalogError(RuntimeError):
    """The authoritative research catalog could not complete a request."""


def catalog_integrity_identity(error: BaseException) -> str | None:
    """Identify a catalog integrity conflict, or return None for other errors."""
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, psycopg.errors.IntegrityError):
            constraint = getattr(getattr(current, "diag", None), "constraint_name", None)
            return f"{type(current).__name__}:{current.sqlstate}:{constraint or ''}"
        current = current.__cause__
    return None


def catalog_query(
    statement: str,
    parameters: Sequence[object] | None = None,
) -> list[dict[str, Any]]:
    """Run one retryable read against the authoritative research catalog."""
    _require_command(statement, ("SELECT",), "catalog_query")
    return _catalog_rows(statement, parameters, retry_transient_errors=True)


def catalog_mutation(
    statement: str,
    parameters: Sequence[object] | None = None,
) -> list[dict[str, Any]]:
    """Run one non-retrying mutation against the authoritative research catalog."""
    _require_command(statement, ("INSERT", "UPDATE", "DELETE"), "catalog_mutation")
    return _catalog_rows(statement, parameters, retry_transient_errors=False)


def _catalog_rows(
    statement: str,
    parameters: Sequence[object] | None,
    *,
    retry_transient_errors: bool,
) -> list[dict[str, Any]]:
    def run() -> list[dict[str, Any]]:
        with _connect() as connection:
            connection.execute("SET LOCAL statement_timeout = '30s'")
            connection.execute("SET LOCAL TIME ZONE 'UTC'")
            rows = connection.execute(cast(LiteralString, statement), parameters or ()).fetchall()
            return [_catalog_row(row) for row in rows]

    return _run_catalog_operation(run, retry_transient_errors=retry_transient_errors)


def catalog_batch(
    statements: Sequence[tuple[str, Sequence[object]]],
    *,
    retry_transient_errors: bool = False,
) -> None:
    """Run one transactional batch against the authoritative research catalog."""
    if not statements:
        return

    def run() -> None:
        with _connect() as connection:
            connection.execute("SET LOCAL statement_timeout = '60s'")
            connection.execute("SET LOCAL TIME ZONE 'UTC'")
            for statement, parameters in statements:
                connection.execute(cast(LiteralString, statement), parameters)

    _run_catalog_operation(run, retry_transient_errors=retry_transient_errors)


def catalog_transaction(
    operation: Callable[[CatalogConnection], Result],
    *,
    retry_transient_errors: bool = False,
) -> Result:
    """Run a callback in one authoritative catalog transaction."""

    def run() -> Result:
        with _connect() as connection:
            connection.execute("SET LOCAL statement_timeout = '60s'")
            connection.execute("SET LOCAL TIME ZONE 'UTC'")
            return operation(connection)

    return _run_catalog_operation(run, retry_transient_errors=retry_transient_errors)


def advance_artifact_current_version_statement(
    artifact_id: str,
    candidate_version_id: str,
) -> Statement:
    """Advance an artifact head by creation time and ID."""
    return (
        """
        UPDATE artifacts
        SET current_version_id = %s, current_run_id = NULL
        WHERE id = %s
          AND (
              current_version_id IS NULL
              OR (
                  SELECT (created_at, id) FROM artifact_versions WHERE id = %s
              ) > (
                  SELECT (created_at, id) FROM artifact_versions
                  WHERE id = artifacts.current_version_id
              )
          )
        """,
        [candidate_version_id, artifact_id, candidate_version_id],
    )


def advance_artifact_current_version_from_run_statement(
    artifact_id: str,
    candidate_version_id: str,
    run_id: str,
) -> Statement:
    """Advance a derived head only while every run input remains current."""
    return (
        """
        UPDATE artifacts
        SET current_version_id = %s, current_run_id = %s
        WHERE id = %s
          AND EXISTS (
              SELECT 1
              FROM run_outputs output
              WHERE output.run_id = %s
                AND output.artifact_version_id = %s
          )
          AND NOT EXISTS (
              SELECT 1
              FROM run_inputs input
              JOIN artifact_versions version ON version.id = input.artifact_version_id
              JOIN artifacts artifact ON artifact.id = version.artifact_id
              WHERE input.run_id = %s
                AND input.role != 'prior_output'
                AND artifact.current_version_id IS DISTINCT FROM input.artifact_version_id
          )
        """,
        [candidate_version_id, run_id, artifact_id, run_id, candidate_version_id, run_id],
    )


def _require_command(statement: str, allowed: tuple[str, ...], operation: str) -> None:
    command = statement.lstrip().split(None, 1)[0].upper() if statement.strip() else ""
    if command not in allowed:
        raise ValueError(f"{operation} does not accept {command or 'empty'} statements")


def _catalog_row(row: Mapping[str, object]) -> dict[str, Any]:
    return {
        name: value.isoformat() if isinstance(value, datetime) else value
        for name, value in row.items()
    }


def _connect() -> CatalogConnection:
    if NEWS_POSTGRES_DSN is None:
        raise RuntimeError("NEWS_POSTGRES_DSN is required")
    return cast(
        CatalogConnection,
        psycopg.connect(NEWS_POSTGRES_DSN, row_factory=cast(Any, dict_row)),
    )


def _run_catalog_operation(
    operation: Callable[[], Result],
    *,
    retry_transient_errors: bool,
) -> Result:
    for attempt in range(5):
        try:
            return operation()
        except psycopg.Error as error:
            if not retry_transient_errors or not _is_transient(error) or attempt == 4:
                raise ResearchCatalogError("PostgreSQL catalog request failed") from error
            time.sleep(min(float(2**attempt), 30.0))
    raise RuntimeError("PostgreSQL retry loop did not return")


def _is_transient(error: psycopg.Error) -> bool:
    return isinstance(
        error,
        psycopg.OperationalError
        | psycopg.errors.DeadlockDetected
        | psycopg.errors.SerializationFailure,
    )
