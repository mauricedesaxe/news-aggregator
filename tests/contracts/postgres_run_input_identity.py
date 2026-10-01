from __future__ import annotations

import psycopg
import pytest
from psycopg.types.json import Jsonb

from tests.postgres_catalog import PostgresCatalog, postgres_catalog_fixture

postgres_catalog = postgres_catalog_fixture("run_input_identity")


def _seed_input_references(catalog: PostgresCatalog) -> None:
    catalog.execute(
        "INSERT INTO runs (id, operation_key, executor_kind, implementation_ref, "
        "parameters_json, actor, status, idempotency_key, started_at) "
        "VALUES ('run-1', 'test.operation', 'test', 'test', '{}'::jsonb, "
        "'test', 'running', 'run-1', now())"
    )
    catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
        "visibility, created_at) VALUES ('artifact-1', 'test', 'Test artifact', "
        "'test', 'active', 'private', now())"
    )
    catalog.execute(
        "INSERT INTO artifact_versions "
        "(id, artifact_id, schema_version, content_digest, created_at) "
        "VALUES ('version-1', 'artifact-1', 1, 'digest-1', now()), "
        "('version-2', 'artifact-1', 1, 'digest-2', now())"
    )


def _insert_input(
    catalog: PostgresCatalog, values: dict[str, object], *, ignore_conflict: bool = False
) -> list[dict[str, object]]:
    statement = (
        "INSERT INTO run_inputs "
        "(run_id, position, artifact_version_id, role, locator_json, "
        "selected_content_digest, selection_method, retrieval_metadata_json) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
    )
    if ignore_conflict:
        statement += "ON CONFLICT DO NOTHING "
    statement += "RETURNING run_id, position"
    return catalog.execute(
        statement,
        (
            values["run_id"],
            values["position"],
            values["artifact_version_id"],
            values["role"],
            values["locator_json"],
            values["selected_content_digest"],
            values["selection_method"],
            values["retrieval_metadata_json"],
        ),
    ).fetchall()


def _input(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "run_id": "run-1",
        "position": 0,
        "artifact_version_id": "version-1",
        "role": "article",
        "locator_json": None,
        "selected_content_digest": "digest-1",
        "selection_method": "direct",
        "retrieval_metadata_json": None,
    }
    values.update(changes)
    return values


def test_run_input_insert_and_identical_retry_keep_one_row(
    postgres_catalog: PostgresCatalog,
) -> None:
    _seed_input_references(postgres_catalog)
    original = _input()

    assert _insert_input(postgres_catalog, original) == [{"run_id": "run-1", "position": 0}]
    assert _insert_input(postgres_catalog, original, ignore_conflict=True) == []
    assert _insert_input(postgres_catalog, _input(position=1)) == [
        {"run_id": "run-1", "position": 1}
    ]
    assert postgres_catalog.execute("SELECT count(*) AS total FROM run_inputs").fetchone() == {
        "total": 2
    }


def test_run_input_conflicting_retry_fails_even_with_on_conflict_do_nothing(
    postgres_catalog: PostgresCatalog,
) -> None:
    _seed_input_references(postgres_catalog)
    _insert_input(postgres_catalog, _input())

    for changes in (
        {"artifact_version_id": "version-2"},
        {"role": "feed"},
        {"selected_content_digest": "digest-2"},
        {"selection_method": "lookup"},
    ):
        for ignore_conflict in (False, True):
            with pytest.raises(
                psycopg.errors.IntegrityConstraintViolation, match="run_inputs identity conflict"
            ):
                _insert_input(postgres_catalog, _input(**changes), ignore_conflict=ignore_conflict)


def test_run_input_retry_compares_nullable_json_fields(postgres_catalog: PostgresCatalog) -> None:
    _seed_input_references(postgres_catalog)
    _insert_input(postgres_catalog, _input())

    for field in ("locator_json", "retrieval_metadata_json"):
        with pytest.raises(
            psycopg.errors.IntegrityConstraintViolation, match="run_inputs identity conflict"
        ):
            _insert_input(
                postgres_catalog,
                _input(**{field: Jsonb({"source": "test"})}),
                ignore_conflict=True,
            )

    populated = _input(
        position=1,
        locator_json=Jsonb({"url": "https://example.test", "kind": "article"}),
        retrieval_metadata_json=Jsonb({"status": 200, "cached": False}),
    )
    _insert_input(postgres_catalog, populated)
    assert (
        _insert_input(
            postgres_catalog,
            _input(
                position=1,
                locator_json=Jsonb({"kind": "article", "url": "https://example.test"}),
                retrieval_metadata_json=Jsonb({"cached": False, "status": 200}),
            ),
            ignore_conflict=True,
        )
        == []
    )

    for field in ("locator_json", "retrieval_metadata_json"):
        with pytest.raises(
            psycopg.errors.IntegrityConstraintViolation, match="run_inputs identity conflict"
        ):
            _insert_input(postgres_catalog, populated | {field: None}, ignore_conflict=True)
