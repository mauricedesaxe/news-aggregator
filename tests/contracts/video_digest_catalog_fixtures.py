from __future__ import annotations

from typing import Any

import psycopg


def sha256_id(value: int) -> str:
    return f"{value:064x}"


def record_active_artifact_versions(
    connection: psycopg.Connection[Any], start: int, count: int
) -> tuple[str, ...]:
    version_ids = tuple(sha256_id(value) for value in range(start, start + count))
    for version_id in version_ids:
        artifact_id = f"artifact-{version_id}"
        connection.execute(
            "INSERT INTO artifacts "
            "(id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
            "VALUES (%s, 'test', 'Test artifact', 'test', 'active', 'private', "
            "CURRENT_TIMESTAMP)",
            (artifact_id,),
        )
        connection.execute(
            "INSERT INTO artifact_versions "
            "(id, artifact_id, schema_version, content_digest, created_at) "
            "VALUES (%s, %s, 1, %s, CURRENT_TIMESTAMP)",
            (version_id, artifact_id, version_id),
        )
    return version_ids
