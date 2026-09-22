from __future__ import annotations

from datetime import datetime

from romanian_news import NewsModel, Sha256
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog_transport import catalog_query
from romanian_news.identity import canonical_json as canonical_json
from romanian_news.identity import sha256 as sha256


class ArtifactFile(NewsModel):
    artifact_id: str
    artifact_kind: str
    title: str
    version_id: Sha256
    content_digest: Sha256
    r2_key: str
    media_type: str
    content: bytes


class CurrentArtifactFile(NewsModel):
    version_id: Sha256
    content_digest: Sha256
    r2_key: str


def artifact_file(
    *,
    artifact_id: str,
    artifact_kind: str,
    title: str,
    content: bytes,
    r2_key: str,
    media_type: str,
) -> ArtifactFile:
    content_digest = sha256(content)
    version_identity = f"{artifact_id}\0{content_digest}"
    return ArtifactFile(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        title=title,
        version_id=sha256(version_identity.encode()),
        content_digest=content_digest,
        r2_key=r2_key,
        media_type=media_type,
        content=content,
    )


def current_artifact_reference(artifact_id: str, kind: str) -> ArtifactReference | None:
    rows = catalog_query(
        """
        SELECT artifact.id AS artifact_id, artifact.current_version_id AS version_id,
               file.content_digest, file.r2_key
        FROM artifacts artifact
        JOIN artifact_files file ON file.artifact_version_id = artifact.current_version_id
        WHERE artifact.id = %s AND artifact.kind = %s
        """,
        [artifact_id, kind],
    )
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError(f"Artifact has more than one current file: {artifact_id}")
    return ArtifactReference.model_validate(rows[0], strict=True)


def artifact_references_by_version_ids(
    version_ids: tuple[Sha256, ...],
) -> dict[Sha256, ArtifactReference]:
    references: dict[Sha256, ArtifactReference] = {}
    for offset in range(0, len(version_ids), 50):
        values = version_ids[offset : offset + 50]
        placeholders = ", ".join("%s" for _ in values)
        rows = catalog_query(
            f"""
            SELECT version.artifact_id, version.id AS version_id,
                   file.content_digest, file.r2_key
            FROM artifact_versions version
            JOIN artifact_files file ON file.artifact_version_id = version.id
            WHERE version.id IN ({placeholders})
            """,
            list(values),
        )
        references.update(
            {
                str(row["version_id"]): ArtifactReference.model_validate(row, strict=True)
                for row in rows
            }
        )
    return references


def current_artifact_references(
    artifact_ids: tuple[str, ...],
) -> tuple[ArtifactReference, ...]:
    found: dict[str, ArtifactReference] = {}
    for offset in range(0, len(artifact_ids), 50):
        values = artifact_ids[offset : offset + 50]
        placeholders = ", ".join("%s" for _ in values)
        rows = catalog_query(
            f"""
            SELECT artifact.id AS artifact_id, artifact.current_version_id AS version_id,
                   file.content_digest, file.r2_key
            FROM artifacts artifact
            JOIN artifact_files file ON file.artifact_version_id = artifact.current_version_id
            WHERE artifact.id IN ({placeholders})
            """,
            list(values),
        )
        references = (ArtifactReference.model_validate(row, strict=True) for row in rows)
        found.update((reference.artifact_id, reference) for reference in references)
    return tuple(found[artifact_id] for artifact_id in sorted(found))


def current_artifact_file(artifact_id: str) -> CurrentArtifactFile | None:
    rows = catalog_query(
        """
        SELECT version.id AS version_id, file.content_digest, file.r2_key
        FROM artifacts artifact
        JOIN artifact_versions version ON version.id = artifact.current_version_id
        JOIN artifact_files file ON file.artifact_version_id = version.id
        WHERE artifact.id = %s
        """,
        [artifact_id],
    )
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError(f"Artifact has more than one current file: {artifact_id}")
    return CurrentArtifactFile.model_validate(rows[0], strict=True)


def current_artifact_version_times(artifact_ids: tuple[str, ...]) -> tuple[datetime, ...]:
    """Return creation timestamps of each artifact's current version."""
    times: list[datetime] = []
    for offset in range(0, len(artifact_ids), 50):
        values = artifact_ids[offset : offset + 50]
        placeholders = ", ".join("%s" for _ in values)
        rows = catalog_query(
            f"""
            SELECT artifact.id, version.created_at
            FROM artifacts artifact
            JOIN artifact_versions version ON version.id = artifact.current_version_id
            WHERE artifact.id IN ({placeholders})
            """,
            list(values),
        )
        times.extend(datetime.fromisoformat(str(row["created_at"])) for row in rows)
    return tuple(times)


def existing_current_artifact_ids(artifact_ids: tuple[str, ...]) -> set[str]:
    """Return artifact IDs that have a current version."""
    found: set[str] = set()
    for offset in range(0, len(artifact_ids), 50):
        values = artifact_ids[offset : offset + 50]
        placeholders = ", ".join("%s" for _ in values)
        rows = catalog_query(
            f"SELECT id FROM artifacts WHERE id IN ({placeholders}) "
            "AND current_version_id IS NOT NULL",
            list(values),
        )
        found.update(str(row["id"]) for row in rows)
    return found


def run_exists(run_id: Sha256) -> bool:
    return run_status(run_id) is not None


def existing_run_ids(run_ids: tuple[Sha256, ...]) -> frozenset[Sha256]:
    if not run_ids:
        return frozenset()
    placeholders = ", ".join("%s" for _ in run_ids)
    rows = catalog_query(f"SELECT id FROM runs WHERE id IN ({placeholders})", list(run_ids))
    return frozenset(str(row["id"]) for row in rows)


def run_status(run_id: Sha256) -> str | None:
    rows = catalog_query("SELECT status FROM runs WHERE id = %s", [run_id])
    return str(rows[0]["status"]) if rows else None


def artifact_statements(
    value: ArtifactFile,
    created_at: str,
    produced_by_run_id: Sha256 | None,
    *,
    authority_class: str | None = None,
) -> list[tuple[str, list[object]]]:
    file_id = sha256(f"{value.version_id}\0{value.r2_key}".encode())
    authority = authority_class or (
        "source"
        if value.artifact_kind in ("news_registry", "news_feed", "news_page", "news_article")
        else "derived"
    )
    return [
        (
            "INSERT INTO artifacts "
            "(id, kind, title, authority_class, lifecycle_state, visibility, "
            "current_version_id, created_at, current_run_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL) ON CONFLICT DO NOTHING",
            [
                value.artifact_id,
                value.artifact_kind,
                value.title,
                authority,
                "current",
                "private",
                None,
                created_at,
            ],
        ),
        (
            "INSERT INTO artifact_versions VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                value.version_id,
                value.artifact_id,
                1,
                value.content_digest,
                produced_by_run_id,
                created_at,
            ],
        ),
        (
            "INSERT INTO artifact_files VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                file_id,
                value.version_id,
                value.r2_key,
                value.media_type,
                value.content_digest,
                len(value.content),
                None,
                None,
            ],
        ),
    ]


def run_output_statement(
    run_id: Sha256,
    value: ArtifactFile,
) -> tuple[str, list[object]]:
    return (
        "INSERT INTO run_outputs VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
        [run_id, 0, value.version_id, "output"],
    )


def version_digests(version_ids: tuple[Sha256, ...]) -> dict[Sha256, Sha256]:
    if not version_ids:
        return {}
    placeholders = ", ".join("%s" for _ in version_ids)
    rows = catalog_query(
        f"SELECT id, content_digest FROM artifact_versions WHERE id IN ({placeholders})",
        list(version_ids),
    )
    result = {str(row["id"]): str(row["content_digest"]) for row in rows}
    if set(result) != set(version_ids):
        raise ValueError("Article inputs reference unknown feed snapshots")
    return result
