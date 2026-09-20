from datetime import date

from romanian_news import Sha256
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog_transport import catalog_query


def read_daily_feed_observation_references(day: date) -> tuple[ArtifactReference, ...]:
    return _references(
        """
        SELECT artifact.id AS artifact_id, version.id AS version_id,
               file.content_digest, file.r2_key
        FROM news_feed_observations observation
        JOIN artifact_versions version ON version.id = observation.artifact_version_id
        JOIN artifacts artifact ON artifact.current_version_id = version.id
        JOIN artifact_files file ON file.artifact_version_id = version.id
        WHERE (observation.scheduled_slot AT TIME ZONE 'Europe/Bucharest')::date = %s
        ORDER BY artifact.id COLLATE "C", version.id COLLATE "C"
        """,
        [day.isoformat()],
    )


def read_daily_article_references(day: date) -> tuple[ArtifactReference, ...]:
    return _references(
        """
        SELECT artifact.id AS artifact_id, version.id AS version_id,
               file.content_digest, file.r2_key
        FROM news_article_versions metadata
        JOIN artifact_versions version ON version.id = metadata.artifact_version_id
        JOIN artifacts artifact ON artifact.current_version_id = version.id
        JOIN artifact_files file ON file.artifact_version_id = version.id
        WHERE metadata.bucharest_day = %s
        ORDER BY artifact.id COLLATE "C", version.id COLLATE "C"
        """,
        [day.isoformat()],
    )


def read_current_artifact_input_version_ids(artifact_id: str) -> tuple[Sha256, ...]:
    rows = catalog_query(
        """
        SELECT input.artifact_version_id
        FROM artifacts artifact
        JOIN run_inputs input ON input.run_id = artifact.current_run_id
        WHERE artifact.id = %s AND input.role != 'prior_output'
        ORDER BY input.position
        """,
        [artifact_id],
    )
    return tuple(str(row["artifact_version_id"]) for row in rows)


def relevance_is_accepted(version_id: Sha256) -> bool:
    rows = catalog_query(
        "SELECT accepted FROM news_relevance_versions WHERE artifact_version_id = %s",
        [version_id],
    )
    return len(rows) == 1 and bool(rows[0]["accepted"])


def _references(sql: str, parameters: list[object]) -> tuple[ArtifactReference, ...]:
    return tuple(
        ArtifactReference.model_validate(row, strict=True) for row in catalog_query(sql, parameters)
    )
