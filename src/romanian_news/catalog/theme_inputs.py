from datetime import date

from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog_transport import catalog_query


def read_recorded_daily_theme_references(day: date) -> tuple[ArtifactReference, ...]:
    rows = catalog_query(
        """SELECT input.position, input.role, version.artifact_id, version.id AS version_id,
                  file.content_digest, file.r2_key FROM artifacts theme
           JOIN run_inputs input ON input.run_id = theme.current_run_id
           JOIN artifact_versions version ON version.id = input.artifact_version_id
           JOIN artifact_files file ON file.artifact_version_id = version.id
           WHERE theme.id = %s AND input.role != 'prior_output' ORDER BY input.position""",
        [f"news:themes:{day.isoformat()}"],
    )
    if not rows or rows[0]["role"] != "cluster_set":
        return ()
    if any(row["role"] != "summary" for row in rows[1:]):
        raise ValueError("Daily theme run recorded an unexpected input role")
    return tuple(
        ArtifactReference(
            artifact_id=str(row["artifact_id"]),
            version_id=str(row["version_id"]),
            content_digest=str(row["content_digest"]),
            r2_key=str(row["r2_key"]),
        )
        for row in rows
    )
