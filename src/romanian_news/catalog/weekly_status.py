from __future__ import annotations

from datetime import UTC, date, datetime

from romanian_news import NewsModel, Sha256
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import (
    CurrentArtifactFile,
    artifact_file,
    artifact_statements,
    current_artifact_file,
    run_output_statement,
    run_status,
)
from romanian_news.catalog_transport import (
    advance_artifact_current_version_from_run_statement,
    catalog_batch,
    catalog_query,
)
from romanian_news.identity import canonical_json
from romanian_news.reports import report_run_id
from romanian_news.storage import (
    ResearchObjectIntegrityError,
    publish_immutable_r2_objects,
    read_verified_r2_object,
)
from romanian_news.weekly_status import WeeklyStatusOutput, WeeklyStatusRead


class WeeklyStatusPublication(NewsModel):
    status: str
    run_id: Sha256
    version_id: Sha256


class WeeklyStatusSummary(NewsModel):
    week_start: date
    version_id: Sha256


class WeeklyStatusNotFound(ValueError):
    pass


def publish_weekly_status(
    output: WeeklyStatusOutput, implementation_ref: str
) -> WeeklyStatusPublication:
    if not implementation_ref:
        raise ValueError("Implementation reference is required")
    artifact_id = f"news:weekly_status:{output.read.week_start.isoformat()}"
    run_id = report_run_id(output.request_id, implementation_ref)
    file = artifact_file(
        artifact_id=artifact_id,
        artifact_kind="news_weekly_read",
        title=f"Romania status, week of {output.read.week_start.isoformat()}",
        content=output.content,
        r2_key=(
            f"news/reports/weekly-status/{output.read.week_start.isoformat()}/"
            f"{output.content_digest}.json"
        ),
        media_type="application/json",
    )
    current = current_artifact_file(artifact_id)
    prior_status = run_status(run_id)
    if prior_status is not None:
        rows = catalog_query(
            "SELECT artifact_version_id FROM run_outputs WHERE run_id = %s "
            "AND position = 0 AND role = 'output'",
            [run_id],
        )
        if len(rows) != 1:
            raise ValueError("An existing weekly status run has no saved output")
        saved_version_id = str(rows[0]["artifact_version_id"])
        catalog_batch(
            [
                advance_artifact_current_version_from_run_statement(
                    file.artifact_id, saved_version_id, run_id
                )
            ]
        )
        return WeeklyStatusPublication(
            status="no_change" if prior_status == "no_change" else "published",
            run_id=run_id,
            version_id=saved_version_id,
        )
    publish_immutable_r2_objects(((file.r2_key, file.content),))
    no_change = current is not None and current.content_digest == file.content_digest
    timestamp = datetime.now(UTC).isoformat()
    statements = _run_statements(
        run_id,
        implementation_ref,
        output,
        current if no_change else None,
        timestamp,
        "no_change" if no_change else "publishing",
    )
    statements.extend(artifact_statements(file, timestamp, produced_by_run_id=run_id))
    statements.append(run_output_statement(run_id, file))
    statements.append(
        advance_artifact_current_version_from_run_statement(
            file.artifact_id, file.version_id, run_id
        )
    )
    statements.append(
        (
            "UPDATE runs SET status = 'completed', completed_at = %s "
            "WHERE id = %s AND status = 'publishing'",
            [timestamp, run_id],
        )
    )
    catalog_batch(statements)
    return WeeklyStatusPublication(
        status="no_change" if no_change else "published",
        run_id=run_id,
        version_id=file.version_id,
    )


def _run_statements(
    run_id: Sha256,
    implementation_ref: str,
    output: WeeklyStatusOutput,
    prior_output: CurrentArtifactFile | None,
    timestamp: str,
    status: str,
) -> list[tuple[str, list[object]]]:
    statements: list[tuple[str, list[object]]] = [
        (
            "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING",
            [
                run_id,
                "news.publish_weekly_status",
                "python",
                implementation_ref,
                canonical_json(
                    {
                        "week_start": output.read.week_start.isoformat(),
                        "policy_request_id": output.request_id,
                    }
                ).decode(),
                "chartly",
                status,
                f"news.publish_weekly_status:{run_id}",
                None,
                timestamp,
                timestamp if status == "no_change" else None,
            ],
        )
    ]
    references: list[tuple[ArtifactReference | CurrentArtifactFile, str]] = [
        (slot.report, "daily_report") for slot in output.read.days if slot.report is not None
    ]
    if prior_output is not None:
        references.append((prior_output, "prior_output"))
    for position, (reference, role) in enumerate(references):
        statements.append(
            (
                "INSERT INTO run_inputs VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                [
                    run_id,
                    position,
                    reference.version_id,
                    role,
                    None,
                    reference.content_digest,
                    "whole_file",
                    None,
                ],
            )
        )
    return statements


def list_weekly_status(limit: int = 20, offset: int = 0) -> tuple[WeeklyStatusSummary, ...]:
    rows = catalog_query(
        """
        SELECT artifact.id, artifact.current_version_id AS version_id
        FROM artifacts artifact
        WHERE artifact.kind = 'news_weekly_read'
          AND artifact.current_version_id IS NOT NULL
        ORDER BY artifact.id DESC
        LIMIT %s OFFSET %s
        """,
        [limit, offset],
    )
    return tuple(
        WeeklyStatusSummary(
            week_start=date.fromisoformat(str(row["id"]).removeprefix("news:weekly_status:")),
            version_id=str(row["version_id"]),
        )
        for row in rows
    )


def read_weekly_status(week_start: date) -> tuple[Sha256, WeeklyStatusRead]:
    rows = catalog_query(
        """
        SELECT artifact.current_version_id AS version_id,
               version.content_digest AS version_digest,
               file.content_digest AS file_digest, file.r2_key
        FROM artifacts artifact
        JOIN artifact_versions version ON version.id = artifact.current_version_id
        JOIN artifact_files file ON file.artifact_version_id = version.id
        WHERE artifact.id = %s AND artifact.kind = 'news_weekly_read'
        """,
        [f"news:weekly_status:{week_start.isoformat()}"],
    )
    if len(rows) != 1:
        raise WeeklyStatusNotFound(f"No weekly status for {week_start.isoformat()}")
    return _read_row(rows[0])


def read_weekly_status_version(version_id: Sha256) -> tuple[Sha256, WeeklyStatusRead]:
    rows = catalog_query(
        """
        SELECT version.id AS version_id, version.content_digest AS version_digest,
               file.content_digest AS file_digest, file.r2_key
        FROM artifact_versions version
        JOIN artifacts artifact ON artifact.id = version.artifact_id
        JOIN artifact_files file ON file.artifact_version_id = version.id
        WHERE version.id = %s AND artifact.kind = 'news_weekly_read'
        """,
        [version_id],
    )
    if len(rows) != 1:
        raise WeeklyStatusNotFound(f"No weekly status version: {version_id}")
    return _read_row(rows[0])


def _read_row(row: dict[str, object]) -> tuple[Sha256, WeeklyStatusRead]:
    if row["version_digest"] != row["file_digest"]:
        raise ResearchObjectIntegrityError("Weekly status catalog digests disagree")
    content = read_verified_r2_object(str(row["r2_key"]), str(row["file_digest"]))
    read = WeeklyStatusRead.model_validate_json(content)
    return str(row["version_id"]), read
