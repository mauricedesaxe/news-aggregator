from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import Field

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import (
    CurrentArtifactFile,
    artifact_file,
    artifact_statements,
    canonical_json,
    current_artifact_file,
    run_output_statement,
    run_status,
)
from romanian_news.catalog_transport import (
    advance_artifact_current_version_from_run_statement,
    catalog_batch,
)
from romanian_news.reports import DailyReportOutput, WeeklyReportOutput, report_run_id
from romanian_news.storage import publish_immutable_r2_objects


class NewsDailyReportPublication(NewsModel):
    status: str
    run_id: Sha256
    version_id: Sha256
    uploaded_objects: Annotated[int, Field(ge=0)]
    reused_objects: Annotated[int, Field(ge=0)]


class NewsWeeklyReportPublication(NewsModel):
    status: str
    run_id: Sha256
    version_id: Sha256
    uploaded_objects: Annotated[int, Field(ge=0)]
    reused_objects: Annotated[int, Field(ge=0)]


def publish_daily_report(
    output: DailyReportOutput,
    implementation_ref: str,
) -> NewsDailyReportPublication:
    """Publish a changed daily report or retain an explicit no-change run."""
    if not implementation_ref:
        raise ValueError("Implementation reference is required")
    artifact_id = f"news:daily:{output.report.day.isoformat()}"
    current = current_artifact_file(artifact_id)
    run_id = report_run_id(output.request_id, implementation_ref)
    file = artifact_file(
        artifact_id=artifact_id,
        artifact_kind="news_daily_report",
        title=f"Romanian news report for {output.report.day.isoformat()}",
        content=output.content,
        r2_key=f"news/reports/daily/{output.report.day.isoformat()}/{output.content_digest}.json",
        media_type="application/json",
    )
    no_change = current is not None and current.content_digest == output.content_digest
    existing_status = run_status(run_id)
    if existing_status is not None:
        catalog_batch(
            [
                advance_artifact_current_version_from_run_statement(
                    file.artifact_id, file.version_id, run_id
                )
            ]
        )
        return NewsDailyReportPublication(
            status="no_change" if existing_status == "no_change" else "published",
            run_id=run_id,
            version_id=file.version_id,
            uploaded_objects=0,
            reused_objects=0,
        )
    publication = publish_immutable_r2_objects(((file.r2_key, file.content),))
    statements = _report_run_statements(
        run_id,
        implementation_ref,
        output,
        status="no_change" if no_change else "publishing",
        prior_output=current if no_change else None,
    )
    timestamp = datetime.now(UTC).isoformat()
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
    return NewsDailyReportPublication(
        status="no_change" if no_change else "published",
        run_id=run_id,
        version_id=file.version_id,
        uploaded_objects=publication.uploaded_objects,
        reused_objects=publication.reused_objects,
    )


def publish_weekly_report(
    output: WeeklyReportOutput,
    implementation_ref: str,
) -> NewsWeeklyReportPublication:
    """Publish a changed weekly rollup or retain an explicit no-change run."""
    if not implementation_ref:
        raise ValueError("Implementation reference is required")
    artifact_id = f"news:weekly:{output.report.week_start.isoformat()}"
    current = current_artifact_file(artifact_id)
    run_id = report_run_id(output.request_id, implementation_ref)
    file = artifact_file(
        artifact_id=artifact_id,
        artifact_kind="news_weekly_report",
        title=f"Romanian news report for week {output.report.week_start.isoformat()}",
        content=output.content,
        r2_key=(
            f"news/reports/weekly/{output.report.week_start.isoformat()}/"
            f"{output.content_digest}.json"
        ),
        media_type="application/json",
    )
    no_change = current is not None and current.content_digest == output.content_digest
    existing_status = run_status(run_id)
    if existing_status is not None:
        catalog_batch(
            [
                advance_artifact_current_version_from_run_statement(
                    file.artifact_id, file.version_id, run_id
                )
            ]
        )
        return NewsWeeklyReportPublication(
            status="no_change" if existing_status == "no_change" else "published",
            run_id=run_id,
            version_id=file.version_id,
            uploaded_objects=0,
            reused_objects=0,
        )
    publication = publish_immutable_r2_objects(((file.r2_key, file.content),))
    statements = _weekly_report_run_statements(
        run_id,
        implementation_ref,
        output,
        status="no_change" if no_change else "publishing",
        prior_output=current if no_change else None,
    )
    timestamp = datetime.now(UTC).isoformat()
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
    return NewsWeeklyReportPublication(
        status="no_change" if no_change else "published",
        run_id=run_id,
        version_id=file.version_id,
        uploaded_objects=publication.uploaded_objects,
        reused_objects=publication.reused_objects,
    )


def _report_run_statements(
    run_id: Sha256,
    implementation_ref: str,
    output: DailyReportOutput,
    *,
    status: str,
    prior_output: CurrentArtifactFile | None,
) -> list[tuple[str, list[object]]]:
    timestamp = datetime.now(UTC).isoformat()
    statements: list[tuple[str, list[object]]] = [
        (
            "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                "news.publish_daily",
                "python",
                implementation_ref,
                canonical_json({"day": output.report.day.isoformat()}).decode(),
                "chartly",
                status,
                f"news.publish_daily:{run_id}",
                None,
                timestamp,
                timestamp if status == "no_change" else None,
            ],
        )
    ]
    references: list[tuple[ArtifactReference | CurrentArtifactFile, str]] = [
        (output.themes, "themes"),
        (output.assessments, "assessments"),
        (output.cluster_set, "cluster_set"),
    ]
    for summary, sentiment in zip(output.summaries, output.sentiments, strict=True):
        references.extend(((summary, "summary"), (sentiment, "sentiment")))
    if prior_output:
        references.append((prior_output, "prior_output"))
    for position, (reference, role) in enumerate(references):
        statements.append(
            (
                "INSERT INTO run_inputs VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
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


def _weekly_report_run_statements(
    run_id: Sha256,
    implementation_ref: str,
    output: WeeklyReportOutput,
    *,
    status: str,
    prior_output: CurrentArtifactFile | None,
) -> list[tuple[str, list[object]]]:
    timestamp = datetime.now(UTC).isoformat()
    statements: list[tuple[str, list[object]]] = [
        (
            "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                "news.publish_weekly",
                "python",
                implementation_ref,
                canonical_json({"week_start": output.report.week_start.isoformat()}).decode(),
                "chartly",
                status,
                f"news.publish_weekly:{run_id}",
                None,
                timestamp,
                timestamp if status == "no_change" else None,
            ],
        )
    ]
    references: list[tuple[ArtifactReference | CurrentArtifactFile, str]] = [
        (reference, "daily_report") for reference in output.daily_reports
    ]
    if prior_output:
        references.append((prior_output, "prior_output"))
    for position, (reference, role) in enumerate(references):
        statements.append(
            (
                "INSERT INTO run_inputs VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
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
