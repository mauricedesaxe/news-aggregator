from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated, Literal

from pydantic import Field

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.attempts import response_cost
from romanian_news.catalog.artifacts import (
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
    catalog_query,
)
from romanian_news.storage import publish_immutable_r2_objects
from romanian_news.subject_assessments import (
    DailySubjectAssessmentOutput,
    EmptySubjectAssessmentConstruction,
    ModelSubjectAssessmentConstruction,
    subject_assessment_run_id,
)


class DailySubjectAssessmentUnavailable(ValueError):
    """The current daily subject assessment artifact does not exist."""


class SubjectAssessmentPublication(NewsModel):
    status: Literal["published", "no_change"]
    run_id: Sha256
    reference: ArtifactReference
    uploaded_objects: Annotated[int, Field(ge=0)]
    reused_objects: Annotated[int, Field(ge=0)]


def read_completed_subject_assessment(run_id: Sha256) -> ArtifactReference | None:
    """Read the exact output of a completed assessment run."""
    rows = catalog_query(
        """SELECT version.artifact_id, version.id AS version_id,
                  file.content_digest, file.r2_key
           FROM runs run
           JOIN run_outputs output ON output.run_id = run.id AND output.role = 'output'
           JOIN artifact_versions version ON version.id = output.artifact_version_id
           JOIN artifact_files file ON file.artifact_version_id = version.id
           WHERE run.id = %s AND run.status IN ('completed', 'no_change')""",
        [run_id],
    )
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("Completed subject assessment run must have one output")
    return ArtifactReference.model_validate(rows[0], strict=True)


def read_daily_subject_assessment_reference(day: date) -> ArtifactReference:
    """Read the current immutable daily subject assessment reference."""
    rows = catalog_query(
        """SELECT artifact.id AS artifact_id, artifact.current_version_id AS version_id,
                  file.content_digest, file.r2_key
           FROM artifacts artifact
           JOIN artifact_files file ON file.artifact_version_id = artifact.current_version_id
           WHERE artifact.id = %s AND artifact.kind = 'news_daily_subject_assessments'""",
        [f"news:subject-assessments:{day.isoformat()}"],
    )
    if len(rows) != 1:
        raise DailySubjectAssessmentUnavailable(
            f"Daily subject assessments are unavailable for {day.isoformat()}"
        )
    return ArtifactReference.model_validate(rows[0], strict=True)


def publish_daily_subject_assessments(
    output: DailySubjectAssessmentOutput,
    implementation_ref: str,
) -> SubjectAssessmentPublication:
    """Publish one assessment set with exact generic-artifact run lineage."""
    assessment_set = output.assessment_set
    run_id = subject_assessment_run_id(assessment_set.request_id, implementation_ref)
    artifact_id = f"news:subject-assessments:{assessment_set.day.isoformat()}"
    file = artifact_file(
        artifact_id=artifact_id,
        artifact_kind="news_daily_subject_assessments",
        title=f"Romanian news subject assessments for {assessment_set.day.isoformat()}",
        content=output.content,
        r2_key=(
            f"news/derived/subject-assessments/{assessment_set.day.isoformat()}/"
            f"{output.content_digest}.json"
        ),
        media_type="application/json",
    )
    if file.content_digest != output.content_digest:
        raise ValueError("Subject assessment output digest does not match its content")
    reference = ArtifactReference(
        artifact_id=file.artifact_id,
        version_id=file.version_id,
        content_digest=file.content_digest,
        r2_key=file.r2_key,
    )
    existing = read_completed_subject_assessment(run_id)
    if existing is not None:
        if existing != reference:
            raise ValueError("Completed subject assessment run has another output")
        catalog_batch(
            [
                advance_artifact_current_version_from_run_statement(
                    artifact_id, file.version_id, run_id
                )
            ]
        )
        return SubjectAssessmentPublication(
            status="published",
            run_id=run_id,
            reference=reference,
            uploaded_objects=0,
            reused_objects=0,
        )
    status = run_status(run_id)
    if status is not None:
        raise RuntimeError(f"Subject assessment run is incomplete: {status}")
    current = current_artifact_file(artifact_id)
    no_change = current is not None and current.content_digest == file.content_digest
    publication = publish_immutable_r2_objects(((file.r2_key, file.content),))
    timestamp = datetime.now(UTC).isoformat()
    construction = assessment_set.construction
    provider = (
        "python" if isinstance(construction, EmptySubjectAssessmentConstruction) else "openrouter"
    )
    statements: list[tuple[str, list[object]]] = [
        (
            "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                "news.assess_daily_subjects",
                provider,
                implementation_ref,
                canonical_json(
                    {
                        "day": assessment_set.day.isoformat(),
                        "policy": assessment_set.policy.model_dump(mode="json"),
                    }
                ).decode(),
                "chartly",
                "publishing",
                f"news.assess_daily_subjects:{run_id}",
                None,
                timestamp,
                None,
            ],
        )
    ]
    inputs = (
        (assessment_set.themes, "themes"),
        *((item, "summary") for item in assessment_set.summary_inputs),
        *((item, "relevance") for item in assessment_set.relevance_inputs),
    )
    statements.extend(
        (
            "INSERT INTO run_inputs VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                position,
                input_reference.version_id,
                role,
                None,
                input_reference.content_digest,
                "whole_file",
                None,
            ],
        )
        for position, (input_reference, role) in enumerate(inputs)
    )
    statements.extend(artifact_statements(file, timestamp, produced_by_run_id=run_id))
    statements.append(run_output_statement(run_id, file))
    if isinstance(construction, ModelSubjectAssessmentConstruction):
        statements.append(
            (
                "INSERT INTO news_model_calls VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    file.version_id,
                    "news.assess_daily_subjects",
                    construction.call.model,
                    construction.call.input_tokens,
                    construction.call.output_tokens,
                    sum(
                        response_cost(attempt.provider_response)
                        for attempt in construction.attempts
                    ),
                    construction.call.latency_ms,
                    len(construction.attempts),
                ],
            )
        )
    statements.append(
        advance_artifact_current_version_from_run_statement(artifact_id, file.version_id, run_id)
    )
    statements.append(
        (
            "UPDATE runs SET status = 'completed', completed_at = %s "
            "WHERE id = %s AND status = 'publishing'",
            [timestamp, run_id],
        )
    )
    catalog_batch(statements)
    return SubjectAssessmentPublication(
        status="no_change" if no_change else "published",
        run_id=run_id,
        reference=reference,
        uploaded_objects=publication.uploaded_objects,
        reused_objects=publication.reused_objects,
    )
