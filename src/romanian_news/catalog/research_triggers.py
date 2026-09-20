from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated, Literal

from pydantic import Field

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.attempts import response_cost
from romanian_news.artifacts import ArtifactReference
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
from romanian_news.research_triggers import (
    DailyResearchTriggerOutput,
    DailyResearchTriggerSet,
    EmptyResearchTriggerConstruction,
    ModelResearchTriggerConstruction,
    parse_daily_research_trigger_set,
    research_trigger_run_id,
)
from romanian_news.storage import publish_immutable_r2_objects, read_verified_r2_object


class DailyResearchTriggerUnavailable(ValueError):
    """The current daily research trigger artifact does not exist."""


class ResearchTriggerPublication(NewsModel):
    status: Literal["published", "no_change"]
    run_id: Sha256
    reference: ArtifactReference
    uploaded_objects: Annotated[int, Field(ge=0)]
    reused_objects: Annotated[int, Field(ge=0)]


def read_completed_research_trigger(run_id: Sha256) -> ArtifactReference | None:
    """Read the exact output of a completed trigger run."""
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
        raise ValueError("Completed research trigger run must have one output")
    return ArtifactReference.model_validate(rows[0], strict=True)


def read_daily_research_trigger_reference(day: date) -> ArtifactReference:
    """Read the current immutable daily research trigger reference."""
    rows = catalog_query(
        """SELECT artifact.id AS artifact_id, artifact.current_version_id AS version_id,
                  file.content_digest, file.r2_key
           FROM artifacts artifact
           JOIN artifact_files file ON file.artifact_version_id = artifact.current_version_id
           WHERE artifact.id = %s AND artifact.kind = 'news_daily_research_triggers'""",
        [f"news:research-triggers:{day.isoformat()}"],
    )
    if len(rows) != 1:
        raise DailyResearchTriggerUnavailable(
            f"Daily research triggers are unavailable for {day.isoformat()}"
        )
    return ArtifactReference.model_validate(rows[0], strict=True)


def read_daily_research_trigger_reference_or_none(day: date) -> ArtifactReference | None:
    """Read the current daily research trigger reference when it exists."""
    try:
        return read_daily_research_trigger_reference(day)
    except DailyResearchTriggerUnavailable:
        return None


def read_daily_research_trigger_set(
    report_version_id: Sha256,
) -> DailyResearchTriggerSet | None:
    """Read the current trigger set pinned to one exact daily report version."""
    rows = catalog_query(
        """SELECT artifact.id AS artifact_id
           FROM artifact_versions version
           JOIN artifacts artifact ON artifact.id = version.artifact_id
           WHERE version.id = %s AND artifact.kind = 'news_daily_report'""",
        [report_version_id],
    )
    if len(rows) != 1:
        raise ValueError(f"Daily report is unavailable: {report_version_id}")
    artifact_id = str(rows[0]["artifact_id"])
    prefix = "news:daily:"
    if not artifact_id.startswith(prefix):
        raise ValueError(f"Invalid daily report artifact identity: {artifact_id}")
    day = date.fromisoformat(artifact_id.removeprefix(prefix))
    reference = read_daily_research_trigger_reference_or_none(day)
    if reference is None:
        return None
    trigger_set = parse_daily_research_trigger_set(
        read_verified_r2_object(reference.r2_key, reference.content_digest)
    )
    if trigger_set.report.version_id != report_version_id:
        return None
    return trigger_set


def publish_daily_research_triggers(
    output: DailyResearchTriggerOutput,
    implementation_ref: str,
) -> ResearchTriggerPublication:
    """Publish one trigger set with exact generic-artifact run lineage."""
    trigger_set = output.trigger_set
    run_id = research_trigger_run_id(trigger_set.request_id, implementation_ref)
    artifact_id = f"news:research-triggers:{trigger_set.day.isoformat()}"
    file = artifact_file(
        artifact_id=artifact_id,
        artifact_kind="news_daily_research_triggers",
        title=f"Romanian news research triggers for {trigger_set.day.isoformat()}",
        content=output.content,
        r2_key=(
            f"news/derived/research-triggers/{trigger_set.day.isoformat()}/"
            f"{output.content_digest}.json"
        ),
        media_type="application/json",
    )
    if file.content_digest != output.content_digest:
        raise ValueError("Research trigger output digest does not match its content")
    reference = ArtifactReference(
        artifact_id=file.artifact_id,
        version_id=file.version_id,
        content_digest=file.content_digest,
        r2_key=file.r2_key,
    )
    existing = read_completed_research_trigger(run_id)
    if existing is not None:
        if existing != reference:
            raise ValueError("Completed research trigger run has another output")
        catalog_batch(
            [
                advance_artifact_current_version_from_run_statement(
                    artifact_id, file.version_id, run_id
                )
            ]
        )
        return ResearchTriggerPublication(
            status="published",
            run_id=run_id,
            reference=reference,
            uploaded_objects=0,
            reused_objects=0,
        )
    status = run_status(run_id)
    if status is not None:
        raise RuntimeError(f"Research trigger run is incomplete: {status}")
    current = current_artifact_file(artifact_id)
    no_change = current is not None and current.content_digest == file.content_digest
    publication = publish_immutable_r2_objects(((file.r2_key, file.content),))
    timestamp = datetime.now(UTC).isoformat()
    construction = trigger_set.construction
    provider = (
        "python" if isinstance(construction, EmptyResearchTriggerConstruction) else "openrouter"
    )
    statements: list[tuple[str, list[object]]] = [
        (
            "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                "news.trigger_daily_research",
                provider,
                implementation_ref,
                canonical_json(
                    {
                        "day": trigger_set.day.isoformat(),
                        "policy": trigger_set.policy.model_dump(mode="json"),
                    }
                ).decode(),
                "chartly",
                "publishing",
                f"news.trigger_daily_research:{run_id}",
                None,
                timestamp,
                None,
            ],
        ),
        (
            "INSERT INTO run_inputs VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                0,
                trigger_set.report.version_id,
                "report",
                None,
                trigger_set.report.content_digest,
                "whole_file",
                None,
            ],
        ),
    ]
    statements.extend(artifact_statements(file, timestamp, produced_by_run_id=run_id))
    statements.append(run_output_statement(run_id, file))
    if isinstance(construction, ModelResearchTriggerConstruction):
        statements.append(
            (
                "INSERT INTO news_model_calls VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    file.version_id,
                    "news.trigger_daily_research",
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
    return ResearchTriggerPublication(
        status="no_change" if no_change else "published",
        run_id=run_id,
        reference=reference,
        uploaded_objects=publication.uploaded_objects,
        reused_objects=publication.reused_objects,
    )
