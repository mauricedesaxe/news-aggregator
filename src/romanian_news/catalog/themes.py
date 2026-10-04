from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated, Literal

from pydantic import Field

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.attempts import ModelCall, response_cost
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import (
    artifact_file,
    artifact_statements,
    current_artifact_file,
    run_output_statement,
    run_status,
)
from romanian_news.catalog.model_calls import ModelCallRegistration, model_call_statement
from romanian_news.catalog_transport import (
    advance_artifact_current_version_from_run_statement,
    catalog_batch,
    catalog_query,
)
from romanian_news.identity import canonical_json
from romanian_news.storage import publish_immutable_r2_objects
from romanian_news.themes import (
    DailyThemeOutput,
    ModelThemeConstruction,
    SparseThemeConstruction,
    ThemeConstructionEvidence,
    ThemeModelAttemptEvidence,
    daily_theme_run_id,
)


class DailyThemeUnavailable(ValueError):
    """The current daily theme artifact does not exist yet."""


class NewsDailyThemePublication(NewsModel):
    status: Literal["published", "no_change"]
    run_id: Sha256
    reference: ArtifactReference
    uploaded_objects: Annotated[int, Field(ge=0)]
    reused_objects: Annotated[int, Field(ge=0)]


def publish_daily_themes(
    output: DailyThemeOutput,
    implementation_ref: str,
) -> NewsDailyThemePublication:
    """Publish one complete daily theme set with exact run lineage."""
    if not implementation_ref:
        raise ValueError("Implementation reference is required")
    theme_set = output.theme_set
    construction = theme_set.construction
    if isinstance(construction, SparseThemeConstruction) and any(
        call.model != theme_set.policy.model for call in _model_calls(construction)
    ):
        raise ValueError("Sparse theme stage model does not match its policy")
    response_ids = _accepted_response_ids(construction)
    run_id = daily_theme_run_id(theme_set.request_id, response_ids)
    artifact_id = f"news:themes:{theme_set.day.isoformat()}"
    file = artifact_file(
        artifact_id=artifact_id,
        artifact_kind="news_daily_themes",
        title=f"Romanian news themes for {theme_set.day.isoformat()}",
        content=output.content,
        r2_key=(f"news/derived/themes/{theme_set.day.isoformat()}/{output.content_digest}.json"),
        media_type="application/json",
    )
    reference = ArtifactReference(
        artifact_id=file.artifact_id,
        version_id=file.version_id,
        content_digest=file.content_digest,
        r2_key=file.r2_key,
    )
    current = current_artifact_file(artifact_id)
    no_change = current is not None and current.content_digest == output.content_digest
    existing = run_status(run_id)
    if existing is not None:
        catalog_batch(
            [
                advance_artifact_current_version_from_run_statement(
                    artifact_id, file.version_id, run_id
                )
            ]
        )
        return NewsDailyThemePublication(
            status="no_change" if existing == "no_change" else "published",
            run_id=run_id,
            reference=reference,
            uploaded_objects=0,
            reused_objects=0,
        )

    publication = publish_immutable_r2_objects(((file.r2_key, file.content),))
    timestamp = datetime.now(UTC).isoformat()
    status = "no_change" if no_change else "publishing"
    statements: list[tuple[str, list[object]]] = [
        (
            "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                "news.construct_daily_themes",
                "openrouter"
                if isinstance(construction, ModelThemeConstruction | SparseThemeConstruction)
                else "python",
                implementation_ref,
                canonical_json(
                    {
                        "day": theme_set.day.isoformat(),
                        "policy": theme_set.policy.model_dump(mode="json"),
                    }
                ).decode(),
                "chartly",
                status,
                f"news.construct_daily_themes:{run_id}",
                None,
                timestamp,
                timestamp if no_change else None,
            ],
        )
    ]
    inputs = (
        (theme_set.cluster_set, "cluster_set"),
        *((summary, "summary") for summary in theme_set.summary_inputs),
    )
    for position, (input_reference, role) in enumerate(inputs):
        statements.append(
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
        )
    statements.extend(artifact_statements(file, timestamp, produced_by_run_id=run_id))
    statements.append(run_output_statement(run_id, file))
    calls = _model_calls(construction)
    if calls:
        attempts = _model_attempts(construction)
        statements.append(
            model_call_statement(
                ModelCallRegistration(
                    artifact_version_id=file.version_id,
                    operation_key="news.construct_daily_themes",
                    model=calls[0].model,
                    input_tokens=sum(call.input_tokens for call in calls),
                    output_tokens=sum(call.output_tokens for call in calls),
                    cost_usd=sum(response_cost(attempt.provider_response) for attempt in attempts),
                    latency_ms=sum(call.latency_ms for call in calls),
                    response_count=len(attempts),
                )
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
    return NewsDailyThemePublication(
        status="no_change" if no_change else "published",
        run_id=run_id,
        reference=reference,
        uploaded_objects=publication.uploaded_objects,
        reused_objects=publication.reused_objects,
    )


def read_daily_theme_reference(day: date) -> ArtifactReference:
    """Read the current immutable daily theme reference."""
    rows = catalog_query(
        """
        SELECT artifact.id AS artifact_id, artifact.current_version_id AS version_id,
               file.content_digest, file.r2_key
        FROM artifacts artifact
        JOIN artifact_files file ON file.artifact_version_id = artifact.current_version_id
        WHERE artifact.id = %s AND artifact.kind = 'news_daily_themes'
        """,
        [f"news:themes:{day.isoformat()}"],
    )
    if len(rows) != 1:
        raise DailyThemeUnavailable(f"Daily themes are unavailable for {day.isoformat()}")
    row = rows[0]
    return ArtifactReference(
        artifact_id=str(row["artifact_id"]),
        version_id=str(row["version_id"]),
        content_digest=str(row["content_digest"]),
        r2_key=str(row["r2_key"]),
    )


def _accepted_response_ids(
    construction: ThemeConstructionEvidence | SparseThemeConstruction,
) -> str | tuple[str, ...]:
    if isinstance(construction, ModelThemeConstruction):
        return construction.call.response_id
    if isinstance(construction, SparseThemeConstruction):
        calls = _model_calls(construction)
        return tuple(call.response_id for call in calls)
    return "empty"


def _model_calls(
    construction: ThemeConstructionEvidence | SparseThemeConstruction,
) -> tuple[ModelCall, ...]:
    if isinstance(construction, ModelThemeConstruction):
        return (construction.call,)
    if isinstance(construction, SparseThemeConstruction):
        if construction.merged_prose is None:
            return (construction.assignment.call,)
        return (construction.assignment.call, construction.merged_prose.call)
    return ()


def _model_attempts(
    construction: ThemeConstructionEvidence | SparseThemeConstruction,
) -> tuple[ThemeModelAttemptEvidence, ...]:
    if isinstance(construction, ModelThemeConstruction):
        return construction.attempts
    if isinstance(construction, SparseThemeConstruction):
        if construction.merged_prose is None:
            return construction.assignment.attempts
        return construction.assignment.attempts + construction.merged_prose.attempts
    return ()
