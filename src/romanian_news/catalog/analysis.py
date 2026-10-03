from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, ClassVar

from pydantic import ConfigDict, Field

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.attempts import ModelCall
from romanian_news.analysis.embeddings import EmbeddingOutput
from romanian_news.analysis.groups.models import GroupSentimentOutput, GroupSummaryOutput
from romanian_news.analysis.relevance import RelevanceOutput
from romanian_news.analysis.relevance_v3 import RelevanceV3Output
from romanian_news.catalog.artifacts import (
    ArtifactFile,
    artifact_file,
    artifact_statements,
    existing_run_ids,
    run_output_statement,
)
from romanian_news.catalog.model_calls import ModelCallRegistration, model_call_statement
from romanian_news.catalog_transport import (
    advance_artifact_current_version_from_run_statement,
    catalog_batch,
)
from romanian_news.identity import canonical_json, sha256
from romanian_news.storage import publish_immutable_r2_objects


class NewsAnalysisPublication(NewsModel):
    run_ids: tuple[Sha256, ...]
    published_outputs: Annotated[int, Field(ge=0)]
    uploaded_objects: Annotated[int, Field(ge=0)]
    reused_objects: Annotated[int, Field(ge=0)]


class _AnalysisProviderPayload(NewsModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True, strict=True)

    provider_response: dict[str, object] | None = None
    provider_responses: tuple[dict[str, object], ...] = ()
    context_provider_responses: tuple[dict[str, object], ...] = ()
    impact_provider_responses: tuple[dict[str, object], ...] = ()


def publish_relevance_outputs(
    outputs: tuple[RelevanceOutput | RelevanceV3Output, ...],
    implementation_ref: str,
) -> NewsAnalysisPublication:
    """Publish validated relevance responses and their exact article inputs."""
    return _publish_analysis_outputs(outputs, implementation_ref)


def publish_embedding_outputs(
    outputs: tuple[EmbeddingOutput, ...],
    implementation_ref: str,
) -> NewsAnalysisPublication:
    """Publish embedding vectors and their accepted relevance inputs."""
    return _publish_analysis_outputs(outputs, implementation_ref)


def publish_group_analysis_outputs(
    outputs: tuple[GroupSummaryOutput | GroupSentimentOutput, ...],
    implementation_ref: str,
) -> NewsAnalysisPublication:
    """Publish group summaries and sentiment results with direct source lineage."""
    return _publish_analysis_outputs(outputs, implementation_ref)


def _publish_analysis_outputs(
    outputs: tuple[
        RelevanceOutput
        | RelevanceV3Output
        | EmbeddingOutput
        | GroupSummaryOutput
        | GroupSentimentOutput,
        ...,
    ],
    implementation_ref: str,
) -> NewsAnalysisPublication:
    if not implementation_ref:
        raise ValueError("Implementation reference is required")
    run_ids = tuple(_analysis_run_id(output) for output in outputs)
    files = tuple(
        _analysis_file(output, run_id) for output, run_id in zip(outputs, run_ids, strict=True)
    )
    publication = publish_immutable_r2_objects((value.r2_key, value.content) for value in files)
    existing = existing_run_ids(run_ids)
    statements = []
    for output, file, run_id in zip(outputs, files, run_ids, strict=True):
        if run_id in existing:
            statements.append(
                advance_artifact_current_version_from_run_statement(
                    file.artifact_id, file.version_id, run_id
                )
            )
        else:
            statements.extend(
                _analysis_catalog_statements(output, file, run_id, implementation_ref)
            )
    catalog_batch(statements)
    return NewsAnalysisPublication(
        run_ids=run_ids,
        published_outputs=len(outputs),
        uploaded_objects=publication.uploaded_objects,
        reused_objects=publication.reused_objects,
    )


def _analysis_file(
    output: RelevanceOutput
    | RelevanceV3Output
    | EmbeddingOutput
    | GroupSummaryOutput
    | GroupSentimentOutput,
    run_id: Sha256,
) -> ArtifactFile:
    if isinstance(output, RelevanceOutput | RelevanceV3Output):
        artifact_id = f"news:relevance:{output.request_id}"
        kind = "news_relevance"
        directory = "relevance"
        title = f"Relevance decision for {output.article.version_id}"
    elif isinstance(output, EmbeddingOutput):
        artifact_id = f"news:embedding:{output.request_id}"
        kind = "news_embedding"
        directory = "embeddings"
        title = f"Embedding for {output.article.version_id}"
    elif isinstance(output, GroupSummaryOutput):
        artifact_id = f"news:summary:{output.request_id}"
        kind = "news_summary"
        directory = "summaries"
        title = f"Summary for news group {output.group_id}"
    else:
        artifact_id = f"news:sentiment:{output.request_id}"
        kind = "news_sentiment"
        directory = "sentiment"
        title = f"Sentiment for news group {output.group_id}"
    content_digest = sha256(output.content)
    return artifact_file(
        artifact_id=artifact_id,
        artifact_kind=kind,
        title=title,
        content=output.content,
        r2_key=f"news/derived/{directory}/{output.request_id}/{content_digest}.json",
        media_type="application/json",
    )


def _analysis_catalog_statements(
    output: RelevanceOutput
    | RelevanceV3Output
    | EmbeddingOutput
    | GroupSummaryOutput
    | GroupSentimentOutput,
    file: ArtifactFile,
    run_id: Sha256,
    implementation_ref: str,
) -> list[tuple[str, list[object]]]:
    timestamp = datetime.now(UTC).isoformat()
    calls = _analysis_model_calls(output)
    model = ",".join(dict.fromkeys(call.model for call in calls))
    if isinstance(output, RelevanceV3Output):
        operation_key = "news.relevance.v3"
        inputs = ((output.article, "article"),)
    elif isinstance(output, RelevanceOutput):
        operation_key = "news.relevance"
        inputs = ((output.article, "article"),)
    elif isinstance(output, EmbeddingOutput):
        operation_key = "news.embed"
        inputs = ((output.article, "article"), (output.relevance, "relevance"))
    elif isinstance(output, GroupSummaryOutput):
        operation_key = "news.summarize_group"
        inputs = (
            (output.cluster_set, "cluster_set"),
            *((article, "article") for article in output.articles),
        )
    else:
        operation_key = "news.score_group_sentiment"
        inputs = (
            (output.cluster_set, "cluster_set"),
            *((article, "article") for article in output.articles),
        )
    statements: list[tuple[str, list[object]]] = [
        (
            "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                operation_key,
                "openrouter",
                implementation_ref,
                canonical_json({"model": model}).decode(),
                "chartly",
                "publishing",
                f"{operation_key}:{run_id}",
                None,
                timestamp,
                None,
            ],
        )
    ]
    for position, (reference, role) in enumerate(inputs):
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
    statements.extend(artifact_statements(file, timestamp, produced_by_run_id=run_id))
    if isinstance(output, RelevanceOutput | RelevanceV3Output):
        statements.append(
            (
                "INSERT INTO news_relevance_versions VALUES (%s, %s) ON CONFLICT DO NOTHING",
                [file.version_id, int(output.accepted)],
            )
        )
    statements.append(run_output_statement(run_id, file))
    payload = _AnalysisProviderPayload.model_validate_json(output.content)
    responses = _analysis_provider_responses(payload)
    cost = sum(_provider_response_cost(response) for response in responses)
    statements.append(
        model_call_statement(
            ModelCallRegistration(
                artifact_version_id=file.version_id,
                operation_key=operation_key,
                model=model,
                input_tokens=sum(call.input_tokens for call in calls),
                output_tokens=sum(call.output_tokens for call in calls),
                cost_usd=cost,
                latency_ms=sum(call.latency_ms for call in calls),
                response_count=len(responses),
            )
        )
    )
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
    return statements


def _analysis_run_id(
    output: RelevanceOutput
    | RelevanceV3Output
    | EmbeddingOutput
    | GroupSummaryOutput
    | GroupSentimentOutput,
) -> Sha256:
    response_ids = "\0".join(call.response_id for call in _analysis_model_calls(output))
    return sha256(f"{output.request_id}\0{response_ids}".encode())


def _analysis_model_calls(
    output: RelevanceOutput
    | RelevanceV3Output
    | EmbeddingOutput
    | GroupSummaryOutput
    | GroupSentimentOutput,
) -> tuple[ModelCall, ...]:
    if isinstance(output, RelevanceV3Output):
        impact = (output.impact.provider.call,) if output.impact is not None else ()
        return (output.context.provider.call, *impact)
    return (output.call,)


def _analysis_provider_responses(
    payload: _AnalysisProviderPayload,
) -> tuple[dict[str, object], ...]:
    responses = (
        *payload.provider_responses,
        *payload.context_provider_responses,
        *payload.impact_provider_responses,
    )
    if responses or payload.provider_response is None:
        return responses
    return (payload.provider_response,)


def _provider_response_cost(response: dict[str, object]) -> float:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return 0.0
    cost = usage.get("cost")
    if isinstance(cost, bool) or not isinstance(cost, int | float):
        return 0.0
    return float(cost)
