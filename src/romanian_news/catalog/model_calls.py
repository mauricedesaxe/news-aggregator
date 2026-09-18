from datetime import UTC, datetime
from typing import Literal

from romanian_news import NewsModel, Sha256
from romanian_news.catalog_transport import (
    catalog_batch,
    catalog_query,
)


class ModelAttemptRecord(NewsModel):
    attempt_id: Sha256
    request_id: Sha256
    operation_key: str
    response_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    status: Literal["accepted", "rejected"]
    error: str | None
    observed_at: datetime


class ModelTraceRecord(NewsModel):
    provider: Literal["langfuse"]
    trace_id: str
    observation_id: str
    project_ref: str
    recorded_at: datetime


class ModelUsage(NewsModel):
    input_tokens: int
    output_tokens: int
    cost_usd: float


class HistoricalModelOutput(NewsModel):
    kind: str
    version_id: Sha256
    created_at: datetime
    r2_key: str
    content_digest: Sha256
    call_registered: bool


class ModelCallRegistration(NewsModel):
    artifact_version_id: Sha256
    operation_key: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    response_count: int


def record_model_attempt(
    attempt: ModelAttemptRecord, trace: ModelTraceRecord | None = None
) -> None:
    statements = [_attempt_statement(attempt)]
    if trace is not None:
        statements.append(
            (
                "INSERT INTO news_model_trace_links "
                "(attempt_id, provider, trace_id, observation_id, project_ref, recorded_at) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    attempt.attempt_id,
                    trace.provider,
                    trace.trace_id,
                    trace.observation_id,
                    trace.project_ref,
                    trace.recorded_at.astimezone(UTC).isoformat(),
                ],
            )
        )
    catalog_batch(statements, retry_transient_errors=True)


def read_model_usage() -> ModelUsage:
    row = catalog_query(
        "SELECT COALESCE(sum(input_tokens), 0) AS input_tokens, "
        "COALESCE(sum(output_tokens), 0) AS output_tokens, "
        "COALESCE(sum(cost_usd), 0) AS cost_usd FROM news_model_attempts"
    )[0]
    return ModelUsage(
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        cost_usd=round(float(row["cost_usd"]), 8),
    )


def read_historical_model_outputs() -> tuple[HistoricalModelOutput, ...]:
    rows = catalog_query(
        """
        SELECT artifact.kind, version.id AS version_id, version.created_at, file.r2_key,
               file.content_digest, call.artifact_version_id AS registered_call_version_id
        FROM artifacts artifact
        JOIN artifact_versions version ON version.artifact_id = artifact.id
        JOIN artifact_files file ON file.artifact_version_id = version.id
        LEFT JOIN news_model_calls call ON call.artifact_version_id = version.id
        WHERE artifact.kind IN ('news_relevance', 'news_embedding', 'news_summary', 'news_sentiment')
        ORDER BY version.created_at, version.id
        """
    )
    return tuple(
        HistoricalModelOutput(
            kind=str(row["kind"]),
            version_id=str(row["version_id"]),
            created_at=datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00")),
            r2_key=str(row["r2_key"]),
            content_digest=str(row["content_digest"]),
            call_registered=row["registered_call_version_id"] is not None,
        )
        for row in rows
    )


def read_existing_model_attempt_ids() -> frozenset[Sha256]:
    return frozenset(
        str(row["attempt_id"])
        for row in catalog_query("SELECT attempt_id FROM news_model_attempts")
    )


def register_historical_model_calls(
    attempts: tuple[ModelAttemptRecord, ...], calls: tuple[ModelCallRegistration, ...]
) -> int:
    statements = [_attempt_statement(attempt) for attempt in attempts]
    statements.extend(
        (
            "INSERT INTO news_model_calls VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                call.artifact_version_id,
                call.operation_key,
                call.model,
                call.input_tokens,
                call.output_tokens,
                call.cost_usd,
                call.latency_ms,
                call.response_count,
            ],
        )
        for call in calls
    )
    catalog_batch(statements)
    return len(statements)


def _attempt_statement(attempt: ModelAttemptRecord) -> tuple[str, list[object]]:
    return (
        "INSERT INTO news_model_attempts "
        "(attempt_id, request_id, operation_key, response_id, model, input_tokens, output_tokens, "
        "cost_usd, latency_ms, status, error, observed_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
        [
            attempt.attempt_id,
            attempt.request_id,
            attempt.operation_key,
            attempt.response_id,
            attempt.model,
            attempt.input_tokens,
            attempt.output_tokens,
            attempt.cost_usd,
            attempt.latency_ms,
            attempt.status,
            attempt.error,
            attempt.observed_at.astimezone(UTC).isoformat(),
        ],
    )
