from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.tracing import ModelTraceReference
from romanian_news.identity import canonical_json as _canonical_json
from romanian_news.identity import sha256 as _sha256


class ModelCall(NewsModel):
    response_id: Annotated[str, Field(min_length=1)]
    model: Annotated[str, Field(min_length=1)]
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    latency_ms: Annotated[int, Field(ge=0)]


class ModelAttempt(NewsModel):
    attempt_id: Sha256
    request_id: Sha256
    operation_key: Annotated[str, Field(min_length=1)]
    response_id: Annotated[str, Field(min_length=1)]
    model: Annotated[str, Field(min_length=1)]
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    cost_usd: Annotated[float, Field(ge=0)]
    latency_ms: Annotated[int, Field(ge=0)]
    status: Literal["accepted", "rejected"]
    error: str | None
    observed_at: datetime

    @model_validator(mode="after")
    def validate_status(self) -> ModelAttempt:
        if self.status == "accepted" and self.error is not None:
            raise ValueError("An accepted model attempt cannot have an error")
        if self.status == "rejected" and not self.error:
            raise ValueError("A rejected model attempt requires an error")
        return self


class ProviderResponse(Protocol):
    def model_dump(self, *, mode: str) -> dict[str, object]: ...


def record_model_attempt(
    response: ProviderResponse,
    *,
    request_id: Sha256,
    operation_key: str,
    attempt_index: int,
    latency_ms: int,
    status: Literal["accepted", "rejected"],
    error: str | None,
    fallback_response_id: str,
    trace: ModelTraceReference | None = None,
) -> ModelAttempt:
    """Record one validated provider response without its raw payload."""
    attempt = model_attempt_from_payload(
        response.model_dump(mode="json"),
        request_id=request_id,
        operation_key=operation_key,
        attempt_index=attempt_index,
        latency_ms=latency_ms,
        status=status,
        error=error,
        observed_at=datetime.now(UTC),
        fallback_response_id=fallback_response_id,
    )
    from romanian_news.catalog.model_calls import (
        ModelAttemptRecord,
        ModelTraceRecord,
    )
    from romanian_news.catalog.model_calls import (
        record_model_attempt as write_model_attempt,
    )

    attempt_record = ModelAttemptRecord.model_validate(attempt.model_dump(), strict=True)
    trace_record = (
        ModelTraceRecord(
            provider=trace.provider,
            trace_id=trace.trace_id,
            observation_id=trace.observation_id,
            project_ref=trace.project_ref,
            recorded_at=trace.recorded_at,
        )
        if trace is not None
        else None
    )
    write_model_attempt(attempt_record, trace_record)
    return attempt


def model_attempt_from_payload(
    response: Mapping[str, object],
    *,
    request_id: Sha256,
    operation_key: str,
    attempt_index: int,
    latency_ms: int,
    status: Literal["accepted", "rejected"],
    error: str | None,
    observed_at: datetime,
    fallback_response_id: str | None = None,
) -> ModelAttempt:
    """Extract durable accounting fields from one provider response."""
    response_id = str(
        response.get("id")
        or fallback_response_id
        or _fallback_response_id(request_id, attempt_index)
    )
    model = str(response.get("model") or "unknown")
    usage_value = response.get("usage")
    usage: Mapping[str, object] = usage_value if isinstance(usage_value, Mapping) else {}
    completion_tokens = usage.get("completion_tokens")
    if operation_key == "news.embed" and completion_tokens is None:
        completion_tokens = 0
    observed = _response_time(response.get("created")) or observed_at
    attempt_id = _sha256(
        _canonical_json(
            {
                "attempt_index": attempt_index,
                "operation_key": operation_key,
                "request_id": request_id,
                "response_id": response_id,
            }
        )
    )
    return ModelAttempt(
        attempt_id=attempt_id,
        request_id=request_id,
        operation_key=operation_key,
        response_id=response_id,
        model=model,
        input_tokens=_payload_int(usage.get("prompt_tokens")),
        output_tokens=_payload_int(completion_tokens),
        cost_usd=_payload_float(usage.get("cost")),
        latency_ms=latency_ms,
        status=status,
        error=error,
        observed_at=observed,
    )


def _payload_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"Expected an integer in the recorded payload, received {value!r}")


def _payload_float(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"Expected a number in the recorded payload, received {value!r}")


def response_cost(response: Mapping[str, object]) -> float:
    """Sum-safe cost of one recorded provider response, zero when absent."""
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return 0.0
    cost = usage.get("cost")
    return float(cost) if isinstance(cost, int | float) and not isinstance(cost, bool) else 0.0


def _response_time(value: object) -> datetime | None:
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _fallback_response_id(request_id: Sha256, attempt_index: int) -> str:
    return request_id if attempt_index == 0 else f"{request_id}:{attempt_index}"
