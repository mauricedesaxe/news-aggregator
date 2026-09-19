from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from typing import Generic, Literal, Protocol, TypedDict, TypeVar
from uuid import UUID, uuid4

from langfuse import Langfuse
from langfuse.types import TraceContext
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema

from romanian_news.config import (
    LANGFUSE_BASE_URL,
    LANGFUSE_PROJECT_ID,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)

logger = logging.getLogger(__name__)
_TRACE_PROVIDER = "langfuse"


class ProviderChatRequest(TypedDict):
    """One typed chat-completions request for the OpenAI-compatible provider."""

    model: str
    messages: list[ChatCompletionMessageParam]
    temperature: float
    max_tokens: int
    response_format: ResponseFormatJSONSchema
    extra_body: dict[str, object]


class ProviderEmbeddingRequest(TypedDict):
    """One typed embeddings request for the OpenAI-compatible provider."""

    model: str
    input: str | list[str]
    dimensions: int
    encoding_format: Literal["float"]


@dataclass(frozen=True)
class ModelTraceReference:
    provider: Literal["langfuse"]
    trace_id: str
    observation_id: str
    project_ref: str
    recorded_at: datetime


class TraceableProviderResponse(Protocol):
    def model_dump(self, *, mode: str) -> dict[str, object]: ...


ProviderResponseT = TypeVar("ProviderResponseT", bound=TraceableProviderResponse)


@dataclass(frozen=True)
class ProviderCallResult(Generic[ProviderResponseT]):
    response: ProviderResponseT
    call_id: UUID
    trace: ModelTraceReference | None


def trace_provider_call(
    operation_key: str,
    request_id: str,
    inputs: Mapping[str, object],
    call: Callable[[], ProviderResponseT],
) -> ProviderCallResult[ProviderResponseT]:
    """Call one provider and return its durable trace identity when available."""
    call_id = uuid4()
    if not langfuse_tracing_available():
        return ProviderCallResult(response=call(), call_id=call_id, trace=None)
    assert LANGFUSE_PROJECT_ID is not None

    try:
        client = _langfuse_client()
        trace_context: TraceContext = {"trace_id": client.create_trace_id(seed=str(call_id))}
        model_value = inputs.get("model")
        observation = client.start_observation(
            trace_context=trace_context,
            name=operation_key,
            as_type="generation",
            input=copy.deepcopy(dict(inputs)),
            metadata={"operation_key": operation_key, "request_id": request_id},
            model=model_value if isinstance(model_value, str) else None,
        )
    except Exception:
        logger.exception("Langfuse trace setup failed for %s", operation_key)
        return ProviderCallResult(response=call(), call_id=call_id, trace=None)

    reference = ModelTraceReference(
        provider=_TRACE_PROVIDER,
        trace_id=observation.trace_id,
        observation_id=observation.id,
        project_ref=LANGFUSE_PROJECT_ID,
        recorded_at=datetime.now(UTC),
    )
    try:
        response = call()
    except Exception as error:
        _finish_observation(
            observation,
            output={"error": f"{type(error).__name__}: {error}"},
            level="ERROR",
            status_message=str(error),
        )
        _flush_trace(client, operation_key)
        raise

    _finish_response_observation(observation, response)
    trace = reference if _flush_trace(client, operation_key) else None
    return ProviderCallResult(response=response, call_id=call_id, trace=trace)


def flush_langfuse_traces() -> None:
    """Flush queued traces without changing the caller result."""
    if not langfuse_tracing_available():
        return
    try:
        _langfuse_client().flush()
    except Exception:
        logger.exception("Langfuse trace flush failed")


@cache
def langfuse_tracing_available() -> bool:
    """Check trace credentials once before provider work starts."""
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY or not LANGFUSE_PROJECT_ID:
        return False
    try:
        return _langfuse_client().auth_check()
    except Exception as error:
        logger.warning("Langfuse tracing disabled: %s", error)
        return False


def _usage_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _usage_float(value: object) -> float:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _finish_response_observation(
    observation: object,
    response: TraceableProviderResponse,
) -> None:
    payload = response.model_dump(mode="json")
    usage = payload.get("usage")
    usage_values: Mapping[str, object] = usage if isinstance(usage, Mapping) else {}
    values: dict[str, object] = {
        "output": {"provider_response": payload},
        "usage_details": {
            "input": _usage_int(usage_values.get("prompt_tokens")),
            "output": _usage_int(usage_values.get("completion_tokens")),
        },
    }
    if cost := _usage_float(usage_values.get("cost")):
        values["cost_details"] = {"total": cost}
    _finish_observation(observation, **values)


def _finish_observation(observation, **values: object) -> None:
    try:
        observation.update(**values)
    except Exception:
        logger.exception("Langfuse trace completion failed")
    try:
        observation.end()
    except Exception:
        logger.exception("Langfuse trace completion failed")


def _flush_trace(client: Langfuse, operation_key: str) -> bool:
    try:
        client.flush()
    except Exception:
        logger.exception("Langfuse trace flush failed for %s", operation_key)
        return False
    return True


@cache
def _langfuse_client() -> Langfuse:
    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        base_url=LANGFUSE_BASE_URL,
    )
