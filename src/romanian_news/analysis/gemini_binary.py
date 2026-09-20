from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Annotated, ClassVar, Literal

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.binary_evaluation import (
    BinaryAttemptCallback,
    BinaryAttemptError,
    BinaryAttemptEvidence,
    BinaryProbabilityObservation,
    BinaryRequest,
    binary_decision,
)
from romanian_news.analysis.client import openrouter_client
from romanian_news.analysis.tracing import ProviderChatRequest

_PURPOSE = "news.binary.gemini.evaluation"
_PROMPT_SERIALIZATION = "canonical-binary-system-and-json-v1"
_RESPONSE_SCHEMA = "forced-probability-tool-v1"
_USAGE_ADAPTER = TypeAdapter(dict[str, object])


class GeminiBinaryExecutionPolicy(NewsModel):
    model: Literal["google/gemini-2.5-flash", "google/gemini-3.8-flash"]
    temperature: Annotated[float, Field(ge=0, le=0)]
    max_tokens: Annotated[int, Field(gt=0)]
    provider_require_parameters: Literal[True]
    sdk_max_retries: Annotated[int, Field(ge=0)]
    timeout_seconds: Annotated[float, Field(gt=0)]


GEMINI_25_FLASH_BINARY_EXECUTION_POLICY = GeminiBinaryExecutionPolicy(
    model="google/gemini-2.5-flash",
    temperature=0.0,
    max_tokens=1024,
    provider_require_parameters=True,
    sdk_max_retries=1,
    timeout_seconds=30,
)
GEMINI_38_FLASH_BINARY_EXECUTION_POLICY = GeminiBinaryExecutionPolicy(
    model="google/gemini-3.8-flash",
    temperature=0.0,
    max_tokens=1024,
    provider_require_parameters=True,
    sdk_max_retries=1,
    timeout_seconds=30,
)
GEMINI_BINARY_EXECUTION_POLICY = GEMINI_25_FLASH_BINARY_EXECUTION_POLICY


class GeminiBinaryResponse(NewsModel):
    probability: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]


class _ProviderModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")


class _ProviderFunction(_ProviderModel):
    name: str
    arguments: str


class _ProviderToolCall(_ProviderModel):
    type: Literal["function"]
    function: _ProviderFunction


class _ProviderMessage(_ProviderModel):
    tool_calls: list[_ProviderToolCall] | None = None


class _ProviderChoice(_ProviderModel):
    message: _ProviderMessage


class _ProviderCompletion(_ProviderModel):
    choices: list[_ProviderChoice]


def evaluate_gemini_binary(
    request: BinaryRequest,
    *,
    execution_ref: str,
    policy: GeminiBinaryExecutionPolicy = GEMINI_BINARY_EXECUTION_POLICY,
    clock: Callable[[], float] = time.monotonic,
    on_attempt: BinaryAttemptCallback | None = None,
) -> BinaryProbabilityObservation:
    """Evaluate one SDK call as one observable attempt.

    The OpenAI SDK may retry transports internally. Those hidden retries remain one boundary
    attempt because the adapter cannot observe their individual responses or accounting.
    """
    request_id = gemini_binary_request_id(request, execution_ref, policy)
    provider_request = _provider_request(request, policy)
    started = _clock_value(clock(), "start")
    response: ChatCompletion | None = None
    validated: GeminiBinaryResponse | None = None
    try:
        response = openrouter_client(
            max_retries=policy.sdk_max_retries,
            timeout_seconds=policy.timeout_seconds,
        ).chat.completions.create(**provider_request)
        validated = _parse_response(response)
        provider_request_id, response_model, input_tokens, output_tokens, cost = _accounting(
            response
        )
    except Exception as error:
        latency_ms = _latency_ms(started, _clock_value(clock(), "finish"))
        (
            provider_request_id,
            response_model,
            input_tokens,
            output_tokens,
            cost,
        ) = _partial_accounting(response)
        probability = (
            validated.probability if validated is not None else _partial_probability(response)
        )
        status_code = _error_status(error)
        _notify_attempt(
            on_attempt,
            BinaryAttemptEvidence(
                attempt_number=1,
                status="terminal_error",
                provider_request_id=provider_request_id or _error_request_id(error),
                actual_model=response_model,
                http_status=status_code,
                error=BinaryAttemptError(
                    error_type=type(error).__name__,
                    message=str(error) or type(error).__name__,
                    retryable=_retryable_error(status_code),
                ),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                probability=probability,
            ),
        )
        raise

    latency_ms = _latency_ms(started, _clock_value(clock(), "finish"))
    probability = validated.probability
    attempt = BinaryAttemptEvidence(
        attempt_number=1,
        status="completed",
        provider_request_id=provider_request_id,
        actual_model=response_model,
        http_status=None,
        error=None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
        probability=probability,
    )
    _notify_attempt(on_attempt, attempt)
    return BinaryProbabilityObservation(
        request_id=request_id,
        provider_request_id=provider_request_id,
        model=response_model,
        probability=probability,
        predicted_accepted=binary_decision(probability, request.question.threshold),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        estimated_cost_usd=cost,
        attempts=(attempt,),
    )


def gemini_binary_execution_policy_digest(
    policy: GeminiBinaryExecutionPolicy = GEMINI_BINARY_EXECUTION_POLICY,
) -> Sha256:
    return _sha256(
        _canonical_json(
            {
                "execution": policy.model_dump(mode="json"),
                "prompt_serialization": _PROMPT_SERIALIZATION,
                "response_schema": _RESPONSE_SCHEMA,
            }
        )
    )


def gemini_binary_request_id(
    request: BinaryRequest,
    execution_ref: str,
    policy: GeminiBinaryExecutionPolicy = GEMINI_BINARY_EXECUTION_POLICY,
) -> Sha256:
    if not execution_ref.strip():
        raise ValueError("Gemini binary evaluation requires an execution reference")
    return _sha256(
        _canonical_json(
            {
                "semantic_digest": request.question.semantic_digest,
                "state_digest": request.state_digest,
                "execution_ref": execution_ref,
                "execution_policy_digest": gemini_binary_execution_policy_digest(policy),
                "purpose": _PURPOSE,
            }
        )
    )


def _provider_request(
    request: BinaryRequest,
    policy: GeminiBinaryExecutionPolicy,
) -> ProviderChatRequest:
    question = request.question
    semantics = {
        "false_criteria": question.false_criteria,
        "state": request.state,
        "true_criteria": question.true_criteria,
    }
    return {
        "model": policy.model,
        "messages": [
            {"role": "system", "content": question.instructions},
            {
                "role": "user",
                "content": _canonical_json(semantics).decode(),
            },
        ],
        "temperature": policy.temperature,
        "max_tokens": policy.max_tokens,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "report_probability",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "probability": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            }
                        },
                        "required": ["probability"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "report_probability"},
        },
        "extra_body": {"provider": {"require_parameters": policy.provider_require_parameters}},
    }


def _parse_response(response: ChatCompletion) -> GeminiBinaryResponse:
    try:
        completion = _ProviderCompletion.model_validate(
            response.model_dump(mode="json"), strict=True
        )
        tool_calls = completion.choices[0].message.tool_calls
    except (AttributeError, IndexError, ValidationError) as error:
        raise ValueError("Gemini response omitted the probability tool call") from error
    if tool_calls is None or len(tool_calls) != 1:
        raise ValueError("Gemini response omitted the probability tool call")
    tool_call = tool_calls[0]
    if tool_call.function.name != "report_probability":
        raise ValueError("Gemini response called the wrong tool")
    return GeminiBinaryResponse.model_validate_json(tool_call.function.arguments, strict=True)


def _accounting(response: ChatCompletion) -> tuple[str, str, int, int, Decimal]:
    provider_request_id = _non_empty_text(response.id, "provider request ID")
    response_model = _non_empty_text(response.model, "response model")
    payload = response.model_dump(mode="json")
    try:
        usage_values = _USAGE_ADAPTER.validate_python(payload.get("usage"), strict=True)
    except ValidationError as error:
        raise ValueError("OpenRouter response omitted usage accounting") from error
    input_tokens = _token_count(usage_values.get("prompt_tokens"), "prompt_tokens")
    output_tokens = _token_count(usage_values.get("completion_tokens"), "completion_tokens")
    cost = _non_negative_number(usage_values.get("cost"), "cost")
    return provider_request_id, response_model, input_tokens, output_tokens, cost


def _partial_accounting(
    response: ChatCompletion | None,
) -> tuple[str | None, str | None, int | None, int | None, Decimal | None]:
    if response is None:
        return None, None, None, None, None
    provider_request_id = _optional_non_empty_text(getattr(response, "id", None))
    response_model = _optional_non_empty_text(getattr(response, "model", None))
    try:
        payload = response.model_dump(mode="json")
    except Exception:
        return provider_request_id, response_model, None, None, None
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return provider_request_id, response_model, None, None, None
    return (
        provider_request_id,
        response_model,
        _optional_token_count(usage.get("prompt_tokens")),
        _optional_token_count(usage.get("completion_tokens")),
        _optional_cost(usage.get("cost")),
    )


def _non_empty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"OpenRouter response omitted {label}")
    return value


def _optional_non_empty_text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _token_count(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"OpenRouter response has invalid {label}")
    return value


def _optional_token_count(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return value


def _non_negative_number(value: object, label: str) -> Decimal:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"OpenRouter response has invalid {label}")
    return Decimal(str(value))


def _optional_cost(value: object) -> Decimal | None:
    try:
        return _non_negative_number(value, "cost")
    except ValueError:
        return None


def _partial_probability(response: ChatCompletion | None) -> Decimal | None:
    if response is None:
        return None
    try:
        completion = _ProviderCompletion.model_validate(
            response.model_dump(mode="json"), strict=True
        )
        tool_calls = completion.choices[0].message.tool_calls
        if tool_calls is None or len(tool_calls) != 1:
            return None
        payload = json.loads(
            tool_calls[0].function.arguments,
            parse_float=Decimal,
            parse_int=Decimal,
        )
    except (AttributeError, IndexError, json.JSONDecodeError, TypeError, ValidationError):
        return None
    if not isinstance(payload, dict):
        return None
    probability = payload.get("probability")
    if isinstance(probability, Decimal) and Decimal(0) <= probability <= Decimal(1):
        return probability
    return None


def _notify_attempt(
    callback: BinaryAttemptCallback | None,
    evidence: BinaryAttemptEvidence,
) -> None:
    if callback is not None:
        callback(evidence)


def _error_status(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
        return value
    return None


def _error_request_id(error: Exception) -> str | None:
    return _optional_non_empty_text(getattr(error, "request_id", None))


def _retryable_error(status_code: int | None) -> bool:
    return status_code in (408, 429) or (status_code is not None and 500 <= status_code < 600)


def _clock_value(value: object, point: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"Gemini binary evaluation {point} latency clock is invalid")
    return float(value)


def _latency_ms(started: float, finished: float) -> int:
    elapsed = finished - started
    if elapsed < 0:
        raise ValueError("Gemini binary evaluation latency cannot be negative")
    return round(elapsed * 1000)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(content: bytes) -> Sha256:
    return hashlib.sha256(content).hexdigest()
