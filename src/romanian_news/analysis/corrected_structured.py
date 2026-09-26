from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, cast

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import ValidationError

from romanian_news import Sha256
from romanian_news.analysis.attempts import ModelCall, record_model_attempt
from romanian_news.analysis.client import openrouter_client
from romanian_news.analysis.tracing import ProviderChatRequest, trace_provider_call
from romanian_news.identity import sha256

_Parsed = TypeVar("_Parsed")
_MISSING = object()
AttemptStatus = Literal["accepted", "rejected"]


@dataclass(frozen=True)
class StructuredMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class StructuredAttempt:
    attempt_id: Sha256
    response_id: str
    status: AttemptStatus
    error: str | None
    response_content: str
    response_content_digest: Sha256
    provider_response: dict[str, object]


@dataclass(frozen=True)
class CorrectedStructuredResult(Generic[_Parsed]):
    value: _Parsed
    messages: tuple[StructuredMessage, ...]
    call: ModelCall
    attempts: tuple[StructuredAttempt, ...]


def run_corrected_structured_openrouter(
    *,
    operation: str,
    request_id: Sha256,
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
    schema_name: str,
    response_schema: dict[str, object],
    initial_messages: tuple[StructuredMessage, StructuredMessage],
    parse: Callable[[str], _Parsed],
    correction_message: Callable[[str], str],
    exhausted_error: Callable[[str], ValueError],
    unreachable_error: str,
    started_at: float,
) -> CorrectedStructuredResult[_Parsed]:
    messages = list(initial_messages)
    attempts: list[StructuredAttempt] = []
    responses: list[ChatCompletion] = []
    accepted: _Parsed | object = _MISSING
    for attempt_index in range(2):
        provider_inputs: ProviderChatRequest = {
            "model": model,
            "messages": [_message_param(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                },
            },
            "extra_body": {
                "provider": {"require_parameters": True},
                "reasoning": {"effort": reasoning_effort},
            },
        }
        attempt_started = time.monotonic()
        provider_call = trace_provider_call(
            operation,
            request_id,
            provider_inputs,
            lambda provider_inputs=provider_inputs: (
                openrouter_client().chat.completions.create(**provider_inputs)
            ),
        )
        response = provider_call.response
        responses.append(response)
        content = response.choices[0].message.content or ""
        error_text = None
        rejection: ValidationError | ValueError | None = None
        try:
            accepted = parse(content)
        except (ValidationError, ValueError) as error:
            rejection = error
            error_text = str(error)
        status: AttemptStatus = "accepted" if error_text is None else "rejected"
        recorded = record_model_attempt(
            response,
            request_id=request_id,
            operation_key=operation,
            attempt_index=attempt_index,
            latency_ms=round((time.monotonic() - attempt_started) * 1000),
            status=status,
            error=error_text,
            fallback_response_id=str(provider_call.call_id),
            trace=provider_call.trace,
        )
        attempts.append(
            StructuredAttempt(
                attempt_id=recorded.attempt_id,
                response_id=recorded.response_id,
                status=status,
                error=error_text,
                response_content=content,
                response_content_digest=sha256(content.encode()),
                provider_response=response.model_dump(mode="json"),
            )
        )
        if accepted is not _MISSING and error_text is None:
            break
        assert error_text is not None
        if attempt_index == 1:
            assert rejection is not None
            raise exhausted_error(error_text) from rejection
        messages.extend(
            (
                StructuredMessage(role="assistant", content=content),
                StructuredMessage(role="user", content=correction_message(error_text)),
            )
        )
    if accepted is _MISSING:
        raise RuntimeError(unreachable_error)
    return CorrectedStructuredResult(
        value=cast(_Parsed, accepted),
        messages=tuple(messages),
        call=ModelCall(
            response_id=attempts[-1].response_id,
            model=str(responses[-1].model),
            input_tokens=sum(item.usage.prompt_tokens if item.usage else 0 for item in responses),
            output_tokens=sum(
                item.usage.completion_tokens if item.usage else 0 for item in responses
            ),
            latency_ms=round((time.monotonic() - started_at) * 1000),
        ),
        attempts=tuple(attempts),
    )


def _message_param(message: StructuredMessage) -> ChatCompletionMessageParam:
    if message.role == "system":
        return {"role": "system", "content": message.content}
    if message.role == "assistant":
        return {"role": "assistant", "content": message.content}
    return {"role": "user", "content": message.content}
