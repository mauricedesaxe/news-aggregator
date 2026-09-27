from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Annotated, Literal

import requests
from pydantic import Field, StringConstraints

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.binary_evaluation import (
    BinaryAttemptCallback,
    BinaryAttemptError,
    BinaryAttemptEvidence,
    BinaryProbabilityObservation,
    BinaryRequest,
    binary_attempt_totals,
    binary_decision,
    build_relevance_binary_request,
)
from romanian_news.analysis.relevance import ArticleAnalysisInput
from romanian_news.config import TYPESAFE_API_KEY
from romanian_news.identity import canonical_json as _canonical_json
from romanian_news.identity import sha256 as _sha256

JEV_RELEVANCE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_INPUT_COST_PER_MILLION_TOKENS_USD = 0.042


class JevExecutionPolicy(NewsModel):
    endpoint: Annotated[str, StringConstraints(min_length=1)]
    model: Literal["jev-1.13.0"]
    timeout_seconds: Annotated[float, Field(gt=0)]
    max_attempts: Annotated[int, Field(gt=0)]
    base_retry_delay_seconds: Annotated[float, Field(ge=0)]
    max_retry_delay_seconds: Annotated[float, Field(ge=0)]
    input_cost_per_million_tokens_usd: Annotated[float, Field(ge=0)]


JEV_EXECUTION_POLICY = JevExecutionPolicy(
    endpoint=JEV_RELEVANCE_ENDPOINT,
    model="jev-1.13.0",
    timeout_seconds=60,
    max_attempts=4,
    base_retry_delay_seconds=1,
    max_retry_delay_seconds=30,
    input_cost_per_million_tokens_usd=JEV_INPUT_COST_PER_MILLION_TOKENS_USD,
)


class NoulAnswer(NewsModel):
    type: Literal["noul"]
    noul: Annotated[Decimal, Field(ge=0, le=1)]


class JevUsage(NewsModel):
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]


class JevNoulResponse(NewsModel):
    model: Literal["jev-1.13.0"]
    answers: dict[str, NoulAnswer]
    usage: JevUsage


class JevRelevanceObservation(BinaryProbabilityObservation):
    pass


def evaluate_jev_relevance(
    value: ArticleAnalysisInput | BinaryRequest,
    *,
    execution_ref: str,
    policy: JevExecutionPolicy = JEV_EXECUTION_POLICY,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    on_attempt: BinaryAttemptCallback | None = None,
) -> JevRelevanceObservation:
    if not execution_ref.strip():
        raise ValueError("Jev relevance evaluation requires an execution reference")
    if not TYPESAFE_API_KEY:
        raise ValueError("TYPESAFE_API_KEY is required for Jev relevance evaluation")
    request = value if isinstance(value, BinaryRequest) else build_relevance_binary_request(value)
    headers = {
        "Authorization": f"Bearer {TYPESAFE_API_KEY}",
        "Content-Type": "application/json",
    }
    attempts: list[BinaryAttemptEvidence] = []
    for attempt_index in range(policy.max_attempts):
        started = _clock_value(clock())
        candidate: requests.Response | None = None
        try:
            candidate = requests.post(
                policy.endpoint,
                headers=headers,
                json=_request_payload(request, policy),
                timeout=policy.timeout_seconds,
            )
            candidate.raise_for_status()
            validated, probability = _parse_response(
                candidate.content, request.question.question_id
            )
        except requests.RequestException as error:
            latency_ms = _elapsed_ms(started, _clock_value(clock()))
            status_code = error.response.status_code if error.response is not None else None
            retryable = status_code is None or status_code in (408, 429) or 500 <= status_code < 600
            will_retry = retryable and attempt_index + 1 < policy.max_attempts
            actual_model, input_tokens, output_tokens, cost_usd, probability = (
                _partial_response_evidence(candidate, policy)
            )
            _emit_attempt(
                attempts,
                BinaryAttemptEvidence(
                    attempt_number=attempt_index + 1,
                    status="retryable_error" if will_retry else "terminal_error",
                    provider_request_id=_provider_request_id(candidate),
                    actual_model=actual_model,
                    http_status=_http_status(status_code),
                    error=_attempt_error(error, retryable=retryable),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    probability=probability,
                ),
                on_attempt,
            )
            if not will_retry:
                raise RuntimeError("Jev relevance request failed") from error
            retry_after_seconds = (
                _retry_after_seconds(candidate.headers) if candidate is not None else None
            )
            delay = (
                retry_after_seconds
                if retry_after_seconds is not None
                else policy.base_retry_delay_seconds * 2.0**attempt_index
            )
            sleep(min(delay, policy.max_retry_delay_seconds))
            continue
        except Exception as error:
            latency_ms = _elapsed_ms(started, _clock_value(clock()))
            actual_model, input_tokens, output_tokens, cost_usd, probability = (
                _partial_response_evidence(candidate, policy)
            )
            _emit_attempt(
                attempts,
                BinaryAttemptEvidence(
                    attempt_number=attempt_index + 1,
                    status="terminal_error",
                    provider_request_id=_provider_request_id(candidate),
                    actual_model=actual_model,
                    http_status=_response_status(candidate),
                    error=_attempt_error(error, retryable=False),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    probability=probability,
                ),
                on_attempt,
            )
            raise

        latency_ms = _elapsed_ms(started, _clock_value(clock()))
        input_tokens = validated.usage.input_tokens
        cost_usd = _estimated_cost(input_tokens, policy)
        _emit_attempt(
            attempts,
            BinaryAttemptEvidence(
                attempt_number=attempt_index + 1,
                status="completed",
                provider_request_id=_provider_request_id(candidate),
                actual_model=validated.model,
                http_status=_response_status(candidate),
                error=None,
                input_tokens=input_tokens,
                output_tokens=validated.usage.output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                probability=probability,
            ),
            on_attempt,
        )
        total_input, total_output, total_cost, total_latency = binary_attempt_totals(
            tuple(attempts)
        )
        return JevRelevanceObservation(
            request_id=jev_relevance_request_id(request, execution_ref, policy),
            provider_request_id=_provider_request_id(candidate),
            model=validated.model,
            probability=probability,
            predicted_accepted=binary_decision(probability, request.question.threshold),
            input_tokens=total_input,
            output_tokens=total_output,
            latency_ms=total_latency,
            estimated_cost_usd=total_cost,
            attempts=tuple(attempts),
        )
    raise AssertionError("A valid Jev retry policy always returns or raises")


def jev_execution_policy_digest(
    policy: JevExecutionPolicy = JEV_EXECUTION_POLICY,
) -> Sha256:
    return _sha256(_canonical_json(policy.model_dump(mode="json")))


def jev_relevance_request_id(
    request: BinaryRequest,
    execution_ref: str,
    policy: JevExecutionPolicy = JEV_EXECUTION_POLICY,
) -> Sha256:
    if not execution_ref.strip():
        raise ValueError("Jev relevance evaluation requires an execution reference")
    return _sha256(
        _canonical_json(
            {
                "question_digest": request.question.semantic_digest,
                "state_digest": request.state_digest,
                "execution_ref": execution_ref,
                "execution_policy_digest": jev_execution_policy_digest(policy),
                "purpose": "news.relevance.jev.evaluation",
            }
        )
    )


def _request_payload(request: BinaryRequest, policy: JevExecutionPolicy) -> dict[str, object]:
    question = request.question
    return {
        "state": request.state,
        "model": policy.model,
        "questions": {
            question.question_id: {
                "type": "noul",
                "instructions": question.instructions,
                "criteria": {"true": question.true_criteria, "false": question.false_criteria},
            }
        },
    }


def _parse_response(content: bytes, question_id: str) -> tuple[JevNoulResponse, Decimal]:
    response = JevNoulResponse.model_validate_json(content, strict=True)
    try:
        answer = response.answers[question_id]
    except KeyError as error:
        raise ValueError(f"Jev response omitted binary question {question_id!r}") from error
    return response, answer.noul


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    milliseconds = headers.get("retry-after-ms")
    seconds = headers.get("retry-after")
    try:
        if milliseconds is not None:
            return max(0.0, float(milliseconds) / 1000)
        if seconds is not None:
            return max(0.0, float(seconds))
    except ValueError:
        return None
    return None


def _emit_attempt(
    attempts: list[BinaryAttemptEvidence],
    evidence: BinaryAttemptEvidence,
    callback: BinaryAttemptCallback | None,
) -> None:
    attempts.append(evidence)
    if callback is not None:
        callback(evidence)


def _attempt_error(error: Exception, *, retryable: bool) -> BinaryAttemptError:
    return BinaryAttemptError(
        error_type=type(error).__name__,
        message=str(error) or type(error).__name__,
        retryable=retryable,
    )


def _provider_request_id(response: requests.Response | None) -> str | None:
    if response is None:
        return None
    value = response.headers.get("x-typesafe-request-id")
    return value if isinstance(value, str) and value else None


def _response_status(response: requests.Response | None) -> int | None:
    return _http_status(response.status_code) if response is not None else None


def _http_status(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _estimated_cost(input_tokens: int, policy: JevExecutionPolicy) -> Decimal:
    return (
        Decimal(input_tokens)
        / Decimal(1_000_000)
        * Decimal(str(policy.input_cost_per_million_tokens_usd))
    )


def _partial_response_evidence(
    response: requests.Response | None,
    policy: JevExecutionPolicy,
) -> tuple[str | None, int | None, int | None, Decimal | None, Decimal | None]:
    if response is None:
        return None, None, None, None, None
    try:
        payload = json.loads(response.content, parse_float=Decimal, parse_int=Decimal)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None, None, None, None, None
    if not isinstance(payload, dict):
        return None, None, None, None, None
    model = payload.get("model")
    actual_model = model if isinstance(model, str) and model else None
    usage = payload.get("usage")
    input_tokens = None
    output_tokens = None
    if isinstance(usage, dict):
        input_tokens = _partial_token_count(usage.get("input_tokens"))
        output_tokens = _partial_token_count(usage.get("output_tokens"))
    answers = payload.get("answers")
    probability = None
    if isinstance(answers, dict):
        for answer in answers.values():
            if isinstance(answer, dict):
                probability = _partial_probability(answer.get("noul"))
                if probability is not None:
                    break
    cost = _estimated_cost(input_tokens, policy) if input_tokens is not None else None
    return actual_model, input_tokens, output_tokens, cost, probability


def _partial_token_count(value: object) -> int | None:
    if isinstance(value, Decimal) and value == value.to_integral_value() and value >= 0:
        return int(value)
    return None


def _partial_probability(value: object) -> Decimal | None:
    if isinstance(value, Decimal) and Decimal(0) <= value <= Decimal(1):
        return value
    return None


def _clock_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("Jev relevance evaluation latency clock is invalid")
    return float(value)


def _elapsed_ms(started: float, finished: float) -> int:
    if finished < started:
        raise ValueError("Jev relevance evaluation latency cannot be negative")
    return round((finished - started) * 1000)
