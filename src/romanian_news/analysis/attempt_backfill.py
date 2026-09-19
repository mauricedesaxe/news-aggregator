import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from romanian_news.analysis.attempts import ModelAttempt, model_attempt_from_payload, response_cost
from romanian_news.catalog.model_calls import (
    HistoricalModelOutput,
    ModelAttemptRecord,
    ModelCallRegistration,
    read_existing_model_attempt_ids,
    read_historical_model_outputs,
    register_historical_model_calls,
)
from romanian_news.storage import read_verified_r2_object


@dataclass(frozen=True)
class _HistoricalDisposition:
    status: Literal["accepted", "rejected"]
    error: str | None
    use_call_latency: bool


@dataclass(frozen=True)
class _BackfillLedger:
    attempt_ids: frozenset[str]
    attempts: tuple[ModelAttempt, ...]
    calls: tuple[ModelCallRegistration, ...]


_ACCEPTED = _HistoricalDisposition(status="accepted", error=None, use_call_latency=True)
_REJECTED = _HistoricalDisposition(
    status="rejected",
    error="Historical correction response inferred as rejected",
    use_call_latency=False,
)
_OPERATION_KEYS = {
    "news_relevance": "news.relevance",
    "news_embedding": "news.embed",
    "news_summary": "news.summarize_group",
    "news_sentiment": "news.score_group_sentiment",
}


def register_existing_model_calls() -> int:
    """Backfill immutable call and attempt ledgers from every model output version."""
    outputs = read_historical_model_outputs()
    ledger = _BackfillLedger(
        attempt_ids=read_existing_model_attempt_ids(),
        attempts=(),
        calls=(),
    )
    for output in outputs:
        ledger = _backfill_version(output, ledger)
    return register_historical_model_calls(
        tuple(
            ModelAttemptRecord.model_validate(attempt.model_dump(), strict=True)
            for attempt in ledger.attempts
        ),
        ledger.calls,
    )


def _backfill_version(output: HistoricalModelOutput, ledger: _BackfillLedger) -> _BackfillLedger:
    payload = json.loads(read_verified_r2_object(output.r2_key, output.content_digest))
    call = payload["call"]
    responses = tuple(
        response
        for response in (payload.get("provider_responses") or [payload.get("provider_response")])
        if response is not None
    )
    operation_key = _OPERATION_KEYS[output.kind]
    attempts = _historical_attempts(payload, call, responses, operation_key, output.created_at)
    new_attempts = tuple(
        attempt for attempt in attempts if attempt.attempt_id not in ledger.attempt_ids
    )
    calls = ledger.calls
    if not output.call_registered:
        calls += (_model_call_registration(output, call, responses, operation_key),)
    return _BackfillLedger(
        attempt_ids=ledger.attempt_ids | frozenset(attempt.attempt_id for attempt in new_attempts),
        attempts=ledger.attempts + new_attempts,
        calls=calls,
    )


def _historical_attempts(
    payload: Mapping[str, object],
    call: Mapping[str, object],
    responses: tuple[Mapping[str, object], ...],
    operation_key: str,
    observed_at: datetime,
) -> tuple[ModelAttempt, ...]:
    dispositions = (_REJECTED,) * max(len(responses) - 1, 0) + (_ACCEPTED,) * min(len(responses), 1)
    raw_latency = call.get("latency_ms")
    latency_ms = int(raw_latency) if isinstance(raw_latency, int | float) else 0
    return tuple(
        model_attempt_from_payload(
            response,
            request_id=str(payload["request_id"]),
            operation_key=operation_key,
            attempt_index=attempt_index,
            latency_ms=latency_ms if disposition.use_call_latency else 0,
            status=disposition.status,
            error=disposition.error,
            observed_at=observed_at,
        )
        for attempt_index, (response, disposition) in enumerate(
            zip(responses, dispositions, strict=True)
        )
    )


def _model_call_registration(
    output: HistoricalModelOutput,
    call: Mapping[str, object],
    responses: tuple[Mapping[str, object], ...],
    operation_key: str,
) -> ModelCallRegistration:
    return ModelCallRegistration(
        artifact_version_id=output.version_id,
        operation_key=operation_key,
        model=str(call["model"]),
        input_tokens=_required_int(call["input_tokens"]),
        output_tokens=_required_int(call["output_tokens"]),
        cost_usd=sum(response_cost(response) for response in responses),
        latency_ms=_required_int(call["latency_ms"]),
        response_count=len(responses),
    )


def _required_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"Expected an integer in the recorded model call, received {value!r}")
