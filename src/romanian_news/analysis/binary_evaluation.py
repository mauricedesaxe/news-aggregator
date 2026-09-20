from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Annotated, Literal, cast

from pydantic import Field, StringConstraints, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.relevance import ArticleAnalysisInput
from romanian_news.analysis.relevance_v3 import relevance_v3_article_text


class BinaryQuestion(NewsModel):
    question_id: Annotated[str, StringConstraints(min_length=1)]
    instructions: Annotated[str, StringConstraints(min_length=1)]
    true_criteria: Annotated[str, StringConstraints(min_length=1)]
    false_criteria: Annotated[str, StringConstraints(min_length=1)]
    threshold: Annotated[Decimal, Field(ge=0, le=1)]

    @property
    def semantic_digest(self) -> Sha256:
        return _sha256(_canonical_json(self.model_dump(mode="json")))


class BinaryRequest(NewsModel):
    question: BinaryQuestion
    state: Annotated[str, StringConstraints(min_length=1)]
    state_digest: Sha256

    @model_validator(mode="after")
    def require_exact_state_digest(self) -> BinaryRequest:
        if self.state_digest != binary_state_digest(self.state):
            raise ValueError("Binary request state digest does not match its rendered state")
        return self


class BinaryAttemptError(NewsModel):
    error_type: Annotated[str, StringConstraints(min_length=1)]
    message: Annotated[str, StringConstraints(min_length=1)]
    retryable: bool


class BinaryAttemptEvidence(NewsModel):
    attempt_number: Annotated[int, Field(gt=0)]
    status: Literal["completed", "retryable_error", "terminal_error"]
    provider_request_id: Annotated[str, StringConstraints(min_length=1)] | None
    actual_model: Annotated[str, StringConstraints(min_length=1)] | None
    http_status: Annotated[int, Field(ge=100, le=599)] | None
    error: BinaryAttemptError | None
    input_tokens: Annotated[int, Field(ge=0)] | None
    output_tokens: Annotated[int, Field(ge=0)] | None
    cost_usd: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)] | None
    latency_ms: Annotated[int, Field(ge=0)]
    probability: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)] | None

    @model_validator(mode="after")
    def require_consistent_status(self) -> BinaryAttemptEvidence:
        if self.status == "completed":
            if self.error is not None:
                raise ValueError("Completed binary attempts cannot contain an error")
            if any(
                value is None
                for value in (
                    self.actual_model,
                    self.input_tokens,
                    self.output_tokens,
                    self.cost_usd,
                    self.probability,
                )
            ):
                raise ValueError("Completed binary attempts require output and accounting evidence")
        elif self.error is None:
            raise ValueError("Failed binary attempts require structured error evidence")
        elif self.status == "retryable_error" and not self.error.retryable:
            raise ValueError("Retryable binary attempt status requires a retryable error")
        return self


BinaryAttemptCallback = Callable[[BinaryAttemptEvidence], None]


class BinaryProbabilityObservation(NewsModel):
    request_id: Sha256
    provider_request_id: Annotated[str, StringConstraints(min_length=1)] | None
    model: Annotated[str, StringConstraints(min_length=1)]
    probability: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
    predicted_accepted: bool
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    latency_ms: Annotated[int, Field(ge=0)]
    estimated_cost_usd: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
    attempts: Annotated[tuple[BinaryAttemptEvidence, ...], Field(min_length=1)] = ()

    @model_validator(mode="before")
    @classmethod
    def populate_legacy_attempt(cls, value: object) -> object:
        if not isinstance(value, Mapping) or "attempts" in value:
            return value
        values = cast(Mapping[str, object], value)
        populated: dict[str, object] = dict(values)
        populated["attempts"] = (
            {
                "attempt_number": 1,
                "status": "completed",
                "provider_request_id": values.get("provider_request_id"),
                "actual_model": values.get("model"),
                "http_status": None,
                "error": None,
                "input_tokens": values.get("input_tokens"),
                "output_tokens": values.get("output_tokens"),
                "cost_usd": values.get("estimated_cost_usd"),
                "latency_ms": values.get("latency_ms"),
                "probability": values.get("probability"),
            },
        )
        return populated

    @model_validator(mode="after")
    def require_matching_completed_attempt(self) -> BinaryProbabilityObservation:
        validate_binary_attempts(self.attempts)
        final = self.attempts[-1]
        if final.status != "completed":
            raise ValueError("Binary observation requires a final completed attempt")
        if (
            final.provider_request_id != self.provider_request_id
            or final.actual_model != self.model
            or final.probability != self.probability
        ):
            raise ValueError("Final completed attempt does not match the binary observation")
        input_tokens, output_tokens, cost_usd, latency_ms = binary_attempt_totals(self.attempts)
        if (
            input_tokens != self.input_tokens
            or output_tokens != self.output_tokens
            or cost_usd != self.estimated_cost_usd
            or latency_ms != self.latency_ms
        ):
            raise ValueError("Binary observation accounting does not match its attempts")
        return self


def validate_binary_attempts(attempts: tuple[BinaryAttemptEvidence, ...]) -> None:
    if tuple(attempt.attempt_number for attempt in attempts) != tuple(range(1, len(attempts) + 1)):
        raise ValueError("Binary attempts must be ordered and contiguous")
    if any(attempt.status != "retryable_error" for attempt in attempts[:-1]):
        raise ValueError("Only retryable errors may precede the final binary attempt")
    if attempts[-1].status == "retryable_error":
        raise ValueError("The final binary attempt must complete or terminate")


def binary_attempt_totals(
    attempts: tuple[BinaryAttemptEvidence, ...],
) -> tuple[int, int, Decimal, int]:
    return (
        sum(attempt.input_tokens or 0 for attempt in attempts),
        sum(attempt.output_tokens or 0 for attempt in attempts),
        sum((attempt.cost_usd or Decimal(0) for attempt in attempts), start=Decimal(0)),
        sum(attempt.latency_ms for attempt in attempts),
    )


RELEVANCE_BINARY_QUESTION = BinaryQuestion(
    question_id="relevant",
    instructions=(
        "Would the Relevance V3 acceptance policy accept this article? Apply the criteria exactly. "
        "Treat a genuinely uncertain context or impact classification as accepted."
    ),
    true_criteria=(
        "Accept an uncertain context classification. For a clear context, the article must be "
        "current, Romania must not be absent, and a Romanian consequence must exist; an incidental "
        "subject must have a direct Romanian consequence. Once that context passes, accept an "
        "uncertain impact classification. If both classifications are clear, a secondary subject "
        "passes only for a direct or attributed actual national-market forecast of routine or major "
        "magnitude with strong political or economic relevance. For a non-secondary subject, accept "
        "a quantified realized foregone public-finance loss of routine or major magnitude. For an "
        "actual consequence, reject an attributed claim that is only committed, then accept any one "
        "of these paths: a principal current subject with hypothetical status or organization scope, "
        "routine or major magnitude, and strong political or economic relevance; a quantified "
        "national-market consequence of routine or major magnitude; a quantified realized "
        "public-finance consequence of routine or major magnitude; or a realized, committed, "
        "proposed, or forecast consequence with sector, broad-population, national-market, "
        "public-finance, or several-systems scope, routine or major magnitude, and strong political "
        "or economic relevance."
    ),
    false_criteria=(
        "The context is clearly outside the true criteria: Romania is absent, the article is a "
        "historical retrospective, no Romanian consequence exists, or an incidental subject lacks "
        "a direct consequence. When both context and impact classifications are clear, a secondary "
        "subject fails unless it meets its exact forecast exception. For other clear classifications, "
        "the effect is a foregone opportunity outside its exact public-finance exception, an "
        "attributed consequence is only committed, or none of the actual-consequence acceptance "
        "paths applies."
    ),
    threshold=Decimal("0.5"),
)


def build_relevance_binary_request(value: ArticleAnalysisInput) -> BinaryRequest:
    state = relevance_v3_article_text(value, 24_000)
    return BinaryRequest(
        question=RELEVANCE_BINARY_QUESTION,
        state=state,
        state_digest=binary_state_digest(state),
    )


def binary_decision(probability: Decimal, threshold: Decimal) -> bool:
    if not Decimal("0") <= probability <= Decimal("1"):
        raise ValueError("probability must be between 0 and 1")
    return probability >= threshold


def binary_state_digest(state: str) -> Sha256:
    return _sha256(state.encode())


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(content: bytes) -> Sha256:
    return hashlib.sha256(content).hexdigest()
