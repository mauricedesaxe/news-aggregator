from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from typing import Annotated, Literal

import requests
from pydantic import Field, StringConstraints

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.relevance import ArticleAnalysisInput
from romanian_news.analysis.relevance_v3 import relevance_v3_article_text
from romanian_news.config import TYPESAFE_API_KEY

JEV_RELEVANCE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_INPUT_COST_PER_MILLION_TOKENS_USD = 0.042


class JevRelevancePolicy(NewsModel):
    policy_id: Annotated[str, StringConstraints(min_length=1)]
    endpoint: Annotated[str, StringConstraints(min_length=1)]
    model: Literal["jev-1.13.0"]
    question_id: Literal["relevant"]
    instructions: Annotated[str, StringConstraints(min_length=1)]
    true_criteria: Annotated[str, StringConstraints(min_length=1)]
    false_criteria: Annotated[str, StringConstraints(min_length=1)]
    input_format: Literal["relevance-v3-article-text-v1"]
    max_body_characters: Annotated[int, Field(gt=0)]
    acceptance_threshold: Annotated[float, Field(ge=0, le=1)]
    timeout_seconds: Annotated[float, Field(gt=0)]
    max_attempts: Annotated[int, Field(gt=0)]
    base_retry_delay_seconds: Annotated[float, Field(ge=0)]
    max_retry_delay_seconds: Annotated[float, Field(ge=0)]
    input_cost_per_million_tokens_usd: Annotated[float, Field(ge=0)]


JEV_RELEVANCE_POLICY = JevRelevancePolicy(
    policy_id="jev-relevance-v3-acceptance-v1",
    endpoint=JEV_RELEVANCE_ENDPOINT,
    model="jev-1.13.0",
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
    input_format="relevance-v3-article-text-v1",
    max_body_characters=24_000,
    acceptance_threshold=0.5,
    timeout_seconds=60,
    max_attempts=4,
    base_retry_delay_seconds=1,
    max_retry_delay_seconds=30,
    input_cost_per_million_tokens_usd=JEV_INPUT_COST_PER_MILLION_TOKENS_USD,
)


class NoulAnswer(NewsModel):
    type: Literal["noul"]
    noul: Annotated[float, Field(ge=0, le=1)]


class JevAnswers(NewsModel):
    relevant: NoulAnswer


class JevUsage(NewsModel):
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]


class JevNoulResponse(NewsModel):
    model: Literal["jev-1.13.0"]
    answers: JevAnswers
    usage: JevUsage


class JevRelevanceObservation(NewsModel):
    request_id: Sha256
    provider_request_id: Annotated[str, StringConstraints(min_length=1)] | None
    model: Annotated[str, StringConstraints(min_length=1)]
    probability: Annotated[float, Field(ge=0, le=1)]
    predicted_accepted: bool
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    latency_ms: Annotated[int, Field(ge=0)]
    estimated_cost_usd: Annotated[float, Field(ge=0)]


def evaluate_jev_relevance(
    value: ArticleAnalysisInput,
    *,
    execution_ref: str,
    policy: JevRelevancePolicy = JEV_RELEVANCE_POLICY,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> JevRelevanceObservation:
    if not execution_ref.strip():
        raise ValueError("Jev relevance evaluation requires an execution reference")
    if not TYPESAFE_API_KEY:
        raise ValueError("TYPESAFE_API_KEY is required for Jev relevance evaluation")
    started = clock()
    headers = {
        "Authorization": f"Bearer {TYPESAFE_API_KEY}",
        "Content-Type": "application/json",
    }
    response: requests.Response | None = None
    for attempt in range(policy.max_attempts):
        candidate: requests.Response | None = None
        try:
            candidate = requests.post(
                policy.endpoint,
                headers=headers,
                json=_request_payload(value, policy),
                timeout=policy.timeout_seconds,
            )
            candidate.raise_for_status()
        except requests.RequestException as error:
            status_code = error.response.status_code if error.response is not None else None
            retryable = status_code is None or status_code in (408, 429) or 500 <= status_code < 600
            if not retryable or attempt + 1 >= policy.max_attempts:
                raise RuntimeError("Jev relevance request failed") from error
            retry_after_seconds = (
                _retry_after_seconds(candidate.headers) if candidate is not None else None
            )
            delay = (
                retry_after_seconds
                if retry_after_seconds is not None
                else policy.base_retry_delay_seconds * 2.0**attempt
            )
            sleep(min(delay, policy.max_retry_delay_seconds))
        else:
            response = candidate
            break
    if response is None:
        raise AssertionError("A valid Jev retry policy always returns or receives a response")
    validated = JevNoulResponse.model_validate_json(response.content, strict=True)
    probability = validated.answers.relevant.noul
    input_tokens = validated.usage.input_tokens
    return JevRelevanceObservation(
        request_id=jev_relevance_request_id(value.reference, execution_ref, policy),
        provider_request_id=response.headers.get("x-typesafe-request-id"),
        model=validated.model,
        probability=probability,
        predicted_accepted=probability >= policy.acceptance_threshold,
        input_tokens=input_tokens,
        output_tokens=validated.usage.output_tokens,
        latency_ms=round((clock() - started) * 1000),
        estimated_cost_usd=input_tokens / 1_000_000 * policy.input_cost_per_million_tokens_usd,
    )


def jev_relevance_policy_digest(
    policy: JevRelevancePolicy = JEV_RELEVANCE_POLICY,
) -> Sha256:
    return _sha256(_canonical_json(policy.model_dump(mode="json")))


def jev_relevance_request_id(
    article: ArtifactReference,
    execution_ref: str,
    policy: JevRelevancePolicy = JEV_RELEVANCE_POLICY,
) -> Sha256:
    if not execution_ref.strip():
        raise ValueError("Jev relevance evaluation requires an execution reference")
    return _sha256(
        _canonical_json(
            {
                "article_version_id": article.version_id,
                "execution_ref": execution_ref,
                "policy_digest": jev_relevance_policy_digest(policy),
                "purpose": "news.relevance.jev.evaluation",
            }
        )
    )


def _request_payload(value: ArticleAnalysisInput, policy: JevRelevancePolicy) -> dict[str, object]:
    return {
        "state": relevance_v3_article_text(value, policy.max_body_characters),
        "model": policy.model,
        "questions": {
            policy.question_id: {
                "type": "noul",
                "instructions": policy.instructions,
                "criteria": {"true": policy.true_criteria, "false": policy.false_criteria},
            }
        },
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(content: bytes) -> Sha256:
    return hashlib.sha256(content).hexdigest()


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
