from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.binary_evaluation import (
    BinaryAttemptEvidence,
    BinaryRequest,
    binary_attempt_totals,
    build_relevance_binary_request,
)
from romanian_news.analysis.jev_relevance import (
    JEV_EXECUTION_POLICY,
    evaluate_jev_relevance,
    jev_execution_policy_digest,
    jev_relevance_request_id,
)
from romanian_news.analysis.relevance import (
    ArticleAnalysisReference,
    load_article_analysis_input,
)
from romanian_news.analysis.relevance_v3 import production_relevance_v3_request_id
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.jev_shadow import (
    JevShadowClaim,
    JevShadowReceipt,
    claim_jev_shadow,
    read_jev_shadow_fallback_reason,
    record_jev_shadow_receipt,
)
from romanian_news.config import NEWS_JEV_RELEVANCE_SHADOW_ENABLED, TYPESAFE_API_KEY
from romanian_news.daily import read_daily_article_references, read_daily_relevance_references
from romanian_news.identity import canonical_json, sha256
from romanian_news.storage import read_verified_r2_object

JEV_SHADOW_EXECUTION_REF = "production-shadow-v1"
JEV_SHADOW_STATE_LIMIT = 70_000
FallbackReason = Literal["none", "over_guard", "provider_rejected", "provider_failed"]


class JevShadowBatchResult(NewsModel):
    day: date
    enabled: bool
    article_count: Annotated[int, Field(ge=0)]
    missing_incumbent: Annotated[int, Field(ge=0)]
    paired: Annotated[int, Field(ge=0)]
    over_guard: Annotated[int, Field(ge=0)]
    failed: Annotated[int, Field(ge=0)]
    already_claimed: Annotated[int, Field(ge=0)]
    unresolved: Annotated[int, Field(ge=0)]


def materialize_jev_relevance_shadow(day: date) -> JevShadowBatchResult:
    if not NEWS_JEV_RELEVANCE_SHADOW_ENABLED:
        return JevShadowBatchResult(
            day=day,
            enabled=False,
            article_count=0,
            missing_incumbent=0,
            paired=0,
            over_guard=0,
            failed=0,
            already_claimed=0,
            unresolved=0,
        )
    if not TYPESAFE_API_KEY:
        raise ValueError("TYPESAFE_API_KEY is required when Jev relevance shadowing is enabled")
    relevance = {value.artifact_id: value for value in read_daily_relevance_references(day).values}
    paired = over_guard = failed = already_claimed = unresolved = missing_incumbent = 0
    articles = read_daily_article_references(day).values
    for article in articles:
        expected_id = f"news:relevance:{production_relevance_v3_request_id(article)}"
        incumbent = relevance.get(expected_id)
        if incumbent is None:
            missing_incumbent += 1
            continue
        outcome = _shadow_one(day, article, incumbent)
        if outcome == "paired":
            paired += 1
        elif outcome == "over_guard":
            over_guard += 1
        elif outcome == "failed":
            failed += 1
        elif outcome == "unresolved":
            unresolved += 1
        else:
            already_claimed += 1
    return JevShadowBatchResult(
        day=day,
        enabled=True,
        article_count=len(articles),
        missing_incumbent=missing_incumbent,
        paired=paired,
        over_guard=over_guard,
        failed=failed,
        already_claimed=already_claimed,
        unresolved=unresolved,
    )


def _shadow_one(day: date, article: ArtifactReference, incumbent: ArtifactReference) -> str:
    analysis_input = load_article_analysis_input(
        ArticleAnalysisReference(reference=article, bucharest_day=day)
    )
    request = build_relevance_binary_request(analysis_input)
    incumbent_request_id = production_relevance_v3_request_id(article)
    incumbent_accepted, incumbent_policy_digest = _incumbent_decision(
        incumbent, article.version_id, incumbent_request_id
    )
    claim = _claim(
        article,
        incumbent,
        request,
        incumbent_request_id,
        incumbent_policy_digest,
        incumbent_accepted,
    )
    if not claim_jev_shadow(claim):
        recorded = read_jev_shadow_fallback_reason(claim.shadow_id)
        if recorded is None:
            return "unresolved"
        if recorded == "over_guard":
            return "over_guard"
        if recorded != "none":
            return "failed"
        return "already_claimed"
    if len(request.state) > JEV_SHADOW_STATE_LIMIT:
        record_jev_shadow_receipt(_receipt(claim.shadow_id, "over_guard", ()))
        return "over_guard"
    attempts: list[BinaryAttemptEvidence] = []
    try:
        observation = evaluate_jev_relevance(
            request,
            execution_ref=JEV_SHADOW_EXECUTION_REF,
            on_attempt=attempts.append,
        )
    except (RuntimeError, ValueError):
        if not attempts:
            raise
        reason = _failure_reason(tuple(attempts))
        record_jev_shadow_receipt(
            _receipt(
                claim.shadow_id,
                reason,
                tuple(attempts),
                jev_request_id=jev_relevance_request_id(request, JEV_SHADOW_EXECUTION_REF),
            )
        )
        return "failed"
    record_jev_shadow_receipt(
        _receipt(
            claim.shadow_id,
            "none",
            observation.attempts,
            jev_request_id=observation.request_id,
            provider_request_id=observation.provider_request_id,
            model=observation.model,
            probability=observation.probability,
            jev_accepted=observation.predicted_accepted,
        )
    )
    return "paired"


def _incumbent_decision(
    reference: ArtifactReference, article_version_id: Sha256, request_id: Sha256
) -> tuple[bool, Sha256]:
    payload = json.loads(read_verified_r2_object(reference.r2_key, reference.content_digest))
    if not isinstance(payload, dict):
        raise ValueError("Incumbent relevance payload is invalid")
    if (
        payload.get("article_version_id") != article_version_id
        or payload.get("request_id") != request_id
        or payload.get("mode") != "production_early_exit"
        or not isinstance(payload.get("accepted"), bool)
    ):
        raise ValueError("Incumbent relevance identity does not match its article")
    policy_digest = payload.get("policy_digest")
    if not isinstance(policy_digest, str) or len(policy_digest) != 64:
        raise ValueError("Incumbent relevance policy digest is missing")
    return payload["accepted"], policy_digest


def _claim(
    article: ArtifactReference,
    incumbent: ArtifactReference,
    request: BinaryRequest,
    incumbent_request_id: Sha256,
    incumbent_policy_digest: Sha256,
    incumbent_accepted: bool,
) -> JevShadowClaim:
    policy_digest = jev_execution_policy_digest(JEV_EXECUTION_POLICY)
    identity = {
        "article_version_id": article.version_id,
        "incumbent_version_id": incumbent.version_id,
        "state_digest": request.state_digest,
        "question_digest": request.question.semantic_digest,
        "execution_policy_digest": policy_digest,
        "execution_ref": JEV_SHADOW_EXECUTION_REF,
    }
    return JevShadowClaim(
        shadow_id=sha256(canonical_json(identity)),
        article_version_id=article.version_id,
        incumbent_version_id=incumbent.version_id,
        article_content_digest=article.content_digest,
        incumbent_content_digest=incumbent.content_digest,
        state_digest=request.state_digest,
        question_digest=request.question.semantic_digest,
        execution_policy_digest=policy_digest,
        incumbent_request_id=incumbent_request_id,
        incumbent_policy_digest=incumbent_policy_digest,
        incumbent_accepted=incumbent_accepted,
        state_length=len(request.state),
        execution_ref=JEV_SHADOW_EXECUTION_REF,
        claimed_at=datetime.now(UTC),
    )


def _failure_reason(
    attempts: tuple[BinaryAttemptEvidence, ...],
) -> FallbackReason:
    if attempts and attempts[-1].http_status is not None:
        status = attempts[-1].http_status
        if 400 <= status < 500 and status not in (408, 429):
            return "provider_rejected"
    return "provider_failed"


def _receipt(
    shadow_id: Sha256,
    fallback_reason: FallbackReason,
    attempts: tuple[BinaryAttemptEvidence, ...],
    *,
    jev_request_id: Sha256 | None = None,
    provider_request_id: str | None = None,
    model: str | None = None,
    probability: Decimal | None = None,
    jev_accepted: bool | None = None,
) -> JevShadowReceipt:
    input_tokens, output_tokens, cost, latency = (
        binary_attempt_totals(attempts) if attempts else (0, 0, Decimal(0), 0)
    )
    final = attempts[-1] if attempts else None
    return JevShadowReceipt(
        shadow_id=shadow_id,
        fallback_reason=fallback_reason,
        jev_request_id=jev_request_id,
        provider_request_id=provider_request_id or (final.provider_request_id if final else None),
        model=model or (final.actual_model if final else None),
        probability=probability,
        jev_accepted=jev_accepted,
        attempts=attempts,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
        latency_ms=latency,
        accounting_complete=fallback_reason == "over_guard"
        or bool(attempts)
        and all(
            attempt.input_tokens is not None
            and attempt.output_tokens is not None
            and attempt.cost_usd is not None
            for attempt in attempts
        ),
        completed_at=datetime.now(UTC),
    )
