from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from psycopg.types.json import Jsonb
from pydantic import Field

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.binary_evaluation import BinaryAttemptEvidence
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog_transport import catalog_mutation, catalog_query


class JevShadowClaim(NewsModel):
    shadow_id: Sha256
    article_version_id: Sha256
    incumbent_version_id: Sha256
    article_content_digest: Sha256
    incumbent_content_digest: Sha256
    state_digest: Sha256
    question_digest: Sha256
    execution_policy_digest: Sha256
    incumbent_request_id: Sha256
    incumbent_policy_digest: Sha256
    incumbent_accepted: bool
    state_length: Annotated[int, Field(gt=0)]
    execution_ref: str
    claimed_at: datetime


class JevShadowReceipt(NewsModel):
    shadow_id: Sha256
    fallback_reason: Literal["none", "over_guard", "provider_rejected", "provider_failed"]
    jev_request_id: Sha256 | None
    provider_request_id: str | None
    model: str | None
    probability: Decimal | None
    jev_accepted: bool | None
    attempts: tuple[BinaryAttemptEvidence, ...]
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: Decimal | None
    latency_ms: Annotated[int, Field(ge=0)]
    accounting_complete: bool
    completed_at: datetime


def claim_jev_shadow(value: JevShadowClaim) -> bool:
    ensure_news_catalog_schema()
    rows = catalog_mutation(
        "INSERT INTO news_jev_shadow_claims "
        "(shadow_id, article_version_id, incumbent_version_id, article_content_digest, "
        "incumbent_content_digest, state_digest, question_digest, execution_policy_digest, "
        "incumbent_request_id, incumbent_policy_digest, incumbent_accepted, state_length, "
        "execution_ref, claimed_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT DO NOTHING RETURNING shadow_id",
        (
            value.shadow_id,
            value.article_version_id,
            value.incumbent_version_id,
            value.article_content_digest,
            value.incumbent_content_digest,
            value.state_digest,
            value.question_digest,
            value.execution_policy_digest,
            value.incumbent_request_id,
            value.incumbent_policy_digest,
            value.incumbent_accepted,
            value.state_length,
            value.execution_ref,
            value.claimed_at,
        ),
    )
    return bool(rows)


def read_jev_shadow_fallback_reason(shadow_id: Sha256) -> str | None:
    rows = catalog_query(
        "SELECT fallback_reason FROM news_jev_shadow_receipts WHERE shadow_id = %s", (shadow_id,)
    )
    return str(rows[0]["fallback_reason"]) if rows else None


def record_jev_shadow_receipt(value: JevShadowReceipt) -> None:
    catalog_mutation(
        "INSERT INTO news_jev_shadow_receipts "
        "(shadow_id, fallback_reason, jev_request_id, provider_request_id, model, probability, "
        "jev_accepted, attempts, input_tokens, output_tokens, estimated_cost_usd, latency_ms, "
        "accounting_complete, completed_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "RETURNING shadow_id",
        (
            value.shadow_id,
            value.fallback_reason,
            value.jev_request_id,
            value.provider_request_id,
            value.model,
            value.probability,
            value.jev_accepted,
            Jsonb([attempt.model_dump(mode="json") for attempt in value.attempts]),
            value.input_tokens,
            value.output_tokens,
            value.estimated_cost_usd,
            value.latency_ms,
            value.accounting_complete,
            value.completed_at,
        ),
    )
