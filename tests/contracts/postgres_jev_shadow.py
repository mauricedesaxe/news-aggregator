from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from psycopg.errors import IntegrityConstraintViolation

from romanian_news.catalog.jev_shadow import (
    JevShadowClaim,
    JevShadowReceipt,
    claim_jev_shadow,
    read_jev_shadow_fallback_reason,
    record_jev_shadow_receipt,
)
from romanian_news.catalog_transport import ResearchCatalogError, catalog_integrity_identity
from tests.postgres_catalog import postgres_catalog_fixture

postgres_catalog = postgres_catalog_fixture("jev_shadow")
NOW = datetime(2026, 9, 25, tzinfo=UTC)
ARTICLE_VERSION = "1" * 64
INCUMBENT_VERSION = "2" * 64


def _seed_versions(catalog) -> None:
    for artifact_id, kind, version_id in (
        ("article", "news_article", ARTICLE_VERSION),
        ("incumbent", "relevance_result", INCUMBENT_VERSION),
    ):
        catalog.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
            "visibility, created_at) VALUES (%s, %s, 'Test', 'source', 'current', 'private', %s)",
            (artifact_id, kind, NOW),
        )
        catalog.execute(
            "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
            "created_at) VALUES (%s, %s, 1, %s, %s)",
            (version_id, artifact_id, "a" * 64, NOW),
        )


def _claim() -> JevShadowClaim:
    return JevShadowClaim(
        shadow_id="b" * 64,
        article_version_id=ARTICLE_VERSION,
        incumbent_version_id=INCUMBENT_VERSION,
        article_content_digest="a" * 64,
        incumbent_content_digest="a" * 64,
        state_digest="c" * 64,
        question_digest="d" * 64,
        execution_policy_digest="e" * 64,
        incumbent_request_id="f" * 64,
        incumbent_policy_digest="0" * 64,
        incumbent_accepted=True,
        state_length=70_000,
        execution_ref="production-shadow-v1",
        claimed_at=NOW,
    )


def test_concurrent_claims_have_one_owner_and_receipt_is_immutable(postgres_catalog) -> None:
    _seed_versions(postgres_catalog)
    claim = _claim()
    with ThreadPoolExecutor(max_workers=2) as executor:
        owners = tuple(executor.map(claim_jev_shadow, (claim, claim)))
    assert sorted(owners) == [False, True]

    receipt = JevShadowReceipt(
        shadow_id=claim.shadow_id,
        fallback_reason="over_guard",
        jev_request_id=None,
        provider_request_id=None,
        model=None,
        probability=None,
        jev_accepted=None,
        attempts=(),
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=Decimal(0),
        latency_ms=0,
        accounting_complete=True,
        completed_at=NOW,
    )
    record_jev_shadow_receipt(receipt)
    assert read_jev_shadow_fallback_reason(claim.shadow_id) == "over_guard"
    with pytest.raises(ResearchCatalogError) as error:
        record_jev_shadow_receipt(receipt)
    assert (
        catalog_integrity_identity(error.value)
        == "UniqueViolation:23505:news_jev_shadow_receipts_pkey"
    )
    with pytest.raises(IntegrityConstraintViolation):
        postgres_catalog.execute(
            "UPDATE news_jev_shadow_receipts SET fallback_reason = 'provider_failed' "
            "WHERE shadow_id = %s",
            (claim.shadow_id,),
        )
    with pytest.raises(IntegrityConstraintViolation):
        postgres_catalog.execute(
            "DELETE FROM news_jev_shadow_claims WHERE shadow_id = %s", (claim.shadow_id,)
        )
