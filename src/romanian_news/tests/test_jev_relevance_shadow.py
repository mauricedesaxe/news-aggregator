from __future__ import annotations

import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from romanian_news import jev_relevance_shadow as shadow
from romanian_news import storage
from romanian_news.analysis.attempts import ModelCall
from romanian_news.analysis.binary_evaluation import (
    RELEVANCE_BINARY_QUESTION,
    BinaryAttemptError,
    BinaryAttemptEvidence,
    BinaryRequest,
    binary_state_digest,
)
from romanian_news.analysis.jev_relevance import JevRelevanceObservation
from romanian_news.analysis.relevance_v3 import (
    RELEVANCE_V3_POLICY,
    ContextDecision,
    ContextGateResult,
    GateCall,
    ImpactDecision,
    ImpactGateResult,
    RelevanceV3Output,
    production_relevance_v3_request_id,
    relevance_v3_policy_digest,
    relevance_v3_policy_payload,
)
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.jev_shadow import JevShadowReceipt
from romanian_news.identity import canonical_json, sha256

sys.path.insert(0, str(Path(__file__).parents[3]))

from tests.worker.conftest import FakeR2Client

DAY = date(2026, 9, 25)
ARTICLE = ArtifactReference(
    artifact_id="news:article:test",
    version_id="a" * 64,
    content_digest="b" * 64,
    r2_key="news/article.json",
)
INCUMBENT = ArtifactReference(
    artifact_id="news:relevance:test",
    version_id="c" * 64,
    content_digest="d" * 64,
    r2_key="news/relevance.json",
)
_TAMPERED_FIELD = object()


def _request(length: int) -> BinaryRequest:
    state = "x" * length
    return BinaryRequest(
        question=RELEVANCE_BINARY_QUESTION,
        state=state,
        state_digest=binary_state_digest(state),
    )


def _prepare(
    monkeypatch: pytest.MonkeyPatch, length: int, receipts: list[JevShadowReceipt]
) -> None:
    monkeypatch.setattr(shadow, "load_article_analysis_input", lambda *_args: object())
    monkeypatch.setattr(shadow, "build_relevance_binary_request", lambda *_args: _request(length))
    monkeypatch.setattr(shadow, "_incumbent_decision", lambda *_args: (True, "e" * 64))
    monkeypatch.setattr(shadow, "claim_jev_shadow", lambda *_args: True)
    monkeypatch.setattr(shadow, "record_jev_shadow_receipt", receipts.append)


def test_disabled_shadow_never_reads_or_calls_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shadow, "NEWS_JEV_RELEVANCE_SHADOW_ENABLED", False)
    monkeypatch.setattr(
        shadow,
        "read_daily_relevance_references",
        lambda *_args: pytest.fail("disabled shadow read production data"),
    )
    result = shadow.materialize_jev_relevance_shadow(DAY)
    assert not result.enabled
    assert result.paired == result.failed == 0


def test_enabled_shadow_without_typesafe_api_key_fails_before_any_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shadow, "NEWS_JEV_RELEVANCE_SHADOW_ENABLED", True)
    monkeypatch.setattr(shadow, "TYPESAFE_API_KEY", None)
    monkeypatch.setattr(
        shadow,
        "read_daily_relevance_references",
        lambda *_args: pytest.fail("missing key must fail before reading production data"),
    )

    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        shadow.materialize_jev_relevance_shadow(DAY)


def test_enabled_shadow_counts_articles_without_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shadow, "NEWS_JEV_RELEVANCE_SHADOW_ENABLED", True)
    monkeypatch.setattr(shadow, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(
        shadow,
        "read_daily_article_references",
        lambda *_args: SimpleNamespace(values=(ARTICLE,)),
    )
    monkeypatch.setattr(
        shadow,
        "read_daily_relevance_references",
        lambda *_args: SimpleNamespace(values=()),
    )
    result = shadow.materialize_jev_relevance_shadow(DAY)
    assert result.article_count == 1
    assert result.missing_incumbent == 1
    assert result.paired == 0


@pytest.mark.parametrize("length,expected", [(70_000, "paired"), (70_001, "over_guard")])
def test_shadow_preflight_guard_sends_only_states_within_the_limit(
    monkeypatch: pytest.MonkeyPatch, length: int, expected: str
) -> None:
    receipts: list[JevShadowReceipt] = []
    _prepare(monkeypatch, length, receipts)
    if expected == "over_guard":
        monkeypatch.setattr(
            shadow,
            "evaluate_jev_relevance",
            lambda *_args: pytest.fail("over-guard state must not reach the provider"),
        )
    else:
        monkeypatch.setattr(shadow, "evaluate_jev_relevance", _successful_evaluate)

    assert shadow._shadow_one(DAY, ARTICLE, INCUMBENT) == expected

    assert receipts[0].fallback_reason == ("none" if expected == "paired" else "over_guard")
    assert receipts[0].accounting_complete


def _successful_evaluate(
    _request: BinaryRequest, *, execution_ref: str, on_attempt
) -> JevRelevanceObservation:
    attempt = BinaryAttemptEvidence(
        attempt_number=1,
        status="completed",
        provider_request_id="provider-1",
        actual_model="jev-1.13.0",
        http_status=200,
        error=None,
        input_tokens=10,
        output_tokens=2,
        cost_usd=Decimal("0.01"),
        latency_ms=20,
        probability=Decimal("0.9"),
    )
    on_attempt(attempt)
    return JevRelevanceObservation(
        request_id="f" * 64,
        provider_request_id="provider-1",
        model="jev-1.13.0",
        probability=Decimal("0.9"),
        predicted_accepted=True,
        input_tokens=10,
        output_tokens=2,
        latency_ms=20,
        estimated_cost_usd=Decimal("0.01"),
        attempts=(attempt,),
    )


def test_provider_rejection_records_failure_without_changing_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts: list[JevShadowReceipt] = []
    _prepare(monkeypatch, 30, receipts)

    def reject(_request: BinaryRequest, *, execution_ref: str, on_attempt) -> None:
        on_attempt(
            BinaryAttemptEvidence(
                attempt_number=1,
                status="terminal_error",
                provider_request_id="provider-rejected",
                actual_model=None,
                http_status=400,
                error=BinaryAttemptError(
                    error_type="HTTPError", message="Rejected", retryable=False
                ),
                input_tokens=None,
                output_tokens=None,
                cost_usd=None,
                latency_ms=12,
                probability=None,
            )
        )
        raise RuntimeError("Provider rejected request")

    monkeypatch.setattr(shadow, "evaluate_jev_relevance", reject)
    assert shadow._shadow_one(DAY, ARTICLE, INCUMBENT) == "failed"
    assert receipts[0].fallback_reason == "provider_rejected"
    assert receipts[0].jev_accepted is None
    assert not receipts[0].accounting_complete


def incumbent_relevance_output(
    article: ArtifactReference, *, accepted: bool, request_id: str
) -> RelevanceV3Output:
    """Build one incumbent payload with the exact field set analyze_relevance_v3 writes."""
    context = ContextGateResult(
        decision=ContextDecision(
            subject_role="principal" if accepted else "absent",
            news_cycle="current_cycle",
            romanian_consequence="direct" if accepted else "absent",
            certainty="clear",
            evidence_quote="Dovadă",
            reason_ro="Relevant" if accepted else "Incidental",
        ),
        provider=GateCall(
            request_id="c" * 64,
            call=ModelCall(
                response_id="context-response",
                model="google/gemini-2.5-flash",
                input_tokens=12,
                output_tokens=7,
                latency_ms=10,
            ),
            cost_usd=0.003,
            response_count=1,
            traces=(),
            accounting_complete=True,
        ),
    )
    impact = (
        ImpactGateResult(
            decision=ImpactDecision(
                consequence_status="realized",
                effect_basis="actual_consequence",
                effect_scope="broad_population",
                magnitude="major",
                political_relevance="strong",
                economic_relevance="none",
                quantified=True,
                certainty="clear",
                evidence_quote="Dovadă",
                reason_ro="Relevant",
            ),
            provider=GateCall(
                request_id="d" * 64,
                call=ModelCall(
                    response_id="impact-response",
                    model="google/gemini-2.5-flash",
                    input_tokens=18,
                    output_tokens=9,
                    latency_ms=20,
                ),
                cost_usd=0.004,
                response_count=1,
                traces=(),
                accounting_complete=True,
            ),
        )
        if accepted
        else None
    )
    output = RelevanceV3Output.model_construct(
        request_id=request_id,
        policy=RELEVANCE_V3_POLICY,
        mode="production_early_exit",
        execution_ref=None,
        article=article,
        context=context,
        impact=impact,
        accepted=accepted,
        content=b"",
    )
    return RelevanceV3Output.model_construct(
        request_id=request_id,
        policy=RELEVANCE_V3_POLICY,
        mode="production_early_exit",
        execution_ref=None,
        article=article,
        context=context,
        impact=impact,
        accepted=accepted,
        content=canonical_json(
            {
                "accepted": output.accepted,
                "article_version_id": output.article.version_id,
                "context": output.context.model_dump(mode="json"),
                "context_provider_responses": [{"usage": {"cost": 0.003}}],
                "impact": output.impact.model_dump(mode="json") if output.impact else None,
                "impact_provider_responses": (
                    [{"usage": {"cost": 0.004}}] if output.impact else []
                ),
                "mode": output.mode,
                "execution_ref": output.execution_ref,
                "policy": relevance_v3_policy_payload(output.policy),
                "policy_digest": relevance_v3_policy_digest(output.policy),
                "request_id": output.request_id,
            }
        ),
    )


def _store_incumbent(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> ArtifactReference:
    fake_r2 = FakeR2Client()
    monkeypatch.setattr(storage, "_r2_client", lambda: fake_r2)
    fake_r2.put_object(Bucket="news-objects", Key=INCUMBENT.r2_key, Body=payload, Metadata={})
    return INCUMBENT.model_copy(update={"content_digest": sha256(payload)})


def _prepare_incumbent_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shadow, "load_article_analysis_input", lambda *_args: object())
    monkeypatch.setattr(shadow, "build_relevance_binary_request", lambda *_args: _request(30))
    monkeypatch.setattr(
        shadow,
        "claim_jev_shadow",
        lambda *_args: pytest.fail("a rejected incumbent must not claim a shadow row"),
    )
    monkeypatch.setattr(
        shadow,
        "evaluate_jev_relevance",
        lambda *_args: pytest.fail("a rejected incumbent must not issue a Jev request"),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("article_version_id", "f" * 64),
        ("request_id", "f" * 64),
        ("mode", "full_evaluation"),
        ("accepted", "yes"),
        ("policy_digest", _TAMPERED_FIELD),
        ("policy_digest", "f" * 63),
    ),
)
def test_shadow_rejects_a_tampered_incumbent_before_any_jev_request(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    request_id = production_relevance_v3_request_id(ARTICLE)
    output = incumbent_relevance_output(ARTICLE, accepted=True, request_id=request_id)
    payload = json.loads(output.content)
    if value is _TAMPERED_FIELD:
        del payload[field]
    else:
        payload[field] = value
    incumbent = _store_incumbent(monkeypatch, json.dumps(payload).encode())
    _prepare_incumbent_guard(monkeypatch)

    with pytest.raises(ValueError, match="Incumbent relevance (identity|policy digest)"):
        shadow._shadow_one(DAY, ARTICLE, incumbent)


@pytest.mark.parametrize("accepted", [True, False])
def test_incumbent_decision_round_trips_the_published_verdict_and_policy_digest(
    monkeypatch: pytest.MonkeyPatch, accepted: bool
) -> None:
    request_id = production_relevance_v3_request_id(ARTICLE)
    output = incumbent_relevance_output(ARTICLE, accepted=accepted, request_id=request_id)
    incumbent = _store_incumbent(monkeypatch, output.content)

    assert shadow._incumbent_decision(incumbent, ARTICLE.version_id, request_id) == (
        accepted,
        relevance_v3_policy_digest(RELEVANCE_V3_POLICY),
    )
