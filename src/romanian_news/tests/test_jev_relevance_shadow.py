from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest

from romanian_news import jev_relevance_shadow as shadow
from romanian_news.analysis.binary_evaluation import (
    RELEVANCE_BINARY_QUESTION,
    BinaryAttemptError,
    BinaryAttemptEvidence,
    BinaryRequest,
    binary_state_digest,
)
from romanian_news.analysis.jev_relevance import JevRelevanceObservation
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.jev_shadow import JevShadowReceipt

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
def test_shadow_preflight_preserves_exact_state_and_incumbent(
    monkeypatch: pytest.MonkeyPatch, length: int, expected: str
) -> None:
    receipts: list[JevShadowReceipt] = []
    _prepare(monkeypatch, length, receipts)
    requests: list[BinaryRequest] = []
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

    def evaluate(request: BinaryRequest, *, execution_ref: str, on_attempt):
        requests.append(request)
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

    monkeypatch.setattr(shadow, "evaluate_jev_relevance", evaluate)
    assert shadow._shadow_one(DAY, ARTICLE, INCUMBENT) == expected
    assert len(requests) == (1 if length == 70_000 else 0)
    if requests:
        assert len(requests[0].state) == length
    assert receipts[0].fallback_reason == ("none" if requests else "over_guard")
    assert receipts[0].accounting_complete


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
