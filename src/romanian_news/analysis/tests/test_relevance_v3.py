import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import HttpUrl, ValidationError

from romanian_news.analysis import relevance_v3
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.relevance import ArticleAnalysisInput
from romanian_news.analysis.relevance_v3 import (
    CONTEXT_OPERATION_KEY,
    IMPACT_OPERATION_KEY,
    RELEVANCE_V3_POLICY,
    ContextDecision,
    ImpactDecision,
    analyze_relevance_v3,
    context_is_accepted,
    production_relevance_v3_request_id,
    relevance_v3_is_accepted,
    relevance_v3_policy_digest,
    relevance_v3_policy_payload,
    relevance_v3_request_id,
)
from romanian_news.analysis.tracing import ModelTraceReference, ProviderCallResult
from romanian_news.articles.models import ExtractedArticle

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_DEFAULT_USAGE = object()


class _FakeResponse:
    def __init__(
        self, response_id: str, content: str, usage_payload: object = _DEFAULT_USAGE
    ) -> None:
        self.id = response_id
        self.model = "test/model"
        self.usage = SimpleNamespace(prompt_tokens=12, completion_tokens=7)
        self.usage_payload = (
            {"prompt_tokens": 12, "completion_tokens": 7, "cost": 0.004}
            if usage_payload is _DEFAULT_USAGE
            else usage_payload
        )
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        return {
            "id": self.id,
            "model": self.model,
            "created": 1_788_172_800,
            "usage": self.usage_payload,
        }


def test_full_evaluation_runs_independent_gates_with_the_same_article(monkeypatch) -> None:
    responses = iter(
        (
            _FakeResponse("context-response", _context_json()),
            _FakeResponse("impact-response", _impact_json()),
        )
    )
    calls = []
    attempts = []

    def trace(operation, request_id, inputs, call):
        calls.append((operation, request_id, inputs))
        return ProviderCallResult(
            response=call(),
            call_id=UUID(int=len(calls)),
            trace=_trace(len(calls)),
        )

    monkeypatch.setattr(
        relevance_v3,
        "openrouter_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: next(responses))
            )
        ),
    )
    monkeypatch.setattr(relevance_v3, "trace_provider_call", trace)
    monkeypatch.setattr(
        relevance_v3,
        "record_model_attempt",
        lambda *_args, **kwargs: attempts.append(kwargs),
    )

    output = analyze_relevance_v3(
        _analysis_input(), mode="full_evaluation", execution_ref="git:test:v3-run-1"
    )

    assert output.impact is not None
    article_message = calls[0][2]["messages"][1]
    assert {
        "accepted": output.accepted,
        "operations": [call[0] for call in calls],
        "identical_article_input": article_message == calls[1][2]["messages"][1],
        "article_text": article_message["content"],
        "distinct_gate_requests": calls[0][1] != calls[1][1],
        "attempt_operations": [attempt["operation_key"] for attempt in attempts],
        "attempts_traced": all(attempt["trace"] is not None for attempt in attempts),
        "execution_ref": output.execution_ref,
        "context_response_count": output.context.provider.response_count,
        "context_accounting": output.context.provider.accounting_complete,
        "context_traces": [trace.trace_id for trace in output.context.provider.traces],
        "impact_response_count": output.impact.provider.response_count,
        "impact_accounting": output.impact.provider.accounting_complete,
        "impact_traces": [trace.trace_id for trace in output.impact.provider.traces],
        "observability": output.observability_complete,
        "input_tokens": output.input_tokens,
        "output_tokens": output.output_tokens,
        "cost_usd": output.cost_usd,
    } == {
        "accepted": True,
        "operations": [CONTEXT_OPERATION_KEY, IMPACT_OPERATION_KEY],
        "identical_article_input": True,
        "article_text": (
            "Publicat: 2026-08-31T09:00:00+00:00\n"
            "Titlu: Titlu\n\nArticol:\nDovada exacta este aici."
        ),
        "distinct_gate_requests": True,
        "attempt_operations": [CONTEXT_OPERATION_KEY, IMPACT_OPERATION_KEY],
        "attempts_traced": True,
        "execution_ref": "git:test:v3-run-1",
        "context_response_count": 1,
        "context_accounting": True,
        "context_traces": ["trace-1"],
        "impact_response_count": 1,
        "impact_accounting": True,
        "impact_traces": ["trace-2"],
        "observability": True,
        "input_tokens": 24,
        "output_tokens": 14,
        "cost_usd": pytest.approx(0.008),
    }


def test_each_correction_response_gets_attempt_and_trace_accounting(monkeypatch) -> None:
    responses = iter(
        (
            _FakeResponse(
                "context-invalid",
                _context_json(evidence_quote="Text absent"),
            ),
            _FakeResponse("context-corrected", _context_json()),
            _FakeResponse("impact-response", _impact_json()),
        )
    )
    attempts = []
    traces = []

    def trace(operation, _request_id, _inputs, call):
        trace_reference = _trace(len(traces) + 1)
        traces.append(trace_reference)
        return ProviderCallResult(
            response=call(),
            call_id=UUID(int=len(traces)),
            trace=trace_reference,
        )

    monkeypatch.setattr(
        relevance_v3,
        "openrouter_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: next(responses))
            )
        ),
    )
    monkeypatch.setattr(relevance_v3, "trace_provider_call", trace)
    monkeypatch.setattr(
        relevance_v3,
        "record_model_attempt",
        lambda response, **kwargs: attempts.append((response.id, kwargs)),
    )

    output = analyze_relevance_v3(
        _analysis_input(), mode="full_evaluation", execution_ref="git:test:v3-run-1"
    )

    assert [(response_id, item["status"]) for response_id, item in attempts] == [
        ("context-invalid", "rejected"),
        ("context-corrected", "accepted"),
        ("impact-response", "accepted"),
    ]
    assert [item["trace"] for _response_id, item in attempts] == traces
    assert [trace.trace_id for trace in output.context.provider.traces] == [
        "trace-1",
        "trace-2",
    ]
    assert output.impact is not None
    assert [trace.trace_id for trace in output.impact.provider.traces] == ["trace-3"]
    assert output.context.provider.call.input_tokens == 24
    assert output.context.provider.call.output_tokens == 14
    assert output.cost_usd == pytest.approx(0.012)


def test_full_evaluation_preserves_result_when_tracing_is_unavailable(monkeypatch) -> None:
    responses = iter(
        (
            _FakeResponse("context-response", _context_json()),
            _FakeResponse("impact-response", _impact_json()),
        )
    )
    attempts = []
    calls = []
    monkeypatch.setattr(
        relevance_v3,
        "openrouter_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: next(responses))
            )
        ),
    )

    def trace(_operation, _request_id, _inputs, call):
        calls.append("provider")
        return ProviderCallResult(
            response=call(),
            call_id=UUID(int=len(calls)),
            trace=None,
        )

    monkeypatch.setattr(relevance_v3, "trace_provider_call", trace)
    monkeypatch.setattr(
        relevance_v3,
        "record_model_attempt",
        lambda value, **_kwargs: attempts.append(value.id),
    )

    output = analyze_relevance_v3(
        _analysis_input(),
        mode="full_evaluation",
        execution_ref="git:test:v3-run-1",
    )

    assert output.accepted is True
    assert output.observability_complete is False
    assert output.context.provider.response_count == 1
    assert output.context.provider.traces == ()
    assert output.impact is not None
    assert output.impact.provider.response_count == 1
    assert output.impact.provider.traces == ()
    assert attempts == ["context-response", "impact-response"]
    assert calls == ["provider", "provider"]


@pytest.mark.parametrize(
    "usage_payload",
    (
        None,
        {"completion_tokens": 7, "cost": 0.004},
        {"prompt_tokens": 12, "completion_tokens": "7", "cost": 0.004},
        {"prompt_tokens": 12, "completion_tokens": 7},
        {"prompt_tokens": 12, "completion_tokens": 7, "cost": "0.004"},
        {"prompt_tokens": "bad", "completion_tokens": 7, "cost": "bad"},
        {"prompt_tokens": 12, "completion_tokens": 7, "cost": -1},
    ),
)
def test_missing_or_malformed_usage_marks_accounting_incomplete(monkeypatch, usage_payload) -> None:
    calls = []
    responses = (
        _FakeResponse("context-response", _context_json(), usage_payload),
        _FakeResponse("impact-response", _impact_json()),
    )
    _patch_provider(monkeypatch, calls, *responses)
    recorded = []
    monkeypatch.setattr(
        relevance_v3,
        "record_model_attempt",
        lambda response, **_kwargs: recorded.append(response.model_dump(mode="json")),
    )

    output = analyze_relevance_v3(_analysis_input(), mode="production_early_exit")

    assert output.accepted is True
    assert output.context.provider.accounting_complete is False
    assert output.observability_complete is False
    assert output.context.provider.call.input_tokens in (0, 12)
    assert output.context.provider.call.output_tokens in (0, 7)
    stored_usage = recorded[0]["usage"]
    assert isinstance(stored_usage, dict)
    assert isinstance(stored_usage["prompt_tokens"], int)
    assert isinstance(stored_usage["completion_tokens"], int)
    assert isinstance(stored_usage["cost"], float)


def test_production_stops_only_after_a_clear_context_rejection(monkeypatch) -> None:
    calls = []
    response = _FakeResponse(
        "context-response",
        _context_json(news_cycle="historical_retrospective", certainty="clear"),
    )
    _patch_provider(monkeypatch, calls, response)

    output = analyze_relevance_v3(_analysis_input(), mode="production_early_exit")

    assert output.accepted is False
    assert output.context_early_exit is True
    assert output.impact is None
    assert output.context.provider.traces == ()
    assert calls == [CONTEXT_OPERATION_KEY]


def test_uncertain_context_rejection_risk_reaches_impact(monkeypatch) -> None:
    calls = []
    responses = iter(
        (
            _FakeResponse(
                "context-response",
                _context_json(
                    news_cycle="historical_retrospective",
                    romanian_consequence="absent",
                    certainty="uncertain",
                ),
            ),
            _FakeResponse("impact-response", _impact_json(certainty="uncertain")),
        )
    )
    _patch_provider(monkeypatch, calls, *responses)

    output = analyze_relevance_v3(_analysis_input(), mode="production_early_exit")

    assert output.accepted is True
    assert output.context_early_exit is False
    assert output.impact is not None
    assert calls == [CONTEXT_OPERATION_KEY, IMPACT_OPERATION_KEY]


@pytest.mark.parametrize(
    ("changes", "accepted"),
    (
        ({"news_cycle": "historical_retrospective", "certainty": "clear"}, False),
        ({"romanian_consequence": "absent", "certainty": "clear"}, False),
        (
            {
                "news_cycle": "historical_retrospective",
                "romanian_consequence": "absent",
                "certainty": "uncertain",
            },
            True,
        ),
        ({"subject_role": "absent", "romanian_consequence": "direct"}, False),
        ({"subject_role": "incidental", "romanian_consequence": "possible"}, False),
        ({"subject_role": "incidental", "romanian_consequence": "direct"}, True),
    ),
)
def test_context_policy_rejects_only_clear_ineligible_context(changes, accepted) -> None:
    decision = ContextDecision.model_validate(
        {
            "subject_role": "principal",
            "news_cycle": "current_cycle",
            "romanian_consequence": "direct",
            "certainty": "clear",
            "evidence_quote": "Dovada",
            "reason_ro": "Motiv",
            **changes,
        }
    )

    assert context_is_accepted(decision, RELEVANCE_V3_POLICY.acceptance.context) is accepted


@pytest.mark.parametrize(
    ("changes", "accepted"),
    (
        ({"effect_scope": "sector", "economic_relevance": "strong"}, True),
        ({"effect_scope": "organization", "economic_relevance": "strong"}, False),
        ({"consequence_status": "hypothetical", "economic_relevance": "strong"}, False),
        (
            {
                "consequence_status": "realized",
                "effect_scope": "public_finances",
                "magnitude": "routine",
                "quantified": True,
            },
            True,
        ),
        (
            {
                "consequence_status": "realized",
                "effect_scope": "public_finances",
                "magnitude": "narrow",
                "quantified": True,
            },
            False,
        ),
        ({"certainty": "uncertain"}, True),
    ),
)
def test_impact_policy_is_recall_first_without_article_special_cases(changes, accepted) -> None:
    decision = ImpactDecision.model_validate(
        {
            "consequence_status": "realized",
            "effect_basis": "actual_consequence",
            "effect_scope": "individual",
            "magnitude": "routine",
            "political_relevance": "none",
            "economic_relevance": "none",
            "quantified": False,
            "certainty": "clear",
            "evidence_quote": "Dovada",
            "reason_ro": "Motiv",
            **changes,
        }
    )

    context = ContextDecision(
        subject_role="incidental",
        news_cycle="current_cycle",
        romanian_consequence="direct",
        certainty="clear",
        evidence_quote="Dovada",
        reason_ro="Motiv",
    )

    assert relevance_v3_is_accepted(context, decision, RELEVANCE_V3_POLICY.acceptance) is accepted


@pytest.mark.parametrize(
    ("context_changes", "impact_changes", "accepted"),
    (
        ({}, {"effect_scope": "local_public_finances"}, False),
        ({}, {"effect_basis": "foregone_opportunity"}, False),
        (
            {},
            {
                "consequence_status": "hypothetical",
                "effect_scope": "individual",
                "economic_relevance": "strong",
            },
            True,
        ),
        (
            {},
            {
                "effect_scope": "organization",
                "political_relevance": "strong",
                "economic_relevance": "none",
            },
            True,
        ),
        (
            {"subject_role": "secondary", "romanian_consequence": "possible"},
            {},
            False,
        ),
        (
            {"romanian_consequence": "quoted"},
            {"consequence_status": "committed"},
            False,
        ),
    ),
)
def test_combined_policy_applies_attempt_two_distinctions(
    context_changes, impact_changes, accepted
) -> None:
    context = ContextDecision.model_validate(
        {
            "subject_role": "principal",
            "news_cycle": "current_cycle",
            "romanian_consequence": "direct",
            "certainty": "clear",
            "evidence_quote": "Dovada",
            "reason_ro": "Motiv",
            **context_changes,
        }
    )
    impact = ImpactDecision.model_validate(
        {
            "consequence_status": "realized",
            "effect_basis": "actual_consequence",
            "effect_scope": "sector",
            "magnitude": "routine",
            "political_relevance": "none",
            "economic_relevance": "strong",
            "quantified": False,
            "certainty": "clear",
            "evidence_quote": "Dovada",
            "reason_ro": "Motiv",
            **impact_changes,
        }
    )

    assert relevance_v3_is_accepted(context, impact, RELEVANCE_V3_POLICY.acceptance) is accepted


@pytest.mark.parametrize(
    "impact_changes",
    (
        {
            "consequence_status": "hypothetical",
            "magnitude": "narrow",
            "political_relevance": "none",
            "economic_relevance": "none",
        },
        {
            "effect_scope": "organization",
            "magnitude": "narrow",
            "political_relevance": "none",
            "economic_relevance": "none",
        },
    ),
)
def test_principal_current_exceptions_still_require_materiality(impact_changes) -> None:
    assert (
        relevance_v3_is_accepted(
            _context_decision(),
            _impact_decision(**impact_changes),
            RELEVANCE_V3_POLICY.acceptance,
        )
        is False
    )


def test_attempt_three_rejects_secondary_except_qualified_market_forecasts() -> None:
    ordinary = _impact_decision()
    assert (
        relevance_v3_is_accepted(
            _context_decision(subject_role="secondary"),
            ordinary,
            RELEVANCE_V3_POLICY.acceptance,
        )
        is False
    )

    qualified = _impact_decision(
        consequence_status="forecast",
        effect_scope="national_market",
    )
    for consequence in ("direct", "quoted"):
        for magnitude in ("routine", "major"):
            assert (
                relevance_v3_is_accepted(
                    _context_decision(
                        subject_role="secondary",
                        romanian_consequence=consequence,
                    ),
                    qualified.model_copy(update={"magnitude": magnitude}),
                    RELEVANCE_V3_POLICY.acceptance,
                )
                is True
            )

    for changes in (
        {"effect_basis": "foregone_opportunity"},
        {"effect_scope": "sector"},
        {"magnitude": "narrow"},
        {"economic_relevance": "weak"},
    ):
        assert (
            relevance_v3_is_accepted(
                _context_decision(subject_role="secondary"),
                _impact_decision(
                    **{
                        "consequence_status": "forecast",
                        "effect_scope": "national_market",
                        **changes,
                    }
                ),
                RELEVANCE_V3_POLICY.acceptance,
            )
            is False
        )


def test_attempt_three_accepts_foregone_quantified_realized_public_finance_loss() -> None:
    loss = _impact_decision(
        consequence_status="realized",
        effect_basis="foregone_opportunity",
        effect_scope="public_finances",
        economic_relevance="weak",
        quantified=True,
    )

    for magnitude in ("routine", "major"):
        assert (
            relevance_v3_is_accepted(
                _context_decision(),
                loss.model_copy(update={"magnitude": magnitude}),
                RELEVANCE_V3_POLICY.acceptance,
            )
            is True
        )
    assert (
        relevance_v3_is_accepted(
            _context_decision(),
            loss.model_copy(update={"quantified": False}),
            RELEVANCE_V3_POLICY.acceptance,
        )
        is False
    )


def test_attempt_three_accepts_quantified_actual_national_market_consequences() -> None:
    market_change = _impact_decision(
        effect_scope="national_market",
        economic_relevance="weak",
        quantified=True,
    )

    for magnitude in ("routine", "major"):
        assert (
            relevance_v3_is_accepted(
                _context_decision(),
                market_change.model_copy(update={"magnitude": magnitude}),
                RELEVANCE_V3_POLICY.acceptance,
            )
            is True
        )
    assert (
        relevance_v3_is_accepted(
            _context_decision(),
            market_change.model_copy(update={"quantified": False}),
            RELEVANCE_V3_POLICY.acceptance,
        )
        is False
    )


def test_policy_requires_identical_article_input_for_both_gates() -> None:
    with pytest.raises(ValidationError, match="same article input"):
        RELEVANCE_V3_POLICY.model_copy(
            update={
                "impact": RELEVANCE_V3_POLICY.impact.model_copy(
                    update={"max_body_characters": 12000}
                )
            }
        ).model_validate(
            {
                **RELEVANCE_V3_POLICY.model_dump(),
                "impact": RELEVANCE_V3_POLICY.impact.model_copy(
                    update={"max_body_characters": 12000}
                ).model_dump(),
            }
        )


def test_policy_and_request_identity_cover_schemas_acceptance_and_mode() -> None:
    payload = relevance_v3_policy_payload(RELEVANCE_V3_POLICY)
    reference = _reference()

    assert payload["acceptance"] == RELEVANCE_V3_POLICY.acceptance.model_dump(mode="json")
    assert RELEVANCE_V3_POLICY.acceptance.acceptance_algorithm_id == (
        "recall-first-combined-v3-attempt-3-material-v3"
    )
    changed_algorithm = RELEVANCE_V3_POLICY.model_copy(
        update={
            "acceptance": RELEVANCE_V3_POLICY.acceptance.model_copy(
                update={"acceptance_algorithm_id": "different-algorithm"}
            )
        }
    )
    assert relevance_v3_policy_digest(changed_algorithm) != relevance_v3_policy_digest()
    assert relevance_v3_policy_digest() != relevance_v3_policy_digest(
        RELEVANCE_V3_POLICY.model_copy(
            update={
                "acceptance": RELEVANCE_V3_POLICY.acceptance.model_copy(
                    update={
                        "impact": RELEVANCE_V3_POLICY.acceptance.impact.model_copy(
                            update={"uncertain_passes": False}
                        )
                    }
                )
            }
        )
    )
    assert relevance_v3_request_id(
        reference, mode="full_evaluation", execution_ref="git:test:v3-run-1"
    ) != relevance_v3_request_id(reference, mode="production_early_exit")
    assert relevance_v3_request_id(
        reference, mode="full_evaluation", execution_ref="git:test:v3-run-1"
    ) != relevance_v3_request_id(
        reference, mode="full_evaluation", execution_ref="git:test:v3-run-2"
    )
    assert relevance_v3_request_id(
        reference, mode="production_early_exit"
    ) == relevance_v3_request_id(reference, mode="production_early_exit")
    assert production_relevance_v3_request_id(reference) == relevance_v3_request_id(
        reference, mode="production_early_exit"
    )
    changed_combined_policy = RELEVANCE_V3_POLICY.model_copy(
        update={
            "acceptance": RELEVANCE_V3_POLICY.acceptance.model_copy(
                update={
                    "combined": RELEVANCE_V3_POLICY.acceptance.combined.model_copy(
                        update={"default_rejected_subject_roles": ()}
                    )
                }
            )
        }
    )
    assert relevance_v3_policy_digest(changed_combined_policy) != relevance_v3_policy_digest()


def test_full_evaluation_requires_an_execution_reference() -> None:
    with pytest.raises(ValueError, match="execution reference"):
        analyze_relevance_v3(_analysis_input(), mode="full_evaluation")


def _context_decision(**changes) -> ContextDecision:
    return ContextDecision.model_validate(
        {
            "subject_role": "principal",
            "news_cycle": "current_cycle",
            "romanian_consequence": "direct",
            "certainty": "clear",
            "evidence_quote": "Dovada",
            "reason_ro": "Motiv",
            **changes,
        }
    )


def _impact_decision(**changes) -> ImpactDecision:
    return ImpactDecision.model_validate(
        {
            "consequence_status": "realized",
            "effect_basis": "actual_consequence",
            "effect_scope": "sector",
            "magnitude": "routine",
            "political_relevance": "none",
            "economic_relevance": "strong",
            "quantified": False,
            "certainty": "clear",
            "evidence_quote": "Dovada",
            "reason_ro": "Motiv",
            **changes,
        }
    )


def _trace(position: int) -> ModelTraceReference:
    return ModelTraceReference(
        provider="langfuse",
        trace_id=f"trace-{position}",
        observation_id=f"observation-{position}",
        project_ref="project-test",
        recorded_at=datetime(2026, 8, 31, 9, tzinfo=UTC),
    )


def _patch_provider(monkeypatch, calls, *responses) -> None:
    values = iter(responses)

    def trace(operation, _request_id, _inputs, call):
        calls.append(operation)
        return ProviderCallResult(response=call(), call_id=UUID(int=len(calls)), trace=None)

    monkeypatch.setattr(
        relevance_v3,
        "openrouter_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: next(values)))
        ),
    )
    monkeypatch.setattr(relevance_v3, "trace_provider_call", trace)
    monkeypatch.setattr(relevance_v3, "record_model_attempt", lambda *_args, **_kwargs: None)


def _context_json(**changes) -> str:
    return json.dumps(
        {
            "subject_role": "principal",
            "news_cycle": "current_cycle",
            "romanian_consequence": "direct",
            "certainty": "clear",
            "evidence_quote": "Dovada exacta",
            "reason_ro": "Consecință românească directă.",
            **changes,
        }
    )


def _impact_json(**changes) -> str:
    return json.dumps(
        {
            "consequence_status": "realized",
            "effect_basis": "actual_consequence",
            "effect_scope": "sector",
            "magnitude": "routine",
            "political_relevance": "none",
            "economic_relevance": "strong",
            "quantified": True,
            "certainty": "clear",
            "evidence_quote": "Dovada exacta",
            "reason_ro": "Efect economic sectorial.",
            **changes,
        }
    )


def _analysis_input() -> ArticleAnalysisInput:
    return ArticleAnalysisInput(
        reference=_reference(),
        article=ExtractedArticle(
            article_id=_A,
            outlet_id="test",
            canonical_url=HttpUrl("https://example.test/article"),
            title="Titlu",
            body="Dovada exacta este aici.",
            author=None,
            published_at=datetime(2026, 8, 31, 9, tzinfo=UTC),
            source_updated_at=None,
            bucharest_day=date(2026, 8, 31),
            material_digest=_B,
            extraction_digest=_C,
        ),
    )


def _reference() -> ArtifactReference:
    return ArtifactReference(
        artifact_id="artifact:test",
        version_id=_A,
        content_digest=_D,
        r2_key="objects/test",
    )
