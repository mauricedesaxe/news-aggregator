from decimal import Decimal
from typing import Literal

import pytest
from pydantic import ValidationError

from romanian_news.analysis.binary_evaluation import (
    BinaryAttemptError,
    BinaryAttemptEvidence,
    BinaryProbabilityObservation,
)
from romanian_news.binary_relevance_evaluation import (
    OPENROUTER_GEMINI_25_TARGET,
    OPENROUTER_GEMINI_38_TARGET,
    TYPESAFE_JEV_TARGET,
    V11_MANIFEST_VERSION,
    V11_SOURCE_ARTIFACT_ID,
    BinaryRelevanceSource,
    binary_relevance_metrics,
    load_v11_binary_relevance_source,
    run_binary_relevance_evaluation,
)
from romanian_news.evaluation import NewsEvaluationManifest, NewsEvaluationPin
from romanian_news.tests import evaluation_factories


def test_runner_reuses_one_request_per_manifest_case_and_orders_full_coverage(
    monkeypatch,
) -> None:
    source = _source()
    articles = {
        case.article.r2_key: evaluation_factories.embedded_article(index).value
        for index, case in enumerate(source.manifest.cases, start=1)
        if case.concern == "relevance"
    }
    monkeypatch.setattr(
        "romanian_news.binary_relevance_evaluation.read_verified_r2_object",
        lambda key, _digest: articles[key].model_dump_json().encode(),
    )
    seen: dict[str, list[int]] = {
        "openrouter-gemini-2.5-flash": [],
        "typesafe-jev": [],
    }

    def evaluator(target_id: str, actual_model: str, probability: Decimal):
        def evaluate(request, trial_ref):
            seen[target_id].append(id(request))
            return BinaryProbabilityObservation(
                request_id=("1" if target_id == "typesafe-jev" else "2") * 64,
                provider_request_id=f"{target_id}:{trial_ref}",
                model=actual_model,
                probability=probability,
                predicted_accepted=probability >= request.question.threshold,
                input_tokens=10,
                output_tokens=2,
                latency_ms=20,
                estimated_cost_usd=Decimal("0.00000021"),
            )

        return evaluate

    result = run_binary_relevance_evaluation(
        source,
        targets=(TYPESAFE_JEV_TARGET, OPENROUTER_GEMINI_25_TARGET),
        trial_refs=("trial-2", "trial-1"),
        evaluators={
            "typesafe-jev": evaluator("typesafe-jev", "jev-1.13.0", Decimal("0.8")),
            "openrouter-gemini-2.5-flash": evaluator(
                "openrouter-gemini-2.5-flash",
                "google/gemini-2.5-flash-001",
                Decimal("0.2"),
            ),
        },
    )

    assert tuple((run.target.target_id, run.trial_ref) for run in result.runs) == (
        ("openrouter-gemini-2.5-flash", "trial-1"),
        ("openrouter-gemini-2.5-flash", "trial-2"),
        ("typesafe-jev", "trial-1"),
        ("typesafe-jev", "trial-2"),
    )
    assert result.case_ids == ("z-last", "a-first")
    assert all(tuple(case.case_id for case in run.cases) == result.case_ids for run in result.runs)
    assert len(result.runs) * len(result.case_ids) == sum(len(run.cases) for run in result.runs)
    assert len({case.request_id for run in result.runs for case in run.cases}) == 8
    for case_index in range(len(result.case_ids)):
        object_ids = {
            seen[target_id][trial_index * len(result.case_ids) + case_index]
            for target_id in seen
            for trial_index in range(2)
        }
        assert len(object_ids) == 1

    gemini_case = result.runs[0].cases[0]
    assert gemini_case.requested_model == "google/gemini-2.5-flash"
    assert gemini_case.actual_model == "google/gemini-2.5-flash-001"
    assert gemini_case.question_id == "relevant"
    assert gemini_case.question_digest == result.question_digest
    assert gemini_case.state_digest
    assert gemini_case.execution_policy_id == result.runs[0].target.execution_policy_id
    assert gemini_case.execution_policy_digest == result.runs[0].target.execution_policy_digest
    assert gemini_case.request_id
    assert gemini_case.adapter_request_id == "2" * 64
    assert gemini_case.provider_request_id == "openrouter-gemini-2.5-flash:trial-1"


def test_runner_keeps_failed_adapter_cases_and_exact_decimal_accounting(monkeypatch) -> None:
    source = _source(one_case=True)
    article = evaluation_factories.embedded_article(1).value
    monkeypatch.setattr(
        "romanian_news.binary_relevance_evaluation.read_verified_r2_object",
        lambda _key, _digest: article.model_dump_json().encode(),
    )

    def success(request, _trial_ref):
        return BinaryProbabilityObservation(
            request_id="3" * 64,
            provider_request_id="provider-success",
            model="jev-1.13.0",
            probability=Decimal("0.75"),
            predicted_accepted=True,
            input_tokens=3,
            output_tokens=1,
            latency_ms=10,
            estimated_cost_usd=Decimal("0.1") + Decimal("0.2"),
        )

    def failure(_request, _trial_ref, *, on_attempt=None):
        attempt = BinaryAttemptEvidence(
            attempt_number=1,
            status="terminal_error",
            provider_request_id="provider-failure",
            actual_model="jev-1.13.0",
            http_status=503,
            error=BinaryAttemptError(
                error_type="RuntimeError",
                message="adapter unavailable",
                retryable=True,
            ),
            input_tokens=5,
            output_tokens=2,
            cost_usd=Decimal("0.125"),
            latency_ms=25,
            probability=None,
        )
        if on_attempt is not None:
            on_attempt(attempt)
        raise RuntimeError("adapter unavailable")

    def evaluate(request, trial_ref, *, on_attempt=None):
        if trial_ref == "failure":
            return failure(request, trial_ref, on_attempt=on_attempt)
        return success(request, trial_ref)

    result = run_binary_relevance_evaluation(
        source,
        targets=(TYPESAFE_JEV_TARGET,),
        trial_refs=("success", "failure"),
        evaluators={"typesafe-jev": evaluate},
    )

    failed, completed = result.runs
    assert failed.trial_ref == "failure"
    assert failed.cases[0].status == "failed"
    assert failed.cases[0].passed is False
    assert failed.cases[0].probability is None
    assert failed.cases[0].wall_latency_ms == 25
    assert failed.cases[0].input_tokens == 5
    assert failed.cases[0].output_tokens == 2
    assert failed.cases[0].cost_usd == Decimal("0.125")
    assert failed.cases[0].attempts[0].provider_request_id == "provider-failure"
    assert failed.cases[0].errors[0].error_type == "RuntimeError"
    assert failed.cases[0].errors[0].message == "adapter unavailable"
    assert failed.metrics.failed_cases == 1
    assert failed.metrics.total_cases == 1
    assert failed.metrics.request_count == 1
    assert failed.metrics.total_attempt_latency_ms == 25
    assert failed.metrics.total_cost_usd == Decimal("0.125")
    assert failed.metrics.accuracy == Decimal(0)
    assert completed.metrics.total_cost_usd == Decimal("0.3")
    assert completed.cases[0].cost_usd == Decimal("0.3")
    assert completed.cases[0].cost_basis == "estimated-input-rate"
    assert sum(len(run.cases) for run in result.runs) == 2


def test_metrics_are_exact_and_validate_against_case_results(monkeypatch) -> None:
    source = _source(one_case=True)
    article = evaluation_factories.embedded_article(1).value
    monkeypatch.setattr(
        "romanian_news.binary_relevance_evaluation.read_verified_r2_object",
        lambda _key, _digest: article.model_dump_json().encode(),
    )
    observations = iter(
        (
            _observation("0.9", True, 10, "0.01"),
            _observation("0.8", True, 20, "0.02"),
            _observation("0.2", False, 30, "0.03"),
            _observation("0.1", False, 40, "0.04"),
        )
    )
    case = source.manifest.cases[0]
    source = source.model_copy(
        update={
            "manifest": source.manifest.model_copy(
                update={
                    "cases": (
                        case.model_copy(
                            update={"case_id": "tp", "expected_accepted": True, "control": True}
                        ),
                        case.model_copy(update={"case_id": "fp", "expected_accepted": False}),
                        case.model_copy(
                            update={"case_id": "fn", "expected_accepted": True, "control": True}
                        ),
                        case.model_copy(update={"case_id": "tn", "expected_accepted": False}),
                    )
                }
            )
        }
    )
    result = run_binary_relevance_evaluation(
        source,
        targets=(OPENROUTER_GEMINI_25_TARGET,),
        trial_refs=("trial-1",),
        evaluators={"openrouter-gemini-2.5-flash": lambda _request, _trial_ref: next(observations)},
    )
    metrics = binary_relevance_metrics(result.runs[0].cases)

    assert metrics.model_dump() == {
        "passed_cases": 2,
        "completed_cases": 4,
        "failed_cases": 0,
        "total_cases": 4,
        "accuracy": Decimal("0.5"),
        "precision": Decimal("0.5"),
        "recall": Decimal("0.5"),
        "positive_control_preservation": Decimal("0.5"),
        "false_negative_ids": ("fn",),
        "request_count": 4,
        "input_tokens": 12,
        "output_tokens": 4,
        "total_tokens": 16,
        "total_cost_usd": Decimal("0.10"),
        "total_attempt_latency_ms": 100,
        "p50_wall_latency_ms": Decimal("25.0"),
        "p95_wall_latency_ms": Decimal("38.50"),
    }


def test_v11_source_rejects_wrong_artifact_or_declared_manifest_version(monkeypatch) -> None:
    source = _source(one_case=True)
    with pytest.raises(ValidationError, match="source artifact"):
        BinaryRelevanceSource(
            pin=source.pin.model_copy(update={"manifest_version_id": "f" * 64}),
            manifest_reference=source.manifest_reference.model_copy(
                update={"version_id": "f" * 64}
            ),
            manifest=source.manifest,
        )
    with pytest.raises(ValidationError, match="declared manifest version"):
        BinaryRelevanceSource(
            pin=source.pin,
            manifest_reference=source.manifest_reference,
            manifest=source.manifest.model_copy(update={"version": "v12"}),
        )
    with pytest.raises(ValueError, match="source artifact"):
        load_v11_binary_relevance_source(
            source.pin.model_copy(update={"manifest_version_id": "f" * 64}).model_dump_json()
        )

    monkeypatch.setattr(
        "romanian_news.binary_relevance_evaluation.read_news_evaluation_artifact_references",
        lambda _ids: {V11_SOURCE_ARTIFACT_ID: source.manifest_reference},
    )
    monkeypatch.setattr(
        "romanian_news.binary_relevance_evaluation.read_verified_r2_object",
        lambda _key, _digest: source.manifest.model_dump_json().encode(),
    )
    loaded = load_v11_binary_relevance_source(source.pin.model_dump_json())
    assert loaded == source


def test_models_are_frozen_and_strict() -> None:
    assert TYPESAFE_JEV_TARGET.model_config.get("frozen") is True
    assert BinaryAttemptEvidence.model_config.get("frozen") is True
    assert BinaryAttemptEvidence.model_config.get("strict") is True
    assert OPENROUTER_GEMINI_25_TARGET.target_id != OPENROUTER_GEMINI_38_TARGET.target_id
    assert (
        OPENROUTER_GEMINI_25_TARGET.execution_policy_digest
        != OPENROUTER_GEMINI_38_TARGET.execution_policy_digest
    )
    with pytest.raises(ValidationError):
        TYPESAFE_JEV_TARGET.__class__.model_validate(
            {**TYPESAFE_JEV_TARGET.model_dump(), "requested_model": 1}, strict=True
        )
    with pytest.raises(ValidationError, match="requires a retryable error"):
        BinaryAttemptEvidence(
            attempt_number=1,
            status="retryable_error",
            provider_request_id=None,
            actual_model=None,
            http_status=None,
            error=BinaryAttemptError(
                error_type="Timeout",
                message="retryable timeout",
                retryable=False,
            ),
            input_tokens=None,
            output_tokens=None,
            cost_usd=None,
            latency_ms=10,
            probability=None,
        )

    with pytest.raises(ValidationError, match="ordered"):
        BinaryProbabilityObservation(
            request_id="a" * 64,
            provider_request_id="provider-request",
            model="model",
            probability=Decimal("0.5"),
            predicted_accepted=True,
            input_tokens=1,
            output_tokens=1,
            latency_ms=2,
            estimated_cost_usd=Decimal("0.1"),
            attempts=(
                _attempt(2, "retryable_error"),
                _attempt(1, "completed"),
            ),
        )


def _source(*, one_case: bool = False) -> BinaryRelevanceSource:
    original = evaluation_factories.synthetic_manifest()
    relevance = next(case for case in original.cases if case.concern == "relevance")
    second_provenance = original.cases[1].provenance
    cases = (
        relevance.model_copy(update={"case_id": "z-last"}),
        relevance.model_copy(
            update={
                "case_id": "a-first",
                "expected_accepted": False,
                "provenance": second_provenance,
            }
        ),
    )
    if one_case:
        cases = cases[:1]
    manifest_payload = original.model_dump(mode="python")
    manifest_payload.update(
        version=V11_MANIFEST_VERSION,
        source_feedback_ids=tuple(
            feedback_id for case in cases for feedback_id in case.provenance.feedback_ids
        ),
        cases=tuple(case.model_dump(mode="python") for case in cases),
    )
    manifest = NewsEvaluationManifest.model_validate(manifest_payload, strict=True)
    reference = evaluation_factories.reference(900, "evaluation-manifest").model_copy(
        update={"version_id": V11_SOURCE_ARTIFACT_ID}
    )
    return BinaryRelevanceSource(
        pin=NewsEvaluationPin(
            manifest_version_id=V11_SOURCE_ARTIFACT_ID,
            baseline_version_id="c" * 64,
        ),
        manifest_reference=reference,
        manifest=manifest,
    )


def _observation(
    probability: str,
    verdict: bool,
    latency_ms: int,
    cost: str,
) -> BinaryProbabilityObservation:
    return BinaryProbabilityObservation(
        request_id="4" * 64,
        provider_request_id="provider-request",
        model="google/gemini-2.5-flash-001",
        probability=Decimal(probability),
        predicted_accepted=verdict,
        input_tokens=3,
        output_tokens=1,
        latency_ms=latency_ms,
        estimated_cost_usd=Decimal(cost),
    )


def _attempt(
    attempt_number: int,
    status: Literal["completed", "retryable_error", "terminal_error"],
) -> BinaryAttemptEvidence:
    completed = status == "completed"
    return BinaryAttemptEvidence(
        attempt_number=attempt_number,
        status=status,
        provider_request_id="provider-request" if completed else None,
        actual_model="model" if completed else None,
        http_status=None,
        error=(
            None
            if completed
            else BinaryAttemptError(error_type="Error", message="retry", retryable=True)
        ),
        input_tokens=1 if completed else None,
        output_tokens=1 if completed else None,
        cost_usd=Decimal("0.1") if completed else None,
        latency_ms=1,
        probability=Decimal("0.5") if completed else None,
    )
