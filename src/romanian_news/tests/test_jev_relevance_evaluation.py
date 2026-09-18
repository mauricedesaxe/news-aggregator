import pytest

from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.jev_relevance import JevRelevanceObservation
from romanian_news.catalog.evaluations import LoadedNewsEvaluationRelease
from romanian_news.evaluation import (
    NewsEvaluationPin,
    build_news_evaluation_baseline,
    evaluate_news_dataset,
)
from romanian_news.evaluation_projection import FreshEvaluationPlan
from romanian_news.jev_relevance_evaluation import (
    JevRelevanceCaseResult,
    JevRelevanceEvaluationResult,
    jev_relevance_metrics,
    run_jev_relevance_evaluation,
)
from romanian_news.tests import evaluation_factories
from romanian_news.tests.evaluation_factories import synthetic_manifest


def test_jev_evaluation_runs_every_fresh_reference_independently(monkeypatch) -> None:
    release = _release()
    plan = FreshEvaluationPlan(implementation_ref="git:test")
    article = evaluation_factories.embedded_article(1).value
    calls = []
    monkeypatch.setattr(
        "romanian_news.jev_relevance_evaluation.read_verified_r2_object",
        lambda key, digest: calls.append(("read", key, digest))
        or article.model_dump_json().encode(),
    )

    def evaluate(value, *, execution_ref, policy):
        calls.append(("evaluate", execution_ref, value.reference.version_id, policy.policy_id))
        return JevRelevanceObservation(
            request_id=str(len(calls)).zfill(64),
            provider_request_id=f"provider-{execution_ref}",
            model="jev-1.13.0",
            probability=0.8,
            predicted_accepted=True,
            input_tokens=10,
            output_tokens=2,
            latency_ms=20,
            estimated_cost_usd=0.00000042,
        )

    monkeypatch.setattr("romanian_news.jev_relevance_evaluation.evaluate_jev_relevance", evaluate)

    result = run_jev_relevance_evaluation(release, plan)

    assert result.plan is plan
    assert tuple(run.execution_ref for run in result.runs) == plan.implementation_refs
    assert [call[1] for call in calls if call[0] == "evaluate"] == list(plan.implementation_refs)
    assert len([call for call in calls if call[0] == "read"]) == 3


def test_jev_metrics_cover_quality_accounting_and_latency() -> None:
    results = (
        _case("true-positive", True, True, control=True, latency_ms=10),
        _case("false-positive", False, True, latency_ms=20),
        _case("false-negative", True, False, control=True, latency_ms=30),
        _case("true-negative", False, False, latency_ms=40),
    )

    metrics = jev_relevance_metrics(results)

    assert metrics.model_dump() == {
        "passed_cases": 2,
        "total_cases": 4,
        "precision": 0.5,
        "recall": 0.5,
        "positive_control_preservation": 0.5,
        "false_negative_ids": ("false-negative",),
        "input_tokens": 40,
        "output_tokens": 8,
        "estimated_cost_usd": 0.04,
        "p50_latency_ms": 25.0,
        "p95_latency_ms": 38.5,
    }


def test_jev_group_requires_exact_plan_reference_coverage() -> None:
    plan = FreshEvaluationPlan(implementation_ref="git:test")
    with pytest.raises(ValueError, match="exact FreshEvaluationPlan reference coverage"):
        JevRelevanceEvaluationResult(
            manifest_artifact_version_id="a" * 64,
            plan=plan,
            runs=(),
        )


def _case(
    case_id: str,
    expected: bool,
    predicted: bool,
    *,
    control: bool = False,
    latency_ms: int,
) -> JevRelevanceCaseResult:
    return JevRelevanceCaseResult(
        case_id=case_id,
        article_version_id="1" * 64,
        execution_ref="trial-1",
        expected_accepted=expected,
        predicted_accepted=predicted,
        pass_probability=0.8,
        control=control,
        passed=expected == predicted,
        model="jev-1.13.0",
        request_id="2" * 64,
        provider_request_id=f"provider-{case_id}",
        input_tokens=10,
        output_tokens=2,
        latency_ms=latency_ms,
        estimated_cost_usd=0.01,
    )


def _release() -> LoadedNewsEvaluationRelease:
    manifest = synthetic_manifest()
    dataset = evaluation_factories.synthetic_dataset()
    manifest_reference = evaluation_factories.reference(900, "evaluation-manifest")
    baseline_reference = ArtifactReference(
        artifact_id="news:evaluation-baseline:test",
        version_id="c" * 64,
        content_digest="d" * 64,
        r2_key="news/evaluations/baselines/test.json",
    )
    return LoadedNewsEvaluationRelease(
        pin=NewsEvaluationPin(
            manifest_version_id=manifest_reference.version_id,
            baseline_version_id=baseline_reference.version_id,
        ),
        manifest_reference=manifest_reference,
        baseline_reference=baseline_reference,
        manifest=manifest,
        baseline=build_news_evaluation_baseline(
            evaluate_news_dataset(dataset),
            manifest_reference,
        ),
        dataset=dataset,
    )
