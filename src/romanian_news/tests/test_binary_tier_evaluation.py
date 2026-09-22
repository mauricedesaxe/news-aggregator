from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from romanian_news.analysis.binary_evaluation import BinaryProbabilityObservation, BinaryRequest
from romanian_news.artifacts import ArtifactReference
from romanian_news.binary_benchmark import (
    TYPESAFE_JEV_TARGET,
    BinaryEvaluator,
    build_binary_evaluators,
)
from romanian_news.binary_relevance_evaluation import (
    V11_MANIFEST_VERSION,
    V11_SOURCE_ARTIFACT_ID,
)
from romanian_news.binary_tier_evaluation import (
    BinaryTierCase,
    BinaryTierCaseResult,
    BinaryTierEventContext,
    BinaryTierSource,
    BinaryTierState,
    BinaryTierSubjectContext,
    adapt_binary_tier_cases,
    binary_tier_metrics,
    compose_binary_tier,
    render_binary_tier_state,
    run_binary_tier_evaluation,
)
from romanian_news.evaluation import NewsEvaluationPin, Tier


def test_tier_cases_adapt_to_exact_registered_25_by_2_shape() -> None:
    cases = _cases()
    adapted = adapt_binary_tier_cases(cases)

    assert len(adapted) == 25
    assert sum(len(case.judgments) for case in adapted) == 50
    assert all(
        tuple(judgment.judgment_id for judgment in case.judgments)
        == ("main_subject", "worth_knowing_if_not_main")
        for case in adapted
    )


@pytest.mark.parametrize(
    ("main", "worth", "expected"),
    [
        (Decimal("0.5"), Decimal("0"), "main"),
        (Decimal("0.499"), Decimal("0.5"), "worth_knowing"),
        (Decimal("0.499"), Decimal("0.499"), "excluded"),
    ],
)
def test_tier_composition_covers_all_ordered_branches(
    main: Decimal, worth: Decimal, expected: Tier
) -> None:
    assert compose_binary_tier(main, worth) == expected


def test_main_case_conditional_judgment_is_not_scored_as_binary_ground_truth() -> None:
    source = _source(tuple(_case(index, "main") for index in range(25)))
    result = run_binary_tier_evaluation(
        source,
        targets=(TYPESAFE_JEV_TARGET,),
        trial_refs=("test:trial-001",),
        evaluators={"typesafe-jev": _constant_evaluator(Decimal("0.9"))},
        execution_ref="git:tier-conditional",
        dry_run=True,
    )

    worth_result = next(
        item
        for item in result.binary_result.results
        if item.case_id == "tier-000" and item.question_id == "worth_knowing_if_not_main"
    )
    composed = result.runs[0].cases[0]

    assert worth_result.passed is False
    assert composed.predicted_tier == "main"
    assert composed.exact_match is True
    assert composed.worth_knowing_if_not_main_correct is None
    assert result.runs[0].metrics.exact_accuracy == Decimal(1)


def test_safe_state_excludes_labels_outputs_order_and_curation_metadata() -> None:
    case = _case(0, "main")

    state = render_binary_tier_state(case)

    for forbidden in (
        "expected_tier",
        "observed_tier",
        "report_group_ids",
        "DailySubjectAssessmentSet",
        "assessment",
        "feedback",
        "control",
        "confidence",
        "uncertainty",
    ):
        assert forbidden not in state
    assert state == render_binary_tier_state(case)


def test_three_class_metrics_report_confusion_recall_and_exact_accuracy() -> None:
    metrics = binary_tier_metrics(
        (
            _case_result("main", "main", 0),
            _case_result("worth_knowing", "excluded", 1),
            _case_result("excluded", "excluded", 2),
        )
    )

    assert metrics.exact_accuracy == Decimal(2) / Decimal(3)
    assert metrics.confusion_matrix.main.main == 1
    assert metrics.confusion_matrix.worth_knowing.excluded == 1
    assert metrics.confusion_matrix.excluded.excluded == 1
    assert metrics.per_tier_recall.main == Decimal(1)
    assert metrics.per_tier_recall.worth_knowing == Decimal(0)
    assert metrics.per_tier_recall.excluded == Decimal(1)


def test_synthetic_tier_source_executes_through_generic_dry_runner() -> None:
    source = _source(_cases())
    evaluators = build_binary_evaluators((TYPESAFE_JEV_TARGET,), dry_run=True)

    result = run_binary_tier_evaluation(
        source,
        targets=(TYPESAFE_JEV_TARGET,),
        trial_refs=("test:dry-run",),
        evaluators=evaluators,
        execution_ref="git:tier-dry-run",
        dry_run=True,
    )

    assert len(result.binary_result.results) == 50
    assert len(result.runs) == 1
    assert result.runs[0].metrics.completed_cases == 25
    assert all(item.status == "completed" for item in result.runs[0].cases)


def test_tier_source_rejects_any_case_count_other_than_reviewed_25() -> None:
    with pytest.raises(ValueError, match="requires 25 cases"):
        _ = _source(_cases()[:-1])


def _cases() -> tuple[BinaryTierCase, ...]:
    tiers: tuple[Tier, ...] = ("main", "worth_knowing", "excluded")
    return tuple(_case(index, tiers[index % len(tiers)]) for index in range(25))


def _case(index: int, expected_tier: Tier) -> BinaryTierCase:
    group_id = f"{index + 1:064x}"
    subject_id = f"{index + 101:064x}"
    event = BinaryTierEventContext(
        group_id=group_id,
        title=f"Eveniment {index}",
        summary=f"Rezumat concret {index}",
        key_points=(f"Detaliu {index}",),
        evidence_quotes=(f"Dovada {index}",),
    )
    return BinaryTierCase(
        case_id=f"tier-{index:03d}",
        report_version_id=f"{index + 201:064x}",
        group_id=group_id,
        control=index == 0,
        expected_tier=expected_tier,
        state=BinaryTierState(
            day=date(2026, 9, 11),
            target_subject_id=subject_id,
            subjects=(
                BinaryTierSubjectContext(
                    subject_id=subject_id,
                    title=f"Subiect {index}",
                    summary=f"Context {index}",
                    events=(event,),
                ),
            ),
        ),
    )


def _source(cases: tuple[BinaryTierCase, ...]) -> BinaryTierSource:
    return BinaryTierSource(
        pin=NewsEvaluationPin(
            manifest_version_id=V11_SOURCE_ARTIFACT_ID,
            baseline_version_id="f" * 64,
        ),
        manifest_reference=ArtifactReference(
            artifact_id="news:evaluation-manifest:v11",
            version_id=V11_SOURCE_ARTIFACT_ID,
            content_digest="e" * 64,
            r2_key="test/evaluation-v11.json",
        ),
        declared_manifest_version=V11_MANIFEST_VERSION,
        cases=cases,
    )


def _constant_evaluator(probability: Decimal) -> BinaryEvaluator:
    def evaluate(request: BinaryRequest, _trial_ref: str) -> BinaryProbabilityObservation:
        return BinaryProbabilityObservation(
            request_id="a" * 64,
            provider_request_id=f"test:{request.question.question_id}",
            model="jev-1.13.0",
            probability=probability,
            predicted_accepted=probability >= request.question.threshold,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            estimated_cost_usd=Decimal(0),
        )

    return evaluate


def _case_result(expected: Tier, predicted: Tier, index: int) -> BinaryTierCaseResult:
    expected_worth = None if expected == "main" else predicted == expected
    return BinaryTierCaseResult(
        status="completed",
        case_id=f"case-{index}",
        target_id="typesafe-jev",
        trial_ref="trial-001",
        expected_tier=expected,
        predicted_tier=predicted,
        main_subject_probability=Decimal("0.9") if predicted == "main" else Decimal("0.1"),
        worth_knowing_if_not_main_probability=(
            Decimal("0.9") if predicted == "worth_knowing" else Decimal("0.1")
        ),
        main_subject_correct=(predicted == "main") == (expected == "main"),
        worth_knowing_if_not_main_correct=expected_worth,
        exact_match=predicted == expected,
    )
