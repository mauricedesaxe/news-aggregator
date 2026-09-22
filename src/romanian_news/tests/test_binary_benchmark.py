from __future__ import annotations

from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

from romanian_news.analysis.binary_benchmark import (
    analyze_binary_benchmark,
    exact_binomial_interval,
    exact_mcnemar_p_value,
    write_binary_analysis,
)
from romanian_news.analysis.binary_evaluation import (
    BinaryProbabilityObservation,
    BinaryQuestion,
    BinaryRequest,
    binary_state_digest,
)
from romanian_news.binary_benchmark import (
    TYPESAFE_JEV_TARGET,
    BenchmarkId,
    BinaryBenchmarkCase,
    BinaryEvaluator,
    BinaryJudgmentCase,
    BinaryJudgmentResult,
    BinaryModelIdentityMismatch,
    BinarySpendLedger,
    BinarySpendLimitExceeded,
    build_execution_identity,
    invoke_binary_evaluator,
    load_binary_checkpoint,
    run_registered_binary_benchmark,
    write_binary_checkpoint,
)
from romanian_news.derived_binary_protocol import DERIVED_BINARY_BENCHMARKS


def test_registered_tier_definition_drives_two_judgments_and_analysis(tmp_path: Path) -> None:
    assert tuple(DERIVED_BINARY_BENCHMARKS) == (
        "grouping",
        "ranking",
        "tier",
        "confidence",
        "daily_theme",
    )
    definition = DERIVED_BINARY_BENCHMARKS["tier"]
    cases = _cases("tier")
    identity = build_execution_identity(
        definition,
        source_artifact_id="source-v1",
        declared_manifest_version="manifest-v1",
        cases=cases,
        targets=(TYPESAFE_JEV_TARGET,),
        trial_refs=("trial-001", "trial-002"),
        execution_mode="dry_run",
        execution_ref="git:test",
    )

    result = run_registered_binary_benchmark(
        definition,
        identity,
        cases,
        {"typesafe-jev": _evaluator()},
    )
    report = analyze_binary_benchmark(definition, result)
    json_path = tmp_path / "analysis.json"
    markdown_path = tmp_path / "analysis.md"
    write_binary_analysis(report, json_path, markdown_path)
    first_json = json_path.read_bytes()
    first_markdown = markdown_path.read_bytes()
    write_binary_analysis(report, json_path, markdown_path)

    assert definition.composition_digest is not None
    assert len(definition.questions) == 2
    assert len(result.results) == 100
    assert all(row.status == "completed" for row in result.results)
    assert exact_binomial_interval(0, 10)["upper"] == pytest.approx(0.3084971078)
    assert exact_mcnemar_p_value(1, 9) == pytest.approx(0.021484375)
    assert json_path.read_bytes() == first_json
    assert markdown_path.read_bytes() == first_markdown
    assert report["pareto"] == {
        "dimensions": [
            "worst_trial_errors",
            "mean_cost_usd",
            "median_trial_p95_latency_ms",
        ],
        "frontier": ["typesafe-jev"],
        "targets": [
            {
                "target_id": "typesafe-jev",
                "error_counts_by_trial": [0, 0],
                "worst_trial_errors": 0,
                "total_attempt_count": 100,
                "total_input_tokens": 100,
                "total_output_tokens": 100,
                "total_cost_usd": 0.0,
                "mean_cost_usd": 0.0,
                "median_trial_p50_latency_ms": 1.0,
                "median_trial_p95_latency_ms": 1.0,
                "non_dominated": True,
            }
        ],
    }


def test_checkpoint_interruption_resume_and_identity_mismatch(tmp_path: Path) -> None:
    definition = DERIVED_BINARY_BENCHMARKS["grouping"]
    cases = _cases("grouping")
    identity = build_execution_identity(
        definition,
        source_artifact_id="source-v1",
        declared_manifest_version="manifest-v1",
        cases=cases,
        targets=(TYPESAFE_JEV_TARGET,),
        trial_refs=("trial-001",),
        execution_mode="dry_run",
        execution_ref="git:resume",
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    calls: Counter[str] = Counter()
    accumulated: list[BinaryJudgmentResult] = []
    spend = BinarySpendLedger(definition.spend_ceiling_usd)

    def checkpoint(result: BinaryJudgmentResult) -> None:
        accumulated.append(result)
        write_binary_checkpoint(checkpoint_path, identity, spend, accumulated)
        if len(accumulated) == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_registered_binary_benchmark(
            definition,
            identity,
            cases,
            {"typesafe-jev": _evaluator(calls)},
            on_result=checkpoint,
            spend=spend,
        )

    loaded, reusable = load_binary_checkpoint(checkpoint_path, identity, retry_failed=False)
    resumed_spend = BinarySpendLedger(
        definition.spend_ceiling_usd,
        actual_spend_usd=loaded.actual_spend_usd,
    )
    resumed = run_registered_binary_benchmark(
        definition,
        identity,
        cases,
        {"typesafe-jev": _evaluator(calls)},
        reusable_results=reusable,
        spend=resumed_spend,
    )

    assert len(resumed.results) == 23
    assert sum(calls.values()) == 23
    mismatched = identity.model_copy(update={"execution_ref": "git:other"})
    with pytest.raises(ValueError, match="identity"):
        load_binary_checkpoint(checkpoint_path, mismatched, retry_failed=False)


def test_retry_loading_omits_failed_judgments(tmp_path: Path) -> None:
    definition = DERIVED_BINARY_BENCHMARKS["confidence"]
    cases = _cases("confidence")
    identity = build_execution_identity(
        definition,
        source_artifact_id="source-v1",
        declared_manifest_version="manifest-v1",
        cases=cases,
        targets=(TYPESAFE_JEV_TARGET,),
        trial_refs=("trial-001",),
        execution_mode="dry_run",
        execution_ref="git:retry",
    )
    result = run_registered_binary_benchmark(
        definition,
        identity,
        cases,
        {"typesafe-jev": _evaluator(fail_state="state-000")},
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    spend = BinarySpendLedger(definition.spend_ceiling_usd)
    write_binary_checkpoint(checkpoint_path, identity, spend, result.results)

    _checkpoint, all_results = load_binary_checkpoint(checkpoint_path, identity, retry_failed=False)
    _checkpoint, completed = load_binary_checkpoint(checkpoint_path, identity, retry_failed=True)

    assert len(all_results) == 4
    assert len(completed) == 3


def test_spend_ceiling_stops_before_unaffordable_request() -> None:
    definition = DERIVED_BINARY_BENCHMARKS["grouping"]
    cases = _cases("grouping")
    identity = build_execution_identity(
        definition,
        source_artifact_id="source-v1",
        declared_manifest_version="manifest-v1",
        cases=cases,
        targets=(TYPESAFE_JEV_TARGET,),
        trial_refs=("trial-001",),
        execution_mode="live",
        execution_ref="git:budget",
    )
    calls: Counter[str] = Counter()
    spend = BinarySpendLedger(Decimal("0.0085"))

    with pytest.raises(BinarySpendLimitExceeded, match="before the next request"):
        run_registered_binary_benchmark(
            definition,
            identity,
            cases,
            {"typesafe-jev": _evaluator(calls, cost=Decimal("0.001"))},
            spend=spend,
        )

    assert sum(calls.values()) == 1
    assert spend.actual_spend_usd == Decimal("0.001")


def test_actual_model_identity_mismatch_stops_execution() -> None:
    definition = DERIVED_BINARY_BENCHMARKS["confidence"]
    cases = _cases("confidence")
    identity = build_execution_identity(
        definition,
        source_artifact_id="source-v1",
        declared_manifest_version="manifest-v1",
        cases=cases,
        targets=(TYPESAFE_JEV_TARGET,),
        trial_refs=("trial-001",),
        execution_mode="live",
        execution_ref="git:model-mismatch",
    )
    calls: Counter[str] = Counter()
    evaluator = _evaluator(calls)

    def mismatched(request: BinaryRequest, trial_ref: str) -> BinaryProbabilityObservation:
        return evaluator(request, trial_ref).model_copy(update={"model": "jev-2.0.0"})

    with pytest.raises(BinaryModelIdentityMismatch, match="does not match"):
        run_registered_binary_benchmark(
            definition,
            identity,
            cases,
            {"typesafe-jev": mismatched},
        )

    assert sum(calls.values()) == 1


def test_callback_evaluator_failure_records_full_wall_latency() -> None:
    def fail_before_callback(
        _request: BinaryRequest,
        _trial_ref: str,
        *,
        on_attempt: object | None = None,
    ) -> BinaryProbabilityObservation:
        _ = on_attempt
        raise RuntimeError("provider failed before callback")

    ticks = iter((10.0, 10.125))
    invocation = invoke_binary_evaluator(
        fail_before_callback,
        _request(DERIVED_BINARY_BENCHMARKS["confidence"].questions[0], "state"),
        "trial-001",
        clock=lambda: next(ticks),
    )

    assert invocation.observation is None
    assert invocation.attempts[0].latency_ms == 125


def _cases(benchmark: BenchmarkId) -> tuple[BinaryBenchmarkCase, ...]:
    definition = DERIVED_BINARY_BENCHMARKS[benchmark]
    return tuple(
        BinaryBenchmarkCase(
            case_id=f"case-{index:03d}",
            identity=(f"identity-{index:03d}",),
            control=index == 0,
            judgments=tuple(
                BinaryJudgmentCase(
                    judgment_id=question.question_id,
                    request=_request(question, f"state-{index:03d}"),
                    expected=True,
                )
                for question in definition.questions
            ),
        )
        for index in range(definition.case_count)
    )


def _request(question: BinaryQuestion, state: str) -> BinaryRequest:
    return BinaryRequest(
        question=question,
        state=state,
        state_digest=binary_state_digest(state),
    )


def _evaluator(
    calls: Counter[str] | None = None,
    *,
    fail_state: str | None = None,
    cost: Decimal | None = None,
) -> BinaryEvaluator:
    selected_cost = Decimal(0) if cost is None else cost

    def evaluate(
        request: BinaryRequest, _trial_ref: str, **_kwargs: object
    ) -> BinaryProbabilityObservation:
        if calls is not None:
            calls[request.state_digest] += 1
        if request.state == fail_state:
            raise RuntimeError("registered failure")
        return BinaryProbabilityObservation(
            request_id="1" * 64,
            provider_request_id=f"fake:{request.state_digest}",
            model="jev-1.13.0",
            probability=Decimal("0.75"),
            predicted_accepted=True,
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            estimated_cost_usd=selected_cost,
        )

    return evaluate
