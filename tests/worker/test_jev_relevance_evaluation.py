import dagster as dg

from romanian_news.evaluation_projection import FreshEvaluationPlan
from romanian_news.jev_relevance_evaluation import (
    JevRelevanceEvaluationResult,
    JevRelevanceRunMetrics,
    JevRelevanceRunResult,
)
from romanian_news.worker import jev_relevance_evaluation
from romanian_news.worker.definitions import defs


def test_manual_jev_evaluation_uses_a_fresh_plan(monkeypatch) -> None:
    release = object()
    seen = {}
    monkeypatch.setattr(jev_relevance_evaluation, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(
        jev_relevance_evaluation,
        "load_news_evaluation_release",
        lambda _content: release,
    )
    monkeypatch.setattr(jev_relevance_evaluation, "IMPLEMENTATION_REF", "git:test")
    monkeypatch.setattr(
        jev_relevance_evaluation,
        "run_jev_relevance_evaluation",
        lambda loaded, plan: seen.update(loaded=loaded, plan=plan) or _result(plan),
    )

    result = jev_relevance_evaluation.jev_relevance_evaluation.execute_in_process()

    assert result.success
    assert seen == {"loaded": release, "plan": FreshEvaluationPlan(implementation_ref="git:test")}
    output = result.output_for_node("jev_relevance_evaluation_op")
    assert isinstance(output, JevRelevanceEvaluationResult)
    assert len(output.runs) == 3


def test_manual_jev_evaluation_is_registered_without_automation() -> None:
    assert defs.get_job_def("jev_relevance_evaluation") is not None
    assert jev_relevance_evaluation.jev_relevance_evaluation_op.pool == "news_model"
    assert all(
        schedule.job_name != "jev_relevance_evaluation"
        for schedule in defs.schedules or ()
        if isinstance(schedule, dg.ScheduleDefinition)
    )
    assert all(sensor.name != "jev_relevance_evaluation" for sensor in defs.sensors or ())


def _result(plan: FreshEvaluationPlan) -> JevRelevanceEvaluationResult:
    metrics = JevRelevanceRunMetrics(
        passed_cases=0,
        total_cases=0,
        precision=0,
        recall=0,
        positive_control_preservation=0,
        false_negative_ids=(),
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=0,
        p50_latency_ms=0,
        p95_latency_ms=0,
    )
    return JevRelevanceEvaluationResult(
        manifest_artifact_version_id="a" * 64,
        plan=plan,
        runs=tuple(
            JevRelevanceRunResult(
                manifest_artifact_version_id="a" * 64,
                execution_ref=execution_ref,
                policy_id="jev-test",
                policy_digest="b" * 64,
                case_results=(),
                metrics=metrics,
            )
            for execution_ref in plan.implementation_refs
        ),
    )
