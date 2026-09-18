from types import SimpleNamespace
from unittest.mock import Mock

import dagster as dg

from romanian_news.evaluation_projection import RelevanceV3SliceMetrics
from romanian_news.worker import relevance_v3_evaluation
from romanian_news.worker.definitions import defs
from romanian_news.worker.relevance_v3_evaluation import RelevanceV3EvaluationSummary


def test_manual_v3_evaluation_returns_all_run_diagnostics(monkeypatch) -> None:
    release = object()
    evaluation = _evaluation()
    seen = {}
    setup = Mock()
    setup.load.return_value = release
    monkeypatch.setattr(relevance_v3_evaluation, "ensure_news_catalog_schema", setup.schema)
    monkeypatch.setattr(relevance_v3_evaluation, "load_news_evaluation_release", setup.load)
    monkeypatch.setattr(
        relevance_v3_evaluation,
        "run_relevance_v3_evaluation",
        lambda loaded, plan: seen.update(loaded=loaded, plan=plan) or evaluation,
    )
    monkeypatch.setattr(relevance_v3_evaluation, "IMPLEMENTATION_REF", "git:test")
    monkeypatch.setattr(
        relevance_v3_evaluation,
        "flush_langfuse_traces",
        lambda: seen.update(flushed=True),
    )

    result = relevance_v3_evaluation.relevance_v3_evaluation.execute_in_process()

    assert result.success
    assert seen["loaded"] is release
    assert seen["plan"] == relevance_v3_evaluation.FreshEvaluationPlan(
        implementation_ref="git:test"
    )
    assert seen.get("flushed") is True
    output = result.output_for_node("relevance_v3_evaluation_op")
    assert isinstance(output, RelevanceV3EvaluationSummary)
    assert output.implementation_ref == "git:test"
    assert len(output.runs) == 3
    assert [run.implementation_ref for run in output.runs] == [
        f"git:test:fresh-trial-{position}-of-3" for position in range(1, 4)
    ]
    assert output.runs[0].model_dump(
        include={
            "experiment_url",
            "full",
            "prior",
            "new",
            "context_early_exit_rate",
            "input_tokens",
            "output_tokens",
            "cost_usd",
            "observability_complete",
        }
    ) == {
        "experiment_url": "https://example.test/run-1",
        "full": {
            "passed_cases": 99,
            "total_cases": 105,
            "precision": 0.91,
            "recall": 0.92,
        },
        "prior": {
            "passed_cases": 55,
            "total_cases": 57,
            "precision": 0.93,
            "recall": 0.94,
        },
        "new": {
            "passed_cases": 44,
            "total_cases": 48,
            "precision": 0.9,
            "recall": 0.91,
        },
        "context_early_exit_rate": 0.2,
        "input_tokens": 100,
        "output_tokens": 20,
        "cost_usd": 0.03,
        "observability_complete": True,
    }
    assert output.false_negative_ids == ("case-a",)
    assert output.threshold_failures == ()


def test_manual_v3_evaluation_fails_with_threshold_details(monkeypatch) -> None:
    monkeypatch.setattr(relevance_v3_evaluation, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(
        relevance_v3_evaluation,
        "load_news_evaluation_release",
        lambda _content: object(),
    )
    monkeypatch.setattr(
        relevance_v3_evaluation,
        "run_relevance_v3_evaluation",
        lambda _release, _plan: _evaluation(threshold_failures=("run 2 recall below 0.90: 0.89",)),
    )
    flushes = []
    monkeypatch.setattr(
        relevance_v3_evaluation,
        "flush_langfuse_traces",
        lambda: flushes.append("flush"),
    )

    result = relevance_v3_evaluation.relevance_v3_evaluation.execute_in_process(
        raise_on_error=False
    )

    assert result.success is False
    failure = result.failure_data_for_node("relevance_v3_evaluation_op")
    assert failure is not None
    error = failure.error
    assert error is not None
    assert "run 2 recall below 0.90: 0.89" in error.message
    assert flushes == ["flush"]


def test_manual_v3_evaluation_is_registered_without_automation() -> None:
    assert defs.get_job_def("relevance_v3_evaluation") is not None
    assert relevance_v3_evaluation.relevance_v3_evaluation_op.pool == "news_model"
    schedules = defs.schedules
    assert schedules is not None
    assert all(
        schedule.job_name != "relevance_v3_evaluation"
        for schedule in schedules
        if isinstance(schedule, dg.ScheduleDefinition)
    )
    sensors = defs.sensors
    assert sensors is not None
    assert all(sensor.name != "relevance_v3_evaluation" for sensor in sensors)


def _evaluation(*, threshold_failures=()):
    full = RelevanceV3SliceMetrics(
        passed_cases=99,
        total_cases=105,
        precision=0.91,
        recall=0.92,
    )
    prior = RelevanceV3SliceMetrics(
        passed_cases=55,
        total_cases=57,
        precision=0.93,
        recall=0.94,
    )
    new = RelevanceV3SliceMetrics(
        passed_cases=44,
        total_cases=48,
        precision=0.90,
        recall=0.91,
    )
    runs = tuple(
        SimpleNamespace(
            implementation_ref=f"git:test:fresh-trial-{position}-of-3",
            policy_id="relevance-v3-recall-first",
            policy_digest=str(position) * 64,
            experiment_url=f"https://example.test/run-{position}",
            full_metrics=full,
            prior_metrics=prior,
            new_metrics=new,
            false_negative_ids=("case-a",) if position == 1 else (),
            context_early_exit_rate=0.2,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.03,
            observability_complete=True,
        )
        for position in range(1, 4)
    )
    return SimpleNamespace(
        manifest_artifact_version_id="a" * 64,
        runs=runs,
        false_negative_ids=("case-a",),
        threshold_failures=threshold_failures,
    )
