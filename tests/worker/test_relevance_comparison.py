from types import SimpleNamespace
from unittest.mock import Mock

import dagster as dg

from romanian_news.worker import relevance_comparison
from romanian_news.worker.definitions import defs
from romanian_news.worker.relevance_comparison import FreshRelevanceComparisonSummary


def test_manual_comparison_applies_schema_and_returns_experiment_metadata(monkeypatch) -> None:
    release = object()
    comparison = _comparison()
    seen = {}
    setup = Mock()
    setup.load.return_value = release
    monkeypatch.setattr(relevance_comparison, "ensure_news_catalog_schema", setup.schema)
    monkeypatch.setattr(relevance_comparison, "load_news_evaluation_release", setup.load)
    monkeypatch.setattr(
        relevance_comparison,
        "compare_relevance_policies",
        lambda loaded, plan: seen.update(loaded=loaded, plan=plan) or comparison,
    )
    monkeypatch.setattr(relevance_comparison, "IMPLEMENTATION_REF", "git:test")
    monkeypatch.setattr(
        relevance_comparison,
        "flush_langfuse_traces",
        lambda: seen.update(flushed=True),
        raising=False,
    )

    result = relevance_comparison.fresh_relevance_comparison.execute_in_process()

    assert result.success
    assert seen["loaded"] is release
    assert seen["plan"] == relevance_comparison.FreshEvaluationPlan(implementation_ref="git:test")
    assert seen.get("flushed") is True
    output = result.output_for_node("fresh_relevance_comparison_op")
    assert isinstance(output, FreshRelevanceComparisonSummary)
    assert output.implementation_ref == "git:test"
    assert output.trial_count == 3
    assert output.promoted is True
    assert len(output.trials) == 3
    assert [trial.implementation_ref for trial in output.trials] == [
        f"git:test:fresh-trial-{ordinal}-of-3" for ordinal in range(1, 4)
    ]
    assert output.trials[0].baseline.precision == 0.5
    assert output.trials[0].candidate.precision == 1.0
    assert output.trials[0].baseline.experiment_url == "https://example.test/baseline-1"
    assert output.trials[0].candidate.experiment_url == "https://example.test/candidate-1"
    assert output.trials[1].passed is True
    assert output.trials[1].promotion_failures == ()
    assert output.promotion_failures == ()


def test_manual_comparison_fails_the_dagster_run_when_gates_fail(monkeypatch) -> None:
    monkeypatch.setattr(relevance_comparison, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(
        relevance_comparison,
        "load_news_evaluation_release",
        lambda _content: object(),
    )
    monkeypatch.setattr(
        relevance_comparison,
        "compare_relevance_policies",
        lambda _release, _plan: _comparison(
            promotion_failures=("trial 2 of 3: candidate recall below gate: 0.81",),
            failing_trial=2,
        ),
    )

    flushes = []
    monkeypatch.setattr(
        relevance_comparison,
        "flush_langfuse_traces",
        lambda: flushes.append("flush"),
        raising=False,
    )

    result = relevance_comparison.fresh_relevance_comparison.execute_in_process(
        raise_on_error=False
    )

    assert result.success is False
    failure = result.failure_data_for_node("fresh_relevance_comparison_op")
    assert failure is not None
    error = failure.error
    assert error is not None
    assert "trial 2 of 3: candidate recall below gate: 0.81" in error.message
    assert flushes == ["flush"]


def test_manual_comparison_is_registered_without_automation() -> None:
    assert defs.get_job_def("fresh_relevance_comparison") is not None
    assert relevance_comparison.fresh_relevance_comparison_op.pool == "news_model"
    schedules = defs.schedules
    assert schedules is not None
    assert all(
        schedule.job_name != "fresh_relevance_comparison"
        for schedule in schedules
        if isinstance(schedule, dg.ScheduleDefinition)
    )
    sensors = defs.sensors
    assert sensors is not None
    assert all(sensor.name != "fresh_relevance_comparison" for sensor in sensors)


def _comparison(*, promotion_failures=(), failing_trial=None):
    metrics = SimpleNamespace(
        recall=1.0,
        positive_control_preservation=1.0,
    )
    trials = tuple(
        SimpleNamespace(
            ordinal=ordinal,
            implementation_ref=f"git:test:fresh-trial-{ordinal}-of-3",
            baseline=SimpleNamespace(
                policy_id="relevance-v1",
                policy_digest="1" * 64,
                metrics=SimpleNamespace(precision=0.5, **vars(metrics)),
                experiment_url=f"https://example.test/baseline-{ordinal}",
            ),
            candidate=SimpleNamespace(
                policy_id="relevance-v2",
                policy_digest="2" * 64,
                metrics=SimpleNamespace(precision=1.0, **vars(metrics)),
                experiment_url=f"https://example.test/candidate-{ordinal}",
            ),
            promotion_failures=(
                ("candidate recall below gate: 0.81",) if ordinal == failing_trial else ()
            ),
            passed=ordinal != failing_trial,
        )
        for ordinal in range(1, 4)
    )
    return SimpleNamespace(
        manifest_artifact_version_id="3" * 64,
        trials=trials,
        promotion_failures=promotion_failures,
        promoted=not promotion_failures,
    )
