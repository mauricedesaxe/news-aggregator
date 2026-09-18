from types import SimpleNamespace

import dagster as dg

from romanian_news.evaluation_projection import FreshThemeMetrics
from romanian_news.worker import theme_comparison
from romanian_news.worker.definitions import defs
from romanian_news.worker.theme_comparison import FreshThemeComparisonSummary


def test_manual_theme_comparison_uses_idempotent_implementation_identity(monkeypatch) -> None:
    release = object()
    seen = {}
    monkeypatch.setattr(theme_comparison, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(
        theme_comparison,
        "load_news_evaluation_release",
        lambda _content: release,
    )
    monkeypatch.setattr(
        theme_comparison,
        "compare_theme_policies",
        lambda loaded, plan: seen.update(loaded=loaded, plan=plan) or _comparison(plan),
    )
    monkeypatch.setattr(theme_comparison, "IMPLEMENTATION_REF", "git:test")
    monkeypatch.setattr(
        theme_comparison,
        "flush_langfuse_traces",
        lambda: seen.update(flushed=True),
    )

    result = theme_comparison.fresh_theme_comparison.execute_in_process()

    assert result.success
    assert seen["loaded"] is release
    assert seen["plan"].implementation_ref == "git:test"
    assert seen.get("flushed") is True
    output = result.output_for_node("fresh_theme_comparison_op")
    assert isinstance(output, FreshThemeComparisonSummary)
    assert output.trial_count == 3
    assert output.promoted
    assert output.trials[0].candidate.metrics.complete_report_usefulness == 1.0


def test_manual_theme_comparison_fails_on_promotion_failure(monkeypatch) -> None:
    monkeypatch.setattr(theme_comparison, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(theme_comparison, "load_news_evaluation_release", lambda _content: object())
    monkeypatch.setattr(
        theme_comparison,
        "compare_theme_policies",
        lambda _release, plan: _comparison(plan, failures=("trial 2 failed",)),
    )
    flushed = []
    monkeypatch.setattr(theme_comparison, "flush_langfuse_traces", lambda: flushed.append(True))

    result = theme_comparison.fresh_theme_comparison.execute_in_process(raise_on_error=False)

    assert result.success is False
    failure = result.failure_data_for_node("fresh_theme_comparison_op")
    assert failure is not None and failure.error is not None
    assert "trial 2 failed" in failure.error.message
    assert flushed == [True]


def test_manual_theme_comparison_is_registered_without_automation() -> None:
    assert defs.get_job_def("fresh_theme_comparison") is not None
    assert theme_comparison.fresh_theme_comparison_op.pool == "news_model"
    assert all(
        schedule.job_name != "fresh_theme_comparison"
        for schedule in defs.schedules or ()
        if isinstance(schedule, dg.ScheduleDefinition)
    )
    assert all(sensor.name != "fresh_theme_comparison" for sensor in defs.sensors or ())


def _comparison(plan, *, failures=()):
    baseline_metrics = FreshThemeMetrics(
        must_link_recall=0.0,
        must_separate_preservation=1.0,
        singleton_rate=1.0,
        complete_report_usefulness=0.0,
    )
    candidate_metrics = FreshThemeMetrics(
        must_link_recall=1.0,
        must_separate_preservation=1.0,
        singleton_rate=0.5,
        complete_report_usefulness=1.0,
    )
    trials = tuple(
        SimpleNamespace(
            ordinal=ordinal,
            implementation_ref=implementation_ref,
            baseline=SimpleNamespace(
                policy_id="daily-reader-themes-sparse-v2",
                policy_digest="1" * 64,
                metrics=baseline_metrics,
                experiment_url="https://example.test/baseline",
            ),
            candidate=SimpleNamespace(
                policy_id="daily-reader-subjects-v3",
                policy_digest="2" * 64,
                metrics=candidate_metrics,
                experiment_url="https://example.test/candidate",
            ),
            promotion_failures=failures,
            passed=not failures,
        )
        for ordinal, implementation_ref in enumerate(plan.implementation_refs, start=1)
    )
    return SimpleNamespace(
        manifest_artifact_version_id="3" * 64,
        trials=trials,
        promotion_failures=failures,
        promoted=not failures,
    )
