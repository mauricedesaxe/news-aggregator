from datetime import UTC, date, datetime
from types import SimpleNamespace

import dagster as dg
import pytest

from romanian_news.reports import RetrospectiveCoverage, RetrospectiveDailyReport
from romanian_news.worker import retrospective_analysis
from romanian_news.worker.definitions import defs

DAY = date(2025, 9, 27)


def _coverage(outlets: tuple[str, ...]) -> RetrospectiveCoverage:
    return RetrospectiveCoverage(
        capture_started_at=datetime(2026, 9, 27, 10, tzinfo=UTC),
        capture_ended_at=datetime(2026, 9, 27, 11, tzinfo=UTC),
        included_outlets=outlets,
        discovered_url_count=50,
        verified_page_count=50,
        captured_article_count=45,
        coverage_note="Verified publisher pages only.",
    )


def test_pilot_runs_analysis_in_report_input_order(monkeypatch) -> None:
    monkeypatch.setattr(
        retrospective_analysis,
        "read_retrospective_coverage",
        lambda _day: _coverage(("digi24", "hotnews")),
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "read_daily_article_references",
        lambda _day: SimpleNamespace(values=(object(),) * 45),
    )
    called = []
    for name in (
        "materialize_relevance",
        "materialize_embeddings",
        "materialize_clusters",
        "materialize_group_summaries",
        "materialize_group_sentiment",
        "materialize_daily_themes",
        "materialize_subject_assessments",
    ):
        monkeypatch.setattr(
            retrospective_analysis,
            name,
            lambda _day, _ref, stage=name: called.append(stage),
        )

    def publish(_day, _ref):
        called.append("publish")
        return SimpleNamespace(version_id="a" * 64)

    monkeypatch.setattr(retrospective_analysis, "publish_retrospective_daily_report", publish)
    assert retrospective_analysis.analyze_retrospective_day(DAY, "git:test") == "a" * 64
    assert called == [
        "materialize_relevance",
        "materialize_embeddings",
        "materialize_clusters",
        "materialize_group_summaries",
        "materialize_group_sentiment",
        "materialize_daily_themes",
        "materialize_subject_assessments",
        "publish",
    ]


def test_pilot_rejects_sparse_coverage_before_model_work(monkeypatch) -> None:
    monkeypatch.setattr(
        retrospective_analysis,
        "read_retrospective_coverage",
        lambda _day: _coverage(("hotnews",)),
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "read_daily_article_references",
        lambda _day: pytest.fail("article inputs should not load for one outlet"),
    )
    with pytest.raises(ValueError, match="two outlets"):
        retrospective_analysis.analyze_retrospective_day(DAY, "git:test")


def _ready_first_month(monkeypatch) -> None:
    later = (date(2025, 10, 1), date(2025, 10, 31))
    monkeypatch.setattr(retrospective_analysis, "next_page_window", lambda *_args: later)
    monkeypatch.setattr(retrospective_analysis, "next_capture_window", lambda *_args: later)
    monkeypatch.setattr(
        retrospective_analysis,
        "read_retrospective_coverage",
        lambda day: _coverage(("digi24", "hotnews")) if day == DAY else None,
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "read_daily_article_references",
        lambda _day: SimpleNamespace(values=(object(),) * 45),
    )


def test_schedule_waits_for_both_outlets_to_finish_the_month(monkeypatch) -> None:
    monkeypatch.setattr(
        retrospective_analysis,
        "next_page_window",
        lambda outlet, *_args: (DAY, date(2025, 9, 30)) if outlet == "digi24" else None,
    )
    monkeypatch.setattr(retrospective_analysis, "next_capture_window", lambda *_args: None)
    monkeypatch.setattr(retrospective_analysis, "list_archive_daily_reports", lambda *_args: ())
    monkeypatch.setattr(
        retrospective_analysis,
        "read_retrospective_coverage",
        lambda _day: pytest.fail("No day is ready while September page checks remain"),
    )

    assert retrospective_analysis.next_automated_day() is None


def test_schedule_selects_a_completed_day_and_refreshes_new_captures(monkeypatch) -> None:
    _ready_first_month(monkeypatch)
    monkeypatch.setattr(retrospective_analysis, "list_archive_daily_reports", lambda *_args: ())
    assert retrospective_analysis.next_automated_day() == DAY

    monkeypatch.setattr(
        retrospective_analysis,
        "list_archive_daily_reports",
        lambda *_args: (SimpleNamespace(day=DAY, version_id="a" * 64),),
    )
    old = RetrospectiveDailyReport(
        day=DAY,
        accepted_article_count=0,
        theme_count=0,
        group_count=0,
        sections=(),
        retrospective=_coverage(("digi24", "hotnews")),
    )
    monkeypatch.setattr(retrospective_analysis, "read_daily_report_version", lambda _id: old)
    assert retrospective_analysis.next_automated_day() is None

    newer = _coverage(("digi24", "hotnews")).model_copy(update={"captured_article_count": 46})
    monkeypatch.setattr(
        retrospective_analysis,
        "read_retrospective_coverage",
        lambda day: newer if day == DAY else None,
    )
    assert retrospective_analysis.next_automated_day() == DAY


def test_schedule_surfaces_days_above_its_article_limit(monkeypatch) -> None:
    _ready_first_month(monkeypatch)
    monkeypatch.setattr(retrospective_analysis, "list_archive_daily_reports", lambda *_args: ())
    monkeypatch.setattr(
        retrospective_analysis,
        "read_daily_article_references",
        lambda _day: SimpleNamespace(values=(object(),) * 201),
    )

    with pytest.raises(ValueError, match="above the 200-article batch limit"):
        retrospective_analysis.next_automated_day()


def test_schedule_launches_only_one_retrospective_day(monkeypatch) -> None:
    monkeypatch.setattr(retrospective_analysis, "next_automated_day", lambda: DAY)
    with dg.instance_for_test() as instance:
        with dg.build_schedule_context(
            instance=instance,
            scheduled_execution_time=datetime(2026, 9, 27, 19, 5, tzinfo=UTC),
            repository_def=defs.get_repository_def(),
        ) as context:
            evaluation = retrospective_analysis.scheduled_retrospective_analysis.evaluate_tick(
                context
            )
    assert evaluation.run_requests is not None
    assert len(evaluation.run_requests) == 1
    request = evaluation.run_requests[0]
    assert request.tags["news/archive_day"] == DAY.isoformat()
    assert request.run_config == {
        "ops": {"retrospective_analysis": {"config": {"day": DAY.isoformat()}}}
    }


def test_schedule_waits_while_a_report_is_running(monkeypatch) -> None:
    monkeypatch.setattr(
        retrospective_analysis,
        "next_automated_day",
        lambda: pytest.fail("Active run should stop candidate selection"),
    )
    with dg.instance_for_test() as instance:
        instance.create_run_for_job(
            defs.resolve_job_def("retrospective_analysis_pilot"),
            run_config={"ops": {"retrospective_analysis": {"config": {"day": DAY.isoformat()}}}},
            status=dg.DagsterRunStatus.STARTED,
        )
        with dg.build_schedule_context(
            instance=instance,
            scheduled_execution_time=datetime(2026, 9, 27, 19, 35, tzinfo=UTC),
            repository_def=defs.get_repository_def(),
        ) as context:
            evaluation = retrospective_analysis.scheduled_retrospective_analysis.evaluate_tick(
                context
            )

    assert evaluation.run_requests == []
    assert evaluation.skip_message is not None
