import re
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import dagster as dg
import pytest

from romanian_news.archive.campaign import ARCHIVE_END, ARCHIVE_START
from romanian_news.catalog.archive_model_spend import ArchiveSpend
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


@pytest.fixture(autouse=True)
def _empty_archive_spend(monkeypatch) -> None:
    monkeypatch.setattr(
        retrospective_analysis,
        "read_archive_spend",
        lambda _day: ArchiveSpend(Decimal(0), Decimal(0), 0),
    )


def test_analysis_rejects_days_before_the_archive_window() -> None:
    with pytest.raises(ValueError, match="outside the one-year archive"):
        retrospective_analysis.analyze_retrospective_day(
            ARCHIVE_START - timedelta(days=1), "git:test"
        )


def test_analysis_rejects_days_after_the_archive_window() -> None:
    with pytest.raises(ValueError, match="outside the one-year archive"):
        retrospective_analysis.analyze_retrospective_day(
            ARCHIVE_END + timedelta(days=1), "git:test"
        )


def _stub_pilot_boundaries(monkeypatch) -> None:
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
    monkeypatch.setattr(
        retrospective_analysis, "read_pending_relevance_references", lambda **_kwargs: ()
    )
    for name in (
        "materialize_relevance",
        "materialize_embeddings",
        "materialize_clusters",
        "materialize_group_summaries",
        "materialize_group_sentiment",
        "materialize_daily_themes",
        "materialize_subject_assessments",
    ):
        monkeypatch.setattr(retrospective_analysis, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        retrospective_analysis,
        "publish_retrospective_daily_report",
        lambda _day, _ref: SimpleNamespace(version_id="a" * 64),
    )


def test_pilot_publishes_and_returns_report_version(monkeypatch) -> None:
    _stub_pilot_boundaries(monkeypatch)
    assert retrospective_analysis.analyze_retrospective_day(DAY, "git:test") == "a" * 64


def test_pilot_records_each_stage_duration_in_the_run_log(monkeypatch) -> None:
    _stub_pilot_boundaries(monkeypatch)
    pattern = re.compile(r"Retrospective stage day=(\S+) stage=(\S+) elapsed_seconds=([0-9.]+)")

    with dg.instance_for_test() as instance:
        result = retrospective_analysis.retrospective_analysis_pilot.execute_in_process(
            instance=instance,
            run_config={"ops": {"retrospective_analysis": {"config": {"day": DAY.isoformat()}}}},
        )
        assert result.success
        log_entries = [
            entry.user_message
            for entry in instance.all_logs(result.run_id)
            if entry.dagster_event is None and "Retrospective stage" in entry.user_message
        ]

    stages: dict[str, float] = {}
    for message in log_entries:
        match = pattern.fullmatch(message)
        assert match is not None, message
        day, stage, elapsed = match.groups()
        assert day == DAY.isoformat()
        stages[stage] = float(elapsed)

    assert set(stages) == {
        "relevance",
        "embeddings",
        "clusters",
        "group_summaries",
        "group_sentiment",
        "daily_themes",
        "subject_assessments",
        "publication",
    }
    assert all(elapsed >= 0 for elapsed in stages.values())


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


def test_over_limit_day_cannot_start_model_work_or_publish(monkeypatch) -> None:
    monkeypatch.setattr(
        retrospective_analysis,
        "read_retrospective_coverage",
        lambda _day: _coverage(("digi24", "hotnews")),
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "read_daily_article_references",
        lambda _day: SimpleNamespace(values=(object(),)),
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "read_archive_spend",
        lambda _day: ArchiveSpend(Decimal("10.01"), Decimal(0), 1),
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "materialize_relevance",
        lambda *_args, **_kwargs: pytest.fail("model work started"),
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "publish_retrospective_daily_report",
        lambda *_args: pytest.fail("report published"),
    )
    with pytest.raises(retrospective_analysis.ArchiveSpendLimitReached):
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


def test_schedule_publishes_ready_day_before_month_finishes(monkeypatch) -> None:
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
        lambda day: _coverage(("digi24", "hotnews")) if day == DAY else None,
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "read_daily_article_references",
        lambda _day: SimpleNamespace(values=(object(),) * 45),
    )

    assert retrospective_analysis.next_automated_day() == DAY

    monkeypatch.setattr(
        retrospective_analysis,
        "list_archive_daily_reports",
        lambda *_args: (SimpleNamespace(day=DAY, version_id="a" * 64),),
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "read_retrospective_coverage",
        lambda day: pytest.fail("Published day should wait for completed month")
        if day == DAY
        else None,
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


def test_large_day_runs_one_relevance_batch_before_downstream_work(monkeypatch) -> None:
    _ready_first_month(monkeypatch)
    monkeypatch.setattr(
        retrospective_analysis,
        "read_daily_article_references",
        lambda _day: SimpleNamespace(values=(object(),) * 201),
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "read_retrospective_coverage",
        lambda day: _coverage(("digi24", "hotnews")).model_copy(
            update={"captured_article_count": 201}
        )
        if day == DAY
        else None,
    )
    monkeypatch.setattr(
        retrospective_analysis, "materialize_relevance", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "read_pending_relevance_references",
        lambda **_kwargs: (object(),),
    )
    monkeypatch.setattr(
        retrospective_analysis, "materialize_embeddings", lambda *_args, **_kwargs: None
    )

    assert retrospective_analysis.analyze_retrospective_day(DAY, "git:test") is None
    monkeypatch.setattr(retrospective_analysis, "list_archive_daily_reports", lambda *_args: ())
    assert retrospective_analysis.next_automated_day() == DAY


def test_schedule_selects_earlier_large_day_before_later_day(monkeypatch) -> None:
    _ready_first_month(monkeypatch)
    following = DAY + timedelta(days=1)
    monkeypatch.setattr(retrospective_analysis, "list_archive_daily_reports", lambda *_args: ())
    monkeypatch.setattr(
        retrospective_analysis,
        "read_retrospective_coverage",
        lambda day: _coverage(("digi24", "hotnews")) if day in (DAY, following) else None,
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "read_daily_article_references",
        lambda day: SimpleNamespace(values=(object(),) * (201 if day == DAY else 45)),
    )

    assert retrospective_analysis.next_automated_day() == DAY


def test_schedule_reaches_dates_after_the_initial_ten_days(monkeypatch) -> None:
    later = date(2025, 11, 3)
    monkeypatch.setattr(retrospective_analysis, "next_page_window", lambda *_args: None)
    monkeypatch.setattr(retrospective_analysis, "next_capture_window", lambda *_args: None)
    monkeypatch.setattr(retrospective_analysis, "list_archive_daily_reports", lambda *_args: ())
    monkeypatch.setattr(
        retrospective_analysis,
        "read_retrospective_coverage",
        lambda day: _coverage(("digi24", "hotnews")) if day == later else None,
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "read_daily_article_references",
        lambda _day: SimpleNamespace(values=(object(),) * 45),
    )

    assert retrospective_analysis.next_automated_day() == later


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
