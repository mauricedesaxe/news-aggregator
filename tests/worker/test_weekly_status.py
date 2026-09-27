from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import dagster as dg

from romanian_news.worker import weekly_status
from romanian_news.worker.definitions import defs


def test_recent_completed_weeks_exclude_current_week() -> None:
    assert weekly_status.recent_completed_week_starts(date(2026, 9, 27), 3) == (
        date(2026, 9, 14),
        date(2026, 9, 7),
        date(2026, 8, 31),
    )


def test_refresh_skips_unchanged_inputs(monkeypatch) -> None:
    inputs = SimpleNamespace(available_days=6, policy="policy", days=("exact",))
    current = SimpleNamespace(policy="policy", days=("exact",))
    monkeypatch.setattr(weekly_status, "read_week_input", lambda _week: inputs)
    monkeypatch.setattr(weekly_status, "read_weekly_status", lambda _week: ("v" * 64, current))
    monkeypatch.setattr(
        weekly_status,
        "load_week_reports",
        lambda _inputs: (_ for _ in ()).throw(AssertionError("Should not load reports")),
    )

    assert weekly_status.refresh_weekly_status(date(2026, 9, 14), "implementation") == "current"


def test_refresh_requires_five_daily_reports(monkeypatch) -> None:
    inputs = SimpleNamespace(available_days=4)
    monkeypatch.setattr(weekly_status, "read_week_input", lambda _week: inputs)
    assert weekly_status.refresh_weekly_status(date(2026, 9, 14), "implementation") == (
        "insufficient"
    )


def test_historical_schedule_selects_first_week_with_five_reports(monkeypatch) -> None:
    first_ready = date(2025, 9, 29)
    monkeypatch.setattr(
        weekly_status,
        "read_week_input",
        lambda week: SimpleNamespace(available_days=5 if week == first_ready else 0),
    )
    monkeypatch.setattr(
        weekly_status,
        "read_weekly_status",
        lambda _week: (_ for _ in ()).throw(weekly_status.WeeklyStatusNotFound()),
    )
    assert weekly_status.next_historical_week(date(2026, 9, 27)) == first_ready

    with dg.instance_for_test() as instance:
        with dg.build_schedule_context(
            instance=instance,
            scheduled_execution_time=datetime(2026, 9, 27, 20, 20, tzinfo=UTC),
            repository_def=defs.get_repository_def(),
        ) as context:
            evaluation = weekly_status.scheduled_historical_weekly_status.evaluate_tick(context)
    assert evaluation.run_requests is not None
    assert len(evaluation.run_requests) == 1
    request = evaluation.run_requests[0]
    assert request.tags["news/archive_week"] == first_ready.isoformat()
    assert request.run_config == {
        "ops": {"refresh_weekly_statuses": {"config": {"week_starts": ["2025-09-29"]}}}
    }


def test_historical_schedule_skips_unchanged_week(monkeypatch) -> None:
    ready = date(2025, 9, 29)
    inputs = SimpleNamespace(available_days=5, policy="policy", days=("exact",))
    monkeypatch.setattr(
        weekly_status,
        "read_week_input",
        lambda week: inputs if week == ready else SimpleNamespace(available_days=0),
    )
    monkeypatch.setattr(
        weekly_status,
        "read_weekly_status",
        lambda _week: ("v" * 64, SimpleNamespace(policy="policy", days=("exact",))),
    )
    assert weekly_status.next_historical_week(date(2026, 9, 27)) is None
