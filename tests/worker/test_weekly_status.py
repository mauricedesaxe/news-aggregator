from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from romanian_news.worker import weekly_status


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
