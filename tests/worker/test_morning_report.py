from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import dagster as dg
import pytest

from romanian_news.worker import morning_report


def _run_op(monkeypatch, *, ensure_calls, reference_calls, pings) -> None:
    monkeypatch.setattr(
        morning_report,
        "_now",
        lambda: datetime(2026, 9, 14, 9, 35, tzinfo=ZoneInfo("Europe/Bucharest")),
    )

    def ensure(day, ref):
        ensure_calls.append((day, ref))
        return SimpleNamespace(
            head=SimpleNamespace(version_id="9" * 64, input_time=datetime(2026, 9, 14, 4, 2))
        )

    monkeypatch.setattr(morning_report, "build_and_publish_current_daily_report", ensure)
    monkeypatch.setattr(
        morning_report,
        "read_daily_report_reference",
        lambda day: reference_calls.append(day),
    )
    monkeypatch.setattr(morning_report, "ping_heartbeat", lambda kind: pings.append(kind))
    morning_report.morning_report_check_op(dg.build_op_context())


def test_morning_check_verifies_both_days_and_pings(monkeypatch) -> None:
    ensure_calls: list[tuple[object, object]] = []
    reference_calls: list[object] = []
    pings: list[str] = []

    _run_op(monkeypatch, ensure_calls=ensure_calls, reference_calls=reference_calls, pings=pings)

    assert ensure_calls == [(datetime(2026, 9, 14).date(), morning_report.IMPLEMENTATION_REF)]
    assert reference_calls == [datetime(2026, 9, 13).date()]
    assert pings == ["morning_report"]


def test_morning_check_does_not_ping_when_yesterdays_edition_is_missing(monkeypatch) -> None:
    ensure_calls: list[tuple[object, object]] = []
    pings: list[str] = []

    def missing(_day: object) -> object:
        raise ValueError("No current artifact exists: news:daily:2026-09-13")

    monkeypatch.setattr(
        morning_report,
        "_now",
        lambda: datetime(2026, 9, 14, 9, 35, tzinfo=ZoneInfo("Europe/Bucharest")),
    )
    monkeypatch.setattr(
        morning_report,
        "build_and_publish_current_daily_report",
        lambda day, ref: ensure_calls.append((day, ref)),
    )
    monkeypatch.setattr(morning_report, "read_daily_report_reference", missing)
    monkeypatch.setattr(morning_report, "ping_heartbeat", lambda kind: pings.append(kind))

    with pytest.raises(ValueError):
        morning_report.morning_report_check_op(dg.build_op_context())

    assert len(ensure_calls) == 1
    assert pings == []
