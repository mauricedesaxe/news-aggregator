from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import dagster as dg
import pytest
from dagster._core.events import StepOutputData

from romanian_news.worker import morning_report


def _freeze_morning(monkeypatch, pings: list[str]) -> SimpleNamespace:
    monkeypatch.setattr(
        morning_report,
        "_now",
        lambda: datetime(2026, 9, 14, 9, 35, tzinfo=ZoneInfo("Europe/Bucharest")),
    )
    head = SimpleNamespace(version_id="9" * 64, input_time=datetime(2026, 9, 14, 4, 2))
    monkeypatch.setattr(
        morning_report,
        "build_and_publish_current_daily_report",
        lambda day, ref: SimpleNamespace(head=head),
    )
    monkeypatch.setattr(morning_report, "read_daily_report_reference", lambda day: None)
    monkeypatch.setattr(morning_report, "ping_heartbeat", lambda kind: pings.append(kind))
    return head


def test_morning_check_verifies_both_days_and_pings(monkeypatch) -> None:
    pings: list[str] = []
    head = _freeze_morning(monkeypatch, pings)

    execution = morning_report.morning_report_check.execute_in_process()

    assert execution.success
    assert execution.output_for_node("morning_report_check_op") == "2026-09-14"
    output_event = next(
        event for event in execution.all_node_events if event.event_type_value == "STEP_OUTPUT"
    )
    event_data = output_event.event_specific_data
    assert isinstance(event_data, StepOutputData)
    metadata = {name: value.value for name, value in event_data.metadata.items()}
    assert metadata["day"] == "2026-09-14"
    assert metadata["report_version_id"] == head.version_id
    assert pings == ["morning_report"]


def test_morning_check_does_not_ping_when_yesterdays_edition_is_missing(monkeypatch) -> None:
    ensure_calls: list[object] = []
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
