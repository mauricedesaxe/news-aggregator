from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import dagster as dg
import pytest
from dagster._core.events import StepMaterializationData, StepOutputData

from romanian_news.worker import morning_report
from tests.daily_report_catalog import daily_report, seed_daily_report
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)


def test_morning_check_reports_the_day_and_pings_on_the_real_catalog(
    postgres_catalog: PostgresCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_daily_report(postgres_catalog, daily_report(date(2026, 9, 13)))
    monkeypatch.setattr(
        morning_report,
        "_now",
        lambda: datetime(2026, 9, 14, 9, 35, tzinfo=ZoneInfo("Europe/Bucharest")),
    )
    monkeypatch.setattr(
        morning_report,
        "build_and_publish_current_daily_report",
        lambda day, ref: SimpleNamespace(
            head=SimpleNamespace(version_id="9" * 64, input_time=datetime(2026, 9, 14, 4, 2))
        ),
    )
    pings: list[str] = []
    monkeypatch.setattr(morning_report, "ping_heartbeat", lambda kind: pings.append(kind))

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
    assert metadata["report_version_id"] == "9" * 64
    materialization_event = next(
        event
        for event in execution.all_node_events
        if event.event_type_value == "ASSET_MATERIALIZATION"
    )
    assert isinstance(materialization_event.event_specific_data, StepMaterializationData)
    materialization = materialization_event.event_specific_data.materialization
    assert materialization.asset_key == dg.AssetKey("daily_reports")
    assert materialization.partition == "2026-09-14"
    assert pings == ["morning_report"]
