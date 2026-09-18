from pathlib import Path

import dagster as dg

from romanian_news.worker.recorded_construction import (
    RecordedDailyConstruction,
    build_recorded_construction,
)

FIXTURE = Path(__file__).parent / "fixtures" / "recorded_daily_construction.json"


@dg.asset
def recorded_construction_asset() -> dg.MaterializeResult[object]:
    recorded = RecordedDailyConstruction.model_validate_json(FIXTURE.read_text(), strict=True)
    output = build_recorded_construction(recorded)
    return dg.MaterializeResult[object](
        metadata={
            "cluster_digest": output.cluster_content_digest,
            "theme_digest": output.daily_themes.content_digest,
            "daily_digest": output.daily_report.content_digest,
            "weekly_digest": output.weekly_report.content_digest,
        }
    )


def test_recorded_inputs_rebuild_reports_and_clusters() -> None:
    recorded = RecordedDailyConstruction.model_validate_json(FIXTURE.read_text(), strict=True)

    first = build_recorded_construction(recorded)
    second = build_recorded_construction(recorded)

    assert first == second
    assert first.cluster_content_digest == recorded.expected_cluster_content_digest
    assert first.daily_themes.content_digest == recorded.expected_daily_theme_content_digest
    assert first.daily_report.content_digest == recorded.expected_daily_report_content_digest
    assert first.weekly_report.content_digest == recorded.expected_weekly_report_content_digest
    assert first.daily_report.report.day.isoformat() == "2026-08-31"
    assert tuple(day.report.day.isoformat() for day in first.weekly_report.report.days) == (
        "2026-08-31",
    )


def test_dagster_materializes_recorded_construction() -> None:
    result = dg.materialize([recorded_construction_asset])

    assert result.success
    materialization = result.asset_materializations_for_node("recorded_construction_asset")[0]
    assert materialization.metadata["cluster_digest"].value
    assert materialization.metadata["theme_digest"].value
    assert materialization.metadata["daily_digest"].value
    assert materialization.metadata["weekly_digest"].value
