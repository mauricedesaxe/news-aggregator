from __future__ import annotations

from datetime import date, datetime, timedelta

import dagster as dg

from romanian_news.worker.assets import (
    DAILY_PARTITIONS,
    WEEKLY_FRESHNESS,
    WEEKLY_PARTITIONS,
)


@dg.asset(partitions_def=DAILY_PARTITIONS)
def daily_reports() -> None:
    pass


@dg.asset(
    deps=[dg.AssetDep(daily_reports, partition_mapping=dg.TimeWindowPartitionMapping())],
    partitions_def=WEEKLY_PARTITIONS,
    code_version="v1",
    automation_condition=WEEKLY_FRESHNESS,
)
def weekly_reports() -> None:
    pass


DEFS = dg.Definitions(assets=[daily_reports, weekly_reports])
EVALUATION_TIME = datetime.fromisoformat("2026-09-24T10:00:00+03:00")


def test_weekly_freshness_recovers_old_weeks_and_tracks_daily_changes() -> None:
    with dg.instance_for_test() as instance:
        for monday in (date(2026, 8, 3), date(2026, 8, 10)):
            for offset in range(7):
                _materialize(
                    instance, "daily_reports", (monday + timedelta(days=offset)).isoformat()
                )
        for offset in range(6):
            _materialize(
                instance, "daily_reports", (date(2026, 8, 17) + timedelta(days=offset)).isoformat()
            )
        _materialize(instance, "weekly_reports", "2026-08-03")

        first = _evaluate(instance)
        assert first.get_requested_partitions(dg.AssetKey("weekly_reports")) == {
            "2026-08-03",
            "2026-08-10",
        }

        _materialize(instance, "weekly_reports", "2026-08-03")
        _materialize(instance, "weekly_reports", "2026-08-10")
        settled = _evaluate(instance, cursor=first.cursor)
        assert settled.get_requested_partitions(dg.AssetKey("weekly_reports")) == set()

        _materialize(instance, "daily_reports", "2026-08-05")
        changed = _evaluate(instance, cursor=settled.cursor)
        assert changed.get_requested_partitions(dg.AssetKey("weekly_reports")) == {"2026-08-03"}


def _materialize(instance: dg.DagsterInstance, name: str, partition: str) -> None:
    instance.report_runless_asset_event(
        dg.AssetMaterialization(asset_key=name, partition=partition)
    )


def _evaluate(instance: dg.DagsterInstance, *, cursor=None):
    return dg.evaluate_automation_conditions(
        DEFS,
        instance,
        asset_selection=dg.AssetSelection.assets(weekly_reports),
        evaluation_time=EVALUATION_TIME,
        cursor=cursor,
    )
