from __future__ import annotations

from datetime import date, datetime, timedelta

import dagster as dg
from dagster._core.definitions.asset_checks.asset_check_evaluation import (
    AssetCheckEvaluationTargetMaterializationData,
)

from romanian_news import BUCHAREST
from romanian_news.worker.assets import (
    DAILY_FRESHNESS,
    DAILY_PARTITIONS,
    EAGER,
    SHADOW_FRESHNESS,
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


@dg.asset(partitions_def=DAILY_PARTITIONS)
def shadow_source() -> None:
    pass


@dg.asset(
    deps=[dg.AssetDep(shadow_source, partition_mapping=dg.TimeWindowPartitionMapping())],
    partitions_def=DAILY_PARTITIONS,
    code_version="v1",
    automation_condition=SHADOW_FRESHNESS,
)
def shadow_report() -> None:
    pass


DEFS = dg.Definitions(assets=[daily_reports, weekly_reports])
SHADOW_DEFS = dg.Definitions(assets=[shadow_source, shadow_report])
EVALUATION_TIME = datetime.fromisoformat("2026-09-24T10:00:00+03:00")


@dg.asset(partitions_def=DAILY_PARTITIONS)
def article_snapshot() -> None:
    pass


@dg.asset(
    deps=[article_snapshot],
    partitions_def=DAILY_PARTITIONS,
    automation_condition=DAILY_FRESHNESS,
)
def recovered_report() -> None:
    pass


DAILY_DEFS = dg.Definitions(assets=[article_snapshot, recovered_report])


@dg.asset(
    name="recovered_report",
    deps=[article_snapshot],
    partitions_def=DAILY_PARTITIONS,
    automation_condition=EAGER,
)
def legacy_report() -> None:
    pass


LEGACY_DAILY_DEFS = dg.Definitions(assets=[article_snapshot, legacy_report])


def test_daily_freshness_limits_automatic_recovery_to_recent_partitions() -> None:
    old_day = "2026-09-10"
    latest_day = "2026-09-24"
    with dg.instance_for_test() as instance:
        _materialize(instance, "article_snapshot", old_day)
        _materialize(instance, "article_snapshot", latest_day)

        first = _evaluate_daily(instance)
        assert first.get_requested_partitions(dg.AssetKey("recovered_report")) == {latest_day}

        _materialize(instance, "recovered_report", old_day)
        _materialize(instance, "recovered_report", latest_day)
        settled = _evaluate_daily(instance, cursor=first.cursor)
        assert settled.get_requested_partitions(dg.AssetKey("recovered_report")) == set()

        _materialize(instance, "article_snapshot", old_day)
        changed = _evaluate_daily(instance, cursor=settled.cursor)
        assert changed.get_requested_partitions(dg.AssetKey("recovered_report")) == set()


def test_daily_freshness_covers_exactly_three_calendar_days() -> None:
    with dg.instance_for_test() as instance:
        for day in ("2026-09-27", "2026-09-28", "2026-09-29", "2026-09-30"):
            _materialize(instance, "article_snapshot", day)

        evaluation = _evaluate_daily(
            instance,
            evaluation_time=datetime.fromisoformat("2026-09-30T12:00:00+03:00"),
        )
        assert evaluation.get_requested_partitions(dg.AssetKey("recovered_report")) == {
            "2026-09-28",
            "2026-09-29",
            "2026-09-30",
        }


def test_daily_freshness_retries_recent_missing_output_hourly() -> None:
    old_day = "2026-09-24"
    with dg.instance_for_test() as instance:
        _materialize(instance, "article_snapshot", old_day)

        first = _evaluate_daily(instance, evaluation_time=EVALUATION_TIME + timedelta(minutes=1))
        assert first.get_requested_partitions(dg.AssetKey("recovered_report")) == {old_day}

        too_soon = _evaluate_daily(
            instance,
            cursor=first.cursor,
            evaluation_time=EVALUATION_TIME + timedelta(minutes=6),
        )
        assert too_soon.get_requested_partitions(dg.AssetKey("recovered_report")) == set()

        retry = _evaluate_daily(
            instance,
            cursor=too_soon.cursor,
            evaluation_time=EVALUATION_TIME + timedelta(hours=1, minutes=6),
        )
        assert retry.get_requested_partitions(dg.AssetKey("recovered_report")) == {old_day}


def test_daily_freshness_keeps_hourly_retry_after_upstream_run_finishes() -> None:
    hour = datetime.now(BUCHAREST).replace(minute=0, second=0, microsecond=0)
    hour += timedelta(hours=1)
    day = (hour - timedelta(seconds=10)).date().isoformat()
    source_job = dg.Definitions(
        assets=[article_snapshot],
        jobs=[
            dg.define_asset_job("source_job", selection=dg.AssetSelection.assets(article_snapshot))
        ],
    ).resolve_job_def("source_job")

    with dg.instance_for_test() as instance:
        _materialize(instance, "article_snapshot", day)
        source_run = instance.create_run_for_job(
            source_job,
            status=dg.DagsterRunStatus.STARTED,
            tags={"dagster/partition": day},
            asset_selection={dg.AssetKey("article_snapshot")},
        )

        before_hour = _evaluate_daily(instance, evaluation_time=hour - timedelta(seconds=10))
        assert before_hour.get_requested_partitions(dg.AssetKey("recovered_report")) == set()

        at_hour = _evaluate_daily(
            instance,
            cursor=before_hour.cursor,
            evaluation_time=hour + timedelta(seconds=10),
        )
        assert at_hour.get_requested_partitions(dg.AssetKey("recovered_report")) == set()

        _ = instance.report_run_failed(source_run)
        after_run = _evaluate_daily(
            instance,
            cursor=at_hour.cursor,
            evaluation_time=hour + timedelta(minutes=1),
        )
        assert after_run.get_requested_partitions(dg.AssetKey("recovered_report")) == {day}


def test_daily_freshness_keeps_upstream_update_until_run_finishes() -> None:
    day = "2026-09-24"
    report_key = dg.AssetKey("report_with_two_sources")

    @dg.asset(partitions_def=DAILY_PARTITIONS)
    def busy_source() -> None:
        pass

    @dg.asset(
        deps=[article_snapshot, busy_source],
        partitions_def=DAILY_PARTITIONS,
        automation_condition=DAILY_FRESHNESS,
    )
    def report_with_two_sources() -> None:
        pass

    definitions = dg.Definitions(assets=[article_snapshot, busy_source, report_with_two_sources])
    source_job = dg.Definitions(
        assets=[busy_source],
        jobs=[dg.define_asset_job("source_job", selection=dg.AssetSelection.assets(busy_source))],
    ).resolve_job_def("source_job")

    with dg.instance_for_test() as instance:
        _materialize(instance, "article_snapshot", day)
        _materialize(instance, "busy_source", day)
        _materialize(instance, "report_with_two_sources", day)
        settled = dg.evaluate_automation_conditions(
            definitions,
            instance,
            asset_selection=dg.AssetSelection.assets(report_with_two_sources),
            evaluation_time=EVALUATION_TIME,
        )
        assert settled.get_requested_partitions(report_key) == set()

        source_run = instance.create_run_for_job(
            source_job,
            status=dg.DagsterRunStatus.STARTED,
            tags={"dagster/partition": day},
            asset_selection={dg.AssetKey("busy_source")},
        )
        _materialize(instance, "article_snapshot", day)
        while_running = dg.evaluate_automation_conditions(
            definitions,
            instance,
            asset_selection=dg.AssetSelection.assets(report_with_two_sources),
            evaluation_time=EVALUATION_TIME + timedelta(minutes=1),
            cursor=settled.cursor,
        )
        assert while_running.get_requested_partitions(report_key) == set()

        _ = instance.report_run_failed(source_run)
        after_run = dg.evaluate_automation_conditions(
            definitions,
            instance,
            asset_selection=dg.AssetSelection.assets(report_with_two_sources),
            evaluation_time=EVALUATION_TIME + timedelta(minutes=2),
            cursor=while_running.cursor,
        )
        assert after_run.get_requested_partitions(report_key) == {day}

        _materialize(instance, "report_with_two_sources", day)
        settled_again = dg.evaluate_automation_conditions(
            definitions,
            instance,
            asset_selection=dg.AssetSelection.assets(report_with_two_sources),
            evaluation_time=EVALUATION_TIME + timedelta(minutes=3),
            cursor=after_run.cursor,
        )
        assert settled_again.get_requested_partitions(report_key) == set()


def test_daily_freshness_retries_failed_check_after_dependency_finishes() -> None:
    hour = datetime.now(BUCHAREST).replace(minute=0, second=0, microsecond=0)
    hour += timedelta(hours=1)
    day = (hour - timedelta(seconds=10)).date().isoformat()

    @dg.asset(partitions_def=DAILY_PARTITIONS)
    def busy_source() -> None:
        pass

    @dg.asset(
        deps=[busy_source],
        partitions_def=DAILY_PARTITIONS,
        automation_condition=DAILY_FRESHNESS,
    )
    def checked_report() -> None:
        pass

    @dg.asset_check(asset=checked_report, name="complete", blocking=True)
    def report_complete() -> dg.AssetCheckResult:
        return dg.AssetCheckResult(passed=False)

    definitions = dg.Definitions(
        assets=[busy_source, checked_report], asset_checks=[report_complete]
    )
    source_job = dg.Definitions(
        assets=[busy_source],
        jobs=[dg.define_asset_job("source_job", selection=dg.AssetSelection.assets(busy_source))],
    ).resolve_job_def("source_job")
    report_key = dg.AssetKey("checked_report")

    with dg.instance_for_test() as instance:
        _materialize(instance, "busy_source", day)
        _materialize(instance, "checked_report", day)
        materialization = instance.fetch_materializations(
            dg.AssetRecordsFilter(asset_key=report_key, asset_partitions=[day]), limit=1
        ).records[0]
        _ = instance.report_runless_asset_event(
            dg.AssetCheckEvaluation(
                asset_key=report_key,
                check_name="complete",
                passed=False,
                partition=day,
                blocking=True,
                target_materialization_data=AssetCheckEvaluationTargetMaterializationData(
                    storage_id=materialization.storage_id,
                    run_id=materialization.event_log_entry.run_id,
                    timestamp=materialization.event_log_entry.timestamp,
                ),
            )
        )
        source_run = instance.create_run_for_job(
            source_job,
            status=dg.DagsterRunStatus.STARTED,
            tags={"dagster/partition": day},
            asset_selection={dg.AssetKey("busy_source")},
        )

        before_hour = dg.evaluate_automation_conditions(
            definitions,
            instance,
            asset_selection=dg.AssetSelection.assets(checked_report),
            evaluation_time=hour - timedelta(seconds=10),
        )
        at_hour = dg.evaluate_automation_conditions(
            definitions,
            instance,
            asset_selection=dg.AssetSelection.assets(checked_report),
            evaluation_time=hour + timedelta(seconds=10),
            cursor=before_hour.cursor,
        )
        assert day not in at_hour.get_requested_partitions(report_key)

        _ = instance.report_run_failed(source_run)
        after_run = dg.evaluate_automation_conditions(
            definitions,
            instance,
            asset_selection=dg.AssetSelection.assets(checked_report),
            evaluation_time=hour + timedelta(minutes=1),
            cursor=at_hour.cursor,
        )
        assert day in after_run.get_requested_partitions(report_key)


def test_recovery_condition_picks_up_days_missed_by_eager_cursor() -> None:
    old_day = "2026-09-24"
    with dg.instance_for_test() as instance:
        _materialize(instance, "article_snapshot", old_day)
        legacy = dg.evaluate_automation_conditions(
            LEGACY_DAILY_DEFS,
            instance,
            asset_selection=dg.AssetSelection.assets(legacy_report),
            evaluation_time=EVALUATION_TIME,
        )
        assert legacy.get_requested_partitions(dg.AssetKey("recovered_report")) == set()

        recovery = _evaluate_daily(instance, cursor=legacy.cursor)
        assert recovery.get_requested_partitions(dg.AssetKey("recovered_report")) == {old_day}


def _evaluate_daily(instance: dg.DagsterInstance, *, cursor=None, evaluation_time=EVALUATION_TIME):
    return dg.evaluate_automation_conditions(
        DAILY_DEFS,
        instance,
        asset_selection=dg.AssetSelection.assets(recovered_report),
        evaluation_time=evaluation_time,
        cursor=cursor,
    )


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


def test_shadow_freshness_only_tracks_the_latest_time_window() -> None:
    with dg.instance_for_test() as instance:
        for day in ("2026-09-10", "2026-09-20"):
            _materialize(instance, "shadow_source", day)

        first = _evaluate_shadow(instance)
        assert first.get_requested_partitions(dg.AssetKey("shadow_report")) == {"2026-09-20"}

        _materialize(instance, "shadow_report", "2026-09-20")
        settled = _evaluate_shadow(instance, cursor=first.cursor)
        assert settled.get_requested_partitions(dg.AssetKey("shadow_report")) == set()

        _materialize(instance, "shadow_source", "2026-09-10")
        old_changed = _evaluate_shadow(instance, cursor=settled.cursor)
        assert old_changed.get_requested_partitions(dg.AssetKey("shadow_report")) == set()

        _materialize(instance, "shadow_source", "2026-09-20")
        recent_changed = _evaluate_shadow(instance, cursor=old_changed.cursor)
        assert recent_changed.get_requested_partitions(dg.AssetKey("shadow_report")) == {
            "2026-09-20"
        }


def _evaluate_shadow(instance: dg.DagsterInstance, *, cursor=None):
    return dg.evaluate_automation_conditions(
        SHADOW_DEFS,
        instance,
        asset_selection=dg.AssetSelection.assets(shadow_report),
        evaluation_time=EVALUATION_TIME,
        cursor=cursor,
    )


def _evaluate(instance: dg.DagsterInstance, *, cursor=None):
    return dg.evaluate_automation_conditions(
        DEFS,
        instance,
        asset_selection=dg.AssetSelection.assets(weekly_reports),
        evaluation_time=EVALUATION_TIME,
        cursor=cursor,
    )
