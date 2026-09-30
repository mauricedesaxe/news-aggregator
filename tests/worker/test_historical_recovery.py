from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import dagster as dg

from romanian_news.worker import historical_recovery as recovery
from romanian_news.worker.definitions import defs

NOW = datetime.fromisoformat("2026-09-30T12:00:00+00:00")


def _materialize(
    instance: dg.DagsterInstance,
    asset: str,
    day: str,
    *,
    parent: str | None = None,
    pointer: int | None = None,
    pointers: dict[str, int] | None = None,
) -> dg.EventLogRecord:
    source_pointers = dict(pointers or {})
    if parent is not None and pointer is not None:
        source_pointers[parent] = pointer
    tags = {
        f"dagster/input_event_pointer/{source}": str(storage_id)
        for source, storage_id in source_pointers.items()
    }
    instance.report_runless_asset_event(
        dg.AssetMaterialization(asset_key=asset, partition=day, tags=tags)
    )
    return instance.fetch_materializations(
        dg.AssetRecordsFilter(asset_key=dg.AssetKey(asset), asset_partitions=[day]),
        limit=1,
    ).records[0]


def _evaluate(instance: dg.DagsterInstance, cursor: str | None = None):
    context = dg.build_sensor_context(
        instance=instance,
        repository_def=defs.get_repository_def(),
        cursor=cursor,
    )
    return recovery.historical_daily_recovery.evaluate_tick(context)


def test_historical_recovery_scans_one_old_day_and_excludes_recent_days(monkeypatch) -> None:
    monkeypatch.setattr(recovery, "_recovery_time", lambda: NOW)
    with dg.instance_for_test() as instance:
        for day in ("2026-09-10", "2026-09-27", "2026-09-28", "2026-09-29"):
            _materialize(instance, "articles", day)

        first = _evaluate(instance)
        assert first.run_requests is not None
        assert len(first.run_requests) == 1
        assert first.run_requests[0].partition_key == "2026-09-10"
        assert first.run_requests[0].asset_selection == [dg.AssetKey("relevance")]
        assert first.cursor == "2026-09-10"

        next_tick = _evaluate(instance, first.cursor)
        assert next_tick.run_requests is not None
        assert len(next_tick.run_requests) == 1
        assert next_tick.run_requests[0].partition_key == "2026-09-27"


def test_historical_recovery_moves_to_next_stage_when_input_matches(monkeypatch) -> None:
    monkeypatch.setattr(recovery, "_recovery_time", lambda: NOW)
    with dg.instance_for_test() as instance:
        article = _materialize(instance, "articles", "2026-09-10")
        _materialize(
            instance,
            "relevance",
            "2026-09-10",
            parent="articles",
            pointer=article.storage_id,
        )

        evaluation = _evaluate(instance)
        assert evaluation.run_requests is not None
        assert len(evaluation.run_requests) == 1
        assert evaluation.run_requests[0].asset_selection == [dg.AssetKey("embeddings")]


def test_historical_recovery_uses_input_pointer_when_parent_updates_during_child_run() -> None:
    with dg.instance_for_test() as instance:
        old_article = _materialize(instance, "articles", "2026-09-10")
        new_article = _materialize(instance, "articles", "2026-09-10")
        child = _materialize(
            instance,
            "relevance",
            "2026-09-10",
            parent="articles",
            pointer=old_article.storage_id,
        )
        assert not recovery._matches_inputs(child, {dg.AssetKey("articles"): new_article}, instance)


def test_historical_recovery_suppresses_active_work(monkeypatch) -> None:
    monkeypatch.setattr(recovery, "_recovery_time", lambda: NOW)
    with dg.instance_for_test() as instance:
        _materialize(instance, "articles", "2026-09-10")
        instance.add_run(
            dg.DagsterRun(
                job_name=recovery.historical_daily_recovery_job.name,
                run_id="active-recovery",
                status=dg.DagsterRunStatus.STARTED,
                tags={"dagster/partition": "2026-09-10"},
            )
        )
        evaluation = _evaluate(instance)
        assert evaluation.run_requests == []
        assert evaluation.skip_message == "A historical daily repair is queued or active."


def test_historical_recovery_retries_failed_generation_after_cooldown() -> None:
    with dg.instance_for_test() as instance:
        _materialize(instance, "articles", "2026-09-10")
        first = recovery._repair_request(instance, "2026-09-10", datetime.now(UTC))
        assert first is not None
        instance.add_run(
            dg.DagsterRun(
                job_name=recovery.historical_daily_recovery_job.name,
                run_id="failed-recovery",
                status=dg.DagsterRunStatus.FAILURE,
                tags=dict(first.tags),
            )
        )
        assert recovery._repair_request(instance, "2026-09-10", datetime.now(UTC)) is None
        retry = recovery._repair_request(
            instance, "2026-09-10", datetime.now(UTC) + timedelta(minutes=16)
        )
        assert retry is not None
        assert retry.run_key != first.run_key


def test_historical_recovery_rotates_after_bounded_scan(monkeypatch) -> None:
    monkeypatch.setattr(recovery, "_recovery_time", lambda: NOW)
    with dg.instance_for_test() as instance:
        days = [f"2026-09-{day:02d}" for day in range(10, 17)]
        for day in days:
            _materialize(instance, "articles", day)
        cursor = None
        seen = []
        for _ in range(len(days) + 1):
            evaluation = _evaluate(instance, cursor)
            assert evaluation.run_requests is not None
            assert len(evaluation.run_requests) == 1
            seen.append(evaluation.run_requests[0].partition_key)
            cursor = evaluation.cursor
        assert seen == [*days, days[0]]


def test_historical_recovery_advances_past_a_full_scan_of_active_days(monkeypatch) -> None:
    monkeypatch.setattr(recovery, "_recovery_time", lambda: NOW)
    with dg.instance_for_test() as instance:
        days = [f"2026-09-{day:02d}" for day in range(10, 16)]
        for day in days:
            _materialize(instance, "articles", day)
        for index, day in enumerate(days[: recovery.MAX_DAYS_PER_TICK]):
            instance.add_run(
                dg.DagsterRun(
                    job_name="__ASSET_JOB",
                    run_id=f"active-{index}",
                    status=dg.DagsterRunStatus.STARTED,
                    tags={"dagster/partition": day},
                )
            )

        first = _evaluate(instance)
        assert first.run_requests == []
        assert first.cursor == days[recovery.MAX_DAYS_PER_TICK - 1]
        second = _evaluate(instance, first.cursor)
        assert second.run_requests is not None
        assert second.run_requests[0].partition_key == days[-1]


def test_historical_recovery_repairs_stale_report_even_with_unchanged_asset_inputs(
    monkeypatch,
) -> None:
    day = "2026-09-10"
    with dg.instance_for_test() as instance:
        latest = {dg.AssetKey("articles"): _materialize(instance, "articles", day)}
        for key, parents in recovery.STAGES:
            latest[key] = _materialize(
                instance,
                key.to_user_string(),
                day,
                pointers={parent.to_user_string(): latest[parent].storage_id for parent in parents},
            )

        monkeypatch.setattr(
            recovery,
            "read_daily_report_freshness",
            lambda _day: SimpleNamespace(kind="stale"),
        )
        stale = recovery._repair_request(instance, day, NOW)
        assert stale is not None
        assert stale.asset_selection == [dg.AssetKey("daily_reports")]

        monkeypatch.setattr(
            recovery,
            "read_daily_report_freshness",
            lambda _day: SimpleNamespace(kind="fresh"),
        )
        assert recovery._repair_request(instance, day, NOW) is None
