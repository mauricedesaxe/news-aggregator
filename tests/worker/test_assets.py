import json
from datetime import date, datetime
from types import SimpleNamespace
from typing import cast
from zoneinfo import ZoneInfo

import dagster as dg
import pytest
from dagster._core.definitions.metadata.metadata_value import JsonMetadataValue

from romanian_news.articles.models import ArticleAcquisitionFailure, ArticleFailureKind
from romanian_news.artifacts import ArtifactReference
from romanian_news.daily import DailyArtifactReferences
from romanian_news.groups import DailyClusterSet, NewsGroup
from romanian_news.jev_relevance_shadow import JevShadowBatchResult
from romanian_news.reports import DailyReportInput
from romanian_news.worker import assets, definitions
from romanian_news.worker.definitions import (
    article_batch_controller,
    article_batch_job,
    daily_morning_report_check,
    daily_report_repair,
    defs,
    hourly_registered_feed_poll,
    morning_report_check,
    news_automation,
    scheduled_video_digest,
    weekly_freshness,
    weekly_report_job,
    youtube_approved_publication,
    youtube_publication_job,
    youtube_relevance_controller,
    youtube_source_job,
    youtube_source_poll,
)
from romanian_news.worker.operations import ArticleBatchResult
from romanian_news.youtube.models import (
    YOUTUBE_SOURCES,
    YouTubePollStatus,
    YouTubePublicationResult,
    YouTubeSourceResult,
    YouTubeVideoId,
)

DAY = "2026-09-02"
BUCHAREST = ZoneInfo(assets.BUCHAREST_TIMEZONE)


def test_daily_partitions_follow_bucharest_dst() -> None:
    spring = assets.DAILY_PARTITIONS.time_window_for_partition_key("2026-03-29")
    autumn = assets.DAILY_PARTITIONS.time_window_for_partition_key("2026-10-25")

    assert (spring.end.timestamp() - spring.start.timestamp()) / 3600 == 23
    assert (autumn.end.timestamp() - autumn.start.timestamp()) / 3600 == 25


def test_weekly_partitions_start_on_monday() -> None:
    window = assets.WEEKLY_PARTITIONS.time_window_for_partition_key("2026-08-31")

    assert window.start.weekday() == 0
    assert window.end.weekday() == 0
    assert window.end.date().isoformat() == "2026-09-07"


def test_automation_starts_running_and_uses_bucharest_time() -> None:
    assert hourly_registered_feed_poll.default_status == dg.DefaultScheduleStatus.RUNNING
    assert daily_morning_report_check.default_status == dg.DefaultScheduleStatus.RUNNING
    assert definitions.scheduled_weekly_status.default_status == dg.DefaultScheduleStatus.RUNNING
    assert weekly_freshness.default_status == dg.DefaultSensorStatus.RUNNING
    assert weekly_freshness.minimum_interval_seconds == 3600
    assert scheduled_video_digest.default_status == dg.DefaultScheduleStatus.STOPPED
    assert news_automation.default_status == dg.DefaultSensorStatus.RUNNING
    assert article_batch_controller.default_status == dg.DefaultSensorStatus.RUNNING
    assert hourly_registered_feed_poll.cron_schedule == "0 * * * *"
    assert daily_morning_report_check.cron_schedule == "35 9 * * *"
    assert hourly_registered_feed_poll.execution_timezone == assets.BUCHAREST_TIMEZONE
    assert daily_morning_report_check.execution_timezone == assets.BUCHAREST_TIMEZONE
    schedules = defs.schedules
    assert schedules is not None
    assert {schedule.name for schedule in schedules} == {
        "hourly_registered_feed_poll",
        "daily_morning_report_check",
        "youtube_source_poll",
        "youtube_approved_publication",
        "quarter_hourly_news_feedback_sync",
        "scheduled_video_digest",
        "scheduled_video_digest_incident_monitor",
        "scheduled_weekly_status",
    }
    assert defs.resolve_job_def("morning_report_check").name == morning_report_check.name
    assert defs.resolve_job_def("weekly_report").name == weekly_report_job.name
    assert assets.weekly_reports.automation_conditions_by_key


def test_youtube_automation_starts_running_and_uses_bucharest_time() -> None:
    assert youtube_source_poll.default_status == dg.DefaultScheduleStatus.RUNNING
    assert youtube_approved_publication.default_status == dg.DefaultScheduleStatus.RUNNING
    assert youtube_relevance_controller.default_status == dg.DefaultSensorStatus.RUNNING
    assert youtube_source_poll.cron_schedule == "*/10 * * * *"
    assert youtube_approved_publication.cron_schedule == "*/10 * * * *"
    assert youtube_source_poll.execution_timezone == assets.BUCHAREST_TIMEZONE


def test_youtube_schedule_suppresses_only_the_active_source() -> None:
    scheduled_at = datetime.fromisoformat("2026-09-05T09:30:00+03:00")
    with dg.instance_for_test() as instance:
        instance.create_run_for_job(
            defs.resolve_job_def("youtube_source_job"),
            status=dg.DagsterRunStatus.STARTED,
            tags={"news/youtube_source_id": "recorder-youtube"},
        )
        with dg.build_schedule_context(
            instance=instance,
            scheduled_execution_time=scheduled_at,
            repository_def=defs.get_repository_def(),
        ) as context:
            evaluation = youtube_source_poll.evaluate_tick(context)
    assert evaluation.run_requests is not None
    assert [request.tags["news/youtube_source_id"] for request in evaluation.run_requests] == [
        "starea-impostorilor-youtube",
        "snoop-youtube",
        "stirile-protv-youtube",
    ]
    assert [request.run_key for request in evaluation.run_requests] == [
        "youtube-source:starea-impostorilor-youtube:2026-09-05T09:30:00+03:00",
        "youtube-source:snoop-youtube:2026-09-05T09:30:00+03:00",
        "youtube-source:stirile-protv-youtube:2026-09-05T09:30:00+03:00",
    ]


def test_youtube_source_and_publication_materializations_expose_identity(monkeypatch) -> None:
    monkeypatch.setattr(
        assets,
        "materialize_youtube_source",
        lambda *_args: YouTubeSourceResult(
            source_id=YOUTUBE_SOURCES.source("recorder-youtube").source_id,
            poll_status=YouTubePollStatus.CURRENT,
            candidate_version_id="c" * 64,
        ),
    )
    source = dg.materialize(
        [assets.youtube_source],
        tags={
            "news/youtube_source_id": "recorder-youtube",
            "news/scheduled_at": "2026-09-05T09:30:00+03:00",
        },
    )
    source_materialization = source.asset_materializations_for_node("youtube_source")[0]
    assert source_materialization.metadata["source_id"].value == "recorder-youtube"
    assert source_materialization.metadata["candidate_version_id"].value == "c" * 64

    monkeypatch.setattr(
        assets,
        "materialize_next_youtube_publication",
        lambda *_args: YouTubePublicationResult(
            source_id=YOUTUBE_SOURCES.source("recorder-youtube").source_id,
            video_id=YouTubeVideoId("abcdefghijk"),
            candidate_version_id="c" * 64,
            article_version_id="a" * 64,
            bucharest_day=date.fromisoformat(DAY),
            accepted_clip_count=1,
        ),
    )
    publication = dg.materialize([assets.youtube_publication])
    materialization = publication.asset_materializations_for_node("youtube_publication")[0]
    assert materialization.metadata["bucharest_day"].value == DAY
    assert materialization.metadata["article_version_id"].value == "a" * 64


def test_youtube_relevance_uses_publication_day_metadata() -> None:
    event = SimpleNamespace(
        asset_materialization=dg.AssetMaterialization(
            asset_key="youtube_publication",
            metadata={"bucharest_day": DAY, "article_version_id": "a" * 64},
        )
    )
    asset_event = cast(dg.EventLogEntry, cast(object, event))
    assert definitions._youtube_publication_day(asset_event) == DAY
    assert definitions._youtube_publication(asset_event) == (DAY, "a" * 64)


def test_youtube_relevance_sensor_suppresses_an_active_run_for_the_same_article(
    monkeypatch,
) -> None:
    article_version_id = "a" * 64
    tag = {"news/youtube_article_version_id": article_version_id}
    monkeypatch.setattr(
        assets,
        "materialize_next_youtube_publication",
        lambda *_args: YouTubePublicationResult(
            source_id=YOUTUBE_SOURCES.source("recorder-youtube").source_id,
            video_id=YouTubeVideoId("abcdefghijk"),
            candidate_version_id="c" * 64,
            article_version_id=article_version_id,
            bucharest_day=date.fromisoformat(DAY),
            accepted_clip_count=1,
        ),
    )
    with dg.instance_for_test() as instance:
        instance.create_run_for_job(
            defs.resolve_job_def("youtube_relevance"),
            status=dg.DagsterRunStatus.STARTED,
            tags=tag,
        )
        assert dg.materialize([assets.youtube_publication], instance=instance).success
        with dg.build_sensor_context(
            instance=instance,
            repository_def=defs.get_repository_def(),
        ) as context:
            evaluation = youtube_relevance_controller.evaluate_tick(context)

    assert evaluation.run_requests == []
    assert (
        evaluation.skip_message == "Relevance is already queued or active for this YouTube article."
    )


def test_youtube_relevance_sensor_retries_a_failed_run_on_a_later_materialization(
    monkeypatch,
) -> None:
    article_version_id = "a" * 64
    monkeypatch.setattr(
        assets,
        "materialize_next_youtube_publication",
        lambda *_args: YouTubePublicationResult(
            source_id=YOUTUBE_SOURCES.source("recorder-youtube").source_id,
            video_id=YouTubeVideoId("abcdefghijk"),
            candidate_version_id="c" * 64,
            article_version_id=article_version_id,
            bucharest_day=date.fromisoformat(DAY),
            accepted_clip_count=1,
        ),
    )
    with dg.instance_for_test() as instance:
        assert dg.materialize([assets.youtube_publication], instance=instance).success
        with dg.build_sensor_context(
            instance=instance,
            repository_def=defs.get_repository_def(),
        ) as first_context:
            first_evaluation = youtube_relevance_controller.evaluate_tick(first_context)
        assert first_evaluation.run_requests
        first = first_evaluation.run_requests[0]
        first_cursor = str(
            instance.fetch_materializations(
                dg.AssetRecordsFilter(asset_key=assets.youtube_publication.key), limit=1
            )
            .records[0]
            .storage_id
        )
        instance.create_run_for_job(
            defs.resolve_job_def("youtube_relevance"),
            status=dg.DagsterRunStatus.FAILURE,
            tags={"news/youtube_article_version_id": article_version_id},
        )
        assert dg.materialize([assets.youtube_publication], instance=instance).success
        with dg.build_sensor_context(
            instance=instance,
            cursor=first_cursor,
            repository_def=defs.get_repository_def(),
        ) as second_context:
            second_evaluation = youtube_relevance_controller.evaluate_tick(second_context)

    assert second_evaluation.run_requests
    second = second_evaluation.run_requests[0]
    assert first.run_key != second.run_key
    assert first.tags["news/youtube_article_version_id"] == article_version_id
    assert second.tags["news/youtube_article_version_id"] == article_version_id


def test_youtube_relevance_sensor_keeps_one_materialization_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(
        assets,
        "materialize_next_youtube_publication",
        lambda *_args: YouTubePublicationResult(
            source_id=YOUTUBE_SOURCES.source("recorder-youtube").source_id,
            video_id=YouTubeVideoId("abcdefghijk"),
            candidate_version_id="c" * 64,
            article_version_id="a" * 64,
            bucharest_day=date.fromisoformat(DAY),
            accepted_clip_count=1,
        ),
    )
    with dg.instance_for_test() as instance:
        assert dg.materialize([assets.youtube_publication], instance=instance).success
        requests = []
        for _ in range(2):
            with dg.build_sensor_context(
                instance=instance,
                repository_def=defs.get_repository_def(),
            ) as context:
                evaluation = youtube_relevance_controller.evaluate_tick(context)
                assert evaluation.run_requests
                requests.append(evaluation.run_requests[0])

    assert requests[0].run_key == requests[1].run_key


def test_each_automated_asset_has_one_owner() -> None:
    asset_graph = defs.get_repository_def().asset_graph
    eager_keys = {key.path[-1] for key in news_automation.asset_selection.resolve(asset_graph)}
    assert assets.youtube_source.automation_conditions_by_key == {}
    assert youtube_relevance_controller.default_status == dg.DefaultSensorStatus.RUNNING
    scheduled_keys = {
        schedule.name: {
            key.path[-1]
            for key in defs.resolve_job_def(schedule.job_name).asset_layer.executable_asset_keys
        }
        for schedule in (
            hourly_registered_feed_poll,
            youtube_source_poll,
            youtube_approved_publication,
        )
    }

    assert eager_keys == {
        "relevance",
        "jev_relevance_shadow",
        "embeddings",
        "daily_clusters",
        "group_summaries",
        "group_sentiment",
        "daily_themes",
        "daily_subject_assessments",
        "daily_reports",
    }
    assert scheduled_keys == {
        "hourly_registered_feed_poll": {"feed_intake"},
        "youtube_source_poll": {"youtube_source"},
        "youtube_approved_publication": {"youtube_publication"},
    }
    assert defs.resolve_job_def("youtube_source_job").name == youtube_source_job.name
    assert defs.resolve_job_def("youtube_publication_job").name == youtube_publication_job.name
    controller_keys = {
        key.path[-1]
        for key in defs.resolve_job_def(article_batch_job.name).asset_layer.executable_asset_keys
    }
    assert controller_keys == {"articles"}
    assert "articles" not in eager_keys
    assert {key.path[-1] for key in weekly_freshness.asset_selection.resolve(asset_graph)} == {
        "weekly_reports"
    }
    assert (
        dg.AssetCheckKey(dg.AssetKey("articles"), "no_quarantined_inputs")
        in asset_graph.asset_check_keys
    )
    assert eager_keys.isdisjoint(set().union(*scheduled_keys.values(), controller_keys))


def test_daily_report_repair_job_selects_only_the_daily_report_asset() -> None:
    job = defs.resolve_job_def(daily_report_repair.name)
    keys = {key.path[-1] for key in job.asset_layer.executable_asset_keys}

    assert keys == {"daily_reports"}
    report_op = cast(dg.OpDefinition, assets.daily_reports.node_def)
    assert report_op.pool == "news_catalog"


def test_subject_assessment_asset_wires_exact_upstream_dependencies() -> None:
    asset_graph = defs.get_repository_def().asset_graph

    assessment_parents = {
        key.path[-1]
        for key in asset_graph.get(dg.AssetKey("daily_subject_assessments")).parent_keys
    }
    report_parents = {
        key.path[-1] for key in asset_graph.get(dg.AssetKey("daily_reports")).parent_keys
    }

    assert assessment_parents == {"daily_themes", "relevance"}
    assert report_parents == {"daily_subject_assessments", "group_sentiment"}
    assert (
        dg.AssetCheckKey(dg.AssetKey("daily_subject_assessments"), "exact_inputs")
        in asset_graph.asset_check_keys
    )


def test_article_controller_blocks_active_legacy_article_automation() -> None:
    with dg.instance_for_test() as instance:
        instance.add_run(
            dg.DagsterRun(
                job_name="__ASSET_JOB",
                run_id="legacy-article",
                status=dg.DagsterRunStatus.STARTED,
                tags={"dagster/auto_materialize": "true"},
                asset_selection={dg.AssetKey("articles")},
            )
        )

        context = dg.build_sensor_context(
            instance=instance,
            repository_def=defs.get_repository_def(),
        )
        evaluation = article_batch_controller.evaluate_tick(context)

    assert not evaluation.run_requests
    assert evaluation.skip_message == "Legacy article automation is queued or active."


def test_article_controller_blocks_unscoped_legacy_automation() -> None:
    with dg.instance_for_test() as instance:
        instance.add_run(
            dg.DagsterRun(
                job_name="__ASSET_JOB",
                run_id="legacy-all-assets",
                status=dg.DagsterRunStatus.STARTED,
                tags={"dagster/auto_materialize": "true"},
                asset_selection=None,
            )
        )

        assert definitions._active_article_batch_work(instance) is None


def test_article_controller_ignores_unrelated_or_terminal_legacy_runs() -> None:
    with dg.instance_for_test() as instance:
        for run in (
            dg.DagsterRun(
                job_name="__ASSET_JOB",
                run_id="unrelated",
                status=dg.DagsterRunStatus.STARTED,
                tags={"dagster/auto_materialize": "true"},
                asset_selection={dg.AssetKey("relevance")},
            ),
            dg.DagsterRun(
                job_name="__ASSET_JOB",
                run_id="manual",
                status=dg.DagsterRunStatus.STARTED,
                tags={},
                asset_selection={dg.AssetKey("articles")},
            ),
            dg.DagsterRun(
                job_name="__ASSET_JOB",
                run_id="completed",
                status=dg.DagsterRunStatus.SUCCESS,
                tags={"dagster/auto_materialize": "true"},
                asset_selection={dg.AssetKey("articles")},
            ),
        ):
            instance.add_run(run)

        assert definitions._active_article_batch_work(instance) == set()


def test_article_controller_blocks_only_the_active_partition(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-09-09T12:00:00+03:00")
    active_day = now.date()
    other_day = datetime.fromisoformat("2026-09-08T12:00:00+03:00").date()
    monkeypatch.setattr(definitions, "_controller_time", lambda: now)
    monkeypatch.setattr(
        definitions,
        "read_article_candidate_days",
        lambda *_args: (other_day, active_day),
    )

    def request(_context, day, _now):
        if day == active_day:
            return None
        return dg.RunRequest(run_key="other", partition_key=other_day.isoformat())

    monkeypatch.setattr(definitions, "_article_batch_request", request)
    with dg.instance_for_test() as instance:
        instance.add_run(
            dg.DagsterRun(
                job_name="article_batch",
                run_id="active-batch",
                status=dg.DagsterRunStatus.STARTED,
                tags={"dagster/partition": active_day.isoformat()},
            )
        )
        context = dg.build_sensor_context(
            instance=instance,
            repository_def=defs.get_repository_def(),
        )
        evaluation = article_batch_controller.evaluate_tick(context)

    assert evaluation.run_requests
    assert evaluation.run_requests[0].partition_key == other_day.isoformat()


def test_article_controller_blocks_everything_while_a_batch_lacks_a_partition(
    monkeypatch,
) -> None:
    now = datetime.fromisoformat("2026-09-09T12:00:00+03:00")
    monkeypatch.setattr(definitions, "_controller_time", lambda: now)
    monkeypatch.setattr(
        definitions,
        "read_article_candidate_days",
        lambda *_args: (datetime.fromisoformat("2026-09-08T12:00:00+03:00").date(),),
    )
    with dg.instance_for_test() as instance:
        instance.add_run(
            dg.DagsterRun(
                job_name="article_batch",
                run_id="unscoped-batch",
                status=dg.DagsterRunStatus.STARTED,
                tags={},
            )
        )
        context = dg.build_sensor_context(
            instance=instance,
            repository_def=defs.get_repository_def(),
        )
        evaluation = article_batch_controller.evaluate_tick(context)

    assert not evaluation.run_requests
    assert evaluation.skip_message == "Legacy article automation is queued or active."


def test_article_controller_prioritizes_today_then_advances_history(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-09-09T12:00:00+03:00")
    historical = datetime.fromisoformat("2026-09-08T12:00:00+03:00").date()
    visited = []
    request = dg.RunRequest(run_key="historical", partition_key=historical.isoformat())
    monkeypatch.setattr(definitions, "_controller_time", lambda: now)
    monkeypatch.setattr(
        definitions,
        "read_article_candidate_days",
        lambda *_args: (historical, now.date()),
    )
    monkeypatch.setattr(
        definitions,
        "_article_batch_request",
        lambda _context, day, _now: visited.append(day) or (request if day == historical else None),
    )
    with dg.instance_for_test() as instance:
        context = dg.build_sensor_context(
            instance=instance,
            repository_def=defs.get_repository_def(),
        )
        evaluation = article_batch_controller.evaluate_tick(context)

    assert evaluation.run_requests is not None
    assert len(evaluation.run_requests) == 1
    assert evaluation.run_requests[0].run_key == request.run_key
    assert evaluation.run_requests[0].partition_key == request.partition_key
    assert visited == [now.date(), historical]


def test_article_controller_runs_four_current_batches_then_newest_history(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-09-09T12:00:00+03:00")
    today = now.date()
    middle = datetime.fromisoformat("2026-09-05T12:00:00+03:00").date()
    oldest = datetime.fromisoformat("2026-09-01T12:00:00+03:00").date()
    monkeypatch.setattr(definitions, "_controller_time", lambda: now)
    monkeypatch.setattr(
        definitions,
        "read_article_candidate_days",
        lambda *_args: (oldest, middle, today),
    )
    monkeypatch.setattr(
        definitions,
        "_article_batch_request",
        lambda _context, day, _now: dg.RunRequest(
            run_key=f"batch:{day.isoformat()}",
            partition_key=day.isoformat(),
        ),
    )
    cursor = None
    partitions = []
    with dg.instance_for_test() as instance:
        for _ in range(6):
            context = dg.build_sensor_context(
                instance=instance,
                repository_def=defs.get_repository_def(),
                cursor=cursor,
            )
            evaluation = article_batch_controller.evaluate_tick(context)
            assert evaluation.run_requests
            partitions.append(evaluation.run_requests[0].partition_key)
            cursor = evaluation.cursor

    assert partitions == [today.isoformat()] * 4 + [middle.isoformat(), today.isoformat()]
    assert cursor == "1"


def test_article_controller_falls_back_without_consuming_preference(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-09-09T12:00:00+03:00")
    today = now.date()
    oldest = datetime.fromisoformat("2026-09-01T12:00:00+03:00").date()
    monkeypatch.setattr(definitions, "_controller_time", lambda: now)
    monkeypatch.setattr(
        definitions,
        "read_article_candidate_days",
        lambda *_args: (oldest, today),
    )

    def request(_context, day, _now):
        if day == oldest:
            return None
        return dg.RunRequest(run_key="current", partition_key=today.isoformat())

    monkeypatch.setattr(definitions, "_article_batch_request", request)
    with dg.instance_for_test() as instance:
        context = dg.build_sensor_context(
            instance=instance,
            repository_def=defs.get_repository_def(),
            cursor="4",
        )
        evaluation = article_batch_controller.evaluate_tick(context)

    assert evaluation.run_requests
    assert evaluation.run_requests[0].partition_key == today.isoformat()
    assert evaluation.cursor == "4"


def test_article_controller_uses_newest_history_when_current_has_no_ready_work(
    monkeypatch,
) -> None:
    now = datetime.fromisoformat("2026-09-09T12:00:00+03:00")
    today = now.date()
    middle = datetime.fromisoformat("2026-09-05T12:00:00+03:00").date()
    oldest = datetime.fromisoformat("2026-09-01T12:00:00+03:00").date()
    monkeypatch.setattr(definitions, "_controller_time", lambda: now)
    monkeypatch.setattr(
        definitions,
        "read_article_candidate_days",
        lambda *_args: (oldest, middle, today),
    )

    def request(_context, day, _now):
        if day == today:
            return None
        return dg.RunRequest(run_key=f"batch:{day.isoformat()}", partition_key=day.isoformat())

    monkeypatch.setattr(definitions, "_article_batch_request", request)
    with dg.instance_for_test() as instance:
        context = dg.build_sensor_context(
            instance=instance,
            repository_def=defs.get_repository_def(),
            cursor="2",
        )
        evaluation = article_batch_controller.evaluate_tick(context)

    assert evaluation.run_requests
    assert evaluation.run_requests[0].partition_key == middle.isoformat()
    assert evaluation.cursor == "0"


def test_article_controller_preserves_cursor_when_no_work_is_ready(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-09-09T12:00:00+03:00")
    monkeypatch.setattr(definitions, "_controller_time", lambda: now)
    monkeypatch.setattr(definitions, "read_article_candidate_days", lambda *_args: ())
    monkeypatch.setattr(definitions, "_article_batch_request", lambda *_args: None)
    with dg.instance_for_test() as instance:
        context = dg.build_sensor_context(
            instance=instance,
            repository_def=defs.get_repository_def(),
            cursor="2",
        )
        evaluation = article_batch_controller.evaluate_tick(context)

    assert not evaluation.run_requests
    assert evaluation.cursor == "2"


def test_article_batch_run_keys_recover_after_a_failed_tick(monkeypatch) -> None:
    day = datetime.fromisoformat(DAY).date()
    event_id = "a" * 64
    plan = SimpleNamespace(
        selected=(
            SimpleNamespace(source=SimpleNamespace(event_id=event_id), work_generation="b" * 64),
        ),
        remaining_entries=0,
        deferred_event_ids=(),
        quarantined_event_ids=(),
        source_covered_days=(day,),
    )
    monkeypatch.setattr(definitions, "plan_article_work", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(definitions, "read_article_attempt_states", lambda _event_ids: {})
    monkeypatch.setattr(definitions, "feed_registry", SimpleNamespace)
    monkeypatch.setattr(
        definitions,
        "read_daily_article_references",
        lambda _day: DailyArtifactReferences(day=day, values=()),
    )
    with dg.instance_for_test() as instance:
        context = dg.build_sensor_context(
            instance=instance,
            repository_def=defs.get_repository_def(),
        )
        first = definitions._article_batch_request(
            context,
            day,
            datetime.fromisoformat("2026-09-09T09:00:00+00:00"),
        )
        second = definitions._article_batch_request(
            context,
            day,
            datetime.fromisoformat("2026-09-09T09:01:00+00:00"),
        )

    assert first is not None and second is not None
    assert first.tags["news/article_batch_key"] == second.tags["news/article_batch_key"]
    assert first.run_key != second.run_key
    assert json.loads(first.tags["news/article_event_ids"]) == [event_id]


def test_article_controller_reports_unchanged_quarantine_once(monkeypatch) -> None:
    day = datetime.fromisoformat(DAY).date()
    plan = SimpleNamespace(
        selected=(),
        remaining_entries=0,
        deferred_event_ids=(),
        quarantined_event_ids=("a" * 64,),
        source_covered_days=(day,),
    )
    monkeypatch.setattr(definitions, "plan_article_work", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(definitions, "read_article_attempt_states", lambda _event_ids: {})
    monkeypatch.setattr(definitions, "feed_registry", SimpleNamespace)
    monkeypatch.setattr(
        definitions,
        "read_daily_article_references",
        lambda _day: DailyArtifactReferences(day=day, values=()),
    )
    with dg.instance_for_test() as instance:
        context = dg.build_sensor_context(
            instance=instance,
            repository_def=defs.get_repository_def(),
        )
        first = definitions._article_batch_request(
            context, day, datetime.fromisoformat("2026-09-09T09:00:00+00:00")
        )
        assert first is not None
        instance.add_run(
            dg.DagsterRun(
                job_name="article_batch",
                run_id="quarantine-reported",
                status=dg.DagsterRunStatus.FAILURE,
                tags=first.tags,
            )
        )
        second = definitions._article_batch_request(
            context, day, datetime.fromisoformat("2026-09-09T09:01:00+00:00")
        )

    assert second is None


def test_mixed_quarantine_batch_failure_does_not_count_as_reported_until_work_drains(
    monkeypatch,
) -> None:
    day = datetime.fromisoformat(DAY).date()
    selected_id = "a" * 64
    quarantined_id = "b" * 64
    mixed_plan = SimpleNamespace(
        selected=(
            SimpleNamespace(source=SimpleNamespace(event_id=selected_id), work_generation="c" * 64),
        ),
        remaining_entries=0,
        deferred_event_ids=(),
        quarantined_event_ids=(quarantined_id,),
        source_covered_days=(day,),
    )
    drained_plan = SimpleNamespace(
        selected=(),
        remaining_entries=0,
        deferred_event_ids=(),
        quarantined_event_ids=(quarantined_id,),
        source_covered_days=(day,),
    )
    plans = {"mixed": mixed_plan, "drained": drained_plan}
    requested = {"mixed": 0, "drained": 0}

    def plan(*_args, **_kwargs):
        if requested["mixed"] < 2:
            requested["mixed"] += 1
            return plans["mixed"]
        requested["drained"] += 1
        return plans["drained"]

    monkeypatch.setattr(definitions, "plan_article_work", plan)
    monkeypatch.setattr(definitions, "read_article_attempt_states", lambda _event_ids: {})
    monkeypatch.setattr(definitions, "feed_registry", SimpleNamespace)
    monkeypatch.setattr(
        definitions,
        "read_daily_article_references",
        lambda _day: DailyArtifactReferences(day=day, values=()),
    )
    with dg.instance_for_test() as instance:
        context = dg.build_sensor_context(
            instance=instance,
            repository_def=defs.get_repository_def(),
        )
        first = definitions._article_batch_request(
            context, day, datetime.fromisoformat("2026-09-09T09:00:00+00:00")
        )
        assert first is not None
        instance.add_run(
            dg.DagsterRun(
                job_name="article_batch",
                run_id="mixed-batch-failed",
                status=dg.DagsterRunStatus.FAILURE,
                tags=first.tags,
            )
        )
        second = definitions._article_batch_request(
            context, day, datetime.fromisoformat("2026-09-09T09:01:00+00:00")
        )
        assert second is not None
        assert second.tags["news/article_batch_key"] == first.tags["news/article_batch_key"]
        assert second.run_key != first.run_key

        third = definitions._article_batch_request(
            context, day, datetime.fromisoformat("2026-09-09T09:02:00+00:00")
        )
        assert third is not None
        assert third.tags["news/article_batch_key"] != first.tags["news/article_batch_key"]
        instance.add_run(
            dg.DagsterRun(
                job_name="article_batch",
                run_id="drained-batch-failed",
                status=dg.DagsterRunStatus.FAILURE,
                tags=third.tags,
            )
        )
        fourth = definitions._article_batch_request(
            context, day, datetime.fromisoformat("2026-09-09T09:03:00+00:00")
        )

    assert fourth is None


def test_mixed_quarantine_batch_fails_the_asset_without_materializing(monkeypatch) -> None:
    day = datetime.fromisoformat(DAY).date()
    quarantined = "b" * 64
    references = DailyArtifactReferences(day=day, values=())
    monkeypatch.setattr(assets, "materialize_feed_intake", lambda *_args: references)
    monkeypatch.setattr(
        assets,
        "materialize_articles",
        lambda *_args, **_kwargs: ArticleBatchResult(
            references=references,
            requested_event_ids=("a" * 64,),
            acquired_event_ids=(),
            skipped_event_ids=(),
            failures=(),
            remaining_entries=1,
            deferred_event_ids=(),
            quarantined_event_ids=(quarantined,),
            source_covered=True,
        ),
    )

    result = dg.materialize(
        [assets.feed_intake, assets.articles],
        partition_key=DAY,
        raise_on_error=False,
        tags={
            "news/scheduled_at": "2026-09-02T12:00:00+03:00",
            "news/article_event_ids": '["' + "a" * 64 + '"]',
        },
    )

    assert not result.success
    assert result.asset_materializations_for_node("articles") == []
    observations = [
        event for event in result.all_events if event.event_type_value == "ASSET_OBSERVATION"
    ]
    assert len(observations) == 1
    metadata = observations[0].asset_observation_data.asset_observation.metadata
    quarantined_metadata = metadata["quarantined_event_ids"]
    assert isinstance(quarantined_metadata, JsonMetadataValue)
    assert quarantined_metadata.data == [quarantined]


def test_article_batch_identity_changes_with_implementation_ref(monkeypatch) -> None:
    day = datetime.fromisoformat(DAY).date()
    event_id = "a" * 64
    plan = SimpleNamespace(
        selected=(
            SimpleNamespace(source=SimpleNamespace(event_id=event_id), work_generation="b" * 64),
        ),
        remaining_entries=0,
        deferred_event_ids=(),
        quarantined_event_ids=(),
        source_covered_days=(day,),
    )
    monkeypatch.setattr(definitions, "plan_article_work", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(definitions, "read_article_attempt_states", lambda *_args: {})
    monkeypatch.setattr(definitions, "feed_registry", SimpleNamespace)
    monkeypatch.setattr(
        definitions,
        "read_daily_article_references",
        lambda _day: DailyArtifactReferences(day=day, values=()),
    )
    with dg.instance_for_test() as instance:
        context = dg.build_sensor_context(
            instance=instance,
            repository_def=defs.get_repository_def(),
        )
        monkeypatch.setattr(definitions, "IMPLEMENTATION_REF", "git:old")
        old = definitions._article_batch_request(
            context,
            day,
            datetime.fromisoformat("2026-09-09T09:00:00+00:00"),
        )
        monkeypatch.setattr(definitions, "IMPLEMENTATION_REF", "git:new")
        new = definitions._article_batch_request(
            context,
            day,
            datetime.fromisoformat("2026-09-09T09:00:00+00:00"),
        )

    assert old is not None and new is not None
    assert old.tags["news/article_batch_key"] != new.tags["news/article_batch_key"]


def test_past_uncovered_article_day_is_complete_once_entries_are_gone() -> None:
    day = date(2026, 9, 22)
    result = ArticleBatchResult(
        references=DailyArtifactReferences(day=day, values=()),
        requested_event_ids=(),
        acquired_event_ids=(),
        skipped_event_ids=(),
        failures=(),
        remaining_entries=0,
        deferred_event_ids=(),
        quarantined_event_ids=(),
        source_covered=False,
    )

    assert result.is_complete(date(2026, 9, 22)) is False
    assert result.is_complete(date(2026, 9, 21)) is False
    assert result.is_complete(date(2026, 9, 23)) is True


def test_network_work_uses_independent_pools() -> None:
    assert assets.feed_intake.op.pool == "news_feed_network"
    assert assets.articles.op.pool == "news_article_network"
    feedback_op = definitions.news_feedback_sync.graph.node_defs[0]
    assert isinstance(feedback_op, dg.OpDefinition)
    assert feedback_op.pool == "news_feedback_network"


def test_partial_article_batch_observes_without_materializing(monkeypatch) -> None:
    day = datetime.fromisoformat(DAY).date()
    references = DailyArtifactReferences(day=day, values=())
    monkeypatch.setattr(assets, "materialize_feed_intake", lambda *_args: references)
    monkeypatch.setattr(
        assets,
        "materialize_articles",
        lambda *_args, **_kwargs: ArticleBatchResult(
            references=references,
            requested_event_ids=("a" * 64,),
            acquired_event_ids=(),
            skipped_event_ids=(),
            failures=(
                ArticleAcquisitionFailure(
                    event_id="a" * 64,
                    kind=ArticleFailureKind.DETERMINISTIC,
                    fingerprint="b" * 64,
                    message="invalid article",
                ),
            ),
            remaining_entries=1,
            deferred_event_ids=("a" * 64,),
            quarantined_event_ids=(),
            source_covered=False,
        ),
    )

    result = dg.materialize(
        [assets.feed_intake, assets.articles],
        partition_key=DAY,
        tags={
            "news/scheduled_at": "2026-09-02T12:00:00+03:00",
            "news/article_event_ids": '["' + "a" * 64 + '"]',
        },
    )

    assert result.success
    assert result.asset_materializations_for_node("articles") == []
    observations = [
        event for event in result.all_events if event.event_type_value == "ASSET_OBSERVATION"
    ]
    assert len(observations) == 1


def test_past_uncovered_empty_batch_materializes_articles(monkeypatch) -> None:
    day = datetime.fromisoformat(DAY).date()
    references = DailyArtifactReferences(day=day, values=())
    monkeypatch.setattr(assets, "materialize_feed_intake", lambda *_args: references)
    monkeypatch.setattr(
        assets,
        "materialize_articles",
        lambda *_args, **_kwargs: ArticleBatchResult(
            references=references,
            requested_event_ids=(),
            acquired_event_ids=(),
            skipped_event_ids=(),
            failures=(),
            remaining_entries=0,
            deferred_event_ids=(),
            quarantined_event_ids=(),
            source_covered=False,
        ),
    )

    result = dg.materialize(
        [assets.feed_intake, assets.articles],
        partition_key=DAY,
        tags={"news/scheduled_at": "2026-09-02T12:00:00+03:00", "news/article_event_ids": "[]"},
    )

    assert result.success
    assert result.asset_materializations_for_node("articles")


def test_quarantined_inputs_fail_without_materializing_articles(monkeypatch) -> None:
    day = datetime.fromisoformat(DAY).date()
    quarantined = "a" * 64
    references = DailyArtifactReferences(day=day, values=())
    monkeypatch.setattr(assets, "materialize_feed_intake", lambda *_args: references)
    monkeypatch.setattr(
        assets,
        "materialize_articles",
        lambda *_args, **_kwargs: ArticleBatchResult(
            references=references,
            requested_event_ids=(),
            acquired_event_ids=(),
            skipped_event_ids=(),
            failures=(),
            remaining_entries=0,
            deferred_event_ids=(),
            quarantined_event_ids=(quarantined,),
            source_covered=True,
        ),
    )

    result = dg.materialize(
        [assets.feed_intake, assets.articles],
        partition_key=DAY,
        raise_on_error=False,
        tags={
            "news/scheduled_at": "2026-09-02T12:00:00+03:00",
            "news/article_event_ids": "[]",
        },
    )

    assert not result.success
    assert result.asset_materializations_for_node("articles") == []
    observations = [
        event for event in result.all_events if event.event_type_value == "ASSET_OBSERVATION"
    ]
    assert len(observations) == 1
    metadata = observations[0].asset_observation_data.asset_observation.metadata
    assert metadata["quarantined_count"].value == 1
    quarantined_metadata = metadata["quarantined_event_ids"]
    assert isinstance(quarantined_metadata, JsonMetadataValue)
    assert quarantined_metadata.data == [quarantined]


def test_feed_asset_materializes_recorded_references(monkeypatch) -> None:
    reference = ArtifactReference(
        artifact_id="news:feed-observation:test",
        version_id="1" * 64,
        content_digest="2" * 64,
        r2_key="news/feed-observations/test.json",
    )
    monkeypatch.setattr(
        assets,
        "materialize_feed_intake",
        lambda *_args: DailyArtifactReferences(
            day=datetime.fromisoformat(DAY).date(),
            values=(reference,),
        ),
    )

    result = dg.materialize(
        [assets.feed_intake],
        partition_key=DAY,
        tags={"news/scheduled_at": "2026-09-02T12:00:00+03:00"},
    )

    assert result.success
    materialization = result.asset_materializations_for_node("feed_intake")[0]
    assert materialization.metadata["artifact_count"].value == 1
    version_ids_metadata = materialization.metadata["version_ids"]
    assert isinstance(version_ids_metadata, JsonMetadataValue)
    assert version_ids_metadata.data == ["1" * 64]


def test_daily_report_completeness_uses_its_cluster_snapshot(monkeypatch) -> None:
    day = datetime.fromisoformat(DAY).date()
    reference = ArtifactReference(
        artifact_id="artifact:test",
        version_id="1" * 64,
        content_digest="2" * 64,
        r2_key="objects/test.json",
    )
    cluster_set = DailyClusterSet(
        day=day,
        algorithm="test",
        threshold=0.72,
        embedding_model="test",
        article_version_ids=("3" * 64,),
        relevance_version_ids=("4" * 64,),
        embedding_version_ids=("5" * 64,),
        merges=(),
        groups=(NewsGroup(id="6" * 64, article_version_ids=("3" * 64,)),),
    )
    report_input = DailyReportInput(
        day=day,
        themes=ArtifactReference(
            artifact_id="news:themes:test",
            version_id="8" * 64,
            content_digest="9" * 64,
            r2_key="objects/themes.json",
        ),
        assessments=ArtifactReference(
            artifact_id="news:subject-assessments:test",
            version_id="4" * 64,
            content_digest="7" * 64,
            r2_key="objects/subject-assessments.json",
        ),
        cluster_set=reference,
        summaries=(reference,),
        sentiments=(reference,),
    )
    monkeypatch.setattr(assets, "read_recorded_daily_report_input", lambda _day: report_input)
    monkeypatch.setattr(
        assets,
        "read_recorded_daily_theme_input",
        lambda _day: SimpleNamespace(
            cluster_set=reference,
            groups=(SimpleNamespace(summary=reference),),
        ),
    )
    monkeypatch.setattr(
        assets,
        "read_verified_r2_object",
        lambda *_args: cluster_set.model_dump_json().encode(),
    )
    monkeypatch.setattr(
        assets,
        "read_daily_cluster_reference",
        lambda _day: (_ for _ in ()).throw(AssertionError("read moving cluster head")),
        raising=False,
    )

    result = assets.daily_report_completeness_result(DAY)

    assert result.passed


@pytest.mark.parametrize(
    (
        "paired",
        "over_guard",
        "failed",
        "already_claimed",
        "unresolved",
        "missing_incumbent",
        "passed",
    ),
    (
        pytest.param(1, 1, 0, 1, 0, 0, True, id="claim_only_outcomes_pass"),
        pytest.param(3, 0, 0, 0, 0, 0, True, id="all_paired_passes"),
        pytest.param(0, 1, 1, 0, 0, 0, False, id="failed_fails"),
        pytest.param(0, 0, 0, 1, 1, 0, False, id="unresolved_fails"),
        pytest.param(0, 0, 0, 0, 0, 1, False, id="missing_incumbent_fails"),
    ),
)
def test_jev_shadow_paired_accounting_check_follows_the_batch_result(
    monkeypatch,
    paired: int,
    over_guard: int,
    failed: int,
    already_claimed: int,
    unresolved: int,
    missing_incumbent: int,
    passed: bool,
) -> None:
    day = datetime.fromisoformat(DAY).date()
    result_value = JevShadowBatchResult(
        day=day,
        enabled=True,
        article_count=paired + over_guard + failed + already_claimed + unresolved,
        missing_incumbent=missing_incumbent,
        paired=paired,
        over_guard=over_guard,
        failed=failed,
        already_claimed=already_claimed,
        unresolved=unresolved,
    )
    monkeypatch.setattr(assets, "materialize_jev_relevance_shadow", lambda _day: result_value)

    result = dg.materialize([assets.jev_relevance_shadow], partition_key=DAY)

    assert result.success
    (evaluation,) = result.get_asset_check_evaluations()
    assert evaluation.passed is passed
    assert evaluation.check_name == "paired_accounting"
    assert {key: value.value for key, value in evaluation.metadata.items()} == (
        result_value.model_dump(mode="json")
    )
