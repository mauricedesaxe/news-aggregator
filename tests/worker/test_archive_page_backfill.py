from datetime import UTC, date, datetime

import dagster as dg

from romanian_news.worker import archive_page_backfill
from romanian_news.worker.definitions import defs


def test_page_schedule_skips_outlet_with_active_batch(monkeypatch) -> None:
    monkeypatch.setattr(
        archive_page_backfill,
        "next_page_window",
        lambda _outlet, _start, _end: (date(2025, 9, 27), date(2025, 9, 30)),
    )
    scheduled_at = datetime(2026, 9, 27, 17, 18, tzinfo=UTC)
    with dg.instance_for_test() as instance:
        instance.create_run_for_job(
            defs.resolve_job_def("archive_page_backfill_job"),
            run_config={"ops": {"archive_page_backfill": {"config": {"outlet": "hotnews"}}}},
            status=dg.DagsterRunStatus.STARTED,
            tags={"news/archive_outlet": "hotnews"},
        )
        with dg.build_schedule_context(
            instance=instance,
            scheduled_execution_time=scheduled_at,
            repository_def=defs.get_repository_def(),
        ) as context:
            evaluation = archive_page_backfill.scheduled_archive_page_backfill.evaluate_tick(
                context
            )

    assert evaluation.run_requests is not None
    assert len(evaluation.run_requests) == 1
    request = evaluation.run_requests[0]
    assert request.tags["news/archive_outlet"] == "digi24"
    assert request.run_key == "archive-page:digi24:2026-09-27T17:18:00+00:00"


def test_page_schedule_skips_when_archive_is_complete(monkeypatch) -> None:
    monkeypatch.setattr(
        archive_page_backfill, "next_page_window", lambda _outlet, _start, _end: None
    )
    with dg.instance_for_test() as instance:
        with dg.build_schedule_context(
            instance=instance,
            scheduled_execution_time=datetime(2026, 9, 27, 17, 18, tzinfo=UTC),
            repository_def=defs.get_repository_def(),
        ) as context:
            evaluation = archive_page_backfill.scheduled_archive_page_backfill.evaluate_tick(
                context
            )

    assert evaluation.run_requests == []
    assert evaluation.skip_message is not None
