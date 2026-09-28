from datetime import UTC, date, datetime

import dagster as dg

from romanian_news.archive.campaign import ARCHIVE_END, ARCHIVE_OUTLETS, ARCHIVE_START
from romanian_news.worker import archive_capture
from romanian_news.worker.definitions import defs


def test_article_schedule_skips_outlet_with_active_capture(monkeypatch) -> None:
    calls = []

    def next_window(outlet, start, end):
        calls.append((outlet, start, end))
        return date(2025, 9, 27), date(2025, 9, 30)

    monkeypatch.setattr(
        archive_capture,
        "next_capture_window",
        next_window,
    )
    scheduled_at = datetime(2026, 9, 27, 17, 25, tzinfo=UTC)
    with dg.instance_for_test() as instance:
        instance.create_run_for_job(
            defs.resolve_job_def("archive_article_capture_batch"),
            run_config={
                "ops": {
                    "archive_article_capture": {
                        "config": {
                            "outlet": "hotnews",
                            "start": "2025-09-27",
                            "end": "2025-09-30",
                        }
                    }
                }
            },
            status=dg.DagsterRunStatus.STARTED,
            tags={"news/archive_outlet": "hotnews"},
        )
        with dg.build_schedule_context(
            instance=instance,
            scheduled_execution_time=scheduled_at,
            repository_def=defs.get_repository_def(),
        ) as context:
            evaluation = archive_capture.scheduled_archive_article_capture.evaluate_tick(context)

    assert evaluation.run_requests is not None
    assert calls == [(outlet, ARCHIVE_START, ARCHIVE_END) for outlet in ARCHIVE_OUTLETS]
    assert len(evaluation.run_requests) == 1
    request = evaluation.run_requests[0]
    assert request.tags["news/archive_outlet"] == "digi24"
    assert request.run_config["ops"]["archive_article_capture"]["config"]["limit"] == 50


def test_article_schedule_skips_when_capture_is_complete(monkeypatch) -> None:
    monkeypatch.setattr(archive_capture, "next_capture_window", lambda _outlet, _start, _end: None)
    with dg.instance_for_test() as instance:
        with dg.build_schedule_context(
            instance=instance,
            scheduled_execution_time=datetime(2026, 9, 27, 17, 25, tzinfo=UTC),
            repository_def=defs.get_repository_def(),
        ) as context:
            evaluation = archive_capture.scheduled_archive_article_capture.evaluate_tick(context)

    assert evaluation.run_requests == []
    assert evaluation.skip_message is not None
