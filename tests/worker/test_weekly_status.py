from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import dagster as dg
import pytest

from romanian_news import BUCHAREST, weekly_status_generation
from romanian_news.catalog.weekly_status import publish_weekly_status, read_weekly_status
from romanian_news.weekly_status import (
    MIN_REPORT_DAYS,
    WEEKLY_STATUS_POLICY,
    AreaAssessment,
    Development,
    WeekInput,
)
from romanian_news.weekly_status_generation import GeneratedStatus, StatusAudit
from romanian_news.worker import weekly_status
from romanian_news.worker.definitions import defs
from tests import daily_report_catalog
from tests.daily_report_catalog import daily_report, seed_daily_report
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog
from tests.worker.conftest import FakeR2Client

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)

WEEK = date(2026, 9, 14)
WEEKLY_ARTIFACT = f"news:weekly_status:{WEEK.isoformat()}"
HISTORICAL_WEEK = date(2026, 6, 1)


class StatusModelStub:
    def __init__(self) -> None:
        self.compositions = 0

    def __call__(self, **kwargs: Any) -> object:
        if kwargs["operation"] == "news.verify_weekly_status":
            audit = StatusAudit(supported=True)
            return SimpleNamespace(value=kwargs["parse"](audit.model_dump_json()))
        self.compositions += 1
        context = json.loads(kwargs["initial_messages"][1].content)
        handle = context["sources"][0]["handle"]
        draft = GeneratedStatus(
            developments=(
                Development(
                    title=f"Model wording {self.compositions}",
                    what_changed="Budget policy stayed central across the whole week.",
                    source_handles=(handle,),
                ),
            ),
            assessments=(
                AreaAssessment(
                    area="overall",
                    judgment="The week centered on budget policy.",
                    what_changed="The draft budget framed each sitting day.",
                    why_it_matters="Fiscal decisions shape the autumn agenda.",
                    source_handles=(handle,),
                    coverage="limited",
                    coverage_note="Only ranked daily excerpts were available.",
                ),
                *(
                    AreaAssessment(
                        area=area,
                        judgment=None,
                        what_changed=None,
                        why_it_matters=None,
                        source_handles=(),
                        coverage="insufficient",
                        coverage_note=f"Not enough cited {area} evidence in the daily reads.",
                    )
                    for area in ("economy", "politics", "society")
                ),
            ),
        )
        return SimpleNamespace(value=kwargs["parse"](draft.model_dump_json()))


def _install_status_model_stub(monkeypatch: pytest.MonkeyPatch) -> StatusModelStub:
    stub = StatusModelStub()
    monkeypatch.setattr(weekly_status_generation, "run_corrected_structured_openrouter", stub)
    return stub


def _seed_published_daily_reports(
    monkeypatch: pytest.MonkeyPatch,
    catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    week_start: date,
    day_count: int,
) -> None:
    for position in range(day_count):
        for name in ("RUN_ID", "REPORT_VERSION", "INPUT_VERSION", "FILE_ID"):
            seed_id = hashlib.sha256(f"weekly-status-seed:{name}:{position}".encode()).hexdigest()
            monkeypatch.setattr(daily_report_catalog, name, seed_id)
        day = week_start + timedelta(days=position)
        payload = seed_daily_report(catalog, daily_report(day))
        digest = hashlib.sha256(payload).hexdigest()
        key = f"news/reports/daily/{day.isoformat()}/{digest}.json"
        fake_r2.objects[key] = payload
        fake_r2.metadata[key] = {"sha256": digest}


def _weekly_status_row_counts(catalog: PostgresCatalog) -> dict[str, int]:
    artifact_count = catalog.execute(
        "SELECT count(*) AS count FROM artifacts WHERE kind = 'news_weekly_read'"
    ).fetchone()
    version_count = catalog.execute(
        "SELECT count(*) AS count FROM artifact_versions WHERE artifact_id = %s",
        (WEEKLY_ARTIFACT,),
    ).fetchone()
    run_count = catalog.execute(
        "SELECT count(*) AS count FROM runs WHERE operation_key = 'news.publish_weekly_status'"
    ).fetchone()
    assert artifact_count is not None
    assert version_count is not None
    assert run_count is not None
    return {
        "artifacts": artifact_count["count"],
        "versions": version_count["count"],
        "runs": run_count["count"],
    }


def _weekly_r2_keys(fake_r2: FakeR2Client, week_start: date) -> list[str]:
    prefix = f"news/reports/weekly-status/{week_start.isoformat()}/"
    return sorted(key for key in fake_r2.objects if key.startswith(prefix))


def test_recent_completed_weeks_exclude_current_week() -> None:
    assert weekly_status.recent_completed_week_starts(date(2026, 9, 27), 3) == (
        date(2026, 9, 14),
        date(2026, 9, 7),
        date(2026, 8, 31),
    )


def test_fresh_refresh_publishes_a_completed_weekly_read_end_to_end(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_published_daily_reports(monkeypatch, postgres_catalog, fake_r2, WEEK, MIN_REPORT_DAYS)
    _install_status_model_stub(monkeypatch)

    result = weekly_status.refresh_weekly_status(WEEK, "git:test")

    assert result == "published"
    status_rows = postgres_catalog.execute(
        "SELECT status FROM runs WHERE operation_key = 'news.publish_weekly_status'"
    ).fetchall()
    assert [row["status"] for row in status_rows] == ["completed"]
    artifact_rows = postgres_catalog.execute(
        "SELECT current_version_id FROM artifacts WHERE id = %s AND kind = 'news_weekly_read'",
        (WEEKLY_ARTIFACT,),
    ).fetchall()
    assert len(artifact_rows) == 1
    published_version = artifact_rows[0]["current_version_id"]
    version_count = postgres_catalog.execute(
        "SELECT count(*) AS count FROM artifact_versions WHERE artifact_id = %s",
        (WEEKLY_ARTIFACT,),
    ).fetchone()
    assert version_count is not None
    assert version_count["count"] == 1
    file_rows = postgres_catalog.execute(
        "SELECT r2_key, content_digest FROM artifact_files WHERE artifact_version_id = %s",
        (published_version,),
    ).fetchall()
    assert len(file_rows) == 1

    version_id, read = read_weekly_status(WEEK)

    assert version_id == published_version
    assert read.week_start == WEEK
    assert read.week_end == WEEK + timedelta(days=6)
    assert sum(slot.report is not None for slot in read.days) == MIN_REPORT_DAYS
    weekly_keys = _weekly_r2_keys(fake_r2, WEEK)
    assert weekly_keys == [file_rows[0]["r2_key"]]
    stored = fake_r2.objects[weekly_keys[0]]
    assert hashlib.sha256(stored).hexdigest() == file_rows[0]["content_digest"]
    assert json.loads(stored)["developments"][0]["title"] == "Model wording 1"


def test_second_refresh_with_unchanged_daily_reports_returns_current_without_new_versions(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_published_daily_reports(monkeypatch, postgres_catalog, fake_r2, WEEK, MIN_REPORT_DAYS)
    stub = _install_status_model_stub(monkeypatch)
    assert weekly_status.refresh_weekly_status(WEEK, "git:test") == "published"

    result = weekly_status.refresh_weekly_status(WEEK, "git:test")

    assert result == "current"
    assert stub.compositions == 1
    assert _weekly_status_row_counts(postgres_catalog) == {
        "artifacts": 1,
        "versions": 1,
        "runs": 1,
    }
    assert len(_weekly_r2_keys(fake_r2, WEEK)) == 1


def test_refresh_regenerates_when_the_saved_policy_changes(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_published_daily_reports(monkeypatch, postgres_catalog, fake_r2, WEEK, MIN_REPORT_DAYS)
    stale_inputs = WeekInput(
        week_start=WEEK,
        policy="completed-week-ranked-sources-v1",
        days=weekly_status.read_week_input(WEEK).days,
    )
    reports = {
        slot.report.version_id: daily_report(slot.day)
        for slot in stale_inputs.days
        if slot.report is not None
    }
    stub = _install_status_model_stub(monkeypatch)
    stale_output = weekly_status_generation.build_weekly_status(stale_inputs, reports)
    stale_publication = publish_weekly_status(stale_output, "git:test")

    assert weekly_status.refresh_weekly_status(WEEK, "git:test") == "published"
    assert stub.compositions == 2
    version, read = read_weekly_status(WEEK)
    assert version != stale_publication.version_id
    assert read.policy == WEEKLY_STATUS_POLICY


def test_retry_after_head_loss_reuses_the_saved_weekly_output_when_model_text_differs(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_published_daily_reports(monkeypatch, postgres_catalog, fake_r2, WEEK, MIN_REPORT_DAYS)
    stub = _install_status_model_stub(monkeypatch)
    assert weekly_status.refresh_weekly_status(WEEK, "git:test") == "published"
    saved_version, saved_read = read_weekly_status(WEEK)
    postgres_catalog.execute(
        "UPDATE artifacts SET current_version_id = NULL, current_run_id = NULL WHERE id = %s",
        (WEEKLY_ARTIFACT,),
    )

    result = weekly_status.refresh_weekly_status(WEEK, "git:test")

    assert result == "published"
    assert stub.compositions == 2
    assert _weekly_status_row_counts(postgres_catalog) == {
        "artifacts": 1,
        "versions": 1,
        "runs": 1,
    }
    head = postgres_catalog.execute(
        "SELECT current_version_id FROM artifacts WHERE id = %s", (WEEKLY_ARTIFACT,)
    ).fetchone()
    assert head is not None
    assert head["current_version_id"] == saved_version
    version_id, read = read_weekly_status(WEEK)
    assert version_id == saved_version
    assert read == saved_read
    assert read.developments[0].title == "Model wording 1"


def test_refresh_returns_insufficient_and_publishes_nothing_below_five_daily_reports(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_published_daily_reports(monkeypatch, postgres_catalog, fake_r2, WEEK, MIN_REPORT_DAYS - 1)
    _install_status_model_stub(monkeypatch)

    result = weekly_status.refresh_weekly_status(WEEK, "git:test")

    assert result == "insufficient"
    assert _weekly_status_row_counts(postgres_catalog) == {
        "artifacts": 0,
        "versions": 0,
        "runs": 0,
    }
    assert _weekly_r2_keys(fake_r2, WEEK) == []


def test_historical_schedule_tick_targets_the_first_week_with_five_daily_reports(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_published_daily_reports(
        monkeypatch, postgres_catalog, fake_r2, HISTORICAL_WEEK, MIN_REPORT_DAYS
    )

    with dg.instance_for_test() as instance:
        with dg.build_schedule_context(
            instance=instance,
            scheduled_execution_time=datetime(2026, 9, 27, 20, 20, tzinfo=UTC),
            repository_def=defs.get_repository_def(),
        ) as context:
            evaluation = weekly_status.scheduled_historical_weekly_status.evaluate_tick(context)

    assert evaluation.run_requests is not None
    assert len(evaluation.run_requests) == 1
    request = evaluation.run_requests[0]
    assert request.tags["news/archive_week"] == HISTORICAL_WEEK.isoformat()
    assert request.run_config == {
        "ops": {
            "refresh_weekly_statuses": {"config": {"week_starts": [HISTORICAL_WEEK.isoformat()]}}
        }
    }


def test_historical_schedule_tick_skips_when_the_ready_week_is_already_published(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_published_daily_reports(
        monkeypatch, postgres_catalog, fake_r2, HISTORICAL_WEEK, MIN_REPORT_DAYS
    )
    _install_status_model_stub(monkeypatch)
    assert weekly_status.refresh_weekly_status(HISTORICAL_WEEK, "git:test") == "published"

    with dg.instance_for_test() as instance:
        with dg.build_schedule_context(
            instance=instance,
            scheduled_execution_time=datetime(2026, 9, 27, 20, 20, tzinfo=UTC),
            repository_def=defs.get_repository_def(),
        ) as context:
            evaluation = weekly_status.scheduled_historical_weekly_status.evaluate_tick(context)

    assert not evaluation.run_requests
    assert evaluation.skip_message == "No historical week has new published daily reports."


def test_weekly_schedule_tick_refreshes_the_four_most_recent_completed_mondays() -> None:
    scheduled_at = datetime(2026, 9, 27, 11, 0, tzinfo=BUCHAREST)

    with dg.instance_for_test() as instance:
        with dg.build_schedule_context(
            instance=instance,
            scheduled_execution_time=scheduled_at,
            repository_def=defs.get_repository_def(),
        ) as context:
            evaluation = weekly_status.scheduled_weekly_status.evaluate_tick(context)

    assert evaluation.run_requests is not None
    (request,) = evaluation.run_requests
    assert request.run_key == "weekly-status:2026-09-27"
    assert request.run_config == {
        "ops": {
            "refresh_weekly_statuses": {
                "config": {
                    "week_starts": [
                        "2026-09-14",
                        "2026-09-07",
                        "2026-08-31",
                        "2026-08-24",
                    ]
                }
            }
        }
    }
