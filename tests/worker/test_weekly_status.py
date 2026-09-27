from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import dagster as dg
import pytest
from botocore.exceptions import ClientError

from romanian_news.analysis import corrected_structured
from romanian_news.catalog.weekly_status import (
    list_weekly_status,
    publish_weekly_status,
    read_weekly_status,
)
from romanian_news.weekly_status import (
    WEEKLY_STATUS_POLICY,
    AreaAssessment,
    Development,
    WeekInput,
)
from romanian_news.weekly_status_generation import (
    GeneratedStatus,
    build_weekly_status,
    read_week_input,
)
from romanian_news.worker import weekly_status
from romanian_news.worker.definitions import defs
from romanian_news.worker.weekly_status import (
    recent_completed_week_starts,
    refresh_weekly_status,
    scheduled_weekly_status,
)
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog, postgres_catalog_fixture
from tests.reader.test_app import _daily_report

postgres_catalog = postgres_catalog_fixture("news_weekly_worker")
WEEK_START = date(2026, 9, 14)
CAPTURED_AT = "2026-09-21T06:00:00+00:00"


class _R2Library:
    """In-memory stand-in for the R2 S3 client at the storage network boundary."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not found"}}, "HeadObject")
        return {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_kwargs: object) -> None:
        self.objects[Key] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}


@pytest.fixture
def r2(monkeypatch: pytest.MonkeyPatch) -> _R2Library:
    library = _R2Library()
    monkeypatch.setattr("romanian_news.storage._r2_client", lambda: cast(Any, library))
    return library


def test_recent_completed_weeks_exclude_current_week() -> None:
    assert recent_completed_week_starts(date(2026, 9, 27), 3) == (
        date(2026, 9, 14),
        date(2026, 9, 7),
        date(2026, 8, 31),
    )


def test_schedule_requests_the_last_four_completed_mondays() -> None:
    scheduled_at = datetime(2026, 9, 27, 8, 0, tzinfo=UTC)

    with dg.build_schedule_context(scheduled_execution_time=scheduled_at) as context:
        evaluation = scheduled_weekly_status.evaluate_tick(context)

    assert evaluation.run_requests is not None
    (request,) = evaluation.run_requests
    assert request.run_key == "weekly-status:2026-09-27"
    weeks = request.run_config["ops"]["refresh_weekly_statuses"]["config"]["week_starts"]
    assert [date.fromisoformat(week) for week in weeks] == [
        date(2026, 9, 14),
        date(2026, 9, 7),
        date(2026, 8, 31),
        date(2026, 8, 24),
    ]


def _seed_daily_report(catalog: PostgresCatalog, library: _R2Library, day: date) -> None:
    payload = _daily_report().model_copy(update={"day": day}).model_dump_json().encode()
    digest = hashlib.sha256(payload).hexdigest()
    artifact_id = f"news:daily:{day.isoformat()}"
    version_id = hashlib.sha256(day.isoformat().encode()).hexdigest()
    r2_key = f"news/reports/daily/{day.isoformat()}/{digest}.json"
    library.objects[r2_key] = payload
    catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
        "visibility, created_at) VALUES (%s, 'news_daily_report', %s, 'derived', "
        "'current', 'private', %s) ON CONFLICT DO NOTHING",
        (artifact_id, f"Daily report {day.isoformat()}", CAPTURED_AT),
    )
    catalog.execute(
        "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
        "created_at) VALUES (%s, %s, 3, %s, %s) ON CONFLICT DO NOTHING",
        (version_id, artifact_id, digest, CAPTURED_AT),
    )
    catalog.execute(
        "INSERT INTO artifact_files (id, artifact_version_id, r2_key, media_type, "
        "content_digest, byte_size, row_count, schema_fingerprint) "
        "VALUES (%s, %s, %s, 'application/json', %s, %s, NULL, NULL)",
        (
            hashlib.sha256(r2_key.encode()).hexdigest(),
            version_id,
            r2_key,
            digest,
            len(payload),
        ),
    )
    catalog.execute(
        "UPDATE artifacts SET current_version_id = %s WHERE id = %s", (version_id, artifact_id)
    )


def _seed_week(catalog: PostgresCatalog, library: _R2Library, present_days: int) -> None:
    for offset in range(present_days):
        _seed_daily_report(catalog, library, WEEK_START + timedelta(days=offset))


def _weekly_provider(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    requests: list[dict[str, object]] = []

    def create(**kwargs):
        requests.append(kwargs)
        user_content = kwargs["messages"][1]["content"]
        handles = tuple(source["handle"] for source in json.loads(user_content)["sources"])
        if kwargs["response_format"]["json_schema"]["name"] == "weekly_status":
            content = _limited_draft(handles).model_dump_json()
        else:
            content = json.dumps({"supported": True, "problems": []})
        return SimpleNamespace(
            id=f"response-{len(requests)}",
            model=str(kwargs["model"]),
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=(SimpleNamespace(message=SimpleNamespace(content=content)),),
            model_dump=lambda *, mode: {"id": f"response-{len(requests)}"},
        )

    monkeypatch.setattr(
        corrected_structured,
        "openrouter_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    monkeypatch.setattr(corrected_structured.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        corrected_structured,
        "record_model_attempt",
        lambda _response, **_kwargs: SimpleNamespace(attempt_id="1" * 64, response_id="response"),
    )
    return requests


def _limited_draft(handles: tuple[str, ...]) -> GeneratedStatus:
    return GeneratedStatus(
        developments=(
            Development(
                title="Budget debate",
                what_changed="The budget discussion continued during the week.",
                source_handles=handles[:2],
            ),
        ),
        assessments=(
            AreaAssessment(
                area="overall",
                judgment="The budget debate dominated the selected reports.",
                what_changed="The draft entered public debate.",
                why_it_matters="The budget could affect national spending.",
                source_handles=handles[:2],
                coverage="limited",
                coverage_note="Selected report highlights only.",
            ),
            *(
                AreaAssessment(
                    area=area,
                    judgment=None,
                    what_changed=None,
                    why_it_matters=None,
                    source_handles=(),
                    coverage="insufficient",
                    coverage_note="Not enough selected reporting.",
                )
                for area in ("economy", "politics", "society")
            ),
        ),
    )


@pytest.mark.skipif(TEST_POSTGRES_DSN is None, reason="NEWS_TEST_POSTGRES_DSN is required")
def test_refresh_publishes_once_then_stays_current_without_model_spend(
    monkeypatch, postgres_catalog: PostgresCatalog, r2: _R2Library
) -> None:
    _seed_week(postgres_catalog, r2, present_days=6)
    provider_requests = _weekly_provider(monkeypatch)

    assert refresh_weekly_status(WEEK_START, "test-implementation") == "published"
    assert len(provider_requests) == 2
    first_version, read = read_weekly_status(WEEK_START)

    assert refresh_weekly_status(WEEK_START, "test-implementation") == "current"
    assert len(provider_requests) == 2
    assert read_weekly_status(WEEK_START) == (first_version, read)


@pytest.mark.skipif(TEST_POSTGRES_DSN is None, reason="NEWS_TEST_POSTGRES_DSN is required")
def test_refresh_requires_five_daily_reports_and_writes_nothing(
    monkeypatch, postgres_catalog: PostgresCatalog, r2: _R2Library
) -> None:
    _seed_week(postgres_catalog, r2, present_days=4)
    provider_requests = _weekly_provider(monkeypatch)

    assert refresh_weekly_status(WEEK_START, "test-implementation") == "insufficient"
    assert provider_requests == []
    assert list_weekly_status() == ()
    inputs = read_week_input(WEEK_START)
    assert inputs.available_days == 4


@pytest.mark.skipif(TEST_POSTGRES_DSN is None, reason="NEWS_TEST_POSTGRES_DSN is required")
def test_refresh_regenerates_when_the_saved_policy_changes(
    monkeypatch, postgres_catalog: PostgresCatalog, r2: _R2Library
) -> None:
    _seed_week(postgres_catalog, r2, present_days=6)
    stale_policy = "completed-week-ranked-sources-v1"
    stale_inputs = WeekInput(
        week_start=WEEK_START,
        policy=stale_policy,
        days=read_week_input(WEEK_START).days,
    )
    reports = {
        slot.report.version_id: _daily_report().model_copy(update={"day": slot.day})
        for slot in stale_inputs.days
        if slot.report is not None
    }
    stale_output = build_weekly_status(
        stale_inputs,
        reports,
        compose=lambda _inputs, sources: _limited_draft(tuple(source.handle for source in sources)),
        verify=lambda *_args: None,
    )
    stale_publication = publish_weekly_status(stale_output, "test-implementation")
    provider_requests = _weekly_provider(monkeypatch)

    result = refresh_weekly_status(WEEK_START, "test-implementation")

    assert result == "published"
    assert provider_requests
    version, read = read_weekly_status(WEEK_START)
    assert version != stale_publication.version_id
    assert read.policy == WEEKLY_STATUS_POLICY


def test_historical_schedule_selects_first_week_with_five_reports(monkeypatch) -> None:
    first_ready = date(2025, 9, 29)
    monkeypatch.setattr(
        weekly_status,
        "read_week_input",
        lambda week: SimpleNamespace(available_days=5 if week == first_ready else 0),
    )
    monkeypatch.setattr(
        weekly_status,
        "read_weekly_status",
        lambda _week: (_ for _ in ()).throw(weekly_status.WeeklyStatusNotFound()),
    )
    assert weekly_status.next_historical_week(date(2026, 9, 27)) == first_ready

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
    assert request.tags["news/archive_week"] == first_ready.isoformat()
    assert request.run_config == {
        "ops": {"refresh_weekly_statuses": {"config": {"week_starts": ["2025-09-29"]}}}
    }


def test_historical_schedule_skips_unchanged_week(monkeypatch) -> None:
    ready = date(2025, 9, 29)
    inputs = SimpleNamespace(available_days=5, policy="policy", days=("exact",))
    monkeypatch.setattr(
        weekly_status,
        "read_week_input",
        lambda week: inputs if week == ready else SimpleNamespace(available_days=0),
    )
    monkeypatch.setattr(
        weekly_status,
        "read_weekly_status",
        lambda _week: ("v" * 64, SimpleNamespace(policy="policy", days=("exact",))),
    )
    assert weekly_status.next_historical_week(date(2026, 9, 27)) is None
