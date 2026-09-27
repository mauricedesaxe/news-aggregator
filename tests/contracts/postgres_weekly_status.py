from __future__ import annotations

import hashlib
import io
from datetime import date, timedelta
from typing import Any, cast

import pytest
from botocore.exceptions import ClientError

from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.weekly_status import (
    WeeklyStatusNotFound,
    WeeklyStatusSummary,
    list_weekly_status,
    publish_weekly_status,
    read_weekly_status,
    read_weekly_status_version,
)
from romanian_news.weekly_status import (
    WeekInput,
    WeekInputDay,
    WeeklyStatusOutput,
)
from romanian_news.weekly_status_generation import GeneratedStatus, build_weekly_status
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog, postgres_catalog_fixture
from tests.reader.test_app import _daily_report

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_DSN is None, reason="NEWS_TEST_POSTGRES_DSN is required"
)

postgres_catalog = postgres_catalog_fixture("weekly_status_contract")
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


def _seed_daily_reference(catalog: PostgresCatalog, day: date) -> str:
    version_id = hashlib.sha256(day.isoformat().encode()).hexdigest()
    catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
        "visibility, created_at) VALUES (%s, 'news_daily_report', %s, 'derived', "
        "'current', 'private', %s) ON CONFLICT DO NOTHING",
        (f"news:daily:{day.isoformat()}", f"Daily report {day.isoformat()}", CAPTURED_AT),
    )
    catalog.execute(
        "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
        "created_at) VALUES (%s, %s, 3, %s, %s) ON CONFLICT DO NOTHING",
        (version_id, f"news:daily:{day.isoformat()}", version_id, CAPTURED_AT),
    )
    catalog.execute(
        "UPDATE artifacts SET current_version_id = %s WHERE id = %s",
        (version_id, f"news:daily:{day.isoformat()}"),
    )
    return version_id


def _week(catalog: PostgresCatalog, present_days: int = 6) -> WeekInput:
    days = []
    for offset in range(7):
        day = WEEK_START + timedelta(days=offset)
        version_id = _seed_daily_reference(catalog, day)
        days.append(
            WeekInputDay(
                day=day,
                report=None
                if offset >= present_days
                else ArtifactReference(
                    artifact_id=f"news:daily:{day.isoformat()}",
                    version_id=version_id,
                    content_digest=version_id,
                    r2_key=f"news/reports/daily/{day.isoformat()}/{version_id}.json",
                ),
            )
        )
    return WeekInput(week_start=WEEK_START, days=tuple(days))


def _draft(sources) -> GeneratedStatus:
    from romanian_news.weekly_status import AreaAssessment, Development

    handles = tuple(source.handle for source in sources)
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


def _output(inputs: WeekInput) -> WeeklyStatusOutput:
    reports = {
        slot.report.version_id: _daily_report().model_copy(update={"day": slot.day})
        for slot in inputs.days
        if slot.report is not None
    }
    return build_weekly_status(
        inputs,
        reports,
        compose=lambda _inputs, sources: _draft(sources),
        verify=lambda *_args: None,
    )


def test_publish_reads_back_and_retries_reuse_the_saved_version(
    postgres_catalog: PostgresCatalog, r2: _R2Library
) -> None:
    inputs = _week(postgres_catalog)
    output = _output(inputs)

    publication = publish_weekly_status(output, "test-implementation")

    assert publication.status == "published"
    version_id, read = read_weekly_status(WEEK_START)
    assert version_id == publication.version_id
    assert read == output.read
    assert list_weekly_status() == (
        WeeklyStatusSummary(week_start=WEEK_START, version_id=publication.version_id),
    )
    assert read_weekly_status_version(publication.version_id) == (version_id, output.read)
    published_bytes = next(iter(r2.objects.values()))
    assert hashlib.sha256(published_bytes).hexdigest() == output.content_digest
    input_rows = postgres_catalog.execute(
        "SELECT role, artifact_version_id FROM run_inputs WHERE run_id = %s ORDER BY position",
        (publication.run_id,),
    ).fetchall()
    assert [(str(row["role"]), str(row["artifact_version_id"])) for row in input_rows] == [
        ("daily_report", slot.report.version_id) for slot in inputs.days if slot.report is not None
    ]

    reworded = WeeklyStatusOutput.model_construct(
        request_id=output.request_id,
        read=output.read,
        content_digest=hashlib.sha256(b"new model wording").hexdigest(),
        content=b"new model wording",
    )
    retry = publish_weekly_status(reworded, "test-implementation")

    assert retry.status == "published"
    assert retry.version_id == publication.version_id
    assert read_weekly_status(WEEK_START) == (publication.version_id, output.read)


def test_missing_weeks_and_versions_are_reported(postgres_catalog: PostgresCatalog) -> None:
    with pytest.raises(WeeklyStatusNotFound):
        read_weekly_status(WEEK_START)
    with pytest.raises(WeeklyStatusNotFound):
        read_weekly_status_version("e" * 64)
    assert list_weekly_status() == ()
