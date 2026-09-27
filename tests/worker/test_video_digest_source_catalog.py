from datetime import UTC, datetime, timedelta

import psycopg
import pytest

import romanian_news.catalog.schema as news_schema
from romanian_news import BUCHAREST
from romanian_news.catalog.artifacts import artifact_file, canonical_json, sha256
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog.video_digest import claim_slot, schedule_slot
from romanian_news.catalog.video_digest_selection import capture_slot_report_source
from romanian_news.video_digest.models import (
    ClaimedSlot,
    EditionIdentity,
    ScheduledSlot,
    SlotName,
    SlotSkipReason,
    edition_id,
    scheduled_slot_id,
)
from romanian_news.video_digest.orchestration import SourceUnavailable
from romanian_news.worker import video_digest_source
from tests import daily_report_catalog
from tests.contracts.postgres_video_digest_generation import _record_artifact_versions
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog
from tests.worker.conftest import FakeR2Client

requires_postgres = pytest.mark.skipif(
    TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)


def _slot(name: SlotName, scheduled_at: datetime) -> ScheduledSlot:
    return ScheduledSlot(
        slot_id=scheduled_slot_id(name, scheduled_at),
        name=name,
        scheduled_at=scheduled_at,
        bucharest_day=scheduled_at.astimezone(BUCHAREST).date(),
    )


@requires_postgres
def test_midday_slot_skips_as_unchanged_when_an_earlier_edition_never_recorded_a_selection(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_news_catalog_schema()
    morning_at = datetime(2026, 9, 27, 9, tzinfo=UTC)
    midday_at = morning_at + timedelta(hours=1)
    report = daily_report_catalog.daily_report(morning_at.astimezone(BUCHAREST).date())
    canonical = canonical_json(report.model_dump(mode="json"))
    report_file = artifact_file(
        artifact_id=f"news:daily:{report.day.isoformat()}",
        artifact_kind="news_daily_report",
        title=f"Romanian news report for {report.day.isoformat()}",
        content=canonical,
        r2_key=f"news/reports/daily/{report.day.isoformat()}/{sha256(canonical)}.json",
        media_type="application/json",
    )
    monkeypatch.setattr(daily_report_catalog, "REPORT_VERSION", report_file.version_id)
    payload = daily_report_catalog.seed_daily_report(postgres_catalog, report)
    payload_key = f"news/reports/daily/{report.day.isoformat()}/{sha256(payload)}.json"
    fake_r2.objects[payload_key] = payload

    morning = _slot(SlotName.MORNING, morning_at)
    schedule_slot(morning, recorded_at=morning_at)
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report_version, policy_version = _record_artifact_versions(connection, 940, 2)
    legacy_edition = EditionIdentity(
        edition_id=edition_id(report_version, policy_version),
        daily_report_version_id=report_version,
        policy_bundle_version_id=policy_version,
    )
    claimed = claim_slot(
        morning.slot_id,
        legacy_edition,
        owner_token="legacy-morning-run",
        now=morning_at,
        lease_duration=timedelta(hours=1),
    )
    assert isinstance(claimed, ClaimedSlot)
    assert capture_slot_report_source(morning) is not None

    midday = _slot(SlotName.MIDDAY, midday_at)
    request = video_digest_source.resolve_video_digest_run_request(midday, owner_token="midday-run")

    assert request.source == SourceUnavailable(reason=SlotSkipReason.UNCHANGED)
    selections = postgres_catalog.execute(
        "SELECT slot_id, selection, selected_at FROM video_digest_slot_sources "
        "WHERE slot_id IN (%s, %s) ORDER BY slot_id",
        (morning.slot_id, midday.slot_id),
    ).fetchall()
    assert len(selections) == 2
    assert {row["slot_id"] for row in selections} == {morning.slot_id, midday.slot_id}
    assert all(row["selection"] is None and row["selected_at"] is None for row in selections)
