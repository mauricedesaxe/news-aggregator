from datetime import UTC, datetime, timedelta

import psycopg

from romanian_news import BUCHAREST
from romanian_news.catalog import schema as news_schema
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.video_digest.models import (
    ClaimedSlot,
    EditionIdentity,
    ScheduledSlot,
    SlotName,
    SlotSkipReason,
    edition_id,
    scheduled_slot_id,
)
from romanian_news.video_digest.orchestration import (
    CatalogPort,
    PlanningResume,
    ScheduledResume,
    SkippedResume,
)
from romanian_news.worker.video_digest_catalog import PostgresVideoDigestCatalog


def test_catalog_adapter_reads_the_persisted_slot_lifecycle(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    scheduled_at = datetime(2099, 9, 20, 5, tzinfo=UTC)
    slot = ScheduledSlot(
        slot_id=scheduled_slot_id(SlotName.MORNING, scheduled_at),
        name=SlotName.MORNING,
        scheduled_at=scheduled_at,
        bucharest_day=scheduled_at.astimezone(BUCHAREST).date(),
    )
    port: CatalogPort = PostgresVideoDigestCatalog()

    port.schedule(slot, recorded_at=scheduled_at)
    assert port.read(slot.slot_id) == ScheduledResume(slot=slot)

    skipped = port.skip(
        slot.slot_id,
        SlotSkipReason.SOURCE_MISSING,
        recorded_at=scheduled_at,
    )
    assert skipped.reason is SlotSkipReason.SOURCE_MISSING
    assert port.read(slot.slot_id) == SkippedResume(slot=slot, reason=SlotSkipReason.SOURCE_MISSING)


def test_catalog_adapter_claims_a_scheduled_slot(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        version_ids = []
        for value in (300, 301):
            version_id = f"{value:064x}"
            connection.execute(
                "INSERT INTO artifacts "
                "(id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
                "VALUES (%s, 'test', 'Test artifact', 'test', 'active', 'private', "
                "CURRENT_TIMESTAMP)",
                (f"artifact-{version_id}",),
            )
            connection.execute(
                "INSERT INTO artifact_versions "
                "(id, artifact_id, schema_version, content_digest, created_at) "
                "VALUES (%s, %s, 1, %s, CURRENT_TIMESTAMP)",
                (version_id, f"artifact-{version_id}", version_id),
            )
            version_ids.append(version_id)
    edition = EditionIdentity(
        edition_id=edition_id(version_ids[0], version_ids[1]),
        daily_report_version_id=version_ids[0],
        policy_bundle_version_id=version_ids[1],
    )
    scheduled_at = datetime(2099, 9, 21, 5, tzinfo=UTC)
    slot = ScheduledSlot(
        slot_id=scheduled_slot_id(SlotName.MIDDAY, scheduled_at),
        name=SlotName.MIDDAY,
        scheduled_at=scheduled_at,
        bucharest_day=scheduled_at.astimezone(BUCHAREST).date(),
    )
    port: CatalogPort = PostgresVideoDigestCatalog()
    port.schedule(slot, recorded_at=datetime.now(UTC))

    claimed = port.claim(
        slot.slot_id,
        edition,
        owner_token="dagster-run",
        now=datetime.now(UTC),
        lease_duration=timedelta(minutes=10),
    )

    assert isinstance(claimed, ClaimedSlot)
    assert claimed.lease.slot_id == slot.slot_id
    assert claimed.lease.edition_id == edition.edition_id
    assert claimed.lease.owner_token == "dagster-run"
    resume = port.read(slot.slot_id)
    assert isinstance(resume, PlanningResume)
    assert resume.lease == claimed.lease
