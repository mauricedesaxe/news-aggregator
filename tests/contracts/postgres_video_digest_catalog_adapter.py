from datetime import UTC, datetime

from romanian_news import BUCHAREST
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.video_digest.models import (
    ScheduledSlot,
    SlotName,
    SlotSkipReason,
    scheduled_slot_id,
)
from romanian_news.video_digest.orchestration import (
    CatalogPort,
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
