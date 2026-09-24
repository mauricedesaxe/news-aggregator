from datetime import UTC, date, datetime, timedelta

import pytest

from romanian_news.video_digest.models import (
    ClaimedSlot,
    EditionIdentity,
    ScheduledSlot,
    SlotLease,
    SlotName,
    edition_id,
    scheduled_slot_id,
)
from romanian_news.video_digest.orchestration import CatalogPort, PlanningResume
from romanian_news.worker import video_digest_catalog as adapter_module


def test_catalog_adapter_passes_claim_and_lease_identity_to_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 9, 25, 5, tzinfo=UTC)
    slot = ScheduledSlot(
        slot_id=scheduled_slot_id(SlotName.MORNING, now),
        name=SlotName.MORNING,
        scheduled_at=now,
        bucharest_day=date(2026, 9, 25),
    )
    edition = EditionIdentity(
        edition_id=edition_id("1" * 64, "2" * 64),
        daily_report_version_id="1" * 64,
        policy_bundle_version_id="2" * 64,
    )
    lease = SlotLease(
        slot_id=slot.slot_id,
        edition_id=edition.edition_id,
        owner_token="dagster-run",
        expires_at=now + timedelta(minutes=10),
        claim_count=1,
    )
    seen: list[tuple[object, ...]] = []

    def claim(
        slot_id: str,
        claimed_edition: EditionIdentity,
        *,
        owner_token: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> ClaimedSlot:
        seen.append((slot_id, claimed_edition, owner_token, now, lease_duration))
        return ClaimedSlot(lease=lease)

    monkeypatch.setattr(adapter_module.catalog, "claim_slot", claim)
    monkeypatch.setattr(
        adapter_module.catalog,
        "read_slot_resume_state",
        lambda slot_id: PlanningResume(slot=slot, lease=lease),
    )
    port: CatalogPort = adapter_module.PostgresVideoDigestCatalog()

    result = port.claim(
        slot.slot_id,
        edition,
        owner_token="dagster-run",
        now=now,
        lease_duration=timedelta(minutes=10),
    )

    assert result == ClaimedSlot(lease=lease)
    assert seen == [(slot.slot_id, edition, "dagster-run", now, timedelta(minutes=10))]
    assert port.read(slot.slot_id) == PlanningResume(slot=slot, lease=lease)
