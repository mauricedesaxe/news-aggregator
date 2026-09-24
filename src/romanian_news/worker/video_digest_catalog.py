from datetime import datetime, timedelta

from romanian_news.catalog import video_digest as catalog
from romanian_news.video_digest.models import (
    BusySlot,
    ClaimedSlot,
    EditionIdentity,
    ScheduledSlot,
    SkippedSlot,
    SlotId,
    SlotLease,
    SlotSkipReason,
    TerminalSlot,
)
from romanian_news.video_digest.orchestration import SlotResumeState


class PostgresVideoDigestCatalog:
    def schedule(self, slot: ScheduledSlot, *, recorded_at: datetime) -> None:
        catalog.schedule_slot(slot, recorded_at=recorded_at)

    def read(self, slot_id: str) -> SlotResumeState:
        return catalog.read_slot_resume_state(SlotId(slot_id))

    def skip(self, slot_id: str, reason: SlotSkipReason, *, recorded_at: datetime) -> SkippedSlot:
        return catalog.skip_slot(SlotId(slot_id), reason, recorded_at=recorded_at)

    def claim(
        self,
        slot_id: str,
        edition: EditionIdentity,
        *,
        owner_token: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> ClaimedSlot | SkippedSlot | TerminalSlot:
        return catalog.claim_slot(
            SlotId(slot_id),
            edition,
            owner_token=owner_token,
            now=now,
            lease_duration=lease_duration,
        )

    def reacquire(
        self,
        slot_id: str,
        *,
        owner_token: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> ClaimedSlot | BusySlot | TerminalSlot:
        return catalog.reacquire_slot(
            SlotId(slot_id),
            owner_token=owner_token,
            now=now,
            lease_duration=lease_duration,
        )

    def renew(self, lease: SlotLease, *, now: datetime, lease_duration: timedelta) -> SlotLease:
        return catalog.renew_slot(lease, now=now, lease_duration=lease_duration)

    def fail_deadline(self, lease: SlotLease, *, recorded_at: datetime) -> TerminalSlot:
        return catalog.fail_slot_deadline(lease, recorded_at=recorded_at)
