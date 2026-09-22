from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from threading import Barrier
from typing import Any

import psycopg
import pytest

import romanian_news.catalog.schema as news_schema
import romanian_news.catalog.video_digest as video_digest_catalog
from romanian_news import BUCHAREST
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.video_digest.errors import (
    VideoDigestCheckpointConflictError,
    VideoDigestLeaseLostError,
)
from romanian_news.video_digest.models import (
    BusySlot,
    ClaimedSlot,
    ClaimResult,
    EditionIdentity,
    ScheduledSlot,
    SkippedSlot,
    SlotId,
    SlotLease,
    SlotName,
    SlotSkipReason,
    TerminalSlot,
    TerminalSlotState,
    edition_id,
    planned_story_id,
    scheduled_slot_id,
)

RECORDED_AT = datetime.now(UTC)
MORNING_AT = datetime(2099, 9, 20, 6, tzinfo=UTC)
MIDDAY_AT = datetime(2099, 9, 20, 9, tzinfo=UTC)


def _sha256_id(value: int) -> str:
    return f"{value:064x}"


def _slot(name: SlotName, scheduled_at: datetime) -> ScheduledSlot:
    return ScheduledSlot(
        slot_id=scheduled_slot_id(name, scheduled_at),
        name=name,
        scheduled_at=scheduled_at,
        bucharest_day=scheduled_at.astimezone(BUCHAREST).date(),
    )


def _edition(report: str, policy: str) -> EditionIdentity:
    return EditionIdentity(
        edition_id=edition_id(report, policy),
        daily_report_version_id=report,
        policy_bundle_version_id=policy,
    )


def _claim(
    slot_id: SlotId,
    identity: EditionIdentity,
    *,
    owner_token: str,
    lease_duration: timedelta = timedelta(hours=1),
) -> ClaimResult:
    return video_digest_catalog.claim_slot(
        slot_id,
        identity,
        owner_token=owner_token,
        now=datetime.now(UTC),
        lease_duration=lease_duration,
    )


def _claim_lease(
    slot_id: SlotId,
    identity: EditionIdentity,
    *,
    owner_token: str,
    lease_duration: timedelta = timedelta(hours=1),
) -> SlotLease:
    result = _claim(slot_id, identity, owner_token=owner_token, lease_duration=lease_duration)
    assert isinstance(result, ClaimedSlot)
    return result.lease


def _reacquire(
    slot_id: SlotId,
    *,
    owner_token: str,
    lease_duration: timedelta = timedelta(hours=1),
) -> ClaimedSlot | BusySlot | TerminalSlot:
    return video_digest_catalog.reacquire_slot(
        slot_id,
        owner_token=owner_token,
        now=datetime.now(UTC),
        lease_duration=lease_duration,
    )


def _record_artifact_versions(
    connection: psycopg.Connection[Any], start: int, count: int
) -> tuple[str, ...]:
    version_ids = tuple(_sha256_id(value) for value in range(start, start + count))
    for version_id in version_ids:
        artifact_id = f"artifact-{version_id}"
        connection.execute(
            "INSERT INTO artifacts "
            "(id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
            "VALUES (%s, 'test', 'Test artifact', 'test', 'active', 'private', "
            "CURRENT_TIMESTAMP)",
            (artifact_id,),
        )
        connection.execute(
            "INSERT INTO artifact_versions "
            "(id, artifact_id, schema_version, content_digest, created_at) "
            "VALUES (%s, %s, 1, %s, CURRENT_TIMESTAMP)",
            (version_id, artifact_id, version_id),
        )
    return version_ids


def _insert_edition(
    connection: psycopg.Connection[Any], edition_id: str, report: str, policy: str
) -> None:
    connection.execute(
        "INSERT INTO video_digest_editions "
        "(edition_id, daily_report_version_id, policy_bundle_version_id, subtitle_state, "
        "created_at, updated_at) VALUES (%s, %s, %s, 'pending', CURRENT_TIMESTAMP, "
        "CURRENT_TIMESTAMP)",
        (edition_id, report, policy),
    )


def _insert_slot_row(
    connection: psycopg.Connection[Any],
    slot_id: SlotId,
    name: str,
    scheduled_at: datetime,
) -> None:
    connection.execute(
        "INSERT INTO video_digest_slots "
        "(slot_id, name, scheduled_at, bucharest_day, stage, claim_count, created_at, "
        "updated_at) VALUES (%s, %s, %s, %s, 'scheduled', 0, CURRENT_TIMESTAMP, "
        "CURRENT_TIMESTAMP)",
        (slot_id, name, scheduled_at, scheduled_at.astimezone(BUCHAREST).date()),
    )


def _insert_story(
    connection: psycopg.Connection[Any],
    story_id: str,
    edition_id: str,
    position: int,
    subject_id: str,
) -> None:
    connection.execute(
        "INSERT INTO video_digest_stories "
        "(story_id, edition_id, position, report_subject_id, title, mandatory, "
        "requested_duration_ms, stage, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, 'Story', TRUE, 1000, 'planned', CURRENT_TIMESTAMP, "
        "CURRENT_TIMESTAMP)",
        (story_id, edition_id, position, subject_id),
    )


def _insert_generation_request(
    connection: psycopg.Connection[Any],
    request_id: str,
    edition_id: str,
    request_version: str,
) -> None:
    row = connection.execute(
        "SELECT slot.scheduled_at, slot.bucharest_day, story.story_id, "
        "edition.policy_bundle_version_id FROM video_digest_slots AS slot "
        "JOIN video_digest_editions AS edition ON edition.edition_id = slot.edition_id "
        "JOIN video_digest_stories AS story ON story.edition_id = edition.edition_id "
        "AND story.position = 0 WHERE edition.edition_id = %s",
        (edition_id,),
    ).fetchone()
    assert row is not None
    scheduled_at, bucharest_day, story_id, policy_version = row
    with connection.transaction():
        connection.execute(
            "INSERT INTO video_digest_generation_requests "
            "(request_id, edition_id, story_position, attempt_index, "
            "request_artifact_version_id, generation_policy_artifact_version_id, "
            "reserved_cost_usd, deadline_at, stage, cost_kind, created_at, updated_at) "
            "VALUES (%s, %s, 0, 0, %s, %s, 3.25632, %s + INTERVAL '90 minutes', "
            "'pending', 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (request_id, edition_id, request_version, policy_version, scheduled_at),
        )
        connection.execute(
            "INSERT INTO video_digest_generation_reservations "
            "(request_id, scope_kind, scope_key, limit_usd, reserved_usd, "
            "generation_policy_artifact_version_id, created_at) VALUES "
            "(%s, 'story', %s, 7, 3.25632, %s, CURRENT_TIMESTAMP), "
            "(%s, 'edition', %s, 7, 3.25632, %s, CURRENT_TIMESTAMP), "
            "(%s, 'bucharest_day', %s, 150, 3.25632, %s, CURRENT_TIMESTAMP), "
            "(%s, 'calendar_month', %s, 1000, 3.25632, %s, CURRENT_TIMESTAMP)",
            (
                request_id,
                story_id,
                policy_version,
                request_id,
                edition_id,
                policy_version,
                request_id,
                bucharest_day.isoformat(),
                policy_version,
                request_id,
                bucharest_day.replace(day=1).isoformat(),
                policy_version,
            ),
        )


def _expire_lease(connection: psycopg.Connection[Any], slot_id: SlotId) -> None:
    connection.execute(
        "UPDATE video_digest_slots SET lease_expires_at = clock_timestamp() "
        "- INTERVAL '1 second', updated_at = clock_timestamp() WHERE slot_id = %s",
        (slot_id,),
    )


def _select_slot_fence(connection: psycopg.Connection[Any], slot_id: SlotId) -> tuple[Any, ...]:
    row = connection.execute(
        "SELECT stage, edition_id, lease_owner_token, lease_expires_at, claim_count, "
        "skip_reason FROM video_digest_slots WHERE slot_id = %s",
        (slot_id,),
    ).fetchone()
    assert row is not None
    return row


def _publish_edition_and_slot(
    connection: psycopg.Connection[Any],
    slot: ScheduledSlot,
    recorded_at: datetime,
    *,
    artifact_start: int,
) -> EditionIdentity:
    (
        report,
        policy,
        plan,
        video,
        subtitle_failure,
        story_verification,
        accepted_clip,
        request_version,
        response_version,
        upload_evidence,
        publication_verification,
        candidate_validation,
        assembly_manifest,
    ) = _record_artifact_versions(connection, artifact_start, 13)
    identity = _edition(report, policy)
    _insert_edition(connection, identity.edition_id, report, policy)
    video_digest_catalog.schedule_slot(slot, recorded_at=recorded_at)
    lease = _claim_lease(
        slot.slot_id, identity, owner_token="publisher", lease_duration=timedelta(hours=1)
    )
    subject_id = _sha256_id(artifact_start + 40)
    story_id = planned_story_id(identity.edition_id, 0, subject_id)
    _insert_story(connection, story_id, identity.edition_id, 0, subject_id)
    connection.execute(
        "INSERT INTO video_digest_planning_attempts "
        "(edition_id, attempt_index, disposition, attempt_evidence_artifact_version_id, "
        "accepted_plan_artifact_version_id, created_at) "
        "VALUES (%s, 0, 'accepted', %s, %s, CURRENT_TIMESTAMP)",
        (identity.edition_id, upload_evidence, plan),
    )
    connection.execute(
        "UPDATE video_digest_editions SET plan_artifact_version_id = %s, "
        "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
        (plan, identity.edition_id),
    )
    connection.execute(
        "UPDATE video_digest_stories SET stage = 'verifying', updated_at = CURRENT_TIMESTAMP "
        "WHERE story_id = %s",
        (story_id,),
    )
    connection.execute(
        "UPDATE video_digest_stories SET stage = 'verified', "
        "verification_evidence_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
        "WHERE story_id = %s",
        (story_verification, story_id),
    )
    connection.execute(
        "UPDATE video_digest_stories SET stage = 'generating', updated_at = CURRENT_TIMESTAMP "
        "WHERE story_id = %s",
        (story_id,),
    )
    connection.execute(
        "UPDATE video_digest_editions SET verification_manifest_artifact_version_id = %s, "
        "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
        (publication_verification, identity.edition_id),
    )
    request_id = _sha256_id(artifact_start + 41)
    _insert_generation_request(connection, request_id, identity.edition_id, request_version)
    connection.execute(
        "UPDATE video_digest_generation_requests SET stage = 'submitted', "
        "provider_receipt_id = 'publisher-receipt', cost_kind = 'estimated', cost_usd = 1, "
        "updated_at = CURRENT_TIMESTAMP WHERE request_id = %s",
        (request_id,),
    )
    connection.execute(
        "UPDATE video_digest_generation_requests SET stage = 'processing', "
        "response_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
        "WHERE request_id = %s",
        (response_version, request_id),
    )
    connection.execute(
        "UPDATE video_digest_generation_requests SET stage = 'accepted', "
        "accepted_clip_artifact_version_id = %s, "
        "validation_evidence_artifact_version_id = %s, "
        "cost_kind = 'measured', cost_usd = 2, "
        "updated_at = CURRENT_TIMESTAMP WHERE request_id = %s",
        (accepted_clip, candidate_validation, request_id),
    )
    connection.execute(
        "UPDATE video_digest_stories SET stage = 'accepted', "
        "accepted_clip_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
        "WHERE story_id = %s",
        (accepted_clip, story_id),
    )
    connection.execute(
        "UPDATE video_digest_editions SET plan_artifact_version_id = %s, "
        "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
        (plan, identity.edition_id),
    )
    connection.execute(
        "UPDATE video_digest_editions SET assembled_video_artifact_version_id = %s, "
        "assembly_manifest_artifact_version_id = %s, "
        "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
        (video, assembly_manifest, identity.edition_id),
    )
    connection.execute(
        "UPDATE video_digest_editions SET subtitle_state = 'failed', "
        "subtitle_failure_evidence_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
        "WHERE edition_id = %s",
        (subtitle_failure, identity.edition_id),
    )
    for stage in ("planning", "generating", "assembling", "subtitling", "publishing"):
        connection.execute(
            "UPDATE video_digest_slots SET stage = %s, updated_at = CURRENT_TIMESTAMP "
            "WHERE slot_id = %s",
            (stage, slot.slot_id),
        )
    publication_id = _sha256_id(artifact_start + 42)
    connection.execute(
        "INSERT INTO video_digest_publication_intents "
        "(publication_id, edition_id, expected_video_key, video_digest, video_byte_size, "
        "video_media_type, source_video_artifact_version_id, stage, created_at, updated_at) "
        "VALUES (%s, %s, 'video-digest/edition.mp4', %s, 100, 'video/mp4', %s, 'pending', "
        "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        (publication_id, identity.edition_id, _sha256_id(artifact_start + 43), video),
    )
    connection.execute(
        "INSERT INTO video_digest_publication_attempts "
        "(publication_id, attempt_index, state, started_at, updated_at) "
        "VALUES (%s, 0, 'started', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        (publication_id,),
    )
    connection.execute(
        "UPDATE video_digest_publication_intents SET stage = 'uploading', "
        "updated_at = CURRENT_TIMESTAMP WHERE publication_id = %s",
        (publication_id,),
    )
    connection.execute(
        "UPDATE video_digest_publication_intents SET stage = 'uploaded', "
        "upload_evidence_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
        "WHERE publication_id = %s",
        (upload_evidence, publication_id),
    )
    connection.execute(
        "UPDATE video_digest_publication_intents SET stage = 'verified', "
        "verification_evidence_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
        "WHERE publication_id = %s",
        (publication_verification, publication_id),
    )
    connection.execute(
        "UPDATE video_digest_publication_attempts SET state = 'succeeded', "
        "updated_at = CURRENT_TIMESTAMP WHERE publication_id = %s AND attempt_index = 0",
        (publication_id,),
    )
    connection.execute(
        "UPDATE video_digest_publication_intents SET stage = 'published', "
        "published_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP "
        "WHERE publication_id = %s",
        (publication_id,),
    )
    connection.execute(
        "UPDATE video_digest_slots SET stage = 'published', lease_owner_token = NULL, "
        "lease_expires_at = NULL, terminal_lease_owner_token = %s, "
        "terminal_lease_expires_at = %s, terminal_claim_count = %s, "
        "updated_at = CURRENT_TIMESTAMP WHERE slot_id = %s",
        (lease.owner_token, lease.expires_at, lease.claim_count, slot.slot_id),
    )
    return identity


def test_schedule_slot_replays_an_exact_identity_idempotently(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    slot = _slot(SlotName.MORNING, MORNING_AT)

    first = video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    second = video_digest_catalog.schedule_slot(
        slot, recorded_at=RECORDED_AT + timedelta(minutes=5)
    )

    assert first == slot
    assert second == slot
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        stored = connection.execute(
            "SELECT name, scheduled_at, bucharest_day, stage, claim_count "
            "FROM video_digest_slots WHERE slot_id = %s",
            (slot.slot_id,),
        ).fetchone()
    assert stored == ("morning", MORNING_AT, slot.bucharest_day, "scheduled", 0)


def test_schedule_slot_normalizes_offset_datetimes_to_utc(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    slot = _slot(SlotName.MORNING, MORNING_AT)
    offset = timezone(timedelta(hours=3))
    offset_slot = slot.model_copy(update={"scheduled_at": MORNING_AT.astimezone(offset)})

    result = video_digest_catalog.schedule_slot(
        offset_slot, recorded_at=RECORDED_AT.astimezone(offset)
    )

    assert result == slot
    assert result.scheduled_at.tzinfo is UTC


def test_schedule_slot_rejects_a_conflicting_stored_identity(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    slot = _slot(SlotName.MORNING, MORNING_AT)
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        _insert_slot_row(connection, slot.slot_id, "midday", MORNING_AT + timedelta(hours=3))

    with pytest.raises(VideoDigestCheckpointConflictError, match="identity conflicts"):
        video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)


def test_schedule_slot_rejects_a_natural_identity_collision(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    slot = _slot(SlotName.MORNING, MORNING_AT)
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        _insert_slot_row(connection, SlotId(_sha256_id(40)), "morning", MORNING_AT)

    with pytest.raises(VideoDigestCheckpointConflictError, match="unavailable"):
        video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)


def test_schedule_slot_concurrent_replays_store_a_single_row(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    slot = _slot(SlotName.MORNING, MORNING_AT)
    barrier = Barrier(2)

    def schedule(recorded_at: datetime) -> ScheduledSlot | BaseException:
        try:
            barrier.wait()
            return video_digest_catalog.schedule_slot(slot, recorded_at=recorded_at)
        except BaseException as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(schedule, (RECORDED_AT, RECORDED_AT + timedelta(minutes=5))))

    assert results == [slot, slot]
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        stored = connection.execute(
            "SELECT count(*) FROM video_digest_slots WHERE slot_id = %s", (slot.slot_id,)
        ).fetchone()
    assert stored == (1,)


def test_skip_slot_records_the_reason_and_replays_it_exactly(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    reason = SlotSkipReason.SOURCE_EMPTY

    first = video_digest_catalog.skip_slot(slot.slot_id, reason, recorded_at=RECORDED_AT)
    second = video_digest_catalog.skip_slot(
        slot.slot_id, reason, recorded_at=RECORDED_AT + timedelta(minutes=1)
    )

    assert first == SkippedSlot(reason=reason)
    assert second == SkippedSlot(reason=reason)
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        row = _select_slot_fence(connection, slot.slot_id)
    assert row == ("skipped", None, None, None, 0, reason.value)


def test_skip_slot_rejects_a_conflicting_reason_on_a_skipped_slot(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    video_digest_catalog.skip_slot(
        slot.slot_id, SlotSkipReason.SOURCE_EMPTY, recorded_at=RECORDED_AT
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="skip request"):
        video_digest_catalog.skip_slot(
            slot.slot_id, SlotSkipReason.UNCHANGED, recorded_at=RECORDED_AT
        )


def test_skip_slot_rejects_a_slot_with_an_active_lease(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 45, 2)
    identity = _edition(report, policy)
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    _claim_lease(slot.slot_id, identity, owner_token="owner-a")

    with pytest.raises(VideoDigestCheckpointConflictError, match="skip request"):
        video_digest_catalog.skip_slot(
            slot.slot_id, SlotSkipReason.SOURCE_EMPTY, recorded_at=RECORDED_AT
        )


def test_claim_slot_returns_the_persisted_skip_reason(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 50, 2)
    identity = _edition(report, policy)
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    video_digest_catalog.skip_slot(
        slot.slot_id, SlotSkipReason.SOURCE_MISSING, recorded_at=RECORDED_AT
    )

    result = _claim(slot.slot_id, identity, owner_token="worker-1")

    assert result == SkippedSlot(reason=SlotSkipReason.SOURCE_MISSING)


def test_claim_slot_reports_a_failed_slot_as_terminal(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy, failure_version = _record_artifact_versions(connection, 55, 3)
    identity = _edition(report, policy)
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    lease = _claim_lease(slot.slot_id, identity, owner_token="owner-a")
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "UPDATE video_digest_slots SET stage = 'failed', lease_owner_token = NULL, "
            "lease_expires_at = NULL, terminal_lease_owner_token = %s, "
            "terminal_lease_expires_at = %s, terminal_claim_count = %s, "
            "failure_evidence_artifact_version_id = %s, "
            "failure_reason = 'terminal_failure', updated_at = CURRENT_TIMESTAMP "
            "WHERE slot_id = %s",
            (
                lease.owner_token,
                lease.expires_at,
                lease.claim_count,
                failure_version,
                slot.slot_id,
            ),
        )

    result = _claim(slot.slot_id, identity, owner_token="owner-b")

    assert result == TerminalSlot(state=TerminalSlotState.FAILED)


def test_claim_slot_reports_a_published_slot_as_terminal(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    slot = _slot(SlotName.MIDDAY, MIDDAY_AT)
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        identity = _publish_edition_and_slot(connection, slot, RECORDED_AT, artifact_start=60)

    result = _claim(slot.slot_id, identity, owner_token="worker-1")

    assert result == TerminalSlot(state=TerminalSlotState.PUBLISHED)


def test_claim_slot_claims_a_scheduled_slot_from_scratch(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 100, 2)
    identity = _edition(report, policy)
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    duration = timedelta(minutes=30)

    before = datetime.now(UTC)
    result = _claim(slot.slot_id, identity, owner_token="worker-1", lease_duration=duration)
    after = datetime.now(UTC)

    assert isinstance(result, ClaimedSlot)
    assert result.lease.slot_id == slot.slot_id
    assert result.lease.edition_id == identity.edition_id
    assert result.lease.owner_token == "worker-1"
    assert result.lease.claim_count == 1
    assert before + duration <= result.lease.expires_at <= after + duration
    assert result.lease.expires_at.tzinfo is UTC
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        row = _select_slot_fence(connection, slot.slot_id)
    assert row == ("claimed", identity.edition_id, "worker-1", result.lease.expires_at, 1, None)


def test_reacquire_slot_replays_an_unexpired_same_owner_lease(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 150, 2)
    identity = _edition(report, policy)
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    lease = _claim_lease(slot.slot_id, identity, owner_token="worker-1")

    result = _reacquire(slot.slot_id, owner_token=" worker-1 ")

    assert isinstance(result, ClaimedSlot)
    assert result.lease == lease
    assert result.lease.expires_at.tzinfo is UTC


def test_reacquire_slot_is_busy_while_another_owner_holds_the_lease(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 200, 2)
    identity = _edition(report, policy)
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    lease = _claim_lease(slot.slot_id, identity, owner_token="owner-a")

    result = _reacquire(slot.slot_id, owner_token="owner-b")

    assert result == BusySlot(retry_at=lease.expires_at)
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        row = _select_slot_fence(connection, slot.slot_id)
    assert row == ("claimed", identity.edition_id, "owner-a", lease.expires_at, 1, None)


def test_claim_slot_rejects_a_claim_for_another_edition_while_the_lease_is_active(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 250, 2)
    identity = _edition(report, policy)
    other = _edition(_sha256_id(260), _sha256_id(261))
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    _claim_lease(slot.slot_id, identity, owner_token="owner-a")

    with pytest.raises(VideoDigestCheckpointConflictError, match="reacquired"):
        _claim(slot.slot_id, other, owner_token="owner-b")


def test_claim_slot_skips_with_active_edition_while_another_slot_runs_it(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 300, 2)
    identity = _edition(report, policy)
    first = _slot(SlotName.MORNING, MORNING_AT)
    second = _slot(SlotName.MIDDAY, MIDDAY_AT)
    for slot in (first, second):
        video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    claimed = _claim(first.slot_id, identity, owner_token="worker-1")
    assert isinstance(claimed, ClaimedSlot)

    result = _claim(second.slot_id, identity, owner_token="worker-2")

    assert result == SkippedSlot(reason=SlotSkipReason.ACTIVE_EDITION)
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        row = _select_slot_fence(connection, second.slot_id)
    assert row == (
        "skipped",
        identity.edition_id,
        None,
        None,
        0,
        SlotSkipReason.ACTIVE_EDITION.value,
    )
    assert _claim(second.slot_id, identity, owner_token="worker-2") == SkippedSlot(
        reason=SlotSkipReason.ACTIVE_EDITION
    )


def test_claim_slot_skips_with_already_published_for_a_published_edition(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    published_slot = _slot(SlotName.MIDDAY, MIDDAY_AT)
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        identity = _publish_edition_and_slot(
            connection, published_slot, RECORDED_AT, artifact_start=350
        )
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)

    result = _claim(slot.slot_id, identity, owner_token="worker-1")

    assert result == SkippedSlot(reason=SlotSkipReason.ALREADY_PUBLISHED)
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        row = _select_slot_fence(connection, slot.slot_id)
    assert row == (
        "skipped",
        identity.edition_id,
        None,
        None,
        0,
        SlotSkipReason.ALREADY_PUBLISHED.value,
    )
    assert _claim(slot.slot_id, identity, owner_token="worker-1") == SkippedSlot(
        reason=SlotSkipReason.ALREADY_PUBLISHED
    )


def test_claim_slot_rejects_a_conflicting_stored_edition_identity(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy, stored_report, stored_policy = _record_artifact_versions(connection, 400, 4)
    identity = _edition(report, policy)
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        _insert_edition(connection, identity.edition_id, stored_report, stored_policy)
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)

    with pytest.raises(VideoDigestCheckpointConflictError, match="edition identity"):
        _claim(slot.slot_id, identity, owner_token="worker-1")


def test_reacquire_slot_recovers_same_owner_expired_lease(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 450, 2)
    identity = _edition(report, policy)
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    prior = _claim_lease(slot.slot_id, identity, owner_token="owner-a")
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        _expire_lease(connection, slot.slot_id)

    result = _reacquire(slot.slot_id, owner_token="owner-a")

    assert isinstance(result, ClaimedSlot)
    assert result.lease.claim_count == prior.claim_count + 1


def test_reacquire_slot_recovers_an_expired_lease_with_an_incremented_fence(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 500, 2)
    identity = _edition(report, policy)
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    prior = _claim_lease(
        slot.slot_id, identity, owner_token="owner-a", lease_duration=timedelta(minutes=10)
    )
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        _expire_lease(connection, slot.slot_id)
    duration = timedelta(hours=1)

    before = datetime.now(UTC)
    result = _reacquire(slot.slot_id, owner_token="owner-b", lease_duration=duration)
    after = datetime.now(UTC)

    assert isinstance(result, ClaimedSlot)
    assert result.lease.owner_token == "owner-b"
    assert result.lease.claim_count == prior.claim_count + 1
    assert before + duration <= result.lease.expires_at <= after + duration
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        row = _select_slot_fence(connection, slot.slot_id)
    assert row == ("claimed", identity.edition_id, "owner-b", result.lease.expires_at, 2, None)


def test_claim_slot_reports_a_missing_slot_as_unavailable(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 550, 2)
    identity = _edition(report, policy)

    with pytest.raises(VideoDigestCheckpointConflictError, match="unavailable"):
        _claim(SlotId(_sha256_id(1)), identity, owner_token="worker-1")


def test_renew_slot_extends_an_exact_unexpired_fence(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 600, 2)
    identity = _edition(report, policy)
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    lease = _claim_lease(
        slot.slot_id, identity, owner_token="worker-1", lease_duration=timedelta(minutes=10)
    )

    renewed = video_digest_catalog.renew_slot(
        lease, now=datetime.now(UTC), lease_duration=timedelta(hours=1)
    )

    assert renewed.slot_id == lease.slot_id
    assert renewed.edition_id == lease.edition_id
    assert renewed.owner_token == lease.owner_token
    assert renewed.claim_count == lease.claim_count
    assert renewed.expires_at > lease.expires_at
    assert renewed.expires_at.tzinfo is UTC
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        row = _select_slot_fence(connection, slot.slot_id)
    assert row == ("claimed", identity.edition_id, "worker-1", renewed.expires_at, 1, None)


def test_renew_slot_rejects_a_stale_fence(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 650, 2)
    identity = _edition(report, policy)
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    lease = _claim_lease(slot.slot_id, identity, owner_token="worker-1")
    stale = lease.model_copy(update={"claim_count": lease.claim_count + 1})

    with pytest.raises(VideoDigestLeaseLostError, match="lease was lost"):
        video_digest_catalog.renew_slot(
            stale, now=datetime.now(UTC), lease_duration=timedelta(hours=1)
        )


def test_renew_slot_rejects_an_expired_fence(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 700, 2)
    identity = _edition(report, policy)
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    lease = _claim_lease(slot.slot_id, identity, owner_token="worker-1")
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        _expire_lease(connection, slot.slot_id)

    with pytest.raises(VideoDigestLeaseLostError, match="lease was lost"):
        video_digest_catalog.renew_slot(
            lease, now=datetime.now(UTC), lease_duration=timedelta(hours=1)
        )


def test_claim_slot_maps_a_unique_violation_to_a_checkpoint_conflict(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 750, 2)
    identity = _edition(report, policy)
    slot = _slot(SlotName.MORNING, MORNING_AT)
    video_digest_catalog.schedule_slot(slot, recorded_at=RECORDED_AT)
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "CREATE FUNCTION reject_slot_claim_uniquely() RETURNS trigger LANGUAGE plpgsql "
            "AS $$ BEGIN IF OLD.stage = 'scheduled' AND NEW.stage = 'claimed' THEN "
            "RAISE EXCEPTION 'slot claim uniqueness guard' "
            "USING ERRCODE = '23505', CONSTRAINT = 'slot_claim_uniqueness_guard'; "
            "END IF; RETURN NEW; END; $$"
        )
        connection.execute(
            "CREATE TRIGGER contract_reject_slot_claim BEFORE UPDATE ON video_digest_slots "
            "FOR EACH ROW EXECUTE FUNCTION reject_slot_claim_uniquely()"
        )

    with pytest.raises(VideoDigestCheckpointConflictError, match="identity conflicts"):
        _claim(slot.slot_id, identity, owner_token="worker-1")

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        rolled_back = _select_slot_fence(connection, slot.slot_id)
        editions = connection.execute("SELECT count(*) FROM video_digest_editions").fetchone()
        connection.execute("DROP TRIGGER contract_reject_slot_claim ON video_digest_slots")
        connection.execute("DROP FUNCTION reject_slot_claim_uniquely()")

    assert rolled_back == ("scheduled", None, None, None, 0, None)
    assert editions == (0,)
    assert isinstance(_claim(slot.slot_id, identity, owner_token="worker-1"), ClaimedSlot)
