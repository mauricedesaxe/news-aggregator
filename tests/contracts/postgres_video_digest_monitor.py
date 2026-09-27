from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import psycopg
import pytest
from psycopg.types.json import Json

from romanian_news import BUCHAREST
from romanian_news.catalog import schema as news_schema
from romanian_news.catalog import video_digest as video_digest_catalog
from romanian_news.catalog import video_digest_monitor
from romanian_news.catalog.artifacts import artifact_file
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.video_digest.models import (
    ClaimedSlot,
    EditionIdentity,
    GenerationAdmission,
    GenerationBudgetLimits,
    GenerationRequestIdentity,
    ScheduledSlot,
    SlotName,
    edition_id,
    generation_request_id,
    scheduled_slot_id,
)
from tests.contracts.video_digest_planning_fixtures import accepted_planning_files


def test_video_incident_queries_match_the_postgres_schema(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert video_digest_monitor.read_due_video_incidents(datetime(2026, 9, 25, tzinfo=UTC)) == ()


def _artifact(connection: psycopg.Connection[Any], version: str) -> None:
    connection.execute(
        "INSERT INTO artifacts "
        "(id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
        "VALUES (%s, 'test', 'Test artifact', 'test', 'active', 'private', now())",
        (f"artifact-{version}",),
    )
    connection.execute(
        "INSERT INTO artifact_versions "
        "(id, artifact_id, schema_version, content_digest, created_at) "
        "VALUES (%s, %s, 1, %s, now())",
        (version, f"artifact-{version}", version),
    )


def test_due_incidents_cover_every_video_failure_category(
    postgres_news_schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    recorded = datetime.now(UTC) - timedelta(minutes=10)
    scheduled = (
        datetime.now(BUCHAREST).replace(hour=12, minute=0, second=0, microsecond=0)
        + timedelta(days=1)
    ).astimezone(UTC)
    monitor_now = scheduled + timedelta(minutes=61)

    report_v, policy_v, video_v, failure_v = ("1" * 64, "2" * 64, "3" * 64, "4" * 64)
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        for version in (report_v, policy_v, video_v, failure_v):
            _artifact(connection, version)
    identity = EditionIdentity(
        edition_id=edition_id(report_v, policy_v),
        daily_report_version_id=report_v,
        policy_bundle_version_id=policy_v,
    )

    deadline_slot = ScheduledSlot(
        slot_id=scheduled_slot_id(SlotName.MORNING, monitor_now - timedelta(days=1)),
        name=SlotName.MORNING,
        scheduled_at=monitor_now - timedelta(days=1),
        bucharest_day=(monitor_now - timedelta(days=1)).astimezone(BUCHAREST).date(),
    )
    video_digest_catalog.schedule_slot(deadline_slot, recorded_at=recorded)
    dead = video_digest_catalog.claim_slot(
        deadline_slot.slot_id,
        identity,
        owner_token="monitor",
        now=recorded,
        lease_duration=timedelta(minutes=30),
    )
    assert isinstance(dead, ClaimedSlot)
    video_digest_catalog.fail_slot_deadline(dead.lease, recorded_at=recorded)

    late_slot = ScheduledSlot(
        slot_id=scheduled_slot_id(SlotName.MIDDAY, scheduled),
        name=SlotName.MIDDAY,
        scheduled_at=scheduled,
        bucharest_day=scheduled.astimezone(BUCHAREST).date(),
    )
    video_digest_catalog.schedule_slot(late_slot, recorded_at=recorded)
    late = video_digest_catalog.claim_slot(
        late_slot.slot_id,
        identity,
        owner_token="monitor",
        now=recorded,
        lease_duration=timedelta(hours=6),
    )
    assert isinstance(late, ClaimedSlot)
    lease = late.lease

    plan, plan_file, planning_attempt_file = accepted_planning_files(
        identity,
        ((("5" * 64), "Monitor story", 15_000),),
        seed="monitor-contract",
    )
    story = plan.stories[0]
    video_digest_catalog.checkpoint_planning_attempt(
        lease,
        0,
        "accepted",
        evidence_file=planning_attempt_file,
        accepted_plan=plan,
        plan_file=plan_file,
        recorded_at=recorded,
    )
    verification_file = artifact_file(
        artifact_id=story.story_id,
        artifact_kind="video_digest_story_verification",
        title="Monitor verification",
        content=b"verified",
        r2_key="contracts/video-digest/monitor-verification.json",
        media_type="application/json",
    )
    video_digest_catalog.checkpoint_story_verification(
        lease,
        story.story_id,
        evidence_file=verification_file,
        recorded_at=recorded,
    )
    manifest_file = artifact_file(
        artifact_id=f"{identity.edition_id}:verification-manifest",
        artifact_kind="video_digest_verification_manifest",
        title="Monitor verification manifest",
        content=b"verified edition",
        r2_key="contracts/video-digest/monitor-verification-manifest.json",
        media_type="application/json",
    )
    video_digest_catalog.checkpoint_edition_verification(
        lease,
        manifest_file=manifest_file,
        recorded_at=recorded,
    )
    request_file = artifact_file(
        artifact_id=f"{identity.edition_id}:0:0:generation-request",
        artifact_kind="video_digest_generation_request",
        title="Monitor generation request",
        content=b"request",
        r2_key="contracts/video-digest/monitor-request.json",
        media_type="application/json",
    )
    request = GenerationRequestIdentity(
        request_id=generation_request_id(identity.edition_id, 0, 0, request_file.version_id),
        edition_id=identity.edition_id,
        story_position=0,
        attempt_index=0,
        request_artifact_version_id=request_file.version_id,
    )
    admission = GenerationAdmission(
        generation_policy_artifact_version_id=policy_v,
        reserved_usd=Decimal("3.25632"),
        limits=GenerationBudgetLimits(
            story_usd=Decimal("7"),
            edition_usd=Decimal("7"),
            bucharest_day_usd=Decimal("4"),
            calendar_month_usd=Decimal("1000"),
        ),
    )
    video_digest_catalog.checkpoint_generation_request(
        lease,
        request,
        request_file=request_file,
        admission=admission,
        recorded_at=recorded,
    )

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "INSERT INTO video_digest_slot_sources "
            "(slot_id, source_record, selection, selection_digest, observed_at, selected_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                late_slot.slot_id,
                Json({"version_id": report_v}),
                Json({"selected_sections": [{"subject": "s"}]}),
                "6" * 64,
                monitor_now - timedelta(hours=30),
                monitor_now - timedelta(hours=25),
            ),
        )
        connection.execute(
            "INSERT INTO video_digest_publication_intents "
            "(publication_id, edition_id, expected_video_key, video_digest, video_byte_size, "
            "video_media_type, source_video_artifact_version_id, stage, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', now(), now())",
            (
                "c" * 64,
                identity.edition_id,
                "news/video-digest/monitor/video.mp4",
                "7" * 64,
                10,
                "video/mp4",
                video_v,
            ),
        )
        connection.execute(
            "UPDATE video_digest_publication_intents "
            "SET stage = 'conflict', failure_evidence_artifact_version_id = %s "
            "WHERE publication_id = %s",
            (failure_v, "c" * 64),
        )

    monkeypatch.setattr(video_digest_monitor, "read_stalled_fal_queue_incidents", lambda _now: ())
    incidents = video_digest_monitor.read_due_video_incidents(monitor_now)

    assert [(incident.category, incident.scope_id) for incident in incidents] == [
        ("deadline", deadline_slot.slot_id),
        ("publication_late", late_slot.slot_id),
        ("publication_conflict", "c" * 64),
        ("no_success_24_hours", late_slot.slot_id),
        (
            "budget_80_percent",
            f"bucharest_day:{scheduled.astimezone(BUCHAREST).date().isoformat()}",
        ),
    ]
    assert incidents[0].alert_id != incidents[1].alert_id
