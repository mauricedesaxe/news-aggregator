from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from romanian_news.catalog import video_digest as video_digest_catalog
from romanian_news.catalog import video_digest_monitor
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.video_digest.models import (
    EditionIdentity,
    ScheduledSlot,
    SlotName,
    edition_id,
    scheduled_slot_id,
)
from tests.postgres_catalog import isolated_postgres_schema

NOW = datetime(2026, 9, 25, 10, tzinfo=UTC)

monkeypatch = pytest.MonkeyPatch()
with isolated_postgres_schema(monkeypatch, "probe_monitor") as (_, fixture_dsn):
    ensure_news_catalog_schema()
    report_v, policy_v, video_v, fail_v = "1" * 64, "2" * 64, "3" * 64, "4" * 64
    with psycopg.connect(fixture_dsn, autocommit=True) as c:
        for v in (report_v, policy_v, video_v, fail_v):
            c.execute(
                "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
                "VALUES (%s,'test','t','test','active','private',now())",
                (f"a-{v}",),
            )
            c.execute(
                "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, created_at) "
                "VALUES (%s,%s,1,%s,now())",
                (v, f"a-{v}", v),
            )
    identity = EditionIdentity(
        edition_id=edition_id(report_v, policy_v),
        daily_report_version_id=report_v,
        policy_bundle_version_id=policy_v,
    )
    dead = ScheduledSlot(
        slot_id=scheduled_slot_id(SlotName.MORNING, NOW - timedelta(days=1)),
        name=SlotName.MORNING,
        scheduled_at=NOW - timedelta(days=1),
        bucharest_day=(NOW - timedelta(days=1)).date(),
    )
    late = ScheduledSlot(
        slot_id=scheduled_slot_id(SlotName.MIDDAY, NOW - timedelta(hours=2)),
        name=SlotName.MIDDAY,
        scheduled_at=NOW - timedelta(hours=2),
        bucharest_day=(NOW - timedelta(hours=2)).date(),
    )
    video_digest_catalog.schedule_slot(dead, recorded_at=NOW - timedelta(days=1, minutes=5))
    dead_lease = video_digest_catalog.claim_slot(
        dead.slot_id,
        identity,
        owner_token="monitor",
        now=NOW - timedelta(days=1),
        lease_duration=timedelta(minutes=30),
    )
    print("claim dead:", type(dead_lease).__name__)
    video_digest_catalog.fail_slot_deadline(dead_lease.lease, recorded_at=NOW - timedelta(days=1))
    video_digest_catalog.schedule_slot(late, recorded_at=NOW - timedelta(hours=2, minutes=5))
    late_lease = video_digest_catalog.claim_slot(
        late.slot_id,
        identity,
        owner_token="monitor",
        now=NOW - timedelta(hours=2),
        lease_duration=timedelta(hours=6),
    )
    print("claim late:", type(late_lease).__name__)
    with psycopg.connect(fixture_dsn, autocommit=True) as c:
        c.execute(
            "INSERT INTO video_digest_slot_sources (slot_id, source_record, selection, selection_digest, observed_at, selected_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                late.slot_id,
                psycopg.types.json.Json({"version_id": report_v}),
                psycopg.types.json.Json({"selected_sections": [{"subject": "s"}]}),
                "6" * 64,
                NOW - timedelta(hours=30),
                NOW - timedelta(hours=25),
            ),
        )
        print("slot_sources: ok")
        c.execute(
            "INSERT INTO video_digest_publication_intents (publication_id, edition_id, expected_video_key, video_digest, video_byte_size, video_media_type, source_video_artifact_version_id, failure_evidence_artifact_version_id, stage, updated_at, created_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,NULL,'pending',now(),now())",
            ("c" * 64, identity.edition_id, "news/video", "7" * 64, 10, "video/mp4", video_v),
        )
        c.execute(
            "UPDATE video_digest_publication_intents SET stage = 'conflict', failure_evidence_artifact_version_id = %s WHERE publication_id = %s",
            (fail_v, "c" * 64),
        )
        print("intent: ok")
        try:
            c.execute(
                "INSERT INTO video_digest_generation_requests (request_id, edition_id, story_position, attempt_index, request_artifact_version_id, stage, generation_policy_artifact_version_id, created_at) "
                "VALUES (%s,%s,0,0,%s,'pending',%s,now())",
                ("r" * 64, identity.edition_id, policy_v, video_v),
            )
            print("gen request: ok")
            c.execute(
                "INSERT INTO video_digest_generation_reservations (request_id, scope_kind, scope_key, limit_usd, reserved_usd, generation_policy_artifact_version_id, created_at) "
                "VALUES (%s,'bucharest_day',%s,10,9,%s,now())",
                ("r" * 64, NOW.date().isoformat(), policy_v),
            )
            print("reservation: ok")
        except Exception as e:
            print("reservation path:", type(e).__name__, str(e)[:90])
    monkeypatch.setattr(video_digest_monitor, "read_stalled_fal_queue_incidents", lambda _now: ())
    for i in video_digest_monitor.read_due_video_incidents(NOW):
        print("incident:", i.category, i.scope_id)
