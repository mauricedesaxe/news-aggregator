from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from threading import Barrier
from typing import Any, LiteralString, cast

import psycopg
import pytest

import romanian_news.catalog.schema as news_schema
import romanian_news.catalog.video_digest as video_digest_catalog
import romanian_news.catalog_transport as catalog_transport
from romanian_news import BUCHAREST
from romanian_news.catalog.artifacts import artifact_file
from romanian_news.catalog.schema import NewsCatalogSchemaError, ensure_news_catalog_schema
from romanian_news.video_digest.errors import (
    VideoDigestCheckpointConflictError,
    VideoDigestLeaseLostError,
)
from romanian_news.video_digest.models import (
    ClaimedSlot,
    ClaimResult,
    EditionId,
    EditionIdentity,
    EstimatedAttemptCost,
    FailedSubtitles,
    GenerationAdmission,
    GenerationBudgetLimits,
    GenerationRequestIdentity,
    GenerationStage,
    MeasuredAttemptCost,
    PublicationIntent,
    ScheduledSlot,
    SkippedSlot,
    SlotName,
    SlotSkipReason,
    UploadedPublication,
    UploadingPublication,
    VerifiedPublication,
    VerifiedPublicObject,
    edition_id,
    generation_request_id,
    publication_id,
    scheduled_slot_id,
)
from tests.contracts.video_digest_planning_fixtures import accepted_planning_files


def _sha256_id(value: int) -> str:
    return f"{value:064x}"


def _record_artifact_versions(
    connection: psycopg.Connection[Any], start: int, count: int
) -> tuple[str, ...]:
    version_ids = tuple(_sha256_id(value) for value in range(start, start + count))
    for version_id in version_ids:
        artifact_id = f"artifact-{version_id}"
        connection.execute(
            "INSERT INTO artifacts "
            "(id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
            "VALUES (%s, 'test', 'Test artifact', 'test', 'active', 'private', CURRENT_TIMESTAMP)",
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
        "created_at, updated_at) VALUES (%s, %s, %s, 'pending', "
        "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        (edition_id, report, policy),
    )


def _insert_slot(connection: psycopg.Connection[Any], slot_id: str) -> None:
    connection.execute(
        "INSERT INTO video_digest_slots "
        "(slot_id, name, scheduled_at, bucharest_day, stage, claim_count, "
        "created_at, updated_at) VALUES (%s, 'morning', CURRENT_TIMESTAMP, "
        "(CURRENT_TIMESTAMP AT TIME ZONE 'Europe/Bucharest')::DATE, 'scheduled', 0, "
        "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        (slot_id,),
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
        "VALUES (%s, %s, %s, %s, 'Story', TRUE, 1000, 'planned', "
        "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        (story_id, edition_id, position, subject_id),
    )


def _insert_planning_attempt(
    connection: psycopg.Connection[Any],
    edition_id: str,
    evidence_version: str,
    plan_version: str,
) -> None:
    connection.execute(
        "INSERT INTO video_digest_planning_attempts "
        "(edition_id, attempt_index, disposition, attempt_evidence_artifact_version_id, "
        "accepted_plan_artifact_version_id, created_at) "
        "VALUES (%s, 0, 'accepted', %s, %s, CURRENT_TIMESTAMP)",
        (edition_id, evidence_version, plan_version),
    )


def _insert_generation_request(
    connection: psycopg.Connection[Any],
    request_id: str,
    edition_id: str,
    request_version: str,
    *,
    attempt_index: int = 0,
) -> None:
    has_admission = connection.execute(
        "SELECT to_regclass(current_schema() || '.video_digest_generation_reservations')"
    ).fetchone()
    assert has_admission is not None
    if has_admission[0] is None:
        connection.execute(
            "INSERT INTO video_digest_generation_requests "
            "(request_id, edition_id, story_position, attempt_index, "
            "request_artifact_version_id, stage, cost_kind, created_at, updated_at) "
            "VALUES (%s, %s, 0, %s, %s, 'pending', 'pending', "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (request_id, edition_id, attempt_index, request_version),
        )
        return
    row = connection.execute(
        "SELECT slot.scheduled_at, slot.bucharest_day, story.story_id, "
        "edition.policy_bundle_version_id FROM video_digest_slots AS slot "
        "JOIN video_digest_editions AS edition ON edition.edition_id = slot.edition_id "
        "JOIN video_digest_stories AS story ON story.edition_id = edition.edition_id "
        "AND story.position = 0 WHERE edition.edition_id = %s",
        (edition_id,),
    ).fetchone()
    if row is None:
        slot_id = edition_id
        _insert_slot(connection, slot_id)
        connection.execute(
            "UPDATE video_digest_slots SET stage = 'claimed', edition_id = %s, "
            "lease_owner_token = 'contract-owner', "
            "lease_expires_at = CURRENT_TIMESTAMP + INTERVAL '1 hour', claim_count = 1, "
            "updated_at = CURRENT_TIMESTAMP WHERE slot_id = %s",
            (edition_id, slot_id),
        )
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
            "VALUES (%s, %s, 0, %s, %s, %s, 3.25632, %s + INTERVAL '90 minutes', "
            "'pending', 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (request_id, edition_id, attempt_index, request_version, policy_version, scheduled_at),
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


def _insert_publication_intent(
    connection: psycopg.Connection[Any],
    publication_id: str,
    edition_id: str,
    video_version: str,
    subtitle_version: str,
) -> None:
    connection.execute(
        "INSERT INTO video_digest_publication_intents "
        "(publication_id, edition_id, expected_video_key, video_digest, video_byte_size, "
        "video_media_type, subtitle_expected_key, subtitle_digest, subtitle_byte_size, "
        "subtitle_media_type, source_video_artifact_version_id, "
        "source_subtitle_artifact_version_id, stage, created_at, updated_at) "
        "VALUES (%s, %s, 'digest.mp4', %s, 100, 'video/mp4', 'digest.vtt', %s, "
        "10, 'text/vtt', %s, %s, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        (
            publication_id,
            edition_id,
            _sha256_id(900),
            _sha256_id(901),
            video_version,
            subtitle_version,
        ),
    )


def _advance_story_to_generating(
    connection: psycopg.Connection[Any], story_id: str, verification_version: str
) -> None:
    connection.execute(
        "UPDATE video_digest_stories SET stage = 'verifying', updated_at = CURRENT_TIMESTAMP "
        "WHERE story_id = %s",
        (story_id,),
    )
    connection.execute(
        "UPDATE video_digest_stories SET stage = 'verified', "
        "verification_evidence_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
        "WHERE story_id = %s",
        (verification_version, story_id),
    )
    connection.execute(
        "UPDATE video_digest_stories SET stage = 'generating', updated_at = CURRENT_TIMESTAMP "
        "WHERE story_id = %s",
        (story_id,),
    )


def _accept_generation(
    connection: psycopg.Connection[Any],
    request_id: str,
    response_version: str,
    clip_version: str,
) -> None:
    connection.execute(
        "UPDATE video_digest_generation_requests SET stage = 'submitted', "
        "provider_receipt_id = 'provider-1', cost_kind = 'estimated', cost_usd = 1, "
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
        "accepted_clip_artifact_version_id = %s, cost_kind = 'measured', cost_usd = 2, "
        "updated_at = CURRENT_TIMESTAMP WHERE request_id = %s",
        (clip_version, request_id),
    )


def test_news_schema_installs_and_verifies_again(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = current_schema()"
            ).fetchall()
        }
        migrations = connection.execute(
            "SELECT version, name, sha256 FROM news_schema_migrations ORDER BY version"
        ).fetchall()

    assert "debt_transcript_projection_items" not in tables
    assert len(tables) == 48
    assert migrations == [
        (1, "initial", news_schema.NEWS_CATALOG_MIGRATIONS[0].sha256),
        (2, "video_digest", news_schema.NEWS_CATALOG_MIGRATIONS[1].sha256),
        (
            3,
            "video_digest_generation_fences",
            news_schema.NEWS_CATALOG_MIGRATIONS[2].sha256,
        ),
        (
            4,
            "video_digest_publication_evidence",
            news_schema.NEWS_CATALOG_MIGRATIONS[3].sha256,
        ),
        (5, "video_digest_planning", news_schema.NEWS_CATALOG_MIGRATIONS[4].sha256),
        (6, "video_digest_generation_admission", news_schema.NEWS_CATALOG_MIGRATIONS[5].sha256),
    ]


def test_planning_migration_grandfathers_existing_editions_without_authorizing_paid_work(
    postgres_news_schema: str,
) -> None:
    assert news_schema.NEWS_POSTGRES_DSN is not None
    migrations = news_schema.NEWS_CATALOG_MIGRATIONS
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        for migration in migrations[:4]:
            connection.execute(cast(LiteralString, migration.path.read_text()), prepare=False)
            connection.execute(
                "INSERT INTO news_schema_migrations (version, name, sha256) VALUES (%s, %s, %s)",
                (migration.version, migration.name, migration.sha256),
            )

        report, policy, plan, request = _record_artifact_versions(connection, 900, 4)
        legacy_edition, legacy_story, request_id = (_sha256_id(value) for value in range(910, 913))
        _insert_edition(connection, legacy_edition, report, policy)
        _insert_story(connection, legacy_story, legacy_edition, 0, _sha256_id(913))
        connection.execute(
            "UPDATE video_digest_editions SET plan_artifact_version_id = %s, "
            "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
            (plan, legacy_edition),
        )
        _insert_generation_request(connection, request_id, legacy_edition, request)

    ensure_news_catalog_schema()

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        assert connection.execute(
            "SELECT planning_contract FROM video_digest_editions WHERE edition_id = %s",
            (legacy_edition,),
        ).fetchone() == ("legacy_unverified",)
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_generation_requests SET stage = 'submitted', "
                "provider_receipt_id = 'legacy-receipt', cost_kind = 'estimated', "
                "cost_usd = 1, updated_at = CURRENT_TIMESTAMP WHERE request_id = %s",
                (request_id,),
            )
        new_edition = _sha256_id(914)
        new_report, new_policy = _record_artifact_versions(connection, 904, 2)
        _insert_edition(connection, new_edition, new_report, new_policy)
        assert connection.execute(
            "SELECT planning_contract FROM video_digest_editions WHERE edition_id = %s",
            (new_edition,),
        ).fetchone() == ("verified_v1",)

    with pytest.raises(VideoDigestCheckpointConflictError, match="Legacy video digest edition"):
        video_digest_catalog.read_planning_attempts(EditionId(legacy_edition))


def test_news_schema_rejects_changed_migration_digest(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        connection.execute("UPDATE news_schema_migrations SET sha256 = %s", ("0" * 64,))

    with pytest.raises(NewsCatalogSchemaError, match="migration 1 differs"):
        ensure_news_catalog_schema()


def test_news_schema_rejects_disabled_triggers(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        connection.execute("ALTER TABLE runs DISABLE TRIGGER runs_protect_identity")

    with pytest.raises(NewsCatalogSchemaError, match="disabled triggers: runs_protect_identity"):
        ensure_news_catalog_schema()


def test_video_digest_inserts_must_start_at_initial_state(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy, plan, video, verification, request_version = _record_artifact_versions(
            connection, 1, 6
        )
        edition_id, slot_id, story_id, subject_id, request_id, publication_id = (
            _sha256_id(value) for value in range(20, 26)
        )

        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "INSERT INTO video_digest_editions "
                "(edition_id, daily_report_version_id, policy_bundle_version_id, "
                "plan_artifact_version_id, subtitle_state, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                (edition_id, report, policy, plan),
            )
        _insert_edition(connection, edition_id, report, policy)

        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "INSERT INTO video_digest_slots "
                "(slot_id, name, scheduled_at, bucharest_day, stage, edition_id, "
                "lease_owner_token, lease_expires_at, claim_count, created_at, updated_at) "
                "VALUES (%s, 'morning', CURRENT_TIMESTAMP, "
                "(CURRENT_TIMESTAMP AT TIME ZONE 'Europe/Bucharest')::DATE, 'claimed', %s, "
                "'owner', CURRENT_TIMESTAMP + INTERVAL '1 hour', 1, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                (slot_id, edition_id),
            )
        _insert_slot(connection, slot_id)

        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "INSERT INTO video_digest_stories "
                "(story_id, edition_id, position, report_subject_id, title, mandatory, "
                "requested_duration_ms, stage, verification_evidence_artifact_version_id, "
                "created_at, updated_at) VALUES (%s, %s, 0, %s, 'Story', TRUE, 1000, "
                "'verified', %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                (story_id, edition_id, subject_id, verification),
            )
        _insert_story(connection, story_id, edition_id, 0, subject_id)

        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "INSERT INTO video_digest_generation_requests "
                "(request_id, edition_id, story_position, attempt_index, "
                "request_artifact_version_id, stage, provider_receipt_id, cost_kind, "
                "created_at, updated_at) VALUES (%s, %s, 0, 0, %s, 'submitted', "
                "'receipt', 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                (request_id, edition_id, request_version),
            )
        _insert_planning_attempt(connection, edition_id, video, plan)
        connection.execute(
            "UPDATE video_digest_editions SET plan_artifact_version_id = %s, "
            "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
            (plan, edition_id),
        )
        _advance_story_to_generating(connection, story_id, verification)
        connection.execute(
            "UPDATE video_digest_editions SET verification_manifest_artifact_version_id = %s, "
            "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
            (verification, edition_id),
        )
        _insert_generation_request(connection, request_id, edition_id, request_version)

        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "INSERT INTO video_digest_publication_intents "
                "(publication_id, edition_id, expected_video_key, video_digest, "
                "video_byte_size, video_media_type, source_video_artifact_version_id, "
                "stage, created_at, updated_at) VALUES (%s, %s, 'digest.mp4', %s, 100, "
                "'video/mp4', %s, 'uploading', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                (publication_id, edition_id, _sha256_id(99), video),
            )


def test_video_digest_plan_membership_is_contiguous_and_frozen(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy, plan, attempt_evidence = _record_artifact_versions(connection, 100, 4)
        edition_id, first_story, second_story, late_story = (
            _sha256_id(value) for value in range(110, 114)
        )
        _insert_edition(connection, edition_id, report, policy)
        _insert_planning_attempt(connection, edition_id, attempt_evidence, plan)
        _insert_story(connection, second_story, edition_id, 1, _sha256_id(120))

        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_editions SET plan_artifact_version_id = %s, "
                "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
                (plan, edition_id),
            )

        _insert_story(connection, first_story, edition_id, 0, _sha256_id(121))
        connection.execute(
            "UPDATE video_digest_editions SET plan_artifact_version_id = %s, "
            "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
            (plan, edition_id),
        )
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            _insert_story(connection, late_story, edition_id, 2, _sha256_id(122))


def test_video_digest_planning_and_generation_authorization_are_database_enforced(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        (
            report,
            policy,
            plan,
            rejected_evidence,
            accepted_evidence,
            verification,
            manifest,
            request,
        ) = _record_artifact_versions(connection, 130, 8)
        edition_id, story_id, request_id = (_sha256_id(value) for value in range(140, 143))
        _insert_edition(connection, edition_id, report, policy)
        _insert_story(connection, story_id, edition_id, 0, _sha256_id(150))

        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "INSERT INTO video_digest_planning_attempts "
                "(edition_id, attempt_index, disposition, "
                "attempt_evidence_artifact_version_id, created_at) "
                "VALUES (%s, 1, 'rejected', %s, CURRENT_TIMESTAMP)",
                (edition_id, rejected_evidence),
            )

        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_editions SET plan_artifact_version_id = %s, "
                "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
                (plan, edition_id),
            )
        connection.execute(
            "INSERT INTO video_digest_planning_attempts "
            "(edition_id, attempt_index, disposition, attempt_evidence_artifact_version_id, "
            "created_at) VALUES (%s, 0, 'rejected', %s, CURRENT_TIMESTAMP)",
            (edition_id, rejected_evidence),
        )
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_planning_attempts SET disposition = 'accepted' "
                "WHERE edition_id = %s AND attempt_index = 0",
                (edition_id,),
            )
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "DELETE FROM video_digest_planning_attempts "
                "WHERE edition_id = %s AND attempt_index = 0",
                (edition_id,),
            )
        connection.execute(
            "INSERT INTO video_digest_planning_attempts "
            "(edition_id, attempt_index, disposition, attempt_evidence_artifact_version_id, "
            "accepted_plan_artifact_version_id, created_at) "
            "VALUES (%s, 1, 'accepted', %s, %s, CURRENT_TIMESTAMP)",
            (edition_id, accepted_evidence, plan),
        )
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "INSERT INTO video_digest_planning_attempts "
                "(edition_id, attempt_index, disposition, "
                "attempt_evidence_artifact_version_id, created_at) "
                "VALUES (%s, 2, 'rejected', %s, CURRENT_TIMESTAMP)",
                (edition_id, verification),
            )
        connection.execute(
            "UPDATE video_digest_editions SET plan_artifact_version_id = %s, "
            "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
            (plan, edition_id),
        )
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "INSERT INTO video_digest_planning_attempts "
                "(edition_id, attempt_index, disposition, attempt_evidence_artifact_version_id, "
                "accepted_plan_artifact_version_id, created_at) "
                "VALUES (%s, 2, 'accepted', %s, %s, CURRENT_TIMESTAMP)",
                (edition_id, accepted_evidence, plan),
            )

        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            _insert_generation_request(connection, request_id, edition_id, request)
        _advance_story_to_generating(connection, story_id, verification)
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            _insert_generation_request(connection, request_id, edition_id, request)
        connection.execute(
            "UPDATE video_digest_editions "
            "SET verification_manifest_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
            "WHERE edition_id = %s",
            (manifest, edition_id),
        )
        _insert_generation_request(connection, request_id, edition_id, request)
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_editions "
                "SET verification_manifest_artifact_version_id = %s, "
                "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
                (verification, edition_id),
            )


def test_video_digest_story_acceptance_requires_matching_generation(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        (
            report,
            policy,
            plan,
            attempt_evidence,
            verification,
            request_version,
            response,
            clip,
        ) = _record_artifact_versions(connection, 200, 8)
        edition_id, story_id, request_id = (_sha256_id(value) for value in range(210, 213))
        _insert_edition(connection, edition_id, report, policy)
        _insert_story(connection, story_id, edition_id, 0, _sha256_id(220))
        _insert_planning_attempt(connection, edition_id, attempt_evidence, plan)
        connection.execute(
            "UPDATE video_digest_editions SET plan_artifact_version_id = %s, "
            "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
            (plan, edition_id),
        )
        _advance_story_to_generating(connection, story_id, verification)
        connection.execute(
            "UPDATE video_digest_editions "
            "SET verification_manifest_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
            "WHERE edition_id = %s",
            (verification, edition_id),
        )
        _insert_generation_request(connection, request_id, edition_id, request_version)

        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_stories SET stage = 'accepted', "
                "accepted_clip_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
                "WHERE story_id = %s",
                (clip, story_id),
            )

        _accept_generation(connection, request_id, response, clip)
        connection.execute(
            "UPDATE video_digest_stories SET stage = 'accepted', "
            "accepted_clip_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
            "WHERE story_id = %s",
            (clip, story_id),
        )
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_generation_requests SET cost_kind = 'estimated', "
                "cost_usd = 1, updated_at = CURRENT_TIMESTAMP WHERE request_id = %s",
                (request_id,),
            )


def test_video_digest_lease_owner_requires_a_fresh_claim(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 300, 2)
        edition_id, slot_id = (_sha256_id(value) for value in range(310, 312))
        _insert_edition(connection, edition_id, report, policy)
        _insert_slot(connection, slot_id)
        connection.execute(
            "UPDATE video_digest_slots SET stage = 'claimed', edition_id = %s, "
            "lease_owner_token = 'owner-a', lease_expires_at = CURRENT_TIMESTAMP + INTERVAL '1 hour', "
            "claim_count = 1, updated_at = CURRENT_TIMESTAMP WHERE slot_id = %s",
            (edition_id, slot_id),
        )

        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_slots SET lease_owner_token = 'owner-b', claim_count = 2, "
                "updated_at = CURRENT_TIMESTAMP WHERE slot_id = %s",
                (slot_id,),
            )

        connection.execute(
            "UPDATE video_digest_slots SET lease_expires_at = CURRENT_TIMESTAMP - INTERVAL '1 second', "
            "updated_at = CURRENT_TIMESTAMP WHERE slot_id = %s",
            (slot_id,),
        )
        connection.execute(
            "UPDATE video_digest_slots SET lease_owner_token = 'owner-b', "
            "lease_expires_at = CURRENT_TIMESTAMP + INTERVAL '1 hour', claim_count = 2, "
            "updated_at = CURRENT_TIMESTAMP WHERE slot_id = %s",
            (slot_id,),
        )


def test_video_digest_catalog_serializes_claim_race_and_fences_recovery(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert catalog_transport.NEWS_POSTGRES_DSN is not None
    fixture_dsn = catalog_transport.NEWS_POSTGRES_DSN
    recorded_at = datetime.now(UTC)

    with psycopg.connect(fixture_dsn, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 350, 2)

    identity = EditionIdentity(
        edition_id=edition_id(report, policy),
        daily_report_version_id=report,
        policy_bundle_version_id=policy,
    )
    slots = tuple(
        ScheduledSlot(
            slot_id=scheduled_slot_id(name, scheduled_at),
            name=name,
            scheduled_at=scheduled_at,
            bucharest_day=scheduled_at.astimezone(BUCHAREST).date(),
        )
        for name, scheduled_at in (
            (SlotName.MORNING, datetime(2026, 9, 20, 6, tzinfo=UTC)),
            (SlotName.MIDDAY, datetime(2026, 9, 20, 9, tzinfo=UTC)),
        )
    )
    for slot in slots:
        video_digest_catalog.schedule_slot(slot, recorded_at=recorded_at)
    barrier = Barrier(2)

    def race(arguments: tuple[ScheduledSlot, str]) -> ClaimResult | BaseException:
        slot, owner_token = arguments
        try:
            barrier.wait()
            return video_digest_catalog.claim_slot(
                slot.slot_id,
                identity,
                owner_token=owner_token,
                now=recorded_at,
                lease_duration=timedelta(hours=1),
            )
        except BaseException as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(race, zip(slots, ("owner-a", "owner-b"), strict=True)))

    errors = [result for result in results if isinstance(result, BaseException)]
    assert errors == []
    claimed = [result for result in results if isinstance(result, ClaimedSlot)]
    skipped = [result for result in results if isinstance(result, SkippedSlot)]
    assert len(claimed) == 1
    assert skipped == [SkippedSlot(reason=SlotSkipReason.ACTIVE_EDITION)]
    prior_lease = claimed[0].lease

    recovery_time = datetime.now(UTC)
    with psycopg.connect(fixture_dsn) as connection:
        connection.execute(
            "UPDATE video_digest_slots SET lease_expires_at = %s, updated_at = %s "
            "WHERE slot_id = %s",
            (recovery_time - timedelta(seconds=1), recovery_time, prior_lease.slot_id),
        )

    recovered = video_digest_catalog.claim_slot(
        prior_lease.slot_id,
        identity,
        owner_token="owner-c",
        now=recovery_time,
        lease_duration=timedelta(hours=1),
    )
    assert isinstance(recovered, ClaimedSlot)
    assert recovered.lease.claim_count == prior_lease.claim_count + 1
    assert recovered.lease.owner_token == "owner-c"

    with pytest.raises(VideoDigestLeaseLostError):
        video_digest_catalog.renew_slot(
            prior_lease,
            now=recovery_time,
            lease_duration=timedelta(hours=1),
        )


def test_video_digest_publication_and_slot_complete_together(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        (
            report,
            policy,
            plan,
            video,
            subtitles,
            verification,
            request_version,
            response,
            clip,
            upload_evidence,
            publication_verification,
        ) = _record_artifact_versions(connection, 400, 11)
        edition_id, slot_id, story_id, request_id, publication_id = (
            _sha256_id(value) for value in range(420, 425)
        )
        _insert_edition(connection, edition_id, report, policy)
        _insert_slot(connection, slot_id)
        _insert_story(connection, story_id, edition_id, 0, _sha256_id(430))
        _insert_publication_intent(connection, publication_id, edition_id, video, subtitles)
        _insert_planning_attempt(connection, edition_id, verification, plan)
        connection.execute(
            "UPDATE video_digest_editions SET plan_artifact_version_id = %s, "
            "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
            (plan, edition_id),
        )
        connection.execute(
            "UPDATE video_digest_slots SET stage = 'claimed', edition_id = %s, "
            "lease_owner_token = 'owner', lease_expires_at = CURRENT_TIMESTAMP + INTERVAL '1 hour', "
            "claim_count = 1, updated_at = CURRENT_TIMESTAMP WHERE slot_id = %s",
            (edition_id, slot_id),
        )
        for stage in ("planning", "generating", "assembling", "subtitling"):
            connection.execute(
                "UPDATE video_digest_slots SET stage = %s, updated_at = CURRENT_TIMESTAMP "
                "WHERE slot_id = %s",
                (stage, slot_id),
            )
        _advance_story_to_generating(connection, story_id, verification)
        connection.execute(
            "UPDATE video_digest_editions "
            "SET verification_manifest_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
            "WHERE edition_id = %s",
            (verification, edition_id),
        )
        _insert_generation_request(connection, request_id, edition_id, request_version)
        _accept_generation(connection, request_id, response, clip)
        connection.execute(
            "UPDATE video_digest_stories SET stage = 'accepted', "
            "accepted_clip_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
            "WHERE story_id = %s",
            (clip, story_id),
        )
        connection.execute(
            "UPDATE video_digest_editions SET assembled_video_artifact_version_id = %s, "
            "updated_at = CURRENT_TIMESTAMP WHERE edition_id = %s",
            (video, edition_id),
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
            "verification_evidence_artifact_version_id = %s, "
            "updated_at = CURRENT_TIMESTAMP WHERE publication_id = %s",
            (publication_verification, publication_id),
        )

        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_publication_intents SET stage = 'published', "
                "published_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP "
                "WHERE publication_id = %s",
                (publication_id,),
            )
        connection.execute(
            "UPDATE video_digest_editions SET subtitle_state = 'available', "
            "subtitle_artifact_version_id = %s, updated_at = CURRENT_TIMESTAMP "
            "WHERE edition_id = %s",
            (subtitles, edition_id),
        )
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_publication_intents SET stage = 'published', "
                "published_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP "
                "WHERE publication_id = %s",
                (publication_id,),
            )

        connection.execute(
            "UPDATE video_digest_slots SET stage = 'publishing', updated_at = CURRENT_TIMESTAMP "
            "WHERE slot_id = %s",
            (slot_id,),
        )
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_slots SET stage = 'published', lease_owner_token = NULL, "
                "lease_expires_at = NULL, terminal_lease_owner_token = lease_owner_token, "
                "terminal_lease_expires_at = lease_expires_at, terminal_claim_count = claim_count, "
                "updated_at = CURRENT_TIMESTAMP WHERE slot_id = %s",
                (slot_id,),
            )

        with connection.transaction():
            connection.execute(
                "UPDATE video_digest_publication_intents SET stage = 'published', "
                "published_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP "
                "WHERE publication_id = %s",
                (publication_id,),
            )
            connection.execute(
                "UPDATE video_digest_slots SET stage = 'published', lease_owner_token = NULL, "
                "lease_expires_at = NULL, terminal_lease_owner_token = lease_owner_token, "
                "terminal_lease_expires_at = lease_expires_at, terminal_claim_count = claim_count, "
                "updated_at = CURRENT_TIMESTAMP WHERE slot_id = %s",
                (slot_id,),
            )


def test_video_digest_generation_checkpoints_complete_atomically(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    recorded_at = datetime.now(UTC)
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, 500, 2)

    identity = EditionIdentity(
        edition_id=edition_id(report, policy),
        daily_report_version_id=report,
        policy_bundle_version_id=policy,
    )
    slot = ScheduledSlot(
        slot_id=scheduled_slot_id(SlotName.MORNING, recorded_at),
        name=SlotName.MORNING,
        scheduled_at=recorded_at,
        bucharest_day=recorded_at.astimezone(BUCHAREST).date(),
    )
    video_digest_catalog.schedule_slot(slot, recorded_at=recorded_at)
    claimed = video_digest_catalog.claim_slot(
        slot.slot_id,
        identity,
        owner_token="generation-contract",
        now=recorded_at,
        lease_duration=timedelta(hours=1),
    )
    assert isinstance(claimed, ClaimedSlot)
    lease = claimed.lease

    plan, plan_file, planning_attempt_file = accepted_planning_files(
        identity,
        ((_sha256_id(502), "Contract story", 15_000),),
        seed="schema-contract",
    )
    story = plan.stories[0]
    video_digest_catalog.checkpoint_planning_attempt(
        lease,
        0,
        "accepted",
        evidence_file=planning_attempt_file,
        accepted_plan=plan,
        plan_file=plan_file,
        recorded_at=recorded_at,
    )
    verification_file = artifact_file(
        artifact_id=story.story_id,
        artifact_kind="video_digest_story_verification",
        title="Contract verification",
        content=b"verified",
        r2_key="contracts/video-digest/verification.json",
        media_type="application/json",
    )
    video_digest_catalog.checkpoint_story_verification(
        lease,
        story.story_id,
        evidence_file=verification_file,
        recorded_at=recorded_at,
    )
    manifest_file = artifact_file(
        artifact_id=f"{identity.edition_id}:verification-manifest",
        artifact_kind="video_digest_verification_manifest",
        title="Contract verification manifest",
        content=b"verified edition",
        r2_key="contracts/video-digest/verification-manifest.json",
        media_type="application/json",
    )
    video_digest_catalog.checkpoint_edition_verification(
        lease,
        manifest_file=manifest_file,
        recorded_at=recorded_at,
    )

    request_file = artifact_file(
        artifact_id=f"{identity.edition_id}:0:0:generation-request",
        artifact_kind="video_digest_generation_request",
        title="Contract generation request",
        content=b"request",
        r2_key="contracts/video-digest/request.json",
        media_type="application/json",
    )
    request = GenerationRequestIdentity(
        request_id=generation_request_id(identity.edition_id, 0, 0, request_file.version_id),
        edition_id=identity.edition_id,
        story_position=0,
        attempt_index=0,
        request_artifact_version_id=request_file.version_id,
    )
    pending = video_digest_catalog.checkpoint_generation_request(
        lease,
        request,
        request_file=request_file,
        admission=GenerationAdmission(
            generation_policy_artifact_version_id=policy,
            reserved_usd=Decimal("3.25632"),
            limits=GenerationBudgetLimits(
                story_usd=Decimal("7"),
                edition_usd=Decimal("7"),
                bucharest_day_usd=Decimal("150"),
                calendar_month_usd=Decimal("1000"),
            ),
        ),
        recorded_at=recorded_at,
    )
    assert pending.created is True
    assert pending.state.stage is GenerationStage.PENDING
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        reservations = connection.execute(
            "SELECT scope_kind, reserved_usd FROM video_digest_generation_reservations "
            "WHERE request_id = %s ORDER BY scope_kind",
            (request.request_id,),
        ).fetchall()
        assert reservations == [
            ("bucharest_day", Decimal("3.25632")),
            ("calendar_month", Decimal("3.25632")),
            ("edition", Decimal("3.25632")),
            ("story", Decimal("3.25632")),
        ]
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "DELETE FROM video_digest_generation_reservations WHERE request_id = %s",
                (request.request_id,),
            )
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_generation_reservations SET reserved_usd = 1 "
                "WHERE request_id = %s",
                (request.request_id,),
            )

    receipt_id = f"fal-{request.request_id}"
    receipt_file = artifact_file(
        artifact_id=receipt_id,
        artifact_kind="video_digest_provider_receipt",
        title="Contract provider receipt",
        content=b"receipt",
        r2_key="contracts/video-digest/receipt.json",
        media_type="application/json",
    )
    submitted = video_digest_catalog.checkpoint_generation_submission(
        lease,
        request.request_id,
        provider_receipt_id=receipt_id,
        receipt_file=receipt_file,
        cost=EstimatedAttemptCost(usd=Decimal("1.25")),
        recorded_at=recorded_at,
    )
    assert submitted.provider_receipt_id == receipt_id

    response_file = artifact_file(
        artifact_id=f"{request.request_id}:response",
        artifact_kind="video_digest_generation_response",
        title="Contract generation response",
        content=b"response",
        r2_key="contracts/video-digest/response.json",
        media_type="application/json",
    )
    processing = video_digest_catalog.checkpoint_generation_response(
        lease,
        request.request_id,
        response_file=response_file,
        recorded_at=recorded_at,
    )
    assert processing.stage is GenerationStage.PROCESSING

    clip_file = artifact_file(
        artifact_id=f"{story.story_id}:accepted-clip",
        artifact_kind="video_digest_accepted_clip",
        title="Contract accepted clip",
        content=b"clip",
        r2_key="contracts/video-digest/clip.mp4",
        media_type="video/mp4",
    )
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "CREATE FUNCTION reject_contract_story_acceptance() RETURNS trigger "
            "LANGUAGE plpgsql AS $$ BEGIN "
            "IF NEW.stage = 'accepted' THEN "
            "RAISE EXCEPTION 'contract rejection' USING ERRCODE = '23000'; "
            "END IF; RETURN NEW; END; $$"
        )
        connection.execute(
            "CREATE TRIGGER contract_reject_story_acceptance "
            "BEFORE UPDATE ON video_digest_stories "
            "FOR EACH ROW EXECUTE FUNCTION reject_contract_story_acceptance()"
        )
    with pytest.raises(VideoDigestCheckpointConflictError):
        video_digest_catalog.checkpoint_generation_acceptance(
            lease,
            request.request_id,
            clip_file=clip_file,
            cost=MeasuredAttemptCost(usd=Decimal("1.10")),
            recorded_at=recorded_at,
        )
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        rolled_back = connection.execute(
            "SELECT request.stage, artifact.id "
            "FROM video_digest_generation_requests AS request "
            "LEFT JOIN artifacts AS artifact ON artifact.id = %s "
            "WHERE request.request_id = %s",
            (clip_file.artifact_id, request.request_id),
        ).fetchone()
        assert rolled_back == ("processing", None)
        connection.execute("DROP TRIGGER contract_reject_story_acceptance ON video_digest_stories")
        connection.execute("DROP FUNCTION reject_contract_story_acceptance()")

    accepted = video_digest_catalog.checkpoint_generation_acceptance(
        lease,
        request.request_id,
        clip_file=clip_file,
        cost=MeasuredAttemptCost(usd=Decimal("1.10")),
        recorded_at=recorded_at,
    )
    assert accepted.stage is GenerationStage.ACCEPTED
    video_digest_catalog.checkpoint_assembly_ready(lease, recorded_at=recorded_at)

    assembled_file = artifact_file(
        artifact_id=f"{identity.edition_id}:assembled-video",
        artifact_kind="video_digest_assembled_video",
        title="Contract assembled video",
        content=b"assembled video",
        r2_key="contracts/video-digest/assembled.mp4",
        media_type="video/mp4",
    )
    video_digest_catalog.checkpoint_assembled_video(
        lease,
        video_file=assembled_file,
        recorded_at=recorded_at,
    )
    subtitle_failure_file = artifact_file(
        artifact_id=f"{identity.edition_id}:subtitle-failure",
        artifact_kind="video_digest_subtitle_failure",
        title="Contract subtitle failure",
        content=b"subtitle failure",
        r2_key="contracts/video-digest/subtitle-failure.json",
        media_type="application/json",
    )
    video_digest_catalog.checkpoint_subtitles(
        lease,
        FailedSubtitles(evidence_artifact_version_id=subtitle_failure_file.version_id),
        artifact_file=subtitle_failure_file,
        recorded_at=recorded_at,
    )
    publication_id_value = publication_id(
        edition_id_value=identity.edition_id,
        expected_video_key="contracts/video-digest/public.mp4",
        video_digest=assembled_file.content_digest,
        video_byte_size=len(assembled_file.content),
        video_media_type=assembled_file.media_type,
        subtitle=None,
        source_video_version_id=assembled_file.version_id,
        source_subtitle_version_id=None,
    )
    publication = PublicationIntent(
        publication_id=publication_id_value,
        edition_id=identity.edition_id,
        expected_video_key="contracts/video-digest/public.mp4",
        video_digest=assembled_file.content_digest,
        video_byte_size=len(assembled_file.content),
        video_media_type=assembled_file.media_type,
        source_video_version_id=assembled_file.version_id,
    )
    video_digest_catalog.record_publication_intent(lease, publication, recorded_at=recorded_at)
    video_digest_catalog.checkpoint_publication_progress(
        lease,
        publication_id_value,
        UploadingPublication(),
        recorded_at=recorded_at,
    )
    upload_file = artifact_file(
        artifact_id=f"{publication_id_value}:upload",
        artifact_kind="video_digest_publication_upload",
        title="Contract upload evidence",
        content=b"uploaded",
        r2_key="contracts/video-digest/upload.json",
        media_type="application/json",
    )
    video_digest_catalog.checkpoint_publication_progress(
        lease,
        publication_id_value,
        UploadedPublication(evidence_artifact_version_id=upload_file.version_id),
        evidence_file=upload_file,
        recorded_at=recorded_at,
    )
    public_verification_file = artifact_file(
        artifact_id=f"{publication_id_value}:verification",
        artifact_kind="video_digest_publication_verification",
        title="Contract public verification",
        content=b"public object verified",
        r2_key="contracts/video-digest/public-verification.json",
        media_type="application/json",
    )
    video_digest_catalog.checkpoint_publication_progress(
        lease,
        publication_id_value,
        VerifiedPublication(
            evidence_artifact_version_id=public_verification_file.version_id,
            video=VerifiedPublicObject(
                content_digest=publication.video_digest,
                byte_size=publication.video_byte_size,
                media_type=publication.video_media_type,
                source_artifact_version_id=publication.source_video_version_id,
            ),
        ),
        evidence_file=public_verification_file,
        recorded_at=recorded_at,
    )
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "CREATE FUNCTION reject_contract_slot_publication() RETURNS trigger "
            "LANGUAGE plpgsql AS $$ BEGIN "
            "IF NEW.stage = 'published' THEN "
            "RAISE EXCEPTION 'contract rejection' USING ERRCODE = '23000'; "
            "END IF; RETURN NEW; END; $$"
        )
        connection.execute(
            "CREATE TRIGGER contract_reject_slot_publication "
            "BEFORE UPDATE ON video_digest_slots "
            "FOR EACH ROW EXECUTE FUNCTION reject_contract_slot_publication()"
        )
    with pytest.raises(VideoDigestCheckpointConflictError):
        video_digest_catalog.complete_publication(
            lease, publication_id_value, recorded_at=recorded_at
        )
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        assert connection.execute(
            "SELECT stage FROM video_digest_publication_intents WHERE publication_id = %s",
            (publication_id_value,),
        ).fetchone() == ("verified",)
        connection.execute("DROP TRIGGER contract_reject_slot_publication ON video_digest_slots")
        connection.execute("DROP FUNCTION reject_contract_slot_publication()")

    published = video_digest_catalog.complete_publication(
        lease, publication_id_value, recorded_at=recorded_at
    )
    assert published == video_digest_catalog.complete_publication(
        lease, publication_id_value, recorded_at=recorded_at + timedelta(minutes=1)
    )
    reader_edition = video_digest_catalog.read_published_edition(
        identity.edition_id,
        public_media_base_url="https://media.example.com",
    )
    assert reader_edition is not None
    assert tuple(story.position for story in reader_edition.stories) == (0,)
    assert reader_edition.subtitle.kind == "failed"

    spend = video_digest_catalog.read_generation_spend(identity.edition_id)
    assert spend.measured_usd == Decimal("1.10")
    assert spend.estimated_usd == Decimal("0")
    assert spend.pending_requests == 0
    assert spend.unknown_requests == 0

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        row = connection.execute(
            "SELECT slot.stage, story.stage, request.stage, artifact.current_version_id "
            "FROM video_digest_slots AS slot "
            "JOIN video_digest_stories AS story ON story.edition_id = slot.edition_id "
            "JOIN video_digest_generation_requests AS request "
            "  ON request.edition_id = story.edition_id "
            " AND request.story_position = story.position "
            "JOIN artifacts AS artifact ON artifact.id = %s "
            "WHERE slot.slot_id = %s",
            (clip_file.artifact_id, slot.slot_id),
        ).fetchone()
        second_request_version = _record_artifact_versions(connection, 503, 1)[0]
        _insert_generation_request(
            connection,
            _sha256_id(504),
            identity.edition_id,
            second_request_version,
            attempt_index=1,
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            connection.execute(
                "UPDATE video_digest_generation_requests "
                "SET stage = 'submitted', provider_receipt_id = %s, "
                "cost_kind = 'estimated', cost_usd = 1, updated_at = CURRENT_TIMESTAMP "
                "WHERE request_id = %s",
                (receipt_id, _sha256_id(504)),
            )
    assert row == ("published", "accepted", "accepted", clip_file.version_id)
