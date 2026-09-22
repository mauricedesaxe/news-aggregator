from __future__ import annotations

from typing import Any, Literal, TypedDict
from uuid import UUID

import psycopg
import pytest

import romanian_news.catalog.schema as news_schema
from romanian_news.catalog import video_digest_feedback as feedback_catalog
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.video_digest.errors import VideoDigestFeedbackConflictError
from romanian_news.video_digest.models import EditionId, StoryId


def _digest(value: int) -> str:
    return f"{value:064x}"


class _SeededEdition(TypedDict):
    artifacts: dict[str, str]
    edition_id: str
    publication_id: str
    story_ids: tuple[str, str]
    request_ids: tuple[str, str]


def _seed_published_edition(connection: psycopg.Connection[Any]) -> _SeededEdition:
    names = (
        "report",
        "policy",
        "plan_evidence",
        "plan",
        "manifest",
        "video",
        "assembly_manifest",
        "assembly_evidence",
        "upload_evidence",
        "publication_evidence",
        "generation_policy",
        "story_0_verification",
        "story_0_request",
        "story_0_response",
        "story_0_clip",
        "story_0_validation",
        "story_1_verification",
        "story_1_request",
        "story_1_response",
        "story_1_clip",
        "story_1_validation",
    )
    artifacts = {name: _digest(index + 1) for index, name in enumerate(names)}
    for name, version_id in artifacts.items():
        connection.execute(
            "INSERT INTO artifacts "
            "(id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
            "VALUES (%s, 'test', %s, 'test', 'active', 'private', CURRENT_TIMESTAMP)",
            (f"artifact:{name}", name),
        )
        connection.execute(
            "INSERT INTO artifact_versions "
            "(id, artifact_id, schema_version, content_digest, created_at) "
            "VALUES (%s, %s, 1, %s, CURRENT_TIMESTAMP)",
            (version_id, f"artifact:{name}", version_id),
        )

    edition_id = _digest(100)
    publication_id = _digest(101)
    slot_id = _digest(102)
    story_ids = (_digest(103), _digest(104))
    request_ids = (_digest(105), _digest(106))
    connection.execute("SET session_replication_role = replica")
    connection.execute(
        """
        INSERT INTO video_digest_editions
            (edition_id, daily_report_version_id, policy_bundle_version_id,
             plan_artifact_version_id, verification_manifest_artifact_version_id,
             assembled_video_artifact_version_id, assembly_manifest_artifact_version_id,
             subtitle_state, planning_contract, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', 'verified_v1',
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (
            edition_id,
            artifacts["report"],
            artifacts["policy"],
            artifacts["plan"],
            artifacts["manifest"],
            artifacts["video"],
            artifacts["assembly_manifest"],
        ),
    )
    connection.execute(
        """
        INSERT INTO video_digest_slots
            (slot_id, name, scheduled_at, bucharest_day, stage, edition_id,
             claim_count, terminal_lease_owner_token, terminal_lease_expires_at,
             terminal_claim_count, terminal_fence_required, created_at, updated_at)
        VALUES (%s, 'morning', CURRENT_TIMESTAMP,
                (CURRENT_TIMESTAMP AT TIME ZONE 'Europe/Bucharest')::DATE,
                'published', %s, 1, 'owner', CURRENT_TIMESTAMP, 1, TRUE,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (slot_id, edition_id),
    )
    connection.execute(
        """
        INSERT INTO video_digest_planning_attempts
            (edition_id, attempt_index, disposition,
             attempt_evidence_artifact_version_id,
             accepted_plan_artifact_version_id, created_at)
        VALUES (%s, 0, 'accepted', %s, %s, CURRENT_TIMESTAMP)
        """,
        (edition_id, artifacts["plan_evidence"], artifacts["plan"]),
    )
    connection.execute(
        """
        INSERT INTO video_digest_assembly_attempts
            (edition_id, attempt_index, disposition, evidence_artifact_version_id,
             assembled_video_artifact_version_id,
             assembly_manifest_artifact_version_id, created_at)
        VALUES (%s, 0, 'succeeded', %s, %s, %s, CURRENT_TIMESTAMP)
        """,
        (
            edition_id,
            artifacts["assembly_evidence"],
            artifacts["video"],
            artifacts["assembly_manifest"],
        ),
    )
    for position, (story_id, request_id) in enumerate(zip(story_ids, request_ids, strict=True)):
        prefix = f"story_{position}"
        connection.execute(
            """
            INSERT INTO video_digest_stories
                (story_id, edition_id, position, report_subject_id, title, mandatory,
                 requested_duration_ms, stage,
                 verification_evidence_artifact_version_id,
                 accepted_clip_artifact_version_id, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, TRUE, 1000, 'accepted', %s, %s,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                story_id,
                edition_id,
                position,
                _digest(200 + position),
                f"Story {position}",
                artifacts[f"{prefix}_verification"],
                artifacts[f"{prefix}_clip"],
            ),
        )
        connection.execute(
            """
            INSERT INTO video_digest_generation_requests
                (request_id, edition_id, story_position, attempt_index,
                 request_artifact_version_id, generation_policy_artifact_version_id,
                 stage, provider_receipt_id, response_artifact_version_id,
                 accepted_clip_artifact_version_id,
                 validation_evidence_artifact_version_id,
                 cost_kind, cost_usd, created_at, updated_at)
            VALUES (%s, %s, %s, 0, %s, %s, 'accepted', %s, %s, %s, %s,
                    'measured', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                request_id,
                edition_id,
                position,
                artifacts[f"{prefix}_request"],
                artifacts["generation_policy"],
                f"receipt-{position}",
                artifacts[f"{prefix}_response"],
                artifacts[f"{prefix}_clip"],
                artifacts[f"{prefix}_validation"],
            ),
        )
    connection.execute(
        """
        INSERT INTO video_digest_publication_intents
            (publication_id, edition_id, expected_video_key, video_digest,
             video_byte_size, video_media_type, source_video_artifact_version_id,
             stage, evidence_required, normalized_keys_required,
             upload_evidence_artifact_version_id,
             verification_evidence_artifact_version_id,
             cache_control, visibility, retention, created_at, updated_at, published_at)
        VALUES (%s, %s, 'digest.mp4', %s, 100, 'video/mp4', %s, 'published',
                TRUE, TRUE, %s, %s, 'public,max-age=31536000,immutable',
                'public', 'permanent', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP)
        """,
        (
            publication_id,
            edition_id,
            _digest(300),
            artifacts["video"],
            artifacts["upload_evidence"],
            artifacts["publication_evidence"],
        ),
    )
    connection.execute("SET session_replication_role = origin")
    return {
        "artifacts": artifacts,
        "edition_id": edition_id,
        "publication_id": publication_id,
        "story_ids": story_ids,
        "request_ids": request_ids,
    }


def _write(
    feedback_id: UUID,
    seeded: _SeededEdition,
    *,
    story_id: str | None = None,
    rating: Literal["positive", "negative"] | None = "positive",
) -> feedback_catalog.VideoDigestFeedbackWrite:
    return feedback_catalog.VideoDigestFeedbackWrite(
        feedback_id=feedback_id,
        edition_id=EditionId(str(seeded["edition_id"])),
        story_id=StoryId(story_id) if story_id is not None else None,
        rating=rating,
        note=None,
        actor="owner",
    )


def test_catalog_snapshots_edition_and_story_lineage(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        seeded = _seed_published_edition(connection)

    feedback_id = UUID("11111111-1111-1111-1111-111111111111")
    write = _write(feedback_id, seeded)
    first = feedback_catalog.append_video_digest_feedback(write)
    second = feedback_catalog.append_video_digest_feedback(write)

    assert second == first
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        parent = connection.execute(
            "SELECT publication_id, daily_report_version_id, policy_bundle_version_id, "
            "plan_artifact_version_id, verification_manifest_artifact_version_id, "
            "assembled_video_artifact_version_id, assembly_manifest_artifact_version_id, "
            "publication_verification_evidence_artifact_version_id "
            "FROM video_digest_feedback WHERE feedback_id = %s",
            (feedback_id,),
        ).fetchone()
        children = connection.execute(
            "SELECT story_id, position, generation_request_id, "
            "generation_request_artifact_version_id, generation_policy_artifact_version_id, "
            "generation_response_artifact_version_id, accepted_clip_artifact_version_id, "
            "validation_evidence_artifact_version_id "
            "FROM video_digest_feedback_stories WHERE feedback_id = %s ORDER BY position",
            (feedback_id,),
        ).fetchall()

    artifacts = seeded["artifacts"]
    assert parent == (
        seeded["publication_id"],
        artifacts["report"],
        artifacts["policy"],
        artifacts["plan"],
        artifacts["manifest"],
        artifacts["video"],
        artifacts["assembly_manifest"],
        artifacts["publication_evidence"],
    )
    assert len(children) == 2
    assert tuple(row[0] for row in children) == seeded["story_ids"]
    assert tuple(row[2] for row in children) == seeded["request_ids"]


def test_story_feedback_snapshots_exactly_one_story_and_changed_retry_conflicts(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        seeded = _seed_published_edition(connection)
    story_id = seeded["story_ids"][1]
    feedback_id = UUID("22222222-2222-2222-2222-222222222222")

    feedback_catalog.append_video_digest_feedback(_write(feedback_id, seeded, story_id=story_id))
    with pytest.raises(VideoDigestFeedbackConflictError):
        feedback_catalog.append_video_digest_feedback(
            _write(feedback_id, seeded, story_id=story_id, rating="negative")
        )

    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        rows = connection.execute(
            "SELECT story_id FROM video_digest_feedback_stories WHERE feedback_id = %s",
            (feedback_id,),
        ).fetchall()
    assert rows == [(story_id,)]


def test_direct_sql_rejects_false_or_incomplete_lineage(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        seeded = _seed_published_edition(connection)
    artifacts = seeded["artifacts"]
    values = (
        seeded["edition_id"],
        seeded["publication_id"],
        artifacts["report"],
        artifacts["policy"],
        artifacts["plan"],
        artifacts["manifest"],
        artifacts["video"],
        artifacts["assembly_manifest"],
        artifacts["publication_evidence"],
    )
    statement = """
        INSERT INTO video_digest_feedback
            (feedback_id, edition_id, rating, actor, publication_id,
             daily_report_version_id, policy_bundle_version_id,
             plan_artifact_version_id, verification_manifest_artifact_version_id,
             assembled_video_artifact_version_id,
             assembly_manifest_artifact_version_id,
             publication_verification_evidence_artifact_version_id, created_at)
        VALUES (%s, %s, 'positive', 'owner', %s, %s, %s, %s, %s, %s, %s, %s,
                CURRENT_TIMESTAMP)
    """
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation, match="exact terminal"):
            connection.execute(
                statement,
                (UUID("33333333-3333-3333-3333-333333333333"), *values[:-1], _digest(999)),
            )
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation, match="incomplete"):
            connection.execute(
                statement,
                (UUID("44444444-4444-4444-4444-444444444444"), *values),
            )


def test_feedback_parent_and_children_are_immutable(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN) as connection:
        seeded = _seed_published_edition(connection)
    feedback_id = UUID("55555555-5555-5555-5555-555555555555")
    feedback_catalog.append_video_digest_feedback(_write(feedback_id, seeded))

    statements = (
        "UPDATE video_digest_feedback SET rating = 'negative' WHERE feedback_id = %s",
        "DELETE FROM video_digest_feedback WHERE feedback_id = %s",
        "UPDATE video_digest_feedback_stories SET position = 9 WHERE feedback_id = %s",
        "DELETE FROM video_digest_feedback_stories WHERE feedback_id = %s",
    )
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        for statement in statements:
            with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
                connection.execute(statement, (feedback_id,))
