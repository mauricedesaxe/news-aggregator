from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from romanian_news import NewsModel
from romanian_news.catalog_transport import (
    CatalogConnection,
    ResearchCatalogError,
    catalog_integrity_identity,
    catalog_transaction,
)
from romanian_news.video_digest.errors import (
    VideoDigestFeedbackConflictError,
    VideoDigestFeedbackTargetUnavailableError,
)
from romanian_news.video_digest.models import EditionId, StoryId


class VideoDigestFeedbackWrite(NewsModel):
    feedback_id: UUID
    edition_id: EditionId
    story_id: StoryId | None
    rating: Literal["positive", "negative"] | None
    note: str | None
    actor: Literal["owner"]


class VideoDigestFeedbackRecord(VideoDigestFeedbackWrite):
    created_at: datetime


def append_video_digest_feedback(value: VideoDigestFeedbackWrite) -> VideoDigestFeedbackRecord:
    def append(connection: CatalogConnection) -> VideoDigestFeedbackRecord:
        connection.execute(
            """
            INSERT INTO video_digest_feedback
                (feedback_id, edition_id, story_id, rating, note, actor,
                 publication_id, daily_report_version_id, policy_bundle_version_id,
                 plan_artifact_version_id, verification_manifest_artifact_version_id,
                 assembled_video_artifact_version_id,
                 assembly_manifest_artifact_version_id,
                 publication_verification_evidence_artifact_version_id, created_at)
            SELECT %s, edition.edition_id, %s, %s, %s, %s,
                   publication.publication_id, edition.daily_report_version_id,
                   edition.policy_bundle_version_id, edition.plan_artifact_version_id,
                   edition.verification_manifest_artifact_version_id,
                   edition.assembled_video_artifact_version_id,
                   edition.assembly_manifest_artifact_version_id,
                   publication.verification_evidence_artifact_version_id,
                   CURRENT_TIMESTAMP
            FROM video_digest_editions AS edition
            JOIN video_digest_publication_intents AS publication
              ON publication.edition_id = edition.edition_id
             AND publication.stage = 'published'
            WHERE edition.edition_id = %s
              AND EXISTS (
                  SELECT 1 FROM video_digest_slots AS slot
                  WHERE slot.edition_id = edition.edition_id
                    AND slot.stage = 'published'
              )
              AND edition.plan_artifact_version_id IS NOT NULL
              AND edition.verification_manifest_artifact_version_id IS NOT NULL
              AND edition.assembled_video_artifact_version_id IS NOT NULL
              AND edition.assembly_manifest_artifact_version_id IS NOT NULL
              AND publication.verification_evidence_artifact_version_id IS NOT NULL
              AND EXISTS (
                  SELECT 1 FROM video_digest_stories AS story
                  WHERE story.edition_id = edition.edition_id AND story.stage = 'accepted'
                    AND (%s::text IS NULL OR story.story_id = %s)
              )
            ON CONFLICT DO NOTHING
            """,
            (
                value.feedback_id,
                value.story_id,
                value.rating,
                value.note,
                value.actor,
                value.edition_id,
                value.story_id,
                value.story_id,
            ),
        )
        row = connection.execute(
            """
            SELECT feedback_id, edition_id, story_id, rating, note, actor, created_at
            FROM video_digest_feedback
            WHERE feedback_id = %s
            FOR UPDATE
            """,
            (value.feedback_id,),
        ).fetchone()
        if row is None:
            raise VideoDigestFeedbackTargetUnavailableError(
                f"Published video digest feedback target is unavailable: {value.edition_id}"
            )
        stored = VideoDigestFeedbackRecord.model_validate(row)
        if stored.model_dump(exclude={"created_at"}) != value.model_dump():
            raise VideoDigestFeedbackConflictError(
                f"Video digest feedback ID conflicts with another payload: {value.feedback_id}"
            )
        connection.execute(
            """
            INSERT INTO video_digest_feedback_stories
                (feedback_id, story_id, position,
                 verification_evidence_artifact_version_id, generation_request_id,
                 generation_request_artifact_version_id,
                 generation_policy_artifact_version_id,
                 generation_response_artifact_version_id,
                 accepted_clip_artifact_version_id,
                 validation_evidence_artifact_version_id)
            SELECT feedback.feedback_id, story.story_id, story.position,
                   story.verification_evidence_artifact_version_id,
                   generation.request_id, generation.request_artifact_version_id,
                   generation.generation_policy_artifact_version_id,
                   generation.response_artifact_version_id,
                   generation.accepted_clip_artifact_version_id,
                   generation.validation_evidence_artifact_version_id
            FROM video_digest_feedback AS feedback
            JOIN video_digest_stories AS story
              ON story.edition_id = feedback.edition_id AND story.stage = 'accepted'
             AND (feedback.story_id IS NULL OR story.story_id = feedback.story_id)
            JOIN video_digest_generation_requests AS generation
              ON generation.edition_id = story.edition_id
             AND generation.story_position = story.position
             AND generation.stage = 'accepted'
             AND generation.accepted_clip_artifact_version_id =
                 story.accepted_clip_artifact_version_id
            WHERE feedback.feedback_id = %s
            ON CONFLICT DO NOTHING
            """,
            (value.feedback_id,),
        )
        return stored

    try:
        return catalog_transaction(append, retry_transient_errors=False)
    except ResearchCatalogError as error:
        identity = catalog_integrity_identity(error)
        if identity is not None and identity.split(":", 2)[1].startswith("23"):
            raise VideoDigestFeedbackConflictError(
                "Stored video digest feedback conflicts with the request"
            ) from error
        raise
