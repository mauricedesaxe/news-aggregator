from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from romanian_news import NewsModel, Sha256
from romanian_news.catalog_transport import (
    catalog_batch,
    catalog_query,
)


class DailyReportRecord(NewsModel):
    artifact_id: str
    report_version_id: str


class DailyReportVersionRecord(NewsModel):
    current_version_id: str


class DailyReportFileRecord(NewsModel):
    kind: str
    version_digest: str
    file_digest: str
    r2_key: str


class FeedbackRecord(NewsModel):
    feedback_id: str
    report_version_id: str
    target_kind: str
    theme_id: str | None
    group_id: str | None
    article_version_id: str | None
    rating: str | None
    note: str | None
    actor: str
    created_at: str


class FeedbackWrite(NewsModel):
    feedback_id: UUID
    report_version_id: Sha256
    target_kind: Literal["report", "theme", "group", "article"]
    theme_id: Sha256 | None
    group_id: Sha256 | None
    article_version_id: Sha256 | None
    rating: Literal["positive", "negative"] | None
    note: str | None
    actor: Literal["owner"]


class CuratedScoreRoute(NewsModel):
    score_id: str
    manifest_artifact_version_id: Sha256
    manifest_version: str
    feedback_id: str
    concern: Literal["language", "relevance", "grouping", "ranking"]
    report_version_id: Sha256
    model_output_version_id: Sha256
    model_attempt_id: Sha256
    polarity: Literal["positive", "negative"]
    rationale: str
    note: str | None
    provider: str | None
    trace_id: str | None
    observation_id: str | None


def list_daily_report_records(limit: int, offset: int = 0) -> tuple[DailyReportRecord, ...]:
    rows = catalog_query(
        """
        SELECT artifact.id AS artifact_id, version.id AS report_version_id
        FROM artifacts artifact
        JOIN artifact_versions version ON version.id = artifact.current_version_id
        WHERE artifact.kind = 'news_daily_report'
        ORDER BY artifact.id DESC
        LIMIT %s OFFSET %s
        """,
        [limit, offset],
    )
    return tuple(DailyReportRecord.model_validate(row, strict=False) for row in rows)


def read_current_daily_report_version(
    report_version_id: Sha256,
) -> DailyReportVersionRecord | None:
    rows = catalog_query(
        """
        SELECT artifact.current_version_id
        FROM artifact_versions version
        JOIN artifacts artifact ON artifact.id = version.artifact_id
        WHERE version.id = %s
          AND artifact.kind = 'news_daily_report'
          AND artifact.current_version_id IS NOT NULL
        """,
        [report_version_id],
    )
    if len(rows) != 1:
        return None
    return DailyReportVersionRecord.model_validate(rows[0], strict=False)


def read_daily_report_file(report_version_id: Sha256) -> DailyReportFileRecord | None:
    rows = catalog_query(
        """
        SELECT artifact.kind, version.content_digest AS version_digest,
               file.content_digest AS file_digest, file.r2_key
        FROM artifact_versions version
        JOIN artifacts artifact ON artifact.id = version.artifact_id
        JOIN artifact_files file ON file.artifact_version_id = version.id
        WHERE version.id = %s AND artifact.kind = 'news_daily_report'
        """,
        [report_version_id],
    )
    if len(rows) != 1:
        return None
    return DailyReportFileRecord.model_validate(rows[0], strict=False)


def append_feedback_event(value: FeedbackWrite) -> FeedbackRecord:
    catalog_batch(
        [
            (
                """INSERT INTO news_feedback
                (feedback_id, report_version_id, target_kind, theme_id, group_id,
                 article_version_id, rating, note, actor, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                [
                    str(value.feedback_id),
                    value.report_version_id,
                    value.target_kind,
                    value.theme_id,
                    value.group_id,
                    value.article_version_id,
                    value.rating,
                    value.note,
                    value.actor,
                    datetime.now(UTC).isoformat(),
                ],
            )
        ]
    )
    rows = catalog_query(
        "SELECT * FROM news_feedback WHERE feedback_id = %s", [str(value.feedback_id)]
    )
    if len(rows) != 1:
        raise RuntimeError(f"Feedback event was not recorded: {value.feedback_id}")
    return FeedbackRecord.model_validate(rows[0], strict=False)


def read_latest_feedback_records(report_version_id: Sha256) -> tuple[FeedbackRecord, ...]:
    rows = catalog_query(
        """
        SELECT feedback_id, report_version_id, target_kind, theme_id, group_id,
               article_version_id, rating, note, actor, created_at
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY target_kind, theme_id, group_id, article_version_id
                ORDER BY created_at DESC, feedback_id DESC
            ) AS position
            FROM news_feedback
            WHERE report_version_id = %s
        ) ranked
        WHERE position = 1
        ORDER BY target_kind, theme_id, group_id, article_version_id
        """,
        [report_version_id],
    )
    return tuple(FeedbackRecord.model_validate(row, strict=False) for row in rows)


def read_pending_feedback_score_routes(
    manifest_artifact_version_id: Sha256,
) -> tuple[CuratedScoreRoute, ...]:
    rows = catalog_query(
        """
        SELECT curation.score_id, curation.manifest_artifact_version_id,
               curation.manifest_version, curation.feedback_id, curation.concern,
               curation.report_version_id, curation.model_output_version_id,
               curation.model_attempt_id, curation.polarity, curation.rationale,
               feedback.note, trace.provider, trace.trace_id, trace.observation_id
        FROM news_feedback_score_curation curation
        JOIN news_feedback feedback ON feedback.feedback_id = curation.feedback_id
        LEFT JOIN news_model_trace_links trace
          ON trace.attempt_id = curation.model_attempt_id
        WHERE curation.manifest_artifact_version_id = %s
          AND curation.decision = 'projected'
          AND NOT EXISTS (
              SELECT 1 FROM news_feedback_concern_score_sync_attempts sync
              WHERE sync.score_id = curation.score_id
                AND sync.provider = 'langfuse'
                AND sync.status = 'completed'
          )
          AND NOT EXISTS (
              SELECT 1 FROM news_feedback_concern_score_dispositions disposition
              WHERE disposition.score_id = curation.score_id
                AND disposition.provider = 'langfuse'
          )
        ORDER BY curation.feedback_id, curation.position
        """,
        [manifest_artifact_version_id],
    )
    return tuple(CuratedScoreRoute.model_validate(row, strict=False) for row in rows)


def record_provider_migration_disposition(score_id: str) -> None:
    catalog_batch(
        [
            (
                """INSERT INTO news_feedback_concern_score_dispositions
                (score_id, provider, disposition, reason, source_provider, recorded_at)
                VALUES (%s, 'langfuse', 'permanently_unresolved',
                        'provider_migration', 'langsmith', %s) ON CONFLICT DO NOTHING""",
                [score_id, datetime.now(UTC).isoformat()],
            )
        ]
    )


def record_feedback_sync_attempt(
    score_id: str,
    status: Literal["completed", "failed"],
    error: Exception | None,
) -> None:
    message = f"{type(error).__name__}: {error}" if error is not None else None
    catalog_batch(
        [
            (
                """INSERT INTO news_feedback_concern_score_sync_attempts
                (sync_attempt_id, score_id, provider, status, error, attempted_at)
                VALUES (%s, %s, 'langfuse', %s, %s, %s)""",
                [
                    str(uuid4()),
                    score_id,
                    status,
                    message,
                    datetime.now(UTC).isoformat(),
                ],
            )
        ]
    )
