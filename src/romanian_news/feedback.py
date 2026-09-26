from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, ValidationError, field_validator, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.catalog import feedback as feedback_catalog
from romanian_news.reports import (
    ArchivedDailyReport,
    DailyReportDocument,
    parse_daily_report,
)
from romanian_news.storage import ResearchObjectIntegrityError, read_verified_r2_object


class DailyReportVersionNotFound(ValueError):
    pass


class ReportFeedbackTarget(NewsModel):
    kind: Literal["report"] = "report"
    report_version_id: Sha256


class ThemeFeedbackTarget(NewsModel):
    kind: Literal["theme"] = "theme"
    report_version_id: Sha256
    theme_id: Sha256


class GroupFeedbackTarget(NewsModel):
    kind: Literal["group"] = "group"
    report_version_id: Sha256
    group_id: Sha256


class ArticleFeedbackTarget(NewsModel):
    kind: Literal["article"] = "article"
    report_version_id: Sha256
    group_id: Sha256
    article_version_id: Sha256


NewsFeedbackTarget = Annotated[
    ReportFeedbackTarget | ThemeFeedbackTarget | GroupFeedbackTarget | ArticleFeedbackTarget,
    Field(discriminator="kind"),
]


class NewsFeedbackCommand(NewsModel):
    feedback_id: UUID
    target: NewsFeedbackTarget
    rating: Literal["positive", "negative"] | None = None
    note: Annotated[str | None, Field(max_length=2000)] = None
    actor: Literal["owner"] = "owner"

    @field_validator("note", mode="before")
    @classmethod
    def normalize_note(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.strip() or None

    @model_validator(mode="after")
    def require_rating_or_note(self) -> NewsFeedbackCommand:
        if self.rating is None and self.note is None:
            raise ValueError("Feedback requires a rating or note")
        return self


class NewsFeedbackEvent(NewsFeedbackCommand):
    created_at: datetime


class DailyReportSummary(NewsModel):
    report_version_id: Sha256
    day: date


def list_daily_reports(limit: int = 30, offset: int = 0) -> tuple[DailyReportSummary, ...]:
    """List current daily reports in reverse chronological order."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Daily report limit must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("Daily report offset must be a non-negative integer")
    return tuple(
        _daily_report_summary(record)
        for record in feedback_catalog.list_daily_report_records(limit, offset)
    )


def resolve_current_daily_report_version(report_version_id: Sha256) -> Sha256:
    """Resolve a daily report version to its artifact's current version."""
    record = feedback_catalog.read_current_daily_report_version(report_version_id)
    if record is None:
        raise DailyReportVersionNotFound(
            f"Daily report version is unavailable: {report_version_id}"
        )
    return record.current_version_id


def read_daily_report_version(report_version_id: Sha256) -> DailyReportDocument:
    """Read and verify one exact daily report artifact version."""
    record = feedback_catalog.read_daily_report_file(report_version_id)
    if record is None:
        raise DailyReportVersionNotFound(
            f"Daily report version is unavailable: {report_version_id}"
        )
    if record.version_digest != record.file_digest:
        raise ResearchObjectIntegrityError(
            f"Daily report catalog digests disagree: {report_version_id}"
        )
    content = read_verified_r2_object(record.r2_key, record.file_digest)
    try:
        return parse_daily_report(content)
    except (ValidationError, ValueError) as error:
        raise ResearchObjectIntegrityError(
            f"Daily report payload is invalid: {report_version_id}"
        ) from error


def submit_news_feedback(command: NewsFeedbackCommand) -> NewsFeedbackEvent:
    """Validate and append one idempotent feedback event."""
    report = read_daily_report_version(command.target.report_version_id)
    _validate_target_membership(command.target, report)
    theme_id, group_id, article_version_id = _target_locator(command.target)
    record = feedback_catalog.append_feedback_event(
        feedback_catalog.FeedbackWrite(
            feedback_id=command.feedback_id,
            report_version_id=command.target.report_version_id,
            target_kind=command.target.kind,
            theme_id=theme_id,
            group_id=group_id,
            article_version_id=article_version_id,
            rating=command.rating,
            note=command.note,
            actor=command.actor,
        )
    )
    return news_feedback_event_from_row(record.model_dump())


def read_latest_news_feedback(report_version_id: Sha256) -> tuple[NewsFeedbackEvent, ...]:
    """Read the newest feedback event for each exact target in one report."""
    return tuple(
        news_feedback_event_from_row(record.model_dump())
        for record in feedback_catalog.read_latest_feedback_records(report_version_id)
    )


def _daily_report_summary(
    record: feedback_catalog.DailyReportRecord,
) -> DailyReportSummary:
    artifact_id = record.artifact_id
    prefix = "news:daily:"
    if not artifact_id.startswith(prefix):
        raise ValueError(f"Invalid daily report artifact identity: {artifact_id}")
    return DailyReportSummary(
        report_version_id=record.report_version_id,
        day=date.fromisoformat(artifact_id.removeprefix(prefix)),
    )


def _validate_target_membership(
    target: NewsFeedbackTarget,
    report: DailyReportDocument,
) -> None:
    if target.kind == "report":
        return
    if isinstance(report, ArchivedDailyReport):
        if target.kind == "theme":
            raise ValueError("Archived reports do not contain themes")
        sections = tuple(
            section for section in report.sections if section.group_id == target.group_id
        )
        if len(sections) != 1:
            raise ValueError(f"Feedback group is not in report version: {target.group_id}")
        if target.kind == "group":
            return
        if not any(
            article.article_version_id == target.article_version_id
            for article in sections[0].articles
        ):
            raise ValueError(
                f"Feedback article is not in report group: {target.article_version_id}"
            )
        return
    if target.kind == "theme":
        if sum(section.theme_id == target.theme_id for section in report.sections) != 1:
            raise ValueError(f"Feedback theme is not in report version: {target.theme_id}")
        return
    events = tuple(
        event
        for section in report.sections
        for event in section.events
        if event.group_id == target.group_id
    )
    if len(events) != 1:
        raise ValueError(f"Feedback group is not in report version: {target.group_id}")
    if target.kind == "group":
        return
    if not any(
        article.article_version_id == target.article_version_id for article in events[0].articles
    ):
        raise ValueError(f"Feedback article is not in report group: {target.article_version_id}")


def _target_locator(
    target: NewsFeedbackTarget,
) -> tuple[Sha256 | None, Sha256 | None, Sha256 | None]:
    if target.kind == "report":
        return None, None, None
    if target.kind == "theme":
        return target.theme_id, None, None
    if target.kind == "group":
        return None, target.group_id, None
    return None, target.group_id, target.article_version_id


def news_feedback_event_from_row(row: dict[str, object]) -> NewsFeedbackEvent:
    """Parse one stored feedback row into its domain event."""
    report_version_id = str(row["report_version_id"])
    theme_id = str(row["theme_id"]) if row.get("theme_id") is not None else None
    group_id = str(row["group_id"]) if row["group_id"] is not None else None
    article_version_id = (
        str(row["article_version_id"]) if row["article_version_id"] is not None else None
    )
    kind = str(row["target_kind"])
    if kind == "report":
        target: NewsFeedbackTarget = ReportFeedbackTarget(report_version_id=report_version_id)
    elif kind == "theme" and theme_id is not None:
        target = ThemeFeedbackTarget(report_version_id=report_version_id, theme_id=theme_id)
    elif kind == "group" and group_id is not None:
        target = GroupFeedbackTarget(report_version_id=report_version_id, group_id=group_id)
    elif kind == "article" and group_id is not None and article_version_id is not None:
        target = ArticleFeedbackTarget(
            report_version_id=report_version_id,
            group_id=group_id,
            article_version_id=article_version_id,
        )
    else:
        raise ValueError(f"Invalid stored feedback target: {kind}")
    note = row["note"]
    rating = row["rating"]
    if rating is not None and rating not in ("positive", "negative"):
        raise ValueError(f"Stored feedback has invalid rating: {rating}")
    actor = row["actor"]
    if actor != "owner":
        raise ValueError(f"Stored feedback has invalid actor: {actor}")
    return NewsFeedbackEvent(
        feedback_id=UUID(str(row["feedback_id"])),
        target=target,
        rating=rating,
        note=str(note) if note is not None else None,
        actor="owner",
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )
