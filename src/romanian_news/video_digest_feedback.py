from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from romanian_news import NewsModel
from romanian_news.catalog import video_digest_feedback as feedback_catalog
from romanian_news.video_digest.models import EditionIdField, StoryId


class VideoDigestFeedbackCommand(NewsModel):
    feedback_id: UUID
    edition_id: EditionIdField
    story_id: Annotated[StoryId | None, Field(pattern=r"^[0-9a-f]{64}$")] = None
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
    def require_rating_or_note(self) -> VideoDigestFeedbackCommand:
        if self.rating is None and self.note is None:
            raise ValueError("Feedback requires a rating or note")
        return self


class VideoDigestFeedbackEvent(VideoDigestFeedbackCommand):
    created_at: datetime


def submit_video_digest_feedback(
    command: VideoDigestFeedbackCommand,
) -> VideoDigestFeedbackEvent:
    record = feedback_catalog.append_video_digest_feedback(
        feedback_catalog.VideoDigestFeedbackWrite.model_validate(command.model_dump())
    )
    return VideoDigestFeedbackEvent.model_validate(record.model_dump())
