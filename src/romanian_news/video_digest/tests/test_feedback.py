from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from romanian_news.catalog import video_digest_feedback as feedback_catalog
from romanian_news.video_digest.models import EditionId, StoryId
from romanian_news.video_digest_feedback import (
    VideoDigestFeedbackCommand,
    submit_video_digest_feedback,
)

EDITION_ID = EditionId("1" * 64)
STORY_ID = StoryId("2" * 64)
FEEDBACK_ID = UUID("11111111-1111-1111-1111-111111111111")


def test_feedback_command_normalizes_note_and_requires_content() -> None:
    command = VideoDigestFeedbackCommand(
        feedback_id=FEEDBACK_ID,
        edition_id=EDITION_ID,
        story_id=STORY_ID,
        note="  useful note  ",
    )

    assert command.note == "useful note"
    with pytest.raises(ValidationError, match="Feedback requires a rating or note"):
        VideoDigestFeedbackCommand(
            feedback_id=FEEDBACK_ID,
            edition_id=EDITION_ID,
            note="   ",
        )


def test_feedback_command_rejects_invalid_identity_and_note() -> None:
    with pytest.raises(ValidationError):
        VideoDigestFeedbackCommand.model_validate(
            {
                "feedback_id": FEEDBACK_ID,
                "edition_id": "not-an-edition",
                "rating": "positive",
            }
        )
    with pytest.raises(ValidationError):
        VideoDigestFeedbackCommand(
            feedback_id=FEEDBACK_ID,
            edition_id=EDITION_ID,
            note="x" * 2001,
        )


def test_submit_feedback_passes_only_command_fields_to_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = VideoDigestFeedbackCommand(
        feedback_id=FEEDBACK_ID,
        edition_id=EDITION_ID,
        rating="negative",
        note="Too abrupt",
    )
    created_at = datetime(2026, 9, 22, 10, tzinfo=UTC)

    def append(
        value: feedback_catalog.VideoDigestFeedbackWrite,
    ) -> feedback_catalog.VideoDigestFeedbackRecord:
        assert value.model_dump() == command.model_dump()
        return feedback_catalog.VideoDigestFeedbackRecord(
            **value.model_dump(), created_at=created_at
        )

    monkeypatch.setattr(feedback_catalog, "append_video_digest_feedback", append)

    event = submit_video_digest_feedback(command)

    assert event.created_at == created_at
    assert event.model_dump(exclude={"created_at"}) == command.model_dump()
