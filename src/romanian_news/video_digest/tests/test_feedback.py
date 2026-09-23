from uuid import UUID

import pytest
from pydantic import ValidationError

from romanian_news.video_digest.models import EditionId, StoryId
from romanian_news.video_digest_feedback import VideoDigestFeedbackCommand

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
