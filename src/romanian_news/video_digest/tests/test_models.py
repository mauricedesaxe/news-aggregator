from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import TypeAdapter, ValidationError

from romanian_news.video_digest.models import (
    AttemptCost,
    ClaimResult,
    DigestPlan,
    EditionId,
    EditionIdentity,
    EstimatedAttemptCost,
    GenerationRequestIdentity,
    MeasuredAttemptCost,
    PendingAttemptCost,
    PlannedStory,
    PublicationId,
    PublicationIntent,
    ScheduledSlot,
    SlotId,
    SlotLease,
    SlotName,
    SubtitleObjectMetadata,
    SubtitleOutcome,
    TerminalSlotState,
    UnknownAttemptCost,
    edition_id,
    generation_request_id,
    planned_story_id,
    publication_id,
    scheduled_slot_id,
)

REPORT_VERSION = "a" * 64
POLICY_VERSION = "b" * 64
SUBJECT_ID = "c" * 64
REQUEST_VERSION = "d" * 64
VIDEO_VERSION = "e" * 64
SUBTITLE_VERSION = "f" * 64
VIDEO_DIGEST = "1" * 64
SUBTITLE_DIGEST = "2" * 64
SCHEDULED_AT = datetime(2026, 9, 20, 5, tzinfo=UTC)


def _edition() -> EditionIdentity:
    identity = edition_id(REPORT_VERSION, POLICY_VERSION)
    return EditionIdentity(
        edition_id=identity,
        daily_report_version_id=REPORT_VERSION,
        policy_bundle_version_id=POLICY_VERSION,
    )


def _subtitle() -> SubtitleObjectMetadata:
    edition = _edition().edition_id
    return SubtitleObjectMetadata(
        expected_key=f"video-digests/{edition}/{SUBTITLE_DIGEST}.vtt",
        content_digest=SUBTITLE_DIGEST,
        byte_size=321,
        media_type="text/vtt",
    )


def _publication_values() -> dict[str, object]:
    edition = _edition().edition_id
    subtitle = _subtitle()
    video_key = f"video-digests/{edition}/{VIDEO_DIGEST}.mp4"
    values: dict[str, object] = {
        "edition_id": edition,
        "expected_video_key": video_key,
        "video_digest": VIDEO_DIGEST,
        "video_byte_size": 123_456,
        "video_media_type": "video/mp4",
        "subtitle": subtitle,
        "source_video_version_id": VIDEO_VERSION,
        "source_subtitle_version_id": SUBTITLE_VERSION,
    }
    values["publication_id"] = publication_id(
        edition_id_value=edition,
        expected_video_key=video_key,
        video_digest=VIDEO_DIGEST,
        video_byte_size=123_456,
        video_media_type="video/mp4",
        subtitle=subtitle,
        source_video_version_id=VIDEO_VERSION,
        source_subtitle_version_id=SUBTITLE_VERSION,
    )
    return values


def test_edition_identity_is_stable_and_sensitive_to_each_input() -> None:
    first = edition_id(REPORT_VERSION, POLICY_VERSION)

    assert first == edition_id(REPORT_VERSION, POLICY_VERSION)
    assert first != edition_id("0" * 64, POLICY_VERSION)
    assert first != edition_id(REPORT_VERSION, "0" * 64)
    assert _edition().edition_id == first


def test_slot_identity_is_stable_and_sensitive() -> None:
    slot_id = scheduled_slot_id(SlotName.MORNING, SCHEDULED_AT)
    slot = ScheduledSlot(
        slot_id=slot_id,
        name=SlotName.MORNING,
        scheduled_at=SCHEDULED_AT,
        bucharest_day=date(2026, 9, 20),
    )

    assert slot.slot_id == scheduled_slot_id(SlotName.MORNING, SCHEDULED_AT)
    assert slot.slot_id != scheduled_slot_id(SlotName.MIDDAY, SCHEDULED_AT)
    assert slot.slot_id != scheduled_slot_id(
        SlotName.MORNING,
        datetime(2026, 9, 20, 6, tzinfo=UTC),
    )


def test_aware_datetimes_and_bucharest_day_are_required() -> None:
    with pytest.raises(ValueError, match="UTC offset"):
        scheduled_slot_id(SlotName.MORNING, datetime(2026, 9, 20, 8))

    with pytest.raises(ValidationError):
        SlotLease(
            slot_id=SlotId("1" * 64),
            edition_id=_edition().edition_id,
            owner_token="worker-1",
            expires_at=datetime(2026, 9, 20, 8),
            claim_count=1,
        )

    with pytest.raises(ValidationError, match="Europe/Bucharest"):
        ScheduledSlot(
            slot_id=scheduled_slot_id(SlotName.MORNING, SCHEDULED_AT),
            name=SlotName.MORNING,
            scheduled_at=SCHEDULED_AT,
            bucharest_day=date(2026, 9, 19),
        )


@pytest.mark.parametrize("bad_hash", ["A" * 64, "a" * 63, "g" * 64])
def test_hash_fields_reject_malformed_sha256_values(bad_hash: str) -> None:
    with pytest.raises(ValidationError):
        EditionIdentity(
            edition_id=EditionId(bad_hash),
            daily_report_version_id=REPORT_VERSION,
            policy_bundle_version_id=POLICY_VERSION,
        )

    with pytest.raises(ValidationError):
        EditionIdentity(
            edition_id=edition_id(REPORT_VERSION, POLICY_VERSION),
            daily_report_version_id=bad_hash,
            policy_bundle_version_id=POLICY_VERSION,
        )


def test_planned_story_accepts_only_valid_mandatory_records() -> None:
    edition = _edition().edition_id
    story = PlannedStory(
        story_id=planned_story_id(edition, 0, SUBJECT_ID),
        edition_id=edition,
        position=0,
        report_subject_id=SUBJECT_ID,
        title="Guvernul prezinta bugetul",
        requested_duration_ms=15_000,
    )

    assert story.mandatory is True

    for field, value in (
        ("position", -1),
        ("title", "  "),
        ("mandatory", False),
        ("requested_duration_ms", 0),
    ):
        values = story.model_dump()
        values[field] = value
        with pytest.raises(ValidationError):
            PlannedStory.model_validate(values)


def test_digest_plan_requires_ordered_unique_stories_from_one_edition() -> None:
    edition = _edition().edition_id
    first = PlannedStory(
        story_id=planned_story_id(edition, 0, SUBJECT_ID),
        edition_id=edition,
        position=0,
        report_subject_id=SUBJECT_ID,
        title="Prima stire",
        requested_duration_ms=15_000,
    )
    second = PlannedStory(
        story_id=planned_story_id(edition, 1, "3" * 64),
        edition_id=edition,
        position=1,
        report_subject_id="3" * 64,
        title="A doua stire",
        requested_duration_ms=12_000,
    )

    plan = DigestPlan(
        edition_id=edition,
        artifact_version_id="4" * 64,
        stories=(first, second),
    )

    assert plan.stories == (first, second)
    with pytest.raises(ValidationError, match="ordered"):
        DigestPlan(
            edition_id=edition,
            artifact_version_id="4" * 64,
            stories=(second, first),
        )
    with pytest.raises(ValidationError, match="subjects"):
        DigestPlan(
            edition_id=edition,
            artifact_version_id="4" * 64,
            stories=(
                first,
                second.model_copy(
                    update={
                        "story_id": planned_story_id(edition, 1, SUBJECT_ID),
                        "report_subject_id": SUBJECT_ID,
                    }
                ),
            ),
        )
    other_edition = EditionId("5" * 64)
    with pytest.raises(ValidationError, match="plan edition"):
        DigestPlan(
            edition_id=edition,
            artifact_version_id="4" * 64,
            stories=(
                first,
                second.model_copy(
                    update={
                        "story_id": planned_story_id(other_edition, 1, "3" * 64),
                        "edition_id": other_edition,
                    }
                ),
            ),
        )


def test_generation_request_identity_limits_attempts_and_tracks_artifact() -> None:
    edition = _edition().edition_id
    request_id = generation_request_id(edition, 0, 1, REQUEST_VERSION)
    request = GenerationRequestIdentity(
        request_id=request_id,
        edition_id=edition,
        story_position=0,
        attempt_index=1,
        request_artifact_version_id=REQUEST_VERSION,
    )

    assert request.request_id != generation_request_id(edition, 0, 0, REQUEST_VERSION)
    assert request.request_id != generation_request_id(edition, 1, 1, REQUEST_VERSION)
    assert request.request_id != generation_request_id(edition, 0, 1, "0" * 64)
    with pytest.raises(ValidationError):
        GenerationRequestIdentity(
            request_id=request_id,
            edition_id=edition,
            story_position=0,
            attempt_index=2,
            request_artifact_version_id=REQUEST_VERSION,
        )


def test_attempt_cost_variants_are_explicit_and_non_negative() -> None:
    adapter = TypeAdapter(AttemptCost)
    costs = (
        PendingAttemptCost(),
        EstimatedAttemptCost(usd=Decimal("1.25")),
        MeasuredAttemptCost(usd=Decimal("1.10")),
        UnknownAttemptCost(reason="Provider omitted billing data"),
    )

    assert [adapter.validate_python(cost).kind for cost in costs] == [
        "pending",
        "estimated",
        "measured",
        "unknown",
    ]
    with pytest.raises(ValidationError):
        EstimatedAttemptCost(usd=Decimal("-0.01"))
    with pytest.raises(ValidationError):
        UnknownAttemptCost(reason=" ")
    with pytest.raises(ValidationError):
        PendingAttemptCost.model_validate({"kind": "pending", "usd": Decimal("0")})


def test_claim_result_rejects_unknown_variants() -> None:
    adapter = TypeAdapter(ClaimResult)

    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "waiting"})


def test_terminal_slot_cannot_discard_a_skip_reason() -> None:
    with pytest.raises(ValueError):
        TerminalSlotState("skipped")


def test_subtitle_outcome_rejects_unknown_variants() -> None:
    adapter = TypeAdapter(SubtitleOutcome)

    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "pending"})


def test_publication_subtitle_metadata_is_all_or_none() -> None:
    values = _publication_values()
    assert PublicationIntent.model_validate(values).subtitle == _subtitle()

    without_source = values | {"source_subtitle_version_id": None}
    with pytest.raises(ValidationError, match="provided together"):
        PublicationIntent.model_validate(without_source)

    partial_subtitle = _subtitle().model_dump()
    del partial_subtitle["byte_size"]
    with pytest.raises(ValidationError):
        PublicationIntent.model_validate(values | {"subtitle": partial_subtitle})


def test_publication_identity_is_stable_and_sensitive_to_every_input() -> None:
    values = _publication_values()
    original = PublicationIntent.model_validate(values)
    edition = _edition().edition_id
    subtitle = _subtitle()

    assert original.publication_id == PublicationIntent.model_validate(values).publication_id
    changed_ids = (
        publication_id(
            edition_id_value=edition,
            expected_video_key=f"video-digests/other/{edition}/{VIDEO_DIGEST}.mp4",
            video_digest=VIDEO_DIGEST,
            video_byte_size=123_456,
            video_media_type="video/mp4",
            subtitle=subtitle,
            source_video_version_id=VIDEO_VERSION,
            source_subtitle_version_id=SUBTITLE_VERSION,
        ),
        publication_id(
            edition_id_value=edition,
            expected_video_key=f"video-digests/{edition}/{VIDEO_DIGEST}.mp4",
            video_digest="3" * 64,
            video_byte_size=123_456,
            video_media_type="video/mp4",
            subtitle=subtitle,
            source_video_version_id=VIDEO_VERSION,
            source_subtitle_version_id=SUBTITLE_VERSION,
        ),
        publication_id(
            edition_id_value=edition,
            expected_video_key=f"video-digests/{edition}/{VIDEO_DIGEST}.mp4",
            video_digest=VIDEO_DIGEST,
            video_byte_size=123_457,
            video_media_type="video/mp4",
            subtitle=subtitle,
            source_video_version_id=VIDEO_VERSION,
            source_subtitle_version_id=SUBTITLE_VERSION,
        ),
        publication_id(
            edition_id_value=edition,
            expected_video_key=f"video-digests/{edition}/{VIDEO_DIGEST}.mp4",
            video_digest=VIDEO_DIGEST,
            video_byte_size=123_456,
            video_media_type="video/webm",
            subtitle=subtitle,
            source_video_version_id=VIDEO_VERSION,
            source_subtitle_version_id=SUBTITLE_VERSION,
        ),
        publication_id(
            edition_id_value=edition,
            expected_video_key=f"video-digests/{edition}/{VIDEO_DIGEST}.mp4",
            video_digest=VIDEO_DIGEST,
            video_byte_size=123_456,
            video_media_type="video/mp4",
            subtitle=subtitle,
            source_video_version_id="4" * 64,
            source_subtitle_version_id=SUBTITLE_VERSION,
        ),
        publication_id(
            edition_id_value=edition,
            expected_video_key=f"video-digests/{edition}/{VIDEO_DIGEST}.mp4",
            video_digest=VIDEO_DIGEST,
            video_byte_size=123_456,
            video_media_type="video/mp4",
            subtitle=subtitle.model_copy(update={"byte_size": 322}),
            source_video_version_id=VIDEO_VERSION,
            source_subtitle_version_id=SUBTITLE_VERSION,
        ),
        publication_id(
            edition_id_value=edition,
            expected_video_key=f"video-digests/{edition}/{VIDEO_DIGEST}.mp4",
            video_digest=VIDEO_DIGEST,
            video_byte_size=123_456,
            video_media_type="video/mp4",
            subtitle=subtitle,
            source_video_version_id=VIDEO_VERSION,
            source_subtitle_version_id="5" * 64,
        ),
    )
    assert all(changed_id != original.publication_id for changed_id in changed_ids)


def test_publication_identity_includes_exact_public_policy() -> None:
    values = _publication_values()
    intent = PublicationIntent.model_validate(values)

    def identity(
        *,
        cache_control: str = intent.cache_control,
        visibility: str = intent.visibility,
        retention: str = intent.retention,
    ) -> PublicationId:
        return publication_id(
            edition_id_value=intent.edition_id,
            expected_video_key=intent.expected_video_key,
            video_digest=intent.video_digest,
            video_byte_size=intent.video_byte_size,
            video_media_type=intent.video_media_type,
            subtitle=intent.subtitle,
            source_video_version_id=intent.source_video_version_id,
            source_subtitle_version_id=intent.source_subtitle_version_id,
            cache_control=cache_control,
            visibility=visibility,
            retention=retention,
        )

    assert identity(cache_control="public,max-age=60") != intent.publication_id
    assert identity(visibility="private") != intent.publication_id
    assert identity(retention="candidate-7d") != intent.publication_id
    for field, value in (
        ("cache_control", "public,max-age=60"),
        ("visibility", "private"),
        ("retention", "candidate-7d"),
        ("video_media_type", "video/webm"),
    ):
        with pytest.raises(ValidationError):
            PublicationIntent.model_validate(values | {field: value})


def test_publication_rejects_invalid_size_and_mismatched_identity() -> None:
    values = _publication_values()
    with pytest.raises(ValidationError):
        PublicationIntent.model_validate(values | {"video_byte_size": 0})
    with pytest.raises(ValidationError, match="does not match"):
        PublicationIntent.model_validate(values | {"publication_id": PublicationId("0" * 64)})


@pytest.mark.parametrize(
    "key",
    ("/absolute.mp4", "video-digests//edition.mp4", "../edition.mp4", "video/./edition.mp4"),
)
def test_publication_rejects_non_normalized_public_object_keys(key: str) -> None:
    values = _publication_values()

    with pytest.raises(ValidationError):
        PublicationIntent.model_validate(values | {"expected_video_key": key})
