from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from romanian_news.catalog.artifacts import ArtifactFile
from romanian_news.catalog.video_digest import PublishedPlanArtifact
from romanian_news.identity import sha256
from romanian_news.storage import ResearchObjectIntegrityError
from romanian_news.video_digest.models import (
    EditionIdentity,
    PublicationId,
    PublishedEdition,
    PublishedEditionSummary,
    PublishedMedia,
    PublishedStory,
    PublishedSubtitleAvailable,
    SlotName,
    edition_id,
)
from romanian_news.video_digest.planning import VerifiedDigestPlan
from romanian_news.video_digest.reader import read_reader_video_digest
from tests.contracts.video_digest_planning_fixtures import accepted_planning_files

REPORT_VERSION = "a" * 64
OTHER_REPORT_VERSION = "c" * 64
PUBLICATION_ID = "d" * 64
DAY = date(2026, 9, 21)


def test_reader_digest_discovers_exact_report_editions_in_catalog_order(monkeypatch) -> None:
    first, first_artifact = _published("new", datetime(2026, 9, 21, 18, tzinfo=UTC))
    second, _second_artifact = _published("old", datetime(2026, 9, 21, 12, tzinfo=UTC))
    first_summary = _summary(first)
    second_summary = _summary(second)
    unrelated = first_summary.model_copy(update={"daily_report_version_id": OTHER_REPORT_VERSION})
    reads: list[str] = []

    monkeypatch.setattr(
        "romanian_news.video_digest.reader.video_digest_catalog.list_published_editions",
        lambda *_args, **_kwargs: (first_summary, unrelated, second_summary),
    )
    monkeypatch.setattr(
        "romanian_news.video_digest.reader.video_digest_catalog.read_published_edition",
        lambda edition, **_kwargs: reads.append(edition) or first,
    )
    monkeypatch.setattr(
        "romanian_news.video_digest.reader.video_digest_catalog.read_published_plan_artifact",
        lambda _edition: _artifact_record(first, first_artifact),
    )
    monkeypatch.setattr(
        "romanian_news.video_digest.reader.read_verified_r2_object",
        lambda key, digest: first_artifact.content,
    )

    digest = read_reader_video_digest(DAY, REPORT_VERSION, None, "https://media.example.com")

    assert digest is not None
    assert tuple(option.edition_id for option in digest.editions) == (
        first.edition_id,
        second.edition_id,
    )
    assert digest.selected is not None
    assert digest.selected.edition_id == first.edition_id
    assert reads == [first.edition_id]
    assert digest.selected.video_url == first.video.url
    assert digest.selected.stories[0].narration.startswith("word0 word1")
    assert "r2_key" not in digest.model_dump()
    assert first_artifact.version_id not in str(digest.model_dump())


def test_reader_digest_does_not_read_an_edition_from_another_report(monkeypatch) -> None:
    published, _artifact = _published("other", datetime(2026, 9, 21, 18, tzinfo=UTC))
    summary = _summary(published).model_copy(
        update={"daily_report_version_id": OTHER_REPORT_VERSION}
    )
    monkeypatch.setattr(
        "romanian_news.video_digest.reader.video_digest_catalog.list_published_editions",
        lambda *_args, **_kwargs: (summary,),
    )

    digest = read_reader_video_digest(
        DAY,
        REPORT_VERSION,
        summary.edition_id,
        "https://media.example.com",
    )

    assert digest is None


@pytest.mark.parametrize("mismatch", ["report", "policy", "story"])
def test_reader_digest_rejects_mismatched_published_lineage(monkeypatch, mismatch: str) -> None:
    published, plan_file = _published("mismatch", datetime(2026, 9, 21, 18, tzinfo=UTC))
    artifact = _artifact_record(published, plan_file)
    edition = published
    if mismatch == "report":
        artifact = artifact.model_copy(update={"daily_report_version_id": OTHER_REPORT_VERSION})
    elif mismatch == "policy":
        artifact = artifact.model_copy(update={"policy_bundle_version_id": "e" * 64})
    else:
        story = edition.stories[0].model_copy(update={"title": "Changed after publication"})
        edition = edition.model_copy(update={"stories": (story,)})
    monkeypatch.setattr(
        "romanian_news.video_digest.reader.video_digest_catalog.list_published_editions",
        lambda *_args, **_kwargs: (_summary(published),),
    )
    monkeypatch.setattr(
        "romanian_news.video_digest.reader.video_digest_catalog.read_published_edition",
        lambda *_args, **_kwargs: edition,
    )
    monkeypatch.setattr(
        "romanian_news.video_digest.reader.video_digest_catalog.read_published_plan_artifact",
        lambda _edition: artifact,
    )
    monkeypatch.setattr(
        "romanian_news.video_digest.reader.read_verified_r2_object",
        lambda key, digest: plan_file.content,
    )

    with pytest.raises(ResearchObjectIntegrityError):
        read_reader_video_digest(
            DAY,
            REPORT_VERSION,
            published.edition_id,
            "https://media.example.com",
        )


def _published(seed: str, published_at: datetime) -> tuple[PublishedEdition, ArtifactFile]:
    policy_version = sha256(f"policy:{seed}".encode())
    identity = EditionIdentity(
        edition_id=edition_id(REPORT_VERSION, policy_version),
        daily_report_version_id=REPORT_VERSION,
        policy_bundle_version_id=policy_version,
    )
    plan, plan_file, _attempt_file = accepted_planning_files(
        identity,
        (("f" * 64, f"Story {seed}", 15_000),),
        seed=seed,
    )
    media = PublishedMedia.model_validate(
        {
            "url": f"https://media.example.com/video/{seed}.mp4",
            "content_digest": "1" * 64,
            "byte_size": 123,
            "media_type": "video/mp4",
        }
    )
    subtitle = PublishedSubtitleAvailable(
        media=PublishedMedia.model_validate(
            {
                "url": f"https://media.example.com/video/{seed}.vtt",
                "content_digest": "2" * 64,
                "byte_size": 42,
                "media_type": "text/vtt",
            }
        )
    )
    summary = PublishedEditionSummary(
        edition_id=identity.edition_id,
        publication_id=PublicationId(PUBLICATION_ID),
        day=DAY,
        slot_name=SlotName.EVENING,
        scheduled_at=published_at - timedelta(minutes=5),
        published_at=published_at,
        daily_report_version_id=REPORT_VERSION,
        video=media,
        subtitle=subtitle,
    )
    return (
        PublishedEdition(
            **summary.model_dump(),
            stories=tuple(
                PublishedStory(
                    story_id=story.story_id,
                    position=story.position,
                    report_subject_id=story.report_subject_id,
                    title=story.title,
                    requested_duration_ms=story.requested_duration_ms,
                )
                for story in plan.stories
            ),
        ),
        plan_file,
    )


def _artifact_record(edition: PublishedEdition, plan_file: ArtifactFile) -> PublishedPlanArtifact:
    verified = VerifiedDigestPlan.model_validate_json(plan_file.content, strict=True)
    return PublishedPlanArtifact(
        edition_id=edition.edition_id,
        daily_report_version_id=edition.daily_report_version_id,
        policy_bundle_version_id=verified.plan.policy_bundle_version_id,
        artifact_id=plan_file.artifact_id,
        artifact_kind=plan_file.artifact_kind,
        title=plan_file.title,
        version_id=plan_file.version_id,
        version_digest=plan_file.content_digest,
        file_digest=plan_file.content_digest,
        r2_key=plan_file.r2_key,
        media_type=plan_file.media_type,
    )


def _summary(edition: PublishedEdition) -> PublishedEditionSummary:
    return PublishedEditionSummary.model_validate(edition.model_dump(exclude={"stories"}))
