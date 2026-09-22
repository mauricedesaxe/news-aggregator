from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, HttpUrl

from romanian_news import NewsModel, Sha256
from romanian_news.catalog import video_digest as video_digest_catalog
from romanian_news.catalog.artifacts import ArtifactFile
from romanian_news.catalog.video_digest import PublishedPlanArtifact
from romanian_news.storage import ResearchObjectIntegrityError, read_verified_r2_object
from romanian_news.video_digest.models import (
    DigestPlan,
    EditionId,
    EditionIdField,
    PublishedEdition,
    PublishedEditionSummary,
    SlotName,
    StoryIdField,
)
from romanian_news.video_digest.planning import VerifiedDigestPlan
from romanian_news.video_digest.planning_artifacts import parse_verified_plan_file


class ReaderEditionOption(NewsModel):
    edition_id: EditionIdField
    slot_name: SlotName
    published_at: AwareDatetime


class ReaderSubtitleAvailable(NewsModel):
    kind: Literal["available"] = "available"
    url: HttpUrl


class ReaderSubtitleFailed(NewsModel):
    kind: Literal["failed"] = "failed"


ReaderSubtitle = Annotated[
    ReaderSubtitleAvailable | ReaderSubtitleFailed,
    Field(discriminator="kind"),
]


class ReaderVideoStory(NewsModel):
    story_id: StoryIdField
    position: Annotated[int, Field(ge=0)]
    title: str
    narration: str
    requested_duration_ms: Annotated[int, Field(gt=0)]


class ReaderVideoEdition(NewsModel):
    edition_id: EditionIdField
    slot_name: SlotName
    published_at: AwareDatetime
    video_url: HttpUrl
    subtitle: ReaderSubtitle
    stories: Annotated[tuple[ReaderVideoStory, ...], Field(min_length=1)]


class ReaderVideoDigest(NewsModel):
    editions: Annotated[tuple[ReaderEditionOption, ...], Field(min_length=1)]
    selected: ReaderVideoEdition | None


def read_reader_video_digest(
    day: date,
    report_version_id: Sha256,
    selected_edition_id: EditionId | None,
    public_media_origin: str,
) -> ReaderVideoDigest | None:
    published = video_digest_catalog.list_published_editions(
        day,
        public_media_base_url=public_media_origin,
    )
    matching = tuple(
        edition
        for edition in published
        if edition.day == day and edition.daily_report_version_id == report_version_id
    )
    if not matching:
        return None
    options = tuple(
        ReaderEditionOption(
            edition_id=edition.edition_id,
            slot_name=edition.slot_name,
            published_at=edition.published_at,
        )
        for edition in matching
    )
    requested = selected_edition_id or matching[0].edition_id
    summary = next((edition for edition in matching if edition.edition_id == requested), None)
    if summary is None:
        return ReaderVideoDigest(editions=options, selected=None)

    edition = video_digest_catalog.read_published_edition(
        requested,
        public_media_base_url=public_media_origin,
    )
    artifact = video_digest_catalog.read_published_plan_artifact(requested)
    if edition is None or artifact is None:
        raise ResearchObjectIntegrityError("Published video digest lineage is incomplete")
    content = read_verified_r2_object(artifact.r2_key, artifact.file_digest)
    if artifact.version_digest != artifact.file_digest:
        raise ResearchObjectIntegrityError("Published video digest plan digests disagree")
    file = ArtifactFile(
        artifact_id=artifact.artifact_id,
        artifact_kind=artifact.artifact_kind,
        title=artifact.title,
        version_id=artifact.version_id,
        content_digest=artifact.file_digest,
        r2_key=artifact.r2_key,
        media_type=artifact.media_type,
        content=content,
    )
    verified, plan = parse_verified_plan_file(file)
    _require_exact_lineage(
        day=day,
        report_version_id=report_version_id,
        summary=summary,
        edition=edition,
        artifact=artifact,
        verified=verified,
        plan=plan,
    )
    subtitle = (
        ReaderSubtitleAvailable(url=edition.subtitle.media.url)
        if edition.subtitle.kind == "available"
        else ReaderSubtitleFailed()
    )
    stories = tuple(
        ReaderVideoStory(
            story_id=published_story.story_id,
            position=published_story.position,
            title=published_story.title,
            narration=screenplay_story.narration,
            requested_duration_ms=published_story.requested_duration_ms,
        )
        for published_story, screenplay_story in zip(
            edition.stories,
            verified.plan.stories,
            strict=True,
        )
    )
    return ReaderVideoDigest(
        editions=options,
        selected=ReaderVideoEdition(
            edition_id=edition.edition_id,
            slot_name=edition.slot_name,
            published_at=edition.published_at,
            video_url=edition.video.url,
            subtitle=subtitle,
            stories=stories,
        ),
    )


def _require_exact_lineage(
    *,
    day: date,
    report_version_id: Sha256,
    summary: PublishedEditionSummary,
    edition: PublishedEdition,
    artifact: PublishedPlanArtifact,
    verified: VerifiedDigestPlan,
    plan: DigestPlan,
) -> None:
    expected_stories = tuple(
        (
            story.story_id,
            story.position,
            story.report_subject_id,
            story.title,
            story.requested_duration_ms,
        )
        for story in plan.stories
    )
    published_stories = tuple(
        (
            story.story_id,
            story.position,
            story.report_subject_id,
            story.title,
            story.requested_duration_ms,
        )
        for story in edition.stories
    )
    summary_fields = summary.model_dump()
    edition_fields = edition.model_dump(exclude={"stories"})
    if (
        summary_fields != edition_fields
        or edition.day != day
        or edition.daily_report_version_id != report_version_id
        or artifact.edition_id != edition.edition_id
        or artifact.daily_report_version_id != report_version_id
        or plan.artifact_version_id != artifact.version_id
        or plan.edition_id != edition.edition_id
        or verified.plan.edition_id != edition.edition_id
        or verified.plan.daily_report_version_id != report_version_id
        or verified.plan.policy_bundle_version_id != artifact.policy_bundle_version_id
        or expected_stories != published_stories
    ):
        raise ResearchObjectIntegrityError("Published video digest lineage does not match its plan")


def edition_label(slot_name: SlotName, published_at: datetime) -> str:
    return f"{slot_name.value.title()} · {published_at:%H:%M}"
