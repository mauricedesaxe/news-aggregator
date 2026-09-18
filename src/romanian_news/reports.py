from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, TypeAdapter

from romanian_news import BUCHAREST, NewsModel, Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.groups.models import GroupSentiment, GroupSummary
from romanian_news.articles.models import ExtractedArticle
from romanian_news.groups import DailyClusterSet
from romanian_news.storage import read_verified_r2_object
from romanian_news.subject_assessments import (
    DailySubjectAssessmentSet,
    SubjectAssessment,
    parse_daily_subject_assessment_set,
    subject_order_key,
)
from romanian_news.themes import (
    AliasedReaderSubjectThemeSet,
    DailyTheme,
    DailyThemeSet,
    ReaderSubjectDailyThemeSet,
    SparseDailyThemeSet,
    parse_daily_theme_set,
)

if TYPE_CHECKING:
    from romanian_news.groups import NewsGroup

WEEKLY_REPORT_POLICY = "seven-current-daily-reports-v1"
_SHA256_ADAPTER = TypeAdapter(Sha256)


class ReportArticle(NewsModel):
    article_version_id: Sha256
    outlet_id: str
    title: Annotated[str, Field(min_length=1)]
    canonical_url: str
    sentiment_label: str
    sentiment_score: float


class ReportEvent(NewsModel):
    group_id: Sha256
    title_ro: Annotated[str, Field(min_length=1)]
    summary_ro: Annotated[str, Field(min_length=1)]
    key_points_ro: tuple[str, ...]
    disagreements_ro: tuple[str, ...]
    uncertainty_ro: Annotated[str, Field(min_length=1)] | None = None
    sentiment_label: str
    sentiment_score: float
    sentiment_rationale_ro: Annotated[str, Field(min_length=1)]
    articles: tuple[ReportArticle, ...]


class DailyReportSectionV2(NewsModel):
    theme_id: Sha256
    title: Annotated[str, Field(min_length=1)]
    summary: Annotated[str, Field(min_length=1)]
    events: Annotated[tuple[ReportEvent, ...], Field(min_length=1)]


class DailyReportV2(NewsModel):
    schema_version: Literal[2] = 2
    day: date
    accepted_article_count: Annotated[int, Field(ge=0)]
    theme_count: Annotated[int, Field(ge=0)]
    group_count: Annotated[int, Field(ge=0)]
    sections: tuple[DailyReportSectionV2, ...]


class ReportSubjectCitation(NewsModel):
    article_version_id: Sha256
    evidence_quote: Annotated[str, Field(min_length=1)]


class DailyReportSection(DailyReportSectionV2):
    tier: Literal["main", "worth_knowing", "excluded"]
    semantic_rank: Annotated[int, Field(ge=1)]
    consequence_rationale: Annotated[str, Field(min_length=1, max_length=300)]
    citations: Annotated[tuple[ReportSubjectCitation, ...], Field(min_length=1)]


class DailyReport(NewsModel):
    schema_version: Literal[3] = 3
    day: date
    accepted_article_count: Annotated[int, Field(ge=0)]
    theme_count: Annotated[int, Field(ge=0)]
    group_count: Annotated[int, Field(ge=0)]
    sections: tuple[DailyReportSection, ...]


class ArchivedDailyReportSection(NewsModel):
    group_id: Sha256
    title_ro: str
    summary_ro: str
    key_points_ro: tuple[str, ...]
    disagreements_ro: tuple[str, ...]
    uncertainty_ro: str | None = None
    sentiment_label: str
    sentiment_score: float
    sentiment_rationale_ro: str
    articles: tuple[ReportArticle, ...]


class ArchivedDailyReport(NewsModel):
    day: date
    accepted_article_count: Annotated[int, Field(ge=0)]
    group_count: Annotated[int, Field(ge=0)]
    sections: tuple[ArchivedDailyReportSection, ...]


DailyReportDocument = DailyReport | DailyReportV2 | ArchivedDailyReport


def report_section_events(
    report: DailyReportDocument,
) -> tuple[ReportEvent | ArchivedDailyReportSection, ...]:
    """Flatten either report shape into the events that carry articles and a group."""
    if isinstance(report, ArchivedDailyReport):
        return report.sections
    return tuple(event for section in report.sections for event in section.events)


class DailyReportInput(NewsModel):
    day: date
    themes: ArtifactReference
    assessments: ArtifactReference
    cluster_set: ArtifactReference
    summaries: tuple[ArtifactReference, ...]
    sentiments: tuple[ArtifactReference, ...]


class DailyReportOutput(NewsModel):
    request_id: Sha256
    report: DailyReportDocument
    themes: ArtifactReference
    assessments: ArtifactReference
    cluster_set: ArtifactReference
    summaries: tuple[ArtifactReference, ...]
    sentiments: tuple[ArtifactReference, ...]
    content_digest: Sha256
    content: bytes


class ReportArticleSource(NewsModel):
    outlet_id: str
    title: str
    canonical_url: str


class DailyReportConstruction(NewsModel):
    inputs: DailyReportInput
    theme_set: (
        DailyThemeSet
        | SparseDailyThemeSet
        | ReaderSubjectDailyThemeSet
        | AliasedReaderSubjectThemeSet
    )
    assessment_set: DailySubjectAssessmentSet
    cluster_set: DailyClusterSet
    articles: dict[Sha256, ReportArticleSource]
    summaries: dict[str, GroupSummary]
    sentiments: dict[str, GroupSentiment]


class WeeklyReportDay(NewsModel):
    daily_report_version_id: Sha256
    report: DailyReportDocument


class WeeklyReport(NewsModel):
    week_start: date
    week_end: date
    accepted_article_count: Annotated[int, Field(ge=0)]
    group_count: Annotated[int, Field(ge=0)]
    days: tuple[WeeklyReportDay, ...]


class WeeklyReportInput(NewsModel):
    week_start: date
    policy: str
    daily_reports: tuple[ArtifactReference, ...]


class WeeklyReportOutput(NewsModel):
    request_id: Sha256
    report: WeeklyReport
    daily_reports: tuple[ArtifactReference, ...]
    content_digest: Sha256
    content: bytes


class WeeklyReportConstruction(NewsModel):
    inputs: WeeklyReportInput
    reports: dict[Sha256, DailyReportDocument]


class ReportInputsUnavailable(ValueError):
    """A report cannot be built until its required artifacts exist."""


def read_daily_report_input(day: date) -> DailyReportInput:
    """Discover one exact theme-anchored daily report input."""
    from romanian_news.analysis.groups.sentiment import sentiment_request_id
    from romanian_news.catalog.subject_assessments import (
        DailySubjectAssessmentUnavailable,
        read_daily_subject_assessment_reference,
    )
    from romanian_news.catalog.themes import DailyThemeUnavailable, read_daily_theme_reference

    try:
        themes = read_daily_theme_reference(day)
    except DailyThemeUnavailable as error:
        raise ReportInputsUnavailable(
            f"No current daily themes exist for {day.isoformat()}"
        ) from error
    theme_set = _load_daily_theme_set(themes)
    try:
        assessments = read_daily_subject_assessment_reference(day)
    except DailySubjectAssessmentUnavailable as error:
        raise ReportInputsUnavailable(
            f"No current subject assessments exist for {day.isoformat()}"
        ) from error
    assessment_set = _load_subject_assessment_set(assessments)
    if assessment_set.themes != themes:
        raise ValueError("Daily subject assessments reference another theme version")
    if theme_set.day != day:
        raise ValueError("Daily theme set belongs to another day")
    cluster_set = _load_cluster_set(theme_set.cluster_set)
    sentiments = tuple(
        _read_analysis_artifact_reference(
            f"news:sentiment:{sentiment_request_id(group)}", "news_sentiment"
        )
        for group in cluster_set.groups
    )
    return DailyReportInput(
        day=day,
        themes=themes,
        assessments=assessments,
        cluster_set=theme_set.cluster_set,
        summaries=theme_set.summary_inputs,
        sentiments=sentiments,
    )


_DAILY_INPUT_CACHE_LIMIT = 8
_daily_input_cache: dict[date, DailyReportInput] = {}
_daily_input_cache_lock = threading.Lock()


@dataclass(frozen=True)
class DailyReportInputIdentity:
    """The exact artifact references whose versions pin one report input."""

    value: DailyReportInput

    @property
    def references(self) -> tuple[ArtifactReference, ...]:
        return (
            self.value.themes,
            self.value.assessments,
            self.value.cluster_set,
            *self.value.summaries,
            *self.value.sentiments,
        )


def clear_daily_report_input_cache() -> None:
    """Drop every cached report input; tests and input re-reads use this."""
    with _daily_input_cache_lock:
        _daily_input_cache.clear()


def read_daily_report_input_cached(day: date) -> DailyReportInput:
    """Read one exact report input, reusing the parsed result while it stays current.

    Artifact versions are immutable, so a cached input stays valid until any of
    its referenced artifacts advances to another current version. Verifying that
    is one catalog query, which keeps repeated reads (reader renders, morning
    checks) off R2 entirely.
    """
    cached = _read_daily_input_cache(day)
    if cached is not None and _daily_report_input_matches(cached):
        return cached.value
    value = read_daily_report_input(day)
    _store_daily_input_cache(day, value)
    return value


def _read_daily_input_cache(day: date) -> DailyReportInputIdentity | None:
    with _daily_input_cache_lock:
        value = _daily_input_cache.get(day)
    return DailyReportInputIdentity(value=value) if value is not None else None


def _store_daily_input_cache(day: date, value: DailyReportInput) -> None:
    with _daily_input_cache_lock:
        _daily_input_cache.pop(day, None)
        _daily_input_cache[day] = value
        while len(_daily_input_cache) > _DAILY_INPUT_CACHE_LIMIT:
            _daily_input_cache.pop(next(iter(_daily_input_cache)))


def _daily_report_input_matches(identity: DailyReportInputIdentity) -> bool:
    from romanian_news.catalog.artifacts import current_artifact_references

    expected = {reference.artifact_id: reference.version_id for reference in identity.references}
    current = {
        reference.artifact_id: reference.version_id
        for reference in current_artifact_references(tuple(expected))
    }
    return current == expected


def read_recorded_daily_report_input(day: date) -> DailyReportInput:
    """Read the immutable inputs recorded by the current daily report run."""
    from romanian_news.catalog.report_inputs import read_recorded_daily_report_input as read_input

    value = read_input(day)
    if value is None:
        raise ValueError(f"Daily report is unavailable for {day.isoformat()}")
    if value.day != day:
        raise ValueError("Daily report run parameters do not match the requested day")
    return DailyReportInput(
        day=value.day,
        themes=_catalog_reference(value.themes),
        assessments=_catalog_reference(value.assessments),
        cluster_set=_catalog_reference(value.cluster_set),
        summaries=tuple(_catalog_reference(item) for item in value.summaries),
        sentiments=tuple(_catalog_reference(item) for item in value.sentiments),
    )


def daily_report_request_id(value: DailyReportInput) -> Sha256:
    """Compute the daily report identity from exact immutable references."""
    return _sha256(
        _canonical_json(
            {
                "theme_version_id": value.themes.version_id,
                "assessment_version_id": value.assessments.version_id,
                "cluster_set_version_id": value.cluster_set.version_id,
                "operation": "news.publish_daily",
                "report_schema": DailyReport.model_json_schema(),
                "sentiment_version_ids": [item.version_id for item in value.sentiments],
                "summary_version_ids": [item.version_id for item in value.summaries],
            }
        )
    )


def report_run_id(request_id: Sha256, implementation_ref: str) -> Sha256:
    """Identify one report run by its exact inputs and implementation."""
    if not implementation_ref:
        raise ValueError("Implementation reference is required")
    return _sha256(f"{request_id}\0{implementation_ref}".encode())


def stale_daily_report_days(
    scheduled_at: datetime,
    implementation_ref: str,
    not_before: date,
) -> tuple[date, ...]:
    """Find ready report days whose current head does not match current inputs."""
    local_time = _bucharest_time(scheduled_at)
    last_day = local_time.date()
    from romanian_news.catalog.report_inputs import read_report_day_heads

    heads = read_report_day_heads(not_before, last_day)
    stale = []
    for head in heads:
        day = head.day
        try:
            request_id = report_run_id(
                daily_report_request_id(read_daily_report_input_cached(day)), implementation_ref
            )
        except ValueError:
            continue
        if head.current_run_id != request_id:
            stale.append(day)
    return tuple(stale)


def build_daily_report_from_input(value: DailyReportInput) -> DailyReportOutput:
    """Build one report from exact inputs selected before the job was claimed."""
    theme_set = _load_daily_theme_set(value.themes)
    cluster_set = _load_cluster_set(value.cluster_set)
    return build_daily_report_from_construction(
        DailyReportConstruction(
            inputs=value,
            theme_set=theme_set,
            assessment_set=_load_subject_assessment_set(value.assessments),
            cluster_set=cluster_set,
            articles=_read_report_articles(cluster_set.article_version_ids),
            summaries={
                reference.artifact_id: _load_group_summary(reference, group.id)
                for group, reference in zip(cluster_set.groups, value.summaries, strict=True)
            },
            sentiments={
                reference.artifact_id: _load_group_sentiment(reference, group.id)
                for group, reference in zip(cluster_set.groups, value.sentiments, strict=True)
            },
        )
    )


def build_daily_report_from_construction(
    value: DailyReportConstruction,
) -> DailyReportOutput:
    """Build one nested daily report from fully loaded immutable inputs."""
    _validate_report_construction(value)
    summary_ids = {
        group.id: summary.artifact_id
        for group, summary in zip(value.cluster_set.groups, value.inputs.summaries, strict=True)
    }
    sentiment_ids = {
        group.id: sentiment.artifact_id
        for group, sentiment in zip(value.cluster_set.groups, value.inputs.sentiments, strict=True)
    }
    groups = {group.id: group for group in value.cluster_set.groups}
    themes = {theme.id: theme for theme in value.theme_set.themes}
    sections = tuple(
        _report_theme_section(
            themes[assessment.theme_id],
            assessment,
            groups,
            value,
            summary_ids,
            sentiment_ids,
        )
        for assessment in sorted(value.assessment_set.assessments, key=subject_order_key)
    )
    report = DailyReport(
        day=value.inputs.day,
        accepted_article_count=len(value.cluster_set.article_version_ids),
        theme_count=len(value.theme_set.themes),
        group_count=len(value.cluster_set.groups),
        sections=sections,
    )
    content = _canonical_json(report.model_dump(mode="json"))
    return DailyReportOutput(
        request_id=daily_report_request_id(value.inputs),
        report=report,
        themes=value.inputs.themes,
        assessments=value.inputs.assessments,
        cluster_set=value.inputs.cluster_set,
        summaries=value.inputs.summaries,
        sentiments=value.inputs.sentiments,
        content_digest=_sha256(content),
        content=content,
    )


def _report_theme_section(
    theme: DailyTheme,
    assessment: SubjectAssessment,
    groups: dict[Sha256, NewsGroup],
    construction: DailyReportConstruction,
    summary_ids: dict[Sha256, str],
    sentiment_ids: dict[Sha256, str],
) -> DailyReportSection:
    return DailyReportSection(
        theme_id=theme.id,
        title=theme.title,
        summary=theme.summary,
        tier=assessment.tier,
        semantic_rank=assessment.semantic_rank,
        consequence_rationale=assessment.rationale,
        citations=tuple(
            ReportSubjectCitation(
                article_version_id=item.article.version_id,
                evidence_quote=item.evidence_quote,
            )
            for item in assessment.evidence
        ),
        events=tuple(
            _report_event(
                groups[group_id],
                construction,
                summary_ids[group_id],
                sentiment_ids[group_id],
            )
            for group_id in theme.group_ids
        ),
    )


def _report_event(
    group: NewsGroup,
    construction: DailyReportConstruction,
    summary_id: str,
    sentiment_id: str,
) -> ReportEvent:
    summary = construction.summaries[summary_id]
    sentiment = construction.sentiments[sentiment_id]
    article_sentiments = {item.article_version_id: item for item in sentiment.articles}
    return ReportEvent(
        group_id=group.id,
        title_ro=summary.title_ro,
        summary_ro=summary.summary_ro,
        key_points_ro=summary.key_points_ro,
        disagreements_ro=summary.disagreements_ro,
        uncertainty_ro=summary.uncertainty_ro,
        sentiment_label=sentiment.overall.label,
        sentiment_score=sentiment.overall.score,
        sentiment_rationale_ro=sentiment.overall.rationale_ro,
        articles=tuple(
            ReportArticle(
                article_version_id=version_id,
                outlet_id=construction.articles[version_id].outlet_id,
                title=construction.articles[version_id].title,
                canonical_url=construction.articles[version_id].canonical_url,
                sentiment_label=article_sentiments[version_id].label,
                sentiment_score=article_sentiments[version_id].score,
            )
            for version_id in group.article_version_ids
        ),
    )


def _validate_report_construction(value: DailyReportConstruction) -> None:
    if value.inputs.day != value.theme_set.day or value.inputs.day != value.cluster_set.day:
        raise ValueError("Daily report inputs belong to different days")
    if value.theme_set.cluster_set != value.inputs.cluster_set:
        raise ValueError("Daily theme set references another cluster revision")
    if value.assessment_set.day != value.inputs.day:
        raise ValueError("Daily subject assessments belong to another day")
    if value.assessment_set.themes != value.inputs.themes:
        raise ValueError("Daily subject assessments reference another theme version")
    if value.inputs.themes.version_id == value.inputs.cluster_set.version_id:
        raise ValueError("Daily report theme and cluster references must be distinct")
    if value.theme_set.groups != value.cluster_set.groups:
        raise ValueError("Daily theme groups do not match the report cluster set")
    if value.theme_set.summary_inputs != value.inputs.summaries:
        raise ValueError("Daily report summaries do not match the theme inputs")
    expected_articles = set(value.cluster_set.article_version_ids)
    if set(value.articles) != expected_articles:
        raise ValueError("Daily report article inputs do not exactly cover the cluster set")
    themes = {theme.id: theme for theme in value.theme_set.themes}
    if set(value.assessment_set.subject_ids) != set(themes):
        raise ValueError("Daily subject assessments do not exactly cover report themes")
    for assessment in value.assessment_set.assessments:
        theme = themes[assessment.theme_id]
        if (assessment.group_ids, assessment.article_version_ids) != (
            theme.group_ids,
            theme.article_version_ids,
        ):
            raise ValueError("Daily subject assessment membership does not match its theme")
    if len(value.inputs.sentiments) != len(value.cluster_set.groups):
        raise ValueError("Daily report sentiments must exactly cover cluster groups")
    for group, summary_reference, sentiment_reference in zip(
        value.cluster_set.groups,
        value.inputs.summaries,
        value.inputs.sentiments,
        strict=True,
    ):
        summary = value.summaries[summary_reference.artifact_id]
        sentiment = value.sentiments[sentiment_reference.artifact_id]
        if not set(summary.cited_article_version_ids) <= set(group.article_version_ids):
            raise ValueError("Daily report summary cites an article outside its event")
        sentiment_ids = tuple(item.article_version_id for item in sentiment.articles)
        if len(sentiment_ids) != len(set(sentiment_ids)) or set(sentiment_ids) != set(
            group.article_version_ids
        ):
            raise ValueError("Daily report sentiment does not exactly cover its event")


def read_weekly_report_input(week_start: date) -> WeeklyReportInput:
    """Discover exact weekly report inputs without constructing report content."""
    if week_start.weekday() != 0:
        raise ValueError("Weekly report start must be a Monday")
    references = tuple(
        _read_daily_report_reference(week_start + timedelta(days=offset)) for offset in range(7)
    )
    return WeeklyReportInput(
        week_start=week_start, policy=WEEKLY_REPORT_POLICY, daily_reports=references
    )


def weekly_report_request_id(value: WeeklyReportInput) -> Sha256:
    """Compute the weekly report identity from exact immutable references."""
    return _sha256(
        _canonical_json(
            {
                "daily_report_version_ids": [item.version_id for item in value.daily_reports],
                "operation": "news.publish_weekly",
                "policy": value.policy,
                "report_schema": WeeklyReport.model_json_schema(),
                "week_start": value.week_start.isoformat(),
            }
        )
    )


def stale_weekly_report_weeks(
    scheduled_at: datetime,
    implementation_ref: str,
    not_before: date,
) -> tuple[date, ...]:
    """Find complete weeks whose current head does not match current daily reports."""
    local_day = _bucharest_time(scheduled_at).date()
    from romanian_news.catalog.report_inputs import (
        read_current_daily_report_days,
        read_weekly_report_heads,
    )

    daily_days = read_current_daily_report_days()
    current_runs = {value.week_start: value.current_run_id for value in read_weekly_report_heads()}
    week_starts = sorted(
        {
            day - timedelta(days=day.weekday())
            for day in daily_days
            if day - timedelta(days=day.weekday()) >= not_before
        }
    )
    stale = []
    for week_start in week_starts:
        week_days = {week_start + timedelta(days=offset) for offset in range(7)}
        if week_start + timedelta(days=6) >= local_day or not week_days <= daily_days:
            continue
        try:
            request_id = report_run_id(
                weekly_report_request_id(read_weekly_report_input(week_start)), implementation_ref
            )
        except ReportInputsUnavailable:
            continue
        if current_runs.get(week_start) != request_id:
            stale.append(week_start)
    return tuple(stale)


def build_weekly_report(week_start: date) -> WeeklyReportOutput:
    """Build one weekly report for the legacy monolithic runner."""
    if week_start.weekday() != 0:
        raise ValueError("Weekly report start must be a Monday")
    values = tuple(_read_daily_report(week_start + timedelta(days=offset)) for offset in range(7))
    references = tuple(reference for reference, _report in values)
    days = tuple(
        WeeklyReportDay(daily_report_version_id=reference.version_id, report=report)
        for reference, report in values
    )
    inputs = WeeklyReportInput(
        week_start=week_start, policy=WEEKLY_REPORT_POLICY, daily_reports=references
    )
    report = WeeklyReport(
        week_start=week_start,
        week_end=week_start + timedelta(days=6),
        accepted_article_count=sum(day.report.accepted_article_count for day in days),
        group_count=sum(day.report.group_count for day in days),
        days=days,
    )
    content = _canonical_json(report.model_dump(mode="json"))
    return WeeklyReportOutput(
        request_id=weekly_report_request_id(inputs),
        report=report,
        daily_reports=references,
        content_digest=_sha256(content),
        content=content,
    )


def build_weekly_report_from_input(value: WeeklyReportInput) -> WeeklyReportOutput:
    """Build one weekly report from exact inputs selected before the job was claimed."""
    return build_weekly_report_from_construction(
        WeeklyReportConstruction(
            inputs=value,
            reports={
                reference.version_id: _load_daily_report(reference)
                for reference in value.daily_reports
            },
        )
    )


def build_weekly_report_from_construction(
    value: WeeklyReportConstruction,
) -> WeeklyReportOutput:
    """Build one weekly report from fully loaded immutable daily reports."""
    days = tuple(
        WeeklyReportDay(
            daily_report_version_id=reference.version_id,
            report=value.reports[reference.version_id],
        )
        for reference in value.inputs.daily_reports
    )
    report = WeeklyReport(
        week_start=value.inputs.week_start,
        week_end=value.inputs.week_start + timedelta(days=6),
        accepted_article_count=sum(day.report.accepted_article_count for day in days),
        group_count=sum(day.report.group_count for day in days),
        days=days,
    )
    content = _canonical_json(report.model_dump(mode="json"))
    return WeeklyReportOutput(
        request_id=weekly_report_request_id(value.inputs),
        report=report,
        daily_reports=value.inputs.daily_reports,
        content_digest=_sha256(content),
        content=content,
    )


def _bucharest_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Report reconciliation time must include a timezone")
    return value.astimezone(BUCHAREST)


def _summary_request_id(group) -> Sha256:
    from romanian_news.analysis.groups.summary import summary_request_id

    return summary_request_id(group)


def _sentiment_request_id(group) -> Sha256:
    from romanian_news.analysis.groups.sentiment import sentiment_request_id

    return sentiment_request_id(group)


def _read_cluster_set(day: date) -> tuple[ArtifactReference, DailyClusterSet]:
    from romanian_news.catalog.artifacts import current_artifact_reference

    reference = current_artifact_reference(f"news:clusters:{day.isoformat()}", "news_clusters")
    if reference is None:
        raise ReportInputsUnavailable(f"No current cluster set exists for {day.isoformat()}")
    value = _catalog_reference(reference)
    return value, _load_cluster_set(value)


def _read_daily_report_reference(day: date) -> ArtifactReference:
    from romanian_news.catalog.artifacts import current_artifact_reference

    reference = current_artifact_reference(f"news:daily:{day.isoformat()}", "news_daily_report")
    if reference is None:
        raise ReportInputsUnavailable(f"No current daily report exists for {day.isoformat()}")
    return _catalog_reference(reference)


def parse_daily_report(content: bytes) -> DailyReportDocument:
    """Parse current and immutable archived daily reports without inventing themes."""
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Daily report payload must be an object")
    if payload.get("schema_version") == 3:
        return DailyReport.model_validate_json(content, strict=True)
    if payload.get("schema_version") == 2:
        return DailyReportV2.model_validate_json(content, strict=True)
    return ArchivedDailyReport.model_validate_json(content, strict=True)


def _load_daily_report(reference: ArtifactReference) -> DailyReportDocument:
    return parse_daily_report(read_verified_r2_object(reference.r2_key, reference.content_digest))


def _load_daily_theme_set(
    reference: ArtifactReference,
) -> (
    DailyThemeSet | SparseDailyThemeSet | ReaderSubjectDailyThemeSet | AliasedReaderSubjectThemeSet
):
    return parse_daily_theme_set(
        read_verified_r2_object(reference.r2_key, reference.content_digest)
    )


def _load_subject_assessment_set(
    reference: ArtifactReference,
) -> DailySubjectAssessmentSet:
    return parse_daily_subject_assessment_set(
        read_verified_r2_object(reference.r2_key, reference.content_digest)
    )


def _load_cluster_set(reference: ArtifactReference) -> DailyClusterSet:
    return DailyClusterSet.model_validate_json(
        read_verified_r2_object(reference.r2_key, reference.content_digest), strict=True
    )


def _load_group_summary(
    reference: ArtifactReference,
    expected_group_id: Sha256 | None = None,
) -> GroupSummary:
    payload = _load_analysis_artifact(reference)
    if expected_group_id is not None and payload.get("group_id") != expected_group_id:
        raise ValueError("Daily report summary references another group")
    return GroupSummary.model_validate_json(
        json.dumps(payload["summary"], ensure_ascii=False), strict=True
    )


def _load_group_sentiment(
    reference: ArtifactReference,
    expected_group_id: Sha256 | None = None,
) -> GroupSentiment:
    payload = _load_analysis_artifact(reference)
    if expected_group_id is not None and payload.get("group_id") != expected_group_id:
        raise ValueError("Daily report sentiment references another group")
    return GroupSentiment.model_validate_json(
        json.dumps(payload["sentiment"], ensure_ascii=False), strict=True
    )


def _read_daily_report(day: date) -> tuple[ArtifactReference, DailyReportDocument]:
    reference = _read_daily_report_reference(day)
    return reference, _load_daily_report(reference)


def _read_group_summary(request_id: Sha256) -> tuple[ArtifactReference, GroupSummary]:
    reference = _read_analysis_artifact_reference(f"news:summary:{request_id}", "news_summary")
    return reference, _load_group_summary(reference)


def _read_group_sentiment(request_id: Sha256) -> tuple[ArtifactReference, GroupSentiment]:
    reference = _read_analysis_artifact_reference(f"news:sentiment:{request_id}", "news_sentiment")
    return reference, _load_group_sentiment(reference)


def _read_analysis_artifact_reference(artifact_id: str, kind: str) -> ArtifactReference:
    from romanian_news.catalog.artifacts import current_artifact_reference

    reference = current_artifact_reference(artifact_id, kind)
    if reference is None:
        raise ReportInputsUnavailable(f"Required analysis artifact is unavailable: {artifact_id}")
    return _catalog_reference(reference)


def _load_analysis_artifact(reference: ArtifactReference) -> dict[str, object]:
    return json.loads(read_verified_r2_object(reference.r2_key, reference.content_digest))


def _read_report_articles(
    version_ids: tuple[Sha256, ...],
) -> dict[Sha256, ReportArticleSource]:
    from romanian_news.catalog.report_inputs import read_report_articles

    records = read_report_articles(version_ids)
    result = {}
    for version_id, record in records.items():
        article = ExtractedArticle.model_validate_json(
            read_verified_r2_object(record.reference.r2_key, record.reference.content_digest),
            strict=True,
        )
        result[version_id] = ReportArticleSource(
            outlet_id=record.outlet_id, canonical_url=record.canonical_url, title=article.title
        )
    if set(result) != set(version_ids):
        raise ValueError("Daily report references unknown article versions")
    return result


def _catalog_reference(value) -> ArtifactReference:
    return ArtifactReference.model_validate(value.model_dump(), strict=True)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(content: bytes) -> Sha256:
    return hashlib.sha256(content).hexdigest()
