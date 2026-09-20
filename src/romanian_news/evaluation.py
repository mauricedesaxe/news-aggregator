from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AliasChoices,
    Field,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.relevance import (
    PRODUCTION_RELEVANCE_POLICY,
    RelevanceDecision,
    relevance_is_accepted,
)
from romanian_news.analysis.relevance_v3 import (
    RELEVANCE_V3_POLICY,
    ContextDecision,
    ImpactDecision,
    relevance_v3_is_accepted,
)
from romanian_news.artifacts import ArtifactReference
from romanian_news.feedback import NewsFeedbackTarget, ThemeFeedbackTarget
from romanian_news.groups import (
    EmbeddedArticle,
    EmbeddedArticleReference,
    NewsGroup,
    cluster_articles,
)
from romanian_news.subject_assessments import (
    DailySubjectAssessmentSet,
    compare_subject_anchors,
    resolve_group_anchor,
)
from romanian_news.themes import (
    AliasedReaderSubjectThemeSet,
    DailyThemeInput,
    DailyThemeSet,
    ReaderSubjectDailyThemeSet,
    SparseDailyThemeSet,
)

PIN_PATH = Path(__file__).with_name("evaluation_pin.json")

FORBIDDEN_TITLE_LABELS = frozenset({"breaking", "live", "update", "video"})
TITLE_MIN_WORDS = 4
TITLE_MAX_WORDS = 12
TITLE_MAX_CHARACTERS = 100
SUMMARY_MAX_WORDS = 60
SUMMARY_MAX_CHARACTERS = 500
SUMMARY_MAX_SENTENCES = 2
SUMMARY_TITLE_ABBREVIATIONS = frozenset({"dl", "dna", "dr", "mr", "mrs", "ms", "prof"})
KEY_POINT_MAX_COUNT = 3
KEY_POINT_MAX_CHARACTERS = 240
RELEVANCE_LIMITATION = (
    "Relevance generation is not deterministic. This run evaluates the current acceptance "
    "policy on frozen typed model responses."
)


class EvaluationProvenance(NewsModel):
    feedback_ids: tuple[UUID, ...]
    report: ArtifactReference | None = None
    group_id: Sha256 | None = None
    articles: tuple[ArtifactReference, ...] = ()
    model_outputs: tuple[ArtifactReference, ...] = ()


class RelevanceV3EvaluationDecision(NewsModel):
    context: ContextDecision
    impact: ImpactDecision | None


class RelevanceEvaluationCase(NewsModel):
    concern: Literal["relevance"] = "relevance"
    case_id: str
    control: bool = False
    provenance: EvaluationProvenance
    expected_accepted: bool
    model_response: RelevanceDecision | RelevanceV3EvaluationDecision


class FrozenGeneratedSummary(NewsModel):
    title: str
    summary: str
    key_points: tuple[str, ...]


class SummaryFormatEvaluationCase(NewsModel):
    concern: Literal["summary_format"] = "summary_format"
    case_id: str
    control: bool = False
    provenance: EvaluationProvenance
    artifact: FrozenGeneratedSummary


class GroupingEvaluationCase(NewsModel):
    concern: Literal["grouping"] = "grouping"
    case_id: str
    control: bool = False
    provenance: EvaluationProvenance
    source_cluster_set: ArtifactReference
    day: date
    articles: tuple[EmbeddedArticle, ...]
    left_article_version_id: Sha256
    right_article_version_id: Sha256
    expected_same_group: bool

    @model_validator(mode="after")
    def require_labeled_articles(self) -> GroupingEvaluationCase:
        article_ids = {article.article.version_id for article in self.articles}
        if self.left_article_version_id not in article_ids:
            raise ValueError("Grouping left article must be present in frozen inputs")
        if self.right_article_version_id not in article_ids:
            raise ValueError("Grouping right article must be present in frozen inputs")
        if self.left_article_version_id == self.right_article_version_id:
            raise ValueError("Grouping labels require two distinct articles")
        return self


class ArticleRelevanceInput(NewsModel):
    article_version_id: Sha256
    decision: RelevanceDecision | ImpactDecision


class RankingEvaluationCase(NewsModel):
    concern: Literal["ranking"] = "ranking"
    case_id: str
    control: bool = False
    provenance: EvaluationProvenance
    groups: tuple[NewsGroup, ...]
    relevance: tuple[ArticleRelevanceInput, ...]
    higher_group_id: Sha256
    lower_group_id: Sha256
    observed_group_order: tuple[Sha256, ...] = ()

    @model_validator(mode="after")
    def require_ranked_groups_and_articles(self) -> RankingEvaluationCase:
        group_ids = {group.id for group in self.groups}
        if {self.higher_group_id, self.lower_group_id} - group_ids:
            raise ValueError("Ranking labels must reference frozen groups")
        article_ids = {item.article_version_id for item in self.relevance}
        required_ids = {
            article_id for group in self.groups for article_id in group.article_version_ids
        }
        if article_ids != required_ids:
            raise ValueError("Ranking relevance inputs must exactly cover frozen group articles")
        if self.observed_group_order and (
            len(self.observed_group_order) != len(set(self.observed_group_order))
            or {self.higher_group_id, self.lower_group_id} - set(self.observed_group_order)
        ):
            raise ValueError("Ranking observation must contain both anchors exactly once")
        return self


Tier = Literal["main", "worth_knowing", "excluded"]


class TierEvaluationCase(NewsModel):
    concern: Literal["tier"] = "tier"
    case_id: str
    control: bool = False
    provenance: EvaluationProvenance
    expected_tier: Tier
    observed_tier: Tier


class ConfidenceEvaluationCase(NewsModel):
    concern: Literal["confidence"] = "confidence"
    case_id: str
    control: bool = False
    provenance: EvaluationProvenance
    expected_sufficient: bool
    observed_sufficient: bool


class KeyPointTextExpectation(NewsModel):
    kind: Literal["key_point_text"] = "key_point_text"
    observed_rem: Annotated[float, Field(gt=0)]
    minimum_rem: Annotated[float, Field(gt=0)]


class SingletonDifferencesExpectation(NewsModel):
    kind: Literal["singleton_differences"] = "singleton_differences"
    distinct_source_count: Annotated[int, Field(ge=0)]
    differences: tuple[str, ...]


class ThemePairExpectation(NewsModel):
    case_id: Annotated[str, Field(min_length=1)]
    feedback_ids: tuple[UUID, ...]
    left_group_id: Sha256
    right_group_id: Sha256
    expected_same_theme: bool
    control: bool = False
    rationale: ScoreCurationRationale

    @model_validator(mode="after")
    def require_distinct_groups(self) -> ThemePairExpectation:
        if self.left_group_id == self.right_group_id:
            raise ValueError("Theme expectations require two distinct groups")
        return self


class ThemeReportExpectation(NewsModel):
    minimum_must_link_recall: Annotated[float, Field(ge=0, le=1)] = 1.0
    minimum_must_separate_preservation: Annotated[float, Field(ge=0, le=1)] = 1.0
    require_non_singleton: bool = True


class ThemeEvaluationDayCase(NewsModel):
    concern: Literal["daily_theme"] = "daily_theme"
    case_id: str
    control: bool = False
    provenance: EvaluationProvenance
    source_report: ArtifactReference
    input: DailyThemeInput
    expectations: Annotated[tuple[ThemePairExpectation, ...], Field(min_length=1)]
    report_expectation: ThemeReportExpectation | None = None
    model_output: ArtifactReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    observed_themes: (
        DailyThemeSet
        | SparseDailyThemeSet
        | ReaderSubjectDailyThemeSet
        | AliasedReaderSubjectThemeSet
        | None
    ) = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def require_complete_observation(self) -> ThemeEvaluationDayCase:
        if (self.model_output is None) != (self.observed_themes is None):
            raise ValueError("Theme evaluation observation must be complete")
        return self

    @model_validator(mode="after")
    def require_frozen_groups(self) -> ThemeEvaluationDayCase:
        group_ids = {item.group.id for item in self.input.groups}
        for expectation in self.expectations:
            if {expectation.left_group_id, expectation.right_group_id} - group_ids:
                raise ValueError("Theme expectations must reference frozen daily groups")
        return self

    @model_validator(mode="after")
    def require_distinct_pair_labels(self) -> ThemeEvaluationDayCase:
        pairs = tuple(
            frozenset((expectation.left_group_id, expectation.right_group_id))
            for expectation in self.expectations
        )
        if len(set(pairs)) != len(pairs):
            raise ValueError("Daily-theme expectation pairs must be unique")
        if self.report_expectation is not None:
            outcomes = {item.expected_same_theme for item in self.expectations}
            if outcomes != {False, True}:
                raise ValueError(
                    "Report-level theme evaluation requires must-link and must-separate labels"
                )
        return self


class ThemeConstraintResult(NewsModel):
    case_id: str
    relation: Literal["must_link", "must_separate"]
    left_group_id: Sha256
    right_group_id: Sha256
    passed: bool


class ThemePolicyDayMetrics(NewsModel):
    must_link_passed: Annotated[int, Field(ge=0)]
    must_link_total: Annotated[int, Field(ge=0)]
    must_link_recall: Annotated[float, Field(ge=0, le=1)]
    must_separate_passed: Annotated[int, Field(ge=0)]
    must_separate_total: Annotated[int, Field(ge=0)]
    must_separate_preservation: Annotated[float, Field(ge=0, le=1)]
    singleton_groups: Annotated[int, Field(ge=0)]
    total_groups: Annotated[int, Field(ge=0)]
    singleton_rate: Annotated[float, Field(ge=0, le=1)]
    constraint_results: tuple[ThemeConstraintResult, ...]
    usefulness_failures: tuple[str, ...]
    useful: bool


def evaluate_theme_output(
    case: ThemeEvaluationDayCase,
    theme_set: SparseDailyThemeSet | ReaderSubjectDailyThemeSet,
) -> ThemePolicyDayMetrics:
    """Score reader-subject membership without changing exact source events."""
    expected_groups = _validate_theme_output_source(case, theme_set)
    theme_by_group = {
        group_id: theme.id for theme in theme_set.themes for group_id in theme.group_ids
    }
    constraint_results = tuple(
        _evaluate_theme_constraint(expectation, theme_by_group) for expectation in case.expectations
    )
    must_links = tuple(item for item in constraint_results if item.relation == "must_link")
    must_separates = tuple(item for item in constraint_results if item.relation == "must_separate")
    must_link_passed = sum(item.passed for item in must_links)
    must_separate_passed = sum(item.passed for item in must_separates)
    must_link_recall = must_link_passed / len(must_links) if must_links else 0.0
    must_separate_preservation = (
        must_separate_passed / len(must_separates) if must_separates else 0.0
    )
    singleton_groups = sum(
        len(theme.group_ids) for theme in theme_set.themes if len(theme.group_ids) == 1
    )
    singleton_rate = singleton_groups / len(expected_groups) if expected_groups else 0.0
    failures = _theme_usefulness_failures(
        case.report_expectation,
        constraint_results,
        must_link_recall,
        must_separate_preservation,
        singleton_rate,
    )
    return ThemePolicyDayMetrics(
        must_link_passed=must_link_passed,
        must_link_total=len(must_links),
        must_link_recall=must_link_recall,
        must_separate_passed=must_separate_passed,
        must_separate_total=len(must_separates),
        must_separate_preservation=must_separate_preservation,
        singleton_groups=singleton_groups,
        total_groups=len(expected_groups),
        singleton_rate=singleton_rate,
        constraint_results=constraint_results,
        usefulness_failures=failures,
        useful=case.report_expectation is not None and not failures,
    )


def _validate_theme_output_source(
    case: ThemeEvaluationDayCase,
    theme_set: SparseDailyThemeSet | ReaderSubjectDailyThemeSet,
) -> tuple[NewsGroup, ...]:
    if theme_set.day != case.input.day:
        raise ValueError("Evaluated themes do not match the frozen day")
    if theme_set.cluster_set != case.input.cluster_set:
        raise ValueError("Evaluated themes do not match the frozen cluster set")
    expected_groups = tuple(item.group for item in case.input.groups)
    if theme_set.groups != expected_groups:
        raise ValueError("Evaluated themes do not match the frozen groups")
    expected_summaries = tuple(item.summary for item in case.input.groups)
    if theme_set.summary_inputs != expected_summaries:
        raise ValueError("Evaluated themes do not match the frozen summaries")
    return expected_groups


def _evaluate_theme_constraint(
    expectation: ThemePairExpectation,
    theme_by_group: dict[Sha256, Sha256],
) -> ThemeConstraintResult:
    return ThemeConstraintResult(
        case_id=expectation.case_id,
        relation="must_link" if expectation.expected_same_theme else "must_separate",
        left_group_id=expectation.left_group_id,
        right_group_id=expectation.right_group_id,
        passed=(
            theme_by_group[expectation.left_group_id] == theme_by_group[expectation.right_group_id]
        )
        == expectation.expected_same_theme,
    )


def _theme_usefulness_failures(
    expectation: ThemeReportExpectation | None,
    constraint_results: tuple[ThemeConstraintResult, ...],
    must_link_recall: float,
    must_separate_preservation: float,
    singleton_rate: float,
) -> tuple[str, ...]:
    failures: list[str] = []
    if expectation is None:
        return ()
    failures.extend(
        f"{item.relation} failed: {item.case_id}" for item in constraint_results if not item.passed
    )
    if must_link_recall < expectation.minimum_must_link_recall:
        failures.append("must-link recall is below the report requirement")
    if must_separate_preservation < expectation.minimum_must_separate_preservation:
        failures.append("must-separate preservation is below the report requirement")
    if expectation.require_non_singleton and singleton_rate == 1.0:
        failures.append("report has no shared reader subject")
    return tuple(failures)


ReaderExpectation = Annotated[
    KeyPointTextExpectation | SingletonDifferencesExpectation,
    Field(discriminator="kind"),
]


class ReaderPresentationEvaluationCase(NewsModel):
    concern: Literal["reader_presentation"] = "reader_presentation"
    case_id: str
    control: bool = False
    provenance: EvaluationProvenance
    expectation: ReaderExpectation


NewsEvaluationCase = Annotated[
    RelevanceEvaluationCase
    | SummaryFormatEvaluationCase
    | GroupingEvaluationCase
    | RankingEvaluationCase
    | TierEvaluationCase
    | ConfidenceEvaluationCase
    | ReaderPresentationEvaluationCase
    | ThemeEvaluationDayCase,
    Field(discriminator="concern"),
]


class ReportInputReferences(NewsModel):
    themes: tuple[ArtifactReference, ...] = ()
    assessments: tuple[ArtifactReference, ...] = ()
    cluster_set: tuple[ArtifactReference, ...]
    relevance: tuple[ArtifactReference, ...] = ()
    summary: tuple[ArtifactReference, ...]
    sentiment: tuple[ArtifactReference, ...]


class ReportGroupInputs(NewsModel):
    group_id: Sha256
    summary: ArtifactReference
    sentiment: ArtifactReference


class ReportGroupObservation(NewsModel):
    group_id: Sha256
    uncertainty_disclosed: bool
    tier: Tier = "main"


class ReportEvaluationSnapshot(NewsModel):
    report: ArtifactReference
    themes: ArtifactReference | None = None
    assessments: ArtifactReference | None = None
    cluster_set: ArtifactReference
    report_article_version_ids: tuple[Sha256, ...]
    report_group_ids: tuple[Sha256, ...]
    cluster_articles: tuple[EmbeddedArticleReference, ...]
    group_inputs: tuple[ReportGroupInputs, ...]
    report_inputs: ReportInputReferences
    group_observations: tuple[ReportGroupObservation, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    report_theme_ids: tuple[Sha256, ...] = Field(default=(), exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def require_valid_runtime_observations(self) -> ReportEvaluationSnapshot:
        observation_ids = tuple(item.group_id for item in self.group_observations)
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("Report group observations must be unique")
        if observation_ids and observation_ids != self.report_group_ids:
            raise ValueError("Report group observations must match report group order")
        if self.report_theme_ids and len(self.report_theme_ids) != len(self.report_group_ids):
            raise ValueError("Report theme positions must match report group order")
        return self


class DailyThemeJudgment(NewsModel):
    judgment_id: Annotated[str, Field(min_length=1)]
    feedback_ids: Annotated[tuple[UUID, ...], Field(min_length=1)]
    report_version_id: Sha256
    left_group_id: Sha256
    right_group_id: Sha256
    expected_same_theme: bool
    rationale: Annotated[str, Field(min_length=1)]

    @field_validator("rationale", mode="before")
    @classmethod
    def normalize_rationale(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Theme judgments require a rationale")
        return normalized

    @model_validator(mode="after")
    def require_distinct_groups(self) -> DailyThemeJudgment:
        if self.left_group_id == self.right_group_id:
            raise ValueError("Theme judgments require two distinct groups")
        return self


class ExcludedFeedback(NewsModel):
    feedback_id: UUID
    report_version_id: Sha256
    reason: Literal["superseded"]
    superseded_by_feedback_id: UUID


class NonExecutableFeedback(NewsModel):
    feedback_id: UUID
    report_version_id: Sha256
    reason: Literal["non_executable"]
    concern: (
        Literal["language", "presentation", "operations", "ambiguous", "contradictory"] | None
    ) = None
    rationale: Annotated[str, Field(min_length=1)] | None = None


FeedbackExclusion = Annotated[
    ExcludedFeedback | NonExecutableFeedback,
    Field(discriminator="reason"),
]


ScoreCurationRationale = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ExecutableConcernDisposition(NewsModel):
    kind: Literal["executable"] = "executable"
    concern: Literal["ranking", "grouping"]
    case_ids: Annotated[tuple[str, ...], Field(min_length=1)]
    rationale: ScoreCurationRationale

    @model_validator(mode="after")
    def require_unique_case_ids(self) -> ExecutableConcernDisposition:
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("Executable concern case IDs must be unique")
        return self


class RelevanceConcernDisposition(NewsModel):
    kind: Literal["represented"] = "represented"
    concern: Literal["relevance"] = "relevance"
    judgment: Literal["relevant", "irrelevant"]
    case_ids: Annotated[tuple[str, ...], Field(min_length=1)]
    rationale: ScoreCurationRationale

    @model_validator(mode="after")
    def require_unique_case_ids(self) -> RelevanceConcernDisposition:
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("Relevance concern case IDs must be unique")
        return self


class TierConcernDisposition(NewsModel):
    kind: Literal["represented"] = "represented"
    concern: Literal["tier"] = "tier"
    judgment: Literal["main", "worth_knowing", "excluded"]
    case_ids: Annotated[tuple[str, ...], Field(min_length=1)]
    rationale: ScoreCurationRationale

    @model_validator(mode="after")
    def require_unique_case_ids(self) -> TierConcernDisposition:
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("Tier concern case IDs must be unique")
        return self


class ConfidenceConcernDisposition(NewsModel):
    kind: Literal["represented"] = "represented"
    concern: Literal["confidence"] = "confidence"
    judgment: Literal["supported", "insufficient"]
    case_ids: Annotated[tuple[str, ...], Field(min_length=1)]
    rationale: ScoreCurationRationale

    @model_validator(mode="after")
    def require_unique_case_ids(self) -> ConfidenceConcernDisposition:
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("Confidence concern case IDs must be unique")
        return self


class ContextConcernDisposition(NewsModel):
    kind: Literal["represented"] = "represented"
    concern: Literal["context"] = "context"
    judgment: Literal["sufficient", "needed"]
    rationale: ScoreCurationRationale


class OverallReportConcernDisposition(NewsModel):
    kind: Literal["represented"] = "represented"
    concern: Literal["overall_report"] = "overall_report"
    judgment: Literal["useful", "needs_tiering"]
    rationale: ScoreCurationRationale


class ExploratoryConcernDisposition(NewsModel):
    kind: Literal["exploratory"] = "exploratory"
    concern: Literal["research"] = "research"
    topics: Annotated[
        tuple[
            Literal[
                "coverage_review",
                "entity_context",
                "external_corroboration",
                "financial_data",
                "historical_data",
                "investigative_report",
            ],
            ...,
        ],
        Field(min_length=1),
    ]
    rationale: ScoreCurationRationale

    @model_validator(mode="after")
    def require_unique_topics(self) -> ExploratoryConcernDisposition:
        if len(set(self.topics)) != len(self.topics):
            raise ValueError("Exploratory research topics must be unique")
        return self


class ExcludedConcernDisposition(NewsModel):
    kind: Literal["excluded"] = "excluded"
    concern: Literal[
        "relevance",
        "ranking",
        "grouping",
        "tier",
        "confidence",
        "context",
        "research",
        "overall_report",
    ]
    reason: Literal["ambiguous", "conditional", "unsupported"]
    rationale: ScoreCurationRationale


class SupersededConcernDisposition(NewsModel):
    kind: Literal["superseded"] = "superseded"
    concern: Literal["submission"] = "submission"
    superseded_by_feedback_id: UUID


RepresentedConcernDisposition = Annotated[
    RelevanceConcernDisposition
    | TierConcernDisposition
    | ConfidenceConcernDisposition
    | ContextConcernDisposition
    | OverallReportConcernDisposition,
    Field(discriminator="concern"),
]

FeedbackConcernDisposition = Annotated[
    ExecutableConcernDisposition
    | RepresentedConcernDisposition
    | ExploratoryConcernDisposition
    | ExcludedConcernDisposition
    | SupersededConcernDisposition,
    Field(discriminator="kind"),
]


class FeedbackReview(NewsModel):
    feedback_id: UUID
    target: NewsFeedbackTarget
    concerns: Annotated[tuple[FeedbackConcernDisposition, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_unique_concerns(self) -> FeedbackReview:
        concerns = tuple(item.concern for item in self.concerns)
        if len(set(concerns)) != len(concerns):
            raise ValueError("Feedback review concerns must be unique")
        superseded = tuple(item for item in self.concerns if item.kind == "superseded")
        if superseded and len(self.concerns) != 1:
            raise ValueError("Superseded feedback cannot retain other concern dispositions")
        return self


class UnreviewedTheme(NewsModel):
    subject_position: Annotated[int, Field(ge=1)]
    target: ThemeFeedbackTarget
    reason: Literal["no_feedback"] = "no_feedback"


class ProjectedFeedbackScore(NewsModel):
    decision: Literal["projected"] = "projected"
    feedback_id: UUID
    concern: Literal["language", "relevance", "grouping", "ranking"]
    report_version_id: Sha256
    model_output: ArtifactReference
    model_attempt_id: Sha256
    polarity: Literal["positive", "negative"]
    rationale: ScoreCurationRationale


class ExcludedFeedbackScore(NewsModel):
    decision: Literal["excluded"] = "excluded"
    feedback_id: UUID
    reason: Literal[
        "presentation",
        "operations",
        "ambiguous",
        "contradictory",
        "superseded",
        "no_model_observation",
        "no_concern_specific_observation",
    ]
    concern: Literal["language", "grouping", "ranking"] | None = None
    report_version_id: Sha256
    polarity: Literal["positive", "negative"] | None = None
    rationale: ScoreCurationRationale

    @model_validator(mode="after")
    def require_concern_for_unobserved_model_score(self) -> ExcludedFeedbackScore:
        if self.reason == "no_model_observation":
            if self.concern not in {"grouping", "ranking"}:
                raise ValueError("No-model-observation exclusions require a deterministic concern")
            return self
        if self.reason == "no_concern_specific_observation":
            if self.concern not in {"language", "grouping"} or self.polarity is None:
                raise ValueError("Combined-observation exclusions require a concern and polarity")
            return self
        if self.concern is not None or self.polarity is not None:
            raise ValueError("Non-model exclusions cannot retain a concern or polarity")
        return self


FeedbackScoreCuration = Annotated[
    ProjectedFeedbackScore | ExcludedFeedbackScore,
    Field(discriminator="decision"),
]


@dataclass(frozen=True)
class FeedbackAssignments:
    source_ids: frozenset[UUID]
    represented_ids: frozenset[UUID]
    excluded_ids: frozenset[UUID]
    superseded_ids: frozenset[UUID]

    @classmethod
    def from_feedback_ids(
        cls,
        source_feedback_ids: tuple[UUID, ...],
        represented_feedback_ids: Iterable[UUID],
        excluded_feedback: tuple[FeedbackExclusion, ...],
    ) -> FeedbackAssignments:
        return cls(
            source_ids=frozenset(source_feedback_ids),
            represented_ids=frozenset(represented_feedback_ids),
            excluded_ids=frozenset(item.feedback_id for item in excluded_feedback),
            superseded_ids=frozenset(
                item.feedback_id for item in excluded_feedback if item.reason == "superseded"
            ),
        )

    @classmethod
    def from_curation(
        cls,
        source_feedback_ids: tuple[UUID, ...],
        cases: tuple[NewsEvaluationCase, ...] | tuple[NewsEvaluationSpec, ...],
        archived_daily_theme_judgments: tuple[DailyThemeJudgment, ...],
        excluded_feedback: tuple[FeedbackExclusion, ...],
    ) -> FeedbackAssignments:
        represented_feedback_ids = (
            feedback_id for case in cases for feedback_id in case.provenance.feedback_ids
        )
        theme_feedback_ids = (
            feedback_id
            for judgment in archived_daily_theme_judgments
            for feedback_id in judgment.feedback_ids
        )
        return cls.from_feedback_ids(
            source_feedback_ids,
            (*represented_feedback_ids, *theme_feedback_ids),
            excluded_feedback,
        )


def _validate_legacy_feedback_assignments(
    assignments: FeedbackAssignments,
    source_feedback_ids: tuple[UUID, ...],
    excluded_feedback: tuple[FeedbackExclusion, ...],
    coverage_message: str,
) -> None:
    if len(assignments.source_ids) != len(source_feedback_ids):
        raise ValueError("Evaluation feedback IDs must be unique")
    if assignments.represented_ids & assignments.excluded_ids:
        raise ValueError("Excluded feedback cannot also be represented")
    if assignments.represented_ids | assignments.excluded_ids != assignments.source_ids:
        raise ValueError(coverage_message)
    if len(assignments.excluded_ids) != len(excluded_feedback):
        raise ValueError("Excluded feedback IDs must be unique")


class EvaluationSpecProvenance(NewsModel):
    feedback_ids: tuple[UUID, ...]
    report_version_id: Sha256 | None = None
    group_id: Sha256 | None = None

    @model_validator(mode="after")
    def require_report_for_group(self) -> EvaluationSpecProvenance:
        if self.group_id is not None and self.report_version_id is None:
            raise ValueError("Evaluation group references require a frozen report")
        return self


class RelevanceEvaluationSpec(NewsModel):
    concern: Literal["relevance"] = "relevance"
    case_id: str
    control: bool = False
    provenance: EvaluationSpecProvenance
    expected_accepted: bool
    article: ArtifactReference
    model_output: ArtifactReference


class SummaryFormatEvaluationSpec(NewsModel):
    concern: Literal["summary_format"] = "summary_format"
    case_id: str
    control: bool = False
    provenance: EvaluationSpecProvenance
    articles: tuple[ArtifactReference, ...] = ()
    model_output: ArtifactReference


class GroupingEvaluationSpec(NewsModel):
    concern: Literal["grouping"] = "grouping"
    case_id: str
    control: bool = False
    provenance: EvaluationSpecProvenance
    source_cluster_set: ArtifactReference
    day: date
    articles: tuple[EmbeddedArticleReference, ...]
    left_article_version_id: Sha256
    right_article_version_id: Sha256
    expected_same_group: bool

    @model_validator(mode="after")
    def require_labeled_articles(self) -> GroupingEvaluationSpec:
        article_ids = {article.article.version_id for article in self.articles}
        if self.left_article_version_id not in article_ids:
            raise ValueError("Grouping left article must be present in frozen inputs")
        if self.right_article_version_id not in article_ids:
            raise ValueError("Grouping right article must be present in frozen inputs")
        if self.left_article_version_id == self.right_article_version_id:
            raise ValueError("Grouping labels require two distinct articles")
        return self


class ArticleRelevanceSpec(NewsModel):
    article: ArtifactReference
    model_output: ArtifactReference


class RankingEvaluationSpec(NewsModel):
    concern: Literal["ranking"] = "ranking"
    case_id: str
    control: bool = False
    provenance: EvaluationSpecProvenance
    relevance: tuple[ArticleRelevanceSpec, ...]
    higher_group_id: Sha256
    lower_group_id: Sha256


class TierEvaluationSpec(NewsModel):
    concern: Literal["tier"] = "tier"
    case_id: str
    control: bool = False
    provenance: EvaluationSpecProvenance
    expected_tier: Tier


class ConfidenceEvaluationSpec(NewsModel):
    concern: Literal["confidence"] = "confidence"
    case_id: str
    control: bool = False
    provenance: EvaluationSpecProvenance
    expected_sufficient: bool


class ReaderPresentationEvaluationSpec(NewsModel):
    concern: Literal["reader_presentation"] = "reader_presentation"
    case_id: str
    control: bool = False
    provenance: EvaluationSpecProvenance
    expectation: ReaderExpectation


class ThemeEvaluationDaySpec(NewsModel):
    concern: Literal["daily_theme"] = "daily_theme"
    case_id: str
    control: bool = False
    provenance: EvaluationSpecProvenance
    day: date
    source_report: ArtifactReference
    cluster_set: ArtifactReference
    summaries: tuple[ArtifactReference, ...]
    expectations: Annotated[tuple[ThemePairExpectation, ...], Field(min_length=1)]
    report_expectation: ThemeReportExpectation | None = None
    model_output: ArtifactReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


NewsEvaluationSpec = Annotated[
    RelevanceEvaluationSpec
    | SummaryFormatEvaluationSpec
    | GroupingEvaluationSpec
    | RankingEvaluationSpec
    | TierEvaluationSpec
    | ConfidenceEvaluationSpec
    | ReaderPresentationEvaluationSpec
    | ThemeEvaluationDaySpec,
    Field(discriminator="concern"),
]


def _evaluation_case_ids(
    cases: Iterable[NewsEvaluationCase | NewsEvaluationSpec],
) -> tuple[str, ...]:
    return tuple(
        case_id
        for case in cases
        for case_id in (
            (case.case_id, *(expectation.case_id for expectation in case.expectations))
            if case.concern == "daily_theme"
            else (case.case_id,)
        )
    )


@dataclass(frozen=True)
class _CaseFeedbackLink:
    concern: str
    feedback_ids: frozenset[UUID]
    report_version_id: Sha256 | None
    expected_judgment: object
    group_ids: tuple[Sha256, ...]


def _case_feedback_links(
    cases: Iterable[NewsEvaluationCase | NewsEvaluationSpec],
) -> dict[str, _CaseFeedbackLink]:
    links: dict[str, _CaseFeedbackLink] = {}
    for case in cases:
        if len(set(case.provenance.feedback_ids)) != len(case.provenance.feedback_ids):
            raise ValueError("Evaluation case provenance feedback IDs must be unique")
        report_version_id = (
            case.provenance.report.version_id
            if isinstance(case.provenance, EvaluationProvenance)
            and case.provenance.report is not None
            else case.provenance.report_version_id
            if isinstance(case.provenance, EvaluationSpecProvenance)
            else None
        )
        if case.concern == "daily_theme":
            for expectation in case.expectations:
                if len(set(expectation.feedback_ids)) != len(expectation.feedback_ids):
                    raise ValueError("Theme expectation feedback IDs must be unique")
                if expectation.feedback_ids:
                    links[expectation.case_id] = _CaseFeedbackLink(
                        concern="grouping",
                        feedback_ids=frozenset(expectation.feedback_ids),
                        report_version_id=report_version_id,
                        expected_judgment=expectation.expected_same_theme,
                        group_ids=(expectation.left_group_id, expectation.right_group_id),
                    )
            if frozenset(case.provenance.feedback_ids) != frozenset(
                feedback_id
                for expectation in case.expectations
                for feedback_id in expectation.feedback_ids
            ):
                raise ValueError("Daily-theme provenance must match its expectation feedback")
        elif case.provenance.feedback_ids:
            links[case.case_id] = _CaseFeedbackLink(
                concern=case.concern,
                feedback_ids=frozenset(case.provenance.feedback_ids),
                report_version_id=report_version_id,
                expected_judgment=_case_expected_judgment(case),
                group_ids=_case_target_group_ids(case),
            )
    return links


def _case_expected_judgment(case: NewsEvaluationCase | NewsEvaluationSpec) -> object:
    if case.concern == "relevance":
        return case.expected_accepted
    if case.concern == "tier":
        return case.expected_tier
    if case.concern == "confidence":
        return case.expected_sufficient
    if case.concern == "grouping":
        return case.expected_same_group
    return None


def _case_target_group_ids(
    case: NewsEvaluationCase | NewsEvaluationSpec,
) -> tuple[Sha256, ...]:
    if case.concern == "ranking":
        return (case.higher_group_id,)
    if case.concern in {"relevance", "grouping", "tier", "confidence"}:
        return (case.provenance.group_id,) if case.provenance.group_id is not None else ()
    return ()


def _validate_feedback_reviews(
    source_feedback_ids: tuple[UUID, ...],
    cases: Iterable[NewsEvaluationCase | NewsEvaluationSpec],
    feedback_reviews: tuple[FeedbackReview, ...],
    unreviewed_themes: tuple[UnreviewedTheme, ...],
    report_version_ids: frozenset[Sha256],
) -> None:
    reviews = _require_exact_feedback_review_coverage(source_feedback_ids, feedback_reviews)
    active_targets = _require_valid_supersession_and_targets(
        feedback_reviews, reviews, report_version_ids
    )
    expected_links = _declared_executable_links(feedback_reviews)
    actual_links = _require_executable_links_match(cases, expected_links)
    _require_linked_case_reports_match(actual_links, expected_links)
    _require_represented_judgments_match(feedback_reviews, actual_links)
    _require_valid_unreviewed_themes(unreviewed_themes, active_targets, report_version_ids)


def _require_exact_feedback_review_coverage(
    source_feedback_ids: tuple[UUID, ...],
    feedback_reviews: tuple[FeedbackReview, ...],
) -> dict[UUID, FeedbackReview]:
    if len(set(source_feedback_ids)) != len(source_feedback_ids):
        raise ValueError("Evaluation feedback IDs must be unique")
    reviews = {review.feedback_id: review for review in feedback_reviews}
    if len(reviews) != len(feedback_reviews):
        raise ValueError("Feedback review IDs must be unique")
    if set(reviews) != set(source_feedback_ids):
        raise ValueError("Feedback reviews must exactly cover every declared feedback ID")
    return reviews


def _require_valid_supersession_and_targets(
    feedback_reviews: tuple[FeedbackReview, ...],
    reviews: dict[UUID, FeedbackReview],
    report_version_ids: frozenset[Sha256],
) -> frozenset[str]:
    active_targets: list[str] = []
    superseded_ids: set[UUID] = set()
    replacement_ids: set[UUID] = set()
    for review in feedback_reviews:
        if review.target.report_version_id not in report_version_ids:
            raise ValueError("Feedback reviews must reference a frozen report")
        disposition = review.concerns[0]
        if not isinstance(disposition, SupersededConcernDisposition):
            active_targets.append(review.target.model_dump_json())
            continue
        superseded_ids.add(review.feedback_id)
        replacement_ids.add(disposition.superseded_by_feedback_id)
        replacement = reviews.get(disposition.superseded_by_feedback_id)
        if replacement is None:
            raise ValueError("Superseding feedback must be declared")
        if replacement.target != review.target:
            raise ValueError("Superseding feedback must keep the exact target")

    if len(set(active_targets)) != len(active_targets):
        raise ValueError("Current feedback reviews must have unique exact targets")
    if replacement_ids & superseded_ids:
        raise ValueError("Superseding feedback cannot itself be superseded")
    return frozenset(active_targets)


def _declared_executable_links(
    feedback_reviews: tuple[FeedbackReview, ...],
) -> dict[str, tuple[str, set[UUID], set[Sha256]]]:
    expected_links: dict[str, tuple[str, set[UUID], set[Sha256]]] = {}
    for review in feedback_reviews:
        if isinstance(review.concerns[0], SupersededConcernDisposition):
            continue
        for disposition in review.concerns:
            if not isinstance(
                disposition,
                ExecutableConcernDisposition
                | RelevanceConcernDisposition
                | TierConcernDisposition
                | ConfidenceConcernDisposition,
            ):
                continue
            for case_id in disposition.case_ids:
                concern, feedback_ids, target_reports = expected_links.setdefault(
                    case_id,
                    (disposition.concern, set(), set()),
                )
                if concern != disposition.concern:
                    raise ValueError("One evaluation case cannot represent several concerns")
                feedback_ids.add(review.feedback_id)
                target_reports.add(review.target.report_version_id)
    return expected_links


def _require_executable_links_match(
    cases: Iterable[NewsEvaluationCase | NewsEvaluationSpec],
    expected_links: dict[str, tuple[str, set[UUID], set[Sha256]]],
) -> dict[str, _CaseFeedbackLink]:
    actual_links = _case_feedback_links(cases)
    normalized_expected = {
        case_id: (concern, frozenset(feedback_ids))
        for case_id, (concern, feedback_ids, _target_reports) in expected_links.items()
    }
    normalized_actual = {
        case_id: (link.concern, link.feedback_ids) for case_id, link in actual_links.items()
    }
    if normalized_actual != normalized_expected:
        raise ValueError("Executable concern links must match evaluation provenance both ways")
    return actual_links


def _require_linked_case_reports_match(
    actual_links: dict[str, _CaseFeedbackLink],
    expected_links: dict[str, tuple[str, set[UUID], set[Sha256]]],
) -> None:
    for case_id, link in actual_links.items():
        target_reports = expected_links[case_id][2]
        if link.report_version_id is None or target_reports != {link.report_version_id}:
            raise ValueError("Linked case and feedback target reports must match")


def _require_represented_judgments_match(
    feedback_reviews: tuple[FeedbackReview, ...],
    actual_links: dict[str, _CaseFeedbackLink],
) -> None:
    for review in feedback_reviews:
        for disposition in review.concerns:
            if isinstance(disposition, RelevanceConcernDisposition):
                expected_judgment: object = disposition.judgment == "relevant"
            elif isinstance(disposition, TierConcernDisposition):
                expected_judgment = disposition.judgment
            elif isinstance(disposition, ConfidenceConcernDisposition):
                expected_judgment = disposition.judgment == "supported"
            else:
                continue
            if any(
                actual_links[case_id].expected_judgment != expected_judgment
                for case_id in disposition.case_ids
            ):
                raise ValueError("Represented judgment must match its linked case")


def _require_valid_unreviewed_themes(
    unreviewed_themes: tuple[UnreviewedTheme, ...],
    active_targets: frozenset[str],
    report_version_ids: frozenset[Sha256],
) -> None:
    positions = tuple(item.subject_position for item in unreviewed_themes)
    targets = tuple(item.target.model_dump_json() for item in unreviewed_themes)
    if len(set(positions)) != len(positions) or len(set(targets)) != len(targets):
        raise ValueError("Unreviewed themes must be unique")
    if set(targets) & active_targets:
        raise ValueError("An unreviewed theme cannot have current feedback")
    if any(item.target.report_version_id not in report_version_ids for item in unreviewed_themes):
        raise ValueError("Unreviewed themes must reference a frozen report")


def _exclude_empty_feedback_reviews(reviews: tuple[FeedbackReview, ...]) -> bool:
    return not reviews


def _exclude_empty_unreviewed_themes(themes: tuple[UnreviewedTheme, ...]) -> bool:
    return not themes


class ReportEvaluationSpec(NewsModel):
    report: ArtifactReference
    themes: ArtifactReference | None = None
    cluster_set: ArtifactReference


class NewsEvaluationManifest(NewsModel):
    version: str
    reviewed_at: datetime
    issue_url: str
    prior_manifest: ArtifactReference | None = None
    source_feedback_ids: tuple[UUID, ...]
    reports: tuple[ReportEvaluationSpec, ...]
    cases: tuple[NewsEvaluationSpec, ...]
    archived_daily_theme_judgments: tuple[DailyThemeJudgment, ...] = Field(
        default=(),
        validation_alias=AliasChoices("daily_theme_judgments", "archived_daily_theme_judgments"),
        exclude=True,
    )
    excluded_feedback: tuple[FeedbackExclusion, ...] = ()
    feedback_reviews: tuple[FeedbackReview, ...] = Field(
        default=(), exclude_if=_exclude_empty_feedback_reviews
    )
    unreviewed_themes: tuple[UnreviewedTheme, ...] = Field(
        default=(), exclude_if=_exclude_empty_unreviewed_themes
    )
    score_curation: tuple[FeedbackScoreCuration, ...] = Field(
        default=(), exclude_if=lambda decisions: not decisions
    )

    @model_validator(mode="after")
    def require_valid_curation(self) -> NewsEvaluationManifest:
        case_ids = _evaluation_case_ids(self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Evaluation case IDs must be unique")
        reports = {item.report.version_id: item for item in self.reports}
        self._validate_feedback_curation(reports)
        if len(reports) != len(self.reports):
            raise ValueError("Evaluation report versions must be unique")
        self._validate_report_references(reports)
        self._validate_theme_judgments(reports)
        self._validate_exclusions(reports)
        self._validate_score_curation(reports)
        return self

    def _validate_feedback_curation(self, reports: dict[Sha256, ReportEvaluationSpec]) -> None:
        if self.feedback_reviews:
            if self.archived_daily_theme_judgments or self.excluded_feedback or self.score_curation:
                raise ValueError("Reviewed feedback cannot use legacy curation fields")
            _validate_feedback_reviews(
                self.source_feedback_ids,
                self.cases,
                self.feedback_reviews,
                self.unreviewed_themes,
                frozenset(reports),
            )
            return
        if self.unreviewed_themes:
            raise ValueError("Legacy curation cannot declare unreviewed themes")
        assignments = FeedbackAssignments.from_curation(
            self.source_feedback_ids,
            self.cases,
            self.archived_daily_theme_judgments,
            self.excluded_feedback,
        )
        _validate_legacy_feedback_assignments(
            assignments,
            self.source_feedback_ids,
            self.excluded_feedback,
            "Evaluation specs, theme judgments, and exclusions must exactly cover every declared feedback ID",
        )

    def _validate_report_references(self, reports: dict[Sha256, ReportEvaluationSpec]) -> None:
        for case in self.cases:
            report_version_id = case.provenance.report_version_id
            if report_version_id is not None and report_version_id not in reports:
                raise ValueError("Evaluation specs must reference a frozen report")

    def _validate_exclusions(self, reports: dict[Sha256, ReportEvaluationSpec]) -> None:
        superseded_ids = {
            item.feedback_id for item in self.excluded_feedback if item.reason == "superseded"
        }
        source_ids = set(self.source_feedback_ids)
        for exclusion in self.excluded_feedback:
            if exclusion.report_version_id not in reports:
                raise ValueError("Excluded feedback must reference a frozen report")
            if exclusion.reason == "superseded" and (
                exclusion.superseded_by_feedback_id not in source_ids
                or exclusion.superseded_by_feedback_id in superseded_ids
            ):
                raise ValueError(
                    "Superseding feedback must be a declared, non-superseded source ID"
                )

    def _validate_score_curation(self, reports: dict[Sha256, ReportEvaluationSpec]) -> None:
        if not self.score_curation:
            return
        curated_feedback_ids = {decision.feedback_id for decision in self.score_curation}
        if curated_feedback_ids != set(self.source_feedback_ids):
            raise ValueError(
                "Score projections and exclusions must exactly cover every declared feedback ID"
            )
        decision_keys: set[tuple[UUID, str, Sha256 | None]] = set()
        for decision in self.score_curation:
            if decision.report_version_id not in reports:
                raise ValueError("Score curation must reference a frozen report")
            if decision.decision == "projected":
                discriminator = decision.concern
                attempt_id = decision.model_attempt_id
            else:
                discriminator = f"{decision.reason}:{decision.concern or ''}"
                attempt_id = None
            key = (decision.feedback_id, discriminator, attempt_id)
            if key in decision_keys:
                raise ValueError("Score curation decisions must be unique")
            decision_keys.add(key)

    def _validate_theme_judgments(self, reports: dict[Sha256, ReportEvaluationSpec]) -> None:
        judgment_ids = {item.judgment_id for item in self.archived_daily_theme_judgments}
        if len(judgment_ids) != len(self.archived_daily_theme_judgments):
            raise ValueError("Theme judgment IDs must be unique")
        pairs: set[tuple[Sha256, Sha256, Sha256]] = set()
        for judgment in self.archived_daily_theme_judgments:
            if judgment.report_version_id not in reports:
                raise ValueError("Theme judgments must reference a frozen report")
            pair_left, pair_right = sorted((judgment.left_group_id, judgment.right_group_id))
            key = (judgment.report_version_id, pair_left, pair_right)
            if key in pairs:
                raise ValueError("Theme judgments cannot repeat a reversed group pair")
            pairs.add(key)


class NewsEvaluationDataset(NewsModel):
    version: str
    reviewed_at: datetime
    issue_url: str
    source_feedback_ids: tuple[UUID, ...]
    reports: tuple[ReportEvaluationSnapshot, ...]
    cases: tuple[NewsEvaluationCase, ...]
    archived_daily_theme_judgments: tuple[DailyThemeJudgment, ...] = Field(
        default=(),
        validation_alias=AliasChoices("daily_theme_judgments", "archived_daily_theme_judgments"),
        exclude=True,
    )
    excluded_feedback: tuple[FeedbackExclusion, ...] = ()
    feedback_reviews: tuple[FeedbackReview, ...] = Field(
        default=(), exclude_if=_exclude_empty_feedback_reviews
    )
    unreviewed_themes: tuple[UnreviewedTheme, ...] = Field(
        default=(), exclude_if=_exclude_empty_unreviewed_themes
    )

    @model_validator(mode="after")
    def require_valid_curation(self) -> NewsEvaluationDataset:
        case_ids = _evaluation_case_ids(self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Evaluation case IDs must be unique")
        snapshots = {snapshot.report.version_id: snapshot for snapshot in self.reports}
        self._validate_feedback_curation(snapshots)
        if len(snapshots) != len(self.reports):
            raise ValueError("Evaluation report versions must be unique")
        for case in self.cases:
            self._validate_case_references(case, snapshots)
        if self.feedback_reviews:
            self._validate_feedback_review_targets(snapshots)
        self._validate_theme_judgments(snapshots)
        self._validate_exclusions(snapshots)
        return self

    def _validate_feedback_curation(
        self, snapshots: dict[Sha256, ReportEvaluationSnapshot]
    ) -> None:
        if self.feedback_reviews:
            if self.archived_daily_theme_judgments or self.excluded_feedback:
                raise ValueError("Reviewed feedback cannot use legacy curation fields")
            _validate_feedback_reviews(
                self.source_feedback_ids,
                self.cases,
                self.feedback_reviews,
                self.unreviewed_themes,
                frozenset(snapshots),
            )
            self._validate_unreviewed_theme_positions(snapshots)
            return
        if self.unreviewed_themes:
            raise ValueError("Legacy curation cannot declare unreviewed themes")
        assignments = FeedbackAssignments.from_curation(
            self.source_feedback_ids,
            self.cases,
            self.archived_daily_theme_judgments,
            self.excluded_feedback,
        )
        _validate_legacy_feedback_assignments(
            assignments,
            self.source_feedback_ids,
            self.excluded_feedback,
            "Evaluation cases, theme judgments, and exclusions must exactly cover every declared feedback ID",
        )

    def _validate_exclusions(self, snapshots: dict[Sha256, ReportEvaluationSnapshot]) -> None:
        superseded_ids = {
            item.feedback_id for item in self.excluded_feedback if item.reason == "superseded"
        }
        source_ids = set(self.source_feedback_ids)
        for exclusion in self.excluded_feedback:
            if exclusion.report_version_id not in snapshots:
                raise ValueError("Excluded feedback must reference a frozen report")
            if exclusion.reason == "superseded" and (
                exclusion.superseded_by_feedback_id not in source_ids
                or exclusion.superseded_by_feedback_id in superseded_ids
            ):
                raise ValueError(
                    "Superseding feedback must be a declared, non-superseded source ID"
                )

    def _validate_unreviewed_theme_positions(
        self,
        snapshots: dict[Sha256, ReportEvaluationSnapshot],
    ) -> None:
        for unreviewed in self.unreviewed_themes:
            snapshot = snapshots[unreviewed.target.report_version_id]
            position = unreviewed.subject_position - 1
            if position >= len(snapshot.report_theme_ids):
                raise ValueError("Unreviewed theme position is outside the frozen report")
            if snapshot.report_theme_ids[position] != unreviewed.target.theme_id:
                raise ValueError("Unreviewed theme target does not match its report position")

    def _validate_feedback_review_targets(
        self,
        snapshots: dict[Sha256, ReportEvaluationSnapshot],
    ) -> None:
        links = _case_feedback_links(self.cases)
        for review in self.feedback_reviews:
            if not isinstance(review.target, ThemeFeedbackTarget):
                continue
            snapshot = snapshots[review.target.report_version_id]
            if len(snapshot.report_theme_ids) != len(snapshot.report_group_ids):
                raise ValueError("Theme feedback requires frozen report theme positions")
            target_group_ids = {
                group_id
                for group_id, theme_id in zip(
                    snapshot.report_group_ids,
                    snapshot.report_theme_ids,
                    strict=True,
                )
                if theme_id == review.target.theme_id
            }
            for disposition in review.concerns:
                if not isinstance(
                    disposition,
                    ExecutableConcernDisposition
                    | RelevanceConcernDisposition
                    | TierConcernDisposition
                    | ConfidenceConcernDisposition,
                ):
                    continue
                if any(
                    target_group_ids.isdisjoint(links[case_id].group_ids)
                    for case_id in disposition.case_ids
                ):
                    raise ValueError("A linked case must belong to its target theme")

    @staticmethod
    def _validate_case_references(
        case: NewsEvaluationCase,
        snapshots: dict[Sha256, ReportEvaluationSnapshot],
    ) -> None:
        report = case.provenance.report
        if report is None:
            if case.provenance.group_id is not None:
                raise ValueError("Evaluation group references require a frozen report")
            return
        snapshot = snapshots.get(report.version_id)
        if snapshot is None or snapshot.report != report:
            raise ValueError("Evaluation cases must reference a frozen report")
        if (
            case.provenance.group_id is not None
            and case.provenance.group_id not in snapshot.report_group_ids
        ):
            raise ValueError("Evaluation cases must reference a frozen report group")
        articles = {item.article.version_id: item for item in snapshot.cluster_articles}
        if any(
            articles.get(reference.version_id) is None
            or articles[reference.version_id].article != reference
            for reference in case.provenance.articles
        ):
            raise ValueError("Evaluation cases must reference frozen report articles")
        NewsEvaluationDataset._validate_case_concern(case, snapshot, articles)

    @staticmethod
    def _validate_case_concern(
        case: NewsEvaluationCase,
        snapshot: ReportEvaluationSnapshot,
        articles: dict[Sha256, EmbeddedArticleReference],
    ) -> None:
        if case.concern == "ranking" and any(
            group.id not in snapshot.report_group_ids for group in case.groups
        ):
            raise ValueError("Ranking cases must reference frozen report groups")
        if isinstance(case, TierEvaluationCase | ConfidenceEvaluationCase):
            NewsEvaluationDataset._validate_observation_case(case, snapshot)
        if (
            case.concern == "daily_theme"
            and case.model_output is not None
            and case.model_output != snapshot.themes
        ):
            raise ValueError("Theme evaluation must use the exact report theme output")
        if isinstance(case, GroupingEvaluationCase):
            NewsEvaluationDataset._validate_grouping_case(case, snapshot, articles)

    @staticmethod
    def _validate_observation_case(
        case: TierEvaluationCase | ConfidenceEvaluationCase,
        snapshot: ReportEvaluationSnapshot,
    ) -> None:
        observations = {item.group_id: item for item in snapshot.group_observations}
        group_id = case.provenance.group_id
        if group_id is None or group_id not in observations:
            raise ValueError("Report observation case requires its frozen report group")
        if case.concern == "tier" and case.observed_tier != observations[group_id].tier:
            raise ValueError("Tier observation differs from the frozen report")
        if case.concern == "confidence" and case.observed_sufficient != (
            not observations[group_id].uncertainty_disclosed
        ):
            raise ValueError("Confidence observation differs from the frozen report")

    @staticmethod
    def _validate_grouping_case(
        case: GroupingEvaluationCase,
        snapshot: ReportEvaluationSnapshot,
        articles: dict[Sha256, EmbeddedArticleReference],
    ) -> None:
        if case.source_cluster_set != snapshot.cluster_set:
            raise ValueError("Grouping cases must reference the report cluster set")
        if any(
            article.article.version_id not in articles
            or articles[article.article.version_id]
            != EmbeddedArticleReference(
                article=article.article,
                relevance=article.relevance,
                embedding=article.embedding,
            )
            for article in case.articles
        ):
            raise ValueError("Grouping cases must use frozen report article inputs")

    def _validate_theme_judgments(
        self,
        snapshots: dict[Sha256, ReportEvaluationSnapshot],
    ) -> None:
        judgment_ids = {item.judgment_id for item in self.archived_daily_theme_judgments}
        if len(judgment_ids) != len(self.archived_daily_theme_judgments):
            raise ValueError("Theme judgment IDs must be unique")
        pairs: set[tuple[Sha256, Sha256, Sha256]] = set()
        for judgment in self.archived_daily_theme_judgments:
            snapshot = snapshots.get(judgment.report_version_id)
            if snapshot is None:
                raise ValueError("Theme judgments must reference a frozen report")
            if {judgment.left_group_id, judgment.right_group_id} - set(snapshot.report_group_ids):
                raise ValueError("Theme judgments must reference frozen report groups")
            pair_left, pair_right = sorted((judgment.left_group_id, judgment.right_group_id))
            key = (judgment.report_version_id, pair_left, pair_right)
            if key in pairs:
                raise ValueError("Theme judgments cannot repeat a reversed group pair")
            pairs.add(key)


class EvaluationCaseResult(NewsModel):
    case_id: str
    concern: str
    passed: bool
    control: bool
    expected_positive: bool
    predicted_positive: bool | None
    detail: str

    @model_validator(mode="after")
    def require_valid_outcome(self) -> EvaluationCaseResult:
        if self.predicted_positive is None:
            if self.passed:
                raise ValueError("An unavailable prediction cannot pass")
            return self
        if self.passed != (self.predicted_positive == self.expected_positive):
            raise ValueError("Evaluation result must match its prediction")
        return self


class DeterministicCheckResult(NewsModel):
    check_id: str
    passed: bool
    detail: str


class EvaluationAggregate(NewsModel):
    concern: Literal["relevance", "grouping", "ranking", "tier", "confidence", "daily_theme"]
    passed_cases: Annotated[int, Field(ge=0)]
    total_cases: Annotated[int, Field(ge=0)]
    precision: Annotated[float, Field(ge=0, le=1)]
    recall: Annotated[float, Field(ge=0, le=1)]
    f1: Annotated[float, Field(ge=0, le=1)]
    positive_control_preservation: Annotated[float, Field(ge=0, le=1)]


class NewsEvaluationReport(NewsModel):
    dataset_version: str
    limitations: tuple[str, ...]
    case_results: tuple[EvaluationCaseResult, ...]
    deterministic_checks: tuple[DeterministicCheckResult, ...]
    aggregates: tuple[EvaluationAggregate, ...]


class BaselineCaseResult(NewsModel):
    case_id: str
    concern: str
    control: bool
    passed: bool


class BaselineDeterministicCheck(NewsModel):
    check_id: str
    passed: bool


class NewsEvaluationBaseline(NewsModel):
    manifest: ArtifactReference
    dataset_version: str

    cases: tuple[BaselineCaseResult, ...]
    deterministic_checks: tuple[BaselineDeterministicCheck, ...]
    aggregates: tuple[EvaluationAggregate, ...]

    @model_validator(mode="after")
    def require_unique_result_ids(self) -> NewsEvaluationBaseline:
        if len({item.case_id for item in self.cases}) != len(self.cases):
            raise ValueError("Baseline case IDs must be unique")
        if len({item.check_id for item in self.deterministic_checks}) != len(
            self.deterministic_checks
        ):
            raise ValueError("Baseline check IDs must be unique")
        if len({item.concern for item in self.aggregates}) != len(self.aggregates):
            raise ValueError("Baseline aggregate concerns must be unique")
        return self


class NewsEvaluationPin(NewsModel):
    manifest_version_id: Sha256
    baseline_version_id: Sha256


def evaluate_news_dataset(dataset: NewsEvaluationDataset) -> NewsEvaluationReport:
    """Run current deterministic policies against frozen production examples."""
    case_results = tuple(
        result for case in dataset.cases for result in _evaluate_case_results(case)
    )
    deterministic_checks = tuple(
        _evaluate_report_completeness(snapshot) for snapshot in dataset.reports
    )
    concerns = ["relevance", "grouping", "ranking"]
    concerns.extend(
        concern
        for concern in ("tier", "confidence")
        if any(result.concern == concern for result in case_results)
    )
    aggregates = tuple(_aggregate(concern, case_results) for concern in concerns)
    return NewsEvaluationReport(
        dataset_version=dataset.version,
        limitations=(RELEVANCE_LIMITATION,),
        case_results=case_results,
        deterministic_checks=deterministic_checks,
        aggregates=aggregates,
    )


def build_news_evaluation_baseline(
    report: NewsEvaluationReport,
    manifest: ArtifactReference,
) -> NewsEvaluationBaseline:
    """Capture the exact reviewed result identities and metric floors."""
    return NewsEvaluationBaseline(
        manifest=manifest,
        dataset_version=report.dataset_version,
        cases=tuple(
            BaselineCaseResult(
                case_id=result.case_id,
                concern=result.concern,
                control=result.control,
                passed=result.passed,
            )
            for result in report.case_results
        ),
        deterministic_checks=tuple(
            BaselineDeterministicCheck(check_id=result.check_id, passed=result.passed)
            for result in report.deterministic_checks
        ),
        aggregates=report.aggregates,
    )


def compare_news_evaluation_to_baseline(
    report: NewsEvaluationReport,
    baseline: NewsEvaluationBaseline,
    manifest: ArtifactReference,
) -> tuple[str, ...]:
    """Return each result or metric that regressed from the reviewed baseline."""
    regressions = []
    if manifest != baseline.manifest:
        regressions.append("manifest reference changed")
    if report.dataset_version != baseline.dataset_version:
        regressions.append(
            f"dataset version changed from {baseline.dataset_version} to {report.dataset_version}"
        )

    current_cases = {result.case_id: result for result in report.case_results}
    baseline_cases = {result.case_id: result for result in baseline.cases}
    regressions.extend(_identity_regressions("case", current_cases, baseline_cases))
    for case_id in sorted(current_cases.keys() & baseline_cases.keys()):
        current = current_cases[case_id]
        reviewed = baseline_cases[case_id]
        if (current.concern, current.control) != (reviewed.concern, reviewed.control):
            regressions.append(f"case metadata changed: {case_id}")
        if reviewed.passed and not current.passed:
            regressions.append(f"case regressed: {case_id}")

    current_checks = {result.check_id: result for result in report.deterministic_checks}
    baseline_checks = {result.check_id: result for result in baseline.deterministic_checks}
    regressions.extend(_identity_regressions("check", current_checks, baseline_checks))
    for check_id in sorted(current_checks.keys() & baseline_checks.keys()):
        if baseline_checks[check_id].passed and not current_checks[check_id].passed:
            regressions.append(f"check regressed: {check_id}")

    current_aggregates = {result.concern: result for result in report.aggregates}
    baseline_aggregates = {result.concern: result for result in baseline.aggregates}
    regressions.extend(_identity_regressions("aggregate", current_aggregates, baseline_aggregates))
    for concern in sorted(current_aggregates.keys() & baseline_aggregates.keys()):
        regressions.extend(
            _aggregate_regressions(current_aggregates[concern], baseline_aggregates[concern])
        )
    return tuple(regressions)


def _identity_regressions(
    result_kind: str, current: Mapping[str, object], baseline: Mapping[str, object]
) -> tuple[str, ...]:
    added = sorted(current.keys() - baseline.keys())
    removed = sorted(baseline.keys() - current.keys())
    if not added and not removed:
        return ()
    return (f"{result_kind} IDs changed: added={added}, removed={removed}",)


def _aggregate_regressions(
    current: EvaluationAggregate, baseline: EvaluationAggregate
) -> tuple[str, ...]:
    regressions = []
    if current.total_cases != baseline.total_cases:
        regressions.append(
            f"{current.concern} total_cases changed from "
            f"{baseline.total_cases} to {current.total_cases}"
        )
    for field in (
        "passed_cases",
        "precision",
        "recall",
        "f1",
        "positive_control_preservation",
    ):
        current_value = getattr(current, field)
        baseline_value = getattr(baseline, field)
        if current_value < baseline_value:
            regressions.append(
                f"{current.concern} {field} regressed from "
                f"{baseline_value:g} to {current_value:g}"
            )
    return tuple(regressions)


def _evaluate_case_results(case: NewsEvaluationCase) -> tuple[EvaluationCaseResult, ...]:
    if case.concern != "daily_theme":
        return (_evaluate_case(case),)
    if case.observed_themes is None:
        return ()
    theme_by_group = {
        group_id: theme.id for theme in case.observed_themes.themes for group_id in theme.group_ids
    }
    results = []
    for expectation in case.expectations:
        try:
            predicted = (
                theme_by_group[expectation.left_group_id]
                == theme_by_group[expectation.right_group_id]
            )
        except KeyError as error:
            raise ValueError(
                f"Theme evaluation output omits a reviewed group: {expectation.case_id}"
            ) from error
        results.append(
            EvaluationCaseResult(
                case_id=expectation.case_id,
                concern="grouping",
                passed=predicted == expectation.expected_same_theme,
                control=expectation.control,
                expected_positive=expectation.expected_same_theme,
                predicted_positive=predicted,
                detail=(
                    f"expected same_theme={expectation.expected_same_theme}, "
                    f"frozen output same_theme={predicted}"
                ),
            )
        )
    return tuple(results)


def _evaluate_case(case: NewsEvaluationCase) -> EvaluationCaseResult:
    if case.concern == "relevance":
        expected = case.expected_accepted
        predicted = (
            relevance_is_accepted(case.model_response, PRODUCTION_RELEVANCE_POLICY)
            if isinstance(case.model_response, RelevanceDecision)
            else case.model_response.impact is not None
            and relevance_v3_is_accepted(
                case.model_response.context,
                case.model_response.impact,
                RELEVANCE_V3_POLICY.acceptance,
            )
        )
        detail = f"expected accepted={expected}, current policy accepted={predicted}"
    elif case.concern == "summary_format":
        failures = _summary_format_failures(case.artifact)
        expected = True
        predicted = not failures
        detail = "passes all format limits" if predicted else "; ".join(failures)
    elif case.concern == "grouping":
        output = cluster_articles(case.day, case.articles)
        expected = case.expected_same_group
        if expected:
            predicted = _all_same_group(
                output.cluster_set.groups,
                tuple(article.article.version_id for article in case.articles),
            )
        else:
            predicted = _same_group(
                output.cluster_set.groups,
                case.left_article_version_id,
                case.right_article_version_id,
            )
        detail = f"expected same_group={expected}, current clustering same_group={predicted}"
    elif case.concern == "ranking":
        if not case.observed_group_order:
            raise ValueError("Ranking case has no frozen report order")
        positions = {
            group_id: position for position, group_id in enumerate(case.observed_group_order)
        }
        expected = True
        predicted = positions[case.higher_group_id] < positions[case.lower_group_id]
        detail = (
            f"higher position={positions[case.higher_group_id]}, "
            f"lower position={positions[case.lower_group_id]}"
        )
    elif case.concern == "tier":
        expected = True
        predicted = case.observed_tier == case.expected_tier
        detail = f"expected tier={case.expected_tier}, frozen report tier={case.observed_tier}"
    elif case.concern == "confidence":
        expected = case.expected_sufficient
        predicted = case.observed_sufficient
        detail = f"expected sufficient={expected}, frozen report sufficient={predicted}"
    elif case.concern == "reader_presentation":
        expected = True
        predicted, detail = _evaluate_reader_expectation(case.expectation)
    elif case.concern == "daily_theme":
        raise ValueError("Daily-theme cases must be expanded before evaluation")
    else:
        raise ValueError(f"Unsupported evaluation concern: {case.concern}")
    return EvaluationCaseResult(
        case_id=case.case_id,
        concern=case.concern,
        passed=predicted == expected,
        control=case.control,
        expected_positive=expected,
        predicted_positive=predicted,
        detail=detail,
    )


class SubjectAssessmentEvaluationResult(NewsModel):
    case_results: tuple[EvaluationCaseResult, ...]
    merged_tier_resolutions: tuple[str, ...] = ()
    merged_pair_count: Annotated[int, Field(ge=0)]
    semantic_tie_count: Annotated[int, Field(ge=0)]
    passed_cases: Annotated[int, Field(ge=0)]
    total_cases: Annotated[int, Field(ge=0)]


class SubjectTierLabelConflict(NewsModel):
    """One merged subject whose frozen tier labels disagree."""

    subject_id: Sha256
    group_ids: tuple[Sha256, ...]
    labels: tuple[Tier, ...]
    governing_case_id: str | None = None
    superseded_case_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_resolution_shape(self) -> SubjectTierLabelConflict:
        if "excluded" in self.labels:
            if self.governing_case_id is not None or self.superseded_case_ids:
                raise ValueError("Excluded tier labels have no reviewed merge rule")
        elif self.governing_case_id is None or not self.superseded_case_ids:
            raise ValueError("Merged tier conflict must name its governing main case")
        return self


def subject_tier_label_conflicts(
    cases: Iterable[NewsEvaluationCase],
    theme_set: ReaderSubjectDailyThemeSet,
) -> tuple[SubjectTierLabelConflict, ...]:
    """Report merged subjects whose frozen tier labels disagree after construction.

    The reviewed merge rule from the September resolutions keeps the main
    judgment governing when worth_knowing subjects merge into it; excluded
    labels have no reviewed merge rule and stay unresolvable.
    """
    labels: dict[Sha256, list[tuple[str, Sha256, Tier]]] = {}
    for case in cases:
        if not isinstance(case, TierEvaluationCase):
            continue
        group_id = case.provenance.group_id
        if group_id is None:
            raise ValueError(f"Tier case has no stable group anchor: {case.case_id}")
        theme_id = resolve_group_anchor(theme_set, group_id)
        labels.setdefault(theme_id, []).append((case.case_id, group_id, case.expected_tier))
    conflicts = []
    for theme_id, values in labels.items():
        tiers: tuple[Tier, ...] = tuple(sorted({tier for _case_id, _group_id, tier in values}))
        if len(tiers) < 2:
            continue
        group_ids = tuple(dict.fromkeys(group_id for _case_id, group_id, _tier in values))
        if "excluded" in tiers:
            conflicts.append(
                SubjectTierLabelConflict(
                    subject_id=theme_id,
                    group_ids=group_ids,
                    labels=tiers,
                )
            )
            continue
        main_cases = tuple(value for value in values if value[2] == "main")
        conflicts.append(
            SubjectTierLabelConflict(
                subject_id=theme_id,
                group_ids=group_ids,
                labels=tiers,
                governing_case_id=main_cases[0][0],
                superseded_case_ids=tuple(
                    sorted(case_id for case_id, _group_id, tier in values if tier != "main")
                ),
            )
        )
    return tuple(conflicts)


def evaluate_subject_assessment(
    cases: Iterable[NewsEvaluationCase],
    theme_set: ReaderSubjectDailyThemeSet,
    assessment_set: DailySubjectAssessmentSet,
) -> SubjectAssessmentEvaluationResult:
    """Score stable group anchors against one trial-specific assessment set."""
    conflicts = subject_tier_label_conflicts(cases, theme_set)
    unresolvable = tuple(conflict for conflict in conflicts if conflict.governing_case_id is None)
    if unresolvable:
        raise ValueError(
            "Conflicting frozen tier labels without a reviewed merge rule: "
            + "; ".join(
                f"subject={conflict.subject_id} labels={conflict.labels} "
                f"groups={sorted(conflict.group_ids)}"
                for conflict in unresolvable
            )
        )
    superseded_case_ids = frozenset(
        case_id for conflict in conflicts for case_id in conflict.superseded_case_ids
    )
    resolutions = tuple(
        f"subject={conflict.subject_id} keeps the governing main case "
        f"{conflict.governing_case_id}; the standalone worth_knowing cases "
        f"{sorted(conflict.superseded_case_ids)} are conditional on the standalone scope"
        for conflict in conflicts
    )
    results = []
    merged_pairs = 0
    semantic_ties = 0
    assessments = {item.theme_id: item for item in assessment_set.assessments}
    for case in cases:
        if isinstance(case, RankingEvaluationCase):
            higher_theme = resolve_group_anchor(theme_set, case.higher_group_id)
            lower_theme = resolve_group_anchor(theme_set, case.lower_group_id)
            if higher_theme == lower_theme:
                merged_pairs += 1
                results.append(
                    EvaluationCaseResult(
                        case_id=case.case_id,
                        concern=case.concern,
                        passed=False,
                        control=case.control,
                        expected_positive=True,
                        predicted_positive=None,
                        detail=(
                            "ranking anchors merged into one subject: "
                            f"{case.higher_group_id}, {case.lower_group_id}, subject={higher_theme}"
                        ),
                    )
                )
                continue
            try:
                predicted = compare_subject_anchors(
                    assessment_set,
                    theme_set,
                    case.higher_group_id,
                    case.lower_group_id,
                )
            except ValueError as error:
                if "tie before the technical ID" not in str(error):
                    raise
                semantic_ties += 1
                results.append(
                    EvaluationCaseResult(
                        case_id=case.case_id,
                        concern=case.concern,
                        passed=False,
                        control=case.control,
                        expected_positive=True,
                        predicted_positive=None,
                        detail=str(error),
                    )
                )
                continue
            results.append(
                EvaluationCaseResult(
                    case_id=case.case_id,
                    concern=case.concern,
                    passed=predicted,
                    control=case.control,
                    expected_positive=True,
                    predicted_positive=predicted,
                    detail=(
                        f"higher subject={higher_theme}, lower subject={lower_theme}, "
                        f"ordered={predicted}"
                    ),
                )
            )
        elif isinstance(case, TierEvaluationCase):
            if case.case_id in superseded_case_ids:
                continue
            group_id = case.provenance.group_id
            if group_id is None:
                raise ValueError(f"Tier case has no stable group anchor: {case.case_id}")
            theme_id = resolve_group_anchor(theme_set, group_id)
            observed = assessments[theme_id].tier
            predicted = observed == case.expected_tier
            results.append(
                EvaluationCaseResult(
                    case_id=case.case_id,
                    concern=case.concern,
                    passed=predicted,
                    control=case.control,
                    expected_positive=True,
                    predicted_positive=predicted,
                    detail=f"expected tier={case.expected_tier}, assessed tier={observed}",
                )
            )
    return SubjectAssessmentEvaluationResult(
        case_results=tuple(results),
        merged_tier_resolutions=resolutions,
        merged_pair_count=merged_pairs,
        semantic_tie_count=semantic_ties,
        passed_cases=sum(item.passed for item in results),
        total_cases=len(results),
    )


def _summary_format_failures(artifact: FrozenGeneratedSummary) -> tuple[str, ...]:
    failures = []
    title_words = _words(artifact.title)
    forbidden = FORBIDDEN_TITLE_LABELS.intersection(word.casefold() for word in title_words)
    if forbidden:
        failures.append(f"forbidden title labels: {', '.join(sorted(forbidden))}")
    if not TITLE_MIN_WORDS <= len(title_words) <= TITLE_MAX_WORDS:
        failures.append(f"title has {len(title_words)} words")
    if len(artifact.title) > TITLE_MAX_CHARACTERS:
        failures.append(f"title has {len(artifact.title)} characters")
    summary_words = _words(artifact.summary)
    if len(summary_words) > SUMMARY_MAX_WORDS:
        failures.append(f"summary has {len(summary_words)} words")
    if len(artifact.summary) > SUMMARY_MAX_CHARACTERS:
        failures.append(f"summary has {len(artifact.summary)} characters")
    sentence_count = _sentence_count(artifact.summary)
    if sentence_count > SUMMARY_MAX_SENTENCES:
        failures.append(f"summary has {sentence_count} sentences")
    if not 1 <= len(artifact.key_points) <= KEY_POINT_MAX_COUNT:
        failures.append(f"summary has {len(artifact.key_points)} key points")
    oversized_points = tuple(
        position
        for position, point in enumerate(artifact.key_points, start=1)
        if len(point) > KEY_POINT_MAX_CHARACTERS
    )
    if oversized_points:
        failures.append(f"oversized key points: {oversized_points}")
    return tuple(failures)


def _sentence_count(value: str) -> int:
    endings = re.finditer(r"[.!?]+(?:[\"'’”»›)\]]+)?(?=\s|$)", value)
    return sum(not _is_name_abbreviation(value, ending) for ending in endings)


def _is_name_abbreviation(value: str, ending: re.Match[str]) -> bool:
    if value[ending.start()] != ".":
        return False
    previous_word = re.search(r"[^\W\d_]+$", value[: ending.start()], flags=re.UNICODE)
    next_word = re.match(r"\s+([^\W\d_]+)", value[ending.end() :], flags=re.UNICODE)
    if previous_word is None or next_word is None or not next_word.group(1)[0].isupper():
        return False
    token = previous_word.group(0)
    return token.casefold() in SUMMARY_TITLE_ABBREVIATIONS or (len(token) == 1 and token.isupper())


def _all_same_group(groups: tuple[NewsGroup, ...], article_ids: tuple[Sha256, ...]) -> bool:
    expected = set(article_ids)
    return any(expected <= set(group.article_version_ids) for group in groups)


def _same_group(groups: tuple[NewsGroup, ...], left_id: Sha256, right_id: Sha256) -> bool:
    return any(
        left_id in group.article_version_ids and right_id in group.article_version_ids
        for group in groups
    )


def _evaluate_reader_expectation(expectation: ReaderExpectation) -> tuple[bool, str]:
    if expectation.kind == "key_point_text":
        passed = expectation.observed_rem >= expectation.minimum_rem
        return passed, (
            f"minimum={expectation.minimum_rem:g}rem, observed={expectation.observed_rem:g}rem"
        )
    passed = expectation.distinct_source_count >= 2 or not expectation.differences
    return passed, (
        f"distinct sources={expectation.distinct_source_count}, "
        f"differences={len(expectation.differences)}"
    )


def _evaluate_report_completeness(
    snapshot: ReportEvaluationSnapshot,
) -> DeterministicCheckResult:
    expected_cluster = (snapshot.cluster_set.version_id,)
    expected_themes = (snapshot.themes.version_id,) if snapshot.themes is not None else ()
    expected_articles = tuple(item.article.version_id for item in snapshot.cluster_articles)
    expected_relevance = tuple(item.relevance.version_id for item in snapshot.cluster_articles)
    expected_groups = tuple(item.group_id for item in snapshot.group_inputs)
    expected_summaries = tuple(item.summary.version_id for item in snapshot.group_inputs)
    expected_sentiments = tuple(item.sentiment.version_id for item in snapshot.group_inputs)
    inputs = snapshot.report_inputs
    failures = []
    checks = (
        (
            tuple(item.version_id for item in inputs.themes),
            expected_themes,
            "report themes input is not exact",
        ),
        (snapshot.report_article_version_ids, expected_articles, "report article set is not exact"),
        (snapshot.report_group_ids, expected_groups, "report group set is not exact"),
        (
            tuple(item.version_id for item in inputs.cluster_set),
            expected_cluster,
            "report cluster-set input is not exact",
        ),
        (
            tuple(item.version_id for item in inputs.relevance),
            expected_relevance,
            "report relevance inputs are not exact",
        ),
        (
            tuple(item.version_id for item in inputs.summary),
            expected_summaries,
            "report summary inputs are not exact",
        ),
        (
            tuple(item.version_id for item in inputs.sentiment),
            expected_sentiments,
            "report sentiment inputs are not exact",
        ),
    )
    for actual, expected, failure in checks:
        if not _same_exact_set(actual, expected):
            failures.append(failure)
    passed = not failures
    return DeterministicCheckResult(
        check_id=f"report-inputs:{snapshot.report.version_id}",
        passed=passed,
        detail="complete exact inputs" if passed else "; ".join(failures),
    )


def _same_exact_set(actual: tuple[Sha256, ...], expected: tuple[Sha256, ...]) -> bool:
    return (
        len(actual) == len(set(actual))
        and len(expected) == len(set(expected))
        and set(actual) == set(expected)
    )


def _aggregate(concern: str, results: tuple[EvaluationCaseResult, ...]) -> EvaluationAggregate:
    selected = tuple(result for result in results if result.concern == concern)
    true_positives = sum(
        result.expected_positive and result.predicted_positive is True for result in selected
    )
    false_positives = sum(
        not result.expected_positive and result.predicted_positive is True for result in selected
    )
    false_negatives = sum(
        result.expected_positive and result.predicted_positive is not True for result in selected
    )
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = _ratio(2 * precision * recall, precision + recall)
    controls = tuple(result for result in selected if result.control)
    return EvaluationAggregate(
        concern=TypeAdapter(
            Literal[
                "relevance",
                "grouping",
                "ranking",
                "tier",
                "confidence",
                "daily_theme",
            ]
        ).validate_python(concern, strict=True),
        passed_cases=sum(result.passed for result in selected),
        total_cases=len(selected),
        precision=precision,
        recall=recall,
        f1=f1,
        positive_control_preservation=_ratio(
            sum(result.passed for result in controls), len(controls)
        ),
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _words(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+(?:['’][^\W_]+)?", value, flags=re.UNICODE))
