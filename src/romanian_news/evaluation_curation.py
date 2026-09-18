from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, TypeAdapter

from romanian_news import Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.articles.models import ExtractedArticle
from romanian_news.catalog import evaluations as evaluation_catalog
from romanian_news.evaluation import (
    ArticleRelevanceInput,
    ArticleRelevanceSpec,
    ConfidenceEvaluationCase,
    ConfidenceEvaluationSpec,
    DailyThemeJudgment,
    EvaluationCaseResult,
    EvaluationProvenance,
    EvaluationSpecProvenance,
    ExcludedFeedback,
    ExcludedFeedbackScore,
    FeedbackAssignments,
    GroupingEvaluationCase,
    GroupingEvaluationSpec,
    NewsEvaluationCase,
    NewsEvaluationDataset,
    NewsEvaluationManifest,
    NewsEvaluationSpec,
    NonExecutableFeedback,
    ProjectedFeedbackScore,
    RankingEvaluationCase,
    RankingEvaluationSpec,
    ReaderPresentationEvaluationSpec,
    RelevanceEvaluationCase,
    RelevanceEvaluationSpec,
    ReportEvaluationSnapshot,
    ReportEvaluationSpec,
    ReportGroupInputs,
    ReportGroupObservation,
    ReportInputReferences,
    SummaryFormatEvaluationSpec,
    ThemeEvaluationDayCase,
    ThemeEvaluationDaySpec,
    ThemePairExpectation,
    Tier,
    TierEvaluationCase,
    TierEvaluationSpec,
    evaluate_news_dataset,
)
from romanian_news.feedback import (
    GroupFeedbackTarget,
    NewsFeedbackEvent,
    NewsFeedbackTarget,
    ReportFeedbackTarget,
    ThemeFeedbackTarget,
)
from romanian_news.groups import (
    DailyClusterSet,
    EmbeddedArticle,
    EmbeddedArticleReference,
    NewsGroup,
)
from romanian_news.reports import (
    ArchivedDailyReport,
    DailyReport,
    DailyReportDocument,
    DailyReportSection,
    parse_daily_report,
    report_section_events,
)
from romanian_news.storage import read_verified_r2_object
from romanian_news.subject_assessments import parse_daily_subject_assessment_set
from romanian_news.themes import (
    AliasedReaderSubjectThemeSet,
    DailyThemeSet,
    ModelThemeConstruction,
    ReaderSubjectDailyThemeSet,
    SparseDailyThemeSet,
    SparseThemeConstruction,
    ThemeStageEvidence,
    load_daily_theme_input,
    parse_daily_theme_set,
)

DATASET_VERSION = "news-evaluation-2026-09-07-v6.2"
REVIEWED_AT = datetime.fromisoformat("2026-09-07T14:12:33+00:00")
ISSUE_URL = "https://github.com/mauricedesaxe/chartly/issues/286"
DEFAULT_OUTPUT_PATH = Path("data/news/evaluations/news-evaluation-2026-09-07-v6.2-manifest.json")
SCORE_SOURCE_MANIFEST_VERSION_ID: Sha256 = (
    "cbccda081c0b8d160aff1f0505662d0c4dcfda87404d0a33349d2dac897fab09"
)
SCORE_DATASET_VERSION = "news-evaluation-2026-09-07-v6.3"
SCORE_ISSUE_URL = "https://github.com/mauricedesaxe/chartly/issues/294"
SCORE_DEFAULT_OUTPUT_PATH = Path(
    "data/news/evaluations/news-evaluation-2026-09-07-v6.3-manifest.json"
)
PRIOR_MANIFEST_REFERENCE = ArtifactReference(
    artifact_id="news:evaluation-manifest:news-evaluation-2026-09-07-v6.1",
    version_id="17b39ccf990097441b87537304df5110f61c05aa730f34de31d6c2873fdb2364",
    content_digest="638c1f32b50db5a95a3a510783aa65db2dfc158ae6508c89feea2b921885ed7e",
    r2_key=(
        "news/evaluations/manifests/news-evaluation-2026-09-07-v6.1/"
        "638c1f32b50db5a95a3a510783aa65db2dfc158ae6508c89feea2b921885ed7e.json"
    ),
)
SOURCE_MANIFEST_REFERENCE = ArtifactReference(
    artifact_id="news:evaluation-manifest:news-evaluation-2026-09-05-v5",
    version_id="b973c0f3f05e65f93d097fbdbfb1461533351a28f1c9bcd0b52e0fe35e5056d6",
    content_digest="c2fad2de91cf4e7d25503caff07d6557713f111feecbcc1110d1f22e55bb4440",
    r2_key=(
        "news/evaluations/manifests/news-evaluation-2026-09-05-v5/"
        "c2fad2de91cf4e7d25503caff07d6557713f111feecbcc1110d1f22e55bb4440.json"
    ),
)
REPORT_2026_09_05 = "98171fdf44f84b155f63b0c65b929e787dbd7cbf871ec22892692e582893d2c8"
REPORT_2026_09_06 = "d7435504e45d1fc2e7b8caa9108ad152c60cc724a23d8fbc97b70576b4e1a81d"
REPORT_VERSION_IDS = (REPORT_2026_09_05, REPORT_2026_09_06)
_SHA256_ADAPTER = TypeAdapter(Sha256)
SOURCE_FEEDBACK_IDS = tuple(
    UUID(value)
    for value in (
        "ab7435ba-5a8e-49c3-b84a-b7a24ea21c68",
        "21a3aebf-bbc2-4604-831d-87646840bbc8",
        "03d3f87b-514f-4c02-b5af-49c10128bb12",
        "6386d5c2-160e-4407-890a-5c3017d03930",
        "c2f829a7-6235-423c-8624-b3449b53759c",
        "dc71753c-9f2b-4119-b602-8fad13c4a2b7",
        "e3b0f013-4dfb-47a7-913f-bb50f2cbc186",
        "f19fb4a2-d2d1-4c98-9306-6d8566c4d5a8",
        "597bd307-65e8-453f-bc27-a537f567ef33",
        "c38eba84-1306-4a8f-9adf-810624313bf2",
        "7f259640-24db-49e0-a3df-768637d913b5",
        "a6433016-7d6d-4a4f-9693-4d73d79f1788",
        "da415f63-dab3-436b-9911-42230b0b7dfa",
        "f0dbe63b-22ea-40ac-8863-ac0a558c20ce",
        "6277d4eb-cbde-4b78-8e55-4116c78106ff",
        "19e282fc-ec82-4a61-af97-a74578fdf7bf",
        "af82a151-c4da-4510-b3eb-b5a1625e69a8",
        "d49247e5-53d3-4f67-807d-84b2455a045f",
        "2b89d953-caaf-481b-93d4-82ede5fe128c",
        "97705a8f-0576-4639-9103-48592fa02204",
        "77ab30f2-4ce2-4700-afa5-5c6e10a03a7c",
        "055fdd89-bd4d-43f6-a5c8-4bfcf6307408",
        "4351d88d-7074-47d5-a9b0-f66c55b230a9",
        "838934cc-f45b-4d0c-811d-cf0afab7195b",
        "e7624708-a00d-4a63-9528-c9e3173865ad",
        "07cefc5b-dc24-420b-8865-65fa6c5cb792",
        "c0112f70-d09c-461e-8008-df059876b499",
        "18344968-08f1-46a9-9298-d289cf329e7a",
        "fe4b48e8-3f56-46ff-8da2-44ffae428e46",
        "c344396f-13be-4bdd-a169-de9b5da5d11a",
    )
)


class _ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    id: str


class _ModelOutputRoutePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    request_id: Sha256
    provider_response: _ProviderResponse | None = None
    provider_responses: tuple[_ProviderResponse, ...] = ()

    @property
    def accepted_response_id(self) -> str:
        responses = self.provider_responses or (
            (self.provider_response,) if self.provider_response is not None else ()
        )
        if not responses:
            raise ValueError("Model output has no provider response")
        return responses[-1].id


@dataclass(frozen=True)
class _ModelObservationRoute:
    output: ArtifactReference
    attempt_id: Sha256


@dataclass(frozen=True)
class RelevanceSpec:
    feedback_id: str
    report_version_id: Sha256
    group_id: Sha256
    expected_accepted: bool
    control: bool = False
    case_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroupingSpec:
    case_id: str
    feedback_id: str
    report_version_id: Sha256
    left_group_id: Sha256
    right_group_id: Sha256 | None = None
    expected_same_group: bool = True
    control: bool = True


@dataclass(frozen=True)
class RankingSpec:
    case_id: str
    feedback_id: str
    report_version_id: Sha256
    higher_group_id: Sha256
    lower_group_id: Sha256
    control: bool = False


@dataclass(frozen=True)
class TierSpec:
    case_id: str
    feedback_id: str
    report_version_id: Sha256
    group_id: Sha256
    expected_tier: Tier
    control: bool = False


@dataclass(frozen=True)
class ConfidenceSpec:
    case_id: str
    feedback_id: str
    report_version_id: Sha256
    group_id: Sha256
    expected_sufficient: bool
    control: bool = False


@dataclass(frozen=True)
class ReportBundle:
    snapshot: ReportEvaluationSnapshot
    report: DailyReportDocument
    cluster_set: DailyClusterSet
    articles: dict[Sha256, EmbeddedArticleReference]
    relevance: dict[Sha256, ArtifactReference]


@dataclass(frozen=True)
class _ReviewedReportExpectation:
    report_version_id: Sha256
    feedback_count: int


@dataclass(frozen=True)
class _FeedbackTargetExpectation:
    feedback_id: UUID
    target: NewsFeedbackTarget


RELEVANCE_SPECS = (
    RelevanceSpec(
        "03d3f87b-514f-4c02-b5af-49c10128bb12",
        REPORT_2026_09_05,
        "fc7ebc4fa1fbc1c13a90ceccfeb8c6172860a167f66d3c9ac8308999690c67d5",
        True,
        True,
    ),
    RelevanceSpec(
        "6386d5c2-160e-4407-890a-5c3017d03930",
        REPORT_2026_09_05,
        "0f90c6cf92f9f4a54cedc4ae9412e3f94a3ad3d57b3b81728febe406aeef6120",
        False,
    ),
    RelevanceSpec(
        "c2f829a7-6235-423c-8624-b3449b53759c",
        REPORT_2026_09_05,
        "80f79778d395d721c20c287d99e848afcfbd0bb32a014458b526cb1846d26778",
        False,
    ),
    RelevanceSpec(
        "e3b0f013-4dfb-47a7-913f-bb50f2cbc186",
        REPORT_2026_09_05,
        "8011e9526d1d0a3dd1049ee4b80c659b694d21fcb6b072cb933800583a98c014",
        True,
        True,
    ),
    RelevanceSpec(
        "f19fb4a2-d2d1-4c98-9306-6d8566c4d5a8",
        REPORT_2026_09_05,
        "907de879e7d432d996f9076ff4278a7b208ed9c310cabd7ca053bbf3683bc0ae",
        True,
        True,
    ),
    RelevanceSpec(
        "d49247e5-53d3-4f67-807d-84b2455a045f",
        REPORT_2026_09_06,
        "6b77cba455cbe8c3a7041b640e0f9e7d6ab20a5ae3ff07628a4a72ad69a54e52",
        False,
    ),
    RelevanceSpec(
        "d49247e5-53d3-4f67-807d-84b2455a045f",
        REPORT_2026_09_06,
        "c6a07e27021c7d1e614339375620a2d1cb704140655f1d245cdcb4f58ff7e89a",
        False,
    ),
    RelevanceSpec(
        "97705a8f-0576-4639-9103-48592fa02204",
        REPORT_2026_09_06,
        "65affae7d0032485df72a51256eaa7711161033f20d68471477659d47bf2902a",
        True,
        True,
    ),
    RelevanceSpec(
        "77ab30f2-4ce2-4700-afa5-5c6e10a03a7c",
        REPORT_2026_09_06,
        "76679f1b4befdfaa72460c73f2c22f2b3acf8295dd47b957b6bcb0fde0b2a0bb",
        False,
    ),
    RelevanceSpec(
        "4351d88d-7074-47d5-a9b0-f66c55b230a9",
        REPORT_2026_09_06,
        "09dd1af7f7649fb3a8f609f8afabc973e4e047923f6fad95024271ae34055f9b",
        True,
        True,
    ),
    RelevanceSpec(
        "07cefc5b-dc24-420b-8865-65fa6c5cb792",
        REPORT_2026_09_06,
        "5063f1b167df128cc2482b1653ac10009457434599da5a5712d5e509754f5973",
        True,
        True,
    ),
    RelevanceSpec(
        "fe4b48e8-3f56-46ff-8da2-44ffae428e46",
        REPORT_2026_09_06,
        "bb4583f083fd208080b838c975f34f9ceb9c6ac9c56897113b4c5502afb3c888",
        True,
        True,
    ),
    RelevanceSpec(
        "c344396f-13be-4bdd-a169-de9b5da5d11a",
        REPORT_2026_09_06,
        "a2551a5e84e04a8e7f1147669941e91eba4e541b52880fc9bce3f86b347b3bb5",
        True,
        True,
    ),
)

GROUPING_SPECS = (
    GroupingSpec(
        "sep06-duplicate-csat-modernization-event",
        "f0dbe63b-22ea-40ac-8863-ac0a558c20ce",
        REPORT_2026_09_06,
        "cdc435382e3e01b08b5351a7bccdf1f565d43b8e4c7eaa0226adab130018aa33",
        "5063f1b167df128cc2482b1653ac10009457434599da5a5712d5e509754f5973",
        True,
        False,
    ),
    GroupingSpec(
        "sep06-duplicate-csat-convocation-event",
        "6277d4eb-cbde-4b78-8e55-4116c78106ff",
        REPORT_2026_09_06,
        "20cf144ef2b791413d9f977fc48c4457cb52a93b8c4ab21cc8b0f95edcafaaca",
        "5063f1b167df128cc2482b1653ac10009457434599da5a5712d5e509754f5973",
        True,
        False,
    ),
    GroupingSpec(
        "sep06-tax-pressure-coverage-control",
        "4351d88d-7074-47d5-a9b0-f66c55b230a9",
        REPORT_2026_09_06,
        "09dd1af7f7649fb3a8f609f8afabc973e4e047923f6fad95024271ae34055f9b",
    ),
    GroupingSpec(
        "sep06-csat-coverage-control",
        "07cefc5b-dc24-420b-8865-65fa6c5cb792",
        REPORT_2026_09_06,
        "5063f1b167df128cc2482b1653ac10009457434599da5a5712d5e509754f5973",
    ),
    GroupingSpec(
        "sep06-prime-minister-coverage-control",
        "fe4b48e8-3f56-46ff-8da2-44ffae428e46",
        REPORT_2026_09_06,
        "bb4583f083fd208080b838c975f34f9ceb9c6ac9c56897113b4c5502afb3c888",
    ),
    GroupingSpec(
        "sep06-pnl-opposition-coverage-control",
        "c344396f-13be-4bdd-a169-de9b5da5d11a",
        REPORT_2026_09_06,
        "a2551a5e84e04a8e7f1147669941e91eba4e541b52880fc9bce3f86b347b3bb5",
    ),
)

RANKING_SPECS = (
    RankingSpec(
        "sep05-prime-minister-ranks-above-fuel-control",
        "03d3f87b-514f-4c02-b5af-49c10128bb12",
        REPORT_2026_09_05,
        "fc7ebc4fa1fbc1c13a90ceccfeb8c6172860a167f66d3c9ac8308999690c67d5",
        "907de879e7d432d996f9076ff4278a7b208ed9c310cabd7ca053bbf3683bc0ae",
        True,
    ),
    RankingSpec(
        "sep05-fuel-ranks-above-pnrr",
        "f19fb4a2-d2d1-4c98-9306-6d8566c4d5a8",
        REPORT_2026_09_05,
        "907de879e7d432d996f9076ff4278a7b208ed9c310cabd7ca053bbf3683bc0ae",
        "8011e9526d1d0a3dd1049ee4b80c659b694d21fcb6b072cb933800583a98c014",
    ),
    RankingSpec(
        "sep05-pnrr-ranks-above-energy-storage-control",
        "e3b0f013-4dfb-47a7-913f-bb50f2cbc186",
        REPORT_2026_09_05,
        "8011e9526d1d0a3dd1049ee4b80c659b694d21fcb6b072cb933800583a98c014",
        "a77c35eee904c9c7e2e6d741d322bd8932cd03d4c3ee29ce039a86d9f6b49201",
        True,
    ),
)

THEME_JUDGMENTS = (
    DailyThemeJudgment(
        judgment_id="sep05-political-deadlock-events-share-theme",
        feedback_ids=(UUID("dc71753c-9f2b-4119-b602-8fad13c4a2b7"),),
        report_version_id=REPORT_2026_09_05,
        left_group_id="b077d46a2b2ec26644fe42b0a835cdbce771d264b987fa495d14b927ce9a5517",
        right_group_id="fc7ebc4fa1fbc1c13a90ceccfeb8c6172860a167f66d3c9ac8308999690c67d5",
        expected_same_theme=True,
        rationale="Both events cover the same Romanian government formation deadlock.",
    ),
    DailyThemeJudgment(
        judgment_id="sep06-retail-and-motorway-events-differ",
        feedback_ids=(UUID("2b89d953-caaf-481b-93d4-82ede5fe128c"),),
        report_version_id=REPORT_2026_09_06,
        left_group_id="65affae7d0032485df72a51256eaa7711161033f20d68471477659d47bf2902a",
        right_group_id="668cf6e7c69b3d0c63841813a76bd77d1697d63a200243487118371a154c2647",
        expected_same_theme=False,
        rationale="Retail contraction and a motorway's regional effect are distinct events.",
    ),
)

EXCLUSIONS = (
    ExcludedFeedback(
        feedback_id=UUID("21a3aebf-bbc2-4604-831d-87646840bbc8"),
        report_version_id=REPORT_2026_09_05,
        reason="superseded",
        superseded_by_feedback_id=UUID("03d3f87b-514f-4c02-b5af-49c10128bb12"),
    ),
    ExcludedFeedback(
        feedback_id=UUID("597bd307-65e8-453f-bc27-a537f567ef33"),
        report_version_id=REPORT_2026_09_05,
        reason="superseded",
        superseded_by_feedback_id=UUID("7f259640-24db-49e0-a3df-768637d913b5"),
    ),
    ExcludedFeedback(
        feedback_id=UUID("c38eba84-1306-4a8f-9adf-810624313bf2"),
        report_version_id=REPORT_2026_09_05,
        reason="superseded",
        superseded_by_feedback_id=UUID("7f259640-24db-49e0-a3df-768637d913b5"),
    ),
    ExcludedFeedback(
        feedback_id=UUID("a6433016-7d6d-4a4f-9693-4d73d79f1788"),
        report_version_id=REPORT_2026_09_06,
        reason="superseded",
        superseded_by_feedback_id=UUID("055fdd89-bd4d-43f6-a5c8-4bfcf6307408"),
    ),
    ExcludedFeedback(
        feedback_id=UUID("da415f63-dab3-436b-9911-42230b0b7dfa"),
        report_version_id=REPORT_2026_09_06,
        reason="superseded",
        superseded_by_feedback_id=UUID("c0112f70-d09c-461e-8008-df059876b499"),
    ),
    ExcludedFeedback(
        feedback_id=UUID("e7624708-a00d-4a63-9528-c9e3173865ad"),
        report_version_id=REPORT_2026_09_06,
        reason="superseded",
        superseded_by_feedback_id=UUID("c0112f70-d09c-461e-8008-df059876b499"),
    ),
    NonExecutableFeedback(
        feedback_id=UUID("ab7435ba-5a8e-49c3-b84a-b7a24ea21c68"),
        report_version_id=REPORT_2026_09_05,
        reason="non_executable",
        concern="ambiguous",
        rationale="The note allows either a ranking or grouping correction.",
    ),
    NonExecutableFeedback(
        feedback_id=UUID("7f259640-24db-49e0-a3df-768637d913b5"),
        report_version_id=REPORT_2026_09_05,
        reason="non_executable",
        concern="operations",
        rationale="Report length has no deterministic expectation.",
    ),
    NonExecutableFeedback(
        feedback_id=UUID("19e282fc-ec82-4a61-af97-a74578fdf7bf"),
        report_version_id=REPORT_2026_09_06,
        reason="non_executable",
        concern="presentation",
        rationale="Collapsed event presentation is outside the evaluator.",
    ),
    NonExecutableFeedback(
        feedback_id=UUID("af82a151-c4da-4510-b3eb-b5a1625e69a8"),
        report_version_id=REPORT_2026_09_06,
        reason="non_executable",
        concern="ambiguous",
        rationale="The note allows ingestion, grouping, or relevance failures.",
    ),
    NonExecutableFeedback(
        feedback_id=UUID("055fdd89-bd4d-43f6-a5c8-4bfcf6307408"),
        report_version_id=REPORT_2026_09_06,
        reason="non_executable",
        concern="language",
        rationale="Theme language has no deterministic evaluator yet.",
    ),
    NonExecutableFeedback(
        feedback_id=UUID("c0112f70-d09c-461e-8008-df059876b499"),
        report_version_id=REPORT_2026_09_06,
        reason="non_executable",
        concern="language",
        rationale="Theme language has no deterministic evaluator yet.",
    ),
    NonExecutableFeedback(
        feedback_id=UUID("18344968-08f1-46a9-9298-d289cf329e7a"),
        report_version_id=REPORT_2026_09_06,
        reason="non_executable",
        concern="language",
        rationale="Theme language has no deterministic evaluator yet.",
    ),
    NonExecutableFeedback(
        feedback_id=UUID("838934cc-f45b-4d0c-811d-cf0afab7195b"),
        report_version_id=REPORT_2026_09_06,
        reason="non_executable",
        concern="contradictory",
        rationale="The negative rating conflicts with the positive note.",
    ),
)

_REVIEWED_REPORT_EXPECTATIONS = (
    _ReviewedReportExpectation(report_version_id=REPORT_2026_09_05, feedback_count=11),
    _ReviewedReportExpectation(report_version_id=REPORT_2026_09_06, feedback_count=19),
)

FEEDBACK_TARGETS: dict[UUID, NewsFeedbackTarget] = {
    UUID("ab7435ba-5a8e-49c3-b84a-b7a24ea21c68"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_05,
        group_id="35adfdcad50bcfa9544ef7fd3b23fb1f1a7b55e0ee3efc91cb88a4b30a61dd26",
    ),
    UUID("21a3aebf-bbc2-4604-831d-87646840bbc8"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_05,
        group_id="fc7ebc4fa1fbc1c13a90ceccfeb8c6172860a167f66d3c9ac8308999690c67d5",
    ),
    UUID("03d3f87b-514f-4c02-b5af-49c10128bb12"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_05,
        group_id="fc7ebc4fa1fbc1c13a90ceccfeb8c6172860a167f66d3c9ac8308999690c67d5",
    ),
    UUID("6386d5c2-160e-4407-890a-5c3017d03930"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_05,
        group_id="0f90c6cf92f9f4a54cedc4ae9412e3f94a3ad3d57b3b81728febe406aeef6120",
    ),
    UUID("c2f829a7-6235-423c-8624-b3449b53759c"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_05,
        group_id="80f79778d395d721c20c287d99e848afcfbd0bb32a014458b526cb1846d26778",
    ),
    UUID("dc71753c-9f2b-4119-b602-8fad13c4a2b7"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_05,
        group_id="b077d46a2b2ec26644fe42b0a835cdbce771d264b987fa495d14b927ce9a5517",
    ),
    UUID("e3b0f013-4dfb-47a7-913f-bb50f2cbc186"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_05,
        group_id="8011e9526d1d0a3dd1049ee4b80c659b694d21fcb6b072cb933800583a98c014",
    ),
    UUID("f19fb4a2-d2d1-4c98-9306-6d8566c4d5a8"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_05,
        group_id="907de879e7d432d996f9076ff4278a7b208ed9c310cabd7ca053bbf3683bc0ae",
    ),
    UUID("597bd307-65e8-453f-bc27-a537f567ef33"): ReportFeedbackTarget(
        report_version_id=REPORT_2026_09_05
    ),
    UUID("c38eba84-1306-4a8f-9adf-810624313bf2"): ReportFeedbackTarget(
        report_version_id=REPORT_2026_09_05
    ),
    UUID("7f259640-24db-49e0-a3df-768637d913b5"): ReportFeedbackTarget(
        report_version_id=REPORT_2026_09_05
    ),
    UUID("a6433016-7d6d-4a4f-9693-4d73d79f1788"): ThemeFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        theme_id="0b2eaa327f4ed1c853fe6ef11f7d3fb7064ccc840e793c5d36fe90cf2caf6f43",
    ),
    UUID("da415f63-dab3-436b-9911-42230b0b7dfa"): ThemeFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        theme_id="4695dea732abe3e63a1ca4897bf8bfb67d781017aaab9fe8a856ad2dfdb0bf2f",
    ),
    UUID("f0dbe63b-22ea-40ac-8863-ac0a558c20ce"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        group_id="cdc435382e3e01b08b5351a7bccdf1f565d43b8e4c7eaa0226adab130018aa33",
    ),
    UUID("6277d4eb-cbde-4b78-8e55-4116c78106ff"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        group_id="20cf144ef2b791413d9f977fc48c4457cb52a93b8c4ab21cc8b0f95edcafaaca",
    ),
    UUID("19e282fc-ec82-4a61-af97-a74578fdf7bf"): ReportFeedbackTarget(
        report_version_id=REPORT_2026_09_06
    ),
    UUID("af82a151-c4da-4510-b3eb-b5a1625e69a8"): ThemeFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        theme_id="4497dfb3b7e36c6d39d8861eadc2e8932f33fe6d50f22a123d699f141f20bcdb",
    ),
    UUID("d49247e5-53d3-4f67-807d-84b2455a045f"): ThemeFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        theme_id="70690e0e1eb6cb4b572faaebe617a76a586495935b690224e53cfdb81b0c46e5",
    ),
    UUID("2b89d953-caaf-481b-93d4-82ede5fe128c"): ThemeFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        theme_id="155711a078906e0a86c8235e396ddcd7f682887b8f1e7cdd6239802467a84f04",
    ),
    UUID("97705a8f-0576-4639-9103-48592fa02204"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        group_id="65affae7d0032485df72a51256eaa7711161033f20d68471477659d47bf2902a",
    ),
    UUID("77ab30f2-4ce2-4700-afa5-5c6e10a03a7c"): ThemeFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        theme_id="cd6cbe4ec79821ec6c4a974e727c10b51247448a7d2c1cbb83e9c37d7d643969",
    ),
    UUID("055fdd89-bd4d-43f6-a5c8-4bfcf6307408"): ThemeFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        theme_id="0b2eaa327f4ed1c853fe6ef11f7d3fb7064ccc840e793c5d36fe90cf2caf6f43",
    ),
    UUID("4351d88d-7074-47d5-a9b0-f66c55b230a9"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        group_id="09dd1af7f7649fb3a8f609f8afabc973e4e047923f6fad95024271ae34055f9b",
    ),
    UUID("838934cc-f45b-4d0c-811d-cf0afab7195b"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        group_id="14f7a1cc3884de51098a8920dd5c71ccdaf896e0b520cfc68a5974cb2d2bb9d3",
    ),
    UUID("e7624708-a00d-4a63-9528-c9e3173865ad"): ThemeFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        theme_id="4695dea732abe3e63a1ca4897bf8bfb67d781017aaab9fe8a856ad2dfdb0bf2f",
    ),
    UUID("07cefc5b-dc24-420b-8865-65fa6c5cb792"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        group_id="5063f1b167df128cc2482b1653ac10009457434599da5a5712d5e509754f5973",
    ),
    UUID("c0112f70-d09c-461e-8008-df059876b499"): ThemeFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        theme_id="4695dea732abe3e63a1ca4897bf8bfb67d781017aaab9fe8a856ad2dfdb0bf2f",
    ),
    UUID("18344968-08f1-46a9-9298-d289cf329e7a"): ThemeFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        theme_id="13d82c201a3a52377b4a16a648b226051a5a347710b02882b8fb615a65e6afc9",
    ),
    UUID("fe4b48e8-3f56-46ff-8da2-44ffae428e46"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        group_id="bb4583f083fd208080b838c975f34f9ceb9c6ac9c56897113b4c5502afb3c888",
    ),
    UUID("c344396f-13be-4bdd-a169-de9b5da5d11a"): GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_06,
        group_id="a2551a5e84e04a8e7f1147669941e91eba4e541b52880fc9bce3f86b347b3bb5",
    ),
}
_FEEDBACK_TARGET_EXPECTATIONS = tuple(
    _FeedbackTargetExpectation(feedback_id=feedback_id, target=target)
    for feedback_id, target in FEEDBACK_TARGETS.items()
)
_REPRESENTED_FEEDBACK_IDS = frozenset(
    {
        *(UUID(spec.feedback_id) for spec in RELEVANCE_SPECS),
        *(UUID(spec.feedback_id) for spec in GROUPING_SPECS),
        *(UUID(spec.feedback_id) for spec in RANKING_SPECS),
        *(feedback_id for judgment in THEME_JUDGMENTS for feedback_id in judgment.feedback_ids),
    }
)


def build_news_evaluation_manifest() -> NewsEvaluationManifest:
    """Build the corrected manifest from the immutable v6.1 release."""
    prior = NewsEvaluationManifest.model_validate_json(
        read_evaluation_artifact(PRIOR_MANIFEST_REFERENCE), strict=True
    )
    report = next(item for item in prior.reports if item.report.version_id == REPORT_2026_09_06)
    cluster_set = DailyClusterSet.model_validate_json(
        read_evaluation_artifact(report.cluster_set), strict=True
    )
    references = evaluation_catalog.read_news_evaluation_artifact_references(
        (
            *cluster_set.article_version_ids,
            *cluster_set.relevance_version_ids,
            *cluster_set.embedding_version_ids,
        )
    )
    articles = {
        article_id: EmbeddedArticleReference(
            article=references[article_id],
            relevance=references[relevance_id],
            embedding=references[embedding_id],
        )
        for article_id, relevance_id, embedding_id in zip(
            cluster_set.article_version_ids,
            cluster_set.relevance_version_ids,
            cluster_set.embedding_version_ids,
            strict=True,
        )
    }
    corrected_specs = {spec.case_id: spec for spec in GROUPING_SPECS if not spec.control}
    cases = tuple(
        _corrected_grouping_spec(case, corrected_specs[case.case_id], cluster_set, articles)
        if case.case_id in corrected_specs
        else case
        for case in prior.cases
    )
    return prior.model_copy(
        update={
            "version": DATASET_VERSION,
            "reviewed_at": REVIEWED_AT,
            "issue_url": ISSUE_URL,
            "prior_manifest": PRIOR_MANIFEST_REFERENCE,
            "cases": cases,
        }
    )


def build_news_feedback_score_manifest() -> NewsEvaluationManifest:
    """Build v6.3 score curation from the exact pinned v6.2 evidence."""
    prior_reference = evaluation_catalog.read_news_evaluation_artifact_references(
        (SCORE_SOURCE_MANIFEST_VERSION_ID,)
    )[SCORE_SOURCE_MANIFEST_VERSION_ID]
    prior = NewsEvaluationManifest.model_validate_json(
        read_evaluation_artifact(prior_reference), strict=True
    )
    if prior.version != DATASET_VERSION:
        raise ValueError(f"Pinned score source must be {DATASET_VERSION}: {prior.version}")
    if tuple(prior.source_feedback_ids) != SOURCE_FEEDBACK_IDS:
        raise ValueError("Pinned v6.2 manifest does not contain the reviewed feedback source")

    dataset = evaluation_catalog.hydrate_news_evaluation_manifest(prior)
    feedback = evaluation_catalog.read_news_evaluation_feedback(SOURCE_FEEDBACK_IDS)
    bundles = {
        report_version_id: read_evaluation_report_bundle(report_version_id)
        for report_version_id in REPORT_VERSION_IDS
    }
    _require_exact_feedback_source(feedback, bundles)
    score_curation = _build_feedback_score_curation(prior, dataset, feedback)
    candidate = prior.model_copy(
        update={
            "version": SCORE_DATASET_VERSION,
            "issue_url": SCORE_ISSUE_URL,
            "prior_manifest": prior_reference,
            "score_curation": score_curation,
        }
    )
    return NewsEvaluationManifest.model_validate_json(candidate.model_dump_json(), strict=True)


def _build_feedback_score_curation(
    manifest: NewsEvaluationManifest,
    dataset: NewsEvaluationDataset,
    feedback: tuple[NewsFeedbackEvent, ...],
) -> tuple[ProjectedFeedbackScore | ExcludedFeedbackScore, ...]:
    source_ids = frozenset(manifest.source_feedback_ids)
    rows = {row.feedback_id: row for row in feedback}
    decisions: dict[UUID, list[ProjectedFeedbackScore | ExcludedFeedbackScore]] = {
        feedback_id: [] for feedback_id in manifest.source_feedback_ids
    }
    unresolved = _append_case_score_curation(
        dataset,
        source_ids,
        decisions,
    )
    unresolved.extend(_append_excluded_feedback_scores(manifest, rows, dataset, decisions))

    if unresolved:
        raise ValueError("Score curation has unresolved cases:\n" + "\n".join(unresolved))
    missing = tuple(
        feedback_id for feedback_id in manifest.source_feedback_ids if not decisions[feedback_id]
    )
    if missing:
        raise ValueError(f"Score curation has unresolved feedback IDs: {missing}")
    return tuple(
        decision
        for feedback_id in manifest.source_feedback_ids
        for decision in decisions[feedback_id]
    )


def _append_case_score_curation(
    dataset: NewsEvaluationDataset,
    source_ids: frozenset[UUID],
    decisions: dict[UUID, list[ProjectedFeedbackScore | ExcludedFeedbackScore]],
) -> list[str]:
    results = {result.case_id: result for result in evaluate_news_dataset(dataset).case_results}
    deterministic_results: dict[
        tuple[UUID, Literal["grouping", "ranking"], Sha256], list[EvaluationCaseResult]
    ] = {}
    unresolved: list[str] = []

    for case in dataset.cases:
        feedback_ids = tuple(
            feedback_id for feedback_id in case.provenance.feedback_ids if feedback_id in source_ids
        )
        if not feedback_ids:
            continue
        if case.concern == "relevance":
            try:
                _append_relevance_scores(case, feedback_ids, results[case.case_id], decisions)
            except ValueError as error:
                unresolved.extend(
                    f"{feedback_id} relevance {case.case_id}: {error}"
                    for feedback_id in feedback_ids
                )
            continue
        deterministic_concern: Literal["grouping", "ranking"] | None = None
        if case.concern == "grouping":
            deterministic_concern = "grouping"
        elif case.concern == "ranking":
            deterministic_concern = "ranking"
        if deterministic_concern is not None:
            result = results[case.case_id]
            report_version_id = _case_report_version(case)
            for feedback_id in feedback_ids:
                deterministic_results.setdefault(
                    (feedback_id, deterministic_concern, report_version_id), []
                ).append(result)
            continue
        if case.concern == "daily_theme":
            try:
                _append_theme_grouping_scores(case, feedback_ids, dataset, decisions)
            except ValueError as error:
                unresolved.extend(
                    f"{feedback_id} grouping {case.case_id}: {error}"
                    for feedback_id in feedback_ids
                )

    _append_deterministic_score_exclusions(deterministic_results, decisions)
    return unresolved


def _append_relevance_scores(
    case: RelevanceEvaluationCase,
    feedback_ids: tuple[UUID, ...],
    result: EvaluationCaseResult,
    decisions: dict[UUID, list[ProjectedFeedbackScore | ExcludedFeedbackScore]],
) -> None:
    model_output = _require_one_reference(
        case.provenance.model_outputs, case.case_id, "model output"
    )
    route = _model_output_observation(model_output)
    report_version_id = _case_report_version(case)
    for feedback_id in feedback_ids:
        decisions[feedback_id].append(
            ProjectedFeedbackScore(
                feedback_id=feedback_id,
                concern="relevance",
                report_version_id=report_version_id,
                model_output=route.output,
                model_attempt_id=route.attempt_id,
                polarity=_result_polarity(result.passed),
                rationale=f"{case.case_id}: {result.detail}",
            )
        )


def _append_deterministic_score_exclusions(
    deterministic_results: dict[
        tuple[UUID, Literal["grouping", "ranking"], Sha256], list[EvaluationCaseResult]
    ],
    decisions: dict[UUID, list[ProjectedFeedbackScore | ExcludedFeedbackScore]],
) -> None:
    for (feedback_id, concern, report_version_id), case_results in deterministic_results.items():
        passed = all(bool(result.passed) for result in case_results)
        detail = "; ".join(f"{result.case_id}: {result.detail}" for result in case_results)
        decisions[feedback_id].append(
            ExcludedFeedbackScore(
                feedback_id=feedback_id,
                reason="no_model_observation",
                concern=concern,
                report_version_id=report_version_id,
                polarity=_result_polarity(passed),
                rationale=detail,
            )
        )


def _append_excluded_feedback_scores(
    manifest: NewsEvaluationManifest,
    rows: dict[UUID, NewsFeedbackEvent],
    dataset: NewsEvaluationDataset,
    decisions: dict[UUID, list[ProjectedFeedbackScore | ExcludedFeedbackScore]],
) -> list[str]:
    unresolved: list[str] = []
    for exclusion in manifest.excluded_feedback:
        if isinstance(exclusion, NonExecutableFeedback) and exclusion.concern == "language":
            try:
                decision = _language_score(exclusion, rows[exclusion.feedback_id], dataset)
            except ValueError as error:
                unresolved.append(f"{exclusion.feedback_id} language: {error}")
            else:
                decisions[exclusion.feedback_id].append(decision)
        else:
            decisions[exclusion.feedback_id].append(_score_exclusion(exclusion))
    return unresolved


def _append_theme_grouping_scores(
    case: ThemeEvaluationDayCase,
    feedback_ids: tuple[UUID, ...],
    dataset: NewsEvaluationDataset,
    decisions: dict[UUID, list[ProjectedFeedbackScore | ExcludedFeedbackScore]],
) -> None:
    snapshot = _dataset_report(dataset, case.source_report.version_id)
    if snapshot.themes is None:
        for feedback_id in feedback_ids:
            decisions[feedback_id].append(
                ExcludedFeedbackScore(
                    feedback_id=feedback_id,
                    reason="no_model_observation",
                    concern="grouping",
                    report_version_id=case.source_report.version_id,
                    rationale=f"{case.case_id}: frozen report has no theme model output.",
                )
            )
        return
    theme_set = _read_theme_set(snapshot)
    route = (
        None
        if isinstance(theme_set.construction, ModelThemeConstruction)
        else _theme_observation(snapshot, "assignment")[1]
    )
    expectations = {
        feedback_id: expectation
        for expectation in case.expectations
        for feedback_id in expectation.feedback_ids
        if feedback_id in feedback_ids
    }
    if set(expectations) != set(feedback_ids):
        raise ValueError(f"Theme case has ambiguous feedback expectations: {case.case_id}")
    for feedback_id in feedback_ids:
        expectation = expectations[feedback_id]
        passed, detail = _frozen_theme_expectation(theme_set, expectation)
        polarity = _result_polarity(passed)
        if route is None:
            decisions[feedback_id].append(
                ExcludedFeedbackScore(
                    feedback_id=feedback_id,
                    reason="no_concern_specific_observation",
                    concern="grouping",
                    report_version_id=case.source_report.version_id,
                    polarity=polarity,
                    rationale=(
                        f"{expectation.case_id}: {detail}; the frozen theme output uses "
                        "one combined assignment and prose observation."
                    ),
                )
            )
        else:
            decisions[feedback_id].append(
                ProjectedFeedbackScore(
                    feedback_id=feedback_id,
                    concern="grouping",
                    report_version_id=case.source_report.version_id,
                    model_output=route.output,
                    model_attempt_id=route.attempt_id,
                    polarity=polarity,
                    rationale=f"{expectation.case_id}: {detail}",
                )
            )


def _language_score(
    exclusion: NonExecutableFeedback,
    row: NewsFeedbackEvent,
    dataset: NewsEvaluationDataset,
) -> ProjectedFeedbackScore | ExcludedFeedbackScore:
    rating = row.rating
    if rating == "positive":
        polarity = "positive"
    elif rating == "negative":
        polarity = "negative"
    else:
        raise ValueError(f"Language feedback has no reviewed rating: {exclusion.feedback_id}")
    if row.target.report_version_id != exclusion.report_version_id:
        raise ValueError(f"Language feedback report differs: {exclusion.feedback_id}")
    snapshot = _dataset_report(dataset, exclusion.report_version_id)
    theme_set = _read_theme_set(snapshot)
    if isinstance(theme_set.construction, ModelThemeConstruction):
        return ExcludedFeedbackScore(
            feedback_id=exclusion.feedback_id,
            reason="no_concern_specific_observation",
            concern="language",
            report_version_id=exclusion.report_version_id,
            polarity=polarity,
            rationale=(
                f"{exclusion.rationale or 'Reviewed explicit language feedback.'} "
                "The frozen theme output uses one combined assignment and prose observation."
            ),
        )
    _theme_set, route = _theme_observation(snapshot, "prose")
    return ProjectedFeedbackScore(
        feedback_id=exclusion.feedback_id,
        concern="language",
        report_version_id=exclusion.report_version_id,
        model_output=route.output,
        model_attempt_id=route.attempt_id,
        polarity=polarity,
        rationale=exclusion.rationale or "Reviewed explicit language feedback.",
    )


def _score_exclusion(
    exclusion: ExcludedFeedback | NonExecutableFeedback,
) -> ExcludedFeedbackScore:
    if isinstance(exclusion, ExcludedFeedback):
        reason = "superseded"
        rationale = f"Superseded by feedback {exclusion.superseded_by_feedback_id}."
    else:
        if exclusion.concern not in {
            "presentation",
            "operations",
            "ambiguous",
            "contradictory",
        }:
            raise ValueError(f"Unsupported score exclusion: {exclusion.feedback_id}")
        reason = exclusion.concern
        rationale = exclusion.rationale or f"Reviewed {reason} feedback is not model-owned."
    return ExcludedFeedbackScore.model_validate(
        {
            "feedback_id": exclusion.feedback_id,
            "reason": reason,
            "report_version_id": exclusion.report_version_id,
            "rationale": rationale,
        },
        strict=True,
    )


def _case_report_version(case: NewsEvaluationCase) -> Sha256:
    if case.provenance.report is None:
        raise ValueError(f"Evaluation case has no frozen report: {case.case_id}")
    return case.provenance.report.version_id


def _result_polarity(passed: bool) -> Literal["positive", "negative"]:
    return "positive" if passed else "negative"


def _dataset_report(
    dataset: NewsEvaluationDataset, report_version_id: Sha256
) -> ReportEvaluationSnapshot:
    matches = tuple(
        report for report in dataset.reports if report.report.version_id == report_version_id
    )
    if len(matches) != 1:
        raise ValueError(f"Dataset has no exact report snapshot: {report_version_id}")
    return matches[0]


def _model_output_observation(reference: ArtifactReference) -> _ModelObservationRoute:
    payload = _ModelOutputRoutePayload.model_validate_json(
        read_evaluation_artifact(reference), strict=True
    )
    return _ModelObservationRoute(
        output=reference,
        attempt_id=_resolve_accepted_attempt(
            payload.request_id,
            payload.accepted_response_id,
        ),
    )


def _theme_observation(
    snapshot: ReportEvaluationSnapshot,
    stage_name: str,
) -> tuple[SparseDailyThemeSet | ReaderSubjectDailyThemeSet, _ModelObservationRoute]:
    output = snapshot.themes
    if output is None:
        raise ValueError(f"Report has no exact theme output: {snapshot.report.version_id}")
    theme_set = _read_theme_set(snapshot)
    if not isinstance(
        theme_set, SparseDailyThemeSet | ReaderSubjectDailyThemeSet
    ) or not isinstance(theme_set.construction, SparseThemeConstruction):
        raise ValueError(f"Theme output has no exact sparse stages: {output.version_id}")
    if stage_name == "assignment":
        stage = theme_set.construction.assignment
    elif stage_name == "prose":
        stage = theme_set.construction.merged_prose
        if stage is None:
            raise ValueError(f"Theme output has no prose stage: {output.version_id}")
    else:
        raise ValueError(f"Unsupported theme stage: {stage_name}")
    return theme_set, _theme_stage_observation(output, stage)


def _read_theme_set(
    snapshot: ReportEvaluationSnapshot,
) -> (
    DailyThemeSet | SparseDailyThemeSet | ReaderSubjectDailyThemeSet | AliasedReaderSubjectThemeSet
):
    if snapshot.themes is None:
        raise ValueError(f"Report has no exact theme output: {snapshot.report.version_id}")
    return parse_daily_theme_set(read_evaluation_artifact(snapshot.themes))


def _theme_stage_observation(
    output: ArtifactReference,
    stage: ThemeStageEvidence,
) -> _ModelObservationRoute:
    accepted = tuple(
        attempt
        for attempt in stage.attempts
        if attempt.status == "accepted" and attempt.response_id == stage.call.response_id
    )
    if len(accepted) != 1:
        raise ValueError(f"Theme stage has no exact accepted attempt: {stage.request_id}")
    attempt_id = _resolve_accepted_attempt(stage.request_id, stage.call.response_id)
    if attempt_id != accepted[0].attempt_id:
        raise ValueError(f"Theme stage attempt differs from the catalog: {stage.request_id}")
    return _ModelObservationRoute(output=output, attempt_id=attempt_id)


def _resolve_accepted_attempt(request_id: Sha256, response_id: str) -> Sha256:
    return evaluation_catalog.read_news_model_attempt_id(request_id, response_id)


def _frozen_theme_expectation(
    theme_set: (
        DailyThemeSet
        | SparseDailyThemeSet
        | ReaderSubjectDailyThemeSet
        | AliasedReaderSubjectThemeSet
    ),
    expectation: ThemePairExpectation,
) -> tuple[bool, str]:
    theme_by_group = {
        group_id: theme.id for theme in theme_set.themes for group_id in theme.group_ids
    }
    if (
        expectation.left_group_id not in theme_by_group
        or expectation.right_group_id not in theme_by_group
    ):
        raise ValueError(f"Theme expectation groups are absent: {expectation.case_id}")
    observed_same = (
        theme_by_group[expectation.left_group_id] == theme_by_group[expectation.right_group_id]
    )
    passed = observed_same == expectation.expected_same_theme
    return passed, (
        f"expected same_theme={expectation.expected_same_theme}, "
        f"frozen output same_theme={observed_same}"
    )


def _corrected_grouping_spec(
    case: NewsEvaluationSpec,
    spec: GroupingSpec,
    cluster_set: DailyClusterSet,
    articles: dict[Sha256, EmbeddedArticleReference],
) -> GroupingEvaluationSpec:
    if not isinstance(case, GroupingEvaluationSpec) or spec.right_group_id is None:
        raise ValueError(f"Correction target is not a cross-group case: {case.case_id}")
    groups = {group.id: group for group in cluster_set.groups}
    left = groups[spec.left_group_id]
    right = groups[spec.right_group_id]
    selected = (*left.article_version_ids, *right.article_version_ids)
    return case.model_copy(
        update={
            "articles": tuple(articles[article_id] for article_id in selected),
            "left_article_version_id": left.article_version_ids[0],
            "right_article_version_id": right.article_version_ids[0],
        }
    )


def _build_hydrated_dataset() -> NewsEvaluationDataset:
    prior = _load_prior_dataset()
    bundles = {
        report_version_id: read_evaluation_report_bundle(report_version_id)
        for report_version_id in REPORT_VERSION_IDS
    }
    feedback = evaluation_catalog.read_news_evaluation_feedback(SOURCE_FEEDBACK_IDS)
    source_feedback_ids = tuple(row.feedback_id for row in feedback)
    _require_exact_feedback_source(feedback, bundles)
    prior_cases = _without_feedback(prior.cases)
    relevance_cases = tuple(
        case
        for spec in RELEVANCE_SPECS
        for case in build_relevance_cases(spec, bundles[spec.report_version_id])
    )
    reports = (*prior.reports, *(bundle.snapshot for bundle in bundles.values()))
    theme_cases = build_theme_day_cases(
        reports,
        (*prior.archived_daily_theme_judgments, *THEME_JUDGMENTS),
        frozenset(source_feedback_ids),
    )
    executable_cases = (
        *prior_cases,
        *relevance_cases,
        *(_grouping_case(spec, bundles[spec.report_version_id]) for spec in GROUPING_SPECS),
        *(build_ranking_case(spec, bundles[spec.report_version_id]) for spec in RANKING_SPECS),
        *theme_cases,
    )
    return NewsEvaluationDataset(
        version=DATASET_VERSION,
        reviewed_at=REVIEWED_AT,
        issue_url=ISSUE_URL,
        source_feedback_ids=source_feedback_ids,
        reports=reports,
        cases=executable_cases,
        archived_daily_theme_judgments=(),
        excluded_feedback=EXCLUSIONS,
    )


def _without_feedback(
    cases: tuple[NewsEvaluationCase, ...],
) -> tuple[NewsEvaluationCase, ...]:
    return tuple(
        case.model_copy(
            update={"provenance": case.provenance.model_copy(update={"feedback_ids": ()})}
        )
        for case in cases
    )


def _load_prior_dataset() -> NewsEvaluationDataset:
    manifest = NewsEvaluationManifest.model_validate_json(
        read_evaluation_artifact(SOURCE_MANIFEST_REFERENCE), strict=True
    )
    return evaluation_catalog.hydrate_news_evaluation_manifest(manifest)


def _manifest_from_dataset(
    dataset: NewsEvaluationDataset,
    *,
    prior_manifest: ArtifactReference | None = None,
) -> NewsEvaluationManifest:
    return NewsEvaluationManifest(
        version=dataset.version,
        reviewed_at=dataset.reviewed_at,
        issue_url=dataset.issue_url,
        prior_manifest=prior_manifest,
        source_feedback_ids=dataset.source_feedback_ids,
        reports=tuple(
            ReportEvaluationSpec(
                report=report.report,
                themes=report.themes,
                cluster_set=report.cluster_set,
            )
            for report in dataset.reports
        ),
        cases=tuple(evaluation_case_spec(case) for case in dataset.cases),
        archived_daily_theme_judgments=dataset.archived_daily_theme_judgments,
        excluded_feedback=dataset.excluded_feedback,
    )


def evaluation_case_spec(case: NewsEvaluationCase) -> NewsEvaluationSpec:
    provenance = EvaluationSpecProvenance(
        feedback_ids=case.provenance.feedback_ids,
        report_version_id=(
            case.provenance.report.version_id if case.provenance.report is not None else None
        ),
        group_id=case.provenance.group_id,
    )
    if case.concern == "daily_theme":
        return ThemeEvaluationDaySpec(
            case_id=case.case_id,
            control=case.control,
            provenance=provenance,
            day=case.input.day,
            source_report=case.source_report,
            cluster_set=case.input.cluster_set,
            summaries=tuple(item.summary for item in case.input.groups),
            expectations=case.expectations,
            report_expectation=case.report_expectation,
            model_output=case.model_output,
        )
    if case.concern == "relevance":
        article = _require_one_reference(case.provenance.articles, case.case_id, "article")
        model_output = _require_one_reference(
            case.provenance.model_outputs, case.case_id, "model output"
        )
        return RelevanceEvaluationSpec(
            case_id=case.case_id,
            control=case.control,
            provenance=provenance,
            expected_accepted=case.expected_accepted,
            article=article,
            model_output=model_output,
        )
    if case.concern == "summary_format":
        return SummaryFormatEvaluationSpec(
            case_id=case.case_id,
            control=case.control,
            provenance=provenance,
            articles=case.provenance.articles,
            model_output=_require_one_reference(
                case.provenance.model_outputs, case.case_id, "model output"
            ),
        )
    if case.concern == "grouping":
        return GroupingEvaluationSpec(
            case_id=case.case_id,
            control=case.control,
            provenance=provenance,
            source_cluster_set=case.source_cluster_set,
            day=case.day,
            articles=tuple(
                EmbeddedArticleReference(
                    article=article.article,
                    relevance=article.relevance,
                    embedding=article.embedding,
                )
                for article in case.articles
            ),
            left_article_version_id=case.left_article_version_id,
            right_article_version_id=case.right_article_version_id,
            expected_same_group=case.expected_same_group,
        )
    if case.concern == "ranking":
        references = _ranking_references(case)
        return RankingEvaluationSpec(
            case_id=case.case_id,
            control=case.control,
            provenance=provenance,
            relevance=tuple(
                ArticleRelevanceSpec(
                    article=article,
                    model_output=references[article.version_id],
                )
                for article in case.provenance.articles
            ),
            higher_group_id=case.higher_group_id,
            lower_group_id=case.lower_group_id,
        )
    if case.concern == "tier":
        return TierEvaluationSpec(
            case_id=case.case_id,
            control=case.control,
            provenance=provenance,
            expected_tier=case.expected_tier,
        )
    if case.concern == "confidence":
        return ConfidenceEvaluationSpec(
            case_id=case.case_id,
            control=case.control,
            provenance=provenance,
            expected_sufficient=case.expected_sufficient,
        )
    return ReaderPresentationEvaluationSpec(
        case_id=case.case_id,
        control=case.control,
        provenance=provenance,
        expectation=case.expectation,
    )


def _require_one_reference(
    references: tuple[ArtifactReference, ...], case_id: str, role: str
) -> ArtifactReference:
    if len(references) != 1:
        raise ValueError(f"{case_id} must have exactly one {role} reference")
    return references[0]


def _ranking_references(case: RankingEvaluationCase) -> dict[Sha256, ArtifactReference]:
    if len(case.provenance.articles) != len(case.provenance.model_outputs):
        raise ValueError(f"{case.case_id} ranking provenance is incomplete")
    references = {
        article.version_id: model_output
        for article, model_output in zip(
            case.provenance.articles, case.provenance.model_outputs, strict=True
        )
    }
    expected = {item.article_version_id for item in case.relevance}
    if set(references) != expected:
        raise ValueError(f"{case.case_id} ranking provenance does not match its inputs")
    return references


def _require_exact_feedback_source(
    feedback: tuple[NewsFeedbackEvent, ...],
    bundles: dict[Sha256, ReportBundle],
) -> None:
    assignments = _require_feedback_assignment_coverage()
    rows = _require_returned_feedback_rows(feedback, bundles, assignments)
    _require_feedback_target_membership(rows, bundles)
    _require_declared_feedback_targets(rows)
    _require_feedback_exclusions(rows)


def _require_feedback_assignment_coverage() -> FeedbackAssignments:
    assignments = FeedbackAssignments.from_feedback_ids(
        SOURCE_FEEDBACK_IDS,
        _REPRESENTED_FEEDBACK_IDS,
        EXCLUSIONS,
    )
    assignment_identity = (
        len(SOURCE_FEEDBACK_IDS),
        len(assignments.source_ids),
        assignments.represented_ids | assignments.excluded_ids,
    )
    if assignment_identity != (30, 30, assignments.source_ids):
        raise ValueError("Curation assignments must exactly cover the reviewed source IDs")
    return assignments


def _require_returned_feedback_rows(
    feedback: tuple[NewsFeedbackEvent, ...],
    bundles: dict[Sha256, ReportBundle],
    assignments: FeedbackAssignments,
) -> dict[UUID, NewsFeedbackEvent]:
    feedback_ids = tuple(row.feedback_id for row in feedback)
    feedback_identity = (len(feedback_ids), frozenset(feedback_ids))
    if feedback_identity != (30, assignments.source_ids):
        raise ValueError("PostgreSQL did not return the exact reviewed 30-event source")
    counts = {
        report_version_id: sum(
            row.target.report_version_id == report_version_id for row in feedback
        )
        for report_version_id in bundles
    }
    expected_counts = {
        expectation.report_version_id: expectation.feedback_count
        for expectation in _REVIEWED_REPORT_EXPECTATIONS
    }
    if counts != expected_counts:
        raise ValueError("Production feedback must contain 11 and 19 exact report events")
    return {row.feedback_id: row for row in feedback}


def _require_feedback_target_membership(
    rows: dict[UUID, NewsFeedbackEvent],
    bundles: dict[Sha256, ReportBundle],
) -> None:
    for row in rows.values():
        target = row.target
        bundle = bundles.get(target.report_version_id)
        if bundle is None:
            raise ValueError("Reviewed feedback must reference a frozen report")
        if target.kind == "report":
            continue
        if target.kind == "theme":
            if (
                isinstance(bundle.report, ArchivedDailyReport)
                or sum(section.theme_id == target.theme_id for section in bundle.report.sections)
                != 1
            ):
                raise ValueError("Reviewed feedback must target a frozen report theme")
            continue
        if target.kind == "group":
            _ = _section(bundle.report, target.group_id)
            continue
        raise ValueError("V6 feedback must target a report, theme, or group")


def _require_declared_feedback_targets(rows: dict[UUID, NewsFeedbackEvent]) -> None:
    for expectation in _FEEDBACK_TARGET_EXPECTATIONS:
        _require_feedback_target(rows, expectation)


def _require_feedback_exclusions(rows: dict[UUID, NewsFeedbackEvent]) -> None:
    for exclusion in EXCLUSIONS:
        row = rows[exclusion.feedback_id]
        if row.target.report_version_id != exclusion.report_version_id:
            raise ValueError("Feedback exclusion does not match its raw report")
        if exclusion.reason == "superseded":
            _require_supersession(rows, exclusion)


def _require_supersession(rows: dict[UUID, NewsFeedbackEvent], exclusion: ExcludedFeedback) -> None:
    old = rows[exclusion.feedback_id]
    replacement = rows[exclusion.superseded_by_feedback_id]
    if old.target != replacement.target:
        raise ValueError("Superseding feedback must target the same exact report target")
    old_order = (old.created_at, str(old.feedback_id))
    replacement_order = (replacement.created_at, str(replacement.feedback_id))
    if replacement_order <= old_order:
        raise ValueError("Superseding feedback must be newer by reader ordering")


def _require_feedback_target(
    rows: dict[UUID, NewsFeedbackEvent],
    expectation: _FeedbackTargetExpectation,
) -> None:
    if rows[expectation.feedback_id].target != expectation.target:
        raise ValueError(
            f"Curation spec does not match raw feedback target: {expectation.feedback_id}"
        )


def read_evaluation_report_bundle(report_version_id: Sha256) -> ReportBundle:
    lineage = evaluation_catalog.read_news_evaluation_report_lineage(report_version_id)
    report_reference = lineage.report
    report = parse_daily_report(read_evaluation_artifact(report_reference))
    inputs = {
        "themes": list(lineage.inputs.themes),
        "assessments": list(lineage.inputs.assessments),
        "cluster_set": list(lineage.inputs.cluster_set),
        "relevance": list(lineage.inputs.relevance),
        "summary": list(lineage.inputs.summary),
        "sentiment": list(lineage.inputs.sentiment),
    }
    cluster_reference = inputs["cluster_set"][0]
    cluster_set = DailyClusterSet.model_validate_json(
        read_evaluation_artifact(cluster_reference), strict=True
    )
    _require_report_matches_cluster(report, cluster_set)
    themes_reference = _report_themes_reference(report, inputs, cluster_reference)
    _require_report_matches_assessment(report, inputs, themes_reference)
    references = evaluation_catalog.read_news_evaluation_artifact_references(
        (
            *cluster_set.article_version_ids,
            *cluster_set.relevance_version_ids,
            *cluster_set.embedding_version_ids,
        )
    )
    articles = {
        article_id: EmbeddedArticleReference(
            article=references[article_id],
            relevance=references[relevance_id],
            embedding=references[embedding_id],
        )
        for article_id, relevance_id, embedding_id in zip(
            cluster_set.article_version_ids,
            cluster_set.relevance_version_ids,
            cluster_set.embedding_version_ids,
            strict=True,
        )
    }
    group_inputs = _report_group_inputs(
        cluster_set.groups,
        tuple(inputs["summary"]),
        tuple(inputs["sentiment"]),
    )
    events = report_section_events(report)
    snapshot = ReportEvaluationSnapshot(
        report=report_reference,
        themes=themes_reference,
        cluster_set=cluster_reference,
        report_article_version_ids=tuple(
            article.article_version_id for event in events for article in event.articles
        ),
        report_group_ids=tuple(event.group_id for event in events),
        cluster_articles=tuple(
            articles[article_id] for article_id in cluster_set.article_version_ids
        ),
        group_inputs=group_inputs,
        report_inputs=ReportInputReferences(
            themes=tuple(inputs["themes"]),
            assessments=tuple(inputs["assessments"]),
            cluster_set=tuple(inputs["cluster_set"]),
            relevance=tuple(inputs["relevance"]),
            summary=tuple(inputs["summary"]),
            sentiment=tuple(inputs["sentiment"]),
        ),
        group_observations=tuple(
            ReportGroupObservation(
                group_id=event.group_id,
                uncertainty_disclosed=event.uncertainty_ro is not None,
                tier=_report_group_tiers(report)[event.group_id],
            )
            for event in events
        ),
        report_theme_ids=(
            ()
            if isinstance(report, ArchivedDailyReport)
            else tuple(section.theme_id for section in report.sections for _event in section.events)
        ),
    )
    relevance = {
        article_id: references[relevance_id]
        for article_id, relevance_id in zip(
            cluster_set.article_version_ids, cluster_set.relevance_version_ids, strict=True
        )
    }
    return ReportBundle(
        snapshot=snapshot,
        report=report,
        cluster_set=cluster_set,
        articles=articles,
        relevance=relevance,
    )


def _report_themes_reference(
    report: DailyReportDocument,
    inputs: dict[str, list[ArtifactReference]],
    cluster_reference: ArtifactReference,
) -> ArtifactReference | None:
    if isinstance(report, ArchivedDailyReport):
        if inputs["themes"]:
            raise ValueError("Archived report cannot have a themes input")
        return None
    if len(inputs["themes"]) != 1:
        raise ValueError("Current report must have one themes input")
    reference = inputs["themes"][0]
    theme_set = parse_daily_theme_set(read_evaluation_artifact(reference))
    if theme_set.day != report.day or theme_set.cluster_set != cluster_reference:
        raise ValueError("Exact report themes belong to another report input")
    if theme_set.summary_inputs != tuple(inputs["summary"]):
        raise ValueError("Exact report themes do not match report summaries")
    report_themes = {section.theme_id: section for section in report.sections}
    source_themes = {theme.id: theme for theme in theme_set.themes}
    if (
        len(report_themes) != len(report.sections)
        or len(source_themes) != len(theme_set.themes)
        or set(report_themes) != set(source_themes)
    ):
        raise ValueError("Exact report themes do not match the theme set")
    for theme_id, section in report_themes.items():
        theme = source_themes[theme_id]
        if (
            section.title != theme.title
            or section.summary != theme.summary
            or set(event.group_id for event in section.events) != set(theme.group_ids)
        ):
            raise ValueError(f"Exact report theme differs from theme input: {theme_id}")
    return reference


def _require_report_matches_assessment(
    report: DailyReportDocument,
    inputs: dict[str, list[ArtifactReference]],
    themes: ArtifactReference | None,
) -> None:
    if not isinstance(report, DailyReport):
        return
    if len(inputs["assessments"]) != 1 or themes is None:
        raise ValueError("Schema-v3 report must have one assessment input")
    assessment_set = parse_daily_subject_assessment_set(
        read_evaluation_artifact(inputs["assessments"][0])
    )
    if assessment_set.themes != themes:
        raise ValueError("Report assessment input references another theme set")
    assessments = {item.theme_id: item for item in assessment_set.assessments}
    for section in report.sections:
        assessment = assessments.get(section.theme_id)
        if assessment is None or (
            section.tier,
            section.semantic_rank,
            section.consequence_rationale,
            tuple(
                (citation.article_version_id, citation.evidence_quote)
                for citation in section.citations
            ),
        ) != (
            assessment.tier,
            assessment.semantic_rank,
            assessment.rationale,
            tuple(
                (evidence.article.version_id, evidence.evidence_quote)
                for evidence in assessment.evidence
            ),
        ):
            raise ValueError("Schema-v3 report differs from its assessment input")


def _report_group_tiers(report: DailyReportDocument) -> dict[Sha256, Tier]:
    if isinstance(report, ArchivedDailyReport):
        return {section.group_id: "main" for section in report.sections}
    return {
        event.group_id: section.tier if isinstance(section, DailyReportSection) else "main"
        for section in report.sections
        for event in section.events
    }


def _require_report_matches_cluster(
    report: DailyReportDocument,
    cluster_set: DailyClusterSet,
) -> None:
    events = (
        report.sections
        if isinstance(report, ArchivedDailyReport)
        else tuple(event for section in report.sections for event in section.events)
    )
    report_groups = {event.group_id: event for event in events}
    cluster_groups = {group.id: group for group in cluster_set.groups}
    if len(report_groups) != len(events) or len(cluster_groups) != len(cluster_set.groups):
        raise ValueError("Exact report or cluster set has duplicate groups")
    if set(report_groups) != set(cluster_groups):
        raise ValueError("Exact report groups do not match the cluster set")
    for group_id, event in report_groups.items():
        report_articles = tuple(article.article_version_id for article in event.articles)
        cluster_articles = cluster_groups[group_id].article_version_ids
        if (
            len(report_articles) != len(set(report_articles))
            or len(cluster_articles) != len(set(cluster_articles))
            or set(report_articles) != set(cluster_articles)
        ):
            raise ValueError(f"Exact report group articles do not match cluster group {group_id}")


def _report_group_inputs(
    groups: tuple[NewsGroup, ...],
    summaries: tuple[ArtifactReference, ...],
    sentiments: tuple[ArtifactReference, ...],
) -> tuple[ReportGroupInputs, ...]:
    summary_by_group = _references_by_group("summary", summaries)
    sentiment_by_group = _references_by_group("sentiment", sentiments)
    expected_group_ids = {group.id for group in groups}
    if set(summary_by_group) != expected_group_ids or set(sentiment_by_group) != expected_group_ids:
        raise ValueError("Exact report group inputs do not cover the cluster groups")
    return tuple(
        ReportGroupInputs(
            group_id=group_id,
            summary=summary_by_group[group_id],
            sentiment=sentiment_by_group[group_id],
        )
        for group_id in sorted(expected_group_ids)
    )


def _references_by_group(
    role: str, references: tuple[ArtifactReference, ...]
) -> dict[Sha256, ArtifactReference]:
    values = {}
    for reference in references:
        payload = json.loads(read_evaluation_artifact(reference))
        group_id = _SHA256_ADAPTER.validate_python(payload.get("group_id"), strict=True)
        if group_id in values:
            raise ValueError(f"Exact report has duplicate {role} inputs for group {group_id}")
        values[group_id] = reference
    return values


def build_relevance_cases(
    spec: RelevanceSpec, bundle: ReportBundle
) -> tuple[RelevanceEvaluationCase, ...]:
    section = _section(bundle.report, spec.group_id)
    if spec.case_ids and len(spec.case_ids) != len(section.articles):
        raise ValueError("Relevance case IDs must exactly cover the reviewed subject articles")
    cases = []
    for position, article in enumerate(section.articles, start=1):
        article_reference = bundle.articles[article.article_version_id].article
        relevance_reference = bundle.relevance[article.article_version_id]
        decision = evaluation_catalog.read_evaluation_relevance_observation(
            relevance_reference, article.article_version_id
        )
        cases.append(
            RelevanceEvaluationCase(
                case_id=(
                    spec.case_ids[position - 1]
                    if spec.case_ids
                    else (
                        f"{bundle.report.day.isoformat()}-{spec.group_id[:12]}-article-{position}-"
                        f"relevance-{relevance_reference.version_id[:12]}"
                    )
                ),
                control=spec.control,
                provenance=EvaluationProvenance(
                    feedback_ids=(UUID(spec.feedback_id),),
                    report=bundle.snapshot.report,
                    group_id=spec.group_id,
                    articles=(article_reference,),
                    model_outputs=(relevance_reference,),
                ),
                expected_accepted=spec.expected_accepted,
                model_response=decision,
            )
        )
    return tuple(cases)


def _grouping_case(spec: GroupingSpec, bundle: ReportBundle) -> GroupingEvaluationCase:
    left_section = _section(bundle.report, spec.left_group_id)
    if spec.right_group_id is None:
        selected = left_section.articles
        if len(selected) < 2:
            raise ValueError(f"Same-group control needs two articles: {spec.case_id}")
        right_article = selected[1]
        expected_same_group = spec.expected_same_group
        if not expected_same_group:
            raise ValueError(f"One-group cases must expect one group: {spec.case_id}")
    else:
        right_section = _section(bundle.report, spec.right_group_id)
        selected = (*left_section.articles, *right_section.articles)
        right_article = right_section.articles[0]
        expected_same_group = spec.expected_same_group
    articles = tuple(_load_embedded(bundle.articles[item.article_version_id]) for item in selected)
    return GroupingEvaluationCase(
        case_id=spec.case_id,
        control=spec.control,
        provenance=EvaluationProvenance(
            feedback_ids=(UUID(spec.feedback_id),),
            report=bundle.snapshot.report,
            group_id=spec.left_group_id,
            articles=tuple(item.article for item in articles),
            model_outputs=tuple(
                reference for item in articles for reference in (item.relevance, item.embedding)
            ),
        ),
        source_cluster_set=bundle.snapshot.cluster_set,
        day=bundle.report.day,
        articles=articles,
        left_article_version_id=selected[0].article_version_id,
        right_article_version_id=right_article.article_version_id,
        expected_same_group=expected_same_group,
    )


def build_theme_day_cases(
    snapshots: tuple[ReportEvaluationSnapshot, ...],
    judgments: tuple[DailyThemeJudgment, ...],
    source_feedback_ids: frozenset[UUID],
    model_outputs: Mapping[Sha256, ArtifactReference] | None = None,
) -> tuple[ThemeEvaluationDayCase, ...]:
    judgments_by_report: dict[Sha256, list[DailyThemeJudgment]] = {}
    for judgment in judgments:
        judgments_by_report.setdefault(judgment.report_version_id, []).append(judgment)
    snapshot_ids = {snapshot.report.version_id for snapshot in snapshots}
    if set(judgments_by_report) - snapshot_ids:
        raise ValueError("Theme judgments must reference a frozen report snapshot")
    return tuple(
        _theme_day_case(
            snapshot,
            tuple(judgments_by_report[snapshot.report.version_id]),
            source_feedback_ids,
            (model_outputs or {}).get(snapshot.report.version_id),
        )
        for snapshot in snapshots
        if snapshot.report.version_id in judgments_by_report
    )


def _theme_day_case(
    snapshot: ReportEvaluationSnapshot,
    judgments: tuple[DailyThemeJudgment, ...],
    source_feedback_ids: frozenset[UUID],
    model_output: ArtifactReference | None,
) -> ThemeEvaluationDayCase:
    group_inputs = {item.group_id: item.summary for item in snapshot.group_inputs}
    cluster_set = DailyClusterSet.model_validate_json(
        read_evaluation_artifact(snapshot.cluster_set), strict=True
    )
    summaries = tuple(group_inputs[group.id] for group in cluster_set.groups)
    value = load_daily_theme_input(snapshot.cluster_set, summaries)
    judged_group_ids = {
        group_id
        for judgment in judgments
        for group_id in (judgment.left_group_id, judgment.right_group_id)
    }
    unrelated = next(
        (item.group.id for item in value.groups if item.group.id not in judged_group_ids),
        None,
    )
    expectations = [
        ThemePairExpectation(
            case_id=judgment.judgment_id,
            feedback_ids=tuple(
                feedback_id
                for feedback_id in judgment.feedback_ids
                if feedback_id in source_feedback_ids
            ),
            left_group_id=judgment.left_group_id,
            right_group_id=judgment.right_group_id,
            expected_same_theme=judgment.expected_same_theme,
            control=not judgment.expected_same_theme,
            rationale=judgment.rationale,
        )
        for judgment in judgments
    ]
    if unrelated is not None:
        expectations.append(
            ThemePairExpectation(
                case_id=(
                    f"{value.day.isoformat()}-{snapshot.report.version_id[:12]}-"
                    "unrelated-theme-control"
                ),
                feedback_ids=(),
                left_group_id=judgments[0].left_group_id,
                right_group_id=unrelated,
                expected_same_theme=False,
                control=True,
                rationale="The control group covers another reviewed report subject.",
            )
        )
    feedback_ids = tuple(
        dict.fromkeys(
            feedback_id for expectation in expectations for feedback_id in expectation.feedback_ids
        )
    )
    if model_output is not None and model_output != snapshot.themes:
        raise ValueError("Theme evaluation output differs from the frozen report input")
    observed_themes = (
        parse_daily_theme_set(read_evaluation_artifact(model_output))
        if model_output is not None
        else None
    )
    return ThemeEvaluationDayCase(
        case_id=f"{value.day.isoformat()}-{snapshot.report.version_id[:12]}-daily-themes",
        provenance=EvaluationProvenance(
            feedback_ids=feedback_ids,
            report=snapshot.report,
        ),
        source_report=snapshot.report,
        input=value,
        expectations=tuple(expectations),
        model_output=model_output,
        observed_themes=observed_themes,
    )


def build_ranking_case(spec: RankingSpec, bundle: ReportBundle) -> RankingEvaluationCase:
    groups_by_id = {group.id: group for group in bundle.cluster_set.groups}
    groups = (groups_by_id[spec.higher_group_id], groups_by_id[spec.lower_group_id])
    article_ids = tuple(article_id for group in groups for article_id in group.article_version_ids)
    relevance = tuple(
        ArticleRelevanceInput(
            article_version_id=article_id,
            decision=evaluation_catalog.read_evaluation_ranking_decision(
                bundle.relevance[article_id], article_id
            ),
        )
        for article_id in article_ids
    )
    return RankingEvaluationCase(
        case_id=spec.case_id,
        control=spec.control,
        provenance=EvaluationProvenance(
            feedback_ids=(UUID(spec.feedback_id),),
            report=bundle.snapshot.report,
            articles=tuple(bundle.articles[article_id].article for article_id in article_ids),
            model_outputs=tuple(bundle.relevance[article_id] for article_id in article_ids),
        ),
        groups=groups,
        relevance=relevance,
        higher_group_id=spec.higher_group_id,
        lower_group_id=spec.lower_group_id,
        observed_group_order=bundle.snapshot.report_group_ids,
    )


def build_tier_case(spec: TierSpec, bundle: ReportBundle) -> TierEvaluationCase:
    _section(bundle.report, spec.group_id)
    observations = {item.group_id: item for item in bundle.snapshot.group_observations}
    observation = observations.get(spec.group_id)
    if observation is None:
        raise ValueError(f"Reviewed group has no report observation: {spec.group_id}")
    return TierEvaluationCase(
        case_id=spec.case_id,
        control=spec.control,
        provenance=EvaluationProvenance(
            feedback_ids=(UUID(spec.feedback_id),),
            report=bundle.snapshot.report,
            group_id=spec.group_id,
        ),
        expected_tier=spec.expected_tier,
        observed_tier=observation.tier,
    )


def build_confidence_case(
    spec: ConfidenceSpec,
    bundle: ReportBundle,
) -> ConfidenceEvaluationCase:
    observations = {item.group_id: item for item in bundle.snapshot.group_observations}
    observation = observations.get(spec.group_id)
    if observation is None:
        raise ValueError(f"Reviewed group has no report observation: {spec.group_id}")
    return ConfidenceEvaluationCase(
        case_id=spec.case_id,
        control=spec.control,
        provenance=EvaluationProvenance(
            feedback_ids=(UUID(spec.feedback_id),),
            report=bundle.snapshot.report,
            group_id=spec.group_id,
        ),
        expected_sufficient=spec.expected_sufficient,
        observed_sufficient=not observation.uncertainty_disclosed,
    )


def _section(report: DailyReportDocument, group_id: Sha256):
    events = (
        report.sections
        if isinstance(report, ArchivedDailyReport)
        else tuple(event for section in report.sections for event in section.events)
    )
    selected = tuple(event for event in events if event.group_id == group_id)
    if len(selected) != 1:
        raise ValueError(f"Reviewed group is absent from exact report: {group_id}")
    return selected[0]


def read_evaluation_artifact(reference: ArtifactReference) -> bytes:
    """Read one artifact payload, cached by its exact content identity."""
    return _read_verified((reference.version_id, reference.content_digest, reference.r2_key))


@cache
def _read_verified(identity: tuple[Sha256, Sha256, str]) -> bytes:
    _version_id, content_digest, r2_key = identity
    return read_verified_r2_object(r2_key, content_digest)


def _load_embedded(reference: EmbeddedArticleReference) -> EmbeddedArticle:
    article = ExtractedArticle.model_validate_json(
        read_evaluation_artifact(reference.article), strict=True
    )
    embedding_payload = json.loads(read_evaluation_artifact(reference.embedding))
    if embedding_payload.get("article_version_id") != reference.article.version_id:
        raise ValueError("Embedding payload references another article version")
    if embedding_payload.get("relevance_version_id") != reference.relevance.version_id:
        raise ValueError("Embedding payload references another relevance version")
    return EmbeddedArticle(
        article=reference.article,
        relevance=reference.relevance,
        embedding=reference.embedding,
        value=article,
        vector=tuple(embedding_payload["vector"]),
    )


def write_news_evaluation_manifest(path: Path = DEFAULT_OUTPUT_PATH) -> NewsEvaluationManifest:
    """Write deterministic reference-only JSON for the reviewed production inputs."""
    manifest = build_news_evaluation_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_news_feedback_score_manifest(
    path: Path = SCORE_DEFAULT_OUTPUT_PATH,
) -> NewsEvaluationManifest:
    """Write deterministic v6.3 score curation without any production mutation."""
    manifest = build_news_feedback_score_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-curation", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.score_curation:
        write_news_feedback_score_manifest(arguments.output or SCORE_DEFAULT_OUTPUT_PATH)
    else:
        write_news_evaluation_manifest(arguments.output or DEFAULT_OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
