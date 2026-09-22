from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.binary_ranking import (
    RankingArticleEvidence,
    RankingGroupEvidence,
    RankingGroupPair,
    build_ranking_binary_request,
    canonical_ranking_groups,
)
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.binary_benchmark import BinaryBenchmarkCase, BinaryJudgmentCase
from romanian_news.catalog.evaluations import load_news_evaluation_release
from romanian_news.derived_binary_protocol import DERIVED_BINARY_BENCHMARKS
from romanian_news.evaluation import (
    NewsEvaluationPin,
    RankingEvaluationCase,
    RankingEvaluationSpec,
)
from romanian_news.storage import read_verified_r2_object

V11_SOURCE_ARTIFACT_ID = "083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"
V11_BASELINE_ARTIFACT_ID = "78f4738f918280d467b7cba6f5e64f09f76cf5f710817640a06f7ae879cea77d"
V11_MANIFEST_VERSION = "news-evaluation-2026-09-11-v11"


class BinaryRankingSourceCase(NewsModel):
    case_id: str
    control: bool
    groups: RankingGroupPair
    higher_group_id: Sha256
    lower_group_id: Sha256

    @model_validator(mode="after")
    def require_exact_labeled_pair(self) -> BinaryRankingSourceCase:
        group_ids = {item.group.id for item in self.groups}
        if len(group_ids) != 2 or group_ids != {self.higher_group_id, self.lower_group_id}:
            raise ValueError("Binary ranking labels must exactly cover two frozen groups")
        return self


class BinaryRankingSource(NewsModel):
    pin: NewsEvaluationPin
    manifest_reference: ArtifactReference
    baseline_reference: ArtifactReference
    declared_manifest_version: Literal["news-evaluation-2026-09-11-v11"]
    cases: tuple[BinaryRankingSourceCase, ...]

    @model_validator(mode="after")
    def require_exact_pinned_workload(self) -> BinaryRankingSource:
        definition = DERIVED_BINARY_BENCHMARKS["ranking"]
        if (
            self.pin.manifest_version_id != V11_SOURCE_ARTIFACT_ID
            or self.manifest_reference.version_id != V11_SOURCE_ARTIFACT_ID
        ):
            raise ValueError(f"Binary ranking source artifact must be {V11_SOURCE_ARTIFACT_ID}")
        if (
            self.pin.baseline_version_id != V11_BASELINE_ARTIFACT_ID
            or self.baseline_reference.version_id != V11_BASELINE_ARTIFACT_ID
        ):
            raise ValueError(f"Binary ranking baseline artifact must be {V11_BASELINE_ARTIFACT_ID}")
        if self.declared_manifest_version != V11_MANIFEST_VERSION:
            raise ValueError(f"Binary ranking manifest version must be {V11_MANIFEST_VERSION}")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(self.cases) != definition.case_count or len(set(case_ids)) != len(case_ids):
            raise ValueError(
                f"Binary ranking benchmark requires exactly {definition.case_count} unique cases"
            )
        return self


def load_v11_binary_ranking_source(pin_content: bytes | str) -> BinaryRankingSource:
    release = load_news_evaluation_release(pin_content)
    if release.manifest.version != V11_MANIFEST_VERSION:
        raise ValueError(f"Binary ranking manifest version must be {V11_MANIFEST_VERSION}")
    specs = tuple(
        case for case in release.manifest.cases if isinstance(case, RankingEvaluationSpec)
    )
    hydrated = tuple(
        case for case in release.dataset.cases if isinstance(case, RankingEvaluationCase)
    )
    if tuple(case.case_id for case in specs) != tuple(case.case_id for case in hydrated):
        raise ValueError("Binary ranking manifest and hydrated cases do not match")
    articles: dict[Sha256, ExtractedArticle] = {}
    return BinaryRankingSource(
        pin=release.pin,
        manifest_reference=release.manifest_reference,
        baseline_reference=release.baseline_reference,
        declared_manifest_version=V11_MANIFEST_VERSION,
        cases=tuple(
            _load_source_case(spec, case, articles)
            for spec, case in zip(specs, hydrated, strict=True)
        ),
    )


def build_ranking_benchmark_cases(
    source: BinaryRankingSource,
) -> tuple[BinaryBenchmarkCase, ...]:
    definition = DERIVED_BINARY_BENCHMARKS["ranking"]
    question = definition.questions[0]
    cases: list[BinaryBenchmarkCase] = []
    for source_case in source.cases:
        groups = canonical_ranking_groups(source_case.groups)
        expected = groups[0].group.id == source_case.higher_group_id
        request = build_ranking_binary_request(groups)
        cases.append(
            BinaryBenchmarkCase(
                case_id=source_case.case_id,
                identity=_case_identity(groups, expected),
                control=source_case.control,
                judgments=(
                    BinaryJudgmentCase(
                        judgment_id=question.question_id,
                        request=request,
                        expected=expected,
                    ),
                ),
            )
        )
    return tuple(cases)


def _load_source_case(
    spec: RankingEvaluationSpec,
    case: RankingEvaluationCase,
    articles: dict[Sha256, ExtractedArticle],
) -> BinaryRankingSourceCase:
    spec_by_article = {item.article.version_id: item for item in spec.relevance}
    decision_by_article = {item.article_version_id: item.decision for item in case.relevance}
    required = {article_id for group in case.groups for article_id in group.article_version_ids}
    if (
        set(spec_by_article) != required
        or set(decision_by_article) != required
        or len(spec_by_article) != len(spec.relevance)
        or len(decision_by_article) != len(case.relevance)
    ):
        raise ValueError("Binary ranking evidence does not exactly cover frozen group articles")
    groups = tuple(
        RankingGroupEvidence(
            group=group,
            articles=tuple(
                RankingArticleEvidence(
                    article=spec_by_article[article_id].article,
                    value=_cached_article(
                        spec_by_article[article_id].article,
                        articles,
                    ),
                    relevance=spec_by_article[article_id].model_output,
                    decision=decision_by_article[article_id],
                )
                for article_id in group.article_version_ids
            ),
        )
        for group in case.groups
    )
    if len(groups) != 2:
        raise ValueError("Binary ranking cases require exactly two frozen groups")
    return BinaryRankingSourceCase(
        case_id=case.case_id,
        control=case.control,
        groups=(groups[0], groups[1]),
        higher_group_id=case.higher_group_id,
        lower_group_id=case.lower_group_id,
    )


def _load_article(reference: ArtifactReference) -> ExtractedArticle:
    return ExtractedArticle.model_validate_json(
        read_verified_r2_object(reference.r2_key, reference.content_digest), strict=True
    )


def _cached_article(
    reference: ArtifactReference,
    articles: dict[Sha256, ExtractedArticle],
) -> ExtractedArticle:
    article = articles.get(reference.version_id)
    if article is None:
        article = _load_article(reference)
        articles[reference.version_id] = article
    return article


def _case_identity(groups: RankingGroupPair, expected: bool) -> tuple[str, ...]:
    values: list[str] = []
    for alias, group in zip(("A", "B"), groups, strict=True):
        values.append(f"group-{alias}:{group.group.id}")
        for article in sorted(group.articles, key=lambda item: item.article.version_id):
            values.extend(
                (
                    f"article-{alias}:{article.article.version_id}:{article.article.content_digest}",
                    f"relevance-{alias}:{article.relevance.version_id}:{article.relevance.content_digest}",
                )
            )
    values.append(f"expected-a-above-b:{str(expected).lower()}")
    return tuple(values)
