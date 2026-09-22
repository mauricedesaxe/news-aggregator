from __future__ import annotations

import json
from typing import Any, cast

from pydantic import model_validator

from romanian_news import NewsModel
from romanian_news.analysis.binary_benchmark import analyze_binary_benchmark
from romanian_news.analysis.binary_evaluation import BinaryRequest, binary_state_digest
from romanian_news.analysis.relevance import RelevanceDecision
from romanian_news.analysis.relevance_v3 import ImpactDecision
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.binary_benchmark import BinaryBenchmarkEvaluationResult
from romanian_news.derived_binary_protocol import DERIVED_BINARY_BENCHMARKS
from romanian_news.groups import NewsGroup


class RankingArticleEvidence(NewsModel):
    article: ArtifactReference
    value: ExtractedArticle
    relevance: ArtifactReference
    decision: RelevanceDecision | ImpactDecision


class RankingGroupEvidence(NewsModel):
    group: NewsGroup
    articles: tuple[RankingArticleEvidence, ...]

    @model_validator(mode="after")
    def require_exact_group_articles(self) -> RankingGroupEvidence:
        article_ids = tuple(item.article.version_id for item in self.articles)
        if len(set(article_ids)) != len(article_ids) or set(article_ids) != set(
            self.group.article_version_ids
        ):
            raise ValueError("Ranking evidence must exactly cover its frozen group articles")
        return self


RankingGroupPair = tuple[RankingGroupEvidence, RankingGroupEvidence]


def canonical_ranking_groups(groups: RankingGroupPair) -> RankingGroupPair:
    if groups[0].group.id == groups[1].group.id:
        raise ValueError("Ranking benchmark requires two distinct frozen groups")
    ordered = sorted(groups, key=lambda item: item.group.id)
    return ordered[0], ordered[1]


def build_ranking_binary_request(groups: RankingGroupPair) -> BinaryRequest:
    ordered = canonical_ranking_groups(groups)
    state = "\n\n".join(
        _render_group(alias, group) for alias, group in zip(("A", "B"), ordered, strict=True)
    )
    question = DERIVED_BINARY_BENCHMARKS["ranking"].questions[0]
    return BinaryRequest(
        question=question,
        state=state,
        state_digest=binary_state_digest(state),
    )


def analyze_binary_ranking(
    result: BinaryBenchmarkEvaluationResult,
) -> dict[str, Any]:
    definition = DERIVED_BINARY_BENCHMARKS["ranking"]
    report = analyze_binary_benchmark(definition, result)
    trials = cast(list[dict[str, Any]], report["trials"])
    report["trials"] = [
        {
            **trial,
            "inversion_count": int(trial["completed"]) - int(trial["correct"]),
        }
        for trial in trials
    ]
    return report


def _render_group(alias: str, evidence: RankingGroupEvidence) -> str:
    articles = sorted(evidence.articles, key=lambda item: item.article.version_id)
    return "\n\n".join(
        (
            f"Group {alias}\nGroup identity: {evidence.group.id}",
            *(_render_article(index, article) for index, article in enumerate(articles, start=1)),
        )
    )


def _render_article(index: int, evidence: RankingArticleEvidence) -> str:
    value = evidence.value
    decision = json.dumps(
        evidence.decision.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "\n".join(
        (
            f"Article {index}",
            f"Article identity: {evidence.article.version_id}",
            f"Published: {value.published_at.isoformat()}",
            f"Outlet: {value.outlet_id}",
            f"Title: {value.title}",
            "Body:",
            value.body,
            f"Relevance identity: {evidence.relevance.version_id}",
            "Relevance evidence:",
            decision,
        )
    )
