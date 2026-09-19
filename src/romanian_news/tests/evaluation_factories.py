from datetime import UTC, date, datetime
from uuid import UUID

from pydantic import HttpUrl

from romanian_news import EMBEDDING_DIMENSIONS
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.relevance import RelevanceDecision
from romanian_news.articles.models import ExtractedArticle
from romanian_news.evaluation import (
    ArticleRelevanceInput,
    ArticleRelevanceSpec,
    EvaluationProvenance,
    EvaluationSpecProvenance,
    FrozenGeneratedSummary,
    GroupingEvaluationCase,
    GroupingEvaluationSpec,
    KeyPointTextExpectation,
    NewsEvaluationDataset,
    NewsEvaluationManifest,
    RankingEvaluationCase,
    RankingEvaluationSpec,
    ReaderPresentationEvaluationCase,
    ReaderPresentationEvaluationSpec,
    RelevanceEvaluationCase,
    RelevanceEvaluationSpec,
    ReportEvaluationSnapshot,
    ReportEvaluationSpec,
    ReportGroupInputs,
    ReportInputReferences,
    SummaryFormatEvaluationCase,
    SummaryFormatEvaluationSpec,
)
from romanian_news.groups import EmbeddedArticle, EmbeddedArticleReference, NewsGroup

DAY = date(2026, 9, 4)
REVIEWED_AT = datetime(2026, 9, 4, tzinfo=UTC)


def reference(index: int, kind: str = "artifact") -> ArtifactReference:
    digest = f"{index:064x}"
    return ArtifactReference(
        artifact_id=f"news:{kind}:{index}",
        version_id=digest,
        content_digest=digest,
        r2_key=f"news/{kind}/{index}.json",
    )


def relevance_decision(*, accepted: bool, major: bool = False) -> RelevanceDecision:
    return RelevanceDecision(
        national_reach="nationwide" if accepted else "none",
        consequence_magnitude="major" if major else ("routine" if accepted else "narrow"),
        political_relevance="strong" if accepted else "none",
        economic_relevance="none",
        romania_relevance="strong" if accepted else "none",
        confidence=0.9,
        evidence_quote="Synthetic evidence",
        reason_ro="Caz sintetic.",
    )


def embedded_article(index: int, *, axis: int = 0) -> EmbeddedArticle:
    article = reference(index, "article")
    relevance = reference(index + 100, "relevance")
    embedding = reference(index + 200, "embedding")
    coordinates = [0.0] * EMBEDDING_DIMENSIONS
    coordinates[axis] = 1.0
    return EmbeddedArticle(
        article=article,
        relevance=relevance,
        embedding=embedding,
        value=ExtractedArticle(
            article_id=reference(index + 300).version_id,
            outlet_id="synthetic",
            canonical_url=HttpUrl(f"https://example.test/articles/{index}"),
            title=f"Synthetic article {index}",
            body="Synthetic body.",
            author=None,
            published_at=REVIEWED_AT,
            source_updated_at=None,
            bucharest_day=DAY,
            material_digest=reference(index + 400).version_id,
            extraction_digest=reference(index + 500).version_id,
        ),
        vector=tuple(coordinates),
    )


def synthetic_dataset() -> NewsEvaluationDataset:
    first = embedded_article(1)
    second = embedded_article(2)
    high_id = reference(10, "group").version_id
    low_id = reference(11, "group").version_id
    report = _complete_report_snapshot(first, second, high_id, low_id)
    cases = (
        RelevanceEvaluationCase(
            case_id="relevance-accepts",
            provenance=EvaluationProvenance(feedback_ids=()),
            expected_accepted=True,
            model_response=relevance_decision(accepted=True),
        ),
        SummaryFormatEvaluationCase(
            case_id="summary-format",
            provenance=EvaluationProvenance(feedback_ids=()),
            artifact=FrozenGeneratedSummary(
                title="Guvernul publică noul plan național",
                summary="Planul se aplică la nivel național.",
                key_points=("Măsura intră în vigoare.",),
            ),
        ),
        GroupingEvaluationCase(
            case_id="grouping-same",
            provenance=EvaluationProvenance(feedback_ids=()),
            source_cluster_set=report.cluster_set,
            day=DAY,
            articles=(first, second),
            left_article_version_id=first.article.version_id,
            right_article_version_id=second.article.version_id,
            expected_same_group=True,
        ),
        RankingEvaluationCase(
            case_id="ranking-major-first",
            provenance=EvaluationProvenance(feedback_ids=()),
            groups=(
                NewsGroup(id=high_id, article_version_ids=(first.article.version_id,)),
                NewsGroup(id=low_id, article_version_ids=(second.article.version_id,)),
            ),
            relevance=(
                ArticleRelevanceInput(
                    article_version_id=first.article.version_id,
                    decision=relevance_decision(accepted=True, major=True),
                ),
                ArticleRelevanceInput(
                    article_version_id=second.article.version_id,
                    decision=relevance_decision(accepted=False),
                ),
            ),
            higher_group_id=high_id,
            lower_group_id=low_id,
            observed_group_order=(high_id, low_id),
        ),
        ReaderPresentationEvaluationCase(
            case_id="reader-key-point-size",
            provenance=EvaluationProvenance(feedback_ids=()),
            expectation=KeyPointTextExpectation(observed_rem=1.0, minimum_rem=0.9),
        ),
    )
    return NewsEvaluationDataset(
        version="synthetic-v1",
        reviewed_at=REVIEWED_AT,
        issue_url="https://example.test/issues/1",
        source_feedback_ids=(),
        reports=(report,),
        cases=cases,
    )


def synthetic_manifest() -> NewsEvaluationManifest:
    feedback_ids = tuple(UUID(f"00000000-0000-4000-8000-{index:012d}") for index in range(1, 6))
    report = reference(702, "report")
    cluster = reference(701, "cluster")
    provenance = tuple(
        EvaluationSpecProvenance(
            feedback_ids=(feedback_id,),
            report_version_id=report.version_id,
        )
        for feedback_id in feedback_ids
    )
    first = embedded_article(1)
    second = embedded_article(2)
    group = reference(10, "group").version_id
    return NewsEvaluationManifest(
        version="synthetic-v1",
        reviewed_at=REVIEWED_AT,
        issue_url="https://example.test/issues/1",
        source_feedback_ids=feedback_ids,
        reports=(ReportEvaluationSpec(report=report, cluster_set=cluster),),
        cases=(
            RelevanceEvaluationSpec(
                case_id="relevance-accepts",
                provenance=provenance[0],
                expected_accepted=True,
                article=first.article,
                model_output=first.relevance,
            ),
            SummaryFormatEvaluationSpec(
                case_id="summary-format",
                provenance=provenance[1],
                articles=(first.article,),
                model_output=reference(700, "summary"),
            ),
            GroupingEvaluationSpec(
                case_id="grouping-same",
                provenance=provenance[2],
                source_cluster_set=cluster,
                day=DAY,
                articles=(
                    EmbeddedArticleReference(
                        article=first.article,
                        relevance=first.relevance,
                        embedding=first.embedding,
                    ),
                    EmbeddedArticleReference(
                        article=second.article,
                        relevance=second.relevance,
                        embedding=second.embedding,
                    ),
                ),
                left_article_version_id=first.article.version_id,
                right_article_version_id=second.article.version_id,
                expected_same_group=True,
            ),
            RankingEvaluationSpec(
                case_id="ranking-major-first",
                provenance=provenance[3],
                relevance=(
                    ArticleRelevanceSpec(article=first.article, model_output=first.relevance),
                    ArticleRelevanceSpec(article=second.article, model_output=second.relevance),
                ),
                higher_group_id=group,
                lower_group_id=reference(11, "group").version_id,
            ),
            ReaderPresentationEvaluationSpec(
                case_id="reader-key-point-size",
                provenance=provenance[4],
                expectation=KeyPointTextExpectation(observed_rem=1.0, minimum_rem=0.9),
            ),
        ),
    )


def _complete_report_snapshot(
    first: EmbeddedArticle,
    second: EmbeddedArticle,
    first_group_id: str,
    second_group_id: str,
) -> ReportEvaluationSnapshot:
    cluster = reference(600, "cluster")
    summaries = (reference(601, "summary"), reference(602, "summary"))
    sentiments = (reference(603, "sentiment"), reference(604, "sentiment"))
    return ReportEvaluationSnapshot(
        report=reference(605, "report"),
        cluster_set=cluster,
        report_article_version_ids=(first.article.version_id, second.article.version_id),
        report_group_ids=(first_group_id, second_group_id),
        cluster_articles=(
            EmbeddedArticleReference(
                article=first.article,
                relevance=first.relevance,
                embedding=first.embedding,
            ),
            EmbeddedArticleReference(
                article=second.article,
                relevance=second.relevance,
                embedding=second.embedding,
            ),
        ),
        group_inputs=(
            ReportGroupInputs(
                group_id=first_group_id,
                summary=summaries[0],
                sentiment=sentiments[0],
            ),
            ReportGroupInputs(
                group_id=second_group_id,
                summary=summaries[1],
                sentiment=sentiments[1],
            ),
        ),
        report_inputs=ReportInputReferences(
            cluster_set=(cluster,),
            relevance=(first.relevance, second.relevance),
            summary=summaries,
            sentiment=sentiments,
        ),
    )
