import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import UUID

from pydantic import HttpUrl

from romanian_news import EMBEDDING_DIMENSIONS
from romanian_news.analysis.embeddings import EmbeddingInput, embed_article
from romanian_news.analysis.groups.models import GroupAnalysisInput
from romanian_news.analysis.groups.summary import summarize_group
from romanian_news.analysis.relevance import ArticleAnalysisInput
from romanian_news.analysis.tracing import ModelTraceReference, ProviderCallResult
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.groups import NewsGroup

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64


class _EmbeddingResponse:
    id = "embedding-response"
    model = "test/embedding"
    usage = SimpleNamespace(prompt_tokens=12)
    data = [SimpleNamespace(embedding=[0.0] * EMBEDDING_DIMENSIONS)]

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        return {
            "id": self.id,
            "model": self.model,
            "usage": {"prompt_tokens": 12},
        }


class _SummaryResponse:
    id = "summary-response"
    model = "test/generation"
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=7)
    choices = [
        SimpleNamespace(
            message=SimpleNamespace(
                content=json.dumps(
                    {
                        "title_ro": "Titlu",
                        "summary_ro": "Rezumat",
                        "key_points_ro": ["Punct"],
                        "disagreements_ro": [],
                        "uncertainty_ro": "Nicio incertitudine",
                        "cited_article_version_ids": ["a1"],
                    }
                )
            )
        )
    ]

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        return {"id": self.id, "model": self.model, "usage": {}}


def test_embedding_forwards_trace_to_attempt(monkeypatch) -> None:
    trace = ModelTraceReference(
        provider="langfuse",
        trace_id="t" * 32,
        observation_id="o" * 32,
        project_ref="project",
        recorded_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    recorded = []
    monkeypatch.setattr(
        "romanian_news.analysis.embeddings.trace_provider_call",
        lambda *_args: ProviderCallResult(
            response=_EmbeddingResponse(),
            call_id=UUID("018f0000-0000-7000-8000-000000000001"),
            trace=trace,
        ),
    )
    monkeypatch.setattr(
        "romanian_news.analysis.embeddings.record_model_attempt",
        lambda *_args, **kwargs: recorded.append(kwargs),
    )

    embed_article(
        EmbeddingInput(
            article=ArticleAnalysisInput(reference=_reference(_A), article=_article()),
            relevance=_reference(_B),
        )
    )

    assert recorded[0]["trace"] is trace
    assert recorded[0]["fallback_response_id"] == "018f0000-0000-7000-8000-000000000001"


def test_summary_forwards_trace_to_attempt(monkeypatch) -> None:
    trace = ModelTraceReference(
        provider="langfuse",
        trace_id="t" * 32,
        observation_id="o" * 32,
        project_ref="project",
        recorded_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    recorded = []
    monkeypatch.setattr(
        "romanian_news.analysis.groups.summary.trace_provider_call",
        lambda *_args: ProviderCallResult(
            response=_SummaryResponse(),
            call_id=UUID("018f0000-0000-7000-8000-000000000002"),
            trace=trace,
        ),
    )
    monkeypatch.setattr(
        "romanian_news.analysis.groups.summary.record_model_attempt",
        lambda *_args, **kwargs: recorded.append(kwargs),
    )

    summarize_group(_group_input(), "article context")

    assert recorded[0]["trace"] is trace
    assert recorded[0]["fallback_response_id"] == "018f0000-0000-7000-8000-000000000002"


def _group_input() -> GroupAnalysisInput:
    return GroupAnalysisInput(
        day=date(2026, 8, 31),
        cluster_set=_reference(_B),
        group=NewsGroup(id=_C, article_version_ids=(_A,)),
        articles=((_reference(_A), _article()),),
        summary_needed=True,
        sentiment_needed=False,
    )


def _article() -> ExtractedArticle:
    return ExtractedArticle(
        article_id=_A,
        outlet_id="test",
        canonical_url=HttpUrl("https://example.test/article"),
        title="Titlu",
        body="Dovada exacta este aici.",
        author=None,
        published_at=datetime(2026, 8, 31, 9, tzinfo=UTC),
        source_updated_at=None,
        bucharest_day=date(2026, 8, 31),
        material_digest=_B,
        extraction_digest=_C,
    )


def _reference(version_id: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"artifact:{version_id[:4]}",
        version_id=version_id,
        content_digest=_D,
        r2_key=f"objects/{version_id}",
    )
