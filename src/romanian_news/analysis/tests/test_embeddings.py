from datetime import UTC, date, datetime
from types import SimpleNamespace

from pydantic import HttpUrl

from romanian_news.analysis import embeddings as embeddings_module
from romanian_news.analysis.embeddings import (
    EMBEDDING_TEXT_POLICY,
    EmbeddingInput,
    embed_article,
    embedding_request_id,
    read_pending_embedding_references,
    read_pending_embeddings,
)
from romanian_news.analysis.relevance import ArticleAnalysisInput
from romanian_news.analysis.relevance_v3 import production_relevance_v3_request_id
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference


def test_embedding_request_identity_stays_stable() -> None:
    article = ArtifactReference(
        artifact_id="news:article:x",
        version_id="a" * 64,
        content_digest="b" * 64,
        r2_key="x",
    )
    relevance = ArtifactReference(
        artifact_id="news:relevance:x",
        version_id="c" * 64,
        content_digest="d" * 64,
        r2_key="y",
    )

    request_id = embedding_request_id(article, relevance)

    assert request_id == embedding_request_id(article, relevance)
    assert embeddings_module.EMBEDDING_TEXT_POLICY == "title-body-1000-v1"
    assert request_id == "9fe0b09cbba8639856757e04ecc3788e193aabc80e43f06d5969f91304caf8f7"


def test_pending_embeddings_read_recent_report_days_first(monkeypatch) -> None:
    queries = []
    monkeypatch.setattr(
        "romanian_news.catalog.analysis_inputs.catalog_query",
        lambda query, *_args: queries.append(" ".join(query.split())) or [],
    )

    assert read_pending_embeddings() == ()
    assert "FROM news_article_versions metadata" in queries[0]
    assert "JOIN artifacts relevance ON relevance.id" in queries[0]
    assert (
        "ORDER BY metadata.bucharest_day DESC, article_version.created_at, article_version.id"
        in queries[0]
    )


def test_pending_embeddings_skip_archived_relevance_before_parsing(monkeypatch) -> None:
    row = {
        "relevance_artifact_id": "news:relevance:archived-policy",
        "relevance_version_id": "c" * 64,
        "relevance_digest": "d" * 64,
        "relevance_r2_key": "news/relevance/archived.json",
        "article_version_id": "a" * 64,
        "article_artifact_id": "news:article:a",
        "article_digest": "b" * 64,
        "article_r2_key": "news/articles/a.json",
        "bucharest_day": "2026-09-05",
    }
    monkeypatch.setattr("romanian_news.catalog.analysis_inputs.catalog_query", lambda *_: [row])
    monkeypatch.setattr(
        "romanian_news.analysis.embeddings.read_verified_r2_object",
        lambda *_: (_ for _ in ()).throw(AssertionError("archived payload was parsed")),
    )

    assert read_pending_embeddings() == ()


def test_pending_embeddings_select_production_v3_relevance(monkeypatch) -> None:
    article = ArtifactReference(
        artifact_id="news:article:a",
        version_id="a" * 64,
        content_digest="b" * 64,
        r2_key="news/articles/a.json",
    )
    relevance_id = production_relevance_v3_request_id(article)
    row = {
        "relevance_artifact_id": f"news:relevance:{relevance_id}",
        "relevance_version_id": "c" * 64,
        "relevance_digest": "d" * 64,
        "relevance_r2_key": "news/relevance/current.json",
        "article_version_id": article.version_id,
        "article_artifact_id": article.artifact_id,
        "article_digest": article.content_digest,
        "article_r2_key": article.r2_key,
        "bucharest_day": "2026-09-05",
    }
    monkeypatch.setattr("romanian_news.catalog.analysis_inputs.catalog_query", lambda *_: [row])
    monkeypatch.setattr(embeddings_module, "existing_current_artifact_ids", lambda *_: set())

    pending = read_pending_embedding_references(day=date(2026, 9, 5))

    assert len(pending) == 1
    assert pending[0].article.reference == article
    assert pending[0].relevance.artifact_id == f"news:relevance:{relevance_id}"


def test_embedding_uses_title_and_first_1000_body_characters(monkeypatch) -> None:
    article_reference = ArtifactReference(
        artifact_id="news:article:x",
        version_id="a" * 64,
        content_digest="b" * 64,
        r2_key="x",
    )
    relevance = ArtifactReference(
        artifact_id="news:relevance:x",
        version_id="c" * 64,
        content_digest="d" * 64,
        r2_key="y",
    )
    article = ExtractedArticle(
        article_id="e" * 64,
        outlet_id="test",
        canonical_url=HttpUrl("https://example.test/article"),
        title="Event title",
        body="x" * 1000 + "excluded",
        author=None,
        published_at=datetime(2026, 8, 31, tzinfo=UTC),
        source_updated_at=None,
        bucharest_day=date(2026, 8, 31),
        material_digest="f" * 64,
        extraction_digest="0" * 64,
    )
    captured = {}
    response = SimpleNamespace(
        model="test/model",
        usage=SimpleNamespace(prompt_tokens=10),
        data=[SimpleNamespace(embedding=[0.0] * embeddings_module.EMBEDDING_DIMENSIONS)],
        model_dump=lambda **_kwargs: {"id": "embedding-response"},
    )
    monkeypatch.setattr(
        embeddings_module,
        "trace_provider_call",
        lambda _operation, _request_id, inputs, _call: (
            captured.update(inputs)
            or SimpleNamespace(response=response, call_id="call", trace=None)
        ),
    )
    monkeypatch.setattr(embeddings_module, "record_model_attempt", lambda *_args, **_kwargs: None)

    embed_article(
        EmbeddingInput(
            article=ArticleAnalysisInput(reference=article_reference, article=article),
            relevance=relevance,
        )
    )

    assert EMBEDDING_TEXT_POLICY == "title-body-1000-v1"
    assert captured["input"] == "Event title\n\n" + "x" * 1000
