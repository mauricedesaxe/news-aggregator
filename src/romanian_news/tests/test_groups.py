import json
import math
from datetime import date, datetime
from threading import Barrier, get_ident
from time import sleep

import pytest
from pydantic import HttpUrl

from romanian_news import EMBEDDING_DIMENSIONS, groups
from romanian_news.analysis.embeddings import embedding_request_id
from romanian_news.analysis.relevance_v3 import production_relevance_v3_request_id
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.cluster_inputs import (
    read_embedded_article_references as read_catalog_embedded_article_references,
)
from romanian_news.groups import (
    DailyClusterSet,
    EmbeddedArticle,
    cluster_articles,
    read_cluster_articles,
    read_embedded_article_references,
)


def test_average_link_clustering_is_independent_of_input_order() -> None:
    articles = (
        _article("a" * 64, (1.0, 0.0)),
        _article("b" * 64, (0.99, 0.1)),
        _article("c" * 64, (0.0, 1.0)),
    )

    first = cluster_articles(date(2026, 8, 31), articles, threshold=0.9)
    second = cluster_articles(date(2026, 8, 31), tuple(reversed(articles)), threshold=0.9)

    assert first.content == second.content
    assert [len(group.article_version_ids) for group in first.cluster_set.groups] == [2, 1]


def test_embedded_article_loading_runs_concurrently_without_reordering(monkeypatch) -> None:
    barrier = Barrier(4)
    worker_threads = []
    monkeypatch.setattr(groups, "read_embedded_article_references", lambda _day: (0, 1, 2, 3))

    def load(reference):
        worker_threads.append(get_ident())
        barrier.wait(timeout=10)
        sleep(0.01 * (3 - reference))
        return reference

    monkeypatch.setattr(groups, "_load_embedded_article", load)

    assert groups.read_embedded_articles(date(2026, 9, 29)) == (0, 1, 2, 3)
    assert len(set(worker_threads)) == 4


def test_average_link_can_join_a_cluster_with_one_weak_pair() -> None:
    articles = (
        _article("a" * 64, (1.0, 0.0)),
        _article("b" * 64, (0.8, 0.6)),
        _article("c" * 64, (0.72, math.sqrt(1 - 0.72**2))),
    )

    result = cluster_articles(date(2026, 8, 31), articles, threshold=0.75)

    assert [len(group.article_version_ids) for group in result.cluster_set.groups] == [3]


def test_average_link_records_stable_tie_history() -> None:
    articles = (
        _article("a" * 64, (1.0, 0.0)),
        _article("b" * 64, (1.0, 0.0)),
        _article("c" * 64, (1.0, 0.0)),
    )

    result = cluster_articles(date(2026, 8, 31), articles, threshold=1.0)
    reversed_result = cluster_articles(date(2026, 8, 31), tuple(reversed(articles)), threshold=1.0)

    assert result.content == reversed_result.content
    assert [
        (merge.left_article_version_ids, merge.right_article_version_ids)
        for merge in result.cluster_set.merges
    ] == [
        (("b" * 64,), ("c" * 64,)),
        (("a" * 64,), ("b" * 64, "c" * 64)),
    ]


def test_average_link_handles_a_full_day_of_similar_articles() -> None:
    articles = tuple(
        _article(
            f"{index:064x}",
            (
                math.cos(2 * math.pi * (index % 8) / 8),
                math.sin(2 * math.pi * (index % 8) / 8),
            ),
        )
        for index in range(256)
    )

    result = cluster_articles(date(2026, 8, 31), articles, threshold=0.9)

    assert len(result.cluster_set.merges) == 248
    assert sorted(len(group.article_version_ids) for group in result.cluster_set.groups) == [32] * 8


def test_default_threshold_groups_articles_with_point_seven_five_similarity() -> None:
    articles = (
        _article("a" * 64, (1.0, 0.0)),
        _article("b" * 64, (0.75, math.sqrt(1 - 0.75**2))),
    )

    result = cluster_articles(date(2026, 8, 31), articles)

    assert [len(group.article_version_ids) for group in result.cluster_set.groups] == [2]


def test_default_cluster_threshold_defines_empty_identity() -> None:
    output = cluster_articles(date(2026, 8, 30), ())

    assert output.cluster_set.article_version_ids == ()
    assert output.cluster_set.groups == ()
    assert output.cluster_set.threshold == 0.72
    assert output.request_id == "3cfa1fee230403844ffd602bc05a821bd16a9602d27e4dfc3079dc0d619b900a"


def test_rejects_articles_from_another_day() -> None:
    article = _article("a" * 64, (1.0, 0.0))

    with pytest.raises(
        ValueError, match="Every clustered article must belong to the requested Romanian day"
    ):
        cluster_articles(date(2026, 8, 30), (article,))


def test_rejects_zero_magnitude_embeddings() -> None:
    article = _article("a" * 64, (0.0, 0.0))

    with pytest.raises(ValueError, match="Embedding vectors must have non-zero magnitude"):
        cluster_articles(date(2026, 8, 31), (article,))


def test_reads_embedded_article_references_in_bounded_pages(monkeypatch) -> None:
    row = {
        "article_artifact_id": "news:article:1",
        "article_version_id": "a" * 64,
        "article_digest": "b" * 64,
        "article_r2_key": "news/articles/1.json",
        "relevance_artifact_id": "news:relevance:1",
        "relevance_version_id": "c" * 64,
        "relevance_digest": "d" * 64,
        "relevance_r2_key": "news/relevance/1.json",
        "embedding_artifact_id": "news:embedding:1",
        "embedding_version_id": "e" * 64,
        "embedding_digest": "f" * 64,
        "embedding_r2_key": "news/embeddings/1.json",
    }
    pages = [[row] * 100, [row]]
    parameters = []

    def query(_sql, params):
        parameters.append(params)
        return pages.pop(0)

    monkeypatch.setattr("romanian_news.catalog.cluster_inputs.catalog_query", query)
    values = read_catalog_embedded_article_references(date(2026, 9, 3))

    assert len(values) == 101
    assert parameters == [["2026-09-03", 0], ["2026-09-03", 100]]


def test_reads_embedded_articles_from_production_v3_analysis(monkeypatch) -> None:
    article = ArtifactReference(
        artifact_id="news:article:a",
        version_id="a" * 64,
        content_digest="b" * 64,
        r2_key="news/articles/a.json",
    )
    relevance = ArtifactReference(
        artifact_id=f"news:relevance:{production_relevance_v3_request_id(article)}",
        version_id="c" * 64,
        content_digest="d" * 64,
        r2_key="news/relevance/a.json",
    )
    row = {
        "article_artifact_id": article.artifact_id,
        "article_version_id": article.version_id,
        "article_digest": article.content_digest,
        "article_r2_key": article.r2_key,
        "relevance_artifact_id": relevance.artifact_id,
        "relevance_version_id": relevance.version_id,
        "relevance_digest": relevance.content_digest,
        "relevance_r2_key": relevance.r2_key,
        "embedding_artifact_id": f"news:embedding:{embedding_request_id(article, relevance)}",
        "embedding_version_id": "e" * 64,
        "embedding_digest": "f" * 64,
        "embedding_r2_key": "news/embeddings/a.json",
    }
    monkeypatch.setattr("romanian_news.catalog.cluster_inputs.catalog_query", lambda *_: [row])

    values = read_embedded_article_references(date(2026, 9, 5))

    assert len(values) == 1
    assert values[0].article == article
    assert values[0].relevance == relevance


def test_replays_recorded_inputs_after_policy_identity_changes(monkeypatch) -> None:
    article = _article("a" * 64, (1.0, 0.0))
    relevance_version = "b" * 64
    embedding_version = "c" * 64
    cluster_set = DailyClusterSet(
        day=article.value.bucharest_day,
        algorithm="complete-link-cosine-v1",
        threshold=0.82,
        embedding_model="recorded-model",
        article_version_ids=(article.article.version_id,),
        relevance_version_ids=(relevance_version,),
        embedding_version_ids=(embedding_version,),
        merges=(),
        groups=(),
    )
    row = {
        "article_artifact_id": article.article.artifact_id,
        "article_version_id": article.article.version_id,
        "article_digest": article.article.content_digest,
        "article_r2_key": article.article.r2_key,
        "relevance_artifact_id": "news:relevance:old-policy",
        "relevance_version_id": relevance_version,
        "relevance_digest": "d" * 64,
        "relevance_r2_key": "news/relevance/old.json",
        "embedding_artifact_id": "news:embedding:old-policy",
        "embedding_version_id": embedding_version,
        "embedding_digest": "e" * 64,
        "embedding_r2_key": "news/embeddings/old.json",
    }
    monkeypatch.setattr("romanian_news.catalog.cluster_inputs.catalog_query", lambda *_: [row])
    monkeypatch.setattr(
        "romanian_news.groups.read_verified_r2_object",
        lambda key, _digest: (
            article.value.model_dump_json().encode()
            if key == article.article.r2_key
            else json.dumps(
                {
                    "article_version_id": article.article.version_id,
                    "relevance_version_id": relevance_version,
                    "vector": article.vector,
                }
            ).encode()
        ),
    )

    replayed = read_cluster_articles(cluster_set)

    assert replayed[0].article.version_id == article.article.version_id
    assert replayed[0].relevance.artifact_id == "news:relevance:old-policy"
    assert replayed[0].embedding.artifact_id == "news:embedding:old-policy"
    assert replayed[0].vector == article.vector


def _article(version_id: str, head: tuple[float, float]) -> EmbeddedArticle:
    vector = head + (0.0,) * (EMBEDDING_DIMENSIONS - len(head))
    reference = ArtifactReference(
        artifact_id=f"news:article:{version_id}",
        version_id=version_id,
        content_digest="d" * 64,
        r2_key=f"news/articles/{version_id}.json",
    )
    return EmbeddedArticle(
        article=reference,
        relevance=reference.model_copy(update={"artifact_id": f"news:relevance:{version_id}"}),
        embedding=reference.model_copy(update={"artifact_id": f"news:embedding:{version_id}"}),
        value=ExtractedArticle(
            article_id=version_id,
            outlet_id="test",
            canonical_url=HttpUrl(f"https://example.test/{version_id}"),
            title=f"Article {version_id[0]}",
            body="Text de test suficient de lung pentru a reprezenta un articol publicat.",
            author=None,
            published_at=datetime.fromisoformat("2026-08-31T08:00:00+00:00"),
            source_updated_at=None,
            bucharest_day=date(2026, 8, 31),
            material_digest="e" * 64,
            extraction_digest="f" * 64,
        ),
        vector=vector,
    )
