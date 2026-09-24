from datetime import UTC, date, datetime
from hashlib import sha256

import pytest
from pydantic import HttpUrl

from romanian_news.analysis.groups import pending
from romanian_news.analysis.groups.models import GroupAnalysisReferenceInput
from romanian_news.analysis.groups.pending import _read_article_references
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.groups import DailyClusterSet, NewsGroup


def test_article_reads_use_bounded_catalog_batches(monkeypatch) -> None:
    version_ids = tuple(sha256(str(index).encode()).hexdigest() for index in range(51))
    digest = sha256(b"article").hexdigest()
    batches = []

    def query(_sql, values):
        batches.append(values)
        return [
            {
                "artifact_id": f"news:article:{version_id}",
                "version_id": version_id,
                "content_digest": digest,
                "r2_key": f"objects/{version_id}",
            }
            for version_id in values
        ]

    monkeypatch.setattr("romanian_news.catalog.artifacts.catalog_query", query)

    assert set(_read_article_references(version_ids)) == set(version_ids)
    assert [len(batch) for batch in batches] == [50, 1]


def test_pending_group_planning_reads_references_without_article_bodies(monkeypatch) -> None:
    version_id = "a" * 64
    cluster_set = DailyClusterSet(
        day=date(2026, 8, 31),
        algorithm="complete-link-cosine-v1",
        threshold=0.72,
        embedding_model="test/model",
        article_version_ids=(version_id,),
        relevance_version_ids=("b" * 64,),
        embedding_version_ids=("c" * 64,),
        merges=(),
        groups=(NewsGroup(id="d" * 64, article_version_ids=(version_id,)),),
    )
    digest = sha256(cluster_set.model_dump_json().encode()).hexdigest()

    def query(sql, _values=None):
        if "cluster.kind" in sql:
            return [
                {
                    "artifact_id": "news:clusters:2026-08-31",
                    "version_id": "e" * 64,
                    "content_digest": digest,
                    "r2_key": "news/clusters/day.json",
                }
            ]
        return [
            {
                "artifact_id": "news:article:test",
                "version_id": version_id,
                "content_digest": "f" * 64,
                "r2_key": "news/articles/article.json",
            }
        ]

    reads = []
    monkeypatch.setattr("romanian_news.catalog.artifacts.catalog_query", query)
    monkeypatch.setattr("romanian_news.catalog.cluster_inputs.catalog_query", query)
    monkeypatch.setattr(pending, "existing_current_artifact_ids", lambda _ids: frozenset())
    monkeypatch.setattr(
        pending,
        "read_verified_r2_object",
        lambda key, _digest: reads.append(key) or cluster_set.model_dump_json().encode(),
    )

    values = pending.read_pending_group_analysis_references()

    assert values[0].articles[0].version_id == version_id
    assert reads == ["news/clusters/day.json"]


def test_group_analysis_input_loads_verified_article_content(monkeypatch) -> None:
    version_id = "a" * 64
    reference = ArtifactReference(
        artifact_id="news:article:test",
        version_id=version_id,
        content_digest="b" * 64,
        r2_key="news/articles/article.json",
    )
    article = ExtractedArticle(
        article_id="c" * 64,
        outlet_id="test",
        canonical_url=HttpUrl("https://example.test/article"),
        title="Titlu",
        body="Dovada exacta este aici.",
        author=None,
        published_at=datetime(2026, 8, 31, 9, tzinfo=UTC),
        source_updated_at=None,
        bucharest_day=date(2026, 8, 31),
        material_digest="d" * 64,
        extraction_digest="e" * 64,
    )
    value = GroupAnalysisReferenceInput(
        day=date(2026, 8, 31),
        cluster_set=ArtifactReference(
            artifact_id="news:clusters:2026-08-31",
            version_id="f" * 64,
            content_digest="1" * 64,
            r2_key="news/clusters/day.json",
        ),
        group=NewsGroup(id="2" * 64, article_version_ids=(version_id,)),
        articles=(reference,),
        summary_needed=True,
        sentiment_needed=False,
    )
    monkeypatch.setattr(
        pending,
        "read_verified_r2_object",
        lambda key, digest: (
            article.model_dump_json().encode()
            if (key, digest) == (reference.r2_key, reference.content_digest)
            else pytest.fail("article loader used a different artifact reference")
        ),
    )

    loaded = pending.load_group_analysis_input(value)

    assert loaded.articles == ((reference, article),)
