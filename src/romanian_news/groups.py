from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from heapq import heappop, heappush
from typing import Annotated

import numpy as np
from pydantic import Field, model_validator

from romanian_news import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, NewsModel, Sha256
from romanian_news.analysis.embeddings import embedding_request_id
from romanian_news.analysis.relevance_v3 import production_relevance_v3_request_id
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.identity import canonical_json as _canonical_json
from romanian_news.identity import sha256 as _sha256
from romanian_news.storage import read_verified_r2_object

CLUSTER_ALGORITHM = "average-link-cosine-v1"
DEFAULT_CLUSTER_THRESHOLD = 0.72


class EmbeddedArticleReference(NewsModel):
    article: ArtifactReference
    relevance: ArtifactReference
    embedding: ArtifactReference


class EmbeddedArticle(EmbeddedArticleReference):
    article: ArtifactReference
    relevance: ArtifactReference
    embedding: ArtifactReference
    value: ExtractedArticle
    vector: tuple[float, ...]

    @model_validator(mode="after")
    def validate_vector(self) -> EmbeddedArticle:
        if len(self.vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Embedding must have {EMBEDDING_DIMENSIONS} dimensions")
        return self


class ClusterMerge(NewsModel):
    left_article_version_ids: tuple[Sha256, ...]
    right_article_version_ids: tuple[Sha256, ...]
    similarity: float


class NewsGroup(NewsModel):
    id: Sha256
    article_version_ids: tuple[Sha256, ...]


class DailyClusterSet(NewsModel):
    day: date
    algorithm: str
    threshold: Annotated[float, Field(ge=-1, le=1)]
    embedding_model: str
    article_version_ids: tuple[Sha256, ...]
    relevance_version_ids: tuple[Sha256, ...]
    embedding_version_ids: tuple[Sha256, ...]
    merges: tuple[ClusterMerge, ...]
    groups: tuple[NewsGroup, ...]


class DailyClusterOutput(NewsModel):
    request_id: Sha256
    content_digest: Sha256
    cluster_set: DailyClusterSet
    articles: tuple[EmbeddedArticle, ...]
    content: bytes


@dataclass(frozen=True)
class _ClusterCandidate:
    similarity: float
    tie_break: tuple[Sha256, ...]
    left_id: int
    right_id: int

    def __lt__(self, other: _ClusterCandidate) -> bool:
        return (self.similarity, self.tie_break) > (other.similarity, other.tie_break)


def parse_daily_cluster_set(content: bytes) -> DailyClusterSet:
    """Parse one durable cluster set at its owning boundary."""
    return DailyClusterSet.model_validate_json(content, strict=True)


def read_cluster_articles(cluster_set: DailyClusterSet) -> tuple[EmbeddedArticle, ...]:
    """Read the exact immutable inputs recorded by a cluster set."""
    from romanian_news.catalog.cluster_inputs import read_cluster_article_references

    values = []
    for article_id, relevance_id, embedding_id in zip(
        cluster_set.article_version_ids,
        cluster_set.relevance_version_ids,
        cluster_set.embedding_version_ids,
        strict=True,
    ):
        catalog_reference = read_cluster_article_references(article_id, relevance_id, embedding_id)
        if catalog_reference is None:
            raise ValueError("Cluster set references unavailable analysis inputs")
        values.append(_load_embedded_article(_embedded_reference(catalog_reference)))
    return tuple(values)


def read_embedded_articles(day: date | None = None) -> tuple[EmbeddedArticle, ...]:
    """Read accepted current articles with verified relevance and embeddings."""
    references = read_embedded_article_references(day)
    if not references:
        return ()
    with ThreadPoolExecutor(max_workers=min(16, len(references))) as executor:
        return tuple(executor.map(_load_embedded_article, references))


def read_embedded_article_references(
    day: date | None = None,
) -> tuple[EmbeddedArticleReference, ...]:
    """Read immutable references for accepted current article analyses."""
    from romanian_news.catalog.cluster_inputs import (
        read_embedded_article_references as read_references,
    )

    values = tuple(_embedded_reference(value) for value in read_references(day))
    return tuple(value for value in values if _uses_current_analysis_policy(value))


def cluster_request_id(
    day: date,
    articles: tuple[EmbeddedArticleReference, ...],
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> Sha256:
    """Compute the cluster request identity without loading article content."""
    ordered = tuple(sorted(articles, key=lambda value: value.article.version_id))
    config_digest = _cluster_config_digest(threshold)
    return _sha256(
        _canonical_json(
            {
                "article_version_ids": [value.article.version_id for value in ordered],
                "config_digest": config_digest,
                "day": day.isoformat(),
                "embedding_version_ids": [value.embedding.version_id for value in ordered],
                "operation": "news.cluster_day",
                "relevance_version_ids": [value.relevance.version_id for value in ordered],
            }
        )
    )


def cluster_articles(
    day: date,
    articles: tuple[EmbeddedArticle, ...],
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> DailyClusterOutput:
    """Group one Romanian day through deterministic average-link clustering."""
    if any(article.value.bucharest_day != day for article in articles):
        raise ValueError("Every clustered article must belong to the requested Romanian day")
    ordered = tuple(sorted(articles, key=lambda value: value.article.version_id))
    article_version_ids = tuple(value.article.version_id for value in ordered)
    similarities = _cosine_similarity_matrix(ordered)
    clusters, merges = _average_link_clusters(
        article_version_ids, similarities, threshold=threshold
    )
    config_digest = _cluster_config_digest(threshold)
    groups = tuple(
        NewsGroup(
            id=_sha256(
                _canonical_json(
                    {
                        "article_version_ids": [article_version_ids[index] for index in cluster],
                        "config_digest": config_digest,
                        "day": day.isoformat(),
                    }
                )
            ),
            article_version_ids=tuple(article_version_ids[index] for index in cluster),
        )
        for cluster in clusters
    )
    cluster_set = DailyClusterSet(
        day=day,
        algorithm=CLUSTER_ALGORITHM,
        threshold=threshold,
        embedding_model=EMBEDDING_MODEL,
        article_version_ids=article_version_ids,
        relevance_version_ids=tuple(value.relevance.version_id for value in ordered),
        embedding_version_ids=tuple(value.embedding.version_id for value in ordered),
        merges=merges,
        groups=groups,
    )
    content = _canonical_json(cluster_set.model_dump(mode="json"))
    request_id = cluster_request_id(day, ordered, threshold=threshold)
    return DailyClusterOutput(
        request_id=request_id,
        content_digest=_sha256(content),
        cluster_set=cluster_set,
        articles=ordered,
        content=content,
    )


def _cosine_similarity_matrix(
    articles: tuple[EmbeddedArticle, ...],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    if not articles:
        return np.empty((0, 0), dtype=np.float64)
    matrix = np.asarray([value.vector for value in articles], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("Embedding vectors must have non-zero magnitude")
    return matrix @ matrix.T / np.outer(norms, norms)


def _average_link_clusters(
    article_version_ids: tuple[Sha256, ...],
    similarities: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    *,
    threshold: float,
) -> tuple[tuple[tuple[int, ...], ...], tuple[ClusterMerge, ...]]:
    clusters: dict[int, tuple[int, ...]] = {
        index: (index,) for index in range(len(article_version_ids))
    }
    candidates: list[_ClusterCandidate] = []
    merges = []

    def add_candidate(left_id: int, right_id: int) -> None:
        left = clusters[left_id]
        right = clusters[right_id]
        if left > right:
            left_id, right_id = right_id, left_id
            left, right = right, left
        similarity = float(np.mean([similarities[a, b] for a in left for b in right]))
        if similarity >= threshold:
            member_ids = tuple(sorted(article_version_ids[index] for index in (*left, *right)))
            heappush(
                candidates,
                _ClusterCandidate(similarity, tuple(reversed(member_ids)), left_id, right_id),
            )

    for left_id in range(len(article_version_ids)):
        for right_id in range(left_id + 1, len(article_version_ids)):
            add_candidate(left_id, right_id)

    next_id = len(article_version_ids)
    while candidates:
        candidate = heappop(candidates)
        if candidate.left_id not in clusters or candidate.right_id not in clusters:
            continue
        left = clusters.pop(candidate.left_id)
        right = clusters.pop(candidate.right_id)
        merges.append(
            ClusterMerge(
                left_article_version_ids=tuple(article_version_ids[index] for index in left),
                right_article_version_ids=tuple(article_version_ids[index] for index in right),
                similarity=candidate.similarity,
            )
        )
        merged = tuple(sorted((*left, *right)))
        clusters[next_id] = merged
        for other_id in tuple(clusters):
            if other_id != next_id:
                add_candidate(other_id, next_id)
        next_id += 1

    ordered = sorted(
        clusters.values(),
        key=lambda value: tuple(article_version_ids[index] for index in value),
    )
    return tuple(ordered), tuple(merges)


def _embedded_reference(value) -> EmbeddedArticleReference:
    return EmbeddedArticleReference(
        article=value.article,
        relevance=value.relevance,
        embedding=value.embedding,
    )


def _uses_current_analysis_policy(value: EmbeddedArticleReference) -> bool:
    return (
        value.relevance.artifact_id
        == f"news:relevance:{production_relevance_v3_request_id(value.article)}"
        and value.embedding.artifact_id
        == f"news:embedding:{embedding_request_id(value.article, value.relevance)}"
    )


def _load_embedded_article(value: EmbeddedArticleReference) -> EmbeddedArticle:
    article = ExtractedArticle.model_validate_json(
        read_verified_r2_object(value.article.r2_key, value.article.content_digest), strict=True
    )
    embedding_payload = json.loads(
        read_verified_r2_object(value.embedding.r2_key, value.embedding.content_digest)
    )
    if embedding_payload.get("article_version_id") != value.article.version_id:
        raise ValueError("Embedding payload references another article version")
    if embedding_payload.get("relevance_version_id") != value.relevance.version_id:
        raise ValueError("Embedding payload references another relevance version")
    return EmbeddedArticle(
        **value.model_dump(), value=article, vector=tuple(embedding_payload["vector"])
    )


def _cluster_config_digest(threshold: float) -> Sha256:
    return _sha256(
        _canonical_json(
            {
                "algorithm": CLUSTER_ALGORITHM,
                "embedding_model": EMBEDDING_MODEL,
                "threshold": threshold,
            }
        )
    )
