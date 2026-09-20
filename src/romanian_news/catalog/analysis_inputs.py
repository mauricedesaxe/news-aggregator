from datetime import date, datetime

from romanian_news import NewsModel
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog_transport import catalog_query


class ArticleAnalysisCatalogReference(NewsModel):
    reference: ArtifactReference
    bucharest_day: date


class EmbeddingCatalogCandidate(NewsModel):
    article: ArticleAnalysisCatalogReference
    relevance: ArtifactReference


def read_current_article_analysis_references(
    through_day: date | None = None, *, day: date | None = None
) -> tuple[ArticleAnalysisCatalogReference, ...]:
    day_filter, parameters = _day_filter(through_day, day)
    rows = catalog_query(
        f"""
        SELECT article.id AS artifact_id, version.id AS version_id,
               file.content_digest, file.r2_key, metadata.bucharest_day
        FROM artifacts article
        JOIN artifact_versions version ON version.id = article.current_version_id
        JOIN artifact_files file ON file.artifact_version_id = version.id
        JOIN news_article_versions metadata ON metadata.artifact_version_id = version.id
        WHERE article.kind = 'news_article' {day_filter}
        ORDER BY metadata.bucharest_day DESC, version.created_at, version.id
        """,
        parameters,
    )
    return tuple(
        ArticleAnalysisCatalogReference(
            reference=_reference(row),
            bucharest_day=datetime.fromisoformat(str(row["bucharest_day"])).date(),
        )
        for row in rows
    )


def read_embedding_candidates(
    through_day: date | None = None, *, day: date | None = None
) -> tuple[EmbeddingCatalogCandidate, ...]:
    day_filter, parameters = _day_filter(through_day, day)
    rows = catalog_query(
        f"""
        SELECT relevance.id AS relevance_artifact_id, relevance.current_version_id AS relevance_version_id,
               relevance_file.content_digest AS relevance_digest, relevance_file.r2_key AS relevance_r2_key,
               article_version.id AS article_version_id, article_version.artifact_id AS article_artifact_id,
               article_file.content_digest AS article_digest, article_file.r2_key AS article_r2_key,
               metadata.bucharest_day
        FROM news_article_versions metadata
        JOIN artifact_versions article_version ON article_version.id = metadata.artifact_version_id
        JOIN artifacts article_artifact ON article_artifact.id = article_version.artifact_id
          AND article_artifact.current_version_id = article_version.id
        JOIN artifact_files article_file ON article_file.artifact_version_id = article_version.id
        JOIN run_inputs input ON input.artifact_version_id = article_version.id AND input.role = 'article'
        JOIN artifact_versions relevance_version ON relevance_version.produced_by_run_id = input.run_id
        JOIN artifacts relevance ON relevance.id = relevance_version.artifact_id
          AND relevance.kind = 'news_relevance' AND relevance.current_version_id = relevance_version.id
        JOIN news_relevance_versions relevance_metadata
          ON relevance_metadata.artifact_version_id = relevance_version.id AND relevance_metadata.accepted = 1
        JOIN artifact_files relevance_file ON relevance_file.artifact_version_id = relevance_version.id
        WHERE metadata.bucharest_day IS NOT NULL {day_filter}
        ORDER BY metadata.bucharest_day DESC, article_version.created_at, article_version.id
        """,
        parameters,
    )
    return tuple(
        EmbeddingCatalogCandidate(
            article=ArticleAnalysisCatalogReference(
                reference=_prefixed_reference(row, "article"),
                bucharest_day=datetime.fromisoformat(str(row["bucharest_day"])).date(),
            ),
            relevance=_prefixed_reference(row, "relevance"),
        )
        for row in rows
    )


def _day_filter(through_day: date | None, day: date | None) -> tuple[str, list[object]]:
    if day is not None:
        return "AND metadata.bucharest_day = %s", [day.isoformat()]
    if through_day is not None:
        return "AND metadata.bucharest_day <= %s", [through_day.isoformat()]
    return "", []


def _reference(row: dict[str, object]) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=str(row["artifact_id"]),
        version_id=str(row["version_id"]),
        content_digest=str(row["content_digest"]),
        r2_key=str(row["r2_key"]),
    )


def _prefixed_reference(row: dict[str, object], prefix: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=str(row[f"{prefix}_artifact_id"]),
        version_id=str(row[f"{prefix}_version_id"]),
        content_digest=str(row[f"{prefix}_digest"]),
        r2_key=str(row[f"{prefix}_r2_key"]),
    )
