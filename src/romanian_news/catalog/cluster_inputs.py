from datetime import date

from romanian_news import NewsModel, Sha256
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog_transport import catalog_query


class EmbeddedArticleCatalogReference(NewsModel):
    article: ArtifactReference
    relevance: ArtifactReference
    embedding: ArtifactReference


def read_current_cluster_references() -> tuple[ArtifactReference, ...]:
    rows = catalog_query(
        """SELECT cluster.id AS artifact_id, cluster.current_version_id AS version_id,
                  file.content_digest, file.r2_key FROM artifacts cluster
           JOIN artifact_files file ON file.artifact_version_id = cluster.current_version_id
           WHERE cluster.kind = 'news_clusters' ORDER BY cluster.id"""
    )
    return tuple(ArtifactReference.model_validate(row, strict=True) for row in rows)


def read_embedded_article_references(
    day: date | None = None,
) -> tuple[EmbeddedArticleCatalogReference, ...]:
    day_clause = "AND metadata.bucharest_day = %s" if day is not None else ""
    parameters: list[object] = [day.isoformat()] if day is not None else []
    rows: list[dict[str, object]] = []
    while True:
        page = catalog_query(
            f"""
            SELECT article_version.artifact_id AS article_artifact_id, article_version.id AS article_version_id,
                   article_file.content_digest AS article_digest, article_file.r2_key AS article_r2_key,
                   relevance.id AS relevance_artifact_id, relevance.current_version_id AS relevance_version_id,
                   relevance_file.content_digest AS relevance_digest, relevance_file.r2_key AS relevance_r2_key,
                   embedding.id AS embedding_artifact_id, embedding.current_version_id AS embedding_version_id,
                   embedding_file.content_digest AS embedding_digest, embedding_file.r2_key AS embedding_r2_key
            FROM artifacts embedding
            JOIN artifact_versions embedding_version ON embedding_version.id = embedding.current_version_id
            JOIN artifact_files embedding_file ON embedding_file.artifact_version_id = embedding_version.id
            JOIN run_inputs article_input ON article_input.run_id = embedding_version.produced_by_run_id AND article_input.role = 'article'
            JOIN run_inputs relevance_input ON relevance_input.run_id = embedding_version.produced_by_run_id AND relevance_input.role = 'relevance'
            JOIN artifact_versions article_version ON article_version.id = article_input.artifact_version_id
            JOIN artifact_files article_file ON article_file.artifact_version_id = article_version.id
            JOIN news_article_versions metadata ON metadata.artifact_version_id = article_version.id
            JOIN artifacts article ON article.id = article_version.artifact_id AND article.current_version_id = article_version.id
            JOIN artifact_versions relevance_version ON relevance_version.id = relevance_input.artifact_version_id
            JOIN artifacts relevance ON relevance.current_version_id = relevance_version.id
            JOIN artifact_files relevance_file ON relevance_file.artifact_version_id = relevance_version.id
            WHERE embedding.kind = 'news_embedding' {day_clause}
            ORDER BY metadata.bucharest_day, article_version.artifact_id, article_version.id
            LIMIT 100 OFFSET %s
            """,
            [*parameters, len(rows)],
        )
        rows.extend(page)
        if len(page) < 100:
            break
    return tuple(_embedded(row) for row in rows)


def read_cluster_article_references(
    article_version_id: Sha256, relevance_version_id: Sha256, embedding_version_id: Sha256
) -> EmbeddedArticleCatalogReference | None:
    rows = catalog_query(
        """
        SELECT article.artifact_id AS article_artifact_id, article.id AS article_version_id,
               article_file.content_digest AS article_digest, article_file.r2_key AS article_r2_key,
               relevance.artifact_id AS relevance_artifact_id, relevance.id AS relevance_version_id,
               relevance_file.content_digest AS relevance_digest, relevance_file.r2_key AS relevance_r2_key,
               embedding.artifact_id AS embedding_artifact_id, embedding.id AS embedding_version_id,
               embedding_file.content_digest AS embedding_digest, embedding_file.r2_key AS embedding_r2_key
        FROM artifact_versions article JOIN artifact_files article_file ON article_file.artifact_version_id = article.id
        JOIN artifact_versions relevance ON relevance.id = %s
        JOIN artifact_files relevance_file ON relevance_file.artifact_version_id = relevance.id
        JOIN artifact_versions embedding ON embedding.id = %s
        JOIN artifact_files embedding_file ON embedding_file.artifact_version_id = embedding.id
        WHERE article.id = %s
        """,
        [relevance_version_id, embedding_version_id, article_version_id],
    )
    return _embedded(rows[0]) if len(rows) == 1 else None


def _embedded(row: dict[str, object]) -> EmbeddedArticleCatalogReference:
    def reference(prefix: str) -> ArtifactReference:
        return ArtifactReference(
            artifact_id=str(row[f"{prefix}_artifact_id"]),
            version_id=str(row[f"{prefix}_version_id"]),
            content_digest=str(row[f"{prefix}_digest"]),
            r2_key=str(row[f"{prefix}_r2_key"]),
        )

    return EmbeddedArticleCatalogReference(
        article=reference("article"),
        relevance=reference("relevance"),
        embedding=reference("embedding"),
    )
