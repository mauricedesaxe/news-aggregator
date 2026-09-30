from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import Field

from romanian_news import NewsModel, Sha256
from romanian_news.catalog.artifacts import (
    artifact_file,
    artifact_statements,
    run_exists,
    run_output_statement,
)
from romanian_news.catalog_transport import (
    advance_artifact_current_version_from_run_statement,
    catalog_batch,
)
from romanian_news.groups import DailyClusterOutput
from romanian_news.identity import canonical_json
from romanian_news.storage import publish_immutable_r2_objects


class NewsClusterPublication(NewsModel):
    run_id: Sha256
    version_id: Sha256
    uploaded_objects: Annotated[int, Field(ge=0)]
    reused_objects: Annotated[int, Field(ge=0)]


def publish_daily_clusters(
    output: DailyClusterOutput,
    implementation_ref: str,
) -> NewsClusterPublication:
    """Publish one complete deterministic cluster assignment for a Romanian day."""
    if not implementation_ref:
        raise ValueError("Implementation reference is required")
    artifact_id = f"news:clusters:{output.cluster_set.day.isoformat()}"
    file = artifact_file(
        artifact_id=artifact_id,
        artifact_kind="news_clusters",
        title=f"Romanian news clusters for {output.cluster_set.day.isoformat()}",
        content=output.content,
        r2_key=(f"news/clusters/{output.cluster_set.day.isoformat()}/{output.content_digest}.json"),
        media_type="application/json",
    )
    if run_exists(output.request_id):
        catalog_batch(
            [
                advance_artifact_current_version_from_run_statement(
                    file.artifact_id, file.version_id, output.request_id
                )
            ]
        )
        return NewsClusterPublication(
            run_id=output.request_id,
            version_id=file.version_id,
            uploaded_objects=0,
            reused_objects=0,
        )
    publication = publish_immutable_r2_objects(((file.r2_key, file.content),))
    timestamp = datetime.now(UTC).isoformat()
    statements: list[tuple[str, list[object]]] = [
        (
            "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                output.request_id,
                "news.cluster_day",
                "python",
                implementation_ref,
                canonical_json(
                    {
                        "algorithm": output.cluster_set.algorithm,
                        "embedding_model": output.cluster_set.embedding_model,
                        "threshold": output.cluster_set.threshold,
                    }
                ).decode(),
                "chartly",
                "publishing",
                f"news.cluster_day:{output.request_id}",
                None,
                timestamp,
                None,
            ],
        )
    ]
    inputs = [
        (article_index * 3 + offset, reference.version_id, role, reference.content_digest)
        for article_index, article in enumerate(output.articles)
        for offset, (reference, role) in enumerate(
            (
                (article.article, "article"),
                (article.relevance, "relevance"),
                (article.embedding, "embedding"),
            )
        )
    ]
    if inputs:
        statements.append(
            (
                "INSERT INTO run_inputs "
                "(run_id, position, artifact_version_id, role, locator_json, "
                "selected_content_digest, selection_method, retrieval_metadata_json) "
                "SELECT %s, input.position, input.artifact_version_id, input.role, "
                "NULL, input.content_digest, 'whole_file', NULL "
                "FROM unnest(%s::bigint[], %s::text[], %s::text[], %s::text[]) "
                "AS input(position, artifact_version_id, role, content_digest) "
                "ON CONFLICT DO NOTHING",
                [output.request_id, *([list(column) for column in zip(*inputs, strict=True)])],
            )
        )
    statements.extend(artifact_statements(file, timestamp, produced_by_run_id=output.request_id))
    statements.append(run_output_statement(output.request_id, file))
    statements.append(
        advance_artifact_current_version_from_run_statement(
            file.artifact_id, file.version_id, output.request_id
        )
    )
    statements.append(
        (
            "UPDATE runs SET status = 'completed', completed_at = %s "
            "WHERE id = %s AND status = 'publishing'",
            [timestamp, output.request_id],
        )
    )
    catalog_batch(statements)
    return NewsClusterPublication(
        run_id=output.request_id,
        version_id=file.version_id,
        uploaded_objects=publication.uploaded_objects,
        reused_objects=publication.reused_objects,
    )
