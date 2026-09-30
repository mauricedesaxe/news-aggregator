"""Publish historical articles from immutable publisher archive captures."""

from __future__ import annotations

from datetime import UTC

from romanian_news import Sha256
from romanian_news.archive.articles import ArchiveArticleCapture
from romanian_news.articles.extraction import normalize_article_url
from romanian_news.catalog.articles import NewsArticlePublication, _advance_article_head_statement
from romanian_news.catalog.artifacts import (
    ArtifactFile,
    artifact_file,
    artifact_statements,
)
from romanian_news.catalog_transport import (
    advance_artifact_current_version_statement,
    catalog_batch,
    catalog_query,
)
from romanian_news.identity import canonical_json, sha256
from romanian_news.storage import publish_immutable_r2_objects


def _capture_file(capture: ArchiveArticleCapture) -> ArtifactFile:
    content = canonical_json(
        {
            "capture_id": capture.capture_id,
            "observation_id": capture.observation_id,
            "discovered_url": str(capture.discovered_url),
            "final_url": str(capture.final_url),
            "page_sha256": capture.page_sha256,
            "fetched_at": capture.fetched_at.astimezone(UTC).isoformat(),
            "published_at": capture.article.published_at.astimezone(UTC).isoformat(),
            "modified_at": capture.article.source_updated_at.astimezone(UTC).isoformat()
            if capture.article.source_updated_at
            else None,
            "publication_evidence": capture.publication_evidence,
        }
    )
    digest = sha256(content)
    return artifact_file(
        artifact_id=f"news:archive-capture:{capture.capture_id}",
        artifact_kind="news_archive_capture",
        title=f"Archive capture: {capture.discovered_url}",
        content=content,
        r2_key=f"news/archive/captures/{capture.capture_id}/{digest}.json",
        media_type="application/json",
    )


def _article_file(capture: ArchiveArticleCapture) -> ArtifactFile:
    article = capture.article
    content = canonical_json(article.model_dump(mode="json"))
    digest = sha256(content)
    return artifact_file(
        artifact_id=f"news:article:{article.article_id}",
        artifact_kind="news_article",
        title=article.title,
        content=content,
        r2_key=f"news/articles/{article.article_id}/{digest}.json",
        media_type="application/json",
    )


def _aliases(capture: ArchiveArticleCapture) -> tuple[tuple[str, str], ...]:
    article = capture.article
    canonical = f"url:{article.canonical_url}"
    aliases = [(canonical, "canonical_url")]
    discovered_host = capture.discovered_url.host
    if discovered_host is None:
        raise ValueError("Archive discovery URL has no host")
    discovered = normalize_article_url(str(capture.discovered_url), (discovered_host,))
    if discovered != str(article.canonical_url):
        aliases.append((f"url:{discovered}", "redirect"))
    return tuple(aliases)


def _current_material(artifact_id: str) -> Sha256 | None:
    rows = catalog_query(
        "SELECT metadata.material_digest FROM artifacts artifact "
        "JOIN news_article_versions metadata "
        "ON metadata.artifact_version_id = artifact.current_version_id "
        "WHERE artifact.id = %s AND artifact.kind = 'news_article'",
        [artifact_id],
    )
    return str(rows[0]["material_digest"]) if rows else None


def _check_aliases(artifact_id: str, aliases: tuple[tuple[str, str], ...]) -> None:
    rows = catalog_query(
        "SELECT alias_key, article_artifact_id FROM news_article_aliases "
        "WHERE alias_key = ANY(%s::text[])",
        [[key for key, _kind in aliases]],
    )
    if any(str(row["article_artifact_id"]) != artifact_id for row in rows):
        raise ValueError("Archive URL already belongs to a different article identity")


def _capture_statements(
    capture: ArchiveArticleCapture, source_file: ArtifactFile
) -> list[tuple[str, list[object]]]:
    article = capture.article
    fetched_at = capture.fetched_at.astimezone(UTC).isoformat()
    statements = artifact_statements(
        source_file, fetched_at, produced_by_run_id=None, authority_class="source"
    )
    statements.append(
        advance_artifact_current_version_statement(source_file.artifact_id, source_file.version_id)
    )
    statements.append(
        (
            "INSERT INTO news_archive_article_captures "
            "(id, observation_id, discovered_url, final_url, capture_artifact_version_id, "
            "page_sha256, fetched_at, published_at, modified_at, publication_evidence) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                capture.capture_id,
                capture.observation_id,
                str(capture.discovered_url),
                str(capture.final_url),
                source_file.version_id,
                capture.page_sha256,
                fetched_at,
                article.published_at.astimezone(UTC).isoformat(),
                article.source_updated_at.astimezone(UTC).isoformat()
                if article.source_updated_at
                else None,
                capture.publication_evidence,
            ],
        )
    )
    return statements


def _article_statements(
    capture: ArchiveArticleCapture,
    source_file: ArtifactFile,
    article_file: ArtifactFile,
    implementation_ref: str,
) -> tuple[Sha256, list[tuple[str, list[object]]]]:
    article = capture.article
    timestamp = capture.fetched_at.astimezone(UTC).isoformat()
    run_id = sha256(
        canonical_json(
            {
                "archive_capture_version_id": source_file.version_id,
                "article_version_id": article_file.version_id,
                "implementation_ref": implementation_ref,
            }
        )
    )
    statements: list[tuple[str, list[object]]] = [
        (
            "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING",
            [
                run_id,
                "news.normalize_archive_article",
                "python",
                implementation_ref,
                canonical_json(
                    {"extractor": "trafilatura-2.2.0", "material_schema": "article-material-v1"}
                ).decode(),
                "chartly",
                "publishing",
                f"news.normalize_archive_article:{run_id}",
                None,
                timestamp,
                None,
            ],
        ),
        (
            "INSERT INTO run_inputs VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING",
            [
                run_id,
                0,
                source_file.version_id,
                "archive_capture",
                canonical_json({"capture_id": capture.capture_id}).decode(),
                source_file.content_digest,
                "whole_file",
                None,
            ],
        ),
    ]
    statements.extend(artifact_statements(article_file, timestamp, produced_by_run_id=run_id))
    statements.append(
        (
            "INSERT INTO news_article_versions "
            "(artifact_version_id, article_artifact_id, outlet_id, canonical_url, "
            "published_at, source_updated_at, bucharest_day, material_digest, "
            "extraction_digest, feed_snapshot_version_id, page_capture_version_id, "
            "captured_at, archive_capture_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL, %s, %s) "
            "ON CONFLICT DO NOTHING",
            [
                article_file.version_id,
                article_file.artifact_id,
                article.outlet_id,
                str(article.canonical_url),
                article.published_at.astimezone(UTC).isoformat(),
                article.source_updated_at.astimezone(UTC).isoformat()
                if article.source_updated_at
                else None,
                article.bucharest_day.isoformat(),
                article.material_digest,
                article.extraction_digest,
                timestamp,
                capture.capture_id,
            ],
        )
    )
    statements.append(
        (
            "INSERT INTO run_outputs VALUES (%s, 0, %s, 'output') ON CONFLICT DO NOTHING",
            [run_id, article_file.version_id],
        )
    )
    statements.append(_advance_article_head_statement(article_file, run_id))
    statements.append(
        (
            "UPDATE runs SET status = 'completed', completed_at = %s "
            "WHERE id = %s AND status = 'publishing'",
            [timestamp, run_id],
        )
    )
    return run_id, statements


def publish_archive_article(
    capture: ArchiveArticleCapture, implementation_ref: str
) -> NewsArticlePublication:
    """Store one archive capture and publish a changed article version."""
    if not implementation_ref:
        raise ValueError("Implementation reference is required")
    source_file = _capture_file(capture)
    article_file = _article_file(capture)
    aliases = _aliases(capture)
    _check_aliases(article_file.artifact_id, aliases)
    changed = _current_material(article_file.artifact_id) != capture.article.material_digest
    files = (source_file, article_file) if changed else (source_file,)
    publication = publish_immutable_r2_objects((value.r2_key, value.content) for value in files)
    statements = _capture_statements(capture, source_file)
    run_ids: tuple[Sha256, ...] = ()
    if changed:
        run_id, article_statements = _article_statements(
            capture, source_file, article_file, implementation_ref
        )
        statements.extend(article_statements)
        run_ids = (run_id,)
    for alias_key, alias_type in aliases:
        statements.append(
            (
                "INSERT INTO news_article_aliases VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    alias_key,
                    article_file.artifact_id,
                    alias_type,
                    capture.fetched_at.astimezone(UTC).isoformat(),
                ],
            )
        )
    catalog_batch(statements, retry_transient_errors=True)
    return NewsArticlePublication(
        run_ids=run_ids,
        published_articles=int(changed),
        unchanged_articles=int(not changed),
        uploaded_objects=publication.uploaded_objects,
        reused_objects=publication.reused_objects,
    )
