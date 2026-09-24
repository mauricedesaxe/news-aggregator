from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import Field

from romanian_news import NewsModel, Sha256
from romanian_news.articles.extraction import normalize_article_url
from romanian_news.articles.models import (
    ArticleAcquisitionResult,
    ArticleCapture,
    ArticleFailureKind,
)
from romanian_news.catalog.artifacts import (
    ArtifactFile,
    artifact_file,
    artifact_statements,
    canonical_json,
    sha256,
    version_digests,
)
from romanian_news.catalog_transport import (
    advance_artifact_current_version_statement,
    catalog_batch,
    catalog_query,
)
from romanian_news.storage import publish_immutable_r2_objects


class NewsArticlePublication(NewsModel):
    run_ids: tuple[Sha256, ...]
    published_articles: Annotated[int, Field(ge=0)]
    unchanged_articles: Annotated[int, Field(ge=0)]
    uploaded_objects: Annotated[int, Field(ge=0)]
    reused_objects: Annotated[int, Field(ge=0)]


class ArticleCatalogState(NewsModel):
    alias_key: str
    published_at: datetime
    source_updated_at: datetime | None
    captured_at: datetime


class ArticleFailureAttempt(NewsModel):
    event_id: Sha256
    implementation_ref: str
    work_generation: Sha256
    failure_kind: ArticleFailureKind
    failure_fingerprint: Sha256
    retry_at: datetime


class ArticleFailureAttemptWrite(ArticleFailureAttempt):
    attempt_id: Sha256
    dagster_run_id: str
    retry_number: Annotated[int, Field(ge=0)]
    error: str
    attempted_at: datetime


class ArticleRecoveryOverride(NewsModel):
    recovery_id: Sha256
    event_id: Sha256
    base_work_generation: Sha256
    expected_work_generation: Sha256
    requested_by: Annotated[str, Field(min_length=1)]
    reason: Annotated[str, Field(min_length=1)]
    requested_at: datetime
    recovery_sequence: Annotated[int, Field(ge=0)] = 0

    @property
    def work_generation(self) -> Sha256:
        return sha256(f"{self.base_work_generation}:recovery:{self.recovery_id}".encode())


def read_article_catalog_states(
    alias_keys: tuple[str, ...],
) -> dict[str, ArticleCatalogState]:
    """Read current article state by normalized URL alias."""
    if not alias_keys:
        return {}
    rows = catalog_query(
        """
        SELECT alias.alias_key,
               COALESCE(checkpoint.published_at, metadata.published_at) AS published_at,
               COALESCE(checkpoint.source_updated_at, metadata.source_updated_at) AS source_updated_at,
               COALESCE(checkpoint.captured_at, metadata.captured_at) AS captured_at
        FROM news_article_aliases alias
        JOIN artifacts article ON article.id = alias.article_artifact_id
        JOIN news_article_versions metadata
          ON metadata.artifact_version_id = article.current_version_id
        LEFT JOIN news_article_checks checkpoint
          ON checkpoint.article_artifact_id = article.id
        WHERE alias.alias_key = ANY(%s)
        """,
        [list(alias_keys)],
    )
    states = tuple(
        ArticleCatalogState(
            alias_key=str(row["alias_key"]),
            published_at=datetime.fromisoformat(str(row["published_at"])),
            source_updated_at=(
                datetime.fromisoformat(str(row["source_updated_at"]))
                if row["source_updated_at"] is not None
                else None
            ),
            captured_at=datetime.fromisoformat(str(row["captured_at"])),
        )
        for row in rows
    )
    return {state.alias_key: state for state in states}


def read_article_failure_attempts(
    event_ids: tuple[Sha256, ...],
) -> tuple[ArticleFailureAttempt, ...]:
    """Read ordered failure attempts for one event set across implementations.

    Attempts carry the implementation that recorded them, but quarantine
    state must span deploys: the batch sensor plans work in the daemon
    process where the deploy sha is unknown, so ref-filtered reads would
    never see what runs recorded and poison articles would never
    quarantine. A changed implementation resets the unchanged counter
    naturally through its fingerprint.
    """
    if not event_ids:
        return ()
    rows = catalog_query(
        "SELECT event_id, implementation_ref, work_generation, failure_kind, "
        "failure_fingerprint, retry_at "
        "FROM news_article_failure_attempts "
        "WHERE event_id = ANY(%s) "
        "ORDER BY attempted_at, attempt_id",
        [list(event_ids)],
    )
    return tuple(
        ArticleFailureAttempt(
            event_id=str(row["event_id"]),
            implementation_ref=str(row["implementation_ref"]),
            work_generation=str(row["work_generation"]),
            failure_kind=ArticleFailureKind(str(row["failure_kind"])),
            failure_fingerprint=str(row["failure_fingerprint"]),
            retry_at=datetime.fromisoformat(str(row["retry_at"])),
        )
        for row in rows
    )


def write_article_failure_attempts(attempts: tuple[ArticleFailureAttemptWrite, ...]) -> None:
    """Persist article failure attempts as one retryable catalog batch."""
    statements = [
        (
            "INSERT INTO news_article_failure_attempts "
            "(attempt_id, event_id, implementation_ref, work_generation, dagster_run_id, "
            "retry_number, failure_kind, failure_fingerprint, error, attempted_at, retry_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                attempt.attempt_id,
                attempt.event_id,
                attempt.implementation_ref,
                attempt.work_generation,
                attempt.dagster_run_id,
                attempt.retry_number,
                attempt.failure_kind.value,
                attempt.failure_fingerprint,
                attempt.error,
                attempt.attempted_at.astimezone(UTC).isoformat(),
                attempt.retry_at.astimezone(UTC).isoformat(),
            ],
        )
        for attempt in attempts
    ]
    if statements:
        catalog_batch(statements, retry_transient_errors=True)


def read_article_recovery_overrides(
    event_ids: tuple[Sha256, ...],
) -> tuple[ArticleRecoveryOverride, ...]:
    if not event_ids:
        return ()
    rows = catalog_query(
        "SELECT recovery_id, recovery_sequence, event_id, base_work_generation, "
        "expected_work_generation, requested_by, reason, requested_at "
        "FROM news_article_recovery_overrides WHERE event_id = ANY(%s) "
        "ORDER BY event_id, recovery_sequence",
        [list(event_ids)],
    )
    return tuple(
        ArticleRecoveryOverride(
            recovery_id=str(row["recovery_id"]),
            event_id=str(row["event_id"]),
            base_work_generation=str(row["base_work_generation"]),
            expected_work_generation=str(row["expected_work_generation"]),
            requested_by=str(row["requested_by"]),
            reason=str(row["reason"]),
            requested_at=datetime.fromisoformat(str(row["requested_at"])),
            recovery_sequence=int(row["recovery_sequence"]),
        )
        for row in rows
    )


def write_article_recovery_overrides(overrides: tuple[ArticleRecoveryOverride, ...]) -> None:
    statements = [
        (
            "INSERT INTO news_article_recovery_overrides "
            "(recovery_id, event_id, base_work_generation, expected_work_generation, "
            "requested_by, reason, requested_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [
                override.recovery_id,
                override.event_id,
                override.base_work_generation,
                override.expected_work_generation,
                override.requested_by,
                override.reason,
                override.requested_at.astimezone(UTC).isoformat(),
            ],
        )
        for override in overrides
    ]
    if statements:
        catalog_batch(statements, retry_transient_errors=True)


def read_article_candidate_times(lower: datetime, upper: datetime) -> tuple[datetime, ...]:
    """Read distinct feed event and observation times in one UTC interval."""
    rows = catalog_query(
        "SELECT DISTINCT candidate_time FROM ("
        "SELECT published_at AS candidate_time FROM news_feed_entry_events "
        "WHERE published_at >= %s AND published_at < %s "
        "UNION ALL "
        "SELECT scheduled_slot AS candidate_time FROM news_feed_observations "
        "WHERE scheduled_slot >= %s AND scheduled_slot < %s) AS candidates",
        [
            lower.astimezone(UTC).isoformat(),
            upper.astimezone(UTC).isoformat(),
            lower.astimezone(UTC).isoformat(),
            upper.astimezone(UTC).isoformat(),
        ],
    )
    return tuple(datetime.fromisoformat(str(row["candidate_time"])) for row in rows)


def publish_articles(
    result: ArticleAcquisitionResult,
    implementation_ref: str,
) -> NewsArticlePublication:
    """Publish only material article changes with exact feed and page inputs."""
    if not implementation_ref:
        raise ValueError("Implementation reference is required")
    captures, identity_statements = _resolve_article_identities(result.captures)
    current_material = _current_article_material(captures)
    changed = tuple(
        capture
        for capture in captures
        if current_material.get(f"news:article:{capture.article.article_id}")
        != capture.article.material_digest
    )
    files = tuple(file for capture in changed for file in _article_files(capture))
    publication = publish_immutable_r2_objects((value.r2_key, value.content) for value in files)
    run_ids = []
    catalog_statements = list(identity_statements)
    feed_digests = version_digests(
        tuple(sorted({capture.source.feed_snapshot_version_id for capture in changed}))
    )
    for capture in changed:
        run_id, statements = _article_catalog_statements(
            capture,
            implementation_ref,
            feed_digests[capture.source.feed_snapshot_version_id],
        )
        catalog_statements.extend(statements)
        run_ids.append(run_id)
    for capture in captures:
        if capture in changed:
            continue
        catalog_statements.extend(_article_alias_statements(capture))
    catalog_statements.extend(_article_check_statements(captures))
    catalog_batch(catalog_statements)
    return NewsArticlePublication(
        run_ids=tuple(run_ids),
        published_articles=len(changed),
        unchanged_articles=len(captures) - len(changed),
        uploaded_objects=publication.uploaded_objects,
        reused_objects=publication.reused_objects,
    )


def _article_check_statements(
    captures: tuple[ArticleCapture, ...],
) -> list[tuple[str, list[object]]]:
    statements = []
    for capture in captures:
        article = capture.article
        statements.append(
            (
                """
                INSERT INTO news_article_checks VALUES (%s, %s, %s, %s)
                ON CONFLICT(article_artifact_id) DO UPDATE SET
                    published_at = excluded.published_at,
                    source_updated_at = excluded.source_updated_at,
                    captured_at = excluded.captured_at
                WHERE excluded.captured_at > news_article_checks.captured_at
                """,
                [
                    f"news:article:{article.article_id}",
                    article.published_at.astimezone(UTC).isoformat(),
                    article.source_updated_at.astimezone(UTC).isoformat()
                    if article.source_updated_at
                    else None,
                    capture.captured_at.astimezone(UTC).isoformat(),
                ],
            )
        )
    return statements


def _resolve_article_identities(
    captures: tuple[ArticleCapture, ...],
) -> tuple[tuple[ArticleCapture, ...], tuple[tuple[str, list[object]], ...]]:
    alias_keys = tuple(
        sorted({alias for capture in captures for alias in _article_aliases(capture)})
    )
    aliases: dict[str, str] = {}
    if alias_keys:
        placeholders = ", ".join("%s" for _ in alias_keys)
        aliases = {
            str(row["alias_key"]): str(row["article_artifact_id"])
            for row in catalog_query(
                f"SELECT alias.alias_key, "
                f"COALESCE(merge.canonical_artifact_id, alias.article_artifact_id) "
                f"AS article_artifact_id FROM news_article_aliases alias "
                f"LEFT JOIN news_article_identity_merges merge "
                f"ON merge.duplicate_artifact_id = alias.article_artifact_id "
                f"WHERE alias.alias_key IN ({placeholders})",
                list(alias_keys),
            )
        }
    resolved = []
    merge_statements = []
    for capture in captures:
        capture_aliases = _article_aliases(capture)
        matches = {aliases[alias] for alias in capture_aliases if alias in aliases}
        if len(matches) > 1:
            canonical_key = f"url:{capture.article.canonical_url}"
            artifact_id = aliases[canonical_key]
            for duplicate_id in sorted(matches - {artifact_id}):
                merge_statements.extend(
                    (
                        (
                            "INSERT INTO news_article_identity_merges VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                            [
                                duplicate_id,
                                artifact_id,
                                capture.captured_at.astimezone(UTC).isoformat(),
                            ],
                        ),
                        (
                            "UPDATE artifacts SET lifecycle_state = 'superseded', "
                            "current_version_id = NULL, current_run_id = NULL "
                            "WHERE id = %s AND kind = 'news_article'",
                            [duplicate_id],
                        ),
                    )
                )
                aliases = {
                    alias: artifact_id if value == duplicate_id else value
                    for alias, value in aliases.items()
                }
        else:
            artifact_id = next(iter(matches), f"news:article:{capture.article.article_id}")
        article_hash = artifact_id.removeprefix("news:article:")
        resolved.append(
            capture.model_copy(
                update={"article": capture.article.model_copy(update={"article_id": article_hash})}
            )
        )
    return tuple(resolved), tuple(merge_statements)


def _current_article_material(captures: tuple[ArticleCapture, ...]) -> dict[str, str]:
    artifact_ids = tuple(
        sorted({f"news:article:{capture.article.article_id}" for capture in captures})
    )
    if not artifact_ids:
        return {}
    placeholders = ", ".join("%s" for _ in artifact_ids)
    rows = catalog_query(
        f"""
        SELECT a.id AS artifact_id, metadata.material_digest
        FROM artifacts a
        JOIN news_article_versions metadata
          ON metadata.artifact_version_id = a.current_version_id
        WHERE a.id IN ({placeholders}) AND a.kind = 'news_article'
        """,
        list(artifact_ids),
    )
    return {str(row["artifact_id"]): str(row["material_digest"]) for row in rows}


def _article_files(capture: ArticleCapture) -> tuple[ArtifactFile, ...]:
    article = capture.article
    article_artifact_id = f"news:article:{article.article_id}"
    article_content = canonical_json(article.model_dump(mode="json"))
    article_content_digest = sha256(article_content)
    article_file = artifact_file(
        artifact_id=article_artifact_id,
        artifact_kind="news_article",
        title=article.title,
        content=article_content,
        r2_key=f"news/articles/{article.article_id}/{article_content_digest}.json",
        media_type="application/json",
    )
    if capture.page_content is None or capture.page_content_digest is None:
        return (article_file,)
    page_file = artifact_file(
        artifact_id=f"news:page:{article.article_id}",
        artifact_kind="news_page",
        title=f"Page capture: {article.canonical_url}",
        content=capture.page_content,
        r2_key=f"news/pages/{article.article_id}/{capture.page_content_digest}.html",
        media_type="text/html",
    )
    return page_file, article_file


def _article_catalog_statements(
    capture: ArticleCapture,
    implementation_ref: str,
    feed_content_digest: Sha256,
) -> tuple[Sha256, list[tuple[str, list[object]]]]:
    article = capture.article
    files = _article_files(capture)
    article_file = files[-1]
    page_file = files[0] if len(files) == 2 else None
    run_id = sha256(
        canonical_json(
            {
                "article_version_id": article_file.version_id,
                "feed_entry_event_id": capture.source.event_id,
                "dlt_load_id": capture.source.dlt_load_id,
                "feed_snapshot_version_id": capture.source.feed_snapshot_version_id,
                "implementation_ref": implementation_ref,
                "page_capture_version_id": page_file.version_id if page_file else None,
            }
        )
    )
    timestamp = capture.captured_at.astimezone(UTC).isoformat()
    statements: list[tuple[str, list[object]]] = []
    if page_file:
        statements.extend(artifact_statements(page_file, timestamp, produced_by_run_id=None))
        statements.append(
            advance_artifact_current_version_statement(page_file.artifact_id, page_file.version_id)
        )
    statements.append(
        (
            "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                "news.normalize_article",
                "python",
                implementation_ref,
                canonical_json(
                    {
                        "extractor": "trafilatura-2.2.0",
                        "material_schema": "article-material-v1",
                    }
                ).decode(),
                "chartly",
                "publishing",
                f"news.normalize_article:{run_id}",
                None,
                timestamp,
                None,
            ],
        )
    )
    statements.append(
        (
            "INSERT INTO run_inputs VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                0,
                capture.source.feed_snapshot_version_id,
                "feed_entry",
                canonical_json(
                    {
                        "event_id": capture.source.event_id,
                        "dlt_load_id": capture.source.dlt_load_id,
                        "feed_id": capture.source.entry.feed_id,
                        "source_id": capture.source.entry.source_id,
                    }
                ).decode(),
                feed_content_digest,
                "dlt_feed_entry",
                None,
            ],
        )
    )
    if page_file:
        statements.append(
            (
                "INSERT INTO run_inputs VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    run_id,
                    1,
                    page_file.version_id,
                    "page_capture",
                    canonical_json({"url": capture.page_url}).decode(),
                    page_file.content_digest,
                    "whole_file",
                    canonical_json(
                        {
                            "latency_ms": capture.latency_ms,
                            "retrieval_error": capture.retrieval_error,
                        }
                    ).decode(),
                ],
            )
        )
    statements.extend(artifact_statements(article_file, timestamp, produced_by_run_id=run_id))
    statements.extend(_article_alias_statements(capture))
    statements.append(
        (
            "INSERT INTO news_article_versions VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
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
                capture.source.feed_snapshot_version_id,
                page_file.version_id if page_file else None,
                timestamp,
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


def _advance_article_head_statement(
    article_file: ArtifactFile,
    run_id: Sha256,
) -> tuple[str, list[object]]:
    return (
        """
        UPDATE artifacts
        SET current_version_id = %s, current_run_id = %s, title = %s
        WHERE id = %s
          AND (
              current_version_id IS NULL
              OR (
                  SELECT (COALESCE(source_updated_at, published_at), captured_at,
                          artifact_version_id)
                  FROM news_article_versions
                  WHERE artifact_version_id = %s
              ) > (
                  SELECT (COALESCE(source_updated_at, published_at), captured_at,
                          artifact_version_id)
                  FROM news_article_versions
                  WHERE artifact_version_id = artifacts.current_version_id
              )
          )
        """,
        [
            article_file.version_id,
            run_id,
            article_file.title,
            article_file.artifact_id,
            article_file.version_id,
        ],
    )


def _article_alias_statements(capture: ArticleCapture) -> list[tuple[str, list[object]]]:
    """Map the capture's canonical and feed URLs onto the article artifact.

    Every capture needs its aliases recorded, changed or not: the article
    planner looks state up by the feed event's own alias, so an unchanged
    capture that skips this write would be re-acquired forever.
    """
    artifact_id = f"news:article:{capture.article.article_id}"
    timestamp = capture.captured_at.astimezone(UTC).isoformat()
    return [
        (
            "INSERT INTO news_article_aliases VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [alias_key, artifact_id, alias_type, timestamp],
        )
        for alias_key, alias_type in _article_aliases(capture).items()
    ]


def _article_aliases(capture: ArticleCapture) -> dict[str, str]:
    aliases = {f"url:{capture.article.canonical_url}": "canonical_url"}
    entry = capture.source.entry
    entry_url_host = entry.url.host
    if entry_url_host is None:
        raise ValueError(f"Feed entry URL has no host: {entry.url}")
    entry_url = normalize_article_url(str(entry.url), (entry_url_host,))
    if entry_url != str(capture.article.canonical_url):
        aliases[f"url:{entry_url}"] = "redirect"
    return aliases
