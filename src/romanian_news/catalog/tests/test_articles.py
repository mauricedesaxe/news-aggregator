import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from pydantic import HttpUrl

import romanian_news.catalog.schema as news_schema
import romanian_news.catalog_transport as catalog_transport
from romanian_news.articles.extraction import extract_article
from romanian_news.articles.models import (
    ArticleAcquisitionResult,
    ArticleCapture,
    ArticleFailureKind,
)
from romanian_news.catalog.articles import (
    ArticleFailureAttemptWrite,
    ArticleRecoveryOverride,
    _resolve_article_identities,
    publish_articles,
    read_article_catalog_states,
    read_article_failure_attempts,
    read_article_recovery_overrides,
    write_article_failure_attempts,
    write_article_recovery_overrides,
)
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog_transport import catalog_batch
from romanian_news.feeds.models import CatalogedFeedEntry, FeedEntry, FeedSpec
from romanian_news.feeds.registry import feed_registry


def test_article_catalog_states_parse_typed_dates(monkeypatch) -> None:
    alias = "url:https://hotnews.ro/article"
    monkeypatch.setattr(
        "romanian_news.catalog.articles.catalog_query",
        lambda _sql, _parameters: [
            {
                "alias_key": alias,
                "published_at": "2026-09-01T08:00:00+00:00",
                "source_updated_at": None,
                "captured_at": "2026-09-01T09:00:00+00:00",
            }
        ],
    )

    state = read_article_catalog_states((alias,))[alias]

    assert state.published_at == datetime.fromisoformat("2026-09-01T08:00:00+00:00")
    assert state.source_updated_at is None


_TEST_POSTGRES_DSN = os.getenv("NEWS_TEST_POSTGRES_DSN")


@pytest.fixture
def article_recovery_postgres(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    base_dsn = _TEST_POSTGRES_DSN
    if base_dsn is None:
        raise RuntimeError("NEWS_TEST_POSTGRES_DSN is required")
    schema = f"article_catalog_{uuid4().hex}"
    with psycopg.connect(base_dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    schema_dsn = make_conninfo(base_dsn, options=f"-csearch_path={schema}")
    monkeypatch.setattr(news_schema, "NEWS_POSTGRES_DSN", schema_dsn)
    monkeypatch.setattr(catalog_transport, "NEWS_POSTGRES_DSN", schema_dsn)
    ensure_news_catalog_schema()
    yield schema_dsn
    with psycopg.connect(base_dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def _seed_cataloged_feed_entry(dsn: str, event_id: str) -> None:
    observed_at = datetime(2026, 9, 1, tzinfo=UTC)
    with psycopg.connect(dsn) as connection:
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
            "visibility, created_at) VALUES ('news:feed:hotnews', 'news_feed', 'HotNews', "
            "'source', 'current', 'private', %s)",
            (observed_at,),
        )
        connection.execute(
            "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
            "created_at) VALUES ('feed-v1', 'news:feed:hotnews', 1, 'digest', %s)",
            (observed_at,),
        )
        connection.execute(
            "INSERT INTO news_dlt_loads (load_id, artifact_version_id, registered_at) "
            "VALUES ('load-1', 'feed-v1', %s)",
            (observed_at,),
        )
        connection.execute(
            "INSERT INTO news_feed_entry_events (event_id, dlt_load_id, registry_version_id, "
            "feed_snapshot_version_id, feed_id, source_id, original_url, published_at, "
            "source_updated_at, observed_at) VALUES (%s, 'load-1', 'registry-v1', 'feed-v1', "
            "'hotnews', 'source-1', 'https://hotnews.ro/stiri/eveniment/article-1', %s, NULL, %s)",
            (event_id, observed_at, observed_at),
        )
        connection.execute(
            "INSERT INTO news_feed_entry_event_versions (event_id, version_id) " "VALUES (%s, %s)",
            (event_id, event_id),
        )


@pytest.mark.skipif(
    _TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_article_recovery_override_round_trip_normalizes_utc_and_orders_by_replay_sequence(
    article_recovery_postgres,
) -> None:
    event_id = "a" * 64
    _seed_cataloged_feed_entry(article_recovery_postgres, event_id)

    assert read_article_recovery_overrides(()) == ()
    assert read_article_recovery_overrides(("f" * 64,)) == ()

    write_article_recovery_overrides(
        (
            ArticleRecoveryOverride(
                recovery_id="b" * 64,
                event_id=event_id,
                base_work_generation="c" * 64,
                expected_work_generation="d" * 64,
                requested_by="operator",
                reason="Parser updated",
                requested_at=datetime.fromisoformat("2026-09-01T06:00:00+03:00"),
            ),
        )
    )
    write_article_recovery_overrides(
        (
            ArticleRecoveryOverride(
                recovery_id="e" * 64,
                event_id=event_id,
                base_work_generation="c" * 64,
                expected_work_generation="0" * 64,
                requested_by="operator",
                reason="Parser updated again",
                requested_at=datetime.fromisoformat("2026-09-01T05:00:00-04:00"),
            ),
        )
    )

    overrides = read_article_recovery_overrides((event_id,))

    assert [override.requested_at for override in overrides] == [
        datetime(2026, 9, 1, 3, tzinfo=UTC),
        datetime(2026, 9, 1, 9, tzinfo=UTC),
    ]
    assert [override.recovery_sequence for override in overrides] == sorted(
        override.recovery_sequence for override in overrides
    )
    assert all(override.work_generation != override.base_work_generation for override in overrides)
    assert ArticleRecoveryOverride.model_validate(overrides[0].model_dump()) == overrides[0]


@contextmanager
def _catalog_connection(dsn: str) -> Iterator[psycopg.Connection[dict[str, Any]]]:
    with psycopg.connect(dsn, row_factory=cast(Any, dict_row)) as connection:
        yield cast(psycopg.Connection[dict[str, Any]], connection)


def _hotnews_feed() -> FeedSpec:
    return next(value for value in feed_registry().feeds if value.id == "hotnews")


def _seed_feed_snapshot_version(dsn: str) -> None:
    observed_at = datetime(2026, 9, 1, tzinfo=UTC)
    with psycopg.connect(dsn) as connection:
        connection.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
            "visibility, created_at) VALUES ('news:feed:hotnews', 'news_feed', 'HotNews', "
            "'source', 'current', 'private', %s)",
            (observed_at,),
        )
        connection.execute(
            "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
            "created_at) VALUES (%s, 'news:feed:hotnews', 1, 'digest', %s)",
            ("c" * 64, observed_at),
        )


def _capture_for_publish(
    entry_url: str,
    event_id: str,
    *,
    canonical_url: str | None = None,
) -> ArticleCapture:
    entry = FeedEntry(
        feed_id="hotnews",
        source_id="source-1",
        url=HttpUrl(entry_url),
        title="Titlu",
        summary="Rezumat suficient pentru extragere. " * 4,
        feed_content=None,
        published_at=datetime.fromisoformat("2026-08-31T12:00:00+03:00"),
        source_updated_at=None,
        author=None,
    )
    article = extract_article(entry, _hotnews_feed(), html=None)
    if canonical_url is not None:
        article = article.model_copy(update={"canonical_url": HttpUrl(canonical_url)})
    return ArticleCapture(
        article=article,
        source=CatalogedFeedEntry(
            event_id=event_id,
            dlt_load_id="load-1",
            registry_version_id="b" * 64,
            feed_snapshot_version_id="c" * 64,
            observed_at=datetime.fromisoformat("2026-08-31T13:00:00+03:00"),
            feed_id=entry.feed_id,
            source_id=entry.source_id,
            url=entry.url,
            published_at=entry.published_at,
            source_updated_at=entry.source_updated_at,
            entry=entry,
        ),
        captured_at=datetime(2026, 9, 1, tzinfo=UTC),
        page_url=None,
        page_content=None,
        page_content_digest=None,
        latency_ms=0,
        retrieval_error=None,
    )


@pytest.mark.skipif(
    _TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_article_publication_records_dlt_event_and_load_lineage_in_run_inputs(
    article_recovery_postgres,
    monkeypatch,
) -> None:
    _seed_feed_snapshot_version(article_recovery_postgres)
    capture = _capture_for_publish("https://hotnews.ro/source-1", "a" * 64)
    monkeypatch.setattr(
        "romanian_news.catalog.articles.publish_immutable_r2_objects",
        lambda _objects: SimpleNamespace(uploaded_objects=0, reused_objects=0),
    )

    publication = publish_articles(
        ArticleAcquisitionResult(captures=(capture,), skipped_entries=0),
        "git:test",
    )

    with _catalog_connection(article_recovery_postgres) as catalog:
        inputs = catalog.execute(
            "SELECT run_inputs.position, run_inputs.artifact_version_id, run_inputs.role, "
            "run_inputs.locator_json, run_inputs.selection_method FROM run_inputs "
            "JOIN runs ON runs.id = run_inputs.run_id "
            "WHERE runs.operation_key = 'news.normalize_article' ORDER BY run_inputs.position"
        ).fetchall()
    assert publication.published_articles == 1
    assert [row["position"] for row in inputs] == [0]
    assert inputs[0]["artifact_version_id"] == "c" * 64
    assert inputs[0]["role"] == "feed_entry"
    assert inputs[0]["selection_method"] == "dlt_feed_entry"
    assert inputs[0]["locator_json"]["event_id"] == "a" * 64
    assert inputs[0]["locator_json"]["dlt_load_id"] == "load-1"


@pytest.mark.skipif(
    _TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_article_identity_prefers_current_page_canonical_and_supersedes_duplicate(
    article_recovery_postgres,
    monkeypatch,
) -> None:
    _seed_feed_snapshot_version(article_recovery_postgres)
    monkeypatch.setattr(
        "romanian_news.catalog.articles.publish_immutable_r2_objects",
        lambda _objects: SimpleNamespace(uploaded_objects=0, reused_objects=0),
    )
    duplicate_url = "https://hotnews.ro/stiri/eveniment/article-1"
    canonical_url = "https://hotnews.ro/opinii/agora/article-1"
    duplicate = _capture_for_publish(duplicate_url, "a" * 64)
    winner = _capture_for_publish(canonical_url, "b" * 64)
    publish_articles(ArticleAcquisitionResult(captures=(duplicate,), skipped_entries=0), "git:test")
    publish_articles(ArticleAcquisitionResult(captures=(winner,), skipped_entries=0), "git:test")
    duplicate_artifact_id = f"news:article:{duplicate.article.article_id}"
    winner_artifact_id = f"news:article:{winner.article.article_id}"
    merged = _capture_for_publish(duplicate_url, "c" * 64, canonical_url=canonical_url)

    resolved, merge_statements = _resolve_article_identities((merged,))

    assert resolved[0].article.article_id == winner.article.article_id
    catalog_batch(merge_statements)

    with _catalog_connection(article_recovery_postgres) as catalog:
        merges = catalog.execute(
            "SELECT duplicate_artifact_id, canonical_artifact_id, merged_at "
            "FROM news_article_identity_merges"
        ).fetchall()
        superseded = catalog.execute(
            "SELECT lifecycle_state, current_version_id FROM artifacts WHERE id = %s",
            (duplicate_artifact_id,),
        ).fetchone()
    assert [(row["duplicate_artifact_id"], row["canonical_artifact_id"]) for row in merges] == [
        (duplicate_artifact_id, winner_artifact_id)
    ]
    assert merges[0]["merged_at"] == merged.captured_at
    assert superseded is not None
    assert superseded["lifecycle_state"] == "superseded"
    assert superseded["current_version_id"] is None

    replay, _statements = _resolve_article_identities(
        (_capture_for_publish(duplicate_url, "d" * 64),)
    )

    assert replay[0].article.article_id == winner.article.article_id


@pytest.mark.skipif(
    _TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_article_failure_write_serializes_enum_and_utc_times(article_recovery_postgres) -> None:
    event_id = "b" * 64
    _seed_cataloged_feed_entry(article_recovery_postgres, event_id)
    attempt = ArticleFailureAttemptWrite(
        attempt_id="a" * 64,
        event_id=event_id,
        implementation_ref="git:test",
        work_generation="c" * 64,
        failure_kind=ArticleFailureKind.DETERMINISTIC,
        failure_fingerprint="d" * 64,
        retry_at=datetime.fromisoformat("2026-09-01T03:15:00+03:00"),
        dagster_run_id="run-1",
        retry_number=2,
        error="invalid article",
        attempted_at=datetime.fromisoformat("2026-09-01T03:00:00+03:00"),
    )

    write_article_failure_attempts((attempt,))

    with _catalog_connection(article_recovery_postgres) as catalog:
        row = catalog.execute(
            "SELECT failure_kind, attempted_at, retry_at, dagster_run_id, retry_number "
            "FROM news_article_failure_attempts WHERE attempt_id = %s",
            (attempt.attempt_id,),
        ).fetchone()
    attempts = read_article_failure_attempts((event_id,))
    assert row is not None
    assert row["failure_kind"] == "deterministic"
    assert row["attempted_at"] == datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    assert row["retry_at"] == datetime(2026, 9, 1, 0, 15, tzinfo=UTC)
    assert row["dagster_run_id"] == "run-1"
    assert row["retry_number"] == 2
    assert [value.retry_at for value in attempts] == [datetime(2026, 9, 1, 0, 15, tzinfo=UTC)]
    assert attempts[0].failure_kind is ArticleFailureKind.DETERMINISTIC


@pytest.mark.skipif(
    _TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_publish_records_feed_aliases_for_unchanged_captures(
    article_recovery_postgres,
    monkeypatch,
) -> None:
    _seed_feed_snapshot_version(article_recovery_postgres)
    capture = _capture_for_publish(
        "https://hotnews.ro/stiri/eveniment/article-1",
        "a" * 64,
        canonical_url="https://hotnews.ro/stiri/politica/article-1",
    )
    artifact_id = f"news:article:{capture.article.article_id}"
    monkeypatch.setattr(
        "romanian_news.catalog.articles.publish_immutable_r2_objects",
        lambda _objects: SimpleNamespace(uploaded_objects=0, reused_objects=0),
    )
    result = ArticleAcquisitionResult(captures=(capture,), skipped_entries=0)

    first = publish_articles(result, "git:test")
    second = publish_articles(result, "git:test")

    assert first.published_articles == 1
    assert second.published_articles == 0
    assert second.unchanged_articles == 1
    assert second.run_ids == ()
    with _catalog_connection(article_recovery_postgres) as catalog:
        aliases = catalog.execute(
            "SELECT alias_key, article_artifact_id, alias_type, first_seen_at "
            "FROM news_article_aliases"
        ).fetchall()
    assert {
        (row["alias_key"], row["article_artifact_id"], row["alias_type"]) for row in aliases
    } == {
        ("url:https://hotnews.ro/stiri/politica/article-1", artifact_id, "canonical_url"),
        ("url:https://hotnews.ro/stiri/eveniment/article-1", artifact_id, "redirect"),
    }
    assert [row["first_seen_at"] for row in aliases] == [capture.captured_at, capture.captured_at]
