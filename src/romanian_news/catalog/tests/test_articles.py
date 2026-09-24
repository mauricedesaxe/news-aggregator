import json
from datetime import UTC, datetime

from pydantic import HttpUrl

from romanian_news.articles.extraction import extract_article
from romanian_news.articles.models import (
    ArticleAcquisitionResult,
    ArticleCapture,
    ArticleFailureKind,
)
from romanian_news.catalog.articles import (
    ArticleFailureAttemptWrite,
    ArticleRecoveryOverride,
    _article_catalog_statements,
    _resolve_article_identities,
    publish_articles,
    read_article_catalog_states,
    read_article_recovery_overrides,
    write_article_failure_attempts,
    write_article_recovery_overrides,
)
from romanian_news.feeds.models import CatalogedFeedEntry, FeedEntry
from romanian_news.feeds.registry import feed_registry


def test_article_run_input_keeps_dlt_event_and_load_lineage() -> None:
    entry = FeedEntry(
        feed_id="hotnews",
        source_id="source-1",
        url=HttpUrl("https://hotnews.ro/source-1"),
        title="Titlu",
        summary="Rezumat suficient pentru extragere. " * 4,
        feed_content=None,
        published_at=datetime.fromisoformat("2026-08-31T12:00:00+03:00"),
        source_updated_at=None,
        author=None,
    )
    source = CatalogedFeedEntry(
        event_id="a" * 64,
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
    )
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    capture = ArticleCapture(
        article=extract_article(entry, feed, html=None),
        source=source,
        captured_at=datetime.now(UTC),
        page_url=None,
        page_content=None,
        page_content_digest=None,
        latency_ms=0,
        retrieval_error=None,
    )

    _run_id, statements = _article_catalog_statements(capture, "test", "d" * 64)

    feed_input = next(
        statement for statement in statements if "INSERT INTO run_inputs" in statement[0]
    )
    locator = json.loads(str(feed_input[1][4]))
    assert feed_input[1][2] == source.feed_snapshot_version_id
    assert feed_input[1][6] == "dlt_feed_entry"
    assert locator["event_id"] == source.event_id
    assert locator["dlt_load_id"] == source.dlt_load_id


def test_article_identity_prefers_current_page_canonical_and_supersedes_duplicate(
    monkeypatch,
) -> None:
    entry = FeedEntry(
        feed_id="hotnews",
        source_id="source-1",
        url=HttpUrl("https://hotnews.ro/opinii/article-1"),
        title="Titlu",
        summary="Rezumat suficient pentru extragere. " * 4,
        feed_content=None,
        published_at=datetime.fromisoformat("2026-08-31T12:00:00+03:00"),
        source_updated_at=None,
        author=None,
    )
    source = CatalogedFeedEntry(
        event_id="a" * 64,
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
    )
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    article = extract_article(entry, feed, html=None).model_copy(
        update={"canonical_url": "https://hotnews.ro/opinii/agora/article-1"}
    )
    capture = ArticleCapture(
        article=article,
        source=source,
        captured_at=datetime.now(UTC),
        page_url=str(article.canonical_url),
        page_content=None,
        page_content_digest=None,
        latency_ms=0,
        retrieval_error=None,
    )
    winner = "news:article:" + "d" * 64
    duplicate = "news:article:" + "e" * 64
    monkeypatch.setattr(
        "romanian_news.catalog.articles.catalog_query",
        lambda *_args: [
            {
                "alias_key": f"url:{article.canonical_url}",
                "article_artifact_id": winner,
            },
            {
                "alias_key": f"url:{entry.url}",
                "article_artifact_id": duplicate,
            },
        ],
    )

    resolved, statements = _resolve_article_identities((capture,))

    assert resolved[0].article.article_id == "d" * 64
    assert [parameters for _sql, parameters in statements] == [
        [duplicate, winner, capture.captured_at.isoformat()],
        [duplicate],
    ]


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


def test_article_recovery_override_read_and_write_preserve_utc(monkeypatch) -> None:
    event_id = "a" * 64
    recovery_id = "b" * 64
    base_generation = "c" * 64
    expected_generation = "d" * 64
    queried = []
    batches = []

    def query(sql, parameters):
        queried.append((sql, parameters))
        return [
            {
                "recovery_id": recovery_id,
                "recovery_sequence": 3,
                "event_id": event_id,
                "base_work_generation": base_generation,
                "expected_work_generation": expected_generation,
                "requested_by": "operator",
                "reason": "Parser updated",
                "requested_at": "2026-09-01T03:00:00+00:00",
            }
        ]

    monkeypatch.setattr("romanian_news.catalog.articles.catalog_query", query)
    monkeypatch.setattr(
        "romanian_news.catalog.articles.catalog_batch",
        lambda statements, **kwargs: batches.append((statements, kwargs)),
    )

    assert read_article_recovery_overrides(()) == ()
    override = read_article_recovery_overrides((event_id,))[0]
    assert queried[0][1] == [[event_id]]
    assert override.requested_at == datetime(2026, 9, 1, 3, tzinfo=UTC)
    assert override.recovery_sequence == 3
    assert override.work_generation != override.base_work_generation

    write_article_recovery_overrides(
        (
            override.model_copy(
                update={"requested_at": datetime.fromisoformat("2026-09-01T06:00:00+03:00")}
            ),
        )
    )
    assert batches[0][0][0][1] == [
        recovery_id,
        event_id,
        base_generation,
        expected_generation,
        "operator",
        "Parser updated",
        "2026-09-01T03:00:00+00:00",
    ]
    assert batches[0][1] == {"retry_transient_errors": True}
    assert "ON CONFLICT" not in batches[0][0][0][0]
    assert ArticleRecoveryOverride.model_validate(override.model_dump()) == override


def test_article_failure_write_serializes_enum_and_utc_times(monkeypatch) -> None:
    batches = []
    monkeypatch.setattr(
        "romanian_news.catalog.articles.catalog_batch",
        lambda statements, **kwargs: batches.append((statements, kwargs)),
    )
    attempt = ArticleFailureAttemptWrite(
        attempt_id="a" * 64,
        event_id="b" * 64,
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

    parameters = batches[0][0][0][1]
    assert parameters[6] == "deterministic"
    assert parameters[9:] == ["2026-09-01T00:00:00+00:00", "2026-09-01T00:15:00+00:00"]
    assert batches[0][1] == {"retry_transient_errors": True}


def test_publish_records_feed_aliases_for_unchanged_captures(monkeypatch) -> None:
    entry = FeedEntry(
        feed_id="hotnews",
        source_id="source-1",
        url=HttpUrl("https://hotnews.ro/stiri/eveniment/article-1"),
        title="Titlu",
        summary="Rezumat suficient pentru extragere. " * 4,
        feed_content=None,
        published_at=datetime.fromisoformat("2026-08-31T12:00:00+03:00"),
        source_updated_at=None,
        author=None,
    )
    source = CatalogedFeedEntry(
        event_id="a" * 64,
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
    )
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    article = extract_article(entry, feed, html=None).model_copy(
        update={"canonical_url": "https://hotnews.ro/stiri/politica/article-1"}
    )
    capture = ArticleCapture(
        article=article,
        source=source,
        captured_at=datetime.now(UTC),
        page_url=str(article.canonical_url),
        page_content=None,
        page_content_digest=None,
        latency_ms=0,
        retrieval_error=None,
    )
    artifact_id = f"news:article:{article.article_id}"
    batches = []
    monkeypatch.setattr(
        "romanian_news.catalog.articles._resolve_article_identities",
        lambda captures: (captures, []),
    )
    monkeypatch.setattr(
        "romanian_news.catalog.articles._current_article_material",
        lambda _captures: {artifact_id: article.material_digest},
    )
    monkeypatch.setattr(
        "romanian_news.catalog.articles.catalog_batch",
        lambda statements, **_kwargs: batches.append(statements),
    )

    publication = publish_articles(
        ArticleAcquisitionResult(captures=(capture,), skipped_entries=0),
        "git:test",
    )

    alias_inserts = [
        parameters
        for statement, parameters in batches[0]
        if statement.startswith("INSERT INTO news_article_aliases")
    ]
    alias_keys = {parameters[0] for parameters in alias_inserts}
    assert publication.published_articles == 0
    assert publication.unchanged_articles == 1
    assert publication.run_ids == ()
    assert f"url:{article.canonical_url}" in alias_keys
    assert "url:https://hotnews.ro/stiri/eveniment/article-1" in alias_keys
    assert all(parameters[1] == artifact_id for parameters in alias_inserts)
