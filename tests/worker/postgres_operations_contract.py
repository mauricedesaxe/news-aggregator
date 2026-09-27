from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import dagster as dg
import dlt
import pytest
import requests
from pydantic import HttpUrl

from romanian_news.analysis.attempts import ModelCall
from romanian_news.analysis.relevance import ArticleAnalysisInput
from romanian_news.analysis.relevance_v3 import (
    RELEVANCE_V3_POLICY,
    ContextDecision,
    ContextGateResult,
    ExecutionMode,
    GateCall,
    RelevanceV3Output,
    production_relevance_v3_request_id,
    relevance_v3_request_id,
)
from romanian_news.articles.extraction import article_id, normalize_article_url
from romanian_news.articles.models import (
    ArticleFailureKind,
    ExtractedArticle,
)
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import (
    artifact_file,
    artifact_statements,
    sha256,
)
from romanian_news.catalog_transport import (
    advance_artifact_current_version_statement,
)
from romanian_news.feeds import acquisition as feed_acquisition
from romanian_news.feeds.registry import feed_registry
from romanian_news.storage import publish_immutable_r2_objects
from romanian_news.worker import definitions, operations
from romanian_news.worker.catalog_status import print_catalog_status
from tests.daily_report_catalog import REPORT_VERSION, daily_report, seed_daily_report
from tests.postgres_catalog import PostgresCatalog

DAY = date(2099, 9, 2)
NOW = datetime(2099, 9, 2, 12, tzinfo=UTC)


def test_feed_intake_publishes_every_registered_feed_and_is_idempotent(
    harness,
    postgres_catalog,
    fake_r2,
    fake_http,
    monkeypatch,
    tmp_path,
    news_day,
) -> None:
    registry = feed_registry()
    for feed in registry.feeds:
        fake_http.serve(str(feed.url), harness.rss(()))
    destination_dir = tmp_path / "dlt-destination"
    destination_dir.mkdir()
    monkeypatch.setattr(
        feed_acquisition,
        "_destination",
        lambda _request: dlt.destinations.filesystem(bucket_url=destination_dir.as_uri()),
    )
    scheduled_at = NOW

    references = operations.materialize_feed_intake(news_day, scheduled_at, tmp_path, "git:test")

    observation_rows = postgres_catalog.execute(
        "SELECT feed_id, status, artifact_version_id FROM news_feed_observations "
        "WHERE scheduled_slot = %s",
        (scheduled_at.isoformat(),),
    ).fetchall()
    assert len(observation_rows) == len(registry.feeds)
    assert {row["status"] for row in observation_rows} == {"ok"}
    assert {row["feed_id"] for row in observation_rows} == {feed.id for feed in registry.feeds}
    assert len(references.values) == len(registry.feeds)
    assert {value.version_id for value in references.values} == {
        row["artifact_version_id"] for row in observation_rows
    }
    for value in references.values:
        stored = fake_r2.objects[value.r2_key]
        assert hashlib.sha256(stored).hexdigest() == value.content_digest
        assert fake_r2.metadata[value.r2_key]["sha256"] == value.content_digest
    snapshot_rows = postgres_catalog.execute(
        "SELECT file.r2_key, file.content_digest FROM artifacts artifact "
        "JOIN artifact_files file ON file.artifact_version_id = artifact.current_version_id "
        "WHERE artifact.kind = 'news_feed'"
    ).fetchall()
    assert len(snapshot_rows) == len(registry.feeds)
    for row in snapshot_rows:
        assert hashlib.sha256(fake_r2.objects[row["r2_key"]]).hexdigest() == row["content_digest"]

    repeat = operations.materialize_feed_intake(news_day, scheduled_at, tmp_path, "git:test")

    assert repeat == references
    assert postgres_catalog.execute(
        "SELECT count(*) AS count FROM news_feed_observations"
    ).fetchone()["count"] == len(registry.feeds)


def _event_id_for_url(catalog: PostgresCatalog, url: str) -> str:
    row = catalog.execute(
        "SELECT event_id FROM news_feed_entry_events WHERE original_url = %s", (url,)
    ).fetchone()
    assert row is not None
    return str(row["event_id"])


def test_article_materializer_publishes_exact_batch_and_rechecks_state(
    harness,
    postgres_catalog,
    fake_r2,
    fake_http,
    monkeypatch,
    news_day,
) -> None:
    feed = harness.feed("testfeed", "testoutlet", "https://feed.test/rss", "feed.test")
    registry = harness.registry(feed)
    monkeypatch.setattr(operations, "feed_registry", lambda: registry)
    url = "https://feed.test/stire-ultima-ora"
    title = "Guvernul a aprobat un pachet de măsuri pentru energia din anul următor"
    paragraphs = (
        "Executivul a anunțat că noile prevederi intră în vigoare la începutul anului viitor.",
        "Ministerul energiei estimează un efect semnificativ asupra facturilor populației.",
        "Analistii economici salută măsura, dar avertizează asupra implementării rapide.",
    )
    page = harness.article(title, paragraphs)
    observed_at = NOW
    capture = harness.seed(
        fake_http,
        registry,
        feed,
        ((url, title, observed_at, "Rezumat pe scurt al știrii."),),
        observed_at,
    )
    fake_http.serve(url, page)
    event_ids = harness.event_ids(capture, registry)
    assert len(event_ids) == 1

    result = operations.materialize_articles(
        news_day, event_ids, "git:test", run_id="run-1", retry_number=0
    )

    assert result.complete
    assert result.acquired_event_ids == event_ids
    assert result.skipped_event_ids == ()
    assert result.failures == ()
    version_rows = postgres_catalog.execute(
        "SELECT version.canonical_url, version.bucharest_day, artifact.current_version_id "
        "FROM news_article_versions version "
        "JOIN artifacts artifact ON artifact.id = version.article_artifact_id "
        "WHERE artifact.kind = 'news_article'"
    ).fetchall()
    assert len(version_rows) == 1
    assert version_rows[0]["canonical_url"] == url
    assert version_rows[0]["bucharest_day"] == news_day
    assert [value.version_id for value in result.references.values] == [
        version_rows[0]["current_version_id"]
    ]
    run_row = postgres_catalog.execute(
        "SELECT id FROM runs WHERE operation_key = 'news.normalize_article' "
        "AND status = 'completed'"
    ).fetchone()
    assert run_row is not None
    assert (
        postgres_catalog.execute(
            "SELECT count(*) AS count FROM run_inputs WHERE run_id = %s", (run_row["id"],)
        ).fetchone()["count"]
        >= 1
    )
    page_rows = postgres_catalog.execute(
        "SELECT file.r2_key, file.content_digest FROM artifact_files file "
        "JOIN news_article_versions version ON version.page_capture_version_id "
        "= file.artifact_version_id"
    ).fetchall()
    assert len(page_rows) == 1
    assert fake_r2.objects[page_rows[0]["r2_key"]] == page
    assert (
        hashlib.sha256(fake_r2.objects[page_rows[0]["r2_key"]]).hexdigest()
        == page_rows[0]["content_digest"]
    )

    second = operations.materialize_articles(
        news_day, event_ids, "git:test", run_id="run-2", retry_number=0
    )

    assert second.complete
    assert second.acquired_event_ids == ()
    assert second.failures == ()
    assert second.references == result.references
    assert (
        postgres_catalog.execute("SELECT count(*) AS count FROM news_article_versions").fetchone()[
            "count"
        ]
        == 1
    )
    assert (
        postgres_catalog.execute(
            "SELECT count(*) AS count FROM runs WHERE operation_key = 'news.normalize_article'"
        ).fetchone()["count"]
        == 1
    )
    assert (
        postgres_catalog.execute(
            "SELECT count(*) AS count FROM news_article_failure_attempts"
        ).fetchone()["count"]
        == 0
    )


def test_article_materializer_records_item_failure_before_the_next_item(
    harness,
    postgres_catalog,
    fake_r2,
    fake_http,
    monkeypatch,
    news_day,
) -> None:
    feed = harness.feed("testfeed", "testoutlet", "https://feed.test/rss", "feed.test")
    registry = harness.registry(feed)
    monkeypatch.setattr(operations, "feed_registry", lambda: registry)
    good_url = "https://feed.test/stire-buna"
    missing_url = "https://feed.test/stire-lipsa"
    good_title = "Parlamentul a adoptat o lege importantă pentru piața energetică"
    good_page = harness.article(
        good_title,
        (
            "Legea modifică modul de calcul al prețurilor pentru energia regenerabilă.",
            "Autoritățile așteaptă efecte pozitive începând cu următorul an bugetar.",
        ),
    )
    observed_at = NOW
    harness.seed(
        fake_http,
        registry,
        feed,
        (
            (good_url, good_title, observed_at, "Descriere pe scurt."),
            (missing_url, "Lipsă", observed_at, ""),
        ),
        observed_at,
    )
    fake_http.serve(good_url, good_page)
    fake_http.serve(missing_url, b"", status=404)
    event_ids = tuple(
        _event_id_for_url(postgres_catalog, value) for value in (good_url, missing_url)
    )

    result = operations.materialize_articles(
        news_day, event_ids, "git:test", run_id="run-1", retry_number=0
    )

    assert result.acquired_event_ids == (event_ids[0],)
    assert [failure.event_id for failure in result.failures] == [event_ids[1]]
    assert result.failures[0].kind == ArticleFailureKind.DETERMINISTIC
    assert result.remaining_entries == 1
    attempt_rows = postgres_catalog.execute(
        "SELECT event_id, failure_kind FROM news_article_failure_attempts"
    ).fetchall()
    assert [(row["event_id"], row["failure_kind"]) for row in attempt_rows] == [
        (event_ids[1], "deterministic")
    ]
    good_rows = postgres_catalog.execute(
        "SELECT version.canonical_url FROM news_article_versions version "
        "JOIN artifacts artifact ON artifact.id = version.article_artifact_id "
        "WHERE artifact.kind = 'news_article'"
    ).fetchall()
    assert [row["canonical_url"] for row in good_rows] == [good_url]
    page_rows = postgres_catalog.execute(
        "SELECT file.r2_key FROM artifact_files file "
        "JOIN news_article_versions version ON version.page_capture_version_id "
        "= file.artifact_version_id"
    ).fetchall()
    assert len(page_rows) == 1
    assert fake_r2.objects[page_rows[0]["r2_key"]] == good_page


def test_article_materializer_publishes_each_success_before_the_next_item(
    harness,
    postgres_catalog,
    fake_r2,
    fake_http,
    monkeypatch,
    news_day,
) -> None:
    feed = harness.feed("testfeed", "testoutlet", "https://feed.test/rss", "feed.test")
    registry = harness.registry(feed)
    monkeypatch.setattr(operations, "feed_registry", lambda: registry)
    good_url = "https://feed.test/stire-prima"
    dead_url = "https://feed.test/stire-cadere"
    good_title = "Banca națională a anunțat o nouă politică monetară pentru trimestru"
    good_page = harness.article(
        good_title,
        (
            "Decizia survine după mai multe luni de dezbateri publice intense.",
            "Economiștii așteaptă stabilizarea ratei inflației în cursul anului.",
        ),
    )
    observed_at = NOW
    harness.seed(
        fake_http,
        registry,
        feed,
        (
            (good_url, good_title, observed_at, "Descriere pe scurt."),
            (dead_url, "Cădere", observed_at, ""),
        ),
        observed_at,
    )
    fake_http.serve(good_url, good_page)
    fake_http.fail(dead_url, requests.ConnectionError("peer hung"))
    event_ids = tuple(_event_id_for_url(postgres_catalog, value) for value in (good_url, dead_url))

    result = operations.materialize_articles(
        news_day, event_ids, "git:test", run_id="run-1", retry_number=0
    )

    assert result.acquired_event_ids == (event_ids[0],)
    assert [failure.event_id for failure in result.failures] == [event_ids[1]]
    assert result.failures[0].kind == ArticleFailureKind.INFRASTRUCTURE
    assert (
        postgres_catalog.execute(
            "SELECT count(*) AS count FROM news_article_failure_attempts "
            "WHERE failure_kind = 'infrastructure'"
        ).fetchone()["count"]
        == 1
    )
    version_rows = postgres_catalog.execute(
        "SELECT version.canonical_url, version.page_capture_version_id "
        "FROM news_article_versions version "
        "JOIN artifacts artifact ON artifact.id = version.article_artifact_id "
        "WHERE artifact.kind = 'news_article'"
    ).fetchall()
    assert [row["canonical_url"] for row in version_rows] == [good_url]
    page_r2_key = postgres_catalog.execute(
        "SELECT r2_key FROM artifact_files WHERE artifact_version_id = %s",
        (version_rows[0]["page_capture_version_id"],),
    ).fetchone()["r2_key"]
    assert fake_r2.objects[page_r2_key] == good_page


def test_article_materializer_dedupes_alias_sharing_events(
    harness,
    postgres_catalog,
    fake_r2,
    fake_http,
    monkeypatch,
    news_day,
) -> None:
    alpha = harness.feed("feed-alpha", "outlet-alpha", "https://alpha.test/rss", "shared.test")
    beta = harness.feed("feed-beta", "outlet-beta", "https://beta.test/rss", "shared.test")
    registry = harness.registry(alpha, beta)
    monkeypatch.setattr(operations, "feed_registry", lambda: registry)
    alpha_url = "https://shared.test/story?utm_source=alpha"
    beta_url = "https://shared.test/story?utm_source=beta"
    title = "Conducerea companiei naționale de transport a prezentat planul anual"
    page = harness.article(
        title,
        (
            "Planul prevede investiții semnificative în infrastructura regională.",
            "Reprezentanții sindicatelor au reacționat cu rezervă față de promisiuni.",
        ),
    )
    observed_at = NOW
    alpha_capture = harness.seed(
        fake_http,
        registry,
        alpha,
        ((alpha_url, title, observed_at, "Descriere alpha."),),
        observed_at,
    )
    beta_capture = harness.seed(
        fake_http,
        registry,
        beta,
        ((beta_url, title, observed_at, "Descriere beta."),),
        observed_at,
    )
    fake_http.serve(alpha_url, page)
    fake_http.serve(beta_url, page)
    alpha_event = harness.event_ids(alpha_capture, registry)[0]
    beta_event = harness.event_ids(beta_capture, registry)[0]

    result = operations.materialize_articles(
        news_day, (alpha_event, beta_event), "git:test", run_id="run-1", retry_number=0
    )

    assert result.complete
    assert len(result.acquired_event_ids) == 1
    assert set(result.acquired_event_ids) | set(result.skipped_event_ids) == {
        alpha_event,
        beta_event,
    }
    assert result.failures == ()
    assert (
        postgres_catalog.execute("SELECT count(*) AS count FROM news_article_versions").fetchone()[
            "count"
        ]
        == 1
    )
    assert (
        postgres_catalog.execute(
            "SELECT count(*) AS count FROM news_article_aliases WHERE alias_key = %s",
            ("url:https://shared.test/story",),
        ).fetchone()["count"]
        == 1
    )


def test_article_materializer_isolates_a_publish_conflict_across_parallel_items(
    harness,
    postgres_catalog,
    fake_r2,
    fake_http,
    monkeypatch,
    news_day,
) -> None:
    feed = harness.feed("testfeed", "testoutlet", "https://feed.test/rss", "feed.test")
    registry = harness.registry(feed)
    monkeypatch.setattr(operations, "feed_registry", lambda: registry)
    clean_url = "https://feed.test/stire-curata"
    conflict_url = "https://feed.test/stire-conflict"
    clean_title = "Ministerul finanțelor a publicat raportul anual despre deficit"
    page = harness.article(
        clean_title,
        (
            "Raportul detaliază evoluția încasărilor bugetare din ultimele luni.",
            "Oppoziția ceră dezbateri parlamentare pe baza concluziilor raportului.",
        ),
    )
    observed_at = NOW
    harness.seed(
        fake_http,
        registry,
        feed,
        (
            (clean_url, clean_title, observed_at, "Descriere."),
            (
                conflict_url,
                "Conflicting headline that is long enough for extraction",
                observed_at,
                "Descriere conflict.",
            ),
        ),
        observed_at,
    )
    fake_http.serve(clean_url, page)
    fake_http.serve(
        conflict_url,
        harness.article(
            "Conflicting headline that is long enough for extraction",
            ("Un paragraf destul de lung pentru a trece de pragul de extragere minimă.",),
        ),
    )
    conflicting_artifact = f"news:article:{article_id(feed.outlet_id, normalize_article_url(conflict_url, feed.article_hosts))}"
    postgres_catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
        "visibility, current_version_id, created_at) "
        "VALUES (%s, 'news_article', 'Conflicting', 'derived', 'current', 'private', NULL, %s)",
        (conflicting_artifact, observed_at.isoformat()),
    )
    event_ids = tuple(
        _event_id_for_url(postgres_catalog, value) for value in (clean_url, conflict_url)
    )

    result = operations.materialize_articles(
        news_day, event_ids, "git:test", run_id="run-1", retry_number=0
    )

    assert result.acquired_event_ids == (event_ids[0],)
    assert [failure.event_id for failure in result.failures] == [event_ids[1]]
    assert result.failures[0].kind == ArticleFailureKind.DETERMINISTIC
    assert "artifacts identity conflict" in result.failures[0].message
    assert (
        postgres_catalog.execute(
            "SELECT count(*) AS count FROM news_article_failure_attempts WHERE event_id = %s",
            (event_ids[1],),
        ).fetchone()["count"]
        == 1
    )
    version_rows = postgres_catalog.execute(
        "SELECT version.canonical_url, version.page_capture_version_id "
        "FROM news_article_versions version "
        "JOIN artifacts artifact ON artifact.id = version.article_artifact_id "
        "WHERE artifact.kind = 'news_article'"
    ).fetchall()
    assert [row["canonical_url"] for row in version_rows] == [clean_url]
    page_r2_key = postgres_catalog.execute(
        "SELECT r2_key FROM artifact_files WHERE artifact_version_id = %s",
        (version_rows[0]["page_capture_version_id"],),
    ).fetchone()["r2_key"]
    assert fake_r2.objects[page_r2_key] == page
    assert (
        postgres_catalog.execute(
            "SELECT count(*) AS count FROM runs WHERE operation_key = 'news.normalize_article'"
        ).fetchone()["count"]
        == 1
    )


def test_article_materializer_rejects_more_than_ten_event_ids() -> None:
    with pytest.raises(ValueError, match="at most 10"):
        operations.materialize_articles(
            DAY,
            tuple(f"{index:064x}" for index in range(11)),
            "git:test",
            run_id="run-1",
            retry_number=0,
        )


def _seed_analysis_article(
    catalog: PostgresCatalog, fake_r2, news_day: date, article: ExtractedArticle
) -> ArtifactReference:
    content = article.model_dump_json().encode()
    r2_key = f"news/articles/{article.article_id}/{sha256(content)}.json"
    snapshot = artifact_file(
        artifact_id="news:feed:testfeed",
        artifact_kind="news_feed",
        title="RSS feed: testfeed",
        content=b"<rss/>",
        r2_key=f"news/feeds/testfeed/{sha256(b'<rss/>')}.xml",
        media_type="application/xml",
    )
    file = artifact_file(
        artifact_id=f"news:article:{article.article_id}",
        artifact_kind="news_article",
        title=article.title,
        content=content,
        r2_key=r2_key,
        media_type="application/json",
    )
    publish_immutable_r2_objects(((snapshot.r2_key, snapshot.content), (r2_key, content)))
    timestamp = NOW.isoformat()
    statements = [
        *artifact_statements(snapshot, timestamp, produced_by_run_id=None),
        *artifact_statements(file, timestamp, produced_by_run_id=None),
        advance_artifact_current_version_statement(snapshot.artifact_id, snapshot.version_id),
        advance_artifact_current_version_statement(file.artifact_id, file.version_id),
        (
            "INSERT INTO news_article_versions "
            "(artifact_version_id, article_artifact_id, outlet_id, canonical_url, "
            "published_at, source_updated_at, bucharest_day, material_digest, "
            "extraction_digest, feed_snapshot_version_id, page_capture_version_id, captured_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                file.version_id,
                file.artifact_id,
                article.outlet_id,
                str(article.canonical_url),
                article.published_at.isoformat(),
                article.source_updated_at.isoformat() if article.source_updated_at else None,
                article.bucharest_day.isoformat(),
                article.material_digest,
                article.extraction_digest,
                snapshot.version_id,
                None,
                timestamp,
            ],
        ),
    ]
    catalog.batch(statements)
    return ArtifactReference(
        artifact_id=file.artifact_id,
        version_id=file.version_id,
        content_digest=file.content_digest,
        r2_key=r2_key,
    )


def test_relevance_materializer_uses_production_v3(
    postgres_catalog,
    fake_r2,
    monkeypatch,
    news_day,
) -> None:
    title = "Analiza politicilor publice pentru infrastructura din România"
    body = (
        "Guvernul a prezentat un plan de investiții care schimbă prioritățile naționale "
        "și reașază bugetul pentru următorii ani."
    )
    article = ExtractedArticle(
        article_id="a" * 64,
        outlet_id="testoutlet",
        canonical_url=HttpUrl("https://feed.test/stire-analiza"),
        title=title,
        body=body,
        author=None,
        published_at=NOW,
        source_updated_at=None,
        bucharest_day=news_day,
        material_digest="1" * 64,
        extraction_digest="2" * 64,
    )
    reference = _seed_analysis_article(postgres_catalog, fake_r2, news_day, article)
    analyzed_modes: list[ExecutionMode] = []

    def analyze(
        value: ArticleAnalysisInput, *, mode: ExecutionMode, **_kwargs: object
    ) -> RelevanceV3Output:
        analyzed_modes.append(mode)
        request_id = relevance_v3_request_id(value.reference, mode=mode)
        context = ContextGateResult(
            decision=ContextDecision(
                subject_role="incidental",
                news_cycle="current_cycle",
                romanian_consequence="absent",
                certainty="clear",
                evidence_quote=value.article.title,
                reason_ro="Articolul nu descrie o consecință românească directă.",
            ),
            provider=GateCall(
                request_id=request_id,
                call=ModelCall(
                    response_id="resp-test",
                    model="test-model",
                    input_tokens=64,
                    output_tokens=32,
                    latency_ms=10,
                ),
                cost_usd=0.0,
                response_count=1,
                traces=(),
                accounting_complete=True,
            ),
        )
        payload = json.dumps(
            {
                "request_id": request_id,
                "mode": mode,
                "context_provider_responses": [
                    {"id": "resp-test", "usage": {"prompt_tokens": 64, "completion_tokens": 32}}
                ],
            },
            sort_keys=True,
        ).encode()
        return RelevanceV3Output(
            request_id=request_id,
            policy=RELEVANCE_V3_POLICY,
            mode=mode,
            execution_ref=None,
            article=value.reference,
            context=context,
            impact=None,
            accepted=False,
            content=payload,
        )

    monkeypatch.setattr(operations, "analyze_relevance_v3", analyze)
    monkeypatch.setattr(operations, "flush_langfuse_traces", lambda: None)

    references = operations.materialize_relevance(news_day, "git:test")

    assert analyzed_modes == ["production_early_exit"]
    assert [value.artifact_id for value in references.values] == [
        f"news:relevance:{production_relevance_v3_request_id(reference)}"
    ]
    relevance_version = references.values[0].version_id
    assert (
        postgres_catalog.execute(
            "SELECT accepted FROM news_relevance_versions WHERE artifact_version_id = %s",
            (relevance_version,),
        ).fetchone()["accepted"]
        == 0
    )
    assert (
        hashlib.sha256(fake_r2.objects[references.values[0].r2_key]).hexdigest()
        == references.values[0].content_digest
    )
    assert (
        postgres_catalog.execute(
            "SELECT count(*) AS count FROM runs WHERE operation_key = 'news.relevance.v3' "
            "AND status = 'completed'"
        ).fetchone()["count"]
        == 1
    )

    repeat = operations.materialize_relevance(news_day, "git:test")

    assert repeat == references
    assert analyzed_modes == ["production_early_exit"]
    assert (
        postgres_catalog.execute(
            "SELECT count(*) AS count FROM runs WHERE operation_key = 'news.relevance.v3'"
        ).fetchone()["count"]
        == 1
    )


@pytest.mark.parametrize(
    ("run", "patches", "message"),
    [
        (
            lambda: operations.materialize_relevance(DAY, "git:test"),
            (
                ("read_pending_relevance_references", lambda **_kwargs: (object(),)),
                ("load_article_analysis_input", lambda _reference: object()),
                (
                    "analyze_relevance_v3",
                    lambda _value, **_kwargs: _raise_model_failure(),
                ),
            ),
            "model failed",
        ),
        (
            lambda: operations.materialize_embeddings(DAY, "git:test"),
            (
                ("read_pending_embedding_references", lambda **_kwargs: (object(),)),
                ("load_embedding_input", lambda _reference: object()),
                ("embed_article", lambda _value: _raise_model_failure()),
            ),
            "model failed",
        ),
        (
            lambda: operations.materialize_group_summaries(DAY, "git:test"),
            (
                (
                    "read_pending_group_analysis_references",
                    lambda _days: (
                        SimpleNamespace(
                            summary_needed=True,
                            sentiment_needed=False,
                        ),
                    ),
                ),
                ("load_group_analysis_input", lambda _pending: object()),
                ("summarize_group", lambda _value: _raise_model_failure()),
            ),
            "summary: model failed",
        ),
        (
            lambda: operations.materialize_group_sentiment(DAY, "git:test"),
            (
                (
                    "read_pending_group_analysis_references",
                    lambda _days: (
                        SimpleNamespace(
                            summary_needed=False,
                            sentiment_needed=True,
                        ),
                    ),
                ),
                ("load_group_analysis_input", lambda _pending: object()),
                ("score_group_sentiment", lambda _value: _raise_model_failure()),
            ),
            "sentiment: model failed",
        ),
    ],
)
def test_model_materializers_flush_traces_without_hiding_model_failure(
    monkeypatch,
    run,
    patches,
    message,
) -> None:
    events = []
    for name, value in patches:
        monkeypatch.setattr(operations, name, value)
    monkeypatch.setattr(
        operations,
        "flush_langfuse_traces",
        lambda: events.append("flush"),
        raising=False,
    )

    with pytest.raises(RuntimeError, match=message):
        run()

    assert events == ["flush"]


@pytest.mark.parametrize(
    ("materialize", "pending", "read_references"),
    [
        (
            operations.materialize_group_summaries,
            SimpleNamespace(summary_needed=False, sentiment_needed=True),
            "read_daily_group_summary_references",
        ),
        (
            operations.materialize_group_sentiment,
            SimpleNamespace(summary_needed=True, sentiment_needed=False),
            "read_daily_group_sentiment_references",
        ),
    ],
)
def test_group_materializers_skip_completed_branches_without_loading_articles(
    monkeypatch,
    materialize,
    pending,
    read_references,
) -> None:
    expected = object()
    monkeypatch.setattr(
        operations,
        "read_pending_group_analysis_references",
        lambda _days: (pending,),
    )
    monkeypatch.setattr(
        operations,
        "load_group_analysis_input",
        lambda _pending: pytest.fail("completed branch loaded article content"),
    )
    monkeypatch.setattr(operations, read_references, lambda _day: expected)
    monkeypatch.setattr(operations, "flush_langfuse_traces", lambda: None)

    assert materialize(DAY, "git:test") is expected


@pytest.mark.parametrize(
    ("materialize", "pending"),
    [
        (
            operations.materialize_group_summaries,
            SimpleNamespace(summary_needed=True, sentiment_needed=False),
        ),
        (
            operations.materialize_group_sentiment,
            SimpleNamespace(summary_needed=False, sentiment_needed=True),
        ),
    ],
)
def test_group_materializers_preserve_article_loading_failures(
    monkeypatch,
    materialize,
    pending,
) -> None:
    events = []
    monkeypatch.setattr(
        operations,
        "read_pending_group_analysis_references",
        lambda _days: (pending,),
    )
    monkeypatch.setattr(
        operations,
        "load_group_analysis_input",
        lambda _pending: (_ for _ in ()).throw(ValueError("article load failed")),
    )
    monkeypatch.setattr(operations, "flush_langfuse_traces", lambda: events.append("flush"))

    with pytest.raises(ValueError, match="article load failed"):
        materialize(DAY, "git:test")

    assert events == ["flush"]


def _raise_model_failure() -> None:
    raise RuntimeError("model failed")


def test_subject_assessment_materializer_reuses_completed_run_before_inference(
    monkeypatch,
) -> None:
    value = object()
    reference = ArtifactReference(
        artifact_id="news:subject-assessments:test",
        version_id="a" * 64,
        content_digest="d" * 64,
        r2_key="news/subject-assessments/test.json",
    )
    monkeypatch.setattr(
        operations,
        "read_daily_subject_assessment_input",
        lambda day: value,
    )
    monkeypatch.setattr(
        operations,
        "subject_assessment_request_id",
        lambda item: "b" * 64,
    )
    monkeypatch.setattr(
        operations,
        "subject_assessment_run_id",
        lambda request_id, implementation_ref: "c" * 64,
    )
    monkeypatch.setattr(
        operations,
        "read_completed_subject_assessment",
        lambda run_id: reference,
    )
    monkeypatch.setattr(
        operations,
        "construct_daily_subject_assessments",
        lambda _value: (_ for _ in ()).throw(AssertionError("unexpected inference")),
    )

    result = operations.materialize_subject_assessments(DAY, "git:test")

    assert result.values == (reference,)


def test_morning_report_check_requires_yesterdays_catalog_edition_before_pinging(
    postgres_catalog: PostgresCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from romanian_news.worker import morning_report

    monkeypatch.setattr(
        morning_report,
        "_now",
        lambda: datetime(2026, 9, 14, 9, 35, tzinfo=ZoneInfo("Europe/Bucharest")),
    )
    monkeypatch.setattr(
        morning_report,
        "build_and_publish_current_daily_report",
        lambda day, ref: SimpleNamespace(
            head=SimpleNamespace(version_id="9" * 64, input_time=datetime(2026, 9, 14, 4, 2))
        ),
    )
    pings: list[str] = []
    monkeypatch.setattr(morning_report, "ping_heartbeat", lambda kind: pings.append(kind))

    with pytest.raises(ValueError, match="news:daily:2026-09-13"):
        morning_report.morning_report_check_op(dg.build_op_context())

    assert pings == []

    seed_daily_report(postgres_catalog, daily_report(date(2026, 9, 13)))

    result = morning_report.morning_report_check_op(dg.build_op_context())

    assert result == "2026-09-14"
    assert pings == ["morning_report"]


def _seed_one_feed_observation(harness, fake_http, observed_at: datetime) -> None:
    registry = feed_registry()
    harness.seed(fake_http, registry, registry.feeds[0], (), observed_at)


def test_article_controller_closes_a_past_day_without_full_feed_coverage(
    harness,
    postgres_catalog,
    fake_r2,
    fake_http,
    monkeypatch,
) -> None:
    _seed_one_feed_observation(harness, fake_http, datetime(2026, 9, 22, 10, tzinfo=UTC))
    now = datetime(2026, 9, 24, 10, tzinfo=UTC)
    monkeypatch.setattr(definitions, "_controller_time", lambda: now)

    with dg.instance_for_test() as instance:
        context = dg.build_sensor_context(
            instance=instance,
            repository_def=definitions.defs.get_repository_def(),
        )
        evaluation = definitions.article_batch_controller.evaluate_tick(context)

    assert evaluation.run_requests
    request = evaluation.run_requests[0]
    assert request.partition_key == "2026-09-22"
    assert json.loads(request.tags["news/article_event_ids"]) == []


def test_article_controller_waits_for_full_coverage_on_the_current_day(
    harness,
    postgres_catalog,
    fake_r2,
    fake_http,
    monkeypatch,
) -> None:
    _seed_one_feed_observation(harness, fake_http, datetime(2026, 9, 24, 10, tzinfo=UTC))
    now = datetime(2026, 9, 24, 15, tzinfo=UTC)
    monkeypatch.setattr(definitions, "_controller_time", lambda: now)

    with dg.instance_for_test() as instance:
        context = dg.build_sensor_context(
            instance=instance,
            repository_def=definitions.defs.get_repository_def(),
        )
        evaluation = definitions.article_batch_controller.evaluate_tick(context)

    assert not evaluation.run_requests
    assert evaluation.skip_message == "No article work is ready."


def test_print_catalog_status_reports_each_day_from_the_real_catalog(
    capsys,
    harness,
    postgres_catalog,
    fake_r2,
    fake_http,
) -> None:
    expected = len(feed_registry().feeds)
    _seed_one_feed_observation(harness, fake_http, datetime(2026, 9, 22, 10, tzinfo=UTC))
    seed_daily_report(postgres_catalog, daily_report(date(2026, 9, 22)))

    print_catalog_status(days=(date(2026, 9, 21), date(2026, 9, 22)))

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"expected_feeds={expected}"
    assert (
        lines[1] == "2026-09-21 feeds=0/" + str(expected) + " freshness=inputs_not_ready report=-"
    )
    assert lines[2] == (
        "2026-09-22 feeds=1/" + str(expected) + " freshness=inputs_not_ready report=" + "a" * 12
    )
    assert REPORT_VERSION.startswith("a")


def test_print_catalog_status_reports_every_day_when_one_freshness_read_fails(
    capsys,
    harness,
    postgres_catalog,
    fake_r2,
    fake_http,
    monkeypatch,
) -> None:
    from romanian_news.worker import catalog_status

    expected = len(feed_registry().feeds)
    _seed_one_feed_observation(harness, fake_http, datetime(2026, 9, 22, 10, tzinfo=UTC))
    seed_daily_report(postgres_catalog, daily_report(date(2026, 9, 22)))
    real_freshness = catalog_status.read_daily_report_freshness

    def failing_freshness(day: date):
        if day == date(2026, 9, 21):
            raise RuntimeError("freshness read failed")
        return real_freshness(day)

    monkeypatch.setattr(catalog_status, "read_daily_report_freshness", failing_freshness)

    print_catalog_status(days=(date(2026, 9, 21), date(2026, 9, 22)))

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"expected_feeds={expected}"
    assert lines[1] == (
        "2026-09-21 feeds=0/" + str(expected) + " freshness=error:freshness read failed report=-"
    )
    assert lines[2] == (
        "2026-09-22 feeds=1/" + str(expected) + " freshness=inputs_not_ready report=" + "a" * 12
    )
