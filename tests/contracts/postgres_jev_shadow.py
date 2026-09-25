from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from psycopg.errors import IntegrityConstraintViolation

from romanian_news import jev_relevance_shadow as shadow
from romanian_news import storage
from romanian_news.analysis import jev_relevance
from romanian_news.analysis.jev_relevance import (
    JEV_EXECUTION_POLICY,
    evaluate_jev_relevance,
)
from romanian_news.analysis.relevance import ArticleAnalysisReference
from romanian_news.analysis.relevance_v3 import production_relevance_v3_request_id
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.analysis import publish_relevance_outputs
from romanian_news.catalog.artifacts import artifact_file, artifact_statements
from romanian_news.catalog.jev_shadow import (
    JevShadowClaim,
    JevShadowReceipt,
    claim_jev_shadow,
    read_jev_shadow_fallback_reason,
    record_jev_shadow_receipt,
)
from romanian_news.catalog_transport import ResearchCatalogError, catalog_integrity_identity
from romanian_news.daily import read_daily_article_references, read_daily_relevance_references
from romanian_news.identity import sha256
from romanian_news.tests.evaluation_factories import embedded_article
from romanian_news.tests.test_jev_relevance_shadow import incumbent_relevance_output
from tests.postgres_catalog import PostgresCatalog, postgres_catalog_fixture
from tests.worker.conftest import FakeR2Client

postgres_catalog = postgres_catalog_fixture("jev_shadow")
NOW = datetime(2026, 9, 25, tzinfo=UTC)
ARTICLE_VERSION = "1" * 64
INCUMBENT_VERSION = "2" * 64
SHADOW_DAY = embedded_article(1).value.bucharest_day
NOUL_ACCEPTED_BODY = json.dumps(
    {
        "model": "jev-1.13.0",
        "answers": {"relevant": {"type": "noul", "noul": 0.7}},
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }
).encode()


@pytest.fixture
def fake_r2(monkeypatch: pytest.MonkeyPatch) -> FakeR2Client:
    client = FakeR2Client()
    monkeypatch.setattr(storage, "_r2_client", lambda: client)
    return client


def _seed_versions(catalog) -> None:
    for artifact_id, kind, version_id in (
        ("article", "news_article", ARTICLE_VERSION),
        ("incumbent", "relevance_result", INCUMBENT_VERSION),
    ):
        catalog.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
            "visibility, created_at) VALUES (%s, %s, 'Test', 'source', 'current', 'private', %s)",
            (artifact_id, kind, NOW),
        )
        catalog.execute(
            "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
            "created_at) VALUES (%s, %s, 1, %s, %s)",
            (version_id, artifact_id, "a" * 64, NOW),
        )


def _claim() -> JevShadowClaim:
    return JevShadowClaim(
        shadow_id="b" * 64,
        article_version_id=ARTICLE_VERSION,
        incumbent_version_id=INCUMBENT_VERSION,
        article_content_digest="a" * 64,
        incumbent_content_digest="a" * 64,
        state_digest="c" * 64,
        question_digest="d" * 64,
        execution_policy_digest="e" * 64,
        incumbent_request_id="f" * 64,
        incumbent_policy_digest="0" * 64,
        incumbent_accepted=True,
        state_length=70_000,
        execution_ref="production-shadow-v1",
        claimed_at=NOW,
    )


def test_concurrent_claims_have_one_owner_and_receipt_is_immutable(postgres_catalog) -> None:
    _seed_versions(postgres_catalog)
    claim = _claim()
    with ThreadPoolExecutor(max_workers=2) as executor:
        owners = tuple(executor.map(claim_jev_shadow, (claim, claim)))
    assert sorted(owners) == [False, True]

    receipt = JevShadowReceipt(
        shadow_id=claim.shadow_id,
        fallback_reason="over_guard",
        jev_request_id=None,
        provider_request_id=None,
        model=None,
        probability=None,
        jev_accepted=None,
        attempts=(),
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=Decimal(0),
        latency_ms=0,
        accounting_complete=True,
        completed_at=NOW,
    )
    record_jev_shadow_receipt(receipt)
    assert read_jev_shadow_fallback_reason(claim.shadow_id) == "over_guard"
    with pytest.raises(ResearchCatalogError) as error:
        record_jev_shadow_receipt(receipt)
    assert (
        catalog_integrity_identity(error.value)
        == "UniqueViolation:23505:news_jev_shadow_receipts_pkey"
    )
    with pytest.raises(IntegrityConstraintViolation):
        postgres_catalog.execute(
            "UPDATE news_jev_shadow_receipts SET fallback_reason = 'provider_failed' "
            "WHERE shadow_id = %s",
            (claim.shadow_id,),
        )
    with pytest.raises(IntegrityConstraintViolation):
        postgres_catalog.execute(
            "DELETE FROM news_jev_shadow_claims WHERE shadow_id = %s", (claim.shadow_id,)
        )


_ScriptedResponse = tuple[int, dict[str, str], bytes]


@contextmanager
def _jev_server(script: list[_ScriptedResponse]) -> Iterator[tuple[str, list[dict[str, object]]]]:
    class Handler(BaseHTTPRequestHandler):
        seen: list[dict[str, object]] = []

        def do_POST(self) -> None:
            type(self).seen.append({"path": self.path})
            status, headers, body = script.pop(0)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", Handler.seen
    finally:
        server.shutdown()
        server.server_close()


def _tick() -> Iterator[float]:
    value = 1.0
    while True:
        yield value
        value += 0.25


def _local_jev_evaluate(base_url: str):
    policy = JEV_EXECUTION_POLICY.model_copy(
        update={"endpoint": f"{base_url}/v1/systemone", "timeout_seconds": 10.0}
    )
    ticks = _tick()

    def evaluate(request, *, execution_ref: str, on_attempt):
        return evaluate_jev_relevance(
            request,
            execution_ref=execution_ref,
            policy=policy,
            sleep=lambda _delay: None,
            clock=ticks.__next__,
            on_attempt=on_attempt,
        )

    return evaluate


def _enable_shadow(monkeypatch: pytest.MonkeyPatch, evaluate) -> None:
    monkeypatch.setattr(shadow, "NEWS_JEV_RELEVANCE_SHADOW_ENABLED", True)
    monkeypatch.setattr(shadow, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(jev_relevance, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(shadow, "evaluate_jev_relevance", evaluate)


def _refuse_provider(*_args: object) -> None:
    pytest.fail("the provider must not be called for this rerun")


def _seed_shadow_day(
    catalog: PostgresCatalog, fake_r2: FakeR2Client
) -> tuple[ArtifactReference, ArtifactReference]:
    article = embedded_article(1).value
    content = article.model_dump_json().encode()
    feed_snapshot = artifact_file(
        artifact_id="news:feed:snapshot:shadow-test",
        artifact_kind="news_feed",
        title="Feed snapshot",
        content=b"feed-snapshot",
        r2_key=f"news/feeds/{sha256(b'feed-snapshot')}.json",
        media_type="application/json",
    )
    article_file = artifact_file(
        artifact_id=f"news:article:{article.article_id}",
        artifact_kind="news_article",
        title=article.title,
        content=content,
        r2_key=f"news/articles/{sha256(content)}.json",
        media_type="application/json",
    )
    timestamp = NOW.isoformat()
    catalog.batch(artifact_statements(feed_snapshot, timestamp, produced_by_run_id=None))
    catalog.batch(artifact_statements(article_file, timestamp, produced_by_run_id=None))
    catalog.execute(
        "UPDATE artifacts SET current_version_id = %s WHERE id = %s",
        (feed_snapshot.version_id, feed_snapshot.artifact_id),
    )
    catalog.execute(
        "UPDATE artifacts SET current_version_id = %s WHERE id = %s",
        (article_file.version_id, article_file.artifact_id),
    )
    catalog.execute(
        "INSERT INTO news_article_versions (artifact_version_id, article_artifact_id, outlet_id, "
        "canonical_url, published_at, source_updated_at, bucharest_day, material_digest, "
        "extraction_digest, feed_snapshot_version_id, page_capture_version_id, captured_at) "
        "VALUES (%s, %s, %s, %s, %s, NULL, %s, %s, %s, %s, NULL, %s)",
        (
            article_file.version_id,
            article_file.artifact_id,
            article.outlet_id,
            str(article.canonical_url),
            article.published_at,
            article.bucharest_day,
            article.material_digest,
            article.extraction_digest,
            feed_snapshot.version_id,
            NOW,
        ),
    )
    fake_r2.put_object(Bucket="news-objects", Key=article_file.r2_key, Body=content, Metadata={})
    current_article = read_daily_article_references(SHADOW_DAY).values[0]
    request_id = production_relevance_v3_request_id(current_article)
    incumbent = incumbent_relevance_output(current_article, accepted=True, request_id=request_id)
    publish_relevance_outputs((incumbent,), "git:test")
    return current_article, read_daily_relevance_references(SHADOW_DAY).values[0]


def _production_claim(
    day: date, article: ArtifactReference, incumbent: ArtifactReference
) -> JevShadowClaim:
    analysis_input = shadow.load_article_analysis_input(
        ArticleAnalysisReference(reference=article, bucharest_day=day)
    )
    request = shadow.build_relevance_binary_request(analysis_input)
    incumbent_request_id = production_relevance_v3_request_id(article)
    accepted, policy_digest = shadow._incumbent_decision(
        incumbent, article.version_id, incumbent_request_id
    )
    return shadow._claim(
        article,
        incumbent,
        request,
        incumbent_request_id,
        policy_digest,
        accepted,
    )


def test_rerunning_a_paired_article_is_already_claimed_without_calling_the_provider(
    postgres_catalog, fake_r2: FakeR2Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_shadow_day(postgres_catalog, fake_r2)
    with _jev_server([(200, {"x-typesafe-request-id": "provider-1"}, NOUL_ACCEPTED_BODY)]) as (
        base_url,
        seen,
    ):
        _enable_shadow(monkeypatch, _local_jev_evaluate(base_url))
        first = shadow.materialize_jev_relevance_shadow(SHADOW_DAY)

    assert len(seen) == 1
    assert first.paired == 1
    receipt = postgres_catalog.execute(
        "SELECT fallback_reason FROM news_jev_shadow_receipts"
    ).fetchone()
    assert receipt is not None
    assert receipt["fallback_reason"] == "none"

    _enable_shadow(monkeypatch, _refuse_provider)
    second = shadow.materialize_jev_relevance_shadow(SHADOW_DAY)

    assert second.article_count == 1
    assert second.already_claimed == 1
    assert second.paired == second.failed == second.unresolved == 0


def test_rerunning_a_bare_claim_without_a_receipt_is_unresolved(
    postgres_catalog, fake_r2: FakeR2Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    article, incumbent = _seed_shadow_day(postgres_catalog, fake_r2)
    assert claim_jev_shadow(_production_claim(SHADOW_DAY, article, incumbent))

    _enable_shadow(monkeypatch, _refuse_provider)
    result = shadow.materialize_jev_relevance_shadow(SHADOW_DAY)

    assert result.unresolved == 1
    assert result.already_claimed == result.failed == 0
    assert (
        postgres_catalog.execute(
            "SELECT count(*) AS count FROM news_jev_shadow_receipts"
        ).fetchone()["count"]
        == 0
    )


def test_rerunning_a_claim_with_a_provider_failed_receipt_reports_failed(
    postgres_catalog, fake_r2: FakeR2Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    article, incumbent = _seed_shadow_day(postgres_catalog, fake_r2)
    claim = _production_claim(SHADOW_DAY, article, incumbent)
    assert claim_jev_shadow(claim)
    record_jev_shadow_receipt(
        JevShadowReceipt(
            shadow_id=claim.shadow_id,
            fallback_reason="provider_failed",
            jev_request_id=None,
            provider_request_id=None,
            model=None,
            probability=None,
            jev_accepted=None,
            attempts=(),
            input_tokens=None,
            output_tokens=None,
            estimated_cost_usd=None,
            latency_ms=0,
            accounting_complete=False,
            completed_at=NOW,
        )
    )

    _enable_shadow(monkeypatch, _refuse_provider)
    result = shadow.materialize_jev_relevance_shadow(SHADOW_DAY)

    assert result.failed == 1
    assert result.already_claimed == result.unresolved == 0


@pytest.mark.parametrize(
    ("status", "headers"),
    ((429, {"retry-after-ms": "250"}), (503, {})),
)
def test_provider_exhaustion_is_classified_as_provider_failed_not_provider_rejected(
    postgres_catalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    headers: dict[str, str],
) -> None:
    _seed_shadow_day(postgres_catalog, fake_r2)
    with _jev_server([(status, headers, b"")] * 4) as (base_url, seen):
        _enable_shadow(monkeypatch, _local_jev_evaluate(base_url))
        result = shadow.materialize_jev_relevance_shadow(SHADOW_DAY)

    assert len(seen) == 4
    assert result.failed == 1
    assert result.paired == result.already_claimed == result.unresolved == 0
    receipt = postgres_catalog.execute(
        "SELECT fallback_reason, attempts FROM news_jev_shadow_receipts"
    ).fetchone()
    assert receipt is not None
    assert receipt["fallback_reason"] == "provider_failed"
    attempts = receipt["attempts"]
    assert [attempt["http_status"] for attempt in attempts] == [status] * 4
    assert attempts[-1]["status"] == "terminal_error"
    assert attempts[-1]["error"]["retryable"] is True
