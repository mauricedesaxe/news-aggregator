import hashlib
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests
from pydantic import ValidationError

from romanian_news.analysis import jev_relevance
from romanian_news.analysis.binary_evaluation import (
    RELEVANCE_BINARY_QUESTION,
    BinaryAttemptEvidence,
    BinaryProbabilityObservation,
    BinaryQuestion,
    BinaryRequest,
    binary_decision,
    build_relevance_binary_request,
)
from romanian_news.analysis.jev_relevance import (
    JEV_EXECUTION_POLICY,
    JevExecutionPolicy,
    JevNoulResponse,
    evaluate_jev_relevance,
    jev_execution_policy_digest,
    jev_relevance_request_id,
)
from romanian_news.analysis.relevance import ArticleAnalysisInput
from romanian_news.analysis.relevance_v3 import relevance_v3_article_text
from romanian_news.tests.evaluation_factories import embedded_article

_ScriptedResponse = tuple[int, dict[str, str], bytes]


@contextmanager
def _jev_server(script: list[_ScriptedResponse]) -> Iterator[tuple[str, type]]:
    class Handler(BaseHTTPRequestHandler):
        seen: list[tuple[str, dict[str, str], dict[str, object]]] = []

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            type(self).seen.append((self.path, dict(self.headers), payload))
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
        yield f"http://127.0.0.1:{server.server_port}", Handler
    finally:
        server.shutdown()
        server.server_close()


def _local_policy(base_url: str) -> JevExecutionPolicy:
    return JEV_EXECUTION_POLICY.model_copy(
        update={"endpoint": f"{base_url}/v1/systemone", "timeout_seconds": 10.0}
    )


def test_jev_boundary_posts_exact_article_and_validates_noul(monkeypatch) -> None:
    article = embedded_article(1)
    body = json.dumps(
        {
            "model": "jev-1.13.0",
            "answers": {"relevant": {"type": "noul", "noul": 0.73}},
            "usage": {"input_tokens": 1_000_000, "output_tokens": 7},
        }
    ).encode()
    monkeypatch.setattr(jev_relevance, "TYPESAFE_API_KEY", "test-key")
    value = ArticleAnalysisInput(reference=article.article, article=article.value)
    binary_request = build_relevance_binary_request(value)

    with _jev_server([(200, {"x-typesafe-request-id": "provider-request-1"}, body)]) as (
        base_url,
        handler,
    ):
        result = evaluate_jev_relevance(
            binary_request,
            execution_ref="git:test:fresh-trial-1-of-3",
            policy=_local_policy(base_url),
        )

    (_path, headers, payload), = handler.seen
    assert headers["Authorization"] == "Bearer test-key"
    assert payload == {
        "state": binary_request.state,
        "model": "jev-1.13.0",
        "questions": {
            "relevant": {
                "type": "noul",
                "instructions": RELEVANCE_BINARY_QUESTION.instructions,
                "criteria": {
                    "true": RELEVANCE_BINARY_QUESTION.true_criteria,
                    "false": RELEVANCE_BINARY_QUESTION.false_criteria,
                },
            }
        },
    }
    assert result.predicted_accepted is True
    assert result.probability == Decimal("0.73")
    assert result.provider_request_id == "provider-request-1"
    assert result.estimated_cost_usd == Decimal("0.042")


def test_jev_boundary_matches_the_v3_article_body_limit(monkeypatch) -> None:
    article = embedded_article(1)
    article = article.model_copy(
        update={"value": article.value.model_copy(update={"body": "a" * 24_001})}
    )
    body = json.dumps(
        {
            "model": "jev-1.13.0",
            "answers": {"relevant": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).encode()
    monkeypatch.setattr(jev_relevance, "TYPESAFE_API_KEY", "test-key")

    with _jev_server([(200, {}, body)]) as (base_url, handler):
        result = evaluate_jev_relevance(
            ArticleAnalysisInput(reference=article.article, article=article.value),
            execution_ref="trial-1",
            policy=_local_policy(base_url),
        )

    (_path, _headers, payload), = handler.seen
    assert payload["state"].endswith("a" * 24_000)
    assert result.predicted_accepted is True


def test_binary_request_captures_the_exact_v3_article_state() -> None:
    article = embedded_article(1)
    value = ArticleAnalysisInput(reference=article.article, article=article.value)

    request = build_relevance_binary_request(value)
    expected = relevance_v3_article_text(value, 24_000).encode()

    assert request.state.encode() == expected
    assert request.state_digest == hashlib.sha256(expected).hexdigest()

    with pytest.raises(ValueError, match="state digest"):
        _ = BinaryRequest(
            question=RELEVANCE_BINARY_QUESTION,
            state=request.state + "changed",
            state_digest=request.state_digest,
        )


def test_binary_decision_accepts_the_exact_threshold() -> None:
    assert binary_decision(Decimal("0.5"), RELEVANCE_BINARY_QUESTION.threshold) is True
    assert binary_decision(Decimal("0.499999"), RELEVANCE_BINARY_QUESTION.threshold) is False


@pytest.mark.parametrize("probability", [Decimal("-0.01"), Decimal("1.01")])
def test_binary_decision_rejects_probability_outside_unit_interval(
    probability: Decimal,
) -> None:
    with pytest.raises(ValueError, match="probability must be between 0 and 1"):
        binary_decision(probability, RELEVANCE_BINARY_QUESTION.threshold)


def test_jev_boundary_retries_transient_failures(monkeypatch) -> None:
    article = embedded_article(1)
    accepted_body = json.dumps(
        {
            "model": "jev-1.13.0",
            "answers": {"relevant": {"type": "noul", "noul": 0.7}},
            "usage": {"input_tokens": 10, "output_tokens": 1},
        }
    ).encode()
    delays: list[float] = []
    attempts: list[BinaryAttemptEvidence] = []
    monkeypatch.setattr(jev_relevance, "TYPESAFE_API_KEY", "test-key")

    with _jev_server(
        [
            (529, {"retry-after-ms": "250"}, b""),
            (200, {"x-typesafe-request-id": "provider-request-2"}, accepted_body),
        ]
    ) as (base_url, handler):
        result = evaluate_jev_relevance(
            ArticleAnalysisInput(reference=article.article, article=article.value),
            execution_ref="trial-1",
            policy=_local_policy(base_url),
            sleep=delays.append,
            clock=iter((1.0, 1.1, 1.35, 1.75)).__next__,
            on_attempt=attempts.append,
        )

    assert len(handler.seen) == 2
    assert delays == [0.25]
    assert tuple(attempt.status for attempt in attempts) == ("retryable_error", "completed")
    assert tuple(attempt.attempt_number for attempt in attempts) == (1, 2)
    assert attempts[0].http_status == 529
    assert attempts[0].error is not None
    assert attempts[0].error.retryable is True
    assert attempts[1].provider_request_id == "provider-request-2"
    assert attempts[1].probability == Decimal("0.7")
    assert result.attempts == tuple(attempts)
    assert result.latency_ms == 500


def test_jev_boundary_reports_the_final_failed_http_attempt(monkeypatch) -> None:
    attempts: list[BinaryAttemptEvidence] = []
    monkeypatch.setattr(jev_relevance, "TYPESAFE_API_KEY", "test-key")

    with _jev_server(
        [(400, {"x-typesafe-request-id": "provider-request-failed"}, b"")]
    ) as (base_url, _handler):
        with pytest.raises(RuntimeError, match="Jev relevance request failed"):
            evaluate_jev_relevance(
                build_relevance_binary_request(
                    ArticleAnalysisInput(
                        reference=embedded_article(1).article,
                        article=embedded_article(1).value,
                    )
                ),
                execution_ref="trial-1",
                policy=_local_policy(base_url),
                clock=iter((2.0, 2.2)).__next__,
                on_attempt=attempts.append,
            )

    assert len(attempts) == 1
    assert attempts[0].status == "terminal_error"
    assert attempts[0].provider_request_id == "provider-request-failed"
    assert attempts[0].http_status == 400
    assert attempts[0].latency_ms == 200


@pytest.mark.parametrize(
    "payload",
    (
        {
            "model": "jev-1.13.0",
            "answers": {"relevant": {"type": "noul", "noul": 1.01}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        {
            "model": "jev-latest",
            "answers": {"relevant": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    ),
)
def test_jev_noul_response_rejects_invalid_wire_shapes(payload) -> None:
    with pytest.raises(ValidationError):
        JevNoulResponse.model_validate(payload, strict=True)


def test_binary_question_and_jev_request_identities_are_stable() -> None:
    article = embedded_article(1)
    request = build_relevance_binary_request(
        ArticleAnalysisInput(reference=article.article, article=article.value)
    )

    same_question = BinaryQuestion.model_validate(
        RELEVANCE_BINARY_QUESTION.model_dump(), strict=True
    )
    changed_question = RELEVANCE_BINARY_QUESTION.model_copy(update={"threshold": Decimal("0.6")})
    changed_execution = JEV_EXECUTION_POLICY.model_copy(
        update={"endpoint": "https://example.test/jev"}
    )

    assert (
        RELEVANCE_BINARY_QUESTION.semantic_digest
        == "5d77b7025c568fec62748abacd89920ea6ddfb5e5d9a4aa272d2e4baa9d2407a"
    )
    assert same_question.semantic_digest == RELEVANCE_BINARY_QUESTION.semantic_digest
    assert changed_question.semantic_digest != RELEVANCE_BINARY_QUESTION.semantic_digest
    assert RELEVANCE_BINARY_QUESTION.semantic_digest == request.question.semantic_digest
    assert jev_execution_policy_digest(changed_execution) != jev_execution_policy_digest()
    first = jev_relevance_request_id(request, "trial-1")
    assert first == jev_relevance_request_id(request, "trial-1")
    assert first != jev_relevance_request_id(request, "trial-2")
