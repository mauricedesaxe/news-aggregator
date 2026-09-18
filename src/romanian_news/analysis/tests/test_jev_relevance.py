import json
from unittest.mock import Mock

import pytest
import requests
from pydantic import ValidationError

from romanian_news.analysis import jev_relevance
from romanian_news.analysis.jev_relevance import (
    JEV_RELEVANCE_POLICY,
    JevNoulResponse,
    evaluate_jev_relevance,
    jev_relevance_policy_digest,
    jev_relevance_request_id,
)
from romanian_news.analysis.relevance import ArticleAnalysisInput
from romanian_news.analysis.relevance_v3 import relevance_v3_article_text
from romanian_news.tests.evaluation_factories import embedded_article


def test_jev_boundary_posts_exact_article_and_validates_noul(monkeypatch) -> None:
    article = embedded_article(1)
    response = Mock()
    response.headers = {"x-typesafe-request-id": "provider-request-1"}
    response.content = json.dumps(
        {
            "model": "jev-1.13.0",
            "answers": {"relevant": {"type": "noul", "noul": 0.73}},
            "usage": {"input_tokens": 1_000_000, "output_tokens": 7},
        }
    ).encode()
    post = Mock(return_value=response)
    monkeypatch.setattr(jev_relevance, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(jev_relevance.requests, "post", post)
    value = ArticleAnalysisInput(reference=article.article, article=article.value)

    result = evaluate_jev_relevance(value, execution_ref="git:test:fresh-trial-1-of-3")

    response.raise_for_status.assert_called_once_with()
    request = post.call_args
    assert request.args == ("https://api.typesafe.ai/v1/systemone",)
    assert request.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert request.kwargs["json"]["model"] == "jev-1.13.0"
    assert request.kwargs["json"]["state"] == relevance_v3_article_text(value, 24_000)
    assert tuple(request.kwargs["json"]["questions"]) == ("relevant",)
    assert result.predicted_accepted is True
    assert result.probability == 0.73
    assert result.provider_request_id == "provider-request-1"
    assert result.estimated_cost_usd == pytest.approx(0.042)


def test_jev_boundary_matches_the_v3_article_body_limit(monkeypatch) -> None:
    article = embedded_article(1)
    article = article.model_copy(
        update={"value": article.value.model_copy(update={"body": "a" * 24_001})}
    )
    response = Mock()
    response.headers = {}
    response.content = json.dumps(
        {
            "model": "jev-1.13.0",
            "answers": {"relevant": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).encode()
    post = Mock(return_value=response)
    monkeypatch.setattr(jev_relevance, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(jev_relevance.requests, "post", post)

    _ = evaluate_jev_relevance(
        ArticleAnalysisInput(reference=article.article, article=article.value),
        execution_ref="trial-1",
    )

    assert post.call_args.kwargs["json"]["state"].endswith("a" * 24_000)


def test_jev_boundary_retries_transient_failures(monkeypatch) -> None:
    article = embedded_article(1)
    overloaded = Mock()
    overloaded.status_code = 529
    overloaded.headers = {"retry-after-ms": "250"}
    overloaded.raise_for_status.side_effect = requests.HTTPError(response=overloaded)
    accepted = Mock()
    accepted.headers = {"x-typesafe-request-id": "provider-request-2"}
    accepted.content = json.dumps(
        {
            "model": "jev-1.13.0",
            "answers": {"relevant": {"type": "noul", "noul": 0.7}},
            "usage": {"input_tokens": 10, "output_tokens": 1},
        }
    ).encode()
    post = Mock(side_effect=(overloaded, accepted))
    delays: list[float] = []
    monkeypatch.setattr(jev_relevance, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(jev_relevance.requests, "post", post)

    result = evaluate_jev_relevance(
        ArticleAnalysisInput(reference=article.article, article=article.value),
        execution_ref="trial-1",
        sleep=delays.append,
        clock=iter((1.0, 1.75)).__next__,
    )

    assert post.call_count == 2
    assert delays == [0.25]
    assert result.latency_ms == 750


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


def test_jev_policy_and_request_identities_are_stable() -> None:
    article = embedded_article(1).article

    assert (
        jev_relevance_policy_digest()
        == "93b8c11b8930809075cfc5c43f054aca264fcdcf14297817e8ac4b07788be4bc"
    )
    first = jev_relevance_request_id(article, "trial-1")
    assert first == jev_relevance_request_id(article, "trial-1")
    assert first != jev_relevance_request_id(article, "trial-2")
    assert (
        jev_relevance_policy_digest(
            JEV_RELEVANCE_POLICY.model_copy(update={"acceptance_threshold": 0.6})
        )
        != jev_relevance_policy_digest()
    )
