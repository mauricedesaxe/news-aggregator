import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import HttpUrl

from romanian_news.analysis.groups.models import (
    GroupAnalysisInput,
)
from romanian_news.analysis.groups.sentiment import (
    _normalize_quote,
    score_group_sentiment,
    sentiment_request_id,
)
from romanian_news.analysis.tracing import ModelTraceReference, ProviderCallResult
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.groups import NewsGroup

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64


class _FakeResponse:
    def __init__(
        self, response_id: str, content: str, input_tokens: int, output_tokens: int
    ) -> None:
        self.id = response_id
        self.model = "test/model"
        self.usage = SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self._payload: dict[str, object] = {
            "id": response_id,
            "model": self.model,
            "created": 1_788_172_800,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "cost": 0.01,
            },
        }

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.calls = []

        def create(**kwargs):
            self.calls.append(kwargs)
            return responses.pop(0)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _assessment_response(response_id: str, evidence_quote: str = "S1") -> _FakeResponse:
    content = json.dumps(
        {
            "label": "positive",
            "score": 0.5,
            "confidence": 0.8,
            "rationale_ro": "Efect pozitiv",
            "evidence_quote": evidence_quote,
        }
    )
    return _FakeResponse(response_id, content, 10, 5)


def _overall_response(response_id: str = "response-overall") -> _FakeResponse:
    content = json.dumps(
        {
            "label": "positive",
            "score": 0.5,
            "confidence": 0.8,
            "rationale_ro": "Efect pozitiv",
            "evidence_quote": "group",
        }
    )
    return _FakeResponse(response_id, content, 12, 6)


def _harness(
    monkeypatch,
    responses: list[_FakeResponse],
    traces: list[ModelTraceReference] | None = None,
):
    client = _FakeClient(responses)
    recorded = []
    monkeypatch.setattr("romanian_news.analysis.groups.sentiment.openrouter_client", lambda: client)

    def trace_call(_operation, _request, _inputs, call):
        response = call()
        trace = traces.pop(0) if traces else None
        return ProviderCallResult(
            response=response,
            call_id=UUID(int=len(recorded) + 1),
            trace=trace,
        )

    monkeypatch.setattr("romanian_news.analysis.groups.sentiment.trace_provider_call", trace_call)
    monkeypatch.setattr(
        "romanian_news.analysis.groups.sentiment.record_model_attempt",
        lambda *_args, **kwargs: recorded.append(kwargs),
    )
    return client, recorded


def test_sentiment_correction_records_rejects_then_accepts(monkeypatch) -> None:
    responses = [
        _assessment_response("response-invalid", evidence_quote="Text absent"),
        _assessment_response("response-accepted"),
        _overall_response(),
    ]
    client, recorded = _harness(monkeypatch, responses)

    output = score_group_sentiment(_group_input())

    assert [attempt["status"] for attempt in recorded] == ["rejected", "accepted", "accepted"]
    assert "one of these sentence IDs: S1" in client.calls[1]["messages"][-1]["content"]
    assert output.call.input_tokens == 32
    assert output.call.output_tokens == 16
    assert len(json.loads(output.content)["provider_responses"]) == 3


def test_article_sentiment_correction_exhaustion_raises_after_repeated_invalid_responses(
    monkeypatch,
) -> None:
    responses = [
        _FakeResponse("response-invalid-1", "{", 10, 5),
        _FakeResponse("response-invalid-2", "{", 10, 5),
    ]
    client, recorded = _harness(monkeypatch, responses)

    with pytest.raises(ValueError, match="Article sentiment remained invalid after correction"):
        score_group_sentiment(_group_input())

    assert [attempt["status"] for attempt in recorded] == ["rejected", "rejected"]
    assert len(client.calls) == 2


def test_sentiment_resolves_sentence_marker_to_verbatim_text(monkeypatch) -> None:
    responses = [_assessment_response("response-article"), _overall_response()]
    _harness(monkeypatch, responses)

    output = score_group_sentiment(_group_input())

    assert output.sentiment.articles[0].evidence_quote == "Titlu\nDovada exacta este aici."
    assert output.sentiment.articles[0].article_version_id == _A
    assert not hasattr(output.sentiment.overall, "evidence_quote")


def test_article_sentiment_schema_only_allows_present_sentence_markers(monkeypatch) -> None:
    responses = [_assessment_response("response-article"), _overall_response()]
    client, _recorded = _harness(monkeypatch, responses)

    score_group_sentiment(_group_input())

    schema = client.calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["evidence_quote"]["enum"] == ["S1"]


def test_quote_normalization_unifies_romanian_comma_and_cedilla_variants() -> None:
    assert _normalize_quote("președintele Nicușor") == _normalize_quote("preşedintele Nicuşor")


def test_sentiment_request_identity_stays_stable() -> None:
    group = NewsGroup(id="e" * 64, article_version_ids=(_A,))

    assert sentiment_request_id(group) == sentiment_request_id(group)


def _group_input() -> GroupAnalysisInput:
    return GroupAnalysisInput(
        day=date(2026, 8, 31),
        cluster_set=_reference(_B),
        group=NewsGroup(id=_C, article_version_ids=(_A,)),
        articles=((_reference(_A), _article()),),
    )


def _article() -> ExtractedArticle:
    return ExtractedArticle(
        article_id=_A,
        outlet_id="test",
        canonical_url=HttpUrl("https://example.test/article"),
        title="Titlu",
        body="Dovada exacta este aici.",
        author=None,
        published_at=datetime(2026, 8, 31, 9, tzinfo=UTC),
        source_updated_at=None,
        bucharest_day=date(2026, 8, 31),
        material_digest=_B,
        extraction_digest=_C,
    )


def _reference(version_id: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"artifact:{version_id[:4]}",
        version_id=version_id,
        content_digest=_D,
        r2_key=f"objects/{version_id}",
    )
