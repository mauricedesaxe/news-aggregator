import json
from datetime import UTC, date, datetime
from types import SimpleNamespace

from pydantic import HttpUrl

from romanian_news.analysis.groups import sentiment as sentiment_module
from romanian_news.analysis.groups import summary as summary_module
from romanian_news.analysis.groups.models import GroupAnalysisInput
from romanian_news.analysis.groups.workflow import analyze_group
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.groups import NewsGroup

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64

_SUMMARY_CONTENT = json.dumps(
    {
        "title_ro": "Title",
        "summary_ro": "Summary",
        "key_points_ro": ("Key point",),
        "disagreements_ro": (),
        "uncertainty_ro": None,
        "cited_article_version_ids": ["a1"],
    }
)


class _ChatResponse:
    def __init__(self, content: str) -> None:
        self.id = "workflow-response"
        self.model = "test/model"
        self.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self._payload: dict[str, object] = {"id": self.id, "model": self.model}

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        return self._payload


def _sentiment_contents() -> list[str]:
    assessment = json.dumps(
        {
            "label": "positive",
            "score": 0.5,
            "confidence": 0.8,
            "rationale_ro": "Efect pozitiv",
            "evidence_quote": "S1",
        }
    )
    overall = json.dumps(
        {
            "label": "positive",
            "score": 0.5,
            "confidence": 0.8,
            "rationale_ro": "Efect pozitiv",
            "evidence_quote": "group",
        }
    )
    return [assessment, overall]


def _trace(calls: list[object], contents: list[str]):
    responses = [_ChatResponse(content) for content in contents]

    def trace(_operation, _request_id, _inputs, _call):
        calls.append(_inputs)
        return SimpleNamespace(response=responses.pop(0), call_id="call", trace=None)

    return trace


def _wire_provider_fakes(
    monkeypatch, summary_contents: list[str]
) -> tuple[list[object], list[object]]:
    summary_calls: list[object] = []
    sentiment_calls: list[object] = []
    monkeypatch.setattr(
        summary_module,
        "trace_provider_call",
        _trace(summary_calls, summary_contents),
    )
    monkeypatch.setattr(
        sentiment_module,
        "trace_provider_call",
        _trace(sentiment_calls, _sentiment_contents()),
    )
    monkeypatch.setattr(summary_module, "record_model_attempt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sentiment_module, "record_model_attempt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        summary_module,
        "openrouter_client",
        lambda: (_ for _ in ()).throw(AssertionError("summary client constructed")),
    )
    monkeypatch.setattr(
        sentiment_module,
        "openrouter_client",
        lambda: (_ for _ in ()).throw(AssertionError("sentiment client constructed")),
    )
    return summary_calls, sentiment_calls


def test_analyze_group_skips_the_summary_model_when_summary_is_not_needed(monkeypatch) -> None:
    summary_calls, _sentiment_calls = _wire_provider_fakes(monkeypatch, [_SUMMARY_CONTENT])

    output = analyze_group(_group_input(summary_needed=False, sentiment_needed=True))

    assert output.summary is None
    assert output.sentiment is not None
    assert output.errors == ()
    assert summary_calls == []


def test_analyze_group_skips_the_sentiment_model_when_sentiment_is_not_needed(monkeypatch) -> None:
    _summary_calls, sentiment_calls = _wire_provider_fakes(monkeypatch, [_SUMMARY_CONTENT])

    output = analyze_group(_group_input(summary_needed=True, sentiment_needed=False))

    assert output.summary is not None
    assert output.sentiment is None
    assert output.errors == ()
    assert sentiment_calls == []


def test_analyze_group_records_a_summary_branch_error_and_still_completes(monkeypatch) -> None:
    _wire_provider_fakes(monkeypatch, ["{", "{"])

    output = analyze_group(_group_input(summary_needed=True, sentiment_needed=False))

    assert output.summary is None
    assert output.sentiment is None
    assert len(output.errors) == 1
    assert output.errors[0].startswith("summary: ")
    assert "remained invalid after correction" in output.errors[0]


def _group_input(*, summary_needed: bool, sentiment_needed: bool) -> GroupAnalysisInput:
    return GroupAnalysisInput(
        day=date(2026, 8, 31),
        cluster_set=_reference(_B),
        group=NewsGroup(id=_C, article_version_ids=(_A,)),
        articles=((_reference(_A), _article()),),
        summary_needed=summary_needed,
        sentiment_needed=sentiment_needed,
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
