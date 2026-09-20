import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import HttpUrl, ValidationError

from romanian_news.analysis import relevance as relevance_module
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.relevance import (
    PRODUCTION_RELEVANCE_POLICY,
    RELEVANCE_POLICY_V1,
    RELEVANCE_POLICY_V2,
    ArticleAnalysisInput,
    RelevanceDecision,
    analyze_relevance,
    relevance_is_accepted,
    relevance_policy_digest,
    relevance_request_id,
)
from romanian_news.analysis.tracing import ProviderCallResult
from romanian_news.articles.models import ExtractedArticle

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.id = "response-relevance"
        self.model = "test/model"
        self.usage = SimpleNamespace(prompt_tokens=12, completion_tokens=7)
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        return {
            "id": self.id,
            "model": self.model,
            "created": 1_788_172_800,
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "cost": 0.004},
        }


class _FakeClient:
    def __init__(self, *responses: _FakeResponse) -> None:
        values = iter(responses)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: next(values))
        )


def test_relevance_uses_exact_title_after_invalid_correction(monkeypatch) -> None:
    response = _FakeResponse(
        json.dumps(
            {
                "national_reach": "nationwide",
                "consequence_magnitude": "routine",
                "political_relevance": "strong",
                "economic_relevance": "none",
                "romania_relevance": "strong",
                "confidence": 0.9,
                "evidence_quote": "Text absent",
                "reason_ro": "Relevant",
            }
        )
    )
    monkeypatch.setattr(
        "romanian_news.analysis.relevance.openrouter_client",
        lambda: _FakeClient(response, response),
    )
    monkeypatch.setattr(
        "romanian_news.analysis.relevance.trace_provider_call",
        lambda _operation, _request, _inputs, call: ProviderCallResult(
            response=call(),
            call_id=UUID("018f0000-0000-7000-8000-000000000001"),
            trace=None,
        ),
    )
    connection = _relevance_database()
    monkeypatch.setattr(
        "romanian_news.catalog.model_calls.catalog_batch", _sqlite_batch(connection)
    )

    output = analyze_relevance(_analysis_input())

    rejected = connection.execute(
        "SELECT * FROM news_model_attempts WHERE status = 'rejected'"
    ).fetchall()
    accepted = connection.execute(
        "SELECT * FROM news_model_attempts WHERE status = 'accepted'"
    ).fetchall()
    assert len(rejected) == 1
    assert "evidence quote" in rejected[0]["error"]
    assert len(accepted) == 1
    assert accepted[0]["input_tokens"] == 12
    assert accepted[0]["output_tokens"] == 7
    assert accepted[0]["cost_usd"] == 0.004
    assert output.decision.evidence_quote == "Titlu"
    assert output.policy == RELEVANCE_POLICY_V2
    assert output.policy_digest == relevance_policy_digest(RELEVANCE_POLICY_V2)
    assert output.accepted is True
    output_fields = output.model_dump()
    assert output_fields["policy"] == RELEVANCE_POLICY_V2.model_dump()
    assert not {"policy_id", "policy_digest", "accepted"}.intersection(output_fields)
    payload = json.loads(output.content)
    assert payload["accepted"] is True


def test_relevance_corrects_an_invalid_evidence_quote_once(monkeypatch) -> None:
    invalid = _FakeResponse(
        json.dumps(
            {
                "national_reach": "nationwide",
                "consequence_magnitude": "routine",
                "political_relevance": "strong",
                "economic_relevance": "none",
                "romania_relevance": "strong",
                "confidence": 0.9,
                "evidence_quote": "Paraphrased evidence",
                "reason_ro": "Relevant",
            }
        )
    )
    corrected = _FakeResponse(
        json.dumps(
            {
                "national_reach": "nationwide",
                "consequence_magnitude": "routine",
                "political_relevance": "strong",
                "economic_relevance": "none",
                "romania_relevance": "strong",
                "confidence": 0.9,
                "evidence_quote": "Dovada exacta",
                "reason_ro": "Relevant",
            }
        )
    )
    client = _FakeClient(invalid, corrected)
    recorded = []
    monkeypatch.setattr("romanian_news.analysis.relevance.openrouter_client", lambda: client)
    monkeypatch.setattr(
        "romanian_news.analysis.relevance.record_model_attempt",
        lambda *_args, **kwargs: recorded.append(kwargs),
    )

    output = analyze_relevance(_analysis_input())

    assert output.decision.evidence_quote == "Dovada exacta"
    assert [value["status"] for value in recorded] == ["rejected", "accepted"]
    payload = json.loads(output.content)
    assert len(payload["provider_responses"]) == 2


def test_relevance_request_identity_stays_stable() -> None:
    request_id = relevance_request_id(_reference(_A))

    assert request_id == relevance_request_id(_reference(_A))
    assert request_id == "1552b3d39663e37142343cbd8f3e69f475a06b55849b8eaad43d16397f68c750"
    assert PRODUCTION_RELEVANCE_POLICY is RELEVANCE_POLICY_V2


@pytest.mark.parametrize(
    ("case", "decision", "accepted"),
    [
        (
            "Putin without a national Romanian consequence",
            {
                "national_reach": "none",
                "consequence_magnitude": "major",
                "political_relevance": "strong",
                "economic_relevance": "none",
                "romania_relevance": "none",
            },
            False,
        ),
        (
            "local traffic restriction",
            {
                "national_reach": "sector_limited",
                "consequence_magnitude": "routine",
                "political_relevance": "strong",
                "economic_relevance": "none",
                "romania_relevance": "strong",
            },
            False,
        ),
        (
            "driver license and registration production fees",
            {
                "national_reach": "nationwide",
                "consequence_magnitude": "narrow",
                "political_relevance": "strong",
                "economic_relevance": "strong",
                "romania_relevance": "strong",
            },
            False,
        ),
        (
            "Romania loses PNRR funds",
            {
                "national_reach": "nationwide",
                "consequence_magnitude": "major",
                "political_relevance": "strong",
                "economic_relevance": "strong",
                "romania_relevance": "strong",
            },
            True,
        ),
    ],
)
def test_relevance_policy_accepts_only_material_nationwide_romanian_consequences(
    case: str,
    decision: dict[str, object],
    accepted: bool,
) -> None:
    value = RelevanceDecision.model_validate(
        {
            **decision,
            "confidence": 0.9,
            "evidence_quote": case,
            "reason_ro": case,
        }
    )

    assert relevance_is_accepted(value, RELEVANCE_POLICY_V1) is accepted
    assert "accepted" not in value.model_dump()


def test_relevance_policies_are_immutable() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        RELEVANCE_POLICY_V1.prompt = "Changed"


@pytest.mark.parametrize(
    "field",
    (
        "policy_id",
        "prompt",
        "response_schema_name",
        "correction_instruction",
        "evidence_policy",
        "text_policy",
    ),
)
def test_relevance_policy_rejects_empty_identity_inputs(field: str) -> None:
    payload = RELEVANCE_POLICY_V1.model_dump()
    payload[field] = ""

    with pytest.raises(ValidationError, match="at least 1 character"):
        relevance_module.RelevancePolicy.model_validate(payload)


def test_policy_identity_changes_for_each_behavior_category() -> None:
    reference = _reference(_A)
    variants = (
        RELEVANCE_POLICY_V1.model_copy(update={"prompt": "Changed prompt"}),
        RELEVANCE_POLICY_V1.model_copy(update={"max_tokens": 801}),
        RELEVANCE_POLICY_V1.model_copy(update={"accepted_consequence_magnitudes": ("major",)}),
    )

    for variant in variants:
        assert relevance_policy_digest(variant) != relevance_policy_digest(RELEVANCE_POLICY_V1)
        assert relevance_request_id(reference, variant) != relevance_request_id(
            reference, RELEVANCE_POLICY_V1
        )


def _relevance_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    schema = Path(__file__).parents[4] / "tests" / "fixtures" / "sqlite_catalog.sql"
    connection.executescript(schema.read_text())
    return connection


def _sqlite_batch(connection: sqlite3.Connection):
    def batch(statements, *, retry_transient_errors=False):
        for sql, params in statements:
            connection.execute(sql.replace("%s", "?"), params)
        connection.commit()

    return batch


def _analysis_input() -> ArticleAnalysisInput:
    return ArticleAnalysisInput(reference=_reference(_A), article=_article())


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
