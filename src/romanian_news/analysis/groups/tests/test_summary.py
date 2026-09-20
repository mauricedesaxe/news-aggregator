import json
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest
from pydantic import HttpUrl, ValidationError

from romanian_news.analysis.groups import summary as summary_module
from romanian_news.analysis.groups.models import GroupAnalysisInput, GroupSummary
from romanian_news.analysis.groups.summary import (
    SUMMARY_COMPARISON_POLICY,
    SUMMARY_CORRECTION_POLICY,
    SUMMARY_PROMPT,
    SUMMARY_TEXT_POLICY,
    summarize_group,
    summary_request_id,
)
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.groups import NewsGroup
from romanian_news.reports import ReportArticle


def test_summary_request_identity_stays_stable() -> None:
    group = NewsGroup(id="e" * 64, article_version_ids=("a" * 64,))
    request_id = summary_request_id(group)

    assert request_id == summary_request_id(group)
    assert request_id == "0f6535e29d4d25932c1d7d11fe2d83fe8f509b82ea77cd4c2977a580d123f9b2"
    assert summary_module.SUMMARY_TITLE_POLICY == "strip-leading-news-labels-v1"
    assert SUMMARY_COMPARISON_POLICY == "distinct-outlets-only-v1"
    assert SUMMARY_CORRECTION_POLICY == "complete-json-once-v1"
    assert SUMMARY_TEXT_POLICY == "article-handle-outlet-published-title-body-12000-v2"


def test_summary_prompt_requires_english_with_archived_field_names() -> None:
    assert "English" in SUMMARY_PROMPT
    assert "legacy field names" in SUMMARY_PROMPT
    assert "values in English" in SUMMARY_PROMPT
    assert set(GroupSummary.model_json_schema()["properties"]) >= {
        "title_ro",
        "summary_ro",
        "key_points_ro",
        "disagreements_ro",
        "uncertainty_ro",
    }
    assert "4-12 word" in SUMMARY_PROMPT
    assert "1-2 sentence" in SUMMARY_PROMPT
    assert "60-word" in SUMMARY_PROMPT
    assert "3 concise" in SUMMARY_PROMPT
    assert "distinct supplied outlets" in SUMMARY_PROMPT
    assert "fewer than two distinct outlets" in SUMMARY_PROMPT
    assert "hedged title" in SUMMARY_PROMPT
    assert "unconfirmed claim" in SUMMARY_PROMPT
    assert "restating the outlet headline as fact" in SUMMARY_PROMPT


def test_summary_accepts_no_material_uncertainty() -> None:
    summary = GroupSummary.model_validate(
        {
            "title_ro": "Title",
            "summary_ro": "Summary",
            "key_points_ro": ("Key point",),
            "disagreements_ro": (),
            "uncertainty_ro": None,
            "cited_article_version_ids": ("a" * 64,),
        }
    )

    assert summary.uncertainty_ro is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title_ro", "x" * 101),
        ("summary_ro", "x" * 501),
        ("key_points_ro", ()),
        ("key_points_ro", ("one", "two", "three", "four")),
        ("key_points_ro", ("x" * 241,)),
    ],
)
def test_generated_group_summary_rejects_values_outside_storage_limits(
    field: str,
    value: object,
) -> None:
    payload = _summary_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        GroupSummary.model_validate(payload)


def test_generated_title_drops_repeated_news_labels_without_changing_source_title() -> None:
    source_title = "BREAKING: LIVE VIDEO UPDATE: România pierde fonduri PNRR"
    summary = GroupSummary.model_validate(
        {
            **_summary_payload(),
            "title_ro": source_title,
        }
    )
    article = ReportArticle(
        article_version_id="b" * 64,
        outlet_id="test",
        title=source_title,
        canonical_url="https://example.test/pnrr",
        sentiment_label="negative",
        sentiment_score=-0.8,
    )

    assert summary.title_ro == "România pierde fonduri PNRR"
    assert article.title == source_title


@pytest.mark.parametrize("title", ["Liverpool faces fiscal pressure", "Updated fiscal outlook"])
def test_generated_title_keeps_words_that_start_like_news_labels(title: str) -> None:
    summary = GroupSummary.model_validate({**_summary_payload(), "title_ro": title})

    assert summary.title_ro == title


def test_single_outlet_summary_clears_provider_differences_in_artifact(monkeypatch) -> None:
    value = _group_input(("same-outlet", "same-outlet"))
    response = SimpleNamespace(
        id="summary-response",
        model="test/model",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            **_summary_payload(),
                            "disagreements_ro": ["Bogus difference"],
                            "cited_article_version_ids": [
                                f"a{index}" for index, _ in enumerate(value.articles, start=1)
                            ],
                        }
                    )
                )
            )
        ],
        model_dump=lambda **_kwargs: {"id": "summary-response"},
    )
    monkeypatch.setattr(
        summary_module,
        "trace_provider_call",
        lambda *_args: SimpleNamespace(response=response, call_id="call", trace=None),
    )
    monkeypatch.setattr(summary_module, "record_model_attempt", lambda *_args, **_kwargs: None)

    output = summarize_group(value, "article context")
    payload = json.loads(output.content)

    assert output.summary.disagreements_ro == ()
    assert payload["summary"]["disagreements_ro"] == []
    assert payload["comparison_policy"] == SUMMARY_COMPARISON_POLICY


def test_summary_corrects_truncated_json_once(monkeypatch) -> None:
    value = _group_input(("outlet",))
    valid = json.dumps(
        {
            **_summary_payload(),
            "cited_article_version_ids": ["a1"],
        }
    )
    responses = [_response("{"), _response(valid)]
    attempts = []
    monkeypatch.setattr(
        summary_module,
        "trace_provider_call",
        lambda *_args: SimpleNamespace(response=responses.pop(0), call_id="call", trace=None),
    )
    monkeypatch.setattr(
        summary_module,
        "record_model_attempt",
        lambda *_args, **kwargs: attempts.append(kwargs),
    )

    output = summarize_group(value, "article context")

    assert [attempt["status"] for attempt in attempts] == ["rejected", "accepted"]
    assert len(json.loads(output.content)["provider_responses"]) == 2


def _group_input(outlets: tuple[str, ...]) -> GroupAnalysisInput:
    article_ids = tuple(f"{index:064x}" for index in range(1, len(outlets) + 1))
    return GroupAnalysisInput(
        day=date(2026, 8, 31),
        cluster_set=_reference("c" * 64),
        group=NewsGroup(id="e" * 64, article_version_ids=article_ids),
        articles=tuple(
            (
                _reference(version_id),
                ExtractedArticle(
                    article_id=f"{index + 10:064x}",
                    outlet_id=outlet,
                    canonical_url=HttpUrl(f"https://example.test/{index}"),
                    title=f"Title {index}",
                    body=f"Body {index}",
                    author=None,
                    published_at=datetime(2026, 8, 31, index, tzinfo=UTC),
                    source_updated_at=None,
                    bucharest_day=date(2026, 8, 31),
                    material_digest=f"{index + 20:064x}",
                    extraction_digest=f"{index + 30:064x}",
                ),
            )
            for index, (version_id, outlet) in enumerate(zip(article_ids, outlets, strict=True))
        ),
        summary_needed=True,
        sentiment_needed=False,
    )


def _reference(version_id: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"news:test:{version_id[:4]}",
        version_id=version_id,
        content_digest="f" * 64,
        r2_key=f"news/test/{version_id}.json",
    )


def _summary_payload() -> dict[str, object]:
    return {
        "title_ro": "Romania loses PNRR funds",
        "summary_ro": "Romania risks losing a material allocation after missed milestones.",
        "key_points_ro": ("The loss affects the national recovery plan.",),
        "disagreements_ro": (),
        "uncertainty_ro": None,
        "cited_article_version_ids": ("a" * 64,),
    }


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="summary-response",
        model="test/model",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model_dump=lambda **_kwargs: {"id": "summary-response"},
    )
