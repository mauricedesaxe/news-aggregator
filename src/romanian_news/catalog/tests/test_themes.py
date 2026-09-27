import hashlib
import json
from datetime import date
from types import SimpleNamespace

import pytest

from romanian_news import themes as construction_module
from romanian_news.analysis import corrected_structured
from romanian_news.analysis.groups.models import GroupSummary
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog import themes
from romanian_news.groups import NewsGroup
from romanian_news.themes import (
    AliasedReaderSubjectThemeSet,
    DailyThemeInput,
    DailyThemeOutput,
    SparseThemeConstruction,
    ThemeGroupInput,
    construct_daily_themes,
    daily_theme_run_id,
)


def test_sparse_publication_rejects_mixed_stage_models(monkeypatch) -> None:
    output = _sparse_output(monkeypatch, merged=True)
    theme_set = output.theme_set
    assert isinstance(theme_set, AliasedReaderSubjectThemeSet)
    construction = theme_set.construction
    assert isinstance(construction, SparseThemeConstruction)
    assert construction.merged_prose is not None
    changed_prose = construction.merged_prose.model_copy(
        update={"call": construction.merged_prose.call.model_copy(update={"model": "other/model"})}
    )
    changed_construction = construction.model_copy(update={"merged_prose": changed_prose})
    changed_output = output.model_copy(
        update={"theme_set": theme_set.model_copy(update={"construction": changed_construction})}
    )

    with pytest.raises(ValueError, match="model does not match"):
        themes.publish_daily_themes(changed_output, "git:test")


def test_version_one_run_identity_keeps_original_hash() -> None:
    request_id = "a" * 64
    response_id = "response-1"

    assert (
        daily_theme_run_id(request_id, response_id)
        == hashlib.sha256(f"{request_id}\0{response_id}".encode()).hexdigest()
    )


def _sparse_output(monkeypatch, *, merged: bool) -> DailyThemeOutput:
    groups = (
        NewsGroup(id="a" * 64, article_version_ids=("1" * 64,)),
        NewsGroup(id="b" * 64, article_version_ids=("2" * 64,)),
    )
    value = DailyThemeInput(
        day=date(2026, 9, 2),
        cluster_set=_reference("clusters", "3"),
        groups=tuple(
            ThemeGroupInput(
                group=group,
                summary=_reference(f"summary-{index}", str(index + 4)),
                value=GroupSummary(
                    title_ro=f"Event {index}",
                    summary_ro=f"Prima {index}. A doua. A treia.",
                    key_points_ro=("Punct",),
                    disagreements_ro=(),
                    cited_article_version_ids=group.article_version_ids,
                ),
            )
            for index, group in enumerate(groups)
        ),
    )
    labels = (1, 1) if merged else (1, 2)
    responses = [
        _response(
            {"assignments": {f"g{index}": label for index, label in enumerate(labels, start=1)}},
            "assignment",
        )
    ]
    if merged:
        responses.append(
            _response(
                {"themes": {"theme_01": {"title": "Tema", "summary": "Rezumat."}}},
                "prose",
            )
        )
    response_iterator = iter(responses)
    monkeypatch.setattr(construction_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(corrected_structured.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        corrected_structured,
        "openrouter_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: next(response_iterator))
            )
        ),
    )
    monkeypatch.setattr(
        corrected_structured,
        "record_model_attempt",
        lambda response, **_kwargs: SimpleNamespace(
            attempt_id=hashlib.sha256(response.id.encode()).hexdigest(),
            response_id=response.id,
        ),
    )
    return construct_daily_themes(value)


def _reference(name: str, character: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"news:{name}",
        version_id=character * 64,
        content_digest="0" * 64,
        r2_key=f"news/{name}.json",
    )


def _response(payload: dict[str, object], response_id: str):
    content = json.dumps(payload)
    provider_payload = {
        "id": response_id,
        "model": "google/gemini-3.8-flash",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01},
        "choices": [{"message": {"content": content}}],
    }
    return SimpleNamespace(
        id=response_id,
        model="google/gemini-3.8-flash",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        choices=(SimpleNamespace(message=SimpleNamespace(content=content)),),
        model_dump=lambda *, mode: provider_payload,
    )
