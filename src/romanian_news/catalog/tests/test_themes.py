import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from romanian_news import themes as construction_module
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.groups.models import GroupSummary
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

FIXTURE = Path(__file__).parents[2] / "tests" / "fixtures" / "recorded_v1_theme_set.json"


def test_daily_theme_publication_records_exact_ordered_inputs_and_model_evidence(
    monkeypatch,
) -> None:
    output = DailyThemeOutput.model_validate_json(FIXTURE.read_text())
    batches = []
    monkeypatch.setattr(themes, "current_artifact_file", lambda _artifact_id: None)
    monkeypatch.setattr(themes, "run_status", lambda _run_id: None)
    monkeypatch.setattr(
        themes,
        "publish_immutable_r2_objects",
        lambda _objects: SimpleNamespace(uploaded_objects=1, reused_objects=0),
    )
    monkeypatch.setattr(themes, "catalog_batch", batches.append)

    publication = themes.publish_daily_themes(output, "git:test")

    statements = batches[0]
    inputs = [params for sql, params in statements if "INTO run_inputs" in sql]
    assert [params[3] for params in inputs] == ["cluster_set", "summary"]
    assert [params[2] for params in inputs] == [
        output.theme_set.cluster_set.version_id,
        output.theme_set.summary_inputs[0].version_id,
    ]
    model_calls = [params for sql, params in statements if "INTO news_model_calls" in sql]
    assert len(model_calls) == 1
    assert model_calls[0][0] == publication.reference.version_id


def test_sparse_publication_aggregates_two_stages_and_run_identity(monkeypatch) -> None:
    output = _sparse_output(monkeypatch, merged=True)
    statements, publication = _publish(monkeypatch, output)
    model_call = next(params for sql, params in statements if "INTO news_model_calls" in sql)

    assert isinstance(output.theme_set, AliasedReaderSubjectThemeSet)
    construction = output.theme_set.construction
    assert isinstance(construction, SparseThemeConstruction)
    assert construction.merged_prose is not None
    response_ids = (
        construction.assignment.call.response_id,
        construction.merged_prose.call.response_id,
    )
    assert publication.run_id == daily_theme_run_id(output.theme_set.request_id, response_ids)
    assert model_call[2:5] == ["google/gemini-3.8-flash", 20, 10]
    assert model_call[5] == pytest.approx(0.02)
    assert model_call[7] == 2


def test_sparse_publication_aggregates_one_stage(monkeypatch) -> None:
    output = _sparse_output(monkeypatch, merged=False)
    statements, publication = _publish(monkeypatch, output)
    model_call = next(params for sql, params in statements if "INTO news_model_calls" in sql)

    assert isinstance(output.theme_set, AliasedReaderSubjectThemeSet)
    construction = output.theme_set.construction
    assert isinstance(construction, SparseThemeConstruction)
    response_ids = (construction.assignment.call.response_id,)
    assert publication.run_id == daily_theme_run_id(output.theme_set.request_id, response_ids)
    assert model_call[2:5] == ["google/gemini-3.8-flash", 10, 5]
    assert model_call[5] == pytest.approx(0.01)
    assert model_call[7] == 1


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
    monkeypatch.setattr(
        construction_module,
        "openrouter_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: next(response_iterator))
            )
        ),
    )
    monkeypatch.setattr(
        construction_module,
        "record_model_attempt",
        lambda response, **_kwargs: SimpleNamespace(
            attempt_id=hashlib.sha256(response.id.encode()).hexdigest(),
            response_id=response.id,
        ),
    )
    return construct_daily_themes(value)


def _publish(monkeypatch, output: DailyThemeOutput):
    batches = []
    monkeypatch.setattr(themes, "current_artifact_file", lambda _artifact_id: None)
    monkeypatch.setattr(themes, "run_status", lambda _run_id: None)
    monkeypatch.setattr(
        themes,
        "publish_immutable_r2_objects",
        lambda _objects: SimpleNamespace(uploaded_objects=1, reused_objects=0),
    )
    monkeypatch.setattr(themes, "catalog_batch", batches.append)
    publication = themes.publish_daily_themes(output, "git:test")
    return batches[0], publication


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
