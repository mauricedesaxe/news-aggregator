import copy
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from romanian_news import themes as themes_module
from romanian_news.analysis import corrected_structured
from romanian_news.analysis.groups.models import GroupSummary
from romanian_news.artifacts import ArtifactReference
from romanian_news.groups import NewsGroup
from romanian_news.themes import (
    LEGACY_SPARSE_THEME_POLICY,
    LEGACY_THEME_POLICY,
    PRODUCTION_THEME_DEFINITION,
    AliasedReaderSubjectThemeSet,
    DailyThemeCorrectionExhaustedError,
    DailyThemeInput,
    DailyThemeOutput,
    DailyThemeSet,
    EmptyThemeConstruction,
    ModelThemeConstruction,
    SparseThemeConstruction,
    SparseThemePolicyDefinition,
    ThemeGroupInput,
    construct_daily_themes,
    daily_theme_id,
    daily_theme_request_id,
    parse_daily_theme_set,
    theme_policy_digest,
)

DAY = date(2026, 9, 2)
FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_v1_theme_set.json"


def test_policy_definition_rejects_prompt_digest_mismatch() -> None:
    with pytest.raises(ValueError, match="assignment prompt digest"):
        SparseThemePolicyDefinition(
            policy=LEGACY_SPARSE_THEME_POLICY,
            assignment_prompt="changed",
            prose_prompt=PRODUCTION_THEME_DEFINITION.prose_prompt,
        )


def test_candidate_uses_versioned_policy_and_its_prompt(monkeypatch) -> None:
    value = _input(2)
    _provider(
        monkeypatch,
        [_response({"assignments": _assignments(value, (1, 2))}, "assignment")],
    )

    output = construct_daily_themes(value)

    assert isinstance(output.theme_set, AliasedReaderSubjectThemeSet)
    assert output.theme_set.schema_version == 4
    assert parse_daily_theme_set(output.content) == output.theme_set


def test_all_singletons_use_one_call_and_source_prose(monkeypatch) -> None:
    value = _input(3)
    calls = _provider(
        monkeypatch,
        [_response({"assignments": _assignments(value, (3, 1, 2))}, "assignment")],
    )

    output = construct_daily_themes(value)

    assert len(calls) == 1
    assert output.theme_set.schema_version == 4
    assert tuple(theme.title for theme in output.theme_set.themes) == (
        "Event 0",
        "Event 1",
        "Event 2",
    )
    assert output.theme_set.themes[0].summary == "Dr. Popescu a vorbit. A plecat apoi."
    construction = output.theme_set.construction
    assert isinstance(construction, SparseThemeConstruction)
    assert construction.merged_prose is None
    assert construction.assignment.request_id != output.theme_set.request_id


def test_singleton_summary_shorter_than_two_sentences_stays_intact(monkeypatch) -> None:
    value = _input(1)
    summary = value.groups[0].value.model_copy(update={"summary_ro": "O singură propoziție."})
    value = value.model_copy(
        update={"groups": (value.groups[0].model_copy(update={"value": summary}),)}
    )
    _provider(
        monkeypatch,
        [_response({"assignments": _assignments(value, (1,))}, "assignment")],
    )

    output = construct_daily_themes(value)

    assert output.theme_set.themes[0].summary == "O singură propoziție."


def test_singleton_summary_does_not_split_an_initialism(monkeypatch) -> None:
    value = _input(1)
    summary = value.groups[0].value.model_copy(
        update={"summary_ro": "România discută cu S.U.A. A doua propoziție. A treia propoziție."}
    )
    value = value.model_copy(
        update={"groups": (value.groups[0].model_copy(update={"value": summary}),)}
    )
    _provider(
        monkeypatch,
        [_response({"assignments": _assignments(value, (1,))}, "assignment")],
    )

    output = construct_daily_themes(value)

    assert output.theme_set.themes[0].summary == ("România discută cu S.U.A. A doua propoziție.")


def test_mixed_flow_calls_assignment_and_merged_prose_once(monkeypatch) -> None:
    value = _input(3)
    calls = _provider(
        monkeypatch,
        [
            _response({"assignments": _assignments(value, (3, 3, 1))}, "assignment"),
            _response(
                {"themes": {"theme_01": {"title": "Tema unită", "summary": "Rezumat unit."}}},
                "prose",
            ),
        ],
    )

    output = construct_daily_themes(value)

    assert len(calls) == 2
    assert output.theme_set.themes[0].group_ids == ("a" * 64, "b" * 64)
    assert output.theme_set.themes[1].title == "Event 2"
    assert "Event 2" not in calls[1]["messages"][1]["content"]
    assert "Event 0" in calls[1]["messages"][1]["content"]
    construction = output.theme_set.construction
    assert isinstance(construction, SparseThemeConstruction)
    assert construction.merged_prose is not None
    assert construction.assignment.request_id != construction.merged_prose.request_id
    schema = calls[1]["response_format"]["json_schema"]["schema"]
    assert tuple(schema["properties"]) == ("themes",)
    theme_schema = schema["properties"]["themes"]
    assert tuple(theme_schema["properties"]) == ("theme_01",)
    assert set(theme_schema["properties"]["theme_01"]["properties"]) == {
        "title",
        "summary",
    }


def test_zero_padded_keys_sort_past_nine(monkeypatch) -> None:
    value = _input(12)
    labels = (1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10)
    calls = _provider(
        monkeypatch,
        [
            _response({"assignments": _assignments(value, labels)}, "assignment"),
            _response(
                {
                    "themes": {
                        "theme_10": {"title": "Ultima temă", "summary": "Două grupuri."},
                        "theme_02": {"title": "A doua temă", "summary": "Alte două grupuri."},
                    }
                },
                "prose",
            ),
        ],
    )

    output = construct_daily_themes(value)

    assert output.theme_set.themes[-1].group_ids == tuple(
        item.group.id for item in value.groups[-2:]
    )
    prose_context = json.loads(calls[1]["messages"][1]["content"])
    assert prose_context["day"] == DAY.isoformat()
    assert tuple(prose_context["themes"]) == ("theme_02", "theme_10")


def test_merged_summary_with_three_sentences_gets_one_correction(monkeypatch) -> None:
    value = _input(2)
    calls = _provider(
        monkeypatch,
        [
            _response({"assignments": _assignments(value, (1, 1))}, "assignment"),
            _response(
                {
                    "themes": {
                        "theme_01": {
                            "title": "Tema",
                            "summary": "Prima. A doua. A treia.",
                        }
                    }
                },
                "prose-invalid",
            ),
            _response(
                {
                    "themes": {
                        "theme_01": {
                            "title": "Tema",
                            "summary": "Prima. A doua.",
                        }
                    }
                },
                "prose-valid",
            ),
        ],
    )

    output = construct_daily_themes(value)

    assert len(calls) == 3
    assert output.theme_set.themes[0].summary == "Prima. A doua."


def test_duplicate_assignment_keys_trigger_one_correction(monkeypatch) -> None:
    value = _input(2)
    duplicate = '{"assignments":{"' + "a" * 64 + '":1,"' + "a" * 64 + '":2,"' + "b" * 64 + '":2}}'
    calls = _provider(
        monkeypatch,
        [
            _response(duplicate, "assignment-1"),
            _response({"assignments": _assignments(value, (1, 2))}, "assignment-2"),
        ],
    )

    output = construct_daily_themes(value)

    assert len(calls) == 2
    assert "repeats property" in calls[1]["messages"][-1]["content"]
    assert tuple(theme.group_ids for theme in output.theme_set.themes) == (
        ("a" * 64,),
        ("b" * 64,),
    )


def test_prose_cannot_add_membership(monkeypatch) -> None:
    value = _input(2)
    invalid: dict[str, object] = {
        "themes": {
            "theme_01": {
                "title": "Tema",
                "summary": "Rezumat.",
                "group_ids": ["a" * 64],
            }
        }
    }
    _provider(
        monkeypatch,
        [
            _response({"assignments": _assignments(value, (1, 1))}, "assignment"),
            _response(invalid, "prose-1"),
            _response(invalid, "prose-2"),
        ],
    )

    with pytest.raises(DailyThemeCorrectionExhaustedError, match="merged prose stage") as raised:
        construct_daily_themes(value)

    assert isinstance(raised.value.__cause__, Exception)


def test_assignment_corrects_once_then_preserves_stage_cause(monkeypatch) -> None:
    value = _input(2)
    _provider(
        monkeypatch,
        [
            _response({"assignments": {"a" * 64: 1}}, "assignment-1"),
            _response({"assignments": {"a" * 64: 1}}, "assignment-2"),
        ],
    )

    with pytest.raises(DailyThemeCorrectionExhaustedError, match="assignment stage") as raised:
        construct_daily_themes(value)

    assert isinstance(raised.value.__cause__, ValueError)


def test_each_stage_gets_one_semantic_correction(monkeypatch) -> None:
    value = _input(2)
    calls = _provider(
        monkeypatch,
        [
            _response({"assignments": _assignments(value, (1, 1))}, "assignment"),
            _response(
                {"themes": {"theme_02": {"title": "Greșit", "summary": "Greșit."}}}, "prose-1"
            ),
            _response(
                {"themes": {"theme_02": {"title": "Greșit", "summary": "Greșit."}}}, "prose-2"
            ),
        ],
    )

    with pytest.raises(DailyThemeCorrectionExhaustedError, match="merged prose stage") as raised:
        construct_daily_themes(value)

    assert len(calls) == 3
    assert isinstance(raised.value.__cause__, ValueError)


def test_empty_input_makes_no_calls(monkeypatch) -> None:
    calls = _provider(monkeypatch, [])
    value = _input(0)

    output = construct_daily_themes(value)

    assert calls == []
    assert output.theme_set.themes == ()


def test_sparse_evidence_rejects_tampered_assignment(monkeypatch) -> None:
    document = _mixed_document(monkeypatch)
    assignment = document["construction"]["assignment"]["attempts"][-1]
    content = json.dumps({"assignments": {"g1": 1, "g2": 2, "g3": 3}})
    _replace_attempt_content(assignment, content)

    with pytest.raises(ValueError, match="memberships do not match"):
        parse_daily_theme_set(json.dumps(document).encode())


def test_sparse_evidence_rejects_tampered_prose(monkeypatch) -> None:
    document = _mixed_document(monkeypatch)
    prose = document["construction"]["merged_prose"]["attempts"][-1]
    content = json.dumps(
        {"themes": {"theme_01": {"title": "Schimbat", "summary": "Rezumat unit."}}}
    )
    _replace_attempt_content(prose, content)

    with pytest.raises(ValueError, match="prose does not match"):
        parse_daily_theme_set(json.dumps(document).encode())


def test_sparse_evidence_rejects_tampered_singleton_prose(monkeypatch) -> None:
    document = _mixed_document(monkeypatch)
    document["themes"][1]["title"] = "Schimbat"

    with pytest.raises(ValueError, match="source summary evidence"):
        parse_daily_theme_set(json.dumps(document).encode())


def test_sparse_evidence_rejects_merged_themes_without_prose_evidence(monkeypatch) -> None:
    document = _mixed_document(monkeypatch)
    document["construction"]["merged_prose"] = None

    with pytest.raises(ValueError, match="Merged themes require merged prose evidence"):
        parse_daily_theme_set(json.dumps(document).encode())


def test_sparse_evidence_rejects_a_stage_call_from_another_model(monkeypatch) -> None:
    document = _mixed_document(monkeypatch)
    document["construction"]["assignment"]["call"]["model"] = "other/model"

    with pytest.raises(ValueError, match="model does not match the sparse policy"):
        parse_daily_theme_set(json.dumps(document).encode())


def test_load_daily_theme_input_accepts_summary_from_prior_cluster_set(monkeypatch) -> None:
    group = NewsGroup(id="a" * 64, article_version_ids=("1" * 64,))
    cluster_set = _reference("clusters", "4")
    summary = _reference("summary-a", "5")
    persisted = {
        cluster_set.r2_key: json.dumps(
            {
                "day": DAY.isoformat(),
                "algorithm": "test",
                "threshold": 0.5,
                "embedding_model": "test/model",
                "article_version_ids": list(group.article_version_ids),
                "relevance_version_ids": ["2" * 64],
                "embedding_version_ids": ["3" * 64],
                "merges": [],
                "groups": [group.model_dump(mode="json")],
            }
        ).encode(),
        summary.r2_key: json.dumps(
            {
                "group_id": group.id,
                "cluster_set_version_id": "3" * 64,
                "summary": {
                    "title_ro": "Eveniment",
                    "summary_ro": "Rezumat",
                    "key_points_ro": ["Punct"],
                    "disagreements_ro": [],
                    "cited_article_version_ids": list(group.article_version_ids),
                },
            }
        ).encode(),
    }
    monkeypatch.setattr(
        themes_module,
        "read_verified_r2_object",
        lambda r2_key, _content_digest: persisted[r2_key],
    )

    value = themes_module.load_daily_theme_input(cluster_set, (summary,))

    assert value.groups[0].value == GroupSummary(
        title_ro="Eveniment",
        summary_ro="Rezumat",
        key_points_ro=("Punct",),
        disagreements_ro=(),
        cited_article_version_ids=group.article_version_ids,
    )


def test_daily_theme_request_identity_covers_exact_inputs_and_order() -> None:
    value = _input(3)
    changed_summary = value.model_copy(
        update={
            "groups": (
                value.groups[0].model_copy(update={"summary": _reference("summary-a", "9")}),
                *value.groups[1:],
            )
        }
    )
    reordered = value.model_copy(update={"groups": tuple(reversed(value.groups))})

    assert daily_theme_request_id(value, LEGACY_THEME_POLICY) != daily_theme_request_id(
        changed_summary, LEGACY_THEME_POLICY
    )
    assert daily_theme_request_id(value, LEGACY_THEME_POLICY) != daily_theme_request_id(
        reordered, LEGACY_THEME_POLICY
    )


def test_prompt_digest_changes_request_and_theme_identities_without_changing_groups() -> None:
    value = _input(3)
    prior_prompt = themes_module.THEME_PROMPT.replace("English", "Romanian")
    prior_policy = LEGACY_THEME_POLICY.model_copy(
        update={"prompt_digest": hashlib.sha256(prior_prompt.encode()).hexdigest()}
    )
    ordered_group_ids = tuple(item.group.id for item in value.groups[:2])

    assert daily_theme_request_id(value, LEGACY_THEME_POLICY) != daily_theme_request_id(
        value, prior_policy
    )
    assert daily_theme_id(
        value.day, ordered_group_ids, theme_policy_digest(LEGACY_THEME_POLICY)
    ) != daily_theme_id(value.day, ordered_group_ids, theme_policy_digest(prior_policy))


def test_sparse_request_identity_changes_with_the_day(monkeypatch) -> None:
    value = _input(2)
    next_day = value.model_copy(update={"day": DAY + timedelta(days=1)})
    _provider(
        monkeypatch,
        [
            _response({"assignments": _assignments(value, (1, 2))}, "assignment-a"),
            _response({"assignments": _assignments(next_day, (1, 2))}, "assignment-b"),
        ],
    )

    first = construct_daily_themes(value).theme_set
    second = construct_daily_themes(next_day).theme_set

    assert first.request_id != second.request_id
    assert first.day == DAY
    assert second.day == DAY + timedelta(days=1)


def test_sparse_request_identity_covers_durable_output_schema(monkeypatch) -> None:
    value = _input(2)
    _provider(
        monkeypatch,
        [
            _response({"assignments": _assignments(value, (1, 2))}, "assignment-a"),
            _response({"assignments": _assignments(value, (1, 2))}, "assignment-b"),
        ],
    )

    original = construct_daily_themes(value).theme_set
    monkeypatch.setattr(
        themes_module,
        "_aliased_reader_subject_theme_set_schema_digest",
        lambda: "f" * 64,
    )
    changed = construct_daily_themes(value).theme_set

    assert changed.request_id != original.request_id


def test_empty_sparse_artifact_round_trips(monkeypatch) -> None:
    _provider(monkeypatch, [])

    output = construct_daily_themes(_input(0))
    parsed = parse_daily_theme_set(output.content)

    assert isinstance(parsed, AliasedReaderSubjectThemeSet)
    assert parsed.schema_version == 4
    assert isinstance(parsed.construction, EmptyThemeConstruction)
    assert parsed.themes == ()


def test_version_one_theme_set_preserves_a_fallback_response_id() -> None:
    document = _version_one_document()
    del document["construction"]["attempts"][0]["provider_response"]["id"]

    theme_set = DailyThemeSet.model_validate_json(json.dumps(document))
    construction = theme_set.construction
    assert isinstance(construction, ModelThemeConstruction)
    assert construction.call.response_id == "recorded-response"


def test_version_one_theme_set_accepts_rejected_null_content_before_final_attempt() -> None:
    document = _version_one_document()
    accepted = document["construction"]["attempts"][0]
    rejected = copy.deepcopy(accepted)
    rejected["attempt_id"] = "e" * 64
    rejected["response_id"] = "response-0"
    rejected["status"] = "rejected"
    rejected["error"] = "The provider returned no content"
    rejected["response_content"] = ""
    rejected["response_content_digest"] = hashlib.sha256(b"").hexdigest()
    rejected["provider_response"]["id"] = "response-0"
    rejected["provider_response"]["choices"][0]["message"]["content"] = None
    document["construction"]["attempts"].insert(0, rejected)

    theme_set = DailyThemeSet.model_validate_json(json.dumps(document))
    construction = theme_set.construction
    assert isinstance(construction, ModelThemeConstruction)
    assert tuple(attempt.status for attempt in construction.attempts) == (
        "rejected",
        "accepted",
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("content", "provider response content"),
        ("digest", "content digest"),
        ("status_error", "accepted model attempt cannot have an error"),
        ("no_accepted", "exactly one accepted"),
        ("accepted_not_final", "accepted model attempt must be final"),
        ("multiple_accepted", "exactly one accepted"),
        ("provider_response_id", "provider response ID"),
        ("selected_response_id", "selected response ID"),
    ),
)
def test_version_one_theme_set_rejects_tampered_model_evidence(tamper: str, message: str) -> None:
    document = _version_one_document()
    construction = document["construction"]
    attempts = construction["attempts"]
    attempt = attempts[0]
    if tamper == "content":
        attempt["response_content"] = "tampered"
        attempt["response_content_digest"] = hashlib.sha256(b"tampered").hexdigest()
    elif tamper == "digest":
        attempt["response_content_digest"] = "0" * 64
    elif tamper == "status_error":
        attempt["error"] = "tampered"
    elif tamper == "no_accepted":
        attempt["status"] = "rejected"
        attempt["error"] = "tampered"
    elif tamper == "accepted_not_final":
        rejected = copy.deepcopy(attempt)
        rejected["attempt_id"] = "e" * 64
        rejected["response_id"] = "response-2"
        rejected["status"] = "rejected"
        rejected["error"] = "tampered"
        rejected["provider_response"]["id"] = "response-2"
        attempts.append(rejected)
    elif tamper == "multiple_accepted":
        accepted = copy.deepcopy(attempt)
        accepted["attempt_id"] = "e" * 64
        accepted["response_id"] = "response-2"
        accepted["provider_response"]["id"] = "response-2"
        attempts.append(accepted)
        construction["call"]["response_id"] = "response-2"
    elif tamper == "provider_response_id":
        attempt["provider_response"]["id"] = "response-2"
    elif tamper == "selected_response_id":
        construction["call"]["response_id"] = "response-2"

    with pytest.raises(ValueError, match=message):
        DailyThemeSet.model_validate_json(json.dumps(document))


def _version_one_document() -> dict[str, Any]:
    return copy.deepcopy(json.loads(FIXTURE.read_text())["theme_set"])


def test_version_one_artifact_remains_readable() -> None:
    recorded = json.loads(FIXTURE.read_text())

    output = DailyThemeOutput.model_validate_json(json.dumps(recorded))
    parsed = parse_daily_theme_set(json.dumps(recorded["theme_set"]).encode())

    assert output.theme_set.schema_version == 1
    assert parsed.schema_version == 1


def test_merged_prose_keys_apply_in_any_property_order(monkeypatch) -> None:
    value = _input(4)
    _provider(
        monkeypatch,
        [
            _response({"assignments": _assignments(value, (1, 1, 2, 2))}, "assignment"),
            _response(
                {
                    "themes": {
                        "theme_02": {"title": "A doua temă", "summary": "Rezumat al doilea."},
                        "theme_01": {"title": "Prima temă", "summary": "Rezumat primul."},
                    }
                },
                "prose",
            ),
        ],
    )

    output = construct_daily_themes(value)

    assert tuple(theme.title for theme in output.theme_set.themes) == (
        "Prima temă",
        "A doua temă",
    )
    assert output.theme_set.themes[0].group_ids == ("a" * 64, "b" * 64)
    assert output.theme_set.themes[1].group_ids == ("c" * 64, "d" * 64)


def test_merged_request_identity_changes_with_the_frozen_plan(monkeypatch) -> None:
    value = _input(3)
    _provider(
        monkeypatch,
        [
            _response({"assignments": _assignments(value, (1, 1, 2))}, "assignment-a"),
            _response(
                {"themes": {"theme_01": {"title": "AB", "summary": "AB summary."}}},
                "prose-a",
            ),
            _response({"assignments": _assignments(value, (1, 2, 2))}, "assignment-b"),
            _response(
                {"themes": {"theme_02": {"title": "BC", "summary": "BC summary."}}},
                "prose-b",
            ),
        ],
    )

    first = construct_daily_themes(value).theme_set
    second = construct_daily_themes(value).theme_set
    first_construction = first.construction
    second_construction = second.construction
    assert isinstance(first_construction, SparseThemeConstruction)
    assert isinstance(second_construction, SparseThemeConstruction)
    assert first_construction.merged_prose is not None
    assert second_construction.merged_prose is not None

    assert first.request_id == second.request_id
    assert first_construction.assignment.request_id == second_construction.assignment.request_id
    assert first_construction.merged_prose.request_id != second_construction.merged_prose.request_id


def test_sparse_evidence_rejects_recomputed_digest_for_tampered_context(monkeypatch) -> None:
    document = _mixed_document(monkeypatch)
    assignment = document["construction"]["assignment"]
    context = json.loads(assignment["messages"][1]["content"])
    context["groups"][0]["event_title"] = "Tampered"
    assignment["messages"][1]["content"] = json.dumps(
        context, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assignment["input_digest"] = _message_document_digest(assignment["messages"])

    with pytest.raises(ValueError, match="request identity"):
        parse_daily_theme_set(json.dumps(document).encode())


def test_sparse_evidence_rejects_a_third_attempt(monkeypatch) -> None:
    document = _mixed_document(monkeypatch)
    attempts = document["construction"]["assignment"]["attempts"]
    attempts.extend((copy.deepcopy(attempts[0]), copy.deepcopy(attempts[0])))

    with pytest.raises(ValueError, match="at most 2 items"):
        parse_daily_theme_set(json.dumps(document).encode())


def test_sparse_evidence_accepts_one_exact_correction(monkeypatch) -> None:
    value = _input(2)
    _provider(
        monkeypatch,
        [
            _response({"assignments": {"a" * 64: 1}}, "assignment-invalid"),
            _response({"assignments": _assignments(value, (1, 2))}, "assignment-valid"),
        ],
    )

    output = construct_daily_themes(value)
    parsed = parse_daily_theme_set(output.content)
    construction = parsed.construction
    assert isinstance(construction, SparseThemeConstruction)
    assert tuple(attempt.status for attempt in construction.assignment.attempts) == (
        "rejected",
        "accepted",
    )


def _message_document_digest(messages: list[dict[str, object]]) -> str:
    content = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(content).hexdigest()


def _mixed_document(monkeypatch) -> dict[str, Any]:
    value = _input(3)
    _provider(
        monkeypatch,
        [
            _response({"assignments": _assignments(value, (1, 1, 2))}, "assignment"),
            _response(
                {"themes": {"theme_01": {"title": "Tema unită", "summary": "Rezumat unit."}}},
                "prose",
            ),
        ],
    )
    return json.loads(construct_daily_themes(value).content)


def _replace_attempt_content(attempt: dict[str, Any], content: str) -> None:
    attempt["response_content"] = content
    attempt["response_content_digest"] = hashlib.sha256(content.encode()).hexdigest()
    attempt["provider_response"]["choices"][0]["message"]["content"] = content


def _provider(monkeypatch, responses):
    calls = []
    response_iterator = iter(responses)
    monkeypatch.setattr(themes_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(corrected_structured.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        corrected_structured,
        "openrouter_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: calls.append(kwargs) or next(response_iterator)
                )
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
    return calls


def _assignments(value: DailyThemeInput, labels: tuple[int, ...]) -> dict[str, int]:
    return {f"g{index}": label for index, label in enumerate(labels, start=1)}


def _input(count: int) -> DailyThemeInput:
    groups = tuple(
        NewsGroup(
            id="abcdef0123456789"[index] * 64,
            article_version_ids=(f"{index + 1:x}" * 64,),
        )
        for index in range(count)
    )
    return DailyThemeInput(
        day=DAY,
        cluster_set=_reference("clusters", "4"),
        groups=tuple(
            ThemeGroupInput(
                group=group,
                summary=_reference(f"summary-{index}", f"{index + 5:x}"),
                value=GroupSummary(
                    title_ro=f"Event {index}",
                    summary_ro=(
                        "Dr. Popescu a vorbit. A plecat apoi. A treia propoziție."
                        if index == 0
                        else f"Prima propoziție {index}. A doua propoziție. A treia propoziție."
                    ),
                    key_points_ro=(f"Point {index}",),
                    disagreements_ro=(),
                    cited_article_version_ids=group.article_version_ids,
                ),
            )
            for index, group in enumerate(groups)
        ),
    )


def _reference(name: str, character: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"news:{name}",
        version_id=character[0] * 64,
        content_digest="0" * 64,
        r2_key=f"news/{name}.json",
    )


def _response(payload: dict[str, object] | str, response_id: str):
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
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
