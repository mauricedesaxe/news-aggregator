import hashlib
import json
from datetime import date
from types import SimpleNamespace

import pytest

from romanian_news import research_triggers as trigger_module
from romanian_news.artifacts import ArtifactReference
from romanian_news.research_triggers import (
    PRODUCTION_RESEARCH_TRIGGER_POLICY as POLICY,
)
from romanian_news.research_triggers import (
    DailyResearchTriggerInput,
    DailyResearchTriggerSet,
    EmptyResearchTriggerConstruction,
    ModelResearchTriggerConstruction,
    ResearchTriggerCorrectionExhausted,
    SubjectResearchFlag,
    SubjectTriggerInput,
    construct_daily_research_triggers,
    parse_daily_research_trigger_set,
    parse_research_trigger_response,
    research_trigger_request_id,
)


def test_construct_daily_research_triggers_rates_every_subject_deterministically(
    monkeypatch,
) -> None:
    value = _input("1" * 64)
    payload = _rated_subjects_payload()
    _provider(monkeypatch, [_response(payload, "trigger-a")])
    first = construct_daily_research_triggers(value)
    _provider(monkeypatch, [_response(payload, "trigger-b")])
    second = construct_daily_research_triggers(value)

    assert parse_daily_research_trigger_set(first.content) == first.trigger_set
    assert first.trigger_set.request_id == second.trigger_set.request_id
    first_construction = first.trigger_set.construction
    second_construction = second.trigger_set.construction
    assert isinstance(first_construction, ModelResearchTriggerConstruction)
    assert isinstance(second_construction, ModelResearchTriggerConstruction)
    assert first_construction.request_id == second_construction.request_id
    assert [attempt.status for attempt in first_construction.attempts] == ["accepted"]
    flags = {flag.theme_id: flag for flag in first.trigger_set.triggers}
    assert set(flags) == {"a" * 64, "b" * 64}
    assert flags["a" * 64].flagged is True
    assert flags["a" * 64].question == "What did European indices do?"
    assert flags["a" * 64].evidence_types == ("financial_data",)
    assert flags["b" * 64].flagged is False
    assert flags["b" * 64].question is None
    assert flags["b" * 64].evidence_types == ()


def test_trigger_response_correction_accepts_the_second_complete_json(monkeypatch) -> None:
    value = _input("1" * 64)
    calls = _provider(
        monkeypatch,
        [
            _response("{", "trigger-invalid"),
            _response(_rated_subjects_payload(), "trigger-valid"),
        ],
    )

    output = construct_daily_research_triggers(value)

    construction = output.trigger_set.construction
    assert isinstance(construction, ModelResearchTriggerConstruction)
    assert [attempt.status for attempt in construction.attempts] == ["rejected", "accepted"]
    assert "Validation error" in calls[1]["messages"][-1]["content"]


def test_trigger_correction_exhaustion_raises_after_repeated_invalid_responses(
    monkeypatch,
) -> None:
    value = _input("1" * 64)
    calls = _provider(
        monkeypatch,
        [_response("{", "trigger-invalid-1"), _response("{", "trigger-invalid-2")],
    )

    with pytest.raises(
        ResearchTriggerCorrectionExhausted,
        match="remained invalid after correction",
    ):
        construct_daily_research_triggers(value)

    assert len(calls) == 2


def _provider(monkeypatch, responses):
    calls = []
    response_iterator = iter(responses)
    monkeypatch.setattr(
        trigger_module,
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
        trigger_module,
        "record_model_attempt",
        lambda response, **_kwargs: SimpleNamespace(
            attempt_id=hashlib.sha256(response.id.encode()).hexdigest(),
            response_id=response.id,
        ),
    )
    return calls


def _rated_subjects_payload() -> dict[str, object]:
    return {
        "subjects": [
            _trigger_subject("subject_01", 0.8),
            _trigger_subject("subject_02", 0.1),
        ]
    }


def _trigger_subject(alias: str, gap_strength: float) -> dict[str, object]:
    flagged = gap_strength >= POLICY.strength_threshold
    return {
        "subject": alias,
        "gap_strength": gap_strength,
        "question": "What did European indices do?" if flagged else None,
        "evidence_types": ["financial_data"] if flagged else [],
    }


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


def _reference(version: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id="news:daily:2026-09-12",
        version_id=version,
        content_digest="c" * 64,
        r2_key="news/reports/daily/2026-09-12/x.json",
    )


def _input(report_version: str) -> DailyResearchTriggerInput:
    return DailyResearchTriggerInput(
        day=date(2026, 9, 12),
        report=_reference(report_version),
        subjects=(
            SubjectTriggerInput(
                alias="subject_01",
                theme_id="a" * 64,
                title="BVB indices decline",
                summary="The BET index fell 1.58%.",
                event_count=1,
                article_count=2,
            ),
            SubjectTriggerInput(
                alias="subject_02",
                theme_id="b" * 64,
                title="Government formation deadlock",
                summary="Parties failed to agree on a coalition.",
                event_count=2,
                article_count=3,
            ),
        ),
    )


def _response_json(*subjects: str) -> str:
    import json

    return json.dumps(
        {
            "subjects": [
                {
                    "subject": alias,
                    "gap_strength": 0.8,
                    "question": "What did European indices do?",
                    "evidence_types": ["financial_data"],
                }
                for alias in subjects
            ]
        }
    )


def test_trigger_request_id_is_deterministic_and_input_sensitive() -> None:
    first = research_trigger_request_id(_input("1" * 64), POLICY)
    second = research_trigger_request_id(_input("1" * 64), POLICY)
    other_report = research_trigger_request_id(_input("2" * 64), POLICY)

    assert first == second
    assert first != other_report
    assert (
        research_trigger_request_id(
            _input("1" * 64), POLICY.model_copy(update={"strength_threshold": 0.4})
        )
        != first
    )


def test_trigger_response_requires_every_subject_exactly_once() -> None:
    value = _input("1" * 64)

    accepted = parse_research_trigger_response(_response_json("subject_01", "subject_02"), value)
    assert tuple(item.subject for item in accepted.subjects) == ("subject_01", "subject_02")

    with pytest.raises(ValueError, match="exactly once"):
        parse_research_trigger_response(_response_json("subject_01"), value)
    with pytest.raises(ValueError, match="exactly once"):
        parse_research_trigger_response(_response_json("subject_01", "subject_01"), value)


def test_flagged_subject_requires_question_and_evidence_types() -> None:
    SubjectResearchFlag(
        theme_id="a" * 64,
        gap_strength=0.8,
        flagged=True,
        question="What did European indices do?",
        evidence_types=("financial_data",),
    )
    SubjectResearchFlag(theme_id="a" * 64, gap_strength=0.1, flagged=False)

    with pytest.raises(ValueError, match="requires a question"):
        SubjectResearchFlag(theme_id="a" * 64, gap_strength=0.8, flagged=True)
    with pytest.raises(ValueError, match="must not carry"):
        SubjectResearchFlag(
            theme_id="a" * 64,
            gap_strength=0.1,
            flagged=False,
            question="Unused",
            evidence_types=("financial_data",),
        )


def test_trigger_set_round_trips_strictly() -> None:
    value = _input("1" * 64)
    trigger_set = DailyResearchTriggerSet(
        day=value.day,
        request_id=research_trigger_request_id(value, POLICY),
        policy=POLICY,
        policy_digest="d" * 64,
        report=value.report,
        construction=EmptyResearchTriggerConstruction(),
        triggers=(),
    )

    parsed = parse_daily_research_trigger_set(trigger_set.model_dump_json().encode())

    assert parsed == trigger_set
