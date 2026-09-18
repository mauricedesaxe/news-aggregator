from datetime import date

import pytest

from romanian_news import research_triggers as trigger_module
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.research_triggers import (
    PRODUCTION_RESEARCH_TRIGGER_POLICY as POLICY,
)
from romanian_news.research_triggers import (
    DailyResearchTriggerInput,
    DailyResearchTriggerSet,
    EmptyResearchTriggerConstruction,
    SubjectResearchFlag,
    SubjectTriggerInput,
    parse_daily_research_trigger_set,
    parse_research_trigger_response,
    research_trigger_request_id,
)


def test_trigger_prompt_requires_english_research_questions() -> None:
    assert "research question in English" in trigger_module.TRIGGER_PROMPT


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
