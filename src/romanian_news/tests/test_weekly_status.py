from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from romanian_news.artifacts import ArtifactReference
from romanian_news.weekly_status import (
    AreaAssessment,
    Development,
    WeekInput,
    WeekInputDay,
    WeeklyStatusRead,
    source_items,
    weekly_status_request_id,
)
from romanian_news.weekly_status_generation import GeneratedStatus, build_weekly_status
from tests.reader.test_app import _daily_report

WEEK_START = date(2026, 9, 14)


def _reference(day: date, version: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"news:daily:{day.isoformat()}",
        version_id=version,
        content_digest="f" * 64,
        r2_key=f"news/reports/daily/{day.isoformat()}/report.json",
    )


def _inputs(present_days: int = 7) -> WeekInput:
    return WeekInput(
        week_start=WEEK_START,
        days=tuple(
            WeekInputDay(
                day=WEEK_START + timedelta(days=offset),
                report=_reference(WEEK_START + timedelta(days=offset), f"{offset + 1:064x}")
                if offset < present_days
                else None,
            )
            for offset in range(7)
        ),
    )


def _reports(inputs: WeekInput):
    return {
        slot.report.version_id: _daily_report().model_copy(update={"day": slot.day})
        for slot in inputs.days
        if slot.report is not None
    }


def _draft(inputs: WeekInput, handles: tuple[str, ...]) -> GeneratedStatus:
    return GeneratedStatus(
        developments=(
            Development(
                title="Budget debate",
                what_changed="The budget discussion continued during the week.",
                source_handles=handles[:2],
            ),
        ),
        assessments=(
            AreaAssessment(
                area="overall",
                judgment="The budget debate dominated the selected reports.",
                what_changed="The draft entered public debate.",
                why_it_matters="The budget could affect national spending.",
                source_handles=handles[:2],
                coverage="limited" if inputs.available_days < 7 else "strong",
                coverage_note="Selected report highlights only.",
            ),
            *(
                AreaAssessment(
                    area=area,
                    judgment=None,
                    what_changed=None,
                    why_it_matters=None,
                    source_handles=(),
                    coverage="insufficient",
                    coverage_note="Not enough selected reporting.",
                )
                for area in ("economy", "politics", "society")
            ),
        ),
    )


def test_week_input_tracks_all_seven_dates_and_missing_report() -> None:
    inputs = _inputs(6)
    assert inputs.available_days == 6
    assert inputs.days[-1].report is None

    with pytest.raises(ValidationError, match="seven ordered dates"):
        WeekInput(week_start=WEEK_START, days=tuple(reversed(inputs.days)))


def test_week_identity_changes_when_daily_version_changes() -> None:
    inputs = _inputs()
    changed = inputs.model_copy(
        update={
            "days": (
                inputs.days[0].model_copy(
                    update={
                        "report": _reference(inputs.days[0].day, "a" * 64),
                    }
                ),
                *inputs.days[1:],
            )
        }
    )
    assert weekly_status_request_id(inputs) != weekly_status_request_id(changed)


def test_build_validates_exact_evidence_and_missing_day_threshold() -> None:
    inputs = _inputs(6)
    reports = _reports(inputs)
    sources = source_items(inputs, reports)
    assert {source.report_version_id for source in sources} == {
        slot.report.version_id for slot in inputs.days if slot.report is not None
    }
    output = build_weekly_status(
        inputs,
        reports,
        compose=lambda value, source_items: _draft(
            value, tuple(source.handle for source in source_items)
        ),
        verify=lambda *_args: None,
    )
    assert output.read.days[-1].report is None
    assert output.read.sources == sources
    assert output.read.assessments[0].coverage == "limited"

    with pytest.raises(ValueError, match="At least five"):
        build_weekly_status(_inputs(4), _reports(_inputs(4)))


def test_read_rejects_citation_outside_captured_reports() -> None:
    inputs = _inputs(5)
    sources = source_items(inputs, _reports(inputs))
    draft = _draft(inputs, tuple(source.handle for source in sources))
    bad = draft.model_copy(
        update={
            "developments": (
                draft.developments[0].model_copy(update={"source_handles": ("wrong",)}),
            )
        }
    )
    with pytest.raises(ValidationError, match="outside its exact inputs"):
        WeeklyStatusRead(
            policy=inputs.policy,
            week_start=WEEK_START,
            week_end=WEEK_START + timedelta(days=6),
            days=inputs.days,
            sources=sources,
            developments=bad.developments,
            assessments=bad.assessments,
        )


def test_insufficient_assessment_cannot_claim_impact() -> None:
    with pytest.raises(ValidationError, match="cannot carry a judgment"):
        AreaAssessment(
            area="economy",
            judgment=None,
            what_changed=None,
            why_it_matters="Household spending may fall.",
            source_handles=(),
            coverage="insufficient",
            coverage_note="Not enough selected reporting.",
        )
