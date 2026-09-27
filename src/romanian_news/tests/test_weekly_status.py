from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from romanian_news import Sha256
from romanian_news.analysis import corrected_structured
from romanian_news.artifacts import ArtifactReference
from romanian_news.reports import (
    ArchivedDailyReport,
    ArchivedDailyReportSection,
    DailyReportDocument,
    DailyReportSectionV2,
    DailyReportV2,
)
from romanian_news.weekly_status import (
    WEEKLY_STATUS_MODEL,
    WEEKLY_STATUS_POLICY,
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
                coverage="limited",
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


def test_generation_and_verification_route_to_the_weekly_status_model(monkeypatch) -> None:
    inputs = _inputs()
    requests, records = _weekly_provider(monkeypatch, inputs)

    output = build_weekly_status(inputs, _reports(inputs))

    assert [request["model"] for request in requests] == [WEEKLY_STATUS_MODEL, WEEKLY_STATUS_MODEL]
    assert [record["operation_key"] for record in records] == [
        "news.compose_weekly_status",
        "news.verify_weekly_status",
    ]
    assert output.read.policy == WEEKLY_STATUS_POLICY
    assert WEEKLY_STATUS_MODEL in output.read.policy


def _weekly_provider(monkeypatch, inputs: WeekInput):
    requests: list[dict[str, object]] = []
    records: list[dict[str, object]] = []

    def create(**kwargs):
        requests.append(kwargs)
        user_content = kwargs["messages"][1]["content"]
        sources = json.loads(user_content)["sources"]
        handles = tuple(source["handle"] for source in sources)
        if kwargs["response_format"]["json_schema"]["name"] == "weekly_status":
            content = _draft(inputs, handles).model_dump_json()
        else:
            content = json.dumps({"supported": True, "problems": []})
        return SimpleNamespace(
            id=f"response-{len(requests)}",
            model=str(kwargs["model"]),
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=(SimpleNamespace(message=SimpleNamespace(content=content)),),
            model_dump=lambda *, mode: {"id": f"response-{len(requests)}"},
        )

    monkeypatch.setattr(
        corrected_structured,
        "openrouter_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    monkeypatch.setattr(
        corrected_structured.time,
        "monotonic",
        lambda: 0.0,
    )

    def record(_response, **kwargs):
        records.append(kwargs)
        return SimpleNamespace(attempt_id=str(len(records)) * 64, response_id="response")

    monkeypatch.setattr(corrected_structured, "record_model_attempt", record)
    return requests, records


def test_source_items_follow_tier_caps_and_ranking() -> None:
    inputs = _inputs(1)
    day = inputs.days[0]
    assert day.report is not None
    base = _daily_report().sections[0]
    sections = tuple(
        base.model_copy(
            update={
                "theme_id": f"{index:064x}",
                "title": f"Main subject {index}",
                "tier": "main",
                "semantic_rank": index,
            }
        )
        for index in range(1, 21)
    ) + tuple(
        base.model_copy(
            update={
                "theme_id": f"{50 + index:064x}",
                "title": f"Other subject {index}",
                "tier": "worth_knowing",
                "semantic_rank": index,
            }
        )
        for index in range(1, 6)
    )
    report = _daily_report().model_copy(update={"day": day.day, "sections": sections})
    reports = {day.report.version_id: report}

    items = source_items(inputs, reports)

    assert [item.title for item in items] == [
        *[f"Main subject {index}" for index in range(1, 16)],
        *[f"Other subject {index}" for index in range(1, 5)],
    ]
    assert [item.handle for item in items] == [f"s{index}" for index in range(1, 20)]


def test_source_items_cap_older_report_shapes_at_twenty() -> None:
    inputs = _inputs(2)
    v2_day, archived_day = inputs.days[0], inputs.days[1]
    assert v2_day.report is not None and archived_day.report is not None
    v2 = DailyReportV2(
        day=v2_day.day,
        accepted_article_count=1,
        theme_count=25,
        group_count=0,
        sections=tuple(
            DailyReportSectionV2(
                theme_id=f"{index:064x}",
                title=f"V2 subject {index}",
                summary=f"Summary {index}",
                events=_daily_report().sections[0].events,
            )
            for index in range(25)
        ),
    )
    archived = ArchivedDailyReport(
        day=archived_day.day,
        accepted_article_count=25,
        group_count=25,
        sections=tuple(
            ArchivedDailyReportSection(
                group_id=f"{index:064x}",
                title_ro=f"Archived subject {index}",
                summary_ro=f"Rezumat {index}",
                key_points_ro=(),
                disagreements_ro=(),
                sentiment_label="mixed",
                sentiment_score=0.0,
                sentiment_rationale_ro="Notă",
                articles=(),
            )
            for index in range(25)
        ),
    )
    reports = {v2_day.report.version_id: v2, archived_day.report.version_id: archived}

    items = source_items(inputs, reports)

    assert [item.title for item in items if item.day == v2_day.day] == [
        f"V2 subject {index}" for index in range(20)
    ]
    assert [item.locator_kind for item in items if item.day == v2_day.day] == ["theme"] * 20
    assert [item.title for item in items if item.day == archived_day.day] == [
        f"Archived subject {index}" for index in range(20)
    ]
    assert [item.locator_kind for item in items if item.day == archived_day.day] == ["group"] * 20


def test_source_items_reject_unknown_report_shapes() -> None:
    inputs = _inputs(1)
    day = inputs.days[0]
    assert day.report is not None
    reports = cast(
        "dict[Sha256, DailyReportDocument]",
        {day.report.version_id: SimpleNamespace(day=day.day)},
    )

    with pytest.raises(TypeError, match="Unsupported daily report shape"):
        source_items(inputs, reports)


def test_read_rejects_invalid_week_bounds_and_assessment_order() -> None:
    inputs = _inputs(7)
    sources = source_items(inputs, _reports(inputs))
    handles = tuple(source.handle for source in sources)
    draft = _draft(inputs, handles)

    def read(**overrides):
        return WeeklyStatusRead(
            policy=inputs.policy,
            week_start=WEEK_START,
            week_end=overrides.pop("week_end", WEEK_START + timedelta(days=6)),
            days=inputs.days,
            sources=sources,
            developments=overrides.pop("developments", draft.developments),
            assessments=overrides.pop("assessments", draft.assessments),
        )

    with pytest.raises(ValidationError, match="must start on Monday"):
        WeekInput(week_start=WEEK_START + timedelta(days=1), days=inputs.days)
    with pytest.raises(ValidationError, match="end must be Sunday"):
        read(week_end=WEEK_START + timedelta(days=5))
    with pytest.raises(ValidationError, match="four areas in order"):
        read(assessments=tuple(reversed(draft.assessments)))
    with pytest.raises(ValidationError, match="cannot repeat"):
        read(
            developments=(
                draft.developments[0].model_copy(
                    update={"source_handles": (*handles[:1], handles[0])}
                ),
            )
        )


def test_missing_days_forbid_strong_coverage() -> None:
    inputs = _inputs(6)
    sources = source_items(inputs, _reports(inputs))
    handles = tuple(source.handle for source in sources)
    draft = _draft(inputs, handles)
    assert draft.assessments[0].coverage == "limited"

    strong = draft.model_copy(
        update={
            "assessments": (
                draft.assessments[0].model_copy(update={"coverage": "strong"}),
                *draft.assessments[1:],
            )
        }
    )

    with pytest.raises(ValidationError, match="strong coverage"):
        WeeklyStatusRead(
            policy=inputs.policy,
            week_start=WEEK_START,
            week_end=WEEK_START + timedelta(days=6),
            days=inputs.days,
            sources=sources,
            developments=strong.developments,
            assessments=strong.assessments,
        )


def test_new_policy_rejects_uncalibrated_strong_coverage_but_old_reads_still_parse() -> None:
    inputs = _inputs()
    sources = source_items(inputs, _reports(inputs))
    draft = _draft(inputs, tuple(source.handle for source in sources))
    strong = draft.assessments[0].model_copy(update={"coverage": "strong"})

    def read(policy: str) -> WeeklyStatusRead:
        return WeeklyStatusRead(
            policy=policy,
            week_start=WEEK_START,
            week_end=WEEK_START + timedelta(days=6),
            days=inputs.days,
            sources=sources,
            developments=draft.developments,
            assessments=(strong, *draft.assessments[1:]),
        )

    with pytest.raises(ValidationError, match="cannot claim strong coverage"):
        read(inputs.policy)

    old_read = read("completed-week-ranked-sources-v1:google/gemini-3.8-flash")
    assert old_read.assessments[0].coverage == "strong"
