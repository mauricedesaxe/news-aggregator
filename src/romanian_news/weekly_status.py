from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from typing import Annotated, Literal, cast

from pydantic import Field, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.artifacts import ArtifactReference
from romanian_news.identity import canonical_json, sha256
from romanian_news.reports import (
    ArchivedDailyReport,
    DailyReport,
    DailyReportDocument,
    DailyReportV2,
)

WEEKLY_STATUS_MODEL = "google/gemini-3.8-flash"
WEEKLY_STATUS_POLICY = f"completed-week-ranked-sources-v1:{WEEKLY_STATUS_MODEL}"
MIN_REPORT_DAYS = 5
Area = Literal["overall", "economy", "politics", "society"]


class WeekInputDay(NewsModel):
    day: date
    report: ArtifactReference | None


class WeekInput(NewsModel):
    week_start: date
    policy: str = WEEKLY_STATUS_POLICY
    days: Annotated[tuple[WeekInputDay, ...], Field(min_length=7, max_length=7)]

    @model_validator(mode="after")
    def check_days(self) -> WeekInput:
        if self.week_start.weekday() != 0:
            raise ValueError("A weekly status must start on Monday")
        if tuple(slot.day for slot in self.days) != tuple(
            self.week_start + timedelta(days=offset) for offset in range(7)
        ):
            raise ValueError("A weekly status needs seven ordered dates")
        for slot in self.days:
            if slot.report is not None and slot.report.artifact_id != (
                f"news:daily:{slot.day.isoformat()}"
            ):
                raise ValueError("Daily reference does not match its date")
        return self

    @property
    def available_days(self) -> int:
        return sum(slot.report is not None for slot in self.days)


class StatusSource(NewsModel):
    handle: str
    day: date
    report_version_id: Sha256
    locator_kind: Literal["theme", "group"]
    locator_id: Sha256
    title: Annotated[str, Field(min_length=1)]
    summary: Annotated[str, Field(min_length=1)]


class Development(NewsModel):
    title: Annotated[str, Field(min_length=1)]
    what_changed: Annotated[str, Field(min_length=1)]
    source_handles: Annotated[tuple[str, ...], Field(min_length=1)]
    uncertainty: str | None = None


class AreaAssessment(NewsModel):
    area: Area
    judgment: str | None
    what_changed: str | None
    why_it_matters: str | None
    source_handles: tuple[str, ...]
    contrary_handles: tuple[str, ...] = ()
    coverage: Literal["strong", "limited", "insufficient"]
    coverage_note: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def check_judgment(self) -> AreaAssessment:
        if self.coverage == "insufficient":
            if any(
                value is not None
                for value in (self.judgment, self.what_changed, self.why_it_matters)
            ):
                raise ValueError("Insufficient coverage cannot carry a judgment")
        elif not (
            self.judgment and self.what_changed and self.why_it_matters and self.source_handles
        ):
            raise ValueError("A judgment needs an explanation and supporting sources")
        return self


class WeeklyStatusRead(NewsModel):
    schema_version: Literal[1] = 1
    policy: str
    week_start: date
    week_end: date
    days: tuple[WeekInputDay, ...]
    sources: tuple[StatusSource, ...]
    developments: tuple[Development, ...]
    assessments: Annotated[tuple[AreaAssessment, ...], Field(min_length=4, max_length=4)]

    @model_validator(mode="after")
    def check_evidence(self) -> WeeklyStatusRead:
        inputs = WeekInput(week_start=self.week_start, days=self.days)
        if self.week_end != self.week_start + timedelta(days=6):
            raise ValueError("Weekly status end must be Sunday")
        handles = {source.handle: source for source in self.sources}
        if len(handles) != len(self.sources):
            raise ValueError("Source handles must be unique")
        versions = {
            slot.day: slot.report.version_id for slot in inputs.days if slot.report is not None
        }
        if any(versions.get(source.day) != source.report_version_id for source in self.sources):
            raise ValueError("A source must belong to the captured daily version")
        if tuple(item.area for item in self.assessments) != (
            "overall",
            "economy",
            "politics",
            "society",
        ):
            raise ValueError("Assessments must cover the four areas in order")
        if inputs.available_days < 7 and any(
            item.coverage == "strong" for item in self.assessments
        ):
            raise ValueError("Missing report days prevent strong coverage")
        for item in (*self.developments, *self.assessments):
            cited = item.source_handles
            contrary = item.contrary_handles if isinstance(item, AreaAssessment) else ()
            if len(cited) != len(set(cited)) or len(contrary) != len(set(contrary)):
                raise ValueError("Source handles cannot repeat")
            if not set((*cited, *contrary)) <= handles.keys():
                raise ValueError("A weekly claim cites a source outside its exact inputs")
        return self


class WeeklyStatusOutput(NewsModel):
    request_id: Sha256
    read: WeeklyStatusRead
    content_digest: Sha256
    content: bytes


def weekly_status_request_id(value: WeekInput) -> Sha256:
    return sha256(
        canonical_json(
            {
                "operation": "news.publish_weekly_status",
                "policy": value.policy,
                "schema": WeeklyStatusRead.model_json_schema(),
                "week_start": value.week_start.isoformat(),
                "days": [
                    (slot.day.isoformat(), slot.report.version_id if slot.report else None)
                    for slot in value.days
                ],
            }
        )
    )


def source_items(
    inputs: WeekInput,
    reports: Mapping[Sha256, DailyReportDocument],
) -> tuple[StatusSource, ...]:
    result: list[StatusSource] = []
    for slot in inputs.days:
        if slot.report is None:
            continue
        report = reports[slot.report.version_id]
        if report.day != slot.day:
            raise ValueError("Daily report date does not match its catalog reference")
        if isinstance(report, DailyReport):
            sections = sorted(
                (section for section in report.sections if section.tier == "main"),
                key=lambda section: section.semantic_rank,
            )[:15]
            extras = sorted(
                (section for section in report.sections if section.tier == "worth_knowing"),
                key=lambda section: section.semantic_rank,
            )[:4]
            selected = (*sections, *extras)
            items = (
                ("theme", section.theme_id, section.title, section.summary) for section in selected
            )
        elif isinstance(report, DailyReportV2):
            items = (
                ("theme", section.theme_id, section.title, section.summary)
                for section in report.sections[:20]
            )
        elif isinstance(report, ArchivedDailyReport):
            items = (
                ("group", section.group_id, section.title_ro, section.summary_ro)
                for section in report.sections[:20]
            )
        else:
            raise TypeError("Unsupported daily report shape")
        for locator_kind, locator_id, title, summary in items:
            result.append(
                StatusSource(
                    handle=f"s{len(result) + 1}",
                    day=slot.day,
                    report_version_id=slot.report.version_id,
                    locator_kind=cast(Literal["theme", "group"], locator_kind),
                    locator_id=locator_id,
                    title=title,
                    summary=summary or title,
                )
            )
    return tuple(result)


def status_output(read: WeeklyStatusRead, request_id: Sha256) -> WeeklyStatusOutput:
    content = canonical_json(read.model_dump(mode="json"))
    return WeeklyStatusOutput(
        request_id=request_id,
        read=read,
        content_digest=sha256(content),
        content=content,
    )
