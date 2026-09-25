from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.catalog.artifacts import canonical_json, sha256
from romanian_news.reports import DailyReport, DailyReportSection


class MaterialChangeJudgment(NewsModel):
    disposition: Literal["material", "unchanged", "uncertain"]
    citation_article_version_id: Sha256 | None
    evidence_quote: str | None
    new_fact_or_consequence: str | None


class SubjectDecision(NewsModel):
    theme_id: Sha256
    disposition: Literal["selected", "unchanged"]
    reason: Literal["first_slot", "new_subject", "exact_match", "material_change", "no_new_fact"]
    prior_theme_ids: tuple[Sha256, ...] = ()
    judgment: MaterialChangeJudgment | None = None
    adjudication_request_id: Sha256 | None = None


class SlotSubjectSelection(NewsModel):
    report_version_id: Sha256
    prior_slot_ids: tuple[Sha256, ...]
    decisions: tuple[SubjectDecision, ...]
    selected_sections: tuple[DailyReportSection, ...]

    @model_validator(mode="after")
    def validate_decisions(self) -> SlotSubjectSelection:
        selected = tuple(item.theme_id for item in self.decisions if item.disposition == "selected")
        if selected != tuple(section.theme_id for section in self.selected_sections):
            raise ValueError("Selected sections do not match ordered decisions")
        if len(set(item.theme_id for item in self.decisions)) != len(self.decisions):
            raise ValueError("Subject decisions must be unique")
        return self

    @property
    def digest(self) -> Sha256:
        return sha256(canonical_json(self.model_dump(mode="json")))

    @property
    def selected_subject_ids(self) -> tuple[Sha256, ...]:
        return tuple(section.theme_id for section in self.selected_sections)

    def validate_report(self, report: DailyReport) -> None:
        main = tuple(section for section in report.sections if section.tier == "main")
        if tuple(item.theme_id for item in self.decisions) != tuple(
            section.theme_id for section in main
        ):
            raise ValueError("Selection decisions do not cover main report sections")
        by_id = {section.theme_id: section for section in main}
        if any(by_id.get(section.theme_id) != section for section in self.selected_sections):
            raise ValueError("Selected sections differ from their exact report content")


Adjudicator = Callable[[DailyReportSection, tuple[DailyReportSection, ...]], MaterialChangeJudgment]


def select_slot_subjects(
    report: DailyReport,
    report_version_id: Sha256,
    prior: tuple[tuple[Sha256, SlotSubjectSelection], ...],
    adjudicate: Adjudicator,
) -> SlotSubjectSelection:
    """Select report-order main subjects using earlier claimed or published selections."""
    prior_sections = tuple(
        section for _, selection in prior for section in selection.selected_sections
    )
    decisions: list[SubjectDecision] = []
    selected: list[DailyReportSection] = []
    for section in report.sections:
        if section.tier != "main":
            continue
        if not prior:
            decision = SubjectDecision(
                theme_id=section.theme_id, disposition="selected", reason="first_slot"
            )
        else:
            related = tuple(
                previous for previous in (*prior_sections, *selected) if _related(section, previous)
            )
            if not related:
                decision = SubjectDecision(
                    theme_id=section.theme_id,
                    disposition="selected",
                    reason="new_subject",
                )
            elif any(section == previous for previous in related):
                decision = SubjectDecision(
                    theme_id=section.theme_id,
                    disposition="unchanged",
                    reason="exact_match",
                    prior_theme_ids=tuple(item.theme_id for item in related),
                )
            else:
                judgment = adjudicate(section, related)
                cited = _supported_new_fact(section, related, judgment)
                decision = SubjectDecision(
                    theme_id=section.theme_id,
                    disposition="selected" if cited else "unchanged",
                    reason="material_change" if cited else "no_new_fact",
                    prior_theme_ids=tuple(item.theme_id for item in related),
                    judgment=judgment,
                    adjudication_request_id=material_change_request_id(section, related),
                )
        decisions.append(decision)
        if decision.disposition == "selected":
            selected.append(section)
    return SlotSubjectSelection(
        report_version_id=report_version_id,
        prior_slot_ids=tuple(slot_id for slot_id, _ in prior),
        decisions=tuple(decisions),
        selected_sections=tuple(selected),
    )


def _related(current: DailyReportSection, previous: DailyReportSection) -> bool:
    if current.theme_id == previous.theme_id:
        return True
    if " ".join(current.title.casefold().split()) == " ".join(previous.title.casefold().split()):
        return True
    current_groups = {event.group_id for event in current.events}
    previous_groups = {event.group_id for event in previous.events}
    if current_groups & previous_groups:
        return True
    current_quotes = {item.evidence_quote for item in current.citations}
    previous_quotes = {item.evidence_quote for item in previous.citations}
    if current_quotes & previous_quotes:
        return True
    current_articles = {
        article.canonical_url for event in current.events for article in event.articles
    }
    previous_articles = {
        article.canonical_url for event in previous.events for article in event.articles
    }
    return bool(current_articles & previous_articles)


def material_change_request_id(
    current: DailyReportSection, previous: tuple[DailyReportSection, ...]
) -> Sha256:
    return sha256(
        canonical_json(
            {
                "adjudication_policy": "cited-new-fact-or-consequence-v1",
                "current": current.model_dump(mode="json"),
                "previous": [item.model_dump(mode="json") for item in previous],
            }
        )
    )


def _supported_new_fact(
    current: DailyReportSection,
    previous: tuple[DailyReportSection, ...],
    judgment: MaterialChangeJudgment,
) -> bool:
    if judgment.disposition != "material" or not judgment.new_fact_or_consequence:
        return False
    if not judgment.citation_article_version_id or not judgment.evidence_quote:
        return False
    cited = any(
        item.article_version_id == judgment.citation_article_version_id
        and item.evidence_quote == judgment.evidence_quote
        for item in current.citations
    )
    seen = any(
        item.evidence_quote == judgment.evidence_quote
        for section in previous
        for item in section.citations
    )
    return cited and not seen
