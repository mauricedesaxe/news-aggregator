from __future__ import annotations

from typing import Any

from fasthtml.common import (
    FT,
    H1,
    H2,
    H3,
    A,
    Article,
    Div,
    Li,
    Main,
    Nav,
    P,
    Section,
    Span,
    Style,
    Title,
    Ul,
)

from romanian_news import Sha256
from romanian_news.catalog.weekly_status import WeeklyStatusSummary
from romanian_news.weekly_status import AreaAssessment, StatusSource, WeeklyStatusRead

_STATUS_STYLES = """
.status { max-width: 62rem; margin: 0 auto; padding: clamp(1rem, 3vw, 2.5rem); }
.status h1 { font-size: clamp(2.4rem, 8vw, 5rem); line-height: 1.02; max-width: 13ch; }
.status-intro { max-width: 44rem; font-size: 1.1rem; line-height: 1.55; }
.status-coverage { border: 1px solid var(--line); border-radius: .5rem; padding: 1rem; margin: 1.5rem 0; }
.status-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
.status-card { border: 1px solid var(--line); border-radius: .5rem; padding: 1.25rem; min-width: 0; }
.status-card h2 { margin-top: .3rem; }
.status-card p { line-height: 1.55; }
.status-card ul, .status-developments ul { padding-left: 1.2rem; }
.status-card li, .status-developments li { margin: .35rem 0; }
.status-label { text-transform: uppercase; letter-spacing: .09em; font-size: .75rem; opacity: .7; }
.status-developments { margin-top: 2.5rem; }
.status-developments article { border-top: 1px solid var(--line); padding: 1rem 0; }
.status-nav { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0 2rem; }
.status-nav a, .status a { text-underline-offset: .2em; }
.status-history { display: grid; gap: .8rem; list-style: none; padding: 0; }
.status-history li { border-bottom: 1px solid var(--line); padding-bottom: .8rem; }
.site-links { display: flex; gap: .8rem; align-items: center; }
@media (max-width: 700px) {
  .status-grid { grid-template-columns: 1fr; }
  .status { padding: 1rem; }
  .status h1 { max-width: none; }
  .site-links { font-size: .85rem; }
}
"""


def _source_links(handles: tuple[str, ...], sources: dict[str, StatusSource]) -> FT | None:
    if not handles:
        return None
    return Ul(
        *(
            Li(
                A(
                    f"{sources[handle].day.strftime('%-d %B')} · {sources[handle].title}",
                    href=(
                        f"/reports/exact/{sources[handle].report_version_id}"
                        f"#source-{sources[handle].locator_id}"
                    ),
                )
            )
            for handle in handles
        )
    )


def _assessment(value: AreaAssessment, sources: dict[str, StatusSource]) -> FT:
    title = "Romania overall" if value.area == "overall" else value.area.capitalize()
    return Article(
        Span(value.coverage, cls="status-label"),
        H2(title),
        H3(value.judgment)
        if value.judgment is not None
        else H3("Not enough reporting to assess this area"),
        P(value.what_changed) if value.what_changed else None,
        P(value.why_it_matters) if value.why_it_matters else None,
        P(value.coverage_note, cls="status-label"),
        _source_links(value.source_handles, sources),
        P("Other evidence:", cls="status-label") if value.contrary_handles else None,
        _source_links(value.contrary_handles, sources),
        cls="status-card",
    )


def render_status(
    read: WeeklyStatusRead,
    version_id: Sha256,
    header: Any,
    *,
    exact_version: bool = False,
) -> tuple[Any, ...]:
    sources = {source.handle: source for source in read.sources}
    missing = tuple(slot.day for slot in read.days if slot.report is None)
    covered = 7 - len(missing)
    date_range = (
        f"{read.week_start.strftime('%-d %B')} to " f"{read.week_end.strftime('%-d %B %Y')}"
    )
    return (
        Title(f"Romania status, {date_range} | Press review"),
        Style(_STATUS_STYLES),
        header,
        Main(
            Div(
                P("Weekly status", cls="status-label"),
                H1("What changed in Romania?"),
                P(date_range, cls="status-intro"),
                Nav(
                    A("All weeks", href="/status/archive"),
                    A("Daily reports", href="/reports"),
                    A("Saved version", href=f"/status/versions/{version_id}")
                    if not exact_version
                    else None,
                    aria_label="Status navigation",
                    cls="status-nav",
                ),
                Div(
                    P(
                        f"Based on reports from {covered} of 7 days."
                        + (
                            " Missing: "
                            + ", ".join(day.strftime("%-d %B") for day in missing)
                            + "."
                            if missing
                            else ""
                        )
                    ),
                    P(
                        "This read uses selected daily highlights. Open a source to check the exact saved report."
                    ),
                    cls="status-coverage",
                ),
                Section(
                    *(_assessment(item, sources) for item in read.assessments), cls="status-grid"
                ),
                Section(
                    H2("Developments across the week"),
                    *(
                        Article(
                            H3(item.title),
                            P(item.what_changed),
                            P(item.uncertainty) if item.uncertainty else None,
                            _source_links(item.source_handles, sources),
                        )
                        for item in read.developments
                    ),
                    cls="status-developments",
                )
                if read.developments
                else None,
                cls="status",
            )
        ),
    )


def render_status_archive(
    summaries: tuple[WeeklyStatusSummary, ...],
    page: int,
    has_more: bool,
    header: Any,
) -> tuple[Any, ...]:
    return (
        Title("Weekly status archive | Press review"),
        Style(_STATUS_STYLES),
        header,
        Main(
            Div(
                P("Archive", cls="status-label"),
                H1("Weekly status"),
                Ul(
                    *(
                        Li(
                            A(
                                f"Week of {item.week_start.strftime('%-d %B %Y')}",
                                href=f"/status/weeks/{item.week_start.isoformat()}",
                            )
                        )
                        for item in summaries
                    ),
                    cls="status-history",
                ),
                Nav(
                    A("← Newer", href=f"/status/archive?page={page - 1}") if page > 1 else None,
                    A("Older →", href=f"/status/archive?page={page + 1}") if has_more else None,
                    aria_label="Status archive pages",
                    cls="status-nav",
                ),
                cls="status",
            )
        ),
    )
