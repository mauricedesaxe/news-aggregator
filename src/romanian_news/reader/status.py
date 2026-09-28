from __future__ import annotations

from datetime import UTC, date, timedelta
from typing import Any

from fasthtml.common import (
    FT,
    H1,
    H2,
    H3,
    A,
    Article,
    Details,
    Div,
    Li,
    Main,
    Nav,
    P,
    Section,
    Style,
    Summary,
    Title,
    Ul,
)

from romanian_news import Sha256
from romanian_news.archive.campaign import ARCHIVE_END, ARCHIVE_START
from romanian_news.catalog.archive_progress import (
    ArchiveDailyReport,
    ArchiveDayEvidence,
    ArchiveDiscoveryMonth,
)
from romanian_news.catalog.weekly_status import WeeklyStatusSummary
from romanian_news.reports import DailyReportDocument, RetrospectiveDailyReport
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
@media (max-width: 700px) {
  .status-grid { grid-template-columns: 1fr; }
  .status { padding: 1rem; }
  .status h1 { max-width: none; }
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
    cited_days = len({sources[handle].day for handle in value.source_handles})
    cited_sources = len(value.source_handles)
    evidence = (
        f"{cited_sources} cited highlight{'s' if cited_sources != 1 else ''} "
        f"across {cited_days} report day{'s' if cited_days != 1 else ''}"
    )
    return Article(
        H2(title),
        H3(value.judgment)
        if value.judgment is not None
        else H3("Not enough reporting to assess this area"),
        P(value.what_changed) if value.what_changed else None,
        P(value.why_it_matters) if value.why_it_matters else None,
        P(evidence) if cited_sources else None,
        P(value.coverage_note),
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
    source_reports: tuple[DailyReportDocument, ...],
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
                P("Saved weekly read" if exact_version else "Weekly status", cls="status-label"),
                H1("What changed in Romania?"),
                P(date_range, cls="status-intro"),
                P(
                    "This saved snapshot keeps this read available if the current week page is updated."
                )
                if exact_version
                else None,
                Nav(
                    A("All weeks", href="/status/archive"),
                    A("Daily reports", href="/reports"),
                    A("Permanent link to this read", href=f"/status/versions/{version_id}")
                    if not exact_version
                    else A(
                        "Current version for this week",
                        href=f"/status/weeks/{read.week_start.isoformat()}",
                    ),
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
                    _retrospective_source_notice(read, source_reports),
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


def _retrospective_source_notice(
    read: WeeklyStatusRead, reports: tuple[DailyReportDocument, ...]
) -> FT | None:
    historical = tuple(report for report in reports if isinstance(report, RetrospectiveDailyReport))
    if not historical:
        return None
    started = min(
        report.retrospective.capture_started_at.astimezone(UTC) for report in historical
    ).strftime("%Y-%m-%d %H:%M UTC")
    ended = max(
        report.retrospective.capture_ended_at.astimezone(UTC) for report in historical
    ).strftime("%Y-%m-%d %H:%M UTC")
    historical_days = {report.day for report in historical}
    versions = {
        slot.day: slot.report.version_id
        for slot in read.days
        if slot.report is not None and slot.day in historical_days
    }
    return Section(
        H2("Historical source capture"),
        P(
            f"This weekly read uses {len(historical)} historical daily "
            f"report{'s' if len(historical) != 1 else ''} whose source pages were captured "
            f"later, from {started} to {ended}."
        ),
        P(
            "Open the exact saved daily reports for outlet and coverage details: ",
            *(
                item
                for index, report in enumerate(historical)
                for item in (
                    A(report.day.isoformat(), href=f"/reports/exact/{versions[report.day]}"),
                    ", " if index < len(historical) - 1 else ".",
                )
            ),
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
                P("Only published weeks appear here. A missing week has no saved read."),
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


def render_archive_progress(
    discovery: tuple[ArchiveDiscoveryMonth, ...],
    archive_reports: tuple[ArchiveDailyReport, ...],
    day_evidence: tuple[ArchiveDayEvidence, ...],
    header: Any,
) -> tuple[Any, ...]:
    return (
        Title("Historical collection | Press review"),
        Style(_STATUS_STYLES),
        header,
        Main(
            Div(_archive_discovery_summary(discovery, archive_reports, day_evidence), cls="status")
        ),
    )


def _archive_discovery_summary(
    months: tuple[ArchiveDiscoveryMonth, ...],
    reports: tuple[ArchiveDailyReport, ...],
    day_evidence: tuple[ArchiveDayEvidence, ...],
) -> Any:
    cards = []
    for outlet_id, label in (("hotnews", "HotNews"), ("digi24", "Digi24")):
        rows = [item for item in months if item.outlet_id == outlet_id]
        if not rows:
            continue
        cards.append(
            Div(
                H3(label),
                P(f"{sum(item.url_entries for item in rows):,} URL entries"),
                P(
                    f"{sum(item.accepted_pages for item in rows):,} pages with verified dates, "
                    f"{sum(item.rejected_pages for item in rows):,} rejected, "
                    f"{sum(item.retryable_pages for item in rows):,} awaiting retry"
                ),
                P(f"{sum(item.captured_articles for item in rows):,} articles captured"),
                Details(
                    Summary("Monthly breakdown"),
                    Ul(
                        *(
                            Li(
                                f"{item.month.strftime('%B %Y')}: "
                                f"{item.url_entries:,} URLs from {item.sitemap_count} sitemap"
                                f"{'s' if item.sitemap_count != 1 else ''}; "
                                f"{item.accepted_pages:,} verified, "
                                f"{item.rejected_pages:,} rejected, "
                                f"{item.retryable_pages:,} awaiting retry; "
                                f"{item.captured_articles:,} articles captured"
                            )
                            for item in rows
                        )
                    ),
                ),
                cls="status-card",
            )
        )
    return Section(
        H2("Historical collection"),
        P(
            "Publisher sitemap URLs found for earlier dates. Verified pages have a readable "
            "original publication date. Captured articles have stored text. Daily reports "
            "and weekly reads appear only when published."
        ),
        P(
            "A zero for verified pages or captured articles means none have been recorded "
            "for that month; URL entries may still exist."
        ),
        Div(*cards, cls="status-grid"),
        H3("Daily coverage by publication date"),
        P(
            f"{len(reports):,} daily "
            f"{'report' if len(reports) == 1 else 'reports'} published. "
            "This includes regular reports and reports reconstructed from archived pages."
        ),
        P(
            "Pending means a report has not been published. Insufficient source means captured "
            "articles exist from fewer than two outlets. Counts cover only pages with a "
            "verified publication date. An empty day does not prove there was no news."
        ),
        _archive_day_list(reports, day_evidence),
        cls="status-coverage",
    )


def _archive_day_list(
    reports: tuple[ArchiveDailyReport, ...],
    evidence: tuple[ArchiveDayEvidence, ...],
) -> Any:
    published = {report.day: report for report in reports}
    observed = {item.day: item for item in evidence}
    months: dict[date, list[Any]] = {}
    month_states: dict[date, dict[str, int]] = {}
    day = ARCHIVE_START
    while day <= ARCHIVE_END:
        item = observed.get(day)
        report = published.get(day)
        if report is not None:
            state = "Published"
        elif item is not None and item.captured_articles and len(item.captured_outlets) < 2:
            state = "Insufficient source"
        else:
            state = "Pending"
        verified = item.verified_pages if item else 0
        captured = item.captured_articles if item else 0
        outlets = len(item.captured_outlets) if item else 0
        label = (
            f"{day:%-d %B %Y}: {state}; {verified} verified, "
            f"{captured} captured from {outlets} outlet{'s' if outlets != 1 else ''}"
        )
        month = day.replace(day=1)
        months.setdefault(month, []).append(
            Li(A(label, href=f"/reports/{report.version_id}") if report else label)
        )
        counts = month_states.setdefault(
            month, {"Published": 0, "Insufficient source": 0, "Pending": 0}
        )
        counts[state] += 1
        day += timedelta(days=1)
    return Div(
        *(
            Details(
                Summary(
                    f"{month:%B %Y}: {month_states[month]['Published']} published, "
                    f"{month_states[month]['Insufficient source']} insufficient source, "
                    f"{month_states[month]['Pending']} pending"
                ),
                Ul(*days, cls="status-history"),
                open=month == ARCHIVE_START.replace(day=1),
            )
            for month, days in months.items()
        )
    )
