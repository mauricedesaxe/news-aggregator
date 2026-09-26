from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from fasthtml.common import (
    FT,
    H2,
    H3,
    A,
    Article,
    Aside,
    Button,
    Details,
    Div,
    Fieldset,
    Form,
    Hidden,
    Input,
    Label,
    Legend,
    Li,
    P,
    Section,
    Small,
    Span,
    Strong,
    Summary,
    Textarea,
    Ul,
)
from pydantic import ValidationError

from romanian_news import Sha256
from romanian_news.feedback import (
    ArticleFeedbackTarget,
    GroupFeedbackTarget,
    NewsFeedbackCommand,
    NewsFeedbackEvent,
    NewsFeedbackTarget,
    ReportFeedbackTarget,
    ThemeFeedbackTarget,
)
from romanian_news.reports import (
    ArchivedDailyReportSection,
    DailyReportSection,
    DailyReportSectionV2,
    ReportArticle,
    ReportEvent,
)
from romanian_news.research_triggers import SubjectResearchFlag


def _report_section(
    section: DailyReportSection | DailyReportSectionV2 | ArchivedDailyReportSection,
    position: int,
    report_version_id: Sha256,
    csrf_token: str,
    latest_feedback: dict[tuple[str, str, str, str], NewsFeedbackEvent],
    research_flags: dict[Sha256, SubjectResearchFlag],
) -> FT:
    if isinstance(section, ArchivedDailyReportSection):
        return _archived_report_section(
            section, position, report_version_id, csrf_token, latest_feedback
        )
    target = ThemeFeedbackTarget(
        report_version_id=report_version_id,
        theme_id=section.theme_id,
    )
    citations = (
        {item.article_version_id: item.evidence_quote for item in section.citations}
        if isinstance(section, DailyReportSection)
        else {}
    )
    return Section(
        P(f"Subject {position:02d}", cls="section-number"),
        H2(section.title),
        P(section.summary, cls="summary"),
        P(section.consequence_rationale, cls="consequence")
        if isinstance(section, DailyReportSection)
        else None,
        _research_flag_control(target, research_flags.get(section.theme_id), csrf_token)
        if section.theme_id in research_flags
        else None,
        _feedback_control(
            target,
            csrf_token,
            latest_feedback.get(_target_key(target)),
        ),
        *(
            _report_event(
                event,
                event_position,
                report_version_id,
                csrf_token,
                latest_feedback,
                citations=citations,
            )
            for event_position, event in enumerate(section.events, start=1)
        ),
        id=f"source-{section.theme_id}",
        cls="story-section",
    )


def _archived_report_section(
    section: ArchivedDailyReportSection,
    position: int,
    report_version_id: Sha256,
    csrf_token: str,
    latest_feedback: dict[tuple[str, str, str, str], NewsFeedbackEvent],
) -> FT:
    return Section(
        P(f"Subject {position:02d}", cls="section-number"),
        _report_event(
            ReportEvent(**section.model_dump()),
            1,
            report_version_id,
            csrf_token,
            latest_feedback,
            show_event_label=False,
        ),
        id=f"source-{section.group_id}",
        cls="story-section",
    )


def _report_event(
    event: ReportEvent,
    position: int,
    report_version_id: Sha256,
    csrf_token: str,
    latest_feedback: dict[tuple[str, str, str, str], NewsFeedbackEvent],
    *,
    show_event_label: bool = True,
    citations: dict[Sha256, str] | None = None,
) -> FT:
    citations = citations or {}
    article_count_label = _format_article_count(len(event.articles))
    target = GroupFeedbackTarget(
        report_version_id=report_version_id,
        group_id=event.group_id,
    )
    distinct_outlets = {article.outlet_id for article in event.articles}
    return Details(
        Summary(
            H3(event.title_ro),
            Span(article_count_label, cls="event-article-count"),
        ),
        P(f"Event {position}", cls="eyebrow") if show_event_label else None,
        P(event.summary_ro, cls="summary"),
        Div(
            _fact_list("Key points", event.key_points_ro, cls="fact key-points"),
            _fact_list("Differences between sources", event.disagreements_ro)
            if len(distinct_outlets) >= 2
            else None,
            cls="facts",
        ),
        Aside(
            H3("Tone assessment"),
            P(
                f"{_sentiment_label(event.sentiment_label)} "
                f"({_format_score(event.sentiment_score)}). "
                f"{event.sentiment_rationale_ro}"
            ),
            cls="assessment",
        ),
        P(Strong("Uncertainty: "), event.uncertainty_ro, cls="uncertainty")
        if event.uncertainty_ro
        else None,
        _feedback_control(
            target,
            csrf_token,
            latest_feedback.get(_target_key(target)),
        ),
        Details(
            Summary(f"Reviewed articles ({article_count_label})"),
            *(
                _article_card(
                    article,
                    report_version_id,
                    event.group_id,
                    csrf_token,
                    latest_feedback,
                    cited_quote=citations.get(article.article_version_id),
                )
                for article in event.articles
            ),
            cls="articles",
        ),
        cls="event-disclosure event-section" if show_event_label else "event-disclosure",
    )


def _format_article_count(count: int) -> str:
    return f"{count} {'article' if count == 1 else 'articles'}"


def _fact_list(title: str, values: tuple[str, ...], *, cls: str = "fact") -> FT | None:
    if not values:
        return None
    return Div(H3(title), Ul(*(Li(value) for value in values)), cls=cls)


def _article_card(
    article: ReportArticle,
    report_version_id: Sha256,
    group_id: Sha256,
    csrf_token: str,
    latest_feedback: dict[tuple[str, str, str, str], NewsFeedbackEvent],
    *,
    cited_quote: str | None = None,
) -> FT:
    target = ArticleFeedbackTarget(
        report_version_id=report_version_id,
        group_id=group_id,
        article_version_id=article.article_version_id,
    )
    return Article(
        Div(
            P(article.outlet_id, cls="article-source"),
            P(
                "Cited evidence",
                title=cited_quote,
                cls="cited",
            )
            if cited_quote is not None
            else None,
            H3(A(article.title, href=article.canonical_url, target="_blank", rel="noreferrer")),
            P(
                f"Tone: {_sentiment_label(article.sentiment_label)} "
                f"({_format_score(article.sentiment_score)})",
                cls="sentiment",
            ),
        ),
        _feedback_control(
            target,
            csrf_token,
            latest_feedback.get(_target_key(target)),
        ),
        cls="article-card",
    )


def _research_flag_control(
    target: ThemeFeedbackTarget,
    flag: SubjectResearchFlag | None,
    csrf_token: str,
    event: NewsFeedbackEvent | None = None,
    *,
    error: str | None = None,
) -> FT:
    control_id = f"research-flag-{target.theme_id[:12]}"
    state = ()
    actions = ()
    if event is not None:
        saved = _research_flag_saved_summary(event)
        state = (
            Div(
                Strong("Flag feedback saved"),
                Small(saved) if saved else None,
                cls="feedback-state",
            ),
        )
    elif flag is not None:
        actions = (
            P(flag.question, cls="research-question"),
            Form(
                Hidden(str(uuid4()), name="feedback_id"),
                Hidden("theme", name="target_kind"),
                Hidden(target.report_version_id, name="report_version_id"),
                Hidden(target.theme_id, name="theme_id"),
                Hidden(csrf_token, name="csrf_token"),
                Hidden("1", name="flag_vote"),
                Fieldset(
                    Legend("Was the research flag right?"),
                    _verdict_radio("right", "Right"),
                    _verdict_radio("wrong", "Wrong"),
                    name="flag_verdict",
                ),
                Textarea(
                    "",
                    name="flag_note",
                    placeholder="Why? What was right or wrong about it?",
                    rows=3,
                ),
                Button("Save feedback", type="submit"),
                method="post",
                action="/feedback",
                hx_post="/feedback",
                hx_target=f"#{control_id}",
                hx_swap="outerHTML",
                hx_disabled_elt=f"#{control_id} button",
                cls="feedback-form",
            ),
        )
    return Aside(
        P(
            Strong("Research suggested"),
            Small(f"gap strength {flag.gap_strength:.2f}") if flag else None,
            cls="research-head",
        ),
        *state,
        P(error, cls="error") if error else None,
        *actions,
        id=control_id,
        cls="research-flag",
    )


def _verdict_radio(value: str, label: str) -> FT:
    return Label(
        Input(type="radio", name="flag_verdict", value=value, required=True),
        label,
        cls="verdict-option",
    )


def _reason_checkbox(value: str, label: str) -> FT:
    return Label(
        Input(type="checkbox", name="flag_reasons", value=value),
        label,
        cls="reason-option",
    )


def _research_flag_saved_summary(event: NewsFeedbackEvent) -> str | None:
    if event.rating not in ("positive", "negative"):
        return event.note
    parts = ["flag was right" if event.rating == "positive" else "flag was wrong"]
    if event.note:
        parts.append(event.note)
    return " · ".join(parts)


def _feedback_control(
    target: NewsFeedbackTarget,
    csrf_token: str,
    event: NewsFeedbackEvent | None = None,
    *,
    error: str | None = None,
    note: str | None = None,
    open_details: bool = False,
) -> FT:
    control_id = _feedback_control_id(target)
    note_id = f"{control_id}-note"
    state = ()
    if event is not None:
        rating = {"positive": "Positive", "negative": "Negative", None: "Note"}[event.rating]
        state = (
            Div(
                Strong(f"Feedback saved: {rating}"),
                Small(event.note) if event.note else None,
                cls="feedback-state",
            ),
        )
    value = event.note if event and event.note else (note or "")
    return Div(
        *state,
        Details(
            Summary(_feedback_label(target)),
            P(error, cls="error") if error else None,
            Form(
                Hidden(str(uuid4()), name="feedback_id"),
                Hidden(target.kind, name="target_kind"),
                Hidden(target.report_version_id, name="report_version_id"),
                Hidden(getattr(target, "theme_id", ""), name="theme_id"),
                Hidden(getattr(target, "group_id", ""), name="group_id"),
                Hidden(getattr(target, "article_version_id", ""), name="article_version_id"),
                Hidden(csrf_token, name="csrf_token"),
                Label("Optional note", fr=note_id, cls="note-label"),
                Textarea(
                    value,
                    id=note_id,
                    name="note",
                    maxlength="2000",
                    placeholder="What was useful, or what should change?",
                ),
                Div(
                    Button("Positive", type="submit", name="rating", value="positive"),
                    Button("Negative", type="submit", name="rating", value="negative"),
                    Button("Save note", type="submit", name="rating", value=""),
                    cls="rating-actions",
                ),
                method="post",
                action="/feedback",
                hx_post="/feedback",
                hx_target=f"#{control_id}",
                hx_swap="outerHTML",
                hx_disabled_elt=f"#{control_id} button",
                cls="feedback-form",
            ),
            open=True if open_details else None,
        ),
        id=control_id,
        cls="feedback-control",
    )


def _feedback_label(target: NewsFeedbackTarget) -> str:
    return {
        "report": "Feedback on this report",
        "theme": "Feedback on this subject",
        "group": "Feedback on this event",
        "article": "Feedback on this article",
    }[target.kind]


def _feedback_command(form: Any) -> NewsFeedbackCommand:
    target: dict[str, object] = {
        "kind": str(form.get("target_kind", "")),
        "report_version_id": str(form.get("report_version_id", "")),
    }
    theme_id = str(form.get("theme_id", ""))
    group_id = str(form.get("group_id", ""))
    article_version_id = str(form.get("article_version_id", ""))
    if theme_id:
        target["theme_id"] = theme_id
    if group_id:
        target["group_id"] = group_id
    if article_version_id:
        target["article_version_id"] = article_version_id
    rating = str(form.get("rating", "")).strip() or None
    flag_verdict = str(form.get("flag_verdict", "")).strip()
    if flag_verdict == "right":
        rating = "positive"
    elif flag_verdict == "wrong":
        rating = "negative"
    note = str(form.get("note", "")) or str(form.get("flag_note", ""))
    return NewsFeedbackCommand.model_validate(
        {
            "feedback_id": UUID(str(form.get("feedback_id", ""))),
            "target": target,
            "rating": rating,
            "note": note,
            "actor": "owner",
        }
    )


def _feedback_target_from_form(form: Any) -> NewsFeedbackTarget | None:
    kinds = {
        "report": ReportFeedbackTarget,
        "theme": ThemeFeedbackTarget,
        "group": GroupFeedbackTarget,
        "article": ArticleFeedbackTarget,
    }
    fields: dict[str, object] = {"report_version_id": str(form.get("report_version_id", ""))}
    theme_id = str(form.get("theme_id", ""))
    group_id = str(form.get("group_id", ""))
    article_version_id = str(form.get("article_version_id", ""))
    if theme_id:
        fields["theme_id"] = theme_id
    if group_id:
        fields["group_id"] = group_id
    if article_version_id:
        fields["article_version_id"] = article_version_id
    try:
        return kinds[str(form.get("target_kind", ""))].model_validate(fields)
    except (KeyError, ValidationError):
        return None


def _target_key(target: NewsFeedbackTarget) -> tuple[str, str, str, str]:
    return (
        target.kind,
        str(getattr(target, "theme_id", "")),
        str(getattr(target, "group_id", "")),
        str(getattr(target, "article_version_id", "")),
    )


def _feedback_control_id(target: NewsFeedbackTarget) -> str:
    suffix = "-".join(part for part in _target_key(target)[1:] if part)
    return f"feedback-{target.kind}{f'-{suffix}' if suffix else ''}"


def _sentiment_label(value: str) -> str:
    return {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "mixed": "mixed",
    }.get(value.lower(), value)


def _format_score(value: float) -> str:
    return f"{value:+.2f}"
