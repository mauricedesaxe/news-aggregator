from __future__ import annotations

import hmac
import json
import secrets
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from ipaddress import ip_address
from itertools import count
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal, TypeVar, cast
from urllib.parse import quote
from uuid import UUID, uuid4

from fasthtml.common import (
    FT,
    H1,
    H2,
    H3,
    A,
    Article,
    Aside,
    Beforeware,
    Button,
    Details,
    Div,
    FastHTML,
    Fieldset,
    Form,
    FtResponse,
    Header,
    Hidden,
    Input,
    Label,
    Legend,
    Li,
    Link,
    Main,
    Meta,
    Nav,
    P,
    RedirectResponse,
    Script,
    Section,
    Small,
    Span,
    Strong,
    Style,
    Summary,
    Textarea,
    Time,
    Title,
    Ul,
    fast_app,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from romanian_news import BUCHAREST, Sha256
from romanian_news.catalog.research_triggers import read_daily_research_trigger_set
from romanian_news.catalog_transport import ResearchCatalogError, catalog_query
from romanian_news.config import APP_PASSWORD, COOKIE_SECURE, SESSION_SECRET, TRUST_PROXY_HEADERS
from romanian_news.current_report import (
    CurrentDailyReport,
    CurrentDailyReportHead,
    DailyReportFreshness,
    read_current_daily_report,
    read_daily_report_freshness,
)
from romanian_news.feedback import (
    ArticleFeedbackTarget,
    DailyReportSummary,
    DailyReportVersionNotFound,
    GroupFeedbackTarget,
    NewsFeedbackCommand,
    NewsFeedbackEvent,
    NewsFeedbackTarget,
    ReportFeedbackTarget,
    ThemeFeedbackTarget,
    list_daily_reports,
    read_daily_report_version,
    read_latest_news_feedback,
    resolve_current_daily_report_version,
    submit_news_feedback,
)
from romanian_news.reader.dagster_repair import (
    DagsterRepairError,
    DagsterRepairUnavailable,
    RepairRun,
    read_daily_report_repair,
    request_daily_report_repair,
)
from romanian_news.reports import (
    ArchivedDailyReportSection,
    DailyReport,
    DailyReportDocument,
    DailyReportSection,
    DailyReportSectionV2,
    ReportArticle,
    ReportEvent,
)
from romanian_news.research_triggers import DailyResearchTriggerSet, SubjectResearchFlag
from romanian_news.storage import (
    ResearchObjectIntegrityError,
    ResearchObjectUnavailable,
    check_r2_access,
)
from romanian_news.telemetry import HttpTracingMiddleware, configure_telemetry

SESSION_COOKIE = "romanian_news_session"
SESSION_MAX_AGE = 14 * 24 * 60 * 60
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_CLIENTS = 1024
READINESS_CACHE_SECONDS = 10


class ReaderSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    app_password: Annotated[str, Field(min_length=12)]
    session_secret: Annotated[str, Field(min_length=32)]
    cookie_secure: bool = False
    trust_proxy_headers: bool = False

    @classmethod
    def from_environment(cls) -> ReaderSettings:
        return cls.model_validate(
            {
                "app_password": APP_PASSWORD,
                "session_secret": SESSION_SECRET,
                "cookie_secure": COOKIE_SECURE,
                "trust_proxy_headers": TRUST_PROXY_HEADERS,
            }
        )


@dataclass(frozen=True)
class ReaderDomain:
    list_reports: Callable[[int], tuple[DailyReportSummary, ...]]
    resolve_current_report_version: Callable[[Sha256], Sha256]
    read_report: Callable[[Sha256], DailyReportDocument]
    read_current_report: Callable[[date], CurrentDailyReport | None]
    read_feedback: Callable[[Sha256], tuple[NewsFeedbackEvent, ...]]
    submit_feedback: Callable[[NewsFeedbackCommand], NewsFeedbackEvent]
    read_research_flags: Callable[[Sha256], DailyResearchTriggerSet | None]
    read_freshness: Callable[[date], DailyReportFreshness] = read_daily_report_freshness
    request_repair: Callable[[date], RepairRun] = request_daily_report_repair
    read_repair: Callable[[str], RepairRun] = read_daily_report_repair
    check_readiness: Callable[[], None] = lambda: _check_storage_readiness()


PRODUCTION_DOMAIN = ReaderDomain(
    list_reports=list_daily_reports,
    resolve_current_report_version=resolve_current_daily_report_version,
    read_report=read_daily_report_version,
    read_current_report=read_current_daily_report,
    read_freshness=read_daily_report_freshness,
    request_repair=request_daily_report_repair,
    read_repair=read_daily_report_repair,
    read_feedback=read_latest_news_feedback,
    submit_feedback=submit_news_feedback,
    read_research_flags=read_daily_research_trigger_set,
)


def _check_storage_readiness() -> None:
    catalog_query("SELECT 1 AS ready")
    check_r2_access()


_SECURITY_HEADERS = (
    (
        b"content-security-policy",
        b"default-src 'self'; script-src 'self' 'inline-speculation-rules'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'",
    ),
    (b"cache-control", b"no-store"),
    (b"referrer-policy", b"no-referrer"),
    (b"strict-transport-security", b"max-age=63072000; includeSubDomains"),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
)

_HTMX_CONFIG = '{"responseHandling":[{"code":"204","swap":false},{"code":"[23]..","swap":true},{"code":"[45]..","swap":true,"error":false}]}'

_STYLES = """
:root {
  color-scheme: light dark;
  --ink: #13231a;
  --ink-soft: #304137;
  --muted: #687267;
  --paper: #f3ecd9;
  --card: #f8f2e2;
  --line: #25382c;
  --line-soft: #b9b39f;
  --acid: #d7ff45;
  --accent: #d5522f;
  --accent-dark: #91351f;
  --positive: #285b3d;
  --negative: #b53d28;
  --header: rgba(243,236,217,.96);
  --assessment: #e6dfca;
  --assessment-ink: #304137;
  --annotation: #eee7d4;
  --control: #fffaf0;
  --control-line: #657064;
  --button-ink: #fffaf0;
  --grid-line: rgba(19,35,26,.055);
  --shadow: #13231a;
  --font-display: Georgia, "Times New Roman", Times, serif;
  --font-reading: Arial, Helvetica, ui-sans-serif, system-ui, sans-serif;
  --font-mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  color: var(--ink);
  background: var(--paper);
  font-family: var(--font-reading);
}
* { box-sizing: border-box; }
html { min-width: 280px; }
body {
  margin: 0;
  background-color: var(--paper);
  background-image:
    linear-gradient(var(--grid-line) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid-line) 1px, transparent 1px);
  background-size: 24px 24px;
  color: var(--ink);
  font-family: var(--font-reading);
  line-height: 1.55;
}
::selection { background: var(--acid); color: #13231a; }
a { color: var(--accent-dark); font-weight: 700; text-decoration-thickness: 1px; text-underline-offset: .2em; }
a:hover { color: var(--accent); }
button, input, textarea { font: inherit; }
button { border-radius: 0; }
.site-header { border-bottom: 3px solid var(--line); background: var(--header); }
.site-header > .header-actions, .reader { width: min(1100px, calc(100% - 2.5rem)); margin: 0 auto; }
.site-header > .header-actions { align-items: stretch; display: flex; justify-content: space-between; min-height: 78px; }
.brand {
  align-items: center;
  color: var(--ink);
  display: flex;
  font-family: var(--font-display);
  font-size: clamp(1.5rem, 4vw, 2.35rem);
  font-weight: 900;
  letter-spacing: -.045em;
  line-height: .9;
  padding-right: 1.25rem;
  text-decoration: none;
}
.brand::after { background: var(--acid); content: ""; height: .38em; margin-left: .45rem; width: .38em; }
.header-actions { align-items: center; display: flex; gap: 1rem; }
.logout { margin: 0; }
.logout button {
  background: transparent;
  border: 0;
  border-left: 1px solid var(--line);
  color: var(--ink);
  cursor: pointer;
  font-family: var(--font-mono);
  font-size: .72rem;
  font-weight: 700;
  height: 100%;
  letter-spacing: .08em;
  min-height: 44px;
  padding: .75rem 1rem;
  text-transform: uppercase;
}
.logout button:hover, .logout button:focus-visible { background: var(--acid); color: #13231a; }
.reader { padding: 1.5rem 0 5rem; }
.date-nav {
  align-items: stretch;
  border: 2px solid var(--line);
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  margin-bottom: .75rem;
  min-height: 52px;
}
.date-nav > * { align-items: center; display: flex; min-width: 0; padding: .75rem 1rem; }
.date-nav a { font-family: var(--font-mono); font-size: .72rem; letter-spacing: .04em; text-decoration: none; text-transform: uppercase; }
.date-nav a:hover, .date-nav a:focus-visible { background: var(--acid); color: #13231a; outline: 0; }
.date-nav a:last-child { justify-content: flex-end; text-align: right; }
.date-nav time {
  background: var(--ink);
  color: var(--paper);
  font-family: var(--font-mono);
  font-size: .76rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-align: center;
  text-transform: uppercase;
}
.report-status {
  background: var(--acid);
  border: 2px solid var(--line);
  color: #13231a;
  font-family: var(--font-mono);
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .02em;
  margin-bottom: .75rem;
  padding: .7rem 1rem;
}
.report-status p { margin: 0; }
.report-status a { color: #13231a; font-weight: 900; margin-left: .4rem; }
.report-tools {
  align-items: start;
  border-bottom: 3px solid var(--line);
  border-top: 1px solid var(--line);
  display: flex;
  gap: 1.25rem 2rem;
  justify-content: space-between;
  margin-bottom: 2.5rem;
  padding: .65rem 0;
}
.eyebrow, .section-number, .article-source, .cited {
  font-family: var(--font-mono);
  font-weight: 800;
  letter-spacing: .1em;
  text-transform: uppercase;
}
.eyebrow { color: var(--accent-dark); font-size: .72rem; }
h1, h2, h3, p { margin-top: 0; }
h1, h2, h3 { font-family: var(--font-display); overflow-wrap: anywhere; }
h1 { font-size: clamp(2.7rem, 8vw, 5.8rem); letter-spacing: -.055em; line-height: .92; margin-bottom: 1.25rem; }
h2 { font-size: clamp(2rem, 5vw, 3.65rem); letter-spacing: -.045em; line-height: .98; margin-bottom: 1rem; max-width: 18ch; }
h3 { font-size: 1.2rem; line-height: 1.25; }
.report-meta {
  color: var(--ink-soft);
  display: flex;
  flex-wrap: wrap;
  font-family: var(--font-mono);
  font-size: .72rem;
  gap: .75rem 1.5rem;
  letter-spacing: .035em;
  text-transform: uppercase;
}
.report-feedback { max-width: 420px; width: 100%; }
.report-tools .feedback-control { margin-top: 0; padding-block: 0; }
.event-section { border-top: 1px solid var(--line); margin-top: 1.75rem; padding-top: .5rem; }
.event-disclosure > summary { cursor: pointer; }
.event-disclosure > summary, .articles > summary, .worth-knowing > summary {
  align-items: center;
  display: flex;
  gap: .75rem;
  justify-content: space-between;
  min-height: 44px;
  padding: .65rem .55rem;
  transition: background-color 120ms ease, box-shadow 120ms ease, color 120ms ease;
}
.event-disclosure > summary:hover, .articles > summary:hover, .worth-knowing > summary:hover {
  background: var(--acid);
  box-shadow: inset 5px 0 0 var(--accent);
  color: #13231a;
}
.event-disclosure > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: .25rem; }
.event-disclosure > summary h3 { display: inline; }
.event-article-count { color: var(--muted); font-family: var(--font-mono); font-size: .7rem; margin-left: .75rem; text-transform: uppercase; white-space: nowrap; }
.event-disclosure[open] > summary { margin-bottom: 1.5rem; }
.story-section {
  background: color-mix(in srgb, var(--paper) 92%, transparent);
  border-bottom: 1px solid var(--line);
  border-top: 4px solid var(--line);
  margin: 0;
  padding: clamp(2rem, 6vw, 4.5rem) clamp(.25rem, 3vw, 2rem);
  position: relative;
}
.story-section + .story-section { border-top-width: 1px; }
.section-number { color: var(--accent-dark); font-size: .72rem; margin-bottom: .65rem; }
.section-number::before { background: var(--acid); content: ""; display: inline-block; height: .75em; margin-right: .55rem; width: .75em; }
.consequence { border-left: 4px solid var(--line); color: var(--ink-soft); font-size: .96rem; line-height: 1.6; margin-bottom: 0; max-width: 72ch; padding-left: 1rem; }
.cited { color: var(--accent-dark); font-size: .68rem; }
.worth-knowing { border-bottom: 3px solid var(--line); border-top: 3px solid var(--line); margin-top: 3rem; }
.worth-knowing > summary { cursor: pointer; }
.worth-knowing > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: .25rem; }
.worth-knowing > summary .wk-title { color: var(--ink); font-family: var(--font-mono); font-size: .78rem; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; }
.worth-knowing > summary .event-article-count { margin-left: .75rem; }
.worth-knowing[open] > summary { margin-bottom: 1.5rem; }
.worth-knowing .story-section:first-of-type { margin-top: 0; }
.summary { font-size: clamp(1.05rem, 2vw, 1.18rem); line-height: 1.72; max-width: 72ch; }
.facts { display: grid; gap: 1.5rem; grid-template-columns: repeat(2, minmax(0, 1fr)); margin: 2rem 0; }
.fact { border-top: 3px solid var(--ink); padding-top: .85rem; }
.fact h3 { font-size: 1rem; }
.key-points { font-size: 1.1rem; }
.fact ul { line-height: 1.55; margin-bottom: 0; padding-left: 1.2rem; }
.fact li + li { margin-top: .55rem; }
.assessment { background: var(--assessment); border-left: 6px solid var(--accent); margin: 1.75rem 0; padding: 1rem 1.2rem; }
.assessment h3 { font-family: var(--font-mono); font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; }
.assessment p { color: var(--assessment-ink); line-height: 1.55; margin-bottom: 0; }
.uncertainty { border-bottom: 1px dashed var(--line-soft); border-top: 1px dashed var(--line-soft); padding: .8rem 0; }
.articles { border-top: 1px solid var(--line); margin-top: 2rem; }
.articles > summary { cursor: pointer; }
.articles > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: .25rem; }
.articles > summary { font-family: var(--font-mono); font-size: .72rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }
.article-card { display: grid; gap: 1.25rem; grid-template-columns: minmax(0, 1fr) minmax(250px, .62fr); padding: 1.4rem .55rem; }
.article-card + .article-card { border-top: 1px solid var(--line); }
.article-card h3 { margin-bottom: .5rem; }
.article-source { color: var(--muted); font-size: .68rem; margin-bottom: .35rem; }
.sentiment { color: var(--muted); font-family: var(--font-mono); font-size: .7rem; }
.feedback-control {
  background: var(--annotation);
  border-left: 2px solid var(--line-soft);
  font-family: var(--font-reading);
  margin-top: .75rem;
  padding: .55rem .75rem;
}
.feedback-state { align-items: baseline; display: flex; flex-wrap: wrap; gap: .4rem .75rem; margin-bottom: .55rem; }
.feedback-state strong { font-family: var(--font-mono); font-size: .74rem; text-transform: uppercase; }
.feedback-state small { color: var(--muted); }
.feedback-control details { padding-top: .2rem; }
.feedback-control summary { color: var(--ink-soft); cursor: pointer; font-family: var(--font-mono); font-size: .72rem; font-weight: 750; list-style-position: inside; min-height: 44px; padding: .65rem 0; }
.feedback-form { margin-top: .8rem; }
.feedback-form.htmx-request { opacity: .7; }
.research-flag { background: var(--annotation); border: 2px solid var(--line); box-shadow: 6px 6px 0 var(--acid); color: var(--ink); margin: 1.4rem 6px 1.4rem 0; padding: 1rem; }
.research-head { align-items: baseline; display: flex; gap: .6rem; margin: 0; }
.research-head strong { color: var(--accent-dark); font-family: var(--font-mono); font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; }
.research-question { font-style: italic; margin: .5rem 0 .7rem; }
.research-flag fieldset { border: 1px solid var(--line); margin: .6rem 0; padding: .6rem .8rem; }
.research-flag legend { font-family: var(--font-mono); font-size: .72rem; font-weight: 700; padding: 0 .3rem; }
.research-flag fieldset[name="flag_verdict"] { display: grid; gap: .65rem; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.verdict-option {
  align-items: center;
  background: var(--control);
  border: 2px solid var(--line);
  cursor: pointer;
  display: flex;
  font-family: var(--font-mono);
  font-size: .75rem;
  font-weight: 900;
  gap: .65rem;
  justify-content: center;
  letter-spacing: .08em;
  min-height: 48px;
  padding: .65rem 1rem;
  text-transform: uppercase;
  transition: background-color 120ms ease, box-shadow 120ms ease, color 120ms ease, transform 120ms ease;
}
.verdict-option:hover { background: var(--acid); color: var(--shadow); transform: translate(-2px, -2px); }
.verdict-option:has(input:checked) { background: var(--ink); box-shadow: 4px 4px 0 var(--acid); color: var(--paper); }
.verdict-option:has(input:focus-visible) { outline: 3px solid var(--accent); outline-offset: 3px; }
.verdict-option input, .reason-option input { accent-color: var(--accent); }
.verdict-option input:checked { accent-color: var(--acid); }
.reason-option { align-items: center; display: flex; gap: .5rem; min-height: 44px; padding: .3rem 0; }
.research-flag textarea { margin: .6rem 0; }
.rating-actions { display: grid; gap: .65rem; grid-template-columns: repeat(3, 1fr); }
.rating-actions button, .research-flag button { border: 2px solid var(--ink); cursor: pointer; font-family: var(--font-mono); font-size: .72rem; font-weight: 800; letter-spacing: .04em; min-height: 44px; padding: .65rem 1rem; text-transform: uppercase; }
.rating-actions button[value="positive"] { background: var(--positive); color: var(--button-ink); }
.rating-actions button[value="negative"] { background: var(--negative); color: var(--button-ink); }
.rating-actions button[value=""] { background: var(--control); color: var(--ink); }
.rating-actions button:hover, .rating-actions button:focus-visible, .research-flag button:hover, .research-flag button:focus-visible { box-shadow: 3px 3px 0 var(--shadow); transform: translate(-2px, -2px); }
.note-label { display: block; font-family: var(--font-mono); font-size: .7rem; font-weight: 700; margin: .8rem 0 .35rem; text-transform: uppercase; }
.feedback-form textarea { background: var(--control); color: var(--ink); border: 1px solid var(--control-line); display: block; min-height: 86px; padding: .7rem; resize: vertical; width: 100%; }
.empty { border: 3px solid var(--line); box-shadow: 10px 10px 0 var(--accent); margin: 12vh auto; max-width: 680px; padding: clamp(2rem, 7vw, 4rem); text-align: left; width: calc(100% - 2.5rem); }
.empty h1 { font-size: clamp(2.5rem, 8vw, 5rem); }
.empty .report-status { margin-top: 1.5rem; }
.login-shell { display: grid; min-height: 100vh; padding: 1.25rem; place-items: center; }
.login-card { background: var(--card); border: 3px solid var(--line); box-shadow: 10px 10px 0 var(--acid), 13px 13px 0 var(--line); max-width: 520px; padding: clamp(1.75rem, 6vw, 3.5rem); width: 100%; }
.login-card h1 { font-size: clamp(3rem, 11vw, 5.5rem); }
.login-card label { font-family: var(--font-mono); font-size: .72rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }
.login-card input { background: var(--control); color: var(--ink); border: 2px solid var(--control-line); margin: .5rem 0 1rem; min-height: 48px; padding: .75rem; width: 100%; }
.login-card input:focus { border-color: var(--ink); box-shadow: 4px 4px 0 var(--acid); outline: 0; }
.login-card button { background: var(--ink); border: 2px solid var(--ink); color: var(--button-ink); cursor: pointer; font-family: var(--font-mono); font-size: .75rem; font-weight: 800; letter-spacing: .08em; min-height: 48px; padding: .7rem 1.2rem; text-transform: uppercase; width: 100%; }
.login-card button:hover, .login-card button:focus-visible { background: var(--acid); color: #13231a; }
.error { color: var(--negative); font-weight: 800; }
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #ece5d1;
    --ink-soft: #c7c0ab;
    --muted: #aaa48f;
    --paper: #171714;
    --card: #211f19;
    --line: #d2cbb7;
    --line-soft: #554f42;
    --acid: #b9dd38;
    --accent: #ef7048;
    --accent-dark: #ff9270;
    --positive: #3d7454;
    --negative: #c84f36;
    --header: rgba(23,23,20,.96);
    --assessment: #2b2922;
    --assessment-ink: #d0c8b5;
    --annotation: #24231e;
    --control: #2a2821;
    --control-line: #746d5e;
    --button-ink: #fffaf0;
    --grid-line: rgba(236,229,209,.045);
    --shadow: #000;
  }
}
@media (max-width: 720px) {
  .site-header > .header-actions, .reader { width: min(100% - 1.25rem, 1100px); }
  .site-header > .header-actions { min-height: 64px; }
  .brand { font-size: 1.55rem; }
  .reader { padding-top: .75rem; }
  .date-nav { grid-template-columns: 1fr 1fr; }
  .date-nav time { grid-column: 1 / -1; grid-row: 1; text-align: center; }
  .date-nav a { grid-row: 2; }
  .facts, .article-card { grid-template-columns: 1fr; }
  .report-tools { display: block; }
  .report-feedback { margin-top: .75rem; max-width: none; }
  .rating-actions { grid-template-columns: 1fr; }
  .article-card { gap: .4rem; }
  .story-section { padding-left: .35rem; padding-right: .35rem; }
  .event-disclosure > summary, .articles > summary, .worth-knowing > summary { align-items: flex-start; }
  h1 { font-size: clamp(2.5rem, 12vw, 4.4rem); }
  h2 { font-size: clamp(2rem, 10vw, 3.25rem); }
}
@media (max-width: 420px) {
  .brand::after { display: none; }
  .logout button { padding-inline: .7rem; }
  .date-nav > * { padding-inline: .65rem; }
  .event-disclosure > summary { display: block; }
  .event-article-count { display: block; margin: .3rem 0 0; }
  .worth-knowing > summary .event-article-count { margin-left: 0; }
  .research-flag fieldset[name="flag_verdict"] { grid-template-columns: 1fr; }
  .empty { margin-top: 7vh; width: calc(100% - 1.25rem); }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
}
"""


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                present = {name.lower() for name, _ in headers}
                headers.extend(
                    (name, value) for name, value in _SECURITY_HEADERS if name not in present
                )
            await send(message)

        await self.app(scope, receive, send_with_headers)


_HandlerT = TypeVar("_HandlerT", bound=Callable[..., object])


_HandlerT = TypeVar("_HandlerT", bound=Callable[..., object])
_HandlerDecorator = Callable[[_HandlerT], _HandlerT]


def _route(
    app: FastHTML, method: Literal["get", "post"], path: str
) -> _HandlerDecorator[_HandlerT]:
    """Return the route decorator FastHTML attaches to itself at import time."""
    return cast("_HandlerDecorator[_HandlerT]", getattr(app, method)(path))


def decode_session_cookie(app: FastHTML, cookie: str) -> dict[str, object]:
    """Decode a signed session cookie through the method FastHTML attaches at import time."""
    decoder = cast("Callable[[str], dict[str, object]]", app.decode_session)
    return decoder(cookie)


def create_app(
    settings: ReaderSettings | None = None,
    domain: ReaderDomain = PRODUCTION_DOMAIN,
) -> FastHTML:
    settings = settings or ReaderSettings.from_environment()
    login_failures: dict[str, deque[float]] = {}
    readiness_cache: tuple[float, bool] | None = None
    readiness_lock = Lock()

    configure_telemetry("romanian-news-reader")
    app, _ = fast_app(
        title="Press review",
        before=Beforeware(
            _require_owner,
            skip=[
                r"/healthz",
                r"/livez",
                r"/readyz",
                r"/login",
                r"/favicon.ico",
                r"/htmx\.min\.js",
            ],
        ),
        default_hdrs=False,
        htmx=False,
        pico=False,
        surreal=False,
        canonical=False,
        static_path=str(Path(__file__).parent / "static"),
        secret_key=settings.session_secret,
        session_cookie=SESSION_COOKIE,
        max_age=SESSION_MAX_AGE,
        same_site="lax",
        sess_https_only=settings.cookie_secure,
        htmlkw={"lang": "en"},
        hdrs=(
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            Meta(name="color-scheme", content="light dark"),
            Meta(name="theme-color", content="#f6f3ea", media="(prefers-color-scheme: light)"),
            Meta(name="theme-color", content="#171714", media="(prefers-color-scheme: dark)"),
            Meta(name="htmx-config", content=_HTMX_CONFIG),
            Link(rel="icon", href="data:,"),
            Script(src="/htmx.min.js", defer=True),
            Style(_STYLES),
        ),
    )

    @_route(app, "get", "/healthz")
    def healthz() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @_route(app, "get", "/livez")
    def livez() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @_route(app, "get", "/readyz")
    def readyz() -> PlainTextResponse:
        nonlocal readiness_cache
        with readiness_lock:
            now = time.monotonic()
            if readiness_cache is None or now - readiness_cache[0] >= READINESS_CACHE_SECONDS:
                try:
                    domain.check_readiness()
                except (ResearchCatalogError, ResearchObjectUnavailable):
                    readiness_cache = (now, False)
                else:
                    readiness_cache = (now, True)
            available = readiness_cache[1]
        return PlainTextResponse(
            "ok" if available else "unavailable",
            status_code=200 if available else 503,
        )

    @_route(app, "get", "/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    @_route(app, "get", "/login")
    def login_form(request: Request) -> tuple[Any, ...]:
        return _login_page(_safe_next(request.query_params.get("next", "/")))

    @_route(app, "post", "/login")
    async def login_submit(request: Request) -> Response | FtResponse:
        return await _submit_login(request, settings, login_failures)

    @_route(app, "get", "/")
    def home(request: Request) -> tuple[Any, ...] | FtResponse:
        return _load_home(request, domain)

    @_route(app, "get", "/today")
    def today(request: Request) -> tuple[Any, ...] | FtResponse:
        return _load_today(request, domain)

    @_route(app, "get", "/reports/{report_version_id}")
    def report_page(
        request: Request, report_version_id: str
    ) -> tuple[Any, ...] | Response | FtResponse:
        return _load_report(request, report_version_id, domain)

    @_route(app, "post", "/report-status/check")
    async def report_status_check(request: Request) -> FtResponse:
        return await _check_report_status(request, domain)

    @_route(app, "get", "/report-status")
    async def report_status(request: Request) -> FtResponse:
        return await _poll_report_status(request, domain)

    @_route(app, "post", "/feedback")
    async def feedback_submit(request: Request) -> Response | FtResponse:
        return await _submit_feedback(request, domain)

    @_route(app, "post", "/logout")
    async def logout(request: Request) -> Response | FtResponse:
        return await _submit_logout(request)

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(HttpTracingMiddleware, routes=app.router.routes)
    return app


def _require_owner(request: Request) -> Response | None:
    if ".." in request.url.path.split("/"):
        return Response(status_code=404)
    if request.session.get("authenticated") is True:
        return None
    next_url = request.url.path
    if request.url.query:
        next_url = f"{next_url}?{request.url.query}"
    login_url = f"/login?next={quote(next_url, safe='')}"
    if "hx-request" in request.headers:
        return Response(status_code=401, headers={"HX-Redirect": login_url})
    return RedirectResponse(login_url, status_code=303)


async def _submit_login(
    request: Request,
    settings: ReaderSettings,
    login_failures: dict[str, deque[float]],
) -> Response | FtResponse:
    form = await request.form()
    password = str(form.get("password", ""))
    next_url = _safe_next(str(form.get("next", "/")))
    client_id = _login_client_id(request, settings.trust_proxy_headers)
    now = time.monotonic()
    recent = login_failures.get(client_id, deque())
    while recent and now - recent[0] >= LOGIN_WINDOW_SECONDS:
        recent.popleft()
    if len(recent) >= LOGIN_MAX_FAILURES:
        return FtResponse(
            _login_page(next_url, "Too many attempts. Try again in a few minutes."),
            status_code=429,
        )
    if hmac.compare_digest(password, settings.app_password):
        login_failures.pop(client_id, None)
        request.session.clear()
        request.session.update({"authenticated": True, "csrf_token": secrets.token_urlsafe(32)})
        return RedirectResponse(next_url, status_code=303)
    if client_id not in login_failures and len(login_failures) >= LOGIN_MAX_CLIENTS:
        login_failures.pop(next(iter(login_failures)))
    recent.append(now)
    login_failures[client_id] = recent
    return FtResponse(_login_page(next_url, "The password is incorrect."), status_code=401)


def _login_client_id(request: Request, trust_proxy_headers: bool) -> str:
    if trust_proxy_headers:
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            try:
                return ip_address(real_ip.strip()).compressed
            except ValueError:
                pass
    return request.client.host if request.client else "unknown"


def _load_home(request: Request, domain: ReaderDomain) -> tuple[Any, ...] | FtResponse:
    return _load_current_day(request, domain)


def _load_today(request: Request, domain: ReaderDomain) -> tuple[Any, ...] | FtResponse:
    return _load_current_day(request, domain)


def _load_current_day(request: Request, domain: ReaderDomain) -> tuple[Any, ...] | FtResponse:
    try:
        reports = domain.list_reports(30)
    except (ResearchCatalogError, ValidationError, ValueError):
        return _unavailable_page(request)
    try:
        current = domain.read_current_report(_bucharest_today())
    except (
        ResearchCatalogError,
        ResearchObjectUnavailable,
        ResearchObjectIntegrityError,
        ValidationError,
        ValueError,
    ):
        return _unavailable_page(request, reports)
    if current is None:
        return _not_ready_page(request, reports)
    summaries = (
        _live_summary(current),
        *(report for report in reports if report.report_version_id != current.head.version_id),
    )
    return _report_page(request, current.head.version_id, summaries, domain, live=current)


def _live_summary(live: CurrentDailyReport) -> DailyReportSummary:
    return DailyReportSummary(report_version_id=live.head.version_id, day=live.head.day)


def _bucharest_today() -> date:
    return datetime.now(BUCHAREST).date()


def _load_report(
    request: Request,
    report_version_id: str,
    domain: ReaderDomain,
) -> tuple[Any, ...] | Response | FtResponse:
    try:
        current_version_id = domain.resolve_current_report_version(report_version_id)
    except DailyReportVersionNotFound:
        return FtResponse(_not_found_page(request), status_code=404)
    except ResearchCatalogError:
        return _unavailable_page(request)
    if current_version_id != report_version_id:
        return RedirectResponse(f"/reports/{current_version_id}", status_code=307)
    try:
        reports = domain.list_reports(30)
    except (ResearchCatalogError, ValueError):
        return _unavailable_page(request)
    try:
        return _report_page(request, report_version_id, reports, domain)
    except DailyReportVersionNotFound:
        return FtResponse(_not_found_page(request), status_code=404)
    except (ResearchCatalogError, ResearchObjectUnavailable, ResearchObjectIntegrityError):
        return _unavailable_page(request, reports)


class ReportStatusRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    day: date
    viewed_version: Sha256 | None = None
    retried_after_success: bool = False

    @field_validator("viewed_version", mode="before")
    @classmethod
    def empty_version_is_none(cls, value: object) -> object:
        return None if value == "" else value


class ReportStatusPoll(ReportStatusRequest):
    run_id: Annotated[str, Field(min_length=1)]


def _initial_status_strip(day: date, viewed_version: Sha256 | None, csrf_token: str) -> FT:
    return _status_strip(
        Form(
            Hidden(csrf_token, name="csrf_token"),
            Hidden(day.isoformat(), name="day"),
            Hidden(viewed_version or "", name="viewed_version"),
            P("Checking report inputs…"),
            hx_post="/report-status/check",
            hx_trigger="load",
            hx_target="#report-status",
            hx_swap="outerHTML",
        )
    )


async def _check_report_status(request: Request, domain: ReaderDomain) -> FtResponse:
    form = await request.form()
    if not _valid_csrf(request, str(form.get("csrf_token", ""))):
        return FtResponse(P("The request expired. Reload the page."), status_code=403)
    try:
        value = ReportStatusRequest.model_validate(
            {
                "day": str(form.get("day", "")),
                "viewed_version": str(form.get("viewed_version", "")),
                "retried_after_success": str(form.get("retried_after_success", "false")),
            }
        )
        if value.day != _bucharest_today():
            raise ValueError("Report status day must be today")
    except (ValidationError, ValueError):
        return FtResponse(P("The report status request is invalid."), status_code=400)
    csrf_token = str(request.session["csrf_token"])
    try:
        freshness = await run_in_threadpool(domain.read_freshness, value.day)
    except (ResearchCatalogError, ValidationError, ValueError):
        return FtResponse(
            _retry_status_post(value, csrf_token, "Report status is temporarily unavailable.")
        )
    if freshness.kind == "inputs_not_ready":
        return FtResponse(
            _retry_status_post(value, csrf_token, "Waiting for today's report inputs.")
        )
    if freshness.kind == "fresh":
        return FtResponse(_ready_status(freshness.head, value.viewed_version))
    try:
        repair = await run_in_threadpool(domain.request_repair, value.day)
    except DagsterRepairUnavailable as error:
        return FtResponse(_status_message(str(error)))
    except DagsterRepairError:
        return FtResponse(
            _retry_status_post(value, csrf_token, "Repair service is temporarily unavailable.")
        )
    return FtResponse(
        _repair_status(
            freshness,
            value.viewed_version,
            repair,
            csrf_token,
            value.retried_after_success,
        )
    )


async def _poll_report_status(request: Request, domain: ReaderDomain) -> FtResponse:
    try:
        value = ReportStatusPoll.model_validate(
            {
                "day": request.query_params.get("day", ""),
                "viewed_version": request.query_params.get("viewed_version", ""),
                "run_id": request.query_params.get("run_id", ""),
                "retried_after_success": request.query_params.get("retried_after_success", "false"),
            }
        )
        if value.day != _bucharest_today():
            raise ValueError("Report status day must be today")
    except (ValidationError, ValueError):
        return FtResponse(P("The report status request is invalid."), status_code=400)
    try:
        freshness = await run_in_threadpool(domain.read_freshness, value.day)
    except (ResearchCatalogError, ValidationError, ValueError):
        return FtResponse(_polling_unavailable(value))
    if freshness.kind == "fresh":
        return FtResponse(_ready_status(freshness.head, value.viewed_version))
    try:
        repair = await run_in_threadpool(domain.read_repair, value.run_id)
    except DagsterRepairError:
        return FtResponse(_polling_unavailable(value))
    return FtResponse(
        _repair_status(
            freshness,
            value.viewed_version,
            repair,
            str(request.session["csrf_token"]),
            value.retried_after_success,
        )
    )


def _retry_status_post(value: ReportStatusRequest, csrf_token: str, message: str) -> FT:
    return _status_strip(
        Form(
            Hidden(csrf_token, name="csrf_token"),
            Hidden(value.day.isoformat(), name="day"),
            Hidden(value.viewed_version or "", name="viewed_version"),
            Hidden(str(value.retried_after_success).lower(), name="retried_after_success"),
            P(message),
            hx_post="/report-status/check",
            hx_trigger="load delay:8s",
            hx_target="#report-status",
            hx_swap="outerHTML",
        )
    )


def _repair_status(
    freshness: DailyReportFreshness,
    viewed_version: Sha256 | None,
    repair: RepairRun,
    csrf_token: str,
    retried_after_success: bool,
) -> FT:
    if repair.kind == "failed":
        return _status_message(
            "The updated report could not be built.",
            A("Open the Dagster run", href=repair.run_url),
        )
    if repair.kind == "succeeded":
        if freshness.kind == "fresh":
            return _ready_status(freshness.head, viewed_version)
        if retried_after_success:
            return _status_message(
                "The report update finished, but newer inputs are still available. Reload to try again.",
                A("Open the Dagster run", href=repair.run_url),
            )
        return _retry_repair_post(freshness, viewed_version, csrf_token)
    message, day = _updating_message(freshness)
    return _status_strip(
        P(message, A("Open the Dagster run", href=repair.run_url)),
        hx_get=_poll_url(day, viewed_version, repair.run_id, retried_after_success),
        hx_trigger="load delay:8s",
        hx_swap="outerHTML",
    )


def _retry_repair_post(
    freshness: DailyReportFreshness,
    viewed_version: Sha256 | None,
    csrf_token: str,
) -> FT:
    message, day = _updating_message(freshness)
    return _retry_status_post(
        ReportStatusRequest(
            day=day,
            viewed_version=viewed_version,
            retried_after_success=True,
        ),
        csrf_token,
        message,
    )


def _updating_message(freshness: DailyReportFreshness) -> tuple[str, date]:
    if freshness.kind == "stale":
        timestamp = freshness.head.input_time.astimezone(BUCHAREST).strftime("%H:%M")
        return (
            f"Showing the {timestamp} input snapshot. An updated report is being built.",
            freshness.head.day,
        )
    if freshness.kind == "fresh":
        return "Today's report is up to date.", freshness.head.day
    if freshness.kind == "inputs_not_ready":
        return "Waiting for today's report inputs.", freshness.day
    return "Today's report is being built.", freshness.day


def _ready_status(head: CurrentDailyReportHead, viewed_version: Sha256 | None) -> FT:
    timestamp = head.input_time.astimezone(BUCHAREST).strftime("%H:%M")
    if viewed_version is None:
        return _status_message(
            f"Inputs current through {timestamp}.",
            A("Open today's report", href=f"/reports/{head.version_id}"),
        )
    if head.version_id != viewed_version:
        return _status_message(
            f"Inputs current through {timestamp}.",
            A("Open update", href=f"/reports/{head.version_id}"),
        )
    return _status_message(f"Inputs current through {timestamp}.")


def _polling_unavailable(value: ReportStatusPoll) -> FT:
    return _status_strip(
        P("Repair status is temporarily unavailable."),
        hx_get=_poll_url(
            value.day,
            value.viewed_version,
            value.run_id,
            value.retried_after_success,
        ),
        hx_trigger="load delay:8s",
        hx_swap="outerHTML",
    )


def _poll_url(
    day: date,
    viewed_version: Sha256 | None,
    run_id: str,
    retried_after_success: bool,
) -> str:
    return (
        f"/report-status?day={day.isoformat()}"
        f"&viewed_version={viewed_version or ''}&run_id={run_id}"
        f"&retried_after_success={str(retried_after_success).lower()}"
    )


def _status_message(message: str, link: FT | None = None) -> FT:
    return _status_strip(P(message, link))


def _status_strip(
    *content: FT,
    hx_get: str | None = None,
    hx_trigger: str | None = None,
    hx_swap: str | None = None,
) -> FT:
    return Div(
        *content,
        id="report-status",
        cls="report-status",
        aria_live="polite",
        aria_atomic="true",
        hx_get=hx_get,
        hx_trigger=hx_trigger,
        hx_swap=hx_swap,
    )


async def _submit_feedback(request: Request, domain: ReaderDomain) -> Response | FtResponse:
    form = await request.form()
    if not _valid_csrf(request, str(form.get("csrf_token", ""))):
        return _feedback_error_response(
            request,
            form,
            "The request expired. Reload the page.",
            403,
        )
    try:
        command = _feedback_command(form)
        event = await run_in_threadpool(domain.submit_feedback, command)
    except (ResearchCatalogError, ResearchObjectUnavailable, ResearchObjectIntegrityError):
        return _feedback_error_response(
            request,
            form,
            "The service is unavailable. Feedback was not saved.",
            503,
        )
    except (DailyReportVersionNotFound, ValidationError, ValueError, TypeError):
        return _feedback_error_response(request, form, "Feedback could not be saved.", 400)
    if _is_htmx(request):
        if str(form.get("flag_vote", "")) == "1" and isinstance(
            command.target, ThemeFeedbackTarget
        ):
            return FtResponse(
                _research_flag_control(command.target, None, request.session["csrf_token"], event)
            )
        return FtResponse(
            _feedback_control(
                command.target, request.session["csrf_token"], event, open_details=True
            )
        )
    return RedirectResponse(f"/reports/{event.target.report_version_id}", status_code=303)


def _feedback_error_response(
    request: Request,
    form: Any,
    message: str,
    status_code: int,
) -> FtResponse:
    target = _feedback_target_from_form(form)
    if _is_htmx(request) and target is not None:
        if str(form.get("flag_vote", "")) == "1" and isinstance(target, ThemeFeedbackTarget):
            return FtResponse(
                _research_flag_control(
                    target,
                    None,
                    request.session["csrf_token"],
                    error=message,
                ),
                status_code=status_code,
            )
        return FtResponse(
            _feedback_control(
                target,
                request.session["csrf_token"],
                error=message,
                note=str(form.get("note", "")),
                open_details=True,
            ),
            status_code=status_code,
        )
    return FtResponse(P(message), status_code=status_code)


async def _submit_logout(request: Request) -> Response | FtResponse:
    form = await request.form()
    if not _valid_csrf(request, str(form.get("csrf_token", ""))):
        return FtResponse(P("The request expired. Reload the page."), status_code=403)
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


def _report_page(
    request: Request,
    report_version_id: Sha256,
    reports: tuple[DailyReportSummary, ...],
    domain: ReaderDomain,
    *,
    live: CurrentDailyReport | None = None,
) -> tuple[Any, ...]:
    report = live.report if live is not None else domain.read_report(report_version_id)
    latest_feedback = {
        _target_key(event.target): event for event in domain.read_feedback(report_version_id)
    }
    csrf_token = str(request.session["csrf_token"])
    selected_index = next(
        (
            index
            for index, value in enumerate(reports)
            if value.report_version_id == report_version_id
        ),
        None,
    )
    report_target = ReportFeedbackTarget(report_version_id=report_version_id)
    research_flags = _research_flags(domain, report_version_id)
    next_live = live is None and report.day < _bucharest_today()
    if isinstance(report, DailyReport):
        meta, sections_body = _tiered_report_body(
            report, report_version_id, csrf_token, latest_feedback, research_flags
        )
    else:
        meta = (
            f"{report.theme_count if isinstance(report, DailyReport) else report.group_count}"
            " subjects"
            f" · {report.group_count} events"
            f" · {report.accepted_article_count} articles in this report"
        )
        sections_body = tuple(
            _report_section(
                section,
                position,
                report_version_id,
                csrf_token,
                latest_feedback,
                research_flags,
            )
            for position, section in enumerate(report.sections, start=1)
        )
    report_tools = Div(
        Div(Span(meta), cls="report-meta"),
        Div(
            _feedback_control(
                report_target,
                csrf_token,
                latest_feedback.get(_target_key(report_target)),
            ),
            cls="report-feedback",
        ),
        cls="report-tools",
    )
    reader = Div(
        *_date_navigation(reports, selected_index, report.day, next_live=next_live),
        _initial_status_strip(report.day, report_version_id, csrf_token)
        if live is not None
        else None,
        report_tools,
        *sections_body,
        cls="reader",
    )
    return (
        Title(f"{_format_date(report.day)} | Press review"),
        _site_header(csrf_token),
        Main(reader),
    )


def _research_flags(
    domain: ReaderDomain, report_version_id: Sha256
) -> dict[Sha256, SubjectResearchFlag]:
    """Load flagged subjects for one report; the reader works without them."""
    try:
        trigger_set = domain.read_research_flags(report_version_id)
    except (ResearchCatalogError, ResearchObjectUnavailable, ResearchObjectIntegrityError):
        return {}
    if trigger_set is None:
        return {}
    return {flag.theme_id: flag for flag in trigger_set.triggers if flag.flagged}


def _tiered_report_body(
    report: DailyReport,
    report_version_id: Sha256,
    csrf_token: str,
    latest_feedback: dict[tuple[str, str, str, str], NewsFeedbackEvent],
    research_flags: dict[Sha256, SubjectResearchFlag],
) -> tuple[str, tuple[FT, ...]]:
    main_sections = tuple(section for section in report.sections if section.tier == "main")
    worth_sections = tuple(
        section for section in report.sections if section.tier == "worth_knowing"
    )
    meta = (
        f"{len(main_sections)} main {'subject' if len(main_sections) == 1 else 'subjects'}"
        f" · {len(worth_sections)} in worth knowing"
        f" · {report.group_count} events"
        f" · {report.accepted_article_count} articles in this report"
    )
    position = _subject_positions()
    main_body = tuple(
        _report_section(
            section,
            next(position),
            report_version_id,
            csrf_token,
            latest_feedback,
            research_flags,
        )
        for section in main_sections
    )
    worth_body = tuple(
        _report_section(
            section,
            next(position),
            report_version_id,
            csrf_token,
            latest_feedback,
            research_flags,
        )
        for section in worth_sections
    )
    worth_block = _worth_knowing_section(len(worth_sections), worth_body)
    body = (*main_body, worth_block) if worth_block is not None else main_body
    return meta, body


def _subject_positions() -> Iterator[int]:
    """Number every visible subject continuously across tiers."""
    return iter(count(1))


def _worth_knowing_section(count_value: int, sections: tuple[FT, ...]) -> FT | None:
    if not sections:
        return None
    return Details(
        Summary(
            Span("Worth knowing", cls="wk-title"),
            Span(
                f"{count_value} {'subject' if count_value == 1 else 'subjects'}",
                cls="event-article-count",
            ),
        ),
        *sections,
        cls="worth-knowing",
    )


def _site_header(csrf_token: str) -> FT:
    return Header(
        Div(
            A("Press review", href="/", cls="brand"),
            Form(
                Hidden(csrf_token, name="csrf_token"),
                Button("Log out", type="submit"),
                method="post",
                action="/logout",
                cls="logout",
            ),
            cls="header-actions",
        ),
        cls="site-header",
    )


def _date_navigation(
    reports: tuple[DailyReportSummary, ...],
    selected_index: int | None,
    report_day: date,
    *,
    next_live: bool = False,
) -> tuple[FT, ...]:
    if selected_index is None:
        return (
            Nav(
                Span(),
                Time(_format_date(report_day), datetime=report_day.isoformat()),
                A("Latest report →", href="/"),
                aria_label="Report date navigation",
                cls="date-nav",
            ),
        )
    newer = reports[selected_index - 1] if selected_index > 0 else None
    older = reports[selected_index + 1] if selected_index + 1 < len(reports) else None
    newer_path = (
        f"/reports/{newer.report_version_id}" if newer else ("/today" if next_live else None)
    )
    older_path = f"/reports/{older.report_version_id}" if older else None
    adjacent_paths = tuple(
        path for path in (older_path, newer_path) if path is not None and path != "/today"
    )
    selected = reports[selected_index]
    navigation = Nav(
        A("← Previous day", href=older_path) if older_path else Span(),
        Time(_format_date(selected.day), datetime=selected.day.isoformat()),
        A("Next day →", href=newer_path) if newer_path else Span(),
        aria_label="Report date navigation",
        cls="date-nav",
    )
    if not adjacent_paths:
        return (navigation,)
    speculation_rules = Script(
        json.dumps(
            {
                "prefetch": [
                    {
                        "source": "list",
                        "urls": adjacent_paths,
                        "eagerness": "immediate",
                    }
                ]
            },
            separators=(",", ":"),
        ),
        type="speculationrules",
    )
    return speculation_rules, navigation


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


def _valid_csrf(request: Request, supplied: str) -> bool:
    expected = request.session.get("csrf_token")
    return isinstance(expected, str) and hmac.compare_digest(supplied, expected)


def _is_htmx(request: Request) -> bool:
    return "hx-request" in request.headers


def _login_page(next_url: str, error: str | None = None) -> tuple[Any, ...]:
    return (
        Title("Sign in | Press review"),
        Main(
            Div(
                P("Private access", cls="eyebrow"),
                H1("Press review"),
                P("Enter the password to open the daily report."),
                P(error, cls="error") if error else None,
                Form(
                    Hidden(next_url, name="next"),
                    Label("Password", fr="password"),
                    Input(
                        type="password",
                        id="password",
                        name="password",
                        required=True,
                        autocomplete="current-password",
                    ),
                    Button("Sign in", type="submit"),
                    method="post",
                    action="/login",
                ),
                cls="login-card",
            ),
            cls="login-shell",
        ),
    )


def _not_ready_page(request: Request, reports: tuple[DailyReportSummary, ...]) -> FtResponse:
    day = _bucharest_today()
    prior_href = _yesterday_report_href(reports)
    csrf_token = str(request.session["csrf_token"])
    return FtResponse(
        (
            Title("Today's report is not ready | Press review"),
            _site_header(csrf_token),
            Main(
                Div(
                    H1("Today's report is not ready yet"),
                    P("The report will stay on this page while its inputs become ready."),
                    A("Open yesterday's report →", href=prior_href) if prior_href else None,
                    _initial_status_strip(day, None, csrf_token),
                    cls="empty",
                )
            ),
        )
    )


def _not_found_page(request: Request) -> tuple[Any, ...]:
    return (
        Title("Report unavailable | Press review"),
        _site_header(str(request.session["csrf_token"])),
        Main(
            Div(
                H1("Report unavailable"),
                P("The requested version is not in the recent archive."),
                A("Open the latest report", href="/"),
                cls="empty",
            )
        ),
    )


def _unavailable_page(
    request: Request,
    reports: tuple[DailyReportSummary, ...] = (),
) -> FtResponse:
    prior_href = _yesterday_report_href(reports)
    return FtResponse(
        (
            Title("Report unavailable | Press review"),
            _site_header(str(request.session["csrf_token"])),
            Main(
                Div(
                    H1("The report could not be loaded"),
                    P("The data is temporarily unavailable. Try again in a few minutes."),
                    A("Retry report →", href="/"),
                    A("Open yesterday's report →", href=prior_href) if prior_href else None,
                    cls="empty",
                )
            ),
        ),
        status_code=503,
    )


def _yesterday_report_href(reports: tuple[DailyReportSummary, ...]) -> str | None:
    yesterday = _bucharest_today() - timedelta(days=1)
    prior = next((report for report in reports if report.day == yesterday), None)
    return f"/reports/{prior.report_version_id}" if prior is not None else None


def _safe_next(value: str) -> str:
    if value.startswith("/") and not value.startswith(("//", "/\\")):
        return value
    return "/"


def _format_date(value: date) -> str:
    months = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    return f"{value.day} {months[value.month - 1]} {value.year}"


def _sentiment_label(value: str) -> str:
    return {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "mixed": "mixed",
    }.get(value.lower(), value)


def _format_score(value: float) -> str:
    return f"{value:+.2f}"
