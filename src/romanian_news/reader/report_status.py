from __future__ import annotations

import hmac
from datetime import date
from typing import TYPE_CHECKING, Annotated

from fasthtml.common import (
    FT,
    A,
    Div,
    Form,
    FtResponse,
    Hidden,
    P,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from romanian_news import BUCHAREST, Sha256
from romanian_news.catalog_transport import ResearchCatalogError
from romanian_news.current_report import (
    CurrentDailyReportHead,
    DailyReportFreshness,
)
from romanian_news.reader import clock
from romanian_news.reader.dagster_repair import (
    DagsterRepairError,
    DagsterRepairUnavailable,
    RepairRun,
)

if TYPE_CHECKING:
    from romanian_news.reader.domain import ReaderDomain


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
        if value.day != clock.bucharest_today():
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
        if value.day != clock.bucharest_today():
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


def _valid_csrf(request: Request, supplied: str) -> bool:
    expected = request.session.get("csrf_token")
    return isinstance(expected, str) and hmac.compare_digest(supplied, expected)
