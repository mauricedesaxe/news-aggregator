from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Annotated

from pydantic import Field

from romanian_news import BUCHAREST, NewsModel, Sha256
from romanian_news.articles.models import ArticleAcquisitionFailure, ArticleFailureKind
from romanian_news.catalog import articles as article_catalog

_QUARANTINE_ATTEMPTS = 3


class ArticleAttemptState(NewsModel):
    event_id: Sha256
    implementation_ref: str
    work_generation: Sha256
    retry_at: datetime
    deterministic_fingerprint: Sha256 | None
    unchanged_deterministic_attempts: Annotated[int, Field(ge=0)]

    @property
    def quarantined(self) -> bool:
        return self.unchanged_deterministic_attempts >= _QUARANTINE_ATTEMPTS

    def deferred(self, now: datetime) -> bool:
        return not self.quarantined and self.retry_at > now


def article_work_generation(event_id: Sha256, completed_at: datetime | None) -> Sha256:
    checkpoint = completed_at.astimezone(UTC).isoformat() if completed_at is not None else "unseen"
    return hashlib.sha256(f"{event_id}:{checkpoint}".encode()).hexdigest()


def read_article_attempt_states(
    work_generations: Mapping[Sha256, Sha256],
) -> dict[Sha256, ArticleAttemptState]:
    event_ids = tuple(work_generations)
    if not event_ids:
        return {}
    rows = article_catalog.read_article_failure_attempts(event_ids)
    grouped: dict[Sha256, list[article_catalog.ArticleFailureAttempt]] = defaultdict(list)
    for row in rows:
        if row.work_generation == work_generations[row.event_id]:
            grouped[row.event_id].append(row)
    return {
        event_id: _derive_attempt_state(
            event_id,
            work_generations[event_id],
            values,
        )
        for event_id, values in grouped.items()
    }


def record_article_failure_attempts(
    failures: tuple[ArticleAcquisitionFailure, ...],
    *,
    work_generations: Mapping[Sha256, Sha256],
    run_id: str,
    retry_number: int,
    implementation_ref: str,
    attempted_at: datetime,
) -> None:
    if not failures:
        return
    if attempted_at.tzinfo is None:
        raise ValueError("Article failure attempt time must include a UTC offset")
    states = read_article_attempt_states(work_generations)
    attempts = []
    for failure in failures:
        work_generation = work_generations[failure.event_id]
        state = states.get(failure.event_id)
        unchanged = (
            state.unchanged_deterministic_attempts
            if state is not None
            and failure.kind == ArticleFailureKind.DETERMINISTIC
            and state.deterministic_fingerprint == failure.fingerprint
            else 0
        )
        delay_minutes = (
            5 if failure.kind == ArticleFailureKind.INFRASTRUCTURE else 15 * 2**unchanged
        )
        retry_at = attempted_at + timedelta(minutes=delay_minutes)
        attempt_id = hashlib.sha256(
            f"{run_id}:{retry_number}:{failure.event_id}".encode()
        ).hexdigest()
        attempts.append(
            article_catalog.ArticleFailureAttemptWrite(
                attempt_id=attempt_id,
                event_id=failure.event_id,
                implementation_ref=implementation_ref,
                work_generation=work_generation,
                dagster_run_id=run_id,
                retry_number=retry_number,
                failure_kind=failure.kind,
                failure_fingerprint=failure.fingerprint,
                error=failure.message,
                attempted_at=attempted_at,
                retry_at=retry_at,
            )
        )
    article_catalog.write_article_failure_attempts(tuple(attempts))


def read_article_candidate_days(start_day: date, end_day: date) -> tuple[date, ...]:
    lower = datetime.combine(start_day - timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    upper = datetime.combine(end_day + timedelta(days=2), datetime.min.time(), tzinfo=UTC)
    candidate_times = article_catalog.read_article_candidate_times(lower, upper)
    days = {value.astimezone(BUCHAREST).date() for value in candidate_times}
    return tuple(sorted(day for day in days if start_day <= day <= end_day))


def _derive_attempt_state(
    event_id: Sha256,
    work_generation: Sha256,
    attempts: list[article_catalog.ArticleFailureAttempt],
) -> ArticleAttemptState:
    deterministic_fingerprint = None
    unchanged = 0
    for attempt in attempts:
        if attempt.failure_kind != ArticleFailureKind.DETERMINISTIC:
            continue
        if attempt.failure_fingerprint == deterministic_fingerprint:
            unchanged += 1
        else:
            deterministic_fingerprint = attempt.failure_fingerprint
            unchanged = 1
    return ArticleAttemptState(
        event_id=event_id,
        implementation_ref=attempts[-1].implementation_ref,
        work_generation=work_generation,
        retry_at=attempts[-1].retry_at,
        deterministic_fingerprint=deterministic_fingerprint,
        unchanged_deterministic_attempts=unchanged,
    )
