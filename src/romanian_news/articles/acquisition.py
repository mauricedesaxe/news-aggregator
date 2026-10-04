from __future__ import annotations

import json
import signal
import threading
import time
from collections import defaultdict, deque
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from types import FrameType

import requests

from romanian_news import BUCHAREST, Sha256
from romanian_news.articles.extraction import extract_article, normalize_article_url
from romanian_news.articles.models import (
    ArticleAcquisitionFailure,
    ArticleAcquisitionResult,
    ArticleBatchCapture,
    ArticleBatchItemResult,
    ArticleBatchSkip,
    ArticleCapture,
    ArticleFailureKind,
    ArticleWorkItem,
    ArticleWorkLane,
    ArticleWorkPlan,
    ArticleWorkStatus,
    MaterializedArticleWorkItem,
)
from romanian_news.articles.recovery import (
    ArticleAttemptState,
    ArticleRecoveryView,
    article_work_generation,
    read_article_recovery_view,
)
from romanian_news.catalog import articles as article_catalog
from romanian_news.catalog import feeds as feed_catalog
from romanian_news.catalog.feeds import (
    read_cataloged_feed_entry_reference,
    read_cataloged_feed_entry_references,
)
from romanian_news.catalog_transport import ResearchCatalogError
from romanian_news.feeds.materialization import materialize_cataloged_feed_entries
from romanian_news.feeds.models import CatalogedFeedEntryReference, FeedRegistry, FeedSpec
from romanian_news.http import UnsafeNewsRedirect, create_news_session, get_with_validated_redirects
from romanian_news.identity import sha256
from romanian_news.storage import (
    ResearchObjectIntegrityError,
    ResearchObjectUnavailable,
)

_ARTICLE_USER_AGENT = (
    "Romanian news aggregator/1.0 (+https://github.com/mauricedesaxe/news-aggregator)"
)
_ARTICLE_TIMEOUT_SECONDS = 10
_ARTICLE_TOTAL_TIMEOUT_SECONDS = 20
_ARTICLE_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
_ARTICLE_STREAM_CHUNK_BYTES = 64 * 1024
_LIVE_UNSEEN_SLOTS = 25
_HISTORICAL_UNSEEN_SLOTS = 25


@dataclass(frozen=True)
class ArticleRecoveryReceipt:
    recovery_ids: tuple[Sha256, ...]
    released_event_ids: tuple[Sha256, ...]


@dataclass(frozen=True)
class ArticleRecoveryStatus:
    event_id: Sha256
    work_generation: Sha256
    quarantined: bool
    deterministic_fingerprint: Sha256 | None


@dataclass(frozen=True)
class _ArticlePage:
    content: bytes | None
    url: str | None
    latency_ms: int
    retrieval_error: str | None
    failure_kind: ArticleFailureKind | None = None
    must_fail: bool = False


def article_work_is_complete(
    source: CatalogedFeedEntryReference,
    registry: FeedRegistry,
    *,
    previous_capture_at: datetime | None,
) -> bool:
    feeds = {feed.id: feed for feed in registry.feeds}
    feed = feeds.get(source.feed_id)
    if feed is None:
        raise ValueError(f"Unknown feed for article work: {source.feed_id}")
    alias = f"url:{normalize_article_url(str(source.url), feed.article_hosts)}"
    state = _article_states((source,), feeds).get(alias)
    if state is None:
        return False
    latest = state.source_updated_at or state.published_at
    captured_at = state.captured_at
    return _effective_time(source) <= latest and (
        previous_capture_at is None or captured_at > previous_capture_at
    )


def plan_article_work(
    registry: FeedRegistry,
    *,
    implementation_ref: str,
    now: datetime,
    start_at: datetime,
    end_at: datetime,
    limit: int = 50,
    revalidate_before: datetime | None = None,
) -> ArticleWorkPlan:
    """Plan exact article jobs while retaining the workflow's lane priorities."""
    feeds = {feed.id: feed for feed in registry.feeds}
    cataloged = read_cataloged_feed_entry_references(start_at=start_at, end_at=end_at)
    sources, _ = _select_article_sources(cataloged, feeds, start_at, end_at)
    states = _article_states(sources, feeds)
    recovery = read_article_recovery_view(
        _article_work_generations(sources, feeds, states),
    )
    plan = _plan_article_work(
        sources,
        feeds,
        now=now,
        limit=limit,
        revalidate_before=revalidate_before,
        attempt_states=recovery.attempt_states,
        work_generations=recovery.work_generations,
        article_states=states,
    )
    return plan.model_copy(
        update={
            "source_covered_days": _covered_news_days(
                registry,
                _requested_days(start_at, end_at),
            )
        }
    )


def read_article_work_status(
    registry: FeedRegistry,
    *,
    implementation_ref: str,
    now: datetime,
    start_at: datetime,
    end_at: datetime,
    revalidate_before: datetime | None = None,
) -> ArticleWorkStatus:
    """Read completion state without producing another batch."""
    feeds = {feed.id: feed for feed in registry.feeds}
    cataloged = read_cataloged_feed_entry_references(start_at=start_at, end_at=end_at)
    sources, _ = _select_article_sources(cataloged, feeds, start_at, end_at)
    states = _article_states(sources, feeds)
    recovery = read_article_recovery_view(
        _article_work_generations(sources, feeds, states),
    )
    state = _plan_article_work(
        sources,
        feeds,
        now=now,
        limit=max(1, len(sources)),
        revalidate_before=revalidate_before,
        attempt_states=recovery.attempt_states,
        work_generations=recovery.work_generations,
        article_states=states,
    )
    return ArticleWorkStatus(
        retryable_entries=len(state.selected) + state.remaining_entries,
        deferred_event_ids=state.deferred_event_ids,
        quarantined_event_ids=state.quarantined_event_ids,
        source_covered_days=_covered_news_days(
            registry,
            _requested_days(start_at, end_at),
        ),
    )


def load_exact_article_work(
    event_ids: tuple[Sha256, ...],
    registry: FeedRegistry,
    *,
    implementation_ref: str,
    now: datetime,
    start_at: datetime,
    end_at: datetime,
    revalidate_before: datetime | None,
) -> tuple[ArticleWorkItem, ...]:
    """Load requested work without selecting replacement items."""
    if len(event_ids) > 10:
        raise ValueError("An article batch cannot exceed 10 event IDs")
    sources = tuple(read_cataloged_feed_entry_reference(event_id) for event_id in event_ids)
    feeds = {feed.id: feed for feed in registry.feeds}
    selected_sources, invalid = _select_article_sources(sources, feeds, start_at, end_at)
    if invalid:
        raise ValueError("Exact article batch contains an invalid source")
    states = _article_states(selected_sources, feeds)
    recovery = read_article_recovery_view(
        _article_work_generations(selected_sources, feeds, states),
    )
    active = tuple(
        source
        for source in selected_sources
        if not (
            recovery.attempt_states.get(source.event_id)
            and recovery.attempt_states[source.event_id].quarantined
        )
    )
    work_items = tuple(
        work
        for source in active
        if (
            work := _article_work_item(
                source,
                states.get(_source_alias(source, feeds)),
                now,
                revalidate_before,
                work_generation=recovery.work_generations[source.event_id],
            )
        )
        is not None
    )
    requested = set(event_ids)
    return tuple(work for work in work_items if work.source.event_id in requested)


def recover_quarantined_article_events(
    event_ids: tuple[Sha256, ...],
    registry: FeedRegistry,
    *,
    requested_by: str,
    reason: str,
    requested_at: datetime,
    expected_work_generations: dict[Sha256, Sha256],
) -> ArticleRecoveryReceipt:
    _validate_recovery_request(
        event_ids, requested_by, reason, requested_at, expected_work_generations
    )
    timestamp = requested_at.astimezone(UTC)
    recovery_ids = _recovery_request_ids(
        event_ids, expected_work_generations, requested_by, reason, timestamp
    )
    existing_ids = {
        override.recovery_id
        for override in article_catalog.read_article_recovery_overrides(event_ids)
    }
    replayed = [recovery_id in existing_ids for recovery_id in recovery_ids]
    if all(replayed):
        return ArticleRecoveryReceipt(recovery_ids=recovery_ids, released_event_ids=event_ids)
    if any(replayed):
        raise ValueError("Recovery request was only partly recorded")
    base_generations = _recovery_base_generations(event_ids, registry)
    view = read_article_recovery_view(base_generations)
    _require_current_quarantine(event_ids, expected_work_generations, view)
    overrides = tuple(
        article_catalog.ArticleRecoveryOverride(
            recovery_id=recovery_id,
            event_id=event_id,
            base_work_generation=base_generations[event_id],
            expected_work_generation=expected_work_generations[event_id],
            requested_by=requested_by.strip(),
            reason=reason.strip(),
            requested_at=timestamp,
        )
        for event_id, recovery_id in zip(event_ids, recovery_ids, strict=True)
    )
    article_catalog.write_article_recovery_overrides(overrides)
    return ArticleRecoveryReceipt(
        recovery_ids=recovery_ids,
        released_event_ids=event_ids,
    )


def _validate_recovery_request(
    event_ids: tuple[Sha256, ...],
    requested_by: str,
    reason: str,
    requested_at: datetime,
    expected_work_generations: dict[Sha256, Sha256],
) -> None:
    if not 1 <= len(event_ids) <= 10 or len(event_ids) != len(set(event_ids)):
        raise ValueError("Recovery requires 1 to 10 unique event IDs")
    if not requested_by.strip() or not reason.strip():
        raise ValueError("Recovery requires a requester and reason")
    if requested_at.tzinfo is None:
        raise ValueError("Recovery time must include a UTC offset")
    if set(expected_work_generations) != set(event_ids):
        raise ValueError("Recovery needs one expected generation per event")


def _recovery_request_ids(
    event_ids: tuple[Sha256, ...],
    expected_work_generations: dict[Sha256, Sha256],
    requested_by: str,
    reason: str,
    requested_at: datetime,
) -> tuple[Sha256, ...]:
    return tuple(
        sha256(
            json.dumps(
                {
                    "event_id": event_id,
                    "expected_work_generation": expected_work_generations[event_id],
                    "requested_by": requested_by.strip(),
                    "reason": reason.strip(),
                    "requested_at": requested_at.isoformat(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for event_id in event_ids
    )


def _recovery_base_generations(
    event_ids: tuple[Sha256, ...],
    registry: FeedRegistry,
) -> dict[Sha256, Sha256]:
    sources = tuple(read_cataloged_feed_entry_reference(event_id) for event_id in event_ids)
    feeds = {feed.id: feed for feed in registry.feeds}
    selected, invalid = _select_article_sources(sources, feeds, None, None)
    if invalid or len(selected) != len(event_ids):
        raise ValueError("Recovery contains an invalid or duplicate article source")
    return _article_work_generations(selected, feeds, _article_states(selected, feeds))


def _require_current_quarantine(
    event_ids: tuple[Sha256, ...],
    expected_work_generations: dict[Sha256, Sha256],
    view: ArticleRecoveryView,
) -> None:
    for event_id in event_ids:
        attempt = view.attempt_states.get(event_id)
        if attempt is None or not attempt.quarantined:
            raise ValueError(f"Article is not quarantined: {event_id}")
        if expected_work_generations[event_id] != view.work_generations[event_id]:
            raise ValueError(f"Article recovery generation changed: {event_id}")


def inspect_article_recovery_event(
    event_id: Sha256,
    registry: FeedRegistry,
) -> ArticleRecoveryStatus:
    source = read_cataloged_feed_entry_reference(event_id)
    feeds = {feed.id: feed for feed in registry.feeds}
    selected, invalid = _select_article_sources((source,), feeds, None, None)
    if invalid or not selected:
        raise ValueError(f"Invalid article source: {event_id}")
    states = _article_states(selected, feeds)
    view = read_article_recovery_view(_article_work_generations(selected, feeds, states))
    attempt = view.attempt_states.get(event_id)
    return ArticleRecoveryStatus(
        event_id=event_id,
        work_generation=view.work_generations[event_id],
        quarantined=attempt.quarantined if attempt is not None else False,
        deterministic_fingerprint=(
            attempt.deterministic_fingerprint if attempt is not None else None
        ),
    )


def acquire_article_batch_item(
    work: ArticleWorkItem,
    registry: FeedRegistry,
) -> ArticleBatchItemResult:
    """Acquire one fixed batch item after a durable completion recheck."""
    try:
        if article_work_is_complete(
            work.source,
            registry,
            previous_capture_at=work.last_captured_at,
        ):
            return ArticleBatchSkip(event_id=work.source.event_id)
        source = materialize_cataloged_feed_entries((work.source,), registry)[0]
    except (ResearchCatalogError, ResearchObjectUnavailable) as error:
        message = f"{work.source.feed_id} {work.source.url}: {error}"
        return _article_failure(work.source.event_id, ArticleFailureKind.INFRASTRUCTURE, message)
    except (ResearchObjectIntegrityError, KeyError, ValueError) as error:
        message = f"{work.source.feed_id} {work.source.url}: {error}"
        return _article_failure(work.source.event_id, ArticleFailureKind.DETERMINISTIC, message)
    materialized = MaterializedArticleWorkItem(
        source=source,
        lane=work.lane,
        last_captured_at=work.last_captured_at,
    )
    _index, _work, page = _fetch_article_page_with_session(0, materialized)
    capture, failure = _article_outcome(
        materialized,
        {feed.id: feed for feed in registry.feeds},
        page,
    )
    if failure is not None:
        return failure
    if capture is None:
        raise RuntimeError("Article acquisition produced neither a capture nor a failure")
    return ArticleBatchCapture(capture=capture)


def acquire_articles(
    registry: FeedRegistry,
    *,
    now: datetime,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
    limit: int = 50,
    revalidate_before: datetime | None = None,
    deadline: float | None = None,
) -> ArticleAcquisitionResult:
    """Fetch article work derived from the immutable feed-entry catalog."""
    if now.tzinfo is None:
        raise ValueError("Article acquisition time must include a UTC offset")
    if limit < 1 or limit > 50:
        raise ValueError("Article limit must be between 1 and 50")
    for name, value in (("start", start_at), ("end", end_at), ("revalidation", revalidate_before)):
        if value is not None and value.tzinfo is None:
            raise ValueError(f"Article {name} time must include a UTC offset")
    effective_end = end_at or now
    feeds = {feed.id: feed for feed in registry.feeds}
    cataloged = read_cataloged_feed_entry_references(start_at=start_at, end_at=effective_end)
    sources, invalid = _select_article_sources(cataloged, feeds, start_at, effective_end)
    plan = _plan_article_work(
        sources,
        feeds,
        now=now,
        limit=limit,
        revalidate_before=revalidate_before,
    )
    materialized_sources = materialize_cataloged_feed_entries(
        tuple(work.source for work in plan.selected),
        registry,
    )
    selected = tuple(
        MaterializedArticleWorkItem(
            source=source,
            lane=work.lane,
            last_captured_at=work.last_captured_at,
        )
        for work, source in zip(plan.selected, materialized_sources, strict=True)
    )
    captures = []
    errors = []
    remaining_entries = plan.remaining_entries
    remaining_days = set(plan.remaining_days)
    outcomes = _fetch_articles_concurrently(selected, feeds, deadline)
    for work, capture, error in outcomes:
        if capture is not None:
            captures.append(capture)
        elif error is not None:
            errors.append(error.message)
            remaining_days.add(_work_day(work))
        else:
            remaining_entries += 1
            remaining_days.add(_work_day(work))
    requested_days = _requested_days(start_at, effective_end)
    return ArticleAcquisitionResult(
        captures=tuple(captures),
        skipped_entries=invalid + plan.complete_entries + remaining_entries,
        remaining_entries=remaining_entries,
        errors=tuple(errors),
        remaining_days=tuple(sorted(remaining_days)),
        source_covered_days=_covered_news_days(registry, requested_days),
    )


def _fetch_articles_concurrently(
    selected: tuple[MaterializedArticleWorkItem, ...],
    feeds: dict[str, FeedSpec],
    deadline: float | None,
) -> tuple[
    tuple[MaterializedArticleWorkItem, ArticleCapture | None, ArticleAcquisitionFailure | None], ...
]:
    if not selected:
        return ()
    if deadline is not None and time.monotonic() + 60 >= deadline:
        return tuple((work, None, None) for work in selected)
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = [
            executor.submit(
                _fetch_article_page_with_session,
                index,
                work,
            )
            for index, work in enumerate(selected)
        ]
        indexed = [future.result() for future in futures]
    indexed.sort(key=lambda value: value[0])
    return tuple((work, *_article_outcome(work, feeds, page)) for _index, work, page in indexed)


def _article_outcome(
    work: MaterializedArticleWorkItem,
    feeds: dict[str, FeedSpec],
    page: _ArticlePage,
) -> tuple[ArticleCapture | None, ArticleAcquisitionFailure | None]:
    feed = feeds[work.source.entry.feed_id]
    if page.must_fail:
        if page.failure_kind is None:
            raise RuntimeError("A required article failure kind is missing")
        message = page.retrieval_error or "Article page retrieval failed"
        return None, _article_failure(work.source.event_id, page.failure_kind, message)
    try:
        return _extract_article_capture(work, feed, page), None
    except (KeyError, ValueError) as error:
        entry = work.source.entry
        message = f"{entry.feed_id} {entry.url}: {error}"
        kind = page.failure_kind or ArticleFailureKind.DETERMINISTIC
        return None, _article_failure(work.source.event_id, kind, message)


def _article_failure(
    event_id: Sha256,
    kind: ArticleFailureKind,
    error: object,
) -> ArticleAcquisitionFailure:
    message = str(error)
    return ArticleAcquisitionFailure(
        event_id=event_id,
        kind=kind,
        fingerprint=sha256(f"{kind.value}:{message}".encode()),
        message=message,
    )


def _fetch_article_page_with_session(
    index: int,
    work: MaterializedArticleWorkItem,
) -> tuple[int, MaterializedArticleWorkItem, _ArticlePage]:
    session = create_news_session()
    try:
        page = _fetch_article_page(session, work)
    finally:
        close = getattr(session, "close", None)
        if close is not None:
            close()
    return index, work, page


def _select_article_sources(
    events: tuple[CatalogedFeedEntryReference, ...],
    feeds: dict[str, FeedSpec],
    start_at: datetime | None,
    end_at: datetime | None,
) -> tuple[tuple[CatalogedFeedEntryReference, ...], int]:
    sources: dict[tuple[str, str], CatalogedFeedEntryReference] = {}
    skipped = 0
    for source in events:
        if start_at is not None and source.published_at < start_at:
            skipped += 1
            continue
        if end_at is not None and source.published_at >= end_at:
            skipped += 1
            continue
        feed = feeds.get(source.feed_id)
        if feed is None:
            skipped += 1
            continue
        try:
            normalized_url = normalize_article_url(str(source.url), feed.article_hosts)
        except ValueError:
            skipped += 1
            continue
        if not feed.accepts_article_url(normalized_url):
            skipped += 1
            continue
        key = (feed.outlet_id, normalized_url)
        existing = sources.get(key)
        if existing is None or _source_order(source) > _source_order(existing):
            sources[key] = source
    return tuple(sources.values()), skipped


def _article_work_item(
    source: CatalogedFeedEntryReference,
    state: article_catalog.ArticleCatalogState | None,
    now: datetime,
    revalidate_before: datetime | None,
    *,
    work_generation: Sha256 | None = None,
) -> ArticleWorkItem | None:
    generation = work_generation or article_work_generation(
        source.event_id, state.captured_at if state is not None else None
    )
    if state is None:
        lane = (
            ArticleWorkLane.LIVE_UNSEEN
            if source.published_at.astimezone(BUCHAREST).date() == now.astimezone(BUCHAREST).date()
            else ArticleWorkLane.HISTORICAL_UNSEEN
        )
        return ArticleWorkItem(source=source, lane=lane, work_generation=generation)
    latest = state.source_updated_at or state.published_at
    captured_at = state.captured_at
    if _effective_time(source) > latest:
        return ArticleWorkItem(
            source=source,
            lane=ArticleWorkLane.SOURCE_UPDATE,
            work_generation=generation,
            last_captured_at=captured_at,
        )
    if revalidate_before is not None and captured_at < revalidate_before:
        return ArticleWorkItem(
            source=source,
            lane=ArticleWorkLane.REVALIDATION,
            work_generation=generation,
            last_captured_at=captured_at,
        )
    return None


def _plan_article_work(
    sources: tuple[CatalogedFeedEntryReference, ...],
    feeds: dict[str, FeedSpec],
    *,
    now: datetime,
    limit: int,
    revalidate_before: datetime | None,
    attempt_states: dict[Sha256, ArticleAttemptState] | None = None,
    work_generations: dict[Sha256, Sha256] | None = None,
    article_states: dict[str, article_catalog.ArticleCatalogState] | None = None,
) -> ArticleWorkPlan:
    states = article_states if article_states is not None else _article_states(sources, feeds)
    attempts = attempt_states or {}
    lanes: dict[ArticleWorkLane, list[ArticleWorkItem]] = {lane: [] for lane in ArticleWorkLane}
    complete = 0
    deferred: list[ArticleWorkItem] = []
    quarantined: list[Sha256] = []
    for source, alias in _source_aliases(sources, feeds):
        state = states.get(alias)
        work = _article_work_item(
            source,
            state,
            now,
            revalidate_before,
            work_generation=_provided_work_generation(source.event_id, work_generations),
        )
        if work is None:
            complete += 1
            continue
        attempt = attempts.get(source.event_id)
        if attempt is not None and attempt.quarantined:
            quarantined.append(source.event_id)
        elif attempt is not None and attempt.deferred(now):
            deferred.append(work)
        else:
            lanes[work.lane].append(work)
    for values in lanes.values():
        values.sort(key=_work_order)

    live = lanes[ArticleWorkLane.LIVE_UNSEEN]
    selected_live = live[: min(_LIVE_UNSEEN_SLOTS, limit)]
    historical_slots = min(_HISTORICAL_UNSEEN_SLOTS, max(0, limit - len(selected_live)))
    historical = _prioritize_historical_unseen(
        lanes[ArticleWorkLane.HISTORICAL_UNSEEN],
        historical_slots,
    )
    selected_historical = historical[:historical_slots]
    selected = [*selected_historical, *selected_live]
    capacity = limit - len(selected)
    if capacity:
        live_extra = live[len(selected_live) : len(selected_live) + capacity]
        selected.extend(live_extra)
        capacity -= len(live_extra)
    if capacity:
        selected.extend(historical[len(selected_historical) : len(selected_historical) + capacity])
    selected_ids = {value.source.event_id for value in selected}
    unseen_remaining = [
        value for value in (*historical, *live) if value.source.event_id not in selected_ids
    ]
    updates = lanes[ArticleWorkLane.SOURCE_UPDATE]
    revalidations = lanes[ArticleWorkLane.REVALIDATION]
    if not unseen_remaining:
        capacity = limit - len(selected)
        selected.extend(updates[:capacity])
        update_remaining = updates[capacity:]
        if not update_remaining:
            capacity = limit - len(selected)
            selected.extend(revalidations[:capacity])
            revalidation_remaining = revalidations[capacity:]
        else:
            revalidation_remaining = revalidations
    else:
        update_remaining = updates
        revalidation_remaining = revalidations
    remaining = [*unseen_remaining, *update_remaining, *revalidation_remaining, *deferred]
    return ArticleWorkPlan(
        selected=tuple(selected),
        remaining_entries=len(remaining),
        remaining_days=tuple(sorted({_work_day(value) for value in remaining})),
        complete_entries=complete,
        deferred_event_ids=tuple(value.source.event_id for value in deferred),
        quarantined_event_ids=tuple(sorted(quarantined)),
    )


def _provided_work_generation(
    event_id: Sha256,
    work_generations: dict[Sha256, Sha256] | None,
) -> Sha256 | None:
    return None if work_generations is None else work_generations.get(event_id)


def _article_work_generations(
    sources: tuple[CatalogedFeedEntryReference, ...],
    feeds: dict[str, FeedSpec],
    states: dict[str, article_catalog.ArticleCatalogState],
) -> dict[Sha256, Sha256]:
    generations = {}
    for source, alias in _source_aliases(sources, feeds):
        state = states.get(alias)
        completed_at = state.captured_at if state is not None else None
        generations[source.event_id] = article_work_generation(source.event_id, completed_at)
    return generations


def _article_states(
    sources: tuple[CatalogedFeedEntryReference, ...],
    feeds: dict[str, FeedSpec],
) -> dict[str, article_catalog.ArticleCatalogState]:
    aliases = tuple(alias for _source, alias in _source_aliases(sources, feeds))
    return article_catalog.read_article_catalog_states(aliases)


def _prioritize_historical_unseen(
    values: list[ArticleWorkItem],
    reservation: int,
) -> list[ArticleWorkItem]:
    by_day_and_feed: dict[date, dict[str, deque[ArticleWorkItem]]] = defaultdict(
        lambda: defaultdict(deque)
    )
    for value in values:
        by_day_and_feed[_work_day(value)][value.source.feed_id].append(value)
    day_queues: dict[date, deque[ArticleWorkItem]] = {}
    for day, feeds in by_day_and_feed.items():
        queue = deque()
        while any(feeds.values()):
            for feed_id in sorted(feeds):
                if feeds[feed_id]:
                    queue.append(feeds[feed_id].popleft())
        day_queues[day] = queue

    days = sorted(day_queues, reverse=True)
    selected: list[ArticleWorkItem] = []
    if days and reservation:
        recent_slots = reservation
        if len(days) > 1 and reservation > 1:
            recent_slots = max(1, int(reservation * 0.8))
        for _ in range(recent_slots):
            if day_queues[days[0]]:
                selected.append(day_queues[days[0]].popleft())
        if len(days) > 1:
            for _ in range(reservation - recent_slots):
                if day_queues[days[-1]]:
                    selected.append(day_queues[days[-1]].popleft())
        for day in days:
            while day_queues[day] and len(selected) < reservation:
                selected.append(day_queues[day].popleft())

    return [*selected, *(value for day in days for value in day_queues[day])]


def _source_alias(
    source: CatalogedFeedEntryReference,
    feeds: dict[str, FeedSpec],
) -> str:
    return f"url:{normalize_article_url(str(source.url), feeds[source.feed_id].article_hosts)}"


def _source_aliases(
    sources: tuple[CatalogedFeedEntryReference, ...],
    feeds: dict[str, FeedSpec],
) -> tuple[tuple[CatalogedFeedEntryReference, str], ...]:
    return tuple(
        (
            source,
            _source_alias(source, feeds),
        )
        for source in sources
    )


class _ArticleDownloadTimeout(RuntimeError):
    pass


class _ArticleResponseTooLarge(ValueError):
    pass


def _raise_article_download_timeout(_signum: int, _frame: FrameType | None) -> None:
    raise _ArticleDownloadTimeout(
        f"Article response exceeded {_ARTICLE_TOTAL_TIMEOUT_SECONDS} seconds"
    )


@contextmanager
def _article_http_deadline() -> Generator[None]:
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    _ = signal.signal(signal.SIGALRM, _raise_article_download_timeout)
    _ = signal.setitimer(signal.ITIMER_REAL, _ARTICLE_TOTAL_TIMEOUT_SECONDS)
    try:
        yield
    finally:
        _ = signal.setitimer(signal.ITIMER_REAL, 0)
        _ = signal.signal(signal.SIGALRM, previous_handler)
        _ = signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _fetch_article_page(
    session: requests.Session,
    work: MaterializedArticleWorkItem,
) -> _ArticlePage:
    entry = work.source.entry
    started = time.monotonic()
    response = None
    page_content = None
    page_url = None
    error_message = None
    failure_kind = None
    must_fail = False
    try:
        with _article_http_deadline():
            response = get_with_validated_redirects(
                session,
                str(entry.url),
                headers={
                    "User-Agent": _ARTICLE_USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                },
                timeout=_ARTICLE_TIMEOUT_SECONDS,
                stream=True,
            )
            response.raise_for_status()
            page_url = response.url
            page_content = _read_bounded_article_content(response, started)
    except _ArticleDownloadTimeout as error:
        error_message = str(error)
        failure_kind = ArticleFailureKind.INFRASTRUCTURE
        must_fail = True
    except _ArticleResponseTooLarge as error:
        error_message = str(error)
        failure_kind = ArticleFailureKind.DETERMINISTIC
        must_fail = True
    except requests.Timeout as error:
        error_message = str(error)
        failure_kind = ArticleFailureKind.INFRASTRUCTURE
        must_fail = True
    except requests.RequestException as error:
        error_message = str(error)
        failure_kind = classify_article_request_failure(error)
    finally:
        if response is not None:
            response.close()
    latency_ms = round((time.monotonic() - started) * 1000)
    return _ArticlePage(
        content=page_content,
        url=page_url,
        latency_ms=latency_ms,
        retrieval_error=error_message,
        failure_kind=failure_kind,
        must_fail=must_fail,
    )


def _read_bounded_article_content(response: requests.Response, started: float) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = 0
        if declared_size > _ARTICLE_MAX_RESPONSE_BYTES:
            raise _ArticleResponseTooLarge(
                f"Article response exceeds {_ARTICLE_MAX_RESPONSE_BYTES} bytes"
            )
    chunks = []
    size = 0
    for chunk in response.iter_content(chunk_size=_ARTICLE_STREAM_CHUNK_BYTES):
        if time.monotonic() - started > _ARTICLE_TOTAL_TIMEOUT_SECONDS:
            raise _ArticleDownloadTimeout(
                f"Article response exceeded {_ARTICLE_TOTAL_TIMEOUT_SECONDS} seconds"
            )
        if not chunk:
            continue
        size += len(chunk)
        if size > _ARTICLE_MAX_RESPONSE_BYTES:
            raise _ArticleResponseTooLarge(
                f"Article response exceeds {_ARTICLE_MAX_RESPONSE_BYTES} bytes"
            )
        chunks.append(chunk)
    if time.monotonic() - started > _ARTICLE_TOTAL_TIMEOUT_SECONDS:
        raise _ArticleDownloadTimeout(
            f"Article response exceeded {_ARTICLE_TOTAL_TIMEOUT_SECONDS} seconds"
        )
    return b"".join(chunks)


def classify_article_request_failure(error: requests.RequestException) -> ArticleFailureKind:
    if isinstance(error, UnsafeNewsRedirect):
        return ArticleFailureKind.DETERMINISTIC
    status = error.response.status_code if error.response is not None else None
    if status is not None and 400 <= status < 500 and status != 429:
        return ArticleFailureKind.DETERMINISTIC
    return ArticleFailureKind.INFRASTRUCTURE


def _extract_article_capture(
    work: MaterializedArticleWorkItem,
    feed: FeedSpec,
    page: _ArticlePage,
) -> ArticleCapture:
    entry = work.source.entry
    return ArticleCapture(
        article=extract_article(entry, feed, html=page.content, final_url=page.url),
        source=work.source,
        captured_at=datetime.now(UTC),
        page_url=page.url,
        page_content=page.content,
        page_content_digest=sha256(page.content) if page.content else None,
        latency_ms=page.latency_ms,
        retrieval_error=page.retrieval_error,
    )


def _covered_news_days(
    registry: FeedRegistry,
    requested_days: tuple[date, ...],
) -> tuple[date, ...]:
    if not requested_days:
        return ()
    counts = feed_catalog.read_successful_feed_counts(requested_days)
    return tuple(day for day in requested_days if counts.get(day) == len(registry.feeds))


def _requested_days(start_at: datetime | None, end_at: datetime) -> tuple[date, ...]:
    if start_at is None:
        return ()
    first = start_at.astimezone(BUCHAREST).date()
    last = (end_at - timedelta(microseconds=1)).astimezone(BUCHAREST).date()
    return tuple(first + timedelta(days=offset) for offset in range((last - first).days + 1))


def _source_order(source: CatalogedFeedEntryReference) -> tuple[datetime, datetime, Sha256]:
    return _effective_time(source), source.observed_at, source.event_id


def _work_order(work: ArticleWorkItem) -> tuple[datetime, datetime, Sha256]:
    return _effective_time(work.source), work.source.observed_at, work.source.event_id


def _work_day(work: ArticleWorkItem | MaterializedArticleWorkItem) -> date:
    return work.source.published_at.astimezone(BUCHAREST).date()


def _effective_time(source: CatalogedFeedEntryReference) -> datetime:
    return source.source_updated_at or source.published_at
