import hashlib
import signal
import threading
from datetime import date, datetime, timedelta

import pytest
import requests
from pydantic import HttpUrl

from romanian_news.articles.acquisition import (
    _article_outcome,
    _article_states,
    _ArticlePage,
    _covered_news_days,
    _fetch_article_page,
    _plan_article_work,
    _select_article_sources,
    acquire_article_batch_item,
    acquire_articles,
    classify_article_request_failure,
    load_exact_article_work,
)
from romanian_news.articles.models import (
    ArticleAcquisitionFailure,
    ArticleBatchSkip,
    ArticleWorkItem,
    ArticleWorkLane,
    MaterializedArticleWorkItem,
)
from romanian_news.articles.recovery import ArticleAttemptState
from romanian_news.catalog.articles import ArticleCatalogState
from romanian_news.catalog_transport import ResearchCatalogError
from romanian_news.feeds.models import (
    CatalogedFeedEntry,
    CatalogedFeedEntryReference,
    FeedEntry,
)
from romanian_news.feeds.registry import feed_registry
from romanian_news.storage import (
    ResearchObjectIntegrityError,
    ResearchObjectUnavailable,
)

_VERSION_ID = "a" * 64
_EVENT_ID = "b" * 64


def test_article_acquisition_rejects_limits_above_tick_capacity() -> None:
    with pytest.raises(ValueError, match="between 1 and 50"):
        acquire_articles(
            feed_registry(),
            now=datetime.fromisoformat("2026-09-01T12:00:00+03:00"),
            limit=51,
        )


def test_article_states_request_all_normalized_aliases_from_catalog(monkeypatch) -> None:
    entries = tuple(_entry(f"article-{index}", "2026-09-09T12:00:00+03:00") for index in range(75))
    sources = tuple(_source(entry, event_id=f"{index:064x}") for index, entry in enumerate(entries))
    registry = feed_registry()
    feeds = {feed.id: feed for feed in registry.feeds}
    calls = []
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.article_catalog.read_article_catalog_states",
        lambda aliases: calls.append(aliases) or {},
    )

    assert _article_states(sources, feeds) == {}
    assert len(calls) == 1
    assert len(calls[0]) == 75


def test_article_acquisition_reads_rolled_off_catalog_entry(monkeypatch) -> None:
    entry = _entry("rolled-off", "2026-08-31T12:00:00+03:00")
    reference = _source(entry, event_id=_EVENT_ID)
    source = CatalogedFeedEntry(**reference.model_dump(), entry=entry)
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.read_cataloged_feed_entry_references",
        lambda **_kwargs: (reference,),
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.article_catalog.read_article_catalog_states",
        lambda _aliases: {},
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.feed_catalog.read_successful_feed_counts",
        lambda _days: {},
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.materialize_cataloged_feed_entries",
        lambda _references, _registry: (source,),
    )
    monkeypatch.setattr("romanian_news.articles.acquisition.create_news_session", object)
    monkeypatch.setattr(
        "romanian_news.articles.acquisition._fetch_article_page",
        lambda _session, _work: _ArticlePage(
            content=None,
            url=None,
            latency_ms=0,
            retrieval_error=None,
        ),
    )

    result = acquire_articles(
        feed_registry(),
        now=datetime.fromisoformat("2026-09-01T12:00:00+03:00"),
        start_at=datetime.fromisoformat("2026-08-31T00:00:00+03:00"),
        end_at=datetime.fromisoformat("2026-09-01T12:00:00+03:00"),
        limit=1,
    )

    assert result.captures[0].source == source


def test_article_acquisition_materializes_only_selected_references(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-09-01T12:00:00+03:00")
    entries = tuple(
        _entry(
            f"historical-{index}",
            datetime.fromisoformat("2026-08-31T00:00:00+03:00").replace(minute=index).isoformat(),
        )
        for index in range(51)
    )
    references = tuple(_source(entry) for entry in entries)
    entries_by_url = {str(entry.url): entry for entry in entries}
    materialized = []

    def materialize(selected, _registry):
        materialized.extend(selected)
        return tuple(
            CatalogedFeedEntry(
                **reference.model_dump(),
                entry=entries_by_url[str(reference.url)],
            )
            for reference in selected
        )

    monkeypatch.setattr(
        "romanian_news.articles.acquisition.read_cataloged_feed_entry_references",
        lambda **_kwargs: references,
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.article_catalog.read_article_catalog_states",
        lambda _aliases: {},
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.feed_catalog.read_successful_feed_counts",
        lambda _days: {},
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.materialize_cataloged_feed_entries", materialize
    )
    monkeypatch.setattr("romanian_news.articles.acquisition.create_news_session", object)
    monkeypatch.setattr(
        "romanian_news.articles.acquisition._fetch_article_page",
        lambda _session, _work: _ArticlePage(
            content=None,
            url=None,
            latency_ms=0,
            retrieval_error=None,
        ),
    )

    result = acquire_articles(
        feed_registry(),
        now=now,
        start_at=datetime.fromisoformat("2026-08-31T00:00:00+03:00"),
        end_at=now,
        limit=50,
    )

    assert len(materialized) == 50
    assert len(result.captures) == 50
    assert references[-1] not in materialized


def test_exact_batch_skips_work_completed_before_retry(monkeypatch) -> None:
    source = _source(_entry("completed", "2026-09-02T08:00:00+03:00"))
    work = ArticleWorkItem(source=source, lane=ArticleWorkLane.LIVE_UNSEEN)
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.article_work_is_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.materialize_cataloged_feed_entries",
        lambda *_args: (_ for _ in ()).throw(AssertionError("materialized completed work")),
    )

    result = acquire_article_batch_item(work, feed_registry())

    assert isinstance(result, ArticleBatchSkip)
    assert result.event_id == source.event_id


def test_exact_batch_load_does_not_replan_or_replace_items(monkeypatch) -> None:
    source = _source(_entry("exact", "2026-09-02T08:00:00+03:00"))
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.read_cataloged_feed_entry_reference",
        lambda _event_id: source,
    )
    monkeypatch.setattr("romanian_news.articles.acquisition._article_states", lambda *_args: {})
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.read_article_attempt_states",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition._plan_article_work",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("replanned batch")),
    )

    work = load_exact_article_work(
        (source.event_id,),
        feed_registry(),
        implementation_ref="git:test",
        now=datetime.fromisoformat("2026-09-02T12:00:00+03:00"),
        start_at=datetime.fromisoformat("2026-09-02T00:00:00+03:00"),
        end_at=datetime.fromisoformat("2026-09-03T00:00:00+03:00"),
        revalidate_before=None,
    )

    assert tuple(item.source.event_id for item in work) == (source.event_id,)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ResearchCatalogError("catalog offline"), "infrastructure"),
        (ResearchObjectUnavailable("object offline"), "infrastructure"),
        (ResearchObjectIntegrityError("object corrupt"), "deterministic"),
        (ValueError("invalid snapshot"), "deterministic"),
    ],
)
def test_exact_batch_converts_snapshot_failures_to_typed_attempts(
    monkeypatch,
    error: Exception,
    expected: str,
) -> None:
    source = _source(_entry("failed", "2026-09-02T08:00:00+03:00"))
    work = ArticleWorkItem(source=source, lane=ArticleWorkLane.LIVE_UNSEEN)
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.article_work_is_complete",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.materialize_cataloged_feed_entries",
        lambda *_args: (_ for _ in ()).throw(error),
    )

    result = acquire_article_batch_item(work, feed_registry())

    assert isinstance(result, ArticleAcquisitionFailure)
    assert result.kind.value == expected
    assert str(source.url) in result.message


def test_exact_batch_does_not_hide_programming_errors(monkeypatch) -> None:
    source = _source(_entry("bug", "2026-09-02T08:00:00+03:00"))
    work = ArticleWorkItem(source=source, lane=ArticleWorkLane.LIVE_UNSEEN)
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.article_work_is_complete",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.materialize_cataloged_feed_entries",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("bug")),
    )

    with pytest.raises(RuntimeError, match="bug"):
        acquire_article_batch_item(work, feed_registry())


def test_article_acquisition_fetches_article_pages_concurrently(monkeypatch) -> None:
    entries = (
        _entry("first", "2026-09-02T08:00:00+03:00"),
        _entry("second", "2026-09-02T08:01:00+03:00"),
    )
    references = tuple(_source(entry) for entry in entries)
    sources = tuple(
        CatalogedFeedEntry(**reference.model_dump(), entry=entry)
        for reference, entry in zip(references, entries, strict=True)
    )
    barrier = threading.Barrier(2)

    def fetch(_session, _work):
        barrier.wait(timeout=1)
        return _ArticlePage(
            content=None,
            url=None,
            latency_ms=0,
            retrieval_error=None,
        )

    monkeypatch.setattr(
        "romanian_news.articles.acquisition.read_cataloged_feed_entry_references",
        lambda **_kwargs: references,
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.article_catalog.read_article_catalog_states",
        lambda _aliases: {},
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.materialize_cataloged_feed_entries",
        lambda _references, _registry: sources,
    )
    monkeypatch.setattr("romanian_news.articles.acquisition.create_news_session", object)
    monkeypatch.setattr("romanian_news.articles.acquisition._fetch_article_page", fetch)

    result = acquire_articles(
        feed_registry(),
        now=datetime.fromisoformat("2026-09-02T12:00:00+03:00"),
        limit=2,
    )

    assert [capture.source.source_id for capture in result.captures] == ["first", "second"]


def test_article_plan_treats_quarantine_as_terminal_and_deferment_as_remaining(
    monkeypatch,
) -> None:
    now = datetime.fromisoformat("2026-09-02T12:00:00+03:00")
    quarantined = _source(_entry("quarantined", "2026-09-02T08:00:00+03:00"))
    deferred = _source(_entry("deferred", "2026-09-02T09:00:00+03:00"))
    states = {
        quarantined.event_id: ArticleAttemptState(
            event_id=quarantined.event_id,
            implementation_ref="git:test",
            work_generation="d" * 64,
            retry_at=now,
            deterministic_fingerprint="c" * 64,
            unchanged_deterministic_attempts=3,
        ),
        deferred.event_id: ArticleAttemptState(
            event_id=deferred.event_id,
            implementation_ref="git:test",
            work_generation="e" * 64,
            retry_at=now + timedelta(minutes=5),
            deterministic_fingerprint=None,
            unchanged_deterministic_attempts=0,
        ),
    }
    monkeypatch.setattr("romanian_news.articles.acquisition._article_states", lambda *_args: {})

    plan = _plan_article_work(
        (quarantined, deferred),
        {feed.id: feed for feed in feed_registry().feeds},
        now=now,
        limit=10,
        revalidate_before=None,
        attempt_states=states,
    )

    assert plan.selected == ()
    assert plan.remaining_entries == 1
    assert plan.deferred_event_ids == (deferred.event_id,)
    assert plan.quarantined_event_ids == (quarantined.event_id,)


def test_article_plan_reserves_live_and_historical_capacity(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-09-01T12:00:00+03:00")
    sources = tuple(
        _source(_entry(f"live-{index}", (now - timedelta(minutes=index)).isoformat()))
        for index in range(45)
    ) + tuple(
        _source(
            _entry(
                f"historical-{index}",
                datetime.fromisoformat("2026-08-31T00:00:00+03:00")
                .replace(minute=index)
                .isoformat(),
            )
        )
        for index in range(30)
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.article_catalog.read_article_catalog_states",
        lambda _aliases: {},
    )

    plan = _plan_article_work(
        sources,
        {feed.id: feed for feed in feed_registry().feeds},
        now=now,
        limit=50,
        revalidate_before=None,
    )

    assert sum(item.lane == ArticleWorkLane.LIVE_UNSEEN for item in plan.selected) == 25
    assert sum(item.lane == ArticleWorkLane.HISTORICAL_UNSEEN for item in plan.selected) == 25
    historical = [
        item.source.published_at
        for item in plan.selected
        if item.lane == ArticleWorkLane.HISTORICAL_UNSEEN
    ]
    assert historical == sorted(historical)


def test_article_plan_borrows_unused_live_capacity(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-09-01T12:00:00+03:00")
    sources = tuple(
        _source(
            _entry(
                f"historical-{index}",
                datetime.fromisoformat("2026-08-31T00:00:00+03:00")
                .replace(minute=index)
                .isoformat(),
            )
        )
        for index in range(50)
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.article_catalog.read_article_catalog_states",
        lambda _aliases: {},
    )

    plan = _plan_article_work(
        sources,
        {feed.id: feed for feed in feed_registry().feeds},
        now=now,
        limit=50,
        revalidate_before=None,
    )

    assert len(plan.selected) == 50
    assert {item.lane for item in plan.selected} == {ArticleWorkLane.HISTORICAL_UNSEEN}


def test_article_plan_blocks_updates_and_revalidation_until_prior_work_fits(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-09-01T12:00:00+03:00")
    unseen = _source(_entry("unseen", "2026-08-31T09:00:00+03:00"))
    update = _source(_entry("update", "2026-08-31T10:00:00+03:00"))
    revalidation = _source(_entry("revalidation", "2026-08-31T11:00:00+03:00"))
    states = {
        "url:https://hotnews.ro/update": ArticleCatalogState(
            alias_key="url:https://hotnews.ro/update",
            published_at=datetime.fromisoformat("2026-08-31T06:00:00+00:00"),
            source_updated_at=None,
            captured_at=datetime.fromisoformat("2026-08-31T07:00:00+00:00"),
        ),
        "url:https://hotnews.ro/revalidation": ArticleCatalogState(
            alias_key="url:https://hotnews.ro/revalidation",
            published_at=revalidation.published_at,
            source_updated_at=None,
            captured_at=datetime.fromisoformat("2026-08-30T07:00:00+00:00"),
        ),
    }
    monkeypatch.setattr("romanian_news.articles.acquisition._article_states", lambda *_: states)
    feeds = {feed.id: feed for feed in feed_registry().feeds}

    one = _plan_article_work(
        (unseen, update, revalidation),
        feeds,
        now=now,
        limit=1,
        revalidate_before=datetime.fromisoformat("2026-08-31T00:00:00+00:00"),
    )
    two = _plan_article_work(
        (unseen, update, revalidation),
        feeds,
        now=now,
        limit=2,
        revalidate_before=datetime.fromisoformat("2026-08-31T00:00:00+00:00"),
    )
    three = _plan_article_work(
        (unseen, update, revalidation),
        feeds,
        now=now,
        limit=3,
        revalidate_before=datetime.fromisoformat("2026-08-31T00:00:00+00:00"),
    )

    assert [item.lane for item in one.selected] == [ArticleWorkLane.HISTORICAL_UNSEEN]
    assert [item.lane for item in two.selected] == [
        ArticleWorkLane.HISTORICAL_UNSEEN,
        ArticleWorkLane.SOURCE_UPDATE,
    ]
    assert [item.lane for item in three.selected] == [
        ArticleWorkLane.HISTORICAL_UNSEEN,
        ArticleWorkLane.SOURCE_UPDATE,
        ArticleWorkLane.REVALIDATION,
    ]


def test_historical_plan_prioritizes_newest_day_and_advances_oldest_day(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-09-01T12:00:00+03:00")
    sources = (
        _source(_entry("oldest", "2026-08-20T18:00:00+03:00")),
        _source(_entry("middle", "2026-08-25T20:00:00+03:00")),
        _source(_entry("recent-1", "2026-08-31T08:00:00+03:00")),
        _source(_entry("recent-2", "2026-08-31T09:00:00+03:00")),
        _source(_entry("recent-3", "2026-08-31T10:00:00+03:00")),
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.article_catalog.read_article_catalog_states",
        lambda _aliases: {},
    )

    plan = _plan_article_work(
        tuple(reversed(sources)),
        {feed.id: feed for feed in feed_registry().feeds},
        now=now,
        limit=3,
        revalidate_before=None,
    )

    assert [item.source.source_id for item in plan.selected] == [
        "recent-1",
        "recent-2",
        "oldest",
    ]


def test_historical_reservation_allocates_eighty_percent_to_newest_day(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-09-01T12:00:00+03:00")
    sources = (
        tuple(
            _source(_entry(f"recent-{index}", f"2026-08-31T{index:02d}:00:00+03:00"))
            for index in range(10)
        )
        + tuple(
            _source(_entry(f"oldest-{index}", f"2026-08-20T{index:02d}:00:00+03:00"))
            for index in range(3)
        )
        + (_source(_entry("middle", "2026-08-25T12:00:00+03:00")),)
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.article_catalog.read_article_catalog_states",
        lambda _aliases: {},
    )

    plan = _plan_article_work(
        sources,
        {feed.id: feed for feed in feed_registry().feeds},
        now=now,
        limit=10,
        revalidate_before=None,
    )

    assert [item.source.source_id for item in plan.selected] == [
        *(f"recent-{index}" for index in range(8)),
        "oldest-0",
        "oldest-1",
    ]


def test_historical_plan_round_robins_feeds_within_each_day(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-09-01T12:00:00+03:00")
    day = "2026-08-31"
    sources = (
        _source(_entry("hotnews-1", f"{day}T08:00:00+03:00")),
        _source(_entry("hotnews-2", f"{day}T09:00:00+03:00")),
        _source(
            _entry(
                "adevarul-1",
                f"{day}T10:00:00+03:00",
                feed_id="adevarul",
                host="adevarul.ro",
            )
        ),
        _source(
            _entry(
                "adevarul-2",
                f"{day}T11:00:00+03:00",
                feed_id="adevarul",
                host="adevarul.ro",
            )
        ),
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.article_catalog.read_article_catalog_states",
        lambda _aliases: {},
    )

    plan = _plan_article_work(
        tuple(reversed(sources)),
        {feed.id: feed for feed in feed_registry().feeds},
        now=now,
        limit=4,
        revalidate_before=None,
    )

    assert [item.source.source_id for item in plan.selected] == [
        "adevarul-1",
        "hotnews-1",
        "adevarul-2",
        "hotnews-2",
    ]


def test_article_download_interrupts_a_response_that_never_yields(monkeypatch) -> None:
    work = _materialized_work("blocked")
    handlers = {"current": signal.SIG_IGN}
    timers = []

    class BlockingResponse(_StreamingResponse):
        def iter_content(self, chunk_size: int):
            assert chunk_size > 0
            handler = handlers["current"]
            assert callable(handler)
            handler(signal.SIGALRM, None)
            yield b"unreachable"

    response = BlockingResponse((), url="https://hotnews.ro/final")
    session = requests.Session()
    monkeypatch.setattr(session, "get", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.signal.getsignal",
        lambda _signal_number: signal.SIG_IGN,
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.signal.getitimer",
        lambda _timer: (3.0, 1.0),
    )

    def set_handler(_signal_number, handler):
        handlers["current"] = handler

    def set_timer(timer, seconds, interval=0.0):
        timers.append((timer, seconds, interval))

    monkeypatch.setattr("romanian_news.articles.acquisition.signal.signal", set_handler)
    monkeypatch.setattr("romanian_news.articles.acquisition.signal.setitimer", set_timer)
    monkeypatch.setattr("romanian_news.articles.acquisition.time.monotonic", lambda: 0.0)

    page = _fetch_article_page(session, work)

    assert page.must_fail
    assert page.failure_kind is not None
    assert page.failure_kind.value == "infrastructure"
    assert handlers["current"] == signal.SIG_IGN
    assert timers[-1] == (signal.ITIMER_REAL, 3.0, 1.0)
    assert response.closed


def test_article_download_stops_a_trickling_response_at_total_deadline(monkeypatch) -> None:
    work = _materialized_work("trickle")
    response = _StreamingResponse((b"a", b"b", b"c"), url="https://hotnews.ro/final")
    session = requests.Session()
    monkeypatch.setattr(session, "get", lambda *_args, **_kwargs: response)
    clock = iter((0.0, 7.0, 14.0, 21.0, 22.0))
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.time.monotonic",
        lambda: next(clock),
    )

    page = _fetch_article_page(session, work)

    assert page.content is None
    assert page.url == "https://hotnews.ro/final"
    assert page.failure_kind is not None
    assert page.failure_kind.value == "infrastructure"
    assert page.must_fail
    assert "exceeded 20 seconds" in str(page.retrieval_error)
    capture, failure = _article_outcome(
        work,
        {feed.id: feed for feed in feed_registry().feeds},
        page,
    )
    assert capture is None
    assert failure is not None
    assert failure.kind.value == "infrastructure"
    assert response.closed


def test_article_download_rejects_oversized_stream_without_buffering_it(monkeypatch) -> None:
    work = _materialized_work("oversized")
    response = _StreamingResponse((b"ab", b"cd"), url="https://hotnews.ro/final")
    session = requests.Session()
    monkeypatch.setattr(session, "get", lambda *_args, **_kwargs: response)
    monkeypatch.setattr("romanian_news.articles.acquisition._ARTICLE_MAX_RESPONSE_BYTES", 3)
    monkeypatch.setattr("romanian_news.articles.acquisition.time.monotonic", lambda: 0.0)

    page = _fetch_article_page(session, work)

    assert page.content is None
    assert page.url == "https://hotnews.ro/final"
    assert page.failure_kind is not None
    assert page.failure_kind.value == "deterministic"
    assert page.must_fail
    assert "exceeds 3 bytes" in str(page.retrieval_error)
    capture, failure = _article_outcome(
        work,
        {feed.id: feed for feed in feed_registry().feeds},
        page,
    )
    assert capture is None
    assert failure is not None
    assert failure.kind.value == "deterministic"
    assert response.closed


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (404, "deterministic"),
        (429, "infrastructure"),
        (503, "infrastructure"),
    ],
)
def test_article_request_failures_are_classified_at_the_http_boundary(
    status: int,
    expected: str,
) -> None:
    response = requests.Response()
    response.status_code = status
    error = requests.HTTPError(str(status), response=response)

    assert classify_article_request_failure(error).value == expected


def test_article_connection_failures_are_infrastructure() -> None:
    assert (
        classify_article_request_failure(requests.ConnectionError("offline")).value
        == "infrastructure"
    )


def test_article_source_selection_excludes_entries_at_the_end_boundary() -> None:
    end_at = datetime.fromisoformat("2026-09-01T12:00:00+03:00")
    future = _source(_entry("future", end_at.isoformat()))

    selected, skipped = _select_article_sources(
        (future,),
        {feed.id: feed for feed in feed_registry().feeds},
        None,
        end_at,
    )

    assert selected == ()
    assert skipped == 1


def test_source_coverage_uses_successful_observations_not_published_entries(monkeypatch) -> None:
    days = (date(2026, 8, 31), date(2026, 9, 1))
    registry = feed_registry()
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.feed_catalog.read_successful_feed_counts",
        lambda _days: {
            date(2026, 8, 31): len(registry.feeds),
            date(2026, 9, 1): len(registry.feeds) - 1,
        },
    )

    assert _covered_news_days(registry, days) == (date(2026, 8, 31),)


class _StreamingResponse:
    def __init__(self, chunks: tuple[bytes, ...], *, url: str) -> None:
        self._chunks = chunks
        self.url = url
        self.headers = {}
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        assert chunk_size > 0
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


def _materialized_work(source_id: str) -> MaterializedArticleWorkItem:
    entry = _entry(source_id, "2026-09-02T08:00:00+03:00")
    reference = _source(entry)
    return MaterializedArticleWorkItem(
        source=CatalogedFeedEntry(**reference.model_dump(), entry=entry),
        lane=ArticleWorkLane.LIVE_UNSEEN,
    )


def _source(
    entry: FeedEntry,
    *,
    event_id: str | None = None,
) -> CatalogedFeedEntryReference:
    event_id = event_id or hashlib.sha256(str(entry.url).encode()).hexdigest()
    return CatalogedFeedEntryReference(
        event_id=event_id,
        dlt_load_id="load-1",
        registry_version_id="c" * 64,
        feed_snapshot_version_id=_VERSION_ID,
        observed_at=entry.published_at + timedelta(minutes=5),
        feed_id=entry.feed_id,
        source_id=entry.source_id,
        url=entry.url,
        published_at=entry.published_at,
        source_updated_at=entry.source_updated_at,
    )


def _entry(
    source_id: str,
    published_at: str,
    *,
    feed_id: str = "hotnews",
    host: str = "hotnews.ro",
) -> FeedEntry:
    return FeedEntry(
        feed_id=feed_id,
        source_id=source_id,
        url=HttpUrl(f"https://{host}/{source_id}"),
        title=f"Titlu {source_id}",
        summary="Rezumat suficient de lung pentru extragerea articolului din flux. " * 3,
        feed_content=None,
        published_at=datetime.fromisoformat(published_at),
        source_updated_at=None,
        author=None,
    )
