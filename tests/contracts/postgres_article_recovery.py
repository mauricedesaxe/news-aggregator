from __future__ import annotations

from datetime import UTC, datetime

import pytest

from romanian_news.articles import acquisition
from romanian_news.articles.models import ArticleAcquisitionFailure, ArticleFailureKind
from romanian_news.articles.recovery import (
    article_work_generation,
    read_article_recovery_view,
    record_article_failure_attempts,
)
from romanian_news.catalog.articles import read_article_recovery_overrides
from romanian_news.feeds.registry import feed_registry
from tests.postgres_catalog import postgres_catalog_fixture

postgres_catalog = postgres_catalog_fixture("article_recovery")
EVENT_ID = "a" * 64
SECOND_EVENT_ID = "b" * 64
REGISTRY_VERSION_ID = "1" * 64
FEED_SNAPSHOT_VERSION_ID = "2" * 64
FINGERPRINT = "f" * 64
PUBLISHED_AT = datetime(2026, 9, 25, 6, tzinfo=UTC)
REQUESTED_AT = datetime(2026, 9, 25, 12, tzinfo=UTC)


def _seed_cataloged_feed_entry(catalog, event_id: str, path: str) -> None:
    catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
        "visibility, created_at) VALUES ('news:feed:hotnews', 'news_feed', 'HotNews', "
        "'source', 'current', 'private', %s) ON CONFLICT DO NOTHING",
        (PUBLISHED_AT,),
    )
    catalog.execute(
        "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
        "created_at) VALUES (%s, 'news:feed:hotnews', 1, 'digest', %s) "
        "ON CONFLICT DO NOTHING",
        (FEED_SNAPSHOT_VERSION_ID, PUBLISHED_AT),
    )
    catalog.execute(
        "INSERT INTO news_dlt_loads (load_id, artifact_version_id, registered_at) "
        "VALUES ('load-1', %s, %s) ON CONFLICT DO NOTHING",
        (FEED_SNAPSHOT_VERSION_ID, PUBLISHED_AT),
    )
    catalog.execute(
        "INSERT INTO news_feed_entry_events (event_id, dlt_load_id, registry_version_id, "
        "feed_snapshot_version_id, feed_id, source_id, original_url, published_at, "
        "source_updated_at, observed_at) VALUES (%s, 'load-1', %s, %s, "
        "'hotnews', 'source-1', %s, %s, NULL, %s)",
        (
            event_id,
            REGISTRY_VERSION_ID,
            FEED_SNAPSHOT_VERSION_ID,
            f"https://hotnews.ro/stiri/eveniment/{path}",
            PUBLISHED_AT,
            PUBLISHED_AT,
        ),
    )
    catalog.execute(
        "INSERT INTO news_feed_entry_event_versions (event_id, version_id) VALUES (%s, %s)",
        (event_id, event_id),
    )
    catalog.execute(
        "INSERT INTO news_feed_entry_occurrences (event_id, dlt_load_id, registry_version_id, "
        "feed_snapshot_version_id, observed_at) VALUES (%s, 'load-1', %s, %s, %s)",
        (event_id, REGISTRY_VERSION_ID, FEED_SNAPSHOT_VERSION_ID, PUBLISHED_AT),
    )


def _quarantine_through_deterministic_failures(catalog, event_id: str) -> None:
    failure = ArticleAcquisitionFailure(
        event_id=event_id,
        kind=ArticleFailureKind.DETERMINISTIC,
        fingerprint=FINGERPRINT,
        message=f"hotnews https://hotnews.ro/stiri/eveniment/{event_id[:6]}: invalid article",
    )
    generation = article_work_generation(event_id, None)
    for attempt in range(3):
        record_article_failure_attempts(
            (failure,),
            work_generations={event_id: generation},
            run_id=f"quarantine-run-{event_id[:1]}-{attempt}",
            retry_number=0,
            implementation_ref="git:test",
            attempted_at=datetime(2026, 9, 25, 7, attempt, tzinfo=UTC),
        )
    assert (
        catalog.query(
            "SELECT count(*) AS count FROM news_article_failure_attempts WHERE event_id = %s",
            [event_id],
        )[0]["count"]
        == 3
    )


def _recover(event_ids, generations):
    return acquisition.recover_quarantined_article_events(
        event_ids,
        feed_registry(),
        requested_by="operator",
        reason="Parser fixed",
        requested_at=REQUESTED_AT,
        expected_work_generations=generations,
    )


def test_quarantined_article_releases_once_and_replays_the_identical_receipt(
    postgres_catalog,
) -> None:
    _seed_cataloged_feed_entry(postgres_catalog, EVENT_ID, "article-quarantine")
    _quarantine_through_deterministic_failures(postgres_catalog, EVENT_ID)
    base_generation = article_work_generation(EVENT_ID, None)

    status = acquisition.inspect_article_recovery_event(EVENT_ID, feed_registry())

    assert status.quarantined
    assert status.deterministic_fingerprint == FINGERPRINT
    assert status.work_generation == base_generation

    receipt = _recover((EVENT_ID,), {EVENT_ID: status.work_generation})

    assert receipt.released_event_ids == (EVENT_ID,)
    recorded = postgres_catalog.query("SELECT recovery_id FROM news_article_recovery_overrides")
    assert [row["recovery_id"] for row in recorded] == list(receipt.recovery_ids)
    assert _recover((EVENT_ID,), {EVENT_ID: status.work_generation}) == receipt
    assert (
        postgres_catalog.query("SELECT count(*) AS count FROM news_article_recovery_overrides")[0][
            "count"
        ]
        == 1
    )

    override = read_article_recovery_overrides((EVENT_ID,))[0]
    view = read_article_recovery_view({EVENT_ID: base_generation})

    assert view.work_generations[EVENT_ID] == override.work_generation
    assert override.work_generation != base_generation
    assert not view.attempt_states
    released = acquisition.inspect_article_recovery_event(EVENT_ID, feed_registry())
    assert not released.quarantined
    assert released.work_generation == override.work_generation
    assert released.deterministic_fingerprint is None


def test_release_with_a_wrong_base_generation_is_rejected(postgres_catalog) -> None:
    _seed_cataloged_feed_entry(postgres_catalog, EVENT_ID, "article-stale-generation")
    _quarantine_through_deterministic_failures(postgres_catalog, EVENT_ID)

    with pytest.raises(ValueError, match="generation changed"):
        _recover((EVENT_ID,), {EVENT_ID: "9" * 64})
    assert acquisition.inspect_article_recovery_event(EVENT_ID, feed_registry()).quarantined
    assert not postgres_catalog.query("SELECT recovery_id FROM news_article_recovery_overrides")


def test_release_of_an_article_that_is_not_quarantined_is_rejected(postgres_catalog) -> None:
    _seed_cataloged_feed_entry(postgres_catalog, EVENT_ID, "article-never-quarantined")

    with pytest.raises(ValueError, match="not quarantined"):
        _recover((EVENT_ID,), {EVENT_ID: article_work_generation(EVENT_ID, None)})
    assert not postgres_catalog.query("SELECT recovery_id FROM news_article_recovery_overrides")


def test_partly_recorded_recovery_replay_is_rejected(postgres_catalog) -> None:
    _seed_cataloged_feed_entry(postgres_catalog, EVENT_ID, "article-partly-first")
    _seed_cataloged_feed_entry(postgres_catalog, SECOND_EVENT_ID, "article-partly-second")
    _quarantine_through_deterministic_failures(postgres_catalog, EVENT_ID)
    _quarantine_through_deterministic_failures(postgres_catalog, SECOND_EVENT_ID)
    generations = {
        event_id: article_work_generation(event_id, None)
        for event_id in (EVENT_ID, SECOND_EVENT_ID)
    }
    _recover((EVENT_ID,), {EVENT_ID: generations[EVENT_ID]})

    with pytest.raises(ValueError, match="only partly recorded"):
        _recover((EVENT_ID, SECOND_EVENT_ID), generations)


@pytest.mark.parametrize(
    ("event_ids", "requested_by", "reason", "requested_at", "generations", "message"),
    (
        (
            (),
            "operator",
            "Parser fixed",
            REQUESTED_AT,
            {},
            "Recovery requires 1 to 10 unique event IDs",
        ),
        (
            (EVENT_ID, EVENT_ID),
            "operator",
            "Parser fixed",
            REQUESTED_AT,
            {EVENT_ID: "c" * 64},
            "Recovery requires 1 to 10 unique event IDs",
        ),
        (
            tuple(f"{index:x}" * 64 for index in range(11)),
            "operator",
            "Parser fixed",
            REQUESTED_AT,
            {f"{index:x}" * 64: "c" * 64 for index in range(11)},
            "Recovery requires 1 to 10 unique event IDs",
        ),
        (
            (EVENT_ID,),
            "  ",
            "Parser fixed",
            REQUESTED_AT,
            {EVENT_ID: "c" * 64},
            "Recovery requires a requester and reason",
        ),
        (
            (EVENT_ID,),
            "operator",
            "",
            REQUESTED_AT,
            {EVENT_ID: "c" * 64},
            "Recovery requires a requester and reason",
        ),
        (
            (EVENT_ID,),
            "operator",
            "Parser fixed",
            datetime(2026, 9, 25, 12),
            {EVENT_ID: "c" * 64},
            "Recovery time must include a UTC offset",
        ),
        (
            (EVENT_ID,),
            "operator",
            "Parser fixed",
            REQUESTED_AT,
            {},
            "Recovery needs one expected generation per event",
        ),
        (
            (EVENT_ID,),
            "operator",
            "Parser fixed",
            REQUESTED_AT,
            {EVENT_ID: "c" * 64, SECOND_EVENT_ID: "d" * 64},
            "Recovery needs one expected generation per event",
        ),
    ),
)
def test_invalid_recovery_requests_are_rejected_before_any_write(
    event_ids, requested_by, reason, requested_at, generations, message
) -> None:
    with pytest.raises(ValueError, match=message):
        acquisition._validate_recovery_request(
            event_ids, requested_by, reason, requested_at, generations
        )
