from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from romanian_news.catalog.articles import (
    ArticleRecoveryOverride,
    read_article_recovery_overrides,
    write_article_recovery_overrides,
)
from romanian_news.catalog_transport import ResearchCatalogError
from tests.postgres_catalog import postgres_catalog_fixture

postgres_catalog = postgres_catalog_fixture("article_recovery_overrides")

EVENT_ID = "a" * 64


def _seed_event_version(catalog) -> None:
    version_id = "e" * 64
    catalog.execute(
        "INSERT INTO artifacts "
        "(id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
        "VALUES (%s, 'test', 'Test artifact', 'test', 'active', 'private', now())",
        (f"artifact-{version_id}",),
    )
    catalog.execute(
        "INSERT INTO artifact_versions "
        "(id, artifact_id, schema_version, content_digest, created_at) "
        "VALUES (%s, %s, 1, %s, now())",
        (version_id, f"artifact-{version_id}", version_id),
    )
    catalog.execute(
        "INSERT INTO news_dlt_loads (load_id, artifact_version_id, registered_at) "
        "VALUES ('load-1', %s, now())",
        (version_id,),
    )
    catalog.execute(
        "INSERT INTO news_feed_entry_events "
        "(event_id, dlt_load_id, registry_version_id, feed_snapshot_version_id, feed_id, "
        "source_id, original_url, published_at, observed_at) "
        "VALUES (%s, 'load-1', 'registry-1', %s, 'feed-1', 'source-1', "
        "'https://example.test/article', now(), now())",
        (EVENT_ID, version_id),
    )
    catalog.execute(
        "INSERT INTO news_feed_entry_event_versions (event_id, version_id) VALUES (%s, %s)",
        (EVENT_ID, "f" * 64),
    )


def _override(**updates: object) -> ArticleRecoveryOverride:
    values: dict[str, Any] = {
        "recovery_id": "1" * 64,
        "event_id": EVENT_ID,
        "base_work_generation": "b" * 64,
        "expected_work_generation": "c" * 64,
        "requested_by": "operator",
        "reason": "Parser updated",
        "requested_at": datetime(2026, 9, 1, 3, tzinfo=UTC),
    }
    values.update(updates)
    return ArticleRecoveryOverride(**values)


def test_article_recovery_overrides_round_trip_in_written_order(postgres_catalog) -> None:
    _seed_event_version(postgres_catalog)
    first = _override()
    second = _override(
        recovery_id="2" * 64,
        expected_work_generation="d" * 64,
        requested_at=datetime.fromisoformat("2026-09-01T06:00:00+03:00"),
    )

    write_article_recovery_overrides((first, second))

    stored = read_article_recovery_overrides((EVENT_ID,))
    assert [override.recovery_id for override in stored] == [first.recovery_id, second.recovery_id]
    assert [override.recovery_sequence for override in stored] == [1, 2]
    assert stored[0].requested_at == datetime(2026, 9, 1, 3, tzinfo=UTC)
    assert stored[1].requested_at == datetime(2026, 9, 1, 3, tzinfo=UTC)
    assert stored[0].work_generation != stored[1].work_generation
    assert read_article_recovery_overrides(()) == ()


def test_article_recovery_overrides_reject_replay_conflict_and_mutation(
    postgres_catalog,
) -> None:
    _seed_event_version(postgres_catalog)
    override = _override()
    write_article_recovery_overrides((override,))

    with pytest.raises(ResearchCatalogError):
        write_article_recovery_overrides((override,))
    with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
        postgres_catalog.execute(
            "UPDATE news_article_recovery_overrides SET reason = 'rewritten'",
        )
    with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
        postgres_catalog.execute(
            "INSERT INTO news_article_recovery_overrides "
            "(recovery_id, event_id, base_work_generation, expected_work_generation, "
            "requested_by, reason, requested_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                "3" * 64,
                EVENT_ID,
                override.base_work_generation,
                override.expected_work_generation,
                override.requested_by,
                "a different reason for the same recovery",
                override.requested_at,
            ),
        )

    assert len(read_article_recovery_overrides((EVENT_ID,))) == 1
