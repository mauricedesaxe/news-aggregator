from __future__ import annotations

import hashlib

import pytest

from romanian_news.catalog.daily import accepted_relevance_version_ids
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog, postgres_catalog_fixture

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_DSN is None, reason="NEWS_TEST_POSTGRES_DSN is required"
)

postgres_catalog = postgres_catalog_fixture("relevance_acceptance")
CAPTURED_AT = "2026-10-01T06:00:00+00:00"


def _id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_accepted_relevance_version_ids_excludes_rejected_and_unrecorded_versions(
    postgres_catalog: PostgresCatalog,
) -> None:
    recorded = tuple(f"{index:064x}" for index in range(60))
    expected = frozenset(recorded[::2])
    unrecorded = tuple(_id(f"unrecorded:{index}") for index in range(5))
    version_ids = recorded + unrecorded
    assert len(version_ids) > 50
    statements: list[tuple[str, list[object]]] = []
    for index, version_id in enumerate(recorded):
        artifact_id = f"news:relevance:{index:064x}"
        statements.extend(
            (
                (
                    "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
                    "visibility, created_at) VALUES (%s, 'news_relevance', %s, 'derived', "
                    "'current', 'private', %s)",
                    [artifact_id, f"Relevance {index}", CAPTURED_AT],
                ),
                (
                    "INSERT INTO artifact_versions (id, artifact_id, schema_version, "
                    "content_digest, created_at) VALUES (%s, %s, 1, %s, %s)",
                    [version_id, artifact_id, _id(f"content:{index}"), CAPTURED_AT],
                ),
                (
                    "INSERT INTO news_relevance_versions (artifact_version_id, accepted) "
                    "VALUES (%s, %s)",
                    [version_id, 1 if index % 2 == 0 else 0],
                ),
            )
        )
    postgres_catalog.batch(statements)

    assert accepted_relevance_version_ids(version_ids) == expected


def test_accepted_relevance_version_ids_accepts_no_versions_without_querying(
    postgres_catalog: PostgresCatalog,
) -> None:
    assert accepted_relevance_version_ids(()) == frozenset()
