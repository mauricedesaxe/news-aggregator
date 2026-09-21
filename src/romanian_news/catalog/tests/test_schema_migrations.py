import re

from romanian_news.catalog.schema import NEWS_CATALOG_MIGRATIONS, _expected_schema_objects


def test_news_schema_contains_its_catalog_boundaries() -> None:
    news_tables, news_triggers = _expected_schema_objects()

    assert {"artifacts", "artifact_versions", "runs", "news_schema_migrations"} <= news_tables
    assert {"artifacts_protect_identity", "runs_protect_identity"} <= news_triggers
    assert {
        "video_digest_slots",
        "video_digest_editions",
        "video_digest_stories",
        "video_digest_planning_attempts",
        "video_digest_generation_requests",
        "video_digest_publication_intents",
    } <= news_tables
    assert {
        "video_digest_editions_require_initial_state",
        "video_digest_slots_require_initial_state",
        "video_digest_slots_protect_transition",
        "video_digest_stories_require_initial_state",
        "video_digest_stories_protect_transition",
        "video_digest_generation_requests_require_initial_state",
        "video_digest_generation_requests_protect_transition",
        "video_digest_publication_intents_require_initial_state",
        "video_digest_publication_intents_require_readiness",
        "video_digest_publication_intents_protect_transition",
        "video_digest_publication_intents_protect_evidence",
        "video_digest_slots_require_terminal_evidence",
        "video_digest_planning_attempts_reject_updates",
        "video_digest_planning_attempts_reject_deletes",
        "video_digest_planning_attempts_require_sequence",
        "video_digest_editions_protect_planning_state",
        "video_digest_generation_requests_require_authorization",
    } <= news_triggers


def test_news_schema_contains_no_debt_objects() -> None:
    migration_sql = "\n".join(migration.path.read_text() for migration in NEWS_CATALOG_MIGRATIONS)

    assert "debt_" not in migration_sql
    assert "_chartly_catalog_schema" not in migration_sql


def test_news_migrations_are_ordered_and_immutable_by_identity() -> None:
    assert tuple((migration.version, migration.name) for migration in NEWS_CATALOG_MIGRATIONS) == (
        (1, "initial"),
        (2, "video_digest"),
        (3, "video_digest_generation_fences"),
        (4, "video_digest_publication_evidence"),
        (5, "video_digest_planning"),
    )
    assert tuple(migration.version for migration in NEWS_CATALOG_MIGRATIONS) == tuple(
        range(1, len(NEWS_CATALOG_MIGRATIONS) + 1)
    )
    assert len({migration.name for migration in NEWS_CATALOG_MIGRATIONS}) == len(
        NEWS_CATALOG_MIGRATIONS
    )
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", migration.sha256) for migration in NEWS_CATALOG_MIGRATIONS
    )
    assert tuple(migration.sha256 for migration in NEWS_CATALOG_MIGRATIONS) == (
        "fdb4743a172787b87e7cd40b8f4a1701c104e4df527f5d1d9622f8ba0639954a",
        "7d36b645f7af54d821b2da1d2eac5ac4da8738c26098d24db56eed2e44f79664",
        "44c9d7bca5165807bf7db40245964ff6fb4527d29771b46c0ce2aa4231874c8b",
        "8dc260a8d8739ded18a246ae759647cc1971f1c04eeb66ea92d090ea4727e0b7",
        "860fa87f9512e2b43e2f2bc7b54426c196eee566dd30e77c4514a60ec1ec9af1",
    )
