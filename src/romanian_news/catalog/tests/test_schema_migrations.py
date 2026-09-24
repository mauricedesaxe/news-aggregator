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
        "video_digest_generation_reservations",
        "video_digest_publication_intents",
        "video_digest_assembly_attempts",
        "video_digest_subtitle_attempts",
        "video_digest_publication_attempts",
        "video_digest_feedback",
        "video_digest_feedback_stories",
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
        "video_digest_generation_requests_require_order",
        "video_digest_generation_reservations_require_budget",
        "video_digest_generation_requests_require_reservations",
        "video_digest_generation_requests_protect_admission",
        "video_digest_generation_requests_protect_media_evidence",
        "video_digest_editions_protect_assembly_manifest",
        "video_digest_generation_reservations_reject_updates",
        "video_digest_generation_reservations_reject_deletes",
        "video_digest_assembly_attempts_require_sequence",
        "video_digest_assembly_attempts_reject_updates",
        "video_digest_assembly_attempts_reject_deletes",
        "video_digest_subtitle_attempts_require_sequence",
        "video_digest_subtitle_attempts_reject_updates",
        "video_digest_subtitle_attempts_reject_deletes",
        "video_digest_publication_attempts_require_sequence",
        "video_digest_publication_attempts_protect_transition",
        "video_digest_publication_attempts_reject_deletes",
        "video_digest_feedback_require_lineage",
        "video_digest_feedback_stories_require_lineage",
        "video_digest_feedback_require_completeness",
        "video_digest_feedback_reject_updates",
        "video_digest_feedback_reject_deletes",
        "video_digest_feedback_stories_reject_updates",
        "video_digest_feedback_stories_reject_deletes",
    } <= news_triggers


def test_news_schema_contains_no_debt_objects() -> None:
    migration_sql = "\n".join(migration.path.read_text() for migration in NEWS_CATALOG_MIGRATIONS)

    assert "debt_" not in migration_sql
    assert "_chartly_catalog_schema" not in migration_sql


def test_schema_verification_uses_only_applied_migrations() -> None:
    tables, triggers = _expected_schema_objects(set(range(1, 7)))

    assert "news_article_recovery_overrides" not in tables
    assert "news_article_recovery_overrides_reject_updates" not in triggers
    assert "news_article_failure_attempts" in tables


def test_news_migrations_are_ordered_and_immutable_by_identity() -> None:
    assert tuple((migration.version, migration.name) for migration in NEWS_CATALOG_MIGRATIONS) == (
        (1, "initial"),
        (2, "video_digest"),
        (3, "video_digest_generation_fences"),
        (4, "video_digest_publication_evidence"),
        (5, "video_digest_planning"),
        (6, "video_digest_generation_admission"),
        (7, "video_digest_media_evidence"),
        (8, "video_digest_orchestration"),
        (9, "video_digest_publication"),
        (10, "video_digest_feedback"),
        (11, "article_recovery_overrides"),
        (12, "daily_report_repair_requests"),
        (13, "youtube_quarantine_releases"),
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
        "66b61b4d9c6f6ce57e0afbc1e53ac736f9d829aa77af983c5386351848f63326",
        "9bb60cac645006d02990a59b761fd2c80bfa42b3f1bfe199d0d0ceb0237e6f09",
        "26cb9c5055f950c2bdb673c37b9c2d78d884560196700e2c18ca31744b37c888",
        "d8c9c938e7a83abfb4263af5280b6851694843b77524d0f84c6dc9ab55d9b4e3",
        "5f733c2e9e4e331a86eceb9fcb362ff507dbc8681a549fceea2c31b4eb328276",
        "69aa3b48f2079044fc85473aadf4c6e1745b62cd63f146c669784c0f1a08c5a4",
        "0a655f2a44f7ab03ed2dd01625abe477c029324782e20749e951e5f4a1227e00",
        "ef288ed9310aa098db525df84bb29a1ac9b1e460c13147e6a7301ca1b3bb342b",
        "6c0cb261d54f9f14053090ba54c70551ae4935d5ade3e8dee01194e64cdbb0a5",
    )
