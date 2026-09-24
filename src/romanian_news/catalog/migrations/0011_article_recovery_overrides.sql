CREATE TABLE news_article_recovery_overrides (
    recovery_id TEXT PRIMARY KEY CHECK (recovery_id ~ '^[0-9a-f]{64}$'),
    recovery_sequence BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    event_id TEXT NOT NULL REFERENCES news_feed_entry_event_versions(event_id),
    base_work_generation TEXT NOT NULL CHECK (base_work_generation ~ '^[0-9a-f]{64}$'),
    expected_work_generation TEXT NOT NULL CHECK (expected_work_generation ~ '^[0-9a-f]{64}$'),
    requested_by TEXT NOT NULL CHECK (length(trim(requested_by)) > 0),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    requested_at TIMESTAMPTZ NOT NULL,
    UNIQUE (event_id, base_work_generation, expected_work_generation)
);

CREATE INDEX news_article_recovery_overrides_by_event_generation_sequence
    ON news_article_recovery_overrides(
        event_id, base_work_generation, recovery_sequence DESC
    );

CREATE TRIGGER news_article_recovery_overrides_reject_conflicting_inserts
BEFORE INSERT ON news_article_recovery_overrides
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('recovery_id,event_id,base_work_generation,expected_work_generation,requested_by,reason,requested_at', 'recovery_id', 'event_id,base_work_generation,expected_work_generation');

CREATE TRIGGER news_article_recovery_overrides_reject_updates
BEFORE UPDATE ON news_article_recovery_overrides
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_article_recovery_overrides_reject_deletes
BEFORE DELETE ON news_article_recovery_overrides
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();
