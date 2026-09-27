CREATE TABLE news_archive_article_captures (
    id TEXT PRIMARY KEY CHECK (id ~ '^[0-9a-f]{64}$'),
    observation_id TEXT NOT NULL,
    discovered_url TEXT NOT NULL,
    final_url TEXT NOT NULL,
    capture_artifact_version_id TEXT NOT NULL UNIQUE REFERENCES artifact_versions(id),
    page_sha256 TEXT NOT NULL CHECK (page_sha256 ~ '^[0-9a-f]{64}$'),
    fetched_at TIMESTAMPTZ NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    modified_at TIMESTAMPTZ,
    publication_evidence TEXT NOT NULL,
    FOREIGN KEY (observation_id, discovered_url)
        REFERENCES news_archive_sitemap_entries(observation_id, canonical_url),
    CHECK (fetched_at >= published_at)
);

CREATE INDEX news_archive_article_captures_day
    ON news_archive_article_captures(published_at, observation_id);

CREATE TRIGGER news_archive_article_captures_reject_updates
BEFORE UPDATE ON news_archive_article_captures
FOR EACH ROW EXECUTE FUNCTION reject_archive_sitemap_mutation();

CREATE TRIGGER news_archive_article_captures_reject_deletes
BEFORE DELETE ON news_archive_article_captures
FOR EACH ROW EXECUTE FUNCTION reject_archive_sitemap_mutation();

ALTER TABLE news_article_versions
    ALTER COLUMN feed_snapshot_version_id DROP NOT NULL,
    ADD COLUMN archive_capture_id TEXT REFERENCES news_archive_article_captures(id),
    ADD CONSTRAINT news_article_versions_exact_source
        CHECK ((feed_snapshot_version_id IS NULL) <> (archive_capture_id IS NULL));
