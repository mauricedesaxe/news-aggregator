CREATE TABLE news_archive_page_checks (
    id TEXT PRIMARY KEY CHECK (id ~ '^[0-9a-f]{64}$'),
    observation_id TEXT NOT NULL REFERENCES news_archive_sitemap_observations(id),
    outlet_id TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    final_url TEXT,
    fetched_at TIMESTAMPTZ NOT NULL,
    page_sha256 TEXT CHECK (page_sha256 IS NULL OR page_sha256 ~ '^[0-9a-f]{64}$'),
    title TEXT,
    published_at TIMESTAMPTZ,
    modified_at TIMESTAMPTZ,
    publication_evidence TEXT,
    rejection TEXT,
    status TEXT NOT NULL CHECK (status IN ('accepted', 'rejected', 'retryable')),
    CHECK ((status = 'accepted') = (published_at IS NOT NULL)),
    CHECK (status != 'accepted' OR (page_sha256 IS NOT NULL AND title IS NOT NULL)),
    UNIQUE (observation_id, canonical_url, page_sha256)
);

CREATE INDEX news_archive_page_checks_candidate
    ON news_archive_page_checks(outlet_id, canonical_url, status);

CREATE TRIGGER news_archive_page_checks_reject_updates
BEFORE UPDATE ON news_archive_page_checks
FOR EACH ROW EXECUTE FUNCTION reject_archive_sitemap_mutation();

CREATE TRIGGER news_archive_page_checks_reject_deletes
BEFORE DELETE ON news_archive_page_checks
FOR EACH ROW EXECUTE FUNCTION reject_archive_sitemap_mutation();
