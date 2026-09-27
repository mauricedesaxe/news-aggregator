CREATE TABLE news_archive_sitemap_observations (
    id TEXT PRIMARY KEY CHECK (id ~ '^[0-9a-f]{64}$'),
    outlet_id TEXT NOT NULL,
    sitemap_url TEXT NOT NULL,
    content_sha256 TEXT NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    fetched_at TIMESTAMPTZ NOT NULL,
    entry_count INTEGER NOT NULL CHECK (entry_count >= 0),
    UNIQUE (outlet_id, sitemap_url, content_sha256)
);

CREATE TABLE news_archive_sitemap_entries (
    observation_id TEXT NOT NULL REFERENCES news_archive_sitemap_observations(id),
    canonical_url TEXT NOT NULL,
    source_url TEXT NOT NULL,
    lastmod_hint TEXT,
    PRIMARY KEY (observation_id, canonical_url)
);

CREATE INDEX news_archive_sitemap_entries_canonical_url
    ON news_archive_sitemap_entries(canonical_url);

CREATE FUNCTION reject_archive_sitemap_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'archive sitemap evidence is immutable' USING ERRCODE = '23000';
END;
$$;

CREATE TRIGGER news_archive_sitemap_observations_reject_updates
BEFORE UPDATE ON news_archive_sitemap_observations
FOR EACH ROW EXECUTE FUNCTION reject_archive_sitemap_mutation();

CREATE TRIGGER news_archive_sitemap_observations_reject_deletes
BEFORE DELETE ON news_archive_sitemap_observations
FOR EACH ROW EXECUTE FUNCTION reject_archive_sitemap_mutation();

CREATE TRIGGER news_archive_sitemap_entries_reject_updates
BEFORE UPDATE ON news_archive_sitemap_entries
FOR EACH ROW EXECUTE FUNCTION reject_archive_sitemap_mutation();

CREATE TRIGGER news_archive_sitemap_entries_reject_deletes
BEFORE DELETE ON news_archive_sitemap_entries
FOR EACH ROW EXECUTE FUNCTION reject_archive_sitemap_mutation();
