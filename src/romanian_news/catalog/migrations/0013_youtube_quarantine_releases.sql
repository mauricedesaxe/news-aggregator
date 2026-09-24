ALTER TABLE youtube_videos ADD COLUMN quarantine_generation BIGINT NOT NULL DEFAULT 0
    CHECK (quarantine_generation >= 0);

UPDATE youtube_videos SET quarantine_generation = 1 WHERE state = 'quarantined';

CREATE TABLE youtube_quarantine_releases (
    request_id UUID PRIMARY KEY,
    source_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    quarantine_generation BIGINT NOT NULL CHECK (quarantine_generation > 0),
    failure_fingerprint TEXT NOT NULL,
    unchanged_failures BIGINT NOT NULL CHECK (unchanged_failures >= 3),
    last_error TEXT NOT NULL,
    requested_by TEXT NOT NULL CHECK (length(trim(requested_by)) > 0),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    requested_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (source_id, video_id) REFERENCES youtube_videos(source_id, video_id),
    UNIQUE (source_id, video_id, quarantine_generation)
);

CREATE TRIGGER youtube_quarantine_releases_reject_updates
BEFORE UPDATE ON youtube_quarantine_releases
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER youtube_quarantine_releases_reject_deletes
BEFORE DELETE ON youtube_quarantine_releases
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();
