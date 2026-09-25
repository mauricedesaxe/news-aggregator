CREATE TABLE video_digest_h3_reference_packs (
    pack_id TEXT PRIMARY KEY CHECK (pack_id ~ '^[0-9a-f]{64}$'),
    manifest_artifact_version_id TEXT NOT NULL UNIQUE REFERENCES artifact_versions(id),
    approval_ref TEXT NOT NULL CHECK (length(trim(approval_ref)) > 0),
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE video_digest_h3_reference_media (
    pack_id TEXT NOT NULL REFERENCES video_digest_h3_reference_packs(pack_id),
    role TEXT NOT NULL CHECK (role IN ('video', 'audio')),
    position BIGINT NOT NULL CHECK (position >= 0),
    artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    PRIMARY KEY (pack_id, role, position)
);

CREATE TRIGGER video_digest_h3_reference_packs_reject_updates
BEFORE UPDATE ON video_digest_h3_reference_packs
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_h3_reference_packs_reject_deletes
BEFORE DELETE ON video_digest_h3_reference_packs
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_h3_reference_media_reject_updates
BEFORE UPDATE ON video_digest_h3_reference_media
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_h3_reference_media_reject_deletes
BEFORE DELETE ON video_digest_h3_reference_media
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();
