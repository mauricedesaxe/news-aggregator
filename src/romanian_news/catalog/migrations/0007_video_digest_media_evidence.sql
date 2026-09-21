DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM video_digest_generation_requests WHERE stage = 'accepted'
    ) OR EXISTS (
        SELECT 1 FROM video_digest_editions
        WHERE assembled_video_artifact_version_id IS NOT NULL
    ) THEN
        RAISE EXCEPTION 'video digest media evidence migration requires no legacy accepted or assembled outputs'
            USING ERRCODE = '23000';
    END IF;
END;
$$;

ALTER TABLE video_digest_generation_requests
    ADD COLUMN validation_evidence_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            validation_evidence_artifact_version_id IS NULL
            OR validation_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'
        );

ALTER TABLE video_digest_editions
    ADD COLUMN assembly_manifest_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            assembly_manifest_artifact_version_id IS NULL
            OR assembly_manifest_artifact_version_id ~ '^[0-9a-f]{64}$'
        );

CREATE FUNCTION protect_video_digest_media_evidence() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.validation_evidence_artifact_version_id IS NOT NULL THEN
            RAISE EXCEPTION 'video digest generation must start without validation evidence'
                USING ERRCODE = '23000';
        END IF;
        RETURN NEW;
    END IF;

    IF OLD.validation_evidence_artifact_version_id IS NULL
       AND NEW.validation_evidence_artifact_version_id IS NOT NULL
       AND NOT (OLD.stage = 'processing' AND NEW.stage = 'accepted') THEN
        RAISE EXCEPTION 'video digest validation evidence requires acceptance'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.validation_evidence_artifact_version_id IS NOT NULL
       AND NEW.validation_evidence_artifact_version_id
           IS DISTINCT FROM OLD.validation_evidence_artifact_version_id THEN
        RAISE EXCEPTION 'video digest validation evidence is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.stage <> 'accepted' AND NEW.stage = 'accepted'
       AND NEW.validation_evidence_artifact_version_id IS NULL THEN
        RAISE EXCEPTION 'accepted video digest generation requires validation evidence'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION protect_video_digest_assembly_manifest() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.assembly_manifest_artifact_version_id IS NOT NULL THEN
            RAISE EXCEPTION 'video digest edition must start without assembly manifest'
                USING ERRCODE = '23000';
        END IF;
        RETURN NEW;
    END IF;

    IF OLD.assembly_manifest_artifact_version_id IS NULL
       AND NEW.assembly_manifest_artifact_version_id IS NOT NULL
       AND NOT (
           OLD.assembled_video_artifact_version_id IS NULL
           AND NEW.assembled_video_artifact_version_id IS NOT NULL
       ) THEN
        RAISE EXCEPTION 'video digest assembly manifest requires assembled video'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.assembly_manifest_artifact_version_id IS NOT NULL
       AND NEW.assembly_manifest_artifact_version_id
           IS DISTINCT FROM OLD.assembly_manifest_artifact_version_id THEN
        RAISE EXCEPTION 'video digest assembly manifest is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.assembled_video_artifact_version_id IS NULL
       AND NEW.assembled_video_artifact_version_id IS NOT NULL
       AND NEW.assembly_manifest_artifact_version_id IS NULL THEN
        RAISE EXCEPTION 'assembled video digest requires an assembly manifest'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER video_digest_generation_requests_protect_media_evidence
BEFORE INSERT OR UPDATE ON video_digest_generation_requests
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_media_evidence();

CREATE TRIGGER video_digest_editions_protect_assembly_manifest
BEFORE INSERT OR UPDATE ON video_digest_editions
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_assembly_manifest();
