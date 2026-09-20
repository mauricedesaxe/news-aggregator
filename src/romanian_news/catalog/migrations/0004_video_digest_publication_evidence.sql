ALTER TABLE video_digest_publication_intents
    ADD COLUMN evidence_required BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN normalized_keys_required BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN upload_evidence_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            upload_evidence_artifact_version_id IS NULL
            OR upload_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    ADD COLUMN verification_evidence_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            verification_evidence_artifact_version_id IS NULL
            OR verification_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    ADD COLUMN failure_evidence_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            failure_evidence_artifact_version_id IS NULL
            OR failure_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    ADD CONSTRAINT video_digest_publication_intents_evidence_shape CHECK (
        NOT evidence_required
        OR (stage IN ('pending', 'uploading')
         AND upload_evidence_artifact_version_id IS NULL
         AND verification_evidence_artifact_version_id IS NULL
         AND failure_evidence_artifact_version_id IS NULL)
        OR (stage = 'uploaded'
            AND upload_evidence_artifact_version_id IS NOT NULL
            AND verification_evidence_artifact_version_id IS NULL
            AND failure_evidence_artifact_version_id IS NULL)
        OR (stage IN ('verified', 'published')
            AND upload_evidence_artifact_version_id IS NOT NULL
            AND verification_evidence_artifact_version_id IS NOT NULL
            AND failure_evidence_artifact_version_id IS NULL)
        OR (stage IN ('conflict', 'failed')
            AND failure_evidence_artifact_version_id IS NOT NULL)
    );

UPDATE video_digest_publication_intents
SET evidence_required = TRUE
WHERE stage IN ('pending', 'uploading');

ALTER TABLE video_digest_publication_intents
    ALTER COLUMN evidence_required SET DEFAULT TRUE,
    ALTER COLUMN normalized_keys_required SET DEFAULT TRUE;

ALTER TABLE video_digest_publication_intents
    ADD CONSTRAINT video_digest_publication_video_key_normalized CHECK (
        NOT normalized_keys_required
        OR (expected_video_key ~ '^[A-Za-z0-9][A-Za-z0-9._/-]*$'
            AND expected_video_key !~ '(^/|(^|/)\.{1,2}(/|$)|//|/$)')
    ),
    ADD CONSTRAINT video_digest_publication_subtitle_key_normalized CHECK (
        NOT normalized_keys_required
        OR subtitle_expected_key IS NULL
        OR (subtitle_expected_key ~ '^[A-Za-z0-9][A-Za-z0-9._/-]*$'
            AND subtitle_expected_key !~ '(^/|(^|/)\.{1,2}(/|$)|//|/$)')
    );

ALTER TABLE video_digest_slots
    ADD COLUMN terminal_fence_required BOOLEAN NOT NULL DEFAULT FALSE;

UPDATE video_digest_slots
SET terminal_fence_required = TRUE
WHERE stage NOT IN ('skipped', 'failed', 'published');

ALTER TABLE video_digest_slots
    ALTER COLUMN terminal_fence_required SET DEFAULT TRUE,
    ALTER COLUMN terminal_fence_required SET NOT NULL;

ALTER TABLE video_digest_slots
    DROP CONSTRAINT video_digest_slots_terminal_fence_complete,
    ADD CONSTRAINT video_digest_slots_terminal_fence_complete CHECK (
        (stage NOT IN ('failed', 'published')
         AND terminal_lease_owner_token IS NULL
         AND terminal_lease_expires_at IS NULL
         AND terminal_claim_count IS NULL)
        OR (stage IN ('failed', 'published') AND (
            NOT terminal_fence_required
            OR (terminal_fence_required
                AND terminal_lease_owner_token IS NOT NULL
                AND length(trim(terminal_lease_owner_token)) > 0
                AND terminal_lease_expires_at IS NOT NULL
                AND terminal_claim_count IS NOT NULL)
        ))
    );

CREATE OR REPLACE FUNCTION require_video_digest_slot_initial_state() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.stage <> 'scheduled'
       OR NEW.claim_count <> 0
       OR NEW.edition_id IS NOT NULL
       OR NEW.lease_owner_token IS NOT NULL
       OR NEW.lease_expires_at IS NOT NULL
       OR NEW.skip_reason IS NOT NULL
       OR NEW.failure_evidence_artifact_version_id IS NOT NULL
       OR NOT NEW.terminal_fence_required THEN
        RAISE EXCEPTION 'video digest slot must start scheduled and unclaimed'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION require_video_digest_publication_intent_initial_state()
RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.stage <> 'pending'
       OR NEW.published_at IS NOT NULL
       OR NOT NEW.evidence_required
       OR NOT NEW.normalized_keys_required
       OR NEW.upload_evidence_artifact_version_id IS NOT NULL
       OR NEW.verification_evidence_artifact_version_id IS NOT NULL
       OR NEW.failure_evidence_artifact_version_id IS NOT NULL THEN
        RAISE EXCEPTION 'video digest publication intent must start pending'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION protect_video_digest_publication_evidence() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.evidence_required IS DISTINCT FROM OLD.evidence_required THEN
        RAISE EXCEPTION 'video digest publication evidence requirement is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.normalized_keys_required IS DISTINCT FROM OLD.normalized_keys_required THEN
        RAISE EXCEPTION 'video digest publication key requirement is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.stage IN ('conflict', 'failed')
       AND OLD.stage NOT IN ('conflict', 'failed')
       AND (NEW.upload_evidence_artifact_version_id,
            NEW.verification_evidence_artifact_version_id)
           IS DISTINCT FROM
           (OLD.upload_evidence_artifact_version_id,
            OLD.verification_evidence_artifact_version_id) THEN
        RAISE EXCEPTION 'video digest publication failure cannot introduce progress evidence'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.upload_evidence_artifact_version_id IS NOT NULL
       AND NEW.upload_evidence_artifact_version_id
           IS DISTINCT FROM OLD.upload_evidence_artifact_version_id THEN
        RAISE EXCEPTION 'video digest publication upload evidence is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.verification_evidence_artifact_version_id IS NOT NULL
       AND NEW.verification_evidence_artifact_version_id
           IS DISTINCT FROM OLD.verification_evidence_artifact_version_id THEN
        RAISE EXCEPTION 'video digest publication verification evidence is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.failure_evidence_artifact_version_id IS NOT NULL
       AND NEW.failure_evidence_artifact_version_id
           IS DISTINCT FROM OLD.failure_evidence_artifact_version_id THEN
        RAISE EXCEPTION 'video digest publication failure evidence is immutable'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION require_video_digest_slot_terminal_evidence() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    publication_stage TEXT;
    publication_evidence TEXT;
BEGIN
    IF NEW.terminal_fence_required IS DISTINCT FROM OLD.terminal_fence_required THEN
        RAISE EXCEPTION 'video digest terminal fence requirement is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.stage <> 'failed' OR NEW.edition_id IS NULL THEN
        RETURN NEW;
    END IF;
    SELECT publication.stage, publication.failure_evidence_artifact_version_id
    INTO publication_stage, publication_evidence
    FROM video_digest_publication_intents AS publication
    WHERE publication.edition_id = NEW.edition_id
    FOR UPDATE;
    IF FOUND AND (
        publication_stage NOT IN ('conflict', 'failed')
        OR publication_evidence IS DISTINCT FROM NEW.failure_evidence_artifact_version_id
    ) THEN
        RAISE EXCEPTION 'video digest slot failure must match publication evidence'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER video_digest_publication_intents_protect_evidence
BEFORE UPDATE ON video_digest_publication_intents
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_publication_evidence();

CREATE TRIGGER video_digest_slots_require_terminal_evidence
BEFORE UPDATE ON video_digest_slots
FOR EACH ROW EXECUTE FUNCTION require_video_digest_slot_terminal_evidence();

CREATE INDEX video_digest_slots_published_day
    ON video_digest_slots(bucharest_day, edition_id)
    WHERE stage = 'published';
