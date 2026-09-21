ALTER TABLE video_digest_slots
    ADD COLUMN failure_reason TEXT CHECK (
        failure_reason IN ('terminal_failure', 'deadline')
    );

UPDATE video_digest_slots
SET failure_reason = 'terminal_failure'
WHERE stage = 'failed';

ALTER TABLE video_digest_slots
    ADD CONSTRAINT video_digest_slots_failure_reason_matches_stage CHECK (
        (stage = 'failed' AND failure_reason IS NOT NULL)
        OR (stage <> 'failed' AND failure_reason IS NULL)
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
       OR NEW.failure_reason IS NOT NULL
       OR NOT NEW.terminal_fence_required THEN
        RAISE EXCEPTION 'video digest slot must start scheduled and unclaimed'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION protect_video_digest_slot_transition() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.slot_id, NEW.name, NEW.scheduled_at, NEW.bucharest_day, NEW.created_at)
       IS DISTINCT FROM
       (OLD.slot_id, OLD.name, OLD.scheduled_at, OLD.bucharest_day, OLD.created_at) THEN
        RAISE EXCEPTION 'video digest slot identity is immutable' USING ERRCODE = '23000';
    END IF;
    IF OLD.stage IN ('skipped', 'failed', 'published') THEN
        RAISE EXCEPTION 'terminal video digest slot cannot change' USING ERRCODE = '23000';
    END IF;
    IF OLD.edition_id IS NOT NULL AND NEW.edition_id IS DISTINCT FROM OLD.edition_id THEN
        RAISE EXCEPTION 'video digest slot edition cannot change' USING ERRCODE = '23000';
    END IF;
    IF OLD.lease_owner_token IS NOT NULL
       AND NEW.lease_owner_token IS NOT NULL
       AND NEW.lease_owner_token IS DISTINCT FROM OLD.lease_owner_token
       AND OLD.lease_expires_at > clock_timestamp() THEN
        RAISE EXCEPTION 'active video digest slot lease owner cannot change'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.claim_count < OLD.claim_count OR NEW.claim_count > OLD.claim_count + 1 THEN
        RAISE EXCEPTION 'video digest slot claim count must increase one claim at a time'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.lease_owner_token IS DISTINCT FROM OLD.lease_owner_token
       AND NEW.lease_owner_token IS NOT NULL
       AND NEW.claim_count <> OLD.claim_count + 1 THEN
        RAISE EXCEPTION 'a new video digest lease owner requires a new claim'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.lease_owner_token IS NOT DISTINCT FROM OLD.lease_owner_token
       AND NEW.claim_count <> OLD.claim_count
       AND NOT (
           OLD.lease_expires_at <= clock_timestamp()
           AND NEW.claim_count = OLD.claim_count + 1
           AND NEW.lease_expires_at > clock_timestamp()
       ) THEN
        RAISE EXCEPTION 'an active video digest lease cannot change the claim count'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.stage <> OLD.stage AND NOT (
        (OLD.stage = 'scheduled' AND NEW.stage IN ('claimed', 'skipped', 'failed'))
        OR (OLD.stage = 'claimed' AND NEW.stage IN ('planning', 'failed'))
        OR (OLD.stage = 'planning' AND NEW.stage IN ('generating', 'failed'))
        OR (OLD.stage = 'generating' AND NEW.stage IN ('assembling', 'failed'))
        OR (OLD.stage = 'assembling' AND NEW.stage IN ('subtitling', 'failed'))
        OR (OLD.stage = 'subtitling' AND NEW.stage IN ('publishing', 'failed'))
        OR (OLD.stage = 'publishing' AND NEW.stage IN ('published', 'failed'))
    ) THEN
        RAISE EXCEPTION 'illegal video digest slot stage transition from % to %',
            OLD.stage, NEW.stage USING ERRCODE = '23000';
    END IF;
    IF OLD.stage = 'publishing'
       AND NEW.stage = 'published'
       AND NOT EXISTS (
           SELECT 1 FROM video_digest_publication_intents AS publication
           WHERE publication.edition_id = NEW.edition_id
             AND publication.stage = 'published'
       ) THEN
        RAISE EXCEPTION 'video digest slot requires a published publication intent'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TABLE video_digest_assembly_attempts (
    edition_id TEXT NOT NULL REFERENCES video_digest_editions(edition_id)
        CHECK (edition_id ~ '^[0-9a-f]{64}$'),
    attempt_index BIGINT NOT NULL CHECK (attempt_index BETWEEN 0 AND 2),
    disposition TEXT NOT NULL CHECK (disposition IN ('failed', 'succeeded')),
    evidence_artifact_version_id TEXT NOT NULL UNIQUE REFERENCES artifact_versions(id)
        CHECK (evidence_artifact_version_id ~ '^[0-9a-f]{64}$'),
    assembled_video_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            assembled_video_artifact_version_id IS NULL
            OR assembled_video_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    assembly_manifest_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            assembly_manifest_artifact_version_id IS NULL
            OR assembly_manifest_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (edition_id, attempt_index),
    CHECK (
        (disposition = 'failed'
         AND assembled_video_artifact_version_id IS NULL
         AND assembly_manifest_artifact_version_id IS NULL)
        OR (disposition = 'succeeded'
            AND assembled_video_artifact_version_id IS NOT NULL
            AND assembly_manifest_artifact_version_id IS NOT NULL)
    )
);

CREATE TABLE video_digest_subtitle_attempts (
    edition_id TEXT NOT NULL REFERENCES video_digest_editions(edition_id)
        CHECK (edition_id ~ '^[0-9a-f]{64}$'),
    attempt_index BIGINT NOT NULL CHECK (attempt_index BETWEEN 0 AND 2),
    strategy TEXT NOT NULL CHECK (strategy IN (
        'whole-edition-v1', 'per-story-v1', 'per-story-without-vad-v1'
    )),
    disposition TEXT NOT NULL CHECK (disposition IN ('failed', 'succeeded')),
    evidence_artifact_version_id TEXT NOT NULL UNIQUE REFERENCES artifact_versions(id)
        CHECK (evidence_artifact_version_id ~ '^[0-9a-f]{64}$'),
    subtitle_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            subtitle_artifact_version_id IS NULL
            OR subtitle_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (edition_id, attempt_index),
    CHECK (
        (disposition = 'failed' AND subtitle_artifact_version_id IS NULL)
        OR (disposition = 'succeeded' AND subtitle_artifact_version_id IS NOT NULL)
    )
);

CREATE FUNCTION require_video_digest_assembly_attempt_sequence() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    prior_count BIGINT;
BEGIN
    PERFORM 1
    FROM video_digest_slots
    WHERE edition_id = NEW.edition_id AND stage = 'assembling'
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'video digest assembly attempt requires an assembling slot'
            USING ERRCODE = '23000';
    END IF;

    SELECT count(*) INTO prior_count
    FROM video_digest_assembly_attempts
    WHERE edition_id = NEW.edition_id;
    IF NEW.attempt_index <> prior_count OR EXISTS (
        SELECT 1 FROM video_digest_assembly_attempts
        WHERE edition_id = NEW.edition_id AND disposition = 'succeeded'
    ) THEN
        RAISE EXCEPTION 'video digest assembly attempts must be contiguous and stop at success'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION require_video_digest_subtitle_attempt_sequence() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    prior_count BIGINT;
    expected_strategy TEXT;
BEGIN
    PERFORM 1
    FROM video_digest_slots
    WHERE edition_id = NEW.edition_id AND stage = 'subtitling'
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'video digest subtitle attempt requires a subtitling slot'
            USING ERRCODE = '23000';
    END IF;

    SELECT count(*) INTO prior_count
    FROM video_digest_subtitle_attempts
    WHERE edition_id = NEW.edition_id;
    expected_strategy := CASE NEW.attempt_index
        WHEN 0 THEN 'whole-edition-v1'
        WHEN 1 THEN 'per-story-v1'
        WHEN 2 THEN 'per-story-without-vad-v1'
    END;
    IF NEW.attempt_index <> prior_count
       OR NEW.strategy <> expected_strategy
       OR EXISTS (
           SELECT 1 FROM video_digest_subtitle_attempts
           WHERE edition_id = NEW.edition_id AND disposition = 'succeeded'
       ) THEN
        RAISE EXCEPTION 'video digest subtitle attempts must follow the fixed strategy sequence'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER video_digest_assembly_attempts_require_sequence
BEFORE INSERT ON video_digest_assembly_attempts
FOR EACH ROW EXECUTE FUNCTION require_video_digest_assembly_attempt_sequence();

CREATE TRIGGER video_digest_assembly_attempts_reject_updates
BEFORE UPDATE ON video_digest_assembly_attempts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_assembly_attempts_reject_deletes
BEFORE DELETE ON video_digest_assembly_attempts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_subtitle_attempts_require_sequence
BEFORE INSERT ON video_digest_subtitle_attempts
FOR EACH ROW EXECUTE FUNCTION require_video_digest_subtitle_attempt_sequence();

CREATE TRIGGER video_digest_subtitle_attempts_reject_updates
BEFORE UPDATE ON video_digest_subtitle_attempts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_subtitle_attempts_reject_deletes
BEFORE DELETE ON video_digest_subtitle_attempts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();
