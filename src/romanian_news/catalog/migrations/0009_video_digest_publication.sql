DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM video_digest_publication_intents) THEN
        RAISE EXCEPTION 'video digest publication policy migration requires no existing intents'
            USING ERRCODE = '23000';
    END IF;
END;
$$;

ALTER TABLE video_digest_publication_intents
    ADD COLUMN cache_control TEXT NOT NULL
        DEFAULT 'public,max-age=31536000,immutable'
        CHECK (cache_control = 'public,max-age=31536000,immutable'),
    ADD COLUMN visibility TEXT NOT NULL DEFAULT 'public'
        CHECK (visibility = 'public'),
    ADD COLUMN retention TEXT NOT NULL DEFAULT 'permanent'
        CHECK (retention = 'permanent'),
    ADD CONSTRAINT video_digest_publication_video_media_type_exact
        CHECK (video_media_type = 'video/mp4'),
    ADD CONSTRAINT video_digest_publication_subtitle_media_type_exact
        CHECK (subtitle_media_type IS NULL OR subtitle_media_type = 'text/vtt');

CREATE OR REPLACE FUNCTION protect_video_digest_publication_intent_transition() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.publication_id, NEW.edition_id, NEW.expected_video_key, NEW.video_digest,
        NEW.video_byte_size, NEW.video_media_type, NEW.subtitle_expected_key,
        NEW.subtitle_digest, NEW.subtitle_byte_size, NEW.subtitle_media_type,
        NEW.source_video_artifact_version_id, NEW.source_subtitle_artifact_version_id,
        NEW.cache_control, NEW.visibility, NEW.retention, NEW.created_at)
       IS DISTINCT FROM
       (OLD.publication_id, OLD.edition_id, OLD.expected_video_key, OLD.video_digest,
        OLD.video_byte_size, OLD.video_media_type, OLD.subtitle_expected_key,
        OLD.subtitle_digest, OLD.subtitle_byte_size, OLD.subtitle_media_type,
        OLD.source_video_artifact_version_id, OLD.source_subtitle_artifact_version_id,
        OLD.cache_control, OLD.visibility, OLD.retention, OLD.created_at) THEN
        RAISE EXCEPTION 'video digest publication intent identity is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.stage IN ('published', 'conflict', 'failed') THEN
        RAISE EXCEPTION 'terminal video digest publication intent cannot change'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.stage = 'published' AND NOT EXISTS (
        SELECT 1
        FROM video_digest_publication_attempts AS attempt
        WHERE attempt.publication_id = NEW.publication_id
          AND attempt.state = 'succeeded'
    ) THEN
        RAISE EXCEPTION 'published video digest requires a successful publication attempt'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.stage <> OLD.stage AND NOT (
        (OLD.stage = 'pending' AND NEW.stage IN ('uploading', 'conflict', 'failed'))
        OR (OLD.stage = 'uploading' AND NEW.stage IN ('uploaded', 'conflict', 'failed'))
        OR (OLD.stage = 'uploaded' AND NEW.stage IN ('verified', 'conflict', 'failed'))
        OR (OLD.stage = 'verified' AND NEW.stage IN ('published', 'conflict', 'failed'))
    ) THEN
        RAISE EXCEPTION 'illegal video digest publication stage transition from % to %',
            OLD.stage, NEW.stage USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TABLE video_digest_publication_attempts (
    publication_id TEXT NOT NULL REFERENCES video_digest_publication_intents(publication_id)
        CHECK (publication_id ~ '^[0-9a-f]{64}$'),
    attempt_index BIGINT NOT NULL CHECK (attempt_index BETWEEN 0 AND 4),
    state TEXT NOT NULL CHECK (state IN (
        'started', 'retryable', 'succeeded', 'conflict', 'failed'
    )),
    retry_at TIMESTAMPTZ,
    failure_evidence_artifact_version_id TEXT UNIQUE REFERENCES artifact_versions(id)
        CHECK (
            failure_evidence_artifact_version_id IS NULL
            OR failure_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    started_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (publication_id, attempt_index),
    CHECK (updated_at >= started_at),
    CHECK (
        (state IN ('started', 'succeeded')
         AND retry_at IS NULL
         AND failure_evidence_artifact_version_id IS NULL)
        OR (state = 'retryable'
            AND attempt_index BETWEEN 0 AND 3
            AND retry_at IS NOT NULL
            AND failure_evidence_artifact_version_id IS NOT NULL)
        OR (state IN ('conflict', 'failed')
            AND retry_at IS NULL
            AND failure_evidence_artifact_version_id IS NOT NULL)
    )
);

CREATE UNIQUE INDEX video_digest_publication_attempts_one_started
    ON video_digest_publication_attempts(publication_id)
    WHERE state = 'started';

CREATE FUNCTION require_video_digest_publication_attempt_sequence() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    prior_count BIGINT;
    prior_state TEXT;
    prior_retry_at TIMESTAMPTZ;
BEGIN
    IF NEW.state <> 'started'
       OR NEW.retry_at IS NOT NULL
       OR NEW.failure_evidence_artifact_version_id IS NOT NULL THEN
        RAISE EXCEPTION 'video digest publication attempt must start active'
            USING ERRCODE = '23000';
    END IF;
    PERFORM 1
    FROM video_digest_publication_intents
    WHERE publication_id = NEW.publication_id
      AND stage IN ('pending', 'uploading', 'uploaded', 'verified')
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'video digest publication attempt requires an active publication'
            USING ERRCODE = '23000';
    END IF;

    SELECT count(*) INTO prior_count
    FROM video_digest_publication_attempts
    WHERE publication_id = NEW.publication_id;
    IF NEW.attempt_index <> prior_count THEN
        RAISE EXCEPTION 'video digest publication attempts must be contiguous'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.attempt_index > 0 THEN
        SELECT state, retry_at INTO prior_state, prior_retry_at
        FROM video_digest_publication_attempts
        WHERE publication_id = NEW.publication_id
          AND attempt_index = NEW.attempt_index - 1;
        IF prior_state <> 'retryable' OR prior_retry_at > clock_timestamp() THEN
            RAISE EXCEPTION 'video digest publication retry is not ready'
                USING ERRCODE = '23000';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION protect_video_digest_publication_attempt_transition() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.publication_id, NEW.attempt_index, NEW.started_at)
       IS DISTINCT FROM
       (OLD.publication_id, OLD.attempt_index, OLD.started_at) THEN
        RAISE EXCEPTION 'video digest publication attempt identity is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.state <> 'started' OR NEW.state = 'started' THEN
        RAISE EXCEPTION 'video digest publication attempt can finish only once'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER video_digest_publication_attempts_require_sequence
BEFORE INSERT ON video_digest_publication_attempts
FOR EACH ROW EXECUTE FUNCTION require_video_digest_publication_attempt_sequence();

CREATE TRIGGER video_digest_publication_attempts_protect_transition
BEFORE UPDATE ON video_digest_publication_attempts
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_publication_attempt_transition();

CREATE TRIGGER video_digest_publication_attempts_reject_deletes
BEFORE DELETE ON video_digest_publication_attempts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();
