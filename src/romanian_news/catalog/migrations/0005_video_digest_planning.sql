ALTER TABLE video_digest_editions
    ADD COLUMN verification_manifest_artifact_version_id TEXT
        REFERENCES artifact_versions(id)
        CHECK (
            verification_manifest_artifact_version_id IS NULL
            OR verification_manifest_artifact_version_id ~ '^[0-9a-f]{64}$'
        );

CREATE TABLE video_digest_planning_attempts (
    edition_id TEXT NOT NULL REFERENCES video_digest_editions(edition_id)
        CHECK (edition_id ~ '^[0-9a-f]{64}$'),
    attempt_index BIGINT NOT NULL CHECK (attempt_index IN (0, 1, 2)),
    disposition TEXT NOT NULL CHECK (disposition IN ('rejected', 'accepted')),
    attempt_evidence_artifact_version_id TEXT NOT NULL UNIQUE REFERENCES artifact_versions(id)
        CHECK (attempt_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'),
    accepted_plan_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            accepted_plan_artifact_version_id IS NULL
            OR accepted_plan_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (edition_id, attempt_index),
    CHECK (
        (disposition = 'rejected' AND accepted_plan_artifact_version_id IS NULL)
        OR (disposition = 'accepted' AND accepted_plan_artifact_version_id IS NOT NULL)
    )
);

CREATE UNIQUE INDEX video_digest_planning_attempts_one_accepted
    ON video_digest_planning_attempts(edition_id)
    WHERE disposition = 'accepted';

CREATE FUNCTION protect_video_digest_planning_attempt() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'video digest planning attempts are immutable' USING ERRCODE = '23000';
END;
$$;

CREATE FUNCTION require_video_digest_planning_attempt_sequence() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM video_digest_planning_attempts AS attempt
        WHERE attempt.edition_id = NEW.edition_id
          AND attempt.disposition = 'accepted'
    ) THEN
        RAISE EXCEPTION 'accepted video digest planning attempt is terminal'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.attempt_index > 0 AND NOT EXISTS (
        SELECT 1
        FROM video_digest_planning_attempts AS prior
        WHERE prior.edition_id = NEW.edition_id
          AND prior.attempt_index = NEW.attempt_index - 1
          AND prior.disposition = 'rejected'
    ) THEN
        RAISE EXCEPTION 'video digest planning attempts must be contiguous after rejection'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION protect_video_digest_planning_state() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.verification_manifest_artifact_version_id IS NOT NULL THEN
            RAISE EXCEPTION 'video digest edition must start without verification manifest'
                USING ERRCODE = '23000';
        END IF;
        RETURN NEW;
    END IF;

    IF OLD.verification_manifest_artifact_version_id IS NOT NULL
       AND NEW.verification_manifest_artifact_version_id
           IS DISTINCT FROM OLD.verification_manifest_artifact_version_id THEN
        RAISE EXCEPTION 'video digest edition verification manifest is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.plan_artifact_version_id IS NULL AND NEW.plan_artifact_version_id IS NOT NULL
       AND NOT EXISTS (
           SELECT 1
           FROM video_digest_planning_attempts AS attempt
           WHERE attempt.edition_id = NEW.edition_id
             AND attempt.disposition = 'accepted'
             AND attempt.accepted_plan_artifact_version_id = NEW.plan_artifact_version_id
       ) THEN
        RAISE EXCEPTION 'video digest plan requires a matching accepted planning attempt'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.verification_manifest_artifact_version_id IS NULL
       AND NEW.verification_manifest_artifact_version_id IS NOT NULL
       AND (NEW.plan_artifact_version_id IS NULL OR EXISTS (
           SELECT 1
           FROM video_digest_stories AS story
           WHERE story.edition_id = NEW.edition_id
             AND story.mandatory
             AND (
                 story.verification_evidence_artifact_version_id IS NULL
                 OR story.stage NOT IN ('generating', 'accepted')
             )
       )) THEN
        RAISE EXCEPTION 'video digest verification manifest requires generation-ready stories'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION require_video_digest_generation_authorization() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM video_digest_editions AS edition
        WHERE edition.edition_id = NEW.edition_id
          AND edition.plan_artifact_version_id IS NOT NULL
          AND edition.verification_manifest_artifact_version_id IS NOT NULL
    ) THEN
        RAISE EXCEPTION 'video digest generation requires an edition verification manifest'
            USING ERRCODE = '23000';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM video_digest_stories AS story
        WHERE story.edition_id = NEW.edition_id
          AND story.mandatory
          AND (
              story.verification_evidence_artifact_version_id IS NULL
              OR story.stage NOT IN ('generating', 'accepted')
          )
    ) THEN
        RAISE EXCEPTION 'video digest generation requires every mandatory story to be ready'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER video_digest_planning_attempts_reject_updates
BEFORE UPDATE ON video_digest_planning_attempts
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_planning_attempt();

CREATE TRIGGER video_digest_planning_attempts_require_sequence
BEFORE INSERT ON video_digest_planning_attempts
FOR EACH ROW EXECUTE FUNCTION require_video_digest_planning_attempt_sequence();

CREATE TRIGGER video_digest_planning_attempts_reject_deletes
BEFORE DELETE ON video_digest_planning_attempts
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_planning_attempt();

CREATE TRIGGER video_digest_editions_protect_planning_state
BEFORE INSERT OR UPDATE ON video_digest_editions
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_planning_state();

CREATE TRIGGER video_digest_generation_requests_require_authorization
BEFORE INSERT ON video_digest_generation_requests
FOR EACH ROW EXECUTE FUNCTION require_video_digest_generation_authorization();
