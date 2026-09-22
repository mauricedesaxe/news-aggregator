CREATE TABLE video_digest_feedback (
    feedback_id UUID PRIMARY KEY,
    edition_id TEXT NOT NULL REFERENCES video_digest_editions(edition_id)
        CHECK (edition_id ~ '^[0-9a-f]{64}$'),
    story_id TEXT REFERENCES video_digest_stories(story_id)
        CHECK (story_id IS NULL OR story_id ~ '^[0-9a-f]{64}$'),
    rating TEXT CHECK (rating IN ('positive', 'negative')),
    note TEXT CHECK (note IS NULL OR length(note) <= 2000),
    actor TEXT NOT NULL CHECK (actor = 'owner'),
    publication_id TEXT NOT NULL REFERENCES video_digest_publication_intents(publication_id)
        CHECK (publication_id ~ '^[0-9a-f]{64}$'),
    daily_report_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (daily_report_version_id ~ '^[0-9a-f]{64}$'),
    policy_bundle_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (policy_bundle_version_id ~ '^[0-9a-f]{64}$'),
    plan_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (plan_artifact_version_id ~ '^[0-9a-f]{64}$'),
    verification_manifest_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (verification_manifest_artifact_version_id ~ '^[0-9a-f]{64}$'),
    assembled_video_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (assembled_video_artifact_version_id ~ '^[0-9a-f]{64}$'),
    assembly_manifest_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (assembly_manifest_artifact_version_id ~ '^[0-9a-f]{64}$'),
    publication_verification_evidence_artifact_version_id TEXT NOT NULL
        REFERENCES artifact_versions(id)
        CHECK (publication_verification_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL,
    CHECK (rating IS NOT NULL OR (note IS NOT NULL AND length(trim(note)) > 0))
);

CREATE TABLE video_digest_feedback_stories (
    feedback_id UUID NOT NULL REFERENCES video_digest_feedback(feedback_id),
    story_id TEXT NOT NULL REFERENCES video_digest_stories(story_id)
        CHECK (story_id ~ '^[0-9a-f]{64}$'),
    position BIGINT NOT NULL CHECK (position >= 0),
    verification_evidence_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (verification_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'),
    generation_request_id TEXT NOT NULL REFERENCES video_digest_generation_requests(request_id)
        CHECK (generation_request_id ~ '^[0-9a-f]{64}$'),
    generation_request_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (generation_request_artifact_version_id ~ '^[0-9a-f]{64}$'),
    generation_policy_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (generation_policy_artifact_version_id ~ '^[0-9a-f]{64}$'),
    generation_response_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (generation_response_artifact_version_id ~ '^[0-9a-f]{64}$'),
    accepted_clip_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (accepted_clip_artifact_version_id ~ '^[0-9a-f]{64}$'),
    validation_evidence_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (validation_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'),
    PRIMARY KEY (feedback_id, story_id),
    UNIQUE (feedback_id, position)
);

CREATE FUNCTION require_video_digest_feedback_lineage() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM video_digest_editions AS edition
        JOIN video_digest_slots AS slot
          ON slot.edition_id = edition.edition_id AND slot.stage = 'published'
        JOIN video_digest_publication_intents AS publication
          ON publication.edition_id = edition.edition_id
         AND publication.stage = 'published'
        JOIN video_digest_planning_attempts AS planning
          ON planning.edition_id = edition.edition_id
         AND planning.disposition = 'accepted'
         AND planning.accepted_plan_artifact_version_id = edition.plan_artifact_version_id
        JOIN video_digest_assembly_attempts AS assembly
          ON assembly.edition_id = edition.edition_id
         AND assembly.disposition = 'succeeded'
         AND assembly.assembled_video_artifact_version_id =
             edition.assembled_video_artifact_version_id
         AND assembly.assembly_manifest_artifact_version_id =
             edition.assembly_manifest_artifact_version_id
        WHERE edition.edition_id = NEW.edition_id
          AND (NEW.story_id IS NULL OR EXISTS (
              SELECT 1 FROM video_digest_stories AS target_story
              WHERE target_story.edition_id = edition.edition_id
                AND target_story.story_id = NEW.story_id
                AND target_story.stage = 'accepted'
          ))
          AND EXISTS (
              SELECT 1 FROM video_digest_stories AS edition_story
              WHERE edition_story.edition_id = edition.edition_id
                AND edition_story.stage = 'accepted'
          )
          AND NOT EXISTS (
              SELECT 1 FROM video_digest_stories AS incomplete_story
              WHERE incomplete_story.edition_id = edition.edition_id
                AND incomplete_story.stage <> 'accepted'
          )
          AND (NEW.publication_id, NEW.daily_report_version_id,
               NEW.policy_bundle_version_id, NEW.plan_artifact_version_id,
               NEW.verification_manifest_artifact_version_id,
               NEW.assembled_video_artifact_version_id,
               NEW.assembly_manifest_artifact_version_id,
               NEW.publication_verification_evidence_artifact_version_id)
              IS NOT DISTINCT FROM
              (publication.publication_id, edition.daily_report_version_id,
               edition.policy_bundle_version_id, edition.plan_artifact_version_id,
               edition.verification_manifest_artifact_version_id,
               edition.assembled_video_artifact_version_id,
               edition.assembly_manifest_artifact_version_id,
               publication.verification_evidence_artifact_version_id)
    ) THEN
        RAISE EXCEPTION 'video digest feedback requires exact terminal publication lineage'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION require_video_digest_feedback_story_lineage() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM video_digest_feedback AS feedback
        JOIN video_digest_stories AS story
          ON story.edition_id = feedback.edition_id
         AND story.story_id = NEW.story_id
         AND story.stage = 'accepted'
        JOIN video_digest_generation_requests AS generation
          ON generation.edition_id = story.edition_id
         AND generation.story_position = story.position
         AND generation.stage = 'accepted'
         AND generation.accepted_clip_artifact_version_id =
             story.accepted_clip_artifact_version_id
        WHERE feedback.feedback_id = NEW.feedback_id
          AND (feedback.story_id IS NULL OR feedback.story_id = story.story_id)
          AND (NEW.position, NEW.verification_evidence_artifact_version_id,
               NEW.generation_request_id,
               NEW.generation_request_artifact_version_id,
               NEW.generation_policy_artifact_version_id,
               NEW.generation_response_artifact_version_id,
               NEW.accepted_clip_artifact_version_id,
               NEW.validation_evidence_artifact_version_id)
              IS NOT DISTINCT FROM
              (story.position, story.verification_evidence_artifact_version_id,
               generation.request_id, generation.request_artifact_version_id,
               generation.generation_policy_artifact_version_id,
               generation.response_artifact_version_id,
               generation.accepted_clip_artifact_version_id,
               generation.validation_evidence_artifact_version_id)
    ) THEN
        RAISE EXCEPTION 'video digest feedback story requires exact accepted generation lineage'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION require_complete_video_digest_feedback() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    expected_count BIGINT;
    recorded_count BIGINT;
BEGIN
    SELECT CASE
        WHEN NEW.story_id IS NULL THEN count(*)
        ELSE count(*) FILTER (WHERE story.story_id = NEW.story_id)
    END INTO expected_count
    FROM video_digest_stories AS story
    WHERE story.edition_id = NEW.edition_id;

    SELECT count(*) INTO recorded_count
    FROM video_digest_feedback_stories AS snapshot
    WHERE snapshot.feedback_id = NEW.feedback_id;

    IF expected_count = 0 OR recorded_count <> expected_count THEN
        RAISE EXCEPTION 'video digest feedback story lineage is incomplete'
            USING ERRCODE = '23000';
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER video_digest_feedback_require_lineage
BEFORE INSERT ON video_digest_feedback
FOR EACH ROW EXECUTE FUNCTION require_video_digest_feedback_lineage();

CREATE TRIGGER video_digest_feedback_stories_require_lineage
BEFORE INSERT ON video_digest_feedback_stories
FOR EACH ROW EXECUTE FUNCTION require_video_digest_feedback_story_lineage();

CREATE CONSTRAINT TRIGGER video_digest_feedback_require_completeness
AFTER INSERT ON video_digest_feedback
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION require_complete_video_digest_feedback();

CREATE TRIGGER video_digest_feedback_reject_updates
BEFORE UPDATE ON video_digest_feedback
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_feedback_reject_deletes
BEFORE DELETE ON video_digest_feedback
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_feedback_stories_reject_updates
BEFORE UPDATE ON video_digest_feedback_stories
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_feedback_stories_reject_deletes
BEFORE DELETE ON video_digest_feedback_stories
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();
