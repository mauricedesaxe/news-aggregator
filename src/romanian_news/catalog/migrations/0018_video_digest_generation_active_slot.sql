CREATE OR REPLACE FUNCTION require_video_digest_generation_order() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    slot_scheduled_at TIMESTAMPTZ;
BEGIN
    SELECT slot.scheduled_at INTO slot_scheduled_at
    FROM video_digest_slots AS slot
    WHERE slot.edition_id = NEW.edition_id
      AND slot.stage IN ('claimed', 'planning', 'generating', 'assembling', 'subtitling', 'publishing')
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'video digest generation request requires an active slot'
            USING ERRCODE = '23503';
    END IF;
    IF NEW.generation_policy_artifact_version_id IS NULL
       OR NEW.reserved_cost_usd IS NULL
       OR NEW.deadline_at IS NULL THEN
        RAISE EXCEPTION 'video digest generation request requires admission evidence'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.deadline_at <> slot_scheduled_at + INTERVAL '90 minutes' THEN
        RAISE EXCEPTION 'video digest generation deadline must be 90 minutes after its slot'
            USING ERRCODE = '23000';
    END IF;
    IF clock_timestamp() >= NEW.deadline_at THEN
        RAISE EXCEPTION 'video digest generation deadline has passed' USING ERRCODE = '23000';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM video_digest_stories AS story
        WHERE story.edition_id = NEW.edition_id
          AND story.position < NEW.story_position
          AND story.stage <> 'accepted'
    ) THEN
        RAISE EXCEPTION 'video digest stories must generate in plan order'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;
