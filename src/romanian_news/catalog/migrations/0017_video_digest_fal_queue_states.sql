CREATE TABLE video_digest_fal_queue_states (
    request_id TEXT NOT NULL REFERENCES video_digest_generation_requests(request_id),
    transition_index BIGINT NOT NULL CHECK (transition_index >= 0),
    provider_status TEXT NOT NULL CHECK (provider_status IN ('IN_QUEUE', 'IN_PROGRESS', 'COMPLETED')),
    entered_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (request_id, transition_index)
);

CREATE FUNCTION require_video_digest_fal_queue_state_transition() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    request_stage TEXT;
    receipt_id TEXT;
    previous_index BIGINT;
    previous_status TEXT;
    previous_entered_at TIMESTAMPTZ;
BEGIN
    SELECT stage, provider_receipt_id INTO request_stage, receipt_id
    FROM video_digest_generation_requests
    WHERE request_id = NEW.request_id
    FOR UPDATE;
    IF request_stage IS DISTINCT FROM 'submitted' OR receipt_id IS NULL THEN
        RAISE EXCEPTION 'Fal queue state requires a submitted generation request'
            USING ERRCODE = '23000';
    END IF;
    SELECT transition_index, provider_status, entered_at
    INTO previous_index, previous_status, previous_entered_at
    FROM video_digest_fal_queue_states
    WHERE request_id = NEW.request_id
    ORDER BY transition_index DESC
    LIMIT 1;
    IF (previous_index IS NULL AND NEW.transition_index <> 0)
       OR (previous_index IS NOT NULL AND (
           NEW.transition_index <> previous_index + 1
           OR NEW.provider_status = previous_status
           OR previous_status = 'COMPLETED'
           OR NEW.entered_at < previous_entered_at
       )) THEN
        RAISE EXCEPTION 'Fal queue state transition is invalid'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER video_digest_fal_queue_states_require_transition
BEFORE INSERT ON video_digest_fal_queue_states
FOR EACH ROW EXECUTE FUNCTION require_video_digest_fal_queue_state_transition();

CREATE TRIGGER video_digest_fal_queue_states_reject_updates
BEFORE UPDATE ON video_digest_fal_queue_states
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_fal_queue_states_reject_deletes
BEFORE DELETE ON video_digest_fal_queue_states
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();
