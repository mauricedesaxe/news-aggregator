ALTER TABLE video_digest_generation_requests
    ADD COLUMN generation_policy_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            generation_policy_artifact_version_id IS NULL
            OR generation_policy_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    ADD COLUMN reserved_cost_usd NUMERIC CHECK (reserved_cost_usd >= 0),
    ADD COLUMN deadline_at TIMESTAMPTZ;

CREATE TABLE video_digest_generation_reservations (
    request_id TEXT NOT NULL REFERENCES video_digest_generation_requests(request_id),
    scope_kind TEXT NOT NULL CHECK (
        scope_kind IN ('story', 'edition', 'bucharest_day', 'calendar_month')
    ),
    scope_key TEXT NOT NULL CHECK (length(trim(scope_key)) > 0),
    limit_usd NUMERIC NOT NULL CHECK (limit_usd >= 0),
    reserved_usd NUMERIC NOT NULL CHECK (reserved_usd >= 0),
    generation_policy_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (generation_policy_artifact_version_id ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (request_id, scope_kind)
);

CREATE FUNCTION require_video_digest_generation_order() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    slot_scheduled_at TIMESTAMPTZ;
BEGIN
    SELECT slot.scheduled_at INTO slot_scheduled_at
    FROM video_digest_slots AS slot
    WHERE slot.edition_id = NEW.edition_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'video digest generation request requires a scheduled slot'
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

CREATE FUNCTION require_video_digest_generation_budget() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    stored_limit NUMERIC;
    used NUMERIC;
BEGIN
    PERFORM pg_advisory_xact_lock(7337801455016519294);

    SELECT reservation.limit_usd INTO stored_limit
    FROM video_digest_generation_reservations AS reservation
    WHERE reservation.scope_kind = NEW.scope_kind
      AND reservation.scope_key = NEW.scope_key
    LIMIT 1;
    IF FOUND AND stored_limit <> NEW.limit_usd THEN
        RAISE EXCEPTION 'video digest generation budget limit changed within a scope'
            USING ERRCODE = '23000';
    END IF;

    SELECT COALESCE(sum(reservation.reserved_usd), 0) INTO used
    FROM video_digest_generation_reservations AS reservation
    WHERE reservation.scope_kind = NEW.scope_kind
      AND reservation.scope_key = NEW.scope_key;
    IF used + NEW.reserved_usd > NEW.limit_usd THEN
        RAISE EXCEPTION 'video digest generation budget exceeded for %', NEW.scope_kind
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION require_video_digest_generation_reservations() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    reservation_count BIGINT;
    distinct_amounts BIGINT;
    distinct_policies BIGINT;
BEGIN
    SELECT count(*), count(DISTINCT reservation.reserved_usd),
           count(DISTINCT reservation.generation_policy_artifact_version_id)
    INTO reservation_count, distinct_amounts, distinct_policies
    FROM video_digest_generation_reservations AS reservation
    WHERE reservation.request_id = NEW.request_id;

    IF reservation_count <> 4 OR distinct_amounts <> 1 OR distinct_policies <> 1 THEN
        RAISE EXCEPTION 'video digest generation request requires four matching reservations'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION reject_video_digest_generation_reservation_change() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'video digest generation reservations are append-only'
        USING ERRCODE = '23000';
END;
$$;

CREATE FUNCTION protect_video_digest_generation_admission() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.generation_policy_artifact_version_id, NEW.reserved_cost_usd, NEW.deadline_at)
       IS DISTINCT FROM
       (OLD.generation_policy_artifact_version_id, OLD.reserved_cost_usd, OLD.deadline_at) THEN
        RAISE EXCEPTION 'video digest generation admission is immutable'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER video_digest_generation_requests_require_order
BEFORE INSERT ON video_digest_generation_requests
FOR EACH ROW EXECUTE FUNCTION require_video_digest_generation_order();

CREATE TRIGGER video_digest_generation_reservations_require_budget
BEFORE INSERT ON video_digest_generation_reservations
FOR EACH ROW EXECUTE FUNCTION require_video_digest_generation_budget();

CREATE TRIGGER video_digest_generation_requests_protect_admission
BEFORE UPDATE ON video_digest_generation_requests
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_generation_admission();

CREATE CONSTRAINT TRIGGER video_digest_generation_requests_require_reservations
AFTER INSERT ON video_digest_generation_requests
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION require_video_digest_generation_reservations();

CREATE TRIGGER video_digest_generation_reservations_reject_updates
BEFORE UPDATE ON video_digest_generation_reservations
FOR EACH ROW EXECUTE FUNCTION reject_video_digest_generation_reservation_change();

CREATE TRIGGER video_digest_generation_reservations_reject_deletes
BEFORE DELETE ON video_digest_generation_reservations
FOR EACH ROW EXECUTE FUNCTION reject_video_digest_generation_reservation_change();
