CREATE TABLE video_digest_editions (
    edition_id TEXT PRIMARY KEY CHECK (edition_id ~ '^[0-9a-f]{64}$'),
    daily_report_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (daily_report_version_id ~ '^[0-9a-f]{64}$'),
    policy_bundle_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (policy_bundle_version_id ~ '^[0-9a-f]{64}$'),
    plan_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (plan_artifact_version_id IS NULL OR plan_artifact_version_id ~ '^[0-9a-f]{64}$'),
    assembled_video_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            assembled_video_artifact_version_id IS NULL
            OR assembled_video_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    subtitle_state TEXT NOT NULL CHECK (subtitle_state IN ('pending', 'available', 'failed')),
    subtitle_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            subtitle_artifact_version_id IS NULL
            OR subtitle_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    subtitle_failure_evidence_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            subtitle_failure_evidence_artifact_version_id IS NULL
            OR subtitle_failure_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    UNIQUE (daily_report_version_id, policy_bundle_version_id),
    CHECK (updated_at >= created_at),
    CHECK (
        (subtitle_state = 'pending'
         AND subtitle_artifact_version_id IS NULL
         AND subtitle_failure_evidence_artifact_version_id IS NULL)
        OR (subtitle_state = 'available'
            AND subtitle_artifact_version_id IS NOT NULL
            AND subtitle_failure_evidence_artifact_version_id IS NULL)
        OR (subtitle_state = 'failed'
            AND subtitle_artifact_version_id IS NULL
            AND subtitle_failure_evidence_artifact_version_id IS NOT NULL)
    )
);

CREATE TABLE video_digest_slots (
    slot_id TEXT PRIMARY KEY CHECK (slot_id ~ '^[0-9a-f]{64}$'),
    name TEXT NOT NULL CHECK (name IN ('morning', 'midday', 'evening')),
    scheduled_at TIMESTAMPTZ NOT NULL,
    bucharest_day DATE NOT NULL,
    stage TEXT NOT NULL CHECK (stage IN (
        'scheduled', 'claimed', 'planning', 'generating', 'assembling',
        'subtitling', 'publishing', 'skipped', 'failed', 'published'
    )),
    edition_id TEXT REFERENCES video_digest_editions(edition_id)
        CHECK (edition_id IS NULL OR edition_id ~ '^[0-9a-f]{64}$'),
    lease_owner_token TEXT CHECK (
        lease_owner_token IS NULL OR length(trim(lease_owner_token)) > 0
    ),
    lease_expires_at TIMESTAMPTZ,
    claim_count BIGINT NOT NULL CHECK (claim_count >= 0),
    skip_reason TEXT CHECK (skip_reason IN (
        'source_missing', 'source_empty', 'unchanged', 'already_published',
        'active_edition', 'overlapping_run'
    )),
    failure_evidence_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            failure_evidence_artifact_version_id IS NULL
            OR failure_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    UNIQUE (name, scheduled_at),
    CHECK (bucharest_day = (scheduled_at AT TIME ZONE 'Europe/Bucharest')::DATE),
    CHECK (updated_at >= created_at),
    CHECK (
        (stage = 'scheduled'
         AND edition_id IS NULL
         AND lease_owner_token IS NULL
         AND lease_expires_at IS NULL
         AND claim_count = 0
         AND skip_reason IS NULL
         AND failure_evidence_artifact_version_id IS NULL)
        OR (stage IN (
                'claimed', 'planning', 'generating', 'assembling',
                'subtitling', 'publishing'
            )
            AND edition_id IS NOT NULL
            AND lease_owner_token IS NOT NULL
            AND lease_expires_at IS NOT NULL
            AND claim_count > 0
            AND skip_reason IS NULL
            AND failure_evidence_artifact_version_id IS NULL)
        OR (stage = 'skipped'
            AND lease_owner_token IS NULL
            AND lease_expires_at IS NULL
            AND skip_reason IS NOT NULL
            AND failure_evidence_artifact_version_id IS NULL)
        OR (stage = 'failed'
            AND lease_owner_token IS NULL
            AND lease_expires_at IS NULL
            AND skip_reason IS NULL
            AND failure_evidence_artifact_version_id IS NOT NULL)
        OR (stage = 'published'
            AND edition_id IS NOT NULL
            AND lease_owner_token IS NULL
            AND lease_expires_at IS NULL
            AND skip_reason IS NULL
            AND failure_evidence_artifact_version_id IS NULL)
    )
);

CREATE TABLE video_digest_stories (
    story_id TEXT PRIMARY KEY CHECK (story_id ~ '^[0-9a-f]{64}$'),
    edition_id TEXT NOT NULL REFERENCES video_digest_editions(edition_id)
        CHECK (edition_id ~ '^[0-9a-f]{64}$'),
    position BIGINT NOT NULL CHECK (position >= 0),
    report_subject_id TEXT NOT NULL CHECK (report_subject_id ~ '^[0-9a-f]{64}$'),
    title TEXT NOT NULL CHECK (length(trim(title)) > 0),
    mandatory BOOLEAN NOT NULL CHECK (mandatory),
    requested_duration_ms BIGINT NOT NULL CHECK (requested_duration_ms > 0),
    stage TEXT NOT NULL CHECK (stage IN (
        'planned', 'verifying', 'verified', 'generating', 'accepted', 'failed'
    )),
    verification_evidence_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            verification_evidence_artifact_version_id IS NULL
            OR verification_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    accepted_clip_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            accepted_clip_artifact_version_id IS NULL
            OR accepted_clip_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    failure_evidence_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            failure_evidence_artifact_version_id IS NULL
            OR failure_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    UNIQUE (edition_id, position),
    UNIQUE (edition_id, report_subject_id),
    CHECK (updated_at >= created_at),
    CHECK (
        (stage IN ('planned', 'verifying')
         AND verification_evidence_artifact_version_id IS NULL
         AND accepted_clip_artifact_version_id IS NULL
         AND failure_evidence_artifact_version_id IS NULL)
        OR (stage IN ('verified', 'generating')
            AND verification_evidence_artifact_version_id IS NOT NULL
            AND accepted_clip_artifact_version_id IS NULL
            AND failure_evidence_artifact_version_id IS NULL)
        OR (stage = 'accepted'
            AND verification_evidence_artifact_version_id IS NOT NULL
            AND accepted_clip_artifact_version_id IS NOT NULL
            AND failure_evidence_artifact_version_id IS NULL)
        OR (stage = 'failed'
            AND accepted_clip_artifact_version_id IS NULL
            AND failure_evidence_artifact_version_id IS NOT NULL)
    )
);

CREATE TABLE video_digest_generation_requests (
    request_id TEXT PRIMARY KEY CHECK (request_id ~ '^[0-9a-f]{64}$'),
    edition_id TEXT NOT NULL CHECK (edition_id ~ '^[0-9a-f]{64}$'),
    story_position BIGINT NOT NULL CHECK (story_position >= 0),
    attempt_index BIGINT NOT NULL CHECK (attempt_index IN (0, 1)),
    request_artifact_version_id TEXT NOT NULL UNIQUE REFERENCES artifact_versions(id)
        CHECK (request_artifact_version_id ~ '^[0-9a-f]{64}$'),
    stage TEXT NOT NULL CHECK (stage IN (
        'pending', 'submitted', 'processing', 'accepted', 'failed'
    )),
    provider_receipt_id TEXT CHECK (
        provider_receipt_id IS NULL OR length(trim(provider_receipt_id)) > 0
    ),
    response_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            response_artifact_version_id IS NULL
            OR response_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    accepted_clip_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            accepted_clip_artifact_version_id IS NULL
            OR accepted_clip_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    failure_evidence_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            failure_evidence_artifact_version_id IS NULL
            OR failure_evidence_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    cost_kind TEXT NOT NULL CHECK (cost_kind IN (
        'pending', 'estimated', 'measured', 'unknown'
    )),
    cost_usd NUMERIC,
    cost_unknown_reason TEXT CHECK (
        cost_unknown_reason IS NULL OR length(trim(cost_unknown_reason)) > 0
    ),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    UNIQUE (edition_id, story_position, attempt_index),
    FOREIGN KEY (edition_id, story_position)
        REFERENCES video_digest_stories(edition_id, position),
    CHECK (updated_at >= created_at),
    CHECK (
        (stage = 'pending'
         AND provider_receipt_id IS NULL
         AND response_artifact_version_id IS NULL
         AND accepted_clip_artifact_version_id IS NULL
         AND failure_evidence_artifact_version_id IS NULL)
        OR (stage = 'submitted'
            AND provider_receipt_id IS NOT NULL
            AND response_artifact_version_id IS NULL
            AND accepted_clip_artifact_version_id IS NULL
            AND failure_evidence_artifact_version_id IS NULL)
        OR (stage = 'processing'
            AND provider_receipt_id IS NOT NULL
            AND accepted_clip_artifact_version_id IS NULL
            AND failure_evidence_artifact_version_id IS NULL)
        OR (stage = 'accepted'
            AND provider_receipt_id IS NOT NULL
            AND response_artifact_version_id IS NOT NULL
            AND accepted_clip_artifact_version_id IS NOT NULL
            AND failure_evidence_artifact_version_id IS NULL)
        OR (stage = 'failed'
            AND accepted_clip_artifact_version_id IS NULL
            AND failure_evidence_artifact_version_id IS NOT NULL)
    ),
    CHECK (
        (cost_kind = 'pending' AND cost_usd IS NULL AND cost_unknown_reason IS NULL)
        OR (cost_kind IN ('estimated', 'measured')
            AND cost_usd IS NOT NULL
            AND cost_usd >= 0
            AND cost_unknown_reason IS NULL)
        OR (cost_kind = 'unknown'
            AND cost_usd IS NULL
            AND cost_unknown_reason IS NOT NULL)
    )
);

CREATE TABLE video_digest_publication_intents (
    publication_id TEXT PRIMARY KEY CHECK (publication_id ~ '^[0-9a-f]{64}$'),
    edition_id TEXT NOT NULL UNIQUE REFERENCES video_digest_editions(edition_id)
        CHECK (edition_id ~ '^[0-9a-f]{64}$'),
    expected_video_key TEXT NOT NULL CHECK (length(trim(expected_video_key)) > 0),
    video_digest TEXT NOT NULL CHECK (video_digest ~ '^[0-9a-f]{64}$'),
    video_byte_size BIGINT NOT NULL CHECK (video_byte_size > 0),
    video_media_type TEXT NOT NULL CHECK (length(trim(video_media_type)) > 0),
    subtitle_expected_key TEXT CHECK (
        subtitle_expected_key IS NULL OR length(trim(subtitle_expected_key)) > 0
    ),
    subtitle_digest TEXT CHECK (
        subtitle_digest IS NULL OR subtitle_digest ~ '^[0-9a-f]{64}$'
    ),
    subtitle_byte_size BIGINT CHECK (subtitle_byte_size IS NULL OR subtitle_byte_size > 0),
    subtitle_media_type TEXT CHECK (
        subtitle_media_type IS NULL OR length(trim(subtitle_media_type)) > 0
    ),
    source_video_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id)
        CHECK (source_video_artifact_version_id ~ '^[0-9a-f]{64}$'),
    source_subtitle_artifact_version_id TEXT REFERENCES artifact_versions(id)
        CHECK (
            source_subtitle_artifact_version_id IS NULL
            OR source_subtitle_artifact_version_id ~ '^[0-9a-f]{64}$'
        ),
    stage TEXT NOT NULL CHECK (stage IN (
        'pending', 'uploading', 'uploaded', 'verified', 'published', 'conflict', 'failed'
    )),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    published_at TIMESTAMPTZ,
    CHECK (updated_at >= created_at),
    CHECK (
        (subtitle_expected_key IS NULL
         AND subtitle_digest IS NULL
         AND subtitle_byte_size IS NULL
         AND subtitle_media_type IS NULL
         AND source_subtitle_artifact_version_id IS NULL)
        OR (subtitle_expected_key IS NOT NULL
            AND subtitle_digest IS NOT NULL
            AND subtitle_byte_size IS NOT NULL
            AND subtitle_media_type IS NOT NULL
            AND source_subtitle_artifact_version_id IS NOT NULL)
    ),
    CHECK (
        (stage = 'published' AND published_at IS NOT NULL)
        OR (stage <> 'published' AND published_at IS NULL)
    )
);

CREATE FUNCTION require_video_digest_edition_initial_state() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.plan_artifact_version_id IS NOT NULL
       OR NEW.assembled_video_artifact_version_id IS NOT NULL
       OR NEW.subtitle_state <> 'pending'
       OR NEW.subtitle_artifact_version_id IS NOT NULL
       OR NEW.subtitle_failure_evidence_artifact_version_id IS NOT NULL THEN
        RAISE EXCEPTION 'video digest edition must start pending without outputs'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION require_video_digest_slot_initial_state() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.stage <> 'scheduled'
       OR NEW.claim_count <> 0
       OR NEW.edition_id IS NOT NULL
       OR NEW.lease_owner_token IS NOT NULL
       OR NEW.lease_expires_at IS NOT NULL
       OR NEW.skip_reason IS NOT NULL
       OR NEW.failure_evidence_artifact_version_id IS NOT NULL THEN
        RAISE EXCEPTION 'video digest slot must start scheduled and unclaimed'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION require_video_digest_story_initial_state() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    edition_plan_artifact_version_id TEXT;
BEGIN
    IF NEW.stage <> 'planned'
       OR NEW.verification_evidence_artifact_version_id IS NOT NULL
       OR NEW.accepted_clip_artifact_version_id IS NOT NULL
       OR NEW.failure_evidence_artifact_version_id IS NOT NULL THEN
        RAISE EXCEPTION 'video digest story must start planned without outputs'
            USING ERRCODE = '23000';
    END IF;

    SELECT edition.plan_artifact_version_id INTO edition_plan_artifact_version_id
    FROM video_digest_editions AS edition
    WHERE edition.edition_id = NEW.edition_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'video digest story requires an edition' USING ERRCODE = '23503';
    END IF;
    IF edition_plan_artifact_version_id IS NOT NULL THEN
        RAISE EXCEPTION 'video digest story cannot be added after plan finalization'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION require_video_digest_generation_request_initial_state() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.stage <> 'pending'
       OR NEW.cost_kind <> 'pending'
       OR NEW.cost_usd IS NOT NULL
       OR NEW.cost_unknown_reason IS NOT NULL
       OR NEW.provider_receipt_id IS NOT NULL
       OR NEW.response_artifact_version_id IS NOT NULL
       OR NEW.accepted_clip_artifact_version_id IS NOT NULL
       OR NEW.failure_evidence_artifact_version_id IS NOT NULL THEN
        RAISE EXCEPTION 'video digest generation request must start pending without outputs'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION require_video_digest_publication_intent_initial_state() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.stage <> 'pending' OR NEW.published_at IS NOT NULL THEN
        RAISE EXCEPTION 'video digest publication intent must start pending'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION protect_video_digest_edition_identity() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    story_count BIGINT;
    first_position BIGINT;
    last_position BIGINT;
BEGIN
    IF (NEW.edition_id, NEW.daily_report_version_id, NEW.policy_bundle_version_id, NEW.created_at)
       IS DISTINCT FROM
       (OLD.edition_id, OLD.daily_report_version_id, OLD.policy_bundle_version_id, OLD.created_at) THEN
        RAISE EXCEPTION 'video digest edition identity is immutable' USING ERRCODE = '23000';
    END IF;
    IF OLD.plan_artifact_version_id IS NOT NULL
       AND NEW.plan_artifact_version_id IS DISTINCT FROM OLD.plan_artifact_version_id THEN
        RAISE EXCEPTION 'video digest edition plan is immutable once recorded'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.plan_artifact_version_id IS NULL AND NEW.plan_artifact_version_id IS NOT NULL THEN
        SELECT count(*), min(story.position), max(story.position)
        INTO story_count, first_position, last_position
        FROM video_digest_stories AS story
        WHERE story.edition_id = NEW.edition_id;
        IF story_count = 0 OR first_position <> 0 OR last_position <> story_count - 1 THEN
            RAISE EXCEPTION 'video digest plan requires contiguous stories from position zero'
                USING ERRCODE = '23000';
        END IF;
    END IF;
    IF OLD.assembled_video_artifact_version_id IS NOT NULL
       AND NEW.assembled_video_artifact_version_id
           IS DISTINCT FROM OLD.assembled_video_artifact_version_id THEN
        RAISE EXCEPTION 'video digest assembled video is immutable once recorded'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.subtitle_state <> 'pending'
       AND (NEW.subtitle_state, NEW.subtitle_artifact_version_id,
            NEW.subtitle_failure_evidence_artifact_version_id)
           IS DISTINCT FROM
           (OLD.subtitle_state, OLD.subtitle_artifact_version_id,
            OLD.subtitle_failure_evidence_artifact_version_id) THEN
        RAISE EXCEPTION 'video digest subtitle outcome is immutable once recorded'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION protect_video_digest_slot_transition() RETURNS trigger
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
       AND OLD.lease_expires_at > CURRENT_TIMESTAMP THEN
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
       AND NEW.claim_count <> OLD.claim_count THEN
        RAISE EXCEPTION 'a video digest lease renewal cannot change the claim count'
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

CREATE FUNCTION protect_video_digest_story_transition() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.story_id, NEW.edition_id, NEW.position, NEW.report_subject_id, NEW.title,
        NEW.mandatory, NEW.requested_duration_ms, NEW.created_at)
       IS DISTINCT FROM
       (OLD.story_id, OLD.edition_id, OLD.position, OLD.report_subject_id, OLD.title,
        OLD.mandatory, OLD.requested_duration_ms, OLD.created_at) THEN
        RAISE EXCEPTION 'video digest story identity is immutable' USING ERRCODE = '23000';
    END IF;
    IF OLD.stage IN ('accepted', 'failed') THEN
        RAISE EXCEPTION 'terminal video digest story cannot change' USING ERRCODE = '23000';
    END IF;
    IF OLD.verification_evidence_artifact_version_id IS NOT NULL
       AND NEW.verification_evidence_artifact_version_id
           IS DISTINCT FROM OLD.verification_evidence_artifact_version_id THEN
        RAISE EXCEPTION 'video digest story verification is immutable once recorded'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.stage <> OLD.stage AND NOT (
        (OLD.stage = 'planned' AND NEW.stage IN ('verifying', 'failed'))
        OR (OLD.stage = 'verifying' AND NEW.stage IN ('verified', 'failed'))
        OR (OLD.stage = 'verified' AND NEW.stage IN ('generating', 'failed'))
        OR (OLD.stage = 'generating' AND NEW.stage IN ('accepted', 'failed'))
    ) THEN
        RAISE EXCEPTION 'illegal video digest story stage transition from % to %',
            OLD.stage, NEW.stage USING ERRCODE = '23000';
    END IF;
    IF OLD.stage <> 'accepted'
       AND NEW.stage = 'accepted'
       AND NOT EXISTS (
           SELECT 1 FROM video_digest_generation_requests AS request
           WHERE request.edition_id = NEW.edition_id
             AND request.story_position = NEW.position
             AND request.stage = 'accepted'
             AND request.accepted_clip_artifact_version_id
                 = NEW.accepted_clip_artifact_version_id
       ) THEN
        RAISE EXCEPTION 'accepted video digest story requires matching accepted generation'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION protect_video_digest_generation_request_transition() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.request_id, NEW.edition_id, NEW.story_position, NEW.attempt_index,
        NEW.request_artifact_version_id, NEW.created_at)
       IS DISTINCT FROM
       (OLD.request_id, OLD.edition_id, OLD.story_position, OLD.attempt_index,
        OLD.request_artifact_version_id, OLD.created_at) THEN
        RAISE EXCEPTION 'video digest generation request identity is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.provider_receipt_id IS NOT NULL
       AND NEW.provider_receipt_id IS DISTINCT FROM OLD.provider_receipt_id THEN
        RAISE EXCEPTION 'video digest provider receipt is immutable once recorded'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.response_artifact_version_id IS NOT NULL
       AND NEW.response_artifact_version_id IS DISTINCT FROM OLD.response_artifact_version_id THEN
        RAISE EXCEPTION 'video digest generation response is immutable once recorded'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.accepted_clip_artifact_version_id IS NOT NULL
       AND NEW.accepted_clip_artifact_version_id
           IS DISTINCT FROM OLD.accepted_clip_artifact_version_id THEN
        RAISE EXCEPTION 'video digest accepted clip is immutable once recorded'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.failure_evidence_artifact_version_id IS NOT NULL
       AND NEW.failure_evidence_artifact_version_id
           IS DISTINCT FROM OLD.failure_evidence_artifact_version_id THEN
        RAISE EXCEPTION 'video digest generation failure is immutable once recorded'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.cost_kind = 'measured'
       AND (NEW.cost_kind, NEW.cost_usd) IS DISTINCT FROM (OLD.cost_kind, OLD.cost_usd) THEN
        RAISE EXCEPTION 'measured video digest generation cost is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.cost_kind = 'unknown'
       AND (NEW.cost_kind, NEW.cost_unknown_reason)
           IS DISTINCT FROM (OLD.cost_kind, OLD.cost_unknown_reason) THEN
        RAISE EXCEPTION 'unknown video digest generation cost is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.cost_kind = 'estimated' AND NEW.cost_kind = 'pending' THEN
        RAISE EXCEPTION 'video digest generation cost cannot return to pending'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.stage <> OLD.stage AND NOT (
        (OLD.stage = 'pending' AND NEW.stage IN ('submitted', 'failed'))
        OR (OLD.stage = 'submitted' AND NEW.stage IN ('processing', 'failed'))
        OR (OLD.stage = 'processing' AND NEW.stage IN ('accepted', 'failed'))
    ) THEN
        RAISE EXCEPTION 'illegal video digest generation request stage transition from % to %',
            OLD.stage, NEW.stage USING ERRCODE = '23000';
    END IF;
    IF OLD.stage <> 'accepted'
       AND NEW.stage = 'accepted'
       AND NOT EXISTS (
           SELECT 1 FROM video_digest_stories AS story
           WHERE story.edition_id = NEW.edition_id
             AND story.position = NEW.story_position
             AND story.stage = 'generating'
       ) THEN
        RAISE EXCEPTION 'accepted video digest generation requires a generating story'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION protect_video_digest_publication_intent_transition() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.publication_id, NEW.edition_id, NEW.expected_video_key, NEW.video_digest,
        NEW.video_byte_size, NEW.video_media_type, NEW.subtitle_expected_key,
        NEW.subtitle_digest, NEW.subtitle_byte_size, NEW.subtitle_media_type,
        NEW.source_video_artifact_version_id, NEW.source_subtitle_artifact_version_id,
        NEW.created_at)
       IS DISTINCT FROM
       (OLD.publication_id, OLD.edition_id, OLD.expected_video_key, OLD.video_digest,
        OLD.video_byte_size, OLD.video_media_type, OLD.subtitle_expected_key,
        OLD.subtitle_digest, OLD.subtitle_byte_size, OLD.subtitle_media_type,
        OLD.source_video_artifact_version_id, OLD.source_subtitle_artifact_version_id,
        OLD.created_at) THEN
        RAISE EXCEPTION 'video digest publication intent identity is immutable'
            USING ERRCODE = '23000';
    END IF;
    IF OLD.stage IN ('published', 'conflict', 'failed') THEN
        RAISE EXCEPTION 'terminal video digest publication intent cannot change'
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

CREATE FUNCTION require_video_digest_publication_readiness() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    edition video_digest_editions%ROWTYPE;
    story_count BIGINT;
    first_position BIGINT;
    last_position BIGINT;
BEGIN
    IF NEW.stage <> 'published' THEN
        RETURN NEW;
    END IF;

    SELECT candidate.* INTO edition
    FROM video_digest_editions AS candidate
    WHERE candidate.edition_id = NEW.edition_id
    FOR UPDATE;

    IF edition.plan_artifact_version_id IS NULL
       OR edition.assembled_video_artifact_version_id IS NULL THEN
        RAISE EXCEPTION 'video digest publication requires a plan and assembled video'
            USING ERRCODE = '23000';
    END IF;
    IF NEW.source_video_artifact_version_id
       IS DISTINCT FROM edition.assembled_video_artifact_version_id THEN
        RAISE EXCEPTION 'video digest publication source video does not match the edition'
            USING ERRCODE = '23000';
    END IF;
    SELECT count(*), min(story.position), max(story.position)
    INTO story_count, first_position, last_position
    FROM video_digest_stories AS story
    WHERE story.edition_id = NEW.edition_id;
    IF story_count = 0 OR first_position <> 0 OR last_position <> story_count - 1 THEN
        RAISE EXCEPTION 'video digest publication requires contiguous stories from position zero'
            USING ERRCODE = '23000';
    END IF;
    IF EXISTS (
        SELECT 1 FROM video_digest_stories AS story
        WHERE story.edition_id = NEW.edition_id
          AND story.mandatory
          AND story.stage <> 'accepted'
    ) THEN
        RAISE EXCEPTION 'video digest publication requires every mandatory story to be accepted'
            USING ERRCODE = '23000';
    END IF;
    IF NOT (
        (edition.subtitle_state = 'available'
         AND NEW.source_subtitle_artifact_version_id
             IS NOT DISTINCT FROM edition.subtitle_artifact_version_id)
        OR (edition.subtitle_state = 'failed'
            AND NEW.source_subtitle_artifact_version_id IS NULL)
    ) THEN
        RAISE EXCEPTION 'video digest publication subtitle source does not match the edition'
            USING ERRCODE = '23000';
    END IF;
    PERFORM 1
    FROM video_digest_slots AS slot
    WHERE slot.edition_id = NEW.edition_id
      AND slot.stage = 'publishing'
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'video digest publication requires its active slot to be publishing'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE INDEX video_digest_slots_due_for_claim
    ON video_digest_slots(scheduled_at, name, slot_id)
    WHERE stage = 'scheduled';

CREATE INDEX video_digest_slots_active_leases
    ON video_digest_slots(lease_expires_at, lease_owner_token, slot_id)
    WHERE lease_owner_token IS NOT NULL;

CREATE UNIQUE INDEX video_digest_slots_one_active_edition
    ON video_digest_slots(edition_id)
    WHERE edition_id IS NOT NULL
      AND stage IN ('claimed', 'planning', 'generating', 'assembling', 'subtitling', 'publishing');

CREATE INDEX video_digest_stories_by_edition_position_stage
    ON video_digest_stories(edition_id, position, stage, mandatory);

CREATE INDEX video_digest_generation_requests_by_receipt
    ON video_digest_generation_requests(provider_receipt_id)
    WHERE provider_receipt_id IS NOT NULL;

CREATE INDEX video_digest_generation_requests_by_edition_cost
    ON video_digest_generation_requests(edition_id, cost_kind, cost_usd);

CREATE INDEX video_digest_publication_intents_published_editions
    ON video_digest_publication_intents(published_at DESC, edition_id)
    WHERE stage = 'published';

CREATE TRIGGER video_digest_editions_require_initial_state
BEFORE INSERT ON video_digest_editions
FOR EACH ROW EXECUTE FUNCTION require_video_digest_edition_initial_state();

CREATE TRIGGER video_digest_editions_protect_identity
BEFORE UPDATE ON video_digest_editions
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_edition_identity();

CREATE TRIGGER video_digest_editions_reject_deletes
BEFORE DELETE ON video_digest_editions
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_slots_require_initial_state
BEFORE INSERT ON video_digest_slots
FOR EACH ROW EXECUTE FUNCTION require_video_digest_slot_initial_state();

CREATE TRIGGER video_digest_slots_protect_transition
BEFORE UPDATE ON video_digest_slots
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_slot_transition();

CREATE TRIGGER video_digest_slots_reject_deletes
BEFORE DELETE ON video_digest_slots
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_stories_require_initial_state
BEFORE INSERT ON video_digest_stories
FOR EACH ROW EXECUTE FUNCTION require_video_digest_story_initial_state();

CREATE TRIGGER video_digest_stories_protect_transition
BEFORE UPDATE ON video_digest_stories
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_story_transition();

CREATE TRIGGER video_digest_stories_reject_deletes
BEFORE DELETE ON video_digest_stories
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_generation_requests_require_initial_state
BEFORE INSERT ON video_digest_generation_requests
FOR EACH ROW EXECUTE FUNCTION require_video_digest_generation_request_initial_state();

CREATE TRIGGER video_digest_generation_requests_protect_transition
BEFORE UPDATE ON video_digest_generation_requests
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_generation_request_transition();

CREATE TRIGGER video_digest_generation_requests_reject_deletes
BEFORE DELETE ON video_digest_generation_requests
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER video_digest_publication_intents_require_initial_state
BEFORE INSERT ON video_digest_publication_intents
FOR EACH ROW EXECUTE FUNCTION require_video_digest_publication_intent_initial_state();

CREATE TRIGGER video_digest_publication_intents_protect_transition
BEFORE UPDATE ON video_digest_publication_intents
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_publication_intent_transition();

CREATE TRIGGER video_digest_publication_intents_require_readiness
BEFORE INSERT OR UPDATE ON video_digest_publication_intents
FOR EACH ROW EXECUTE FUNCTION require_video_digest_publication_readiness();

CREATE TRIGGER video_digest_publication_intents_reject_deletes
BEFORE DELETE ON video_digest_publication_intents
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();
