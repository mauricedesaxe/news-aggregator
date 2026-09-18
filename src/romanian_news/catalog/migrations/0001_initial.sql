CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    operation_key TEXT NOT NULL,
    executor_kind TEXT NOT NULL,
    implementation_ref TEXT NOT NULL,
    parameters_json JSONB NOT NULL,
    actor TEXT NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    error TEXT,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ
);

CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    authority_class TEXT NOT NULL,
    lifecycle_state TEXT NOT NULL,
    visibility TEXT NOT NULL,
    current_version_id TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    current_run_id TEXT REFERENCES runs(id),
    CHECK (current_run_id IS NULL OR current_version_id IS NOT NULL)
);

CREATE TABLE artifact_versions (
    id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    schema_version BIGINT NOT NULL CHECK (schema_version > 0),
    content_digest TEXT NOT NULL,
    produced_by_run_id TEXT REFERENCES runs(id),
    created_at TIMESTAMPTZ NOT NULL,
    UNIQUE (artifact_id, content_digest),
    UNIQUE (artifact_id, id)
);

CREATE TABLE artifact_files (
    id TEXT PRIMARY KEY,
    artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    r2_key TEXT NOT NULL UNIQUE,
    media_type TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    byte_size BIGINT NOT NULL CHECK (byte_size >= 0),
    row_count BIGINT CHECK (row_count IS NULL OR row_count >= 0),
    schema_fingerprint TEXT
);

CREATE TABLE run_inputs (
    run_id TEXT NOT NULL REFERENCES runs(id),
    position BIGINT NOT NULL CHECK (position >= 0),
    artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    role TEXT NOT NULL,
    locator_json JSONB,
    selected_content_digest TEXT NOT NULL,
    selection_method TEXT NOT NULL,
    retrieval_metadata_json JSONB,
    PRIMARY KEY (run_id, position)
);

CREATE TABLE run_outputs (
    run_id TEXT NOT NULL REFERENCES runs(id),
    position BIGINT NOT NULL CHECK (position >= 0),
    artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    role TEXT NOT NULL,
    PRIMARY KEY (run_id, position),
    UNIQUE (run_id, artifact_version_id)
);

CREATE TABLE news_feed_observations (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    feed_snapshot_version_id TEXT REFERENCES artifact_versions(id),
    feed_id TEXT NOT NULL,
    scheduled_slot TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'not_modified', 'failed')),
    http_status BIGINT,
    entry_count BIGINT NOT NULL CHECK (entry_count >= 0),
    observed_at TIMESTAMPTZ NOT NULL,
    latency_ms BIGINT NOT NULL CHECK (latency_ms >= 0),
    error TEXT
);

CREATE TABLE news_article_aliases (
    alias_key TEXT PRIMARY KEY,
    article_artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    alias_type TEXT NOT NULL CHECK (alias_type IN ('canonical_url', 'feed_guid', 'redirect')),
    first_seen_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE news_article_identity_merges (
    duplicate_artifact_id TEXT PRIMARY KEY REFERENCES artifacts(id),
    canonical_artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    merged_at TIMESTAMPTZ NOT NULL,
    CHECK (duplicate_artifact_id <> canonical_artifact_id)
);

CREATE TABLE news_article_versions (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    article_artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    outlet_id TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    source_updated_at TIMESTAMPTZ,
    bucharest_day DATE NOT NULL,
    material_digest TEXT NOT NULL,
    extraction_digest TEXT NOT NULL,
    feed_snapshot_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    page_capture_version_id TEXT REFERENCES artifact_versions(id),
    captured_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE news_article_checks (
    article_artifact_id TEXT PRIMARY KEY REFERENCES artifacts(id),
    published_at TIMESTAMPTZ NOT NULL,
    source_updated_at TIMESTAMPTZ,
    captured_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE news_relevance_versions (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    accepted BIGINT NOT NULL CHECK (accepted IN (0, 1))
);

CREATE TABLE news_dlt_loads (
    load_id TEXT PRIMARY KEY,
    artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    registered_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE news_feed_entry_events (
    event_id TEXT PRIMARY KEY,
    dlt_load_id TEXT NOT NULL REFERENCES news_dlt_loads(load_id),
    registry_version_id TEXT NOT NULL,
    feed_snapshot_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    feed_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    original_url TEXT NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    source_updated_at TIMESTAMPTZ,
    observed_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE news_feed_entry_event_versions (
    event_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL
);

CREATE TABLE news_feed_entry_occurrences (
    event_id TEXT NOT NULL REFERENCES news_feed_entry_events(event_id),
    dlt_load_id TEXT NOT NULL REFERENCES news_dlt_loads(load_id),
    registry_version_id TEXT NOT NULL,
    feed_snapshot_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    observed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (
        event_id,
        dlt_load_id,
        registry_version_id,
        feed_snapshot_version_id,
        observed_at
    )
);

CREATE TABLE news_feed_entry_projection_loads (
    load_id TEXT PRIMARY KEY REFERENCES news_dlt_loads(load_id),
    projected_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE news_article_failure_attempts (
    attempt_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES news_feed_entry_event_versions(event_id),
    implementation_ref TEXT NOT NULL CHECK (length(trim(implementation_ref)) > 0),
    work_generation TEXT NOT NULL,
    dagster_run_id TEXT NOT NULL,
    retry_number BIGINT NOT NULL CHECK (retry_number >= 0),
    failure_kind TEXT NOT NULL CHECK (failure_kind IN ('deterministic', 'infrastructure')),
    failure_fingerprint TEXT NOT NULL,
    error TEXT NOT NULL CHECK (length(error) > 0),
    attempted_at TIMESTAMPTZ NOT NULL,
    retry_at TIMESTAMPTZ NOT NULL,
    UNIQUE (dagster_run_id, retry_number, event_id)
);

CREATE TABLE news_model_calls (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    operation_key TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens BIGINT NOT NULL CHECK (input_tokens >= 0),
    output_tokens BIGINT NOT NULL CHECK (output_tokens >= 0),
    cost_usd DOUBLE PRECISION NOT NULL CHECK (cost_usd >= 0),
    latency_ms BIGINT NOT NULL CHECK (latency_ms >= 0),
    response_count BIGINT NOT NULL CHECK (response_count > 0)
);

CREATE TABLE news_model_attempts (
    attempt_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    operation_key TEXT NOT NULL,
    response_id TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens BIGINT NOT NULL CHECK (input_tokens >= 0),
    output_tokens BIGINT NOT NULL CHECK (output_tokens >= 0),
    cost_usd DOUBLE PRECISION NOT NULL CHECK (cost_usd >= 0),
    latency_ms BIGINT NOT NULL CHECK (latency_ms >= 0),
    status TEXT NOT NULL CHECK (status IN ('accepted', 'rejected')),
    error TEXT,
    observed_at TIMESTAMPTZ NOT NULL,
    CHECK (
        (status = 'accepted' AND error IS NULL)
        OR (status = 'rejected' AND length(error) > 0)
    )
);

CREATE TABLE news_model_trace_links (
    attempt_id TEXT PRIMARY KEY REFERENCES news_model_attempts(attempt_id),
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    trace_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    project_ref TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE youtube_source_state (
    source_id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL UNIQUE,
    baseline_poll_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    last_poll_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE youtube_source_polls (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    source_id TEXT NOT NULL,
    scheduled_at TIMESTAMPTZ NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('baseline', 'current', 'overlap_lost')),
    entry_count BIGINT NOT NULL CHECK (entry_count > 0),
    UNIQUE (source_id, scheduled_at)
);

CREATE TABLE youtube_videos (
    source_id TEXT NOT NULL,
    video_id TEXT NOT NULL CHECK (length(video_id) = 11),
    first_poll_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    state TEXT NOT NULL CHECK (
        state IN ('baseline', 'pending', 'running', 'deferred', 'quarantined', 'candidate')
    ),
    published_at TIMESTAMPTZ NOT NULL,
    title TEXT NOT NULL CHECK (length(trim(title)) > 0),
    metadata_version_id TEXT REFERENCES artifact_versions(id),
    candidate_version_id TEXT REFERENCES artifact_versions(id),
    owner_token TEXT,
    lease_expires_at TIMESTAMPTZ,
    retry_at TIMESTAMPTZ,
    deterministic_failure_fingerprint TEXT,
    unchanged_deterministic_failures BIGINT NOT NULL DEFAULT 0
        CHECK (unchanged_deterministic_failures >= 0),
    last_error TEXT,
    PRIMARY KEY (source_id, video_id),
    CHECK (
        (state = 'running' AND owner_token IS NOT NULL AND lease_expires_at IS NOT NULL)
        OR (state <> 'running' AND owner_token IS NULL AND lease_expires_at IS NULL)
    ),
    CHECK (
        (state = 'deferred' AND retry_at IS NOT NULL)
        OR (state <> 'deferred' AND retry_at IS NULL)
    ),
    CHECK (
        (state = 'candidate' AND candidate_version_id IS NOT NULL)
        OR (state <> 'candidate' AND candidate_version_id IS NULL)
    ),
    CHECK (state <> 'quarantined' OR unchanged_deterministic_failures >= 3)
);

CREATE TABLE youtube_model_receipts (
    request_id TEXT NOT NULL,
    operation_key TEXT NOT NULL,
    attempt_index BIGINT NOT NULL CHECK (attempt_index >= 0),
    artifact_version_id TEXT NOT NULL UNIQUE REFERENCES artifact_versions(id),
    requested_model TEXT NOT NULL CHECK (length(trim(requested_model)) > 0),
    http_status BIGINT NOT NULL CHECK (http_status BETWEEN 100 AND 599),
    status TEXT NOT NULL CHECK (status IN ('unhandled', 'rejected', 'accepted')),
    handled_attempt_id TEXT REFERENCES news_model_attempts(attempt_id),
    error TEXT,
    received_at TIMESTAMPTZ NOT NULL,
    handled_at TIMESTAMPTZ,
    PRIMARY KEY (request_id, operation_key, attempt_index),
    CHECK (
        (status = 'unhandled' AND handled_attempt_id IS NULL AND error IS NULL AND handled_at IS NULL)
        OR (status = 'accepted' AND handled_attempt_id IS NOT NULL AND error IS NULL AND handled_at IS NOT NULL)
        OR (status = 'rejected' AND length(error) > 0 AND handled_at IS NOT NULL)
    )
);

CREATE TABLE youtube_clip_analyses (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    source_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    metadata_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    start_second BIGINT NOT NULL CHECK (start_second >= 0),
    duration_seconds BIGINT NOT NULL CHECK (duration_seconds > 0 AND duration_seconds <= 300),
    model TEXT NOT NULL CHECK (length(trim(model)) > 0),
    request_digest TEXT NOT NULL,
    model_attempt_id TEXT NOT NULL REFERENCES news_model_attempts(attempt_id),
    accepted_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (source_id, video_id) REFERENCES youtube_videos(source_id, video_id),
    UNIQUE (source_id, video_id, metadata_version_id, start_second, duration_seconds, model, request_digest)
);

CREATE TABLE youtube_merge_analyses (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    source_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    request_id TEXT NOT NULL UNIQUE,
    model_attempt_id TEXT NOT NULL REFERENCES news_model_attempts(attempt_id),
    accepted_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (source_id, video_id) REFERENCES youtube_videos(source_id, video_id)
);

CREATE TABLE youtube_candidates (
    candidate_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    source_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    policy TEXT NOT NULL CHECK (policy IN ('automatic', 'owner_review')),
    metadata_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    merge_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    owner_token TEXT NOT NULL CHECK (length(trim(owner_token)) > 0),
    created_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (source_id, video_id) REFERENCES youtube_videos(source_id, video_id),
    UNIQUE (source_id, video_id)
);

CREATE TABLE youtube_candidate_decisions (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    candidate_version_id TEXT NOT NULL UNIQUE REFERENCES youtube_candidates(candidate_version_id),
    source_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('automatic_approved', 'owner_approved', 'owner_rejected')),
    actor TEXT NOT NULL CHECK (actor IN ('system', 'owner')),
    note TEXT CHECK (note IS NULL OR (length(trim(note)) > 0 AND length(note) <= 2000)),
    decided_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE youtube_publications (
    candidate_version_id TEXT PRIMARY KEY REFERENCES youtube_candidates(candidate_version_id),
    source_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    article_version_id TEXT NOT NULL UNIQUE REFERENCES artifact_versions(id),
    published_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (source_id, video_id) REFERENCES youtube_videos(source_id, video_id)
);

CREATE TABLE news_feedback (
    feedback_id TEXT PRIMARY KEY,
    report_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    target_kind TEXT NOT NULL CHECK (target_kind IN ('report', 'theme', 'group', 'article')),
    theme_id TEXT,
    group_id TEXT,
    article_version_id TEXT REFERENCES artifact_versions(id),
    rating TEXT CHECK (rating IN ('positive', 'negative')),
    note TEXT CHECK (note IS NULL OR length(note) <= 2000),
    actor TEXT NOT NULL CHECK (actor = 'owner'),
    created_at TIMESTAMPTZ NOT NULL,
    UNIQUE (feedback_id, report_version_id),
    CHECK (
        (target_kind = 'report' AND theme_id IS NULL AND group_id IS NULL
            AND article_version_id IS NULL)
        OR (target_kind = 'theme' AND theme_id IS NOT NULL AND group_id IS NULL
            AND article_version_id IS NULL)
        OR (target_kind = 'group' AND theme_id IS NULL AND group_id IS NOT NULL
            AND article_version_id IS NULL)
        OR (target_kind = 'article' AND theme_id IS NULL AND group_id IS NOT NULL
            AND article_version_id IS NOT NULL)
    ),
    CHECK (rating IS NOT NULL OR (note IS NOT NULL AND length(trim(note)) > 0))
);

CREATE TABLE news_evaluation_feedback_sources (
    evaluation_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    position BIGINT NOT NULL CHECK (position >= 0),
    feedback_id TEXT NOT NULL REFERENCES news_feedback(feedback_id),
    disposition TEXT NOT NULL CHECK (disposition IN ('represented', 'excluded')),
    PRIMARY KEY (evaluation_artifact_version_id, position),
    UNIQUE (evaluation_artifact_version_id, feedback_id)
);

CREATE TABLE news_feedback_score_curation (
    manifest_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    manifest_version TEXT NOT NULL CHECK (length(trim(manifest_version)) > 0),
    position BIGINT NOT NULL CHECK (position >= 0),
    feedback_id TEXT NOT NULL REFERENCES news_feedback(feedback_id),
    concern TEXT CHECK (concern IN ('language', 'relevance', 'grouping', 'ranking')),
    exclusion_reason TEXT CHECK (exclusion_reason IN (
        'presentation', 'operations', 'ambiguous', 'contradictory', 'superseded',
        'no_model_observation', 'no_concern_specific_observation'
    )),
    decision TEXT NOT NULL CHECK (decision IN ('projected', 'excluded')),
    report_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    model_output_version_id TEXT REFERENCES artifact_versions(id),
    model_attempt_id TEXT REFERENCES news_model_attempts(attempt_id),
    polarity TEXT CHECK (polarity IN ('positive', 'negative')),
    rationale TEXT NOT NULL CHECK (length(trim(rationale)) > 0),
    score_id TEXT UNIQUE,
    PRIMARY KEY (manifest_artifact_version_id, position),
    FOREIGN KEY (feedback_id, report_version_id)
        REFERENCES news_feedback(feedback_id, report_version_id),
    CHECK (
        (decision = 'projected'
         AND concern IS NOT NULL
         AND exclusion_reason IS NULL
         AND model_output_version_id IS NOT NULL
         AND model_attempt_id IS NOT NULL
         AND polarity IS NOT NULL
         AND score_id IS NOT NULL)
        OR (decision = 'excluded'
            AND exclusion_reason IS NOT NULL
            AND model_output_version_id IS NULL
            AND model_attempt_id IS NULL
            AND score_id IS NULL
            AND (
                (exclusion_reason = 'no_model_observation'
                 AND concern IN ('grouping', 'ranking'))
                OR (exclusion_reason = 'no_concern_specific_observation'
                    AND concern IN ('language', 'grouping')
                    AND polarity IS NOT NULL)
                OR (exclusion_reason NOT IN (
                        'no_model_observation', 'no_concern_specific_observation'
                    )
                    AND concern IS NULL
                    AND polarity IS NULL)
            ))
    )
);

CREATE TABLE news_feedback_concern_score_sync_attempts (
    sync_attempt_id TEXT PRIMARY KEY,
    score_id TEXT NOT NULL REFERENCES news_feedback_score_curation(score_id),
    provider TEXT NOT NULL CHECK (provider = 'langfuse'),
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
    error TEXT,
    attempted_at TIMESTAMPTZ NOT NULL,
    CHECK (
        (status = 'completed' AND error IS NULL)
        OR (status = 'failed' AND length(error) > 0)
    )
);

CREATE TABLE news_feedback_concern_score_dispositions (
    score_id TEXT NOT NULL REFERENCES news_feedback_score_curation(score_id),
    provider TEXT NOT NULL CHECK (provider = 'langfuse'),
    disposition TEXT NOT NULL CHECK (disposition = 'permanently_unresolved'),
    reason TEXT NOT NULL CHECK (reason = 'provider_migration'),
    source_provider TEXT NOT NULL CHECK (source_provider = 'langsmith'),
    recorded_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (score_id, provider)
);

CREATE TABLE news_evaluation_projections (
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    manifest_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    projection_kind TEXT NOT NULL CHECK (projection_kind IN ('dataset', 'experiment')),
    dataset_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    experiment_id TEXT,
    experiment_name TEXT,
    experiment_url TEXT,
    implementation_ref TEXT NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (provider, manifest_artifact_version_id, projection_kind, implementation_ref),
    CHECK (
        (projection_kind = 'dataset'
         AND implementation_ref = ''
         AND experiment_id IS NULL
         AND experiment_name IS NULL
         AND experiment_url IS NULL)
        OR (projection_kind = 'experiment'
            AND length(trim(implementation_ref)) > 0
            AND experiment_id IS NOT NULL
            AND experiment_name IS NOT NULL)
    )
);

CREATE TABLE news_relevance_experiments (
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    manifest_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    policy_digest TEXT NOT NULL,
    implementation_ref TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    experiment_url TEXT,
    completed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (provider, manifest_artifact_version_id, policy_digest, implementation_ref),
    CHECK (length(trim(policy_digest)) > 0),
    CHECK (length(trim(implementation_ref)) > 0),
    CHECK (length(trim(policy_id)) > 0)
);

CREATE TABLE news_theme_experiments (
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    manifest_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    policy_digest TEXT NOT NULL,
    implementation_ref TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    experiment_url TEXT,
    completed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (provider, manifest_artifact_version_id, policy_digest, implementation_ref),
    CHECK (length(trim(policy_digest)) > 0),
    CHECK (length(trim(implementation_ref)) > 0),
    CHECK (length(trim(policy_id)) > 0)
);

CREATE TABLE news_feedback_sync_attempts (
    sync_attempt_id TEXT PRIMARY KEY,
    feedback_id TEXT NOT NULL REFERENCES news_feedback(feedback_id),

    model_attempt_id TEXT NOT NULL REFERENCES news_model_attempts(attempt_id),
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    score_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
    error TEXT,
    attempted_at TIMESTAMPTZ NOT NULL,
    CHECK (
        (status = 'completed' AND error IS NULL)
        OR (status = 'failed' AND length(error) > 0)
    )
);

CREATE TABLE news_feedback_sync_dispositions (
    feedback_id TEXT NOT NULL REFERENCES news_feedback(feedback_id),
    provider TEXT NOT NULL CHECK (provider = 'langfuse'),
    disposition TEXT NOT NULL CHECK (disposition = 'permanently_unresolved'),
    reason TEXT NOT NULL CHECK (reason = 'provider_migration'),
    source_provider TEXT NOT NULL CHECK (source_provider = 'langsmith'),
    recorded_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (feedback_id, provider)
);

ALTER TABLE artifacts
    ADD CONSTRAINT artifacts_current_version_owned
    FOREIGN KEY (id, current_version_id)
    REFERENCES artifact_versions (artifact_id, id)
    DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE artifacts
    ADD CONSTRAINT artifacts_current_run_output
    FOREIGN KEY (current_run_id, current_version_id)
    REFERENCES run_outputs (run_id, artifact_version_id)
    DEFERRABLE INITIALLY DEFERRED;

CREATE FUNCTION reject_immutable_catalog_change() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% is immutable', TG_TABLE_NAME USING ERRCODE = '23000';
END;
$$;

CREATE FUNCTION project_catalog_columns(document JSONB, columns_csv TEXT) RETURNS JSONB
LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(jsonb_object_agg(column_name, document -> column_name), '{}'::JSONB)
    FROM unnest(string_to_array(columns_csv, ',')) AS column_name;
$$;

CREATE FUNCTION reject_catalog_identity_conflict() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    existing_row JSONB;
    new_row JSONB := to_jsonb(NEW);
    selector TEXT;
    selector_values JSONB;
BEGIN
    FOREACH selector IN ARRAY TG_ARGV[1:] LOOP
        selector_values := project_catalog_columns(new_row, selector);
        IF EXISTS (SELECT 1 FROM jsonb_each(selector_values) WHERE value = 'null'::JSONB) THEN
            CONTINUE;
        END IF;
        EXECUTE format(
            'SELECT to_jsonb(candidate) FROM %I AS candidate '
            'WHERE to_jsonb(candidate) @> $1 LIMIT 1',
            TG_TABLE_NAME
        )
        INTO existing_row
        USING selector_values;
        EXIT WHEN existing_row IS NOT NULL;
    END LOOP;

    IF existing_row IS NOT NULL
       AND project_catalog_columns(existing_row, TG_ARGV[0])
           IS DISTINCT FROM project_catalog_columns(new_row, TG_ARGV[0]) THEN
        RAISE EXCEPTION '% identity conflict', TG_TABLE_NAME USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION protect_artifact_identity() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.id, NEW.kind, NEW.authority_class, NEW.created_at)
       IS DISTINCT FROM (OLD.id, OLD.kind, OLD.authority_class, OLD.created_at) THEN
        RAISE EXCEPTION 'artifact identity is immutable' USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION protect_run_identity() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.id, NEW.operation_key, NEW.executor_kind, NEW.implementation_ref,
        NEW.parameters_json, NEW.actor, NEW.idempotency_key, NEW.started_at)
       IS DISTINCT FROM
       (OLD.id, OLD.operation_key, OLD.executor_kind, OLD.implementation_ref,
        OLD.parameters_json, OLD.actor, OLD.idempotency_key, OLD.started_at) THEN
        RAISE EXCEPTION 'run identity is immutable' USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION protect_feed_entry_event_version() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.event_id IS DISTINCT FROM OLD.event_id
       OR (OLD.version_id IS DISTINCT FROM OLD.event_id
           AND NEW.version_id IS DISTINCT FROM OLD.version_id) THEN
        RAISE EXCEPTION 'news_feed_entry_event_versions identity conflict'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION require_youtube_candidate_lease() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM youtube_videos AS video
        WHERE video.source_id = NEW.source_id
          AND video.video_id = NEW.video_id
          AND video.state = 'running'
          AND video.owner_token = NEW.owner_token
          AND video.lease_expires_at > NEW.created_at
    ) THEN
        RAISE EXCEPTION 'YouTube candidate lease was lost' USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION require_youtube_candidate_policy() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM youtube_candidates AS candidate
        WHERE candidate.candidate_version_id = NEW.candidate_version_id
          AND candidate.source_id = NEW.source_id
          AND (
              (candidate.policy = 'automatic' AND NEW.kind = 'automatic_approved'
                  AND NEW.actor = 'system' AND NEW.note IS NULL)
              OR (candidate.policy = 'owner_review'
                  AND NEW.kind IN ('owner_approved', 'owner_rejected')
                  AND NEW.actor = 'owner')
          )
    ) THEN
        RAISE EXCEPTION 'YouTube candidate decision violates source policy'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION require_youtube_publication_approval() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM youtube_candidates AS candidate
        JOIN youtube_candidate_decisions AS decision
          ON decision.candidate_version_id = candidate.candidate_version_id
        WHERE candidate.candidate_version_id = NEW.candidate_version_id
          AND candidate.source_id = NEW.source_id
          AND candidate.video_id = NEW.video_id
          AND decision.kind IN ('automatic_approved', 'owner_approved')
    ) THEN
        RAISE EXCEPTION 'YouTube publication requires exact approval' USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE INDEX artifact_versions_by_run
    ON artifact_versions(produced_by_run_id);

CREATE INDEX artifacts_by_kind_current_version
    ON artifacts(kind, current_version_id);

CREATE INDEX artifacts_by_current_version
    ON artifacts(current_version_id);

CREATE INDEX artifact_files_by_version
    ON artifact_files(artifact_version_id);

CREATE INDEX run_inputs_by_version
    ON run_inputs(artifact_version_id);

CREATE INDEX run_outputs_by_version
    ON run_outputs(artifact_version_id);

CREATE INDEX news_feed_entry_events_by_publication
    ON news_feed_entry_events(published_at, feed_id, event_id);

CREATE INDEX news_article_versions_by_day
    ON news_article_versions(bucharest_day, artifact_version_id);

CREATE INDEX news_article_failure_attempts_by_ref_event_generation_time
    ON news_article_failure_attempts(
        implementation_ref,
        event_id,
        work_generation,
        attempted_at,
        attempt_id
    );

CREATE INDEX youtube_videos_by_state_time
    ON youtube_videos(source_id, state, retry_at, lease_expires_at, published_at);

CREATE INDEX youtube_model_receipts_by_request_status
    ON youtube_model_receipts(request_id, operation_key, status, attempt_index);

CREATE INDEX news_feedback_by_report_target_time
    ON news_feedback(
        report_version_id,
        target_kind,
        theme_id,
        group_id,
        article_version_id,
        created_at DESC
    );

CREATE INDEX news_feedback_score_curation_by_manifest_decision
    ON news_feedback_score_curation(manifest_artifact_version_id, decision, position);

CREATE INDEX news_feedback_concern_score_sync_attempts_by_score_status
    ON news_feedback_concern_score_sync_attempts(provider, score_id, status);

CREATE INDEX news_feedback_sync_attempts_by_pair_status
    ON news_feedback_sync_attempts(provider, feedback_id, model_attempt_id, status);

CREATE INDEX news_feedback_sync_attempts_by_time
    ON news_feedback_sync_attempts(attempted_at);

CREATE INDEX news_feedback_sync_dispositions_by_provider
    ON news_feedback_sync_dispositions(provider, disposition, feedback_id);

CREATE TRIGGER artifacts_reject_conflicting_inserts
BEFORE INSERT ON artifacts
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('id,kind,authority_class', 'id');

CREATE TRIGGER runs_reject_idempotency_conflicts
BEFORE INSERT ON runs
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('id,operation_key,executor_kind,implementation_ref,parameters_json,actor,idempotency_key,started_at', 'id', 'idempotency_key');

CREATE TRIGGER artifact_versions_reject_conflicting_inserts
BEFORE INSERT ON artifact_versions
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('id,artifact_id,schema_version,content_digest', 'id', 'artifact_id,content_digest');

CREATE TRIGGER artifact_files_reject_conflicting_inserts
BEFORE INSERT ON artifact_files
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('id,artifact_version_id,r2_key,media_type,content_digest,byte_size,row_count,schema_fingerprint', 'id', 'r2_key');

CREATE TRIGGER run_inputs_reject_conflicting_inserts
BEFORE INSERT ON run_inputs
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('run_id,position,artifact_version_id,role,locator_json,selected_content_digest,selection_method,retrieval_metadata_json', 'run_id,position');

CREATE TRIGGER run_outputs_reject_conflicting_inserts
BEFORE INSERT ON run_outputs
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('run_id,position,artifact_version_id,role', 'run_id,position', 'run_id,artifact_version_id');

CREATE TRIGGER news_article_aliases_reject_conflicting_inserts
BEFORE INSERT ON news_article_aliases
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('alias_key,article_artifact_id,alias_type', 'alias_key');

CREATE TRIGGER news_article_failure_attempts_reject_conflicting_inserts
BEFORE INSERT ON news_article_failure_attempts
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('attempt_id,event_id,implementation_ref,work_generation,dagster_run_id,retry_number,failure_kind,failure_fingerprint,error,attempted_at,retry_at', 'attempt_id', 'dagster_run_id,retry_number,event_id');

CREATE TRIGGER news_feed_entry_events_reject_conflicting_inserts
BEFORE INSERT ON news_feed_entry_events
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('event_id,feed_id,source_id,original_url,published_at,source_updated_at', 'event_id');

CREATE TRIGGER news_feed_entry_projection_loads_reject_conflicting_inserts
BEFORE INSERT ON news_feed_entry_projection_loads
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('load_id,projected_at', 'load_id');

CREATE TRIGGER news_model_attempts_reject_conflicting_inserts
BEFORE INSERT ON news_model_attempts
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('attempt_id,request_id,operation_key,response_id,model,input_tokens,output_tokens,cost_usd,latency_ms,status,error,observed_at', 'attempt_id');

CREATE TRIGGER news_model_trace_links_reject_conflicting_inserts
BEFORE INSERT ON news_model_trace_links
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('attempt_id,provider,trace_id,observation_id,project_ref,recorded_at', 'attempt_id');

CREATE TRIGGER youtube_source_polls_reject_conflicting_inserts
BEFORE INSERT ON youtube_source_polls
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('artifact_version_id,source_id,scheduled_at,observed_at,status,entry_count', 'artifact_version_id', 'source_id,scheduled_at');

CREATE TRIGGER youtube_model_receipts_reject_conflicting_inserts
BEFORE INSERT ON youtube_model_receipts
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('request_id,operation_key,attempt_index,artifact_version_id,requested_model,http_status,received_at', 'request_id,operation_key,attempt_index', 'artifact_version_id');

CREATE TRIGGER youtube_clip_analyses_reject_conflicting_inserts
BEFORE INSERT ON youtube_clip_analyses
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('artifact_version_id,source_id,video_id,metadata_version_id,start_second,duration_seconds,model,request_digest,model_attempt_id', 'artifact_version_id', 'source_id,video_id,metadata_version_id,start_second,duration_seconds,model,request_digest');

CREATE TRIGGER youtube_merge_analyses_reject_conflicting_inserts
BEFORE INSERT ON youtube_merge_analyses
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('artifact_version_id,source_id,video_id,request_id,model_attempt_id', 'artifact_version_id', 'request_id');

CREATE TRIGGER youtube_candidates_reject_conflicting_inserts
BEFORE INSERT ON youtube_candidates
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('candidate_version_id,source_id,video_id,policy,metadata_version_id,merge_version_id', 'candidate_version_id', 'source_id,video_id');

CREATE TRIGGER youtube_candidate_decisions_reject_conflicts
BEFORE INSERT ON youtube_candidate_decisions
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('candidate_version_id,source_id,kind,actor,note', 'candidate_version_id');

CREATE TRIGGER youtube_publications_reject_conflicting_inserts
BEFORE INSERT ON youtube_publications
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('candidate_version_id,source_id,video_id,article_version_id', 'candidate_version_id', 'article_version_id');

CREATE TRIGGER news_feedback_reject_conflicting_inserts
BEFORE INSERT ON news_feedback
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('feedback_id,report_version_id,target_kind,theme_id,group_id,article_version_id,rating,note,actor', 'feedback_id');

CREATE TRIGGER news_evaluation_feedback_sources_reject_conflicting_inserts
BEFORE INSERT ON news_evaluation_feedback_sources
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('evaluation_artifact_version_id,position,feedback_id,disposition', 'evaluation_artifact_version_id,position', 'evaluation_artifact_version_id,feedback_id');

CREATE TRIGGER news_feedback_score_curation_reject_conflicting_inserts
BEFORE INSERT ON news_feedback_score_curation
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('manifest_artifact_version_id,manifest_version,position,feedback_id,concern,exclusion_reason,decision,report_version_id,model_output_version_id,model_attempt_id,polarity,rationale,score_id', 'manifest_artifact_version_id,position', 'score_id');

CREATE TRIGGER news_evaluation_projections_reject_conflicting_inserts
BEFORE INSERT ON news_evaluation_projections
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('provider,manifest_artifact_version_id,projection_kind,dataset_id,dataset_name,experiment_id,experiment_name,experiment_url,implementation_ref', 'provider,manifest_artifact_version_id,projection_kind,implementation_ref');

CREATE TRIGGER news_relevance_experiments_reject_conflicting_inserts
BEFORE INSERT ON news_relevance_experiments
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('provider,manifest_artifact_version_id,policy_digest,implementation_ref,policy_id,dataset_id,dataset_name,experiment_id,experiment_name,experiment_url', 'provider,manifest_artifact_version_id,policy_digest,implementation_ref');

CREATE TRIGGER news_theme_experiments_reject_conflicting_inserts
BEFORE INSERT ON news_theme_experiments
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('provider,manifest_artifact_version_id,policy_digest,implementation_ref,policy_id,dataset_id,dataset_name,experiment_id,experiment_name,experiment_url', 'provider,manifest_artifact_version_id,policy_digest,implementation_ref');

CREATE TRIGGER news_feedback_sync_attempts_reject_conflicting_inserts
BEFORE INSERT ON news_feedback_sync_attempts
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('sync_attempt_id,feedback_id,model_attempt_id,provider,score_id,status,error,attempted_at', 'sync_attempt_id');

CREATE TRIGGER news_feedback_sync_dispositions_reject_conflicting_inserts
BEFORE INSERT ON news_feedback_sync_dispositions
FOR EACH ROW EXECUTE FUNCTION reject_catalog_identity_conflict('feedback_id,provider,disposition,reason,source_provider', 'feedback_id,provider');

CREATE TRIGGER artifacts_protect_identity
BEFORE UPDATE ON artifacts
FOR EACH ROW EXECUTE FUNCTION protect_artifact_identity();

CREATE TRIGGER runs_protect_identity
BEFORE UPDATE ON runs
FOR EACH ROW EXECUTE FUNCTION protect_run_identity();

CREATE TRIGGER news_feed_entry_event_versions_reject_conflicting_updates
BEFORE UPDATE ON news_feed_entry_event_versions
FOR EACH ROW EXECUTE FUNCTION protect_feed_entry_event_version();

CREATE TRIGGER news_feed_entry_event_versions_reject_deletes
BEFORE DELETE ON news_feed_entry_event_versions
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER artifact_versions_reject_updates
BEFORE UPDATE ON artifact_versions
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER artifact_versions_reject_deletes
BEFORE DELETE ON artifact_versions
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER artifact_files_reject_updates
BEFORE UPDATE ON artifact_files
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER artifact_files_reject_deletes
BEFORE DELETE ON artifact_files
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER run_inputs_reject_updates
BEFORE UPDATE ON run_inputs
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER run_inputs_reject_deletes
BEFORE DELETE ON run_inputs
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER run_outputs_reject_updates
BEFORE UPDATE ON run_outputs
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER run_outputs_reject_deletes
BEFORE DELETE ON run_outputs
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feed_observations_reject_updates
BEFORE UPDATE ON news_feed_observations
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feed_observations_reject_deletes
BEFORE DELETE ON news_feed_observations
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_article_aliases_reject_updates
BEFORE UPDATE ON news_article_aliases
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_article_aliases_reject_deletes
BEFORE DELETE ON news_article_aliases
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_article_versions_reject_updates
BEFORE UPDATE ON news_article_versions
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_article_versions_reject_deletes
BEFORE DELETE ON news_article_versions
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_article_failure_attempts_reject_updates
BEFORE UPDATE ON news_article_failure_attempts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_article_failure_attempts_reject_deletes
BEFORE DELETE ON news_article_failure_attempts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_dlt_loads_reject_updates
BEFORE UPDATE ON news_dlt_loads
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_dlt_loads_reject_deletes
BEFORE DELETE ON news_dlt_loads
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feed_entry_events_reject_updates
BEFORE UPDATE ON news_feed_entry_events
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feed_entry_events_reject_deletes
BEFORE DELETE ON news_feed_entry_events
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feed_entry_occurrences_reject_updates
BEFORE UPDATE ON news_feed_entry_occurrences
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feed_entry_occurrences_reject_deletes
BEFORE DELETE ON news_feed_entry_occurrences
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feed_entry_projection_loads_reject_updates
BEFORE UPDATE ON news_feed_entry_projection_loads
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feed_entry_projection_loads_reject_deletes
BEFORE DELETE ON news_feed_entry_projection_loads
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_model_calls_reject_updates
BEFORE UPDATE ON news_model_calls
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_model_calls_reject_deletes
BEFORE DELETE ON news_model_calls
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_model_attempts_reject_updates
BEFORE UPDATE ON news_model_attempts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_model_attempts_reject_deletes
BEFORE DELETE ON news_model_attempts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_model_trace_links_reject_updates
BEFORE UPDATE ON news_model_trace_links
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_model_trace_links_reject_deletes
BEFORE DELETE ON news_model_trace_links
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER youtube_candidates_require_lease
BEFORE INSERT ON youtube_candidates
FOR EACH ROW EXECUTE FUNCTION require_youtube_candidate_lease();

CREATE TRIGGER youtube_candidate_decisions_require_candidate_policy
BEFORE INSERT ON youtube_candidate_decisions
FOR EACH ROW EXECUTE FUNCTION require_youtube_candidate_policy();

CREATE TRIGGER youtube_publications_require_approval
BEFORE INSERT ON youtube_publications
FOR EACH ROW EXECUTE FUNCTION require_youtube_publication_approval();

CREATE TRIGGER youtube_source_polls_reject_updates
BEFORE UPDATE ON youtube_source_polls
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER youtube_source_polls_reject_deletes
BEFORE DELETE ON youtube_source_polls
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER youtube_clip_analyses_reject_updates
BEFORE UPDATE ON youtube_clip_analyses
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER youtube_clip_analyses_reject_deletes
BEFORE DELETE ON youtube_clip_analyses
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER youtube_merge_analyses_reject_updates
BEFORE UPDATE ON youtube_merge_analyses
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER youtube_merge_analyses_reject_deletes
BEFORE DELETE ON youtube_merge_analyses
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER youtube_candidates_reject_updates
BEFORE UPDATE ON youtube_candidates
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER youtube_candidates_reject_deletes
BEFORE DELETE ON youtube_candidates
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER youtube_candidate_decisions_reject_updates
BEFORE UPDATE ON youtube_candidate_decisions
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER youtube_candidate_decisions_reject_deletes
BEFORE DELETE ON youtube_candidate_decisions
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER youtube_publications_reject_updates
BEFORE UPDATE ON youtube_publications
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER youtube_publications_reject_deletes
BEFORE DELETE ON youtube_publications
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feedback_reject_updates
BEFORE UPDATE ON news_feedback
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feedback_reject_deletes
BEFORE DELETE ON news_feedback
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_evaluation_feedback_sources_reject_updates
BEFORE UPDATE ON news_evaluation_feedback_sources
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_evaluation_feedback_sources_reject_deletes
BEFORE DELETE ON news_evaluation_feedback_sources
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feedback_score_curation_reject_updates
BEFORE UPDATE ON news_feedback_score_curation
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feedback_score_curation_reject_deletes
BEFORE DELETE ON news_feedback_score_curation
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feedback_concern_score_sync_attempts_reject_updates
BEFORE UPDATE ON news_feedback_concern_score_sync_attempts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feedback_concern_score_sync_attempts_reject_deletes
BEFORE DELETE ON news_feedback_concern_score_sync_attempts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feedback_concern_score_dispositions_reject_updates
BEFORE UPDATE ON news_feedback_concern_score_dispositions
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feedback_concern_score_dispositions_reject_deletes
BEFORE DELETE ON news_feedback_concern_score_dispositions
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_evaluation_projections_reject_updates
BEFORE UPDATE ON news_evaluation_projections
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_evaluation_projections_reject_deletes
BEFORE DELETE ON news_evaluation_projections
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_relevance_experiments_reject_updates
BEFORE UPDATE ON news_relevance_experiments
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_relevance_experiments_reject_deletes
BEFORE DELETE ON news_relevance_experiments
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_theme_experiments_reject_updates
BEFORE UPDATE ON news_theme_experiments
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_theme_experiments_reject_deletes
BEFORE DELETE ON news_theme_experiments
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feedback_sync_attempts_reject_updates
BEFORE UPDATE ON news_feedback_sync_attempts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feedback_sync_attempts_reject_deletes
BEFORE DELETE ON news_feedback_sync_attempts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feedback_sync_dispositions_reject_updates
BEFORE UPDATE ON news_feedback_sync_dispositions
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_feedback_sync_dispositions_reject_deletes
BEFORE DELETE ON news_feedback_sync_dispositions
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TABLE news_schema_migrations (
    version BIGINT PRIMARY KEY CHECK (version > 0),
    name TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
