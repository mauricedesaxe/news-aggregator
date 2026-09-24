PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    authority_class TEXT NOT NULL,
    lifecycle_state TEXT NOT NULL,
    visibility TEXT NOT NULL,
    current_version_id TEXT,
    created_at TEXT NOT NULL,
    current_run_id TEXT REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    operation_key TEXT NOT NULL,
    executor_kind TEXT NOT NULL,
    implementation_ref TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    actor TEXT NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    error TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS artifact_versions (
    id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    schema_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    produced_by_run_id TEXT REFERENCES runs(id),
    created_at TEXT NOT NULL,
    UNIQUE (artifact_id, content_digest)
);

CREATE TABLE IF NOT EXISTS artifact_files (
    id TEXT PRIMARY KEY,
    artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    r2_key TEXT NOT NULL UNIQUE,
    media_type TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    row_count INTEGER,
    schema_fingerprint TEXT
);

CREATE TABLE IF NOT EXISTS run_inputs (
    run_id TEXT NOT NULL REFERENCES runs(id),
    position INTEGER NOT NULL,
    artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    role TEXT NOT NULL,
    locator_json TEXT,
    selected_content_digest TEXT NOT NULL,
    selection_method TEXT NOT NULL,
    retrieval_metadata_json TEXT,
    PRIMARY KEY (run_id, position)
);

CREATE TABLE IF NOT EXISTS run_outputs (
    run_id TEXT NOT NULL REFERENCES runs(id),
    position INTEGER NOT NULL,
    artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    role TEXT NOT NULL,
    PRIMARY KEY (run_id, position)
);

CREATE TABLE IF NOT EXISTS news_feed_observations (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    feed_snapshot_version_id TEXT REFERENCES artifact_versions(id),
    feed_id TEXT NOT NULL,
    scheduled_slot TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'not_modified', 'failed')),
    http_status INTEGER,
    entry_count INTEGER NOT NULL CHECK (entry_count >= 0),
    observed_at TEXT NOT NULL,
    latency_ms INTEGER NOT NULL CHECK (latency_ms >= 0),
    error TEXT
);

CREATE TABLE IF NOT EXISTS news_article_aliases (
    alias_key TEXT PRIMARY KEY,
    article_artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    alias_type TEXT NOT NULL CHECK (alias_type IN ('canonical_url', 'feed_guid', 'redirect')),
    first_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS news_article_identity_merges (
    duplicate_artifact_id TEXT PRIMARY KEY REFERENCES artifacts(id),
    canonical_artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    merged_at TEXT NOT NULL,
    CHECK (duplicate_artifact_id != canonical_artifact_id)
);

CREATE TABLE IF NOT EXISTS news_article_versions (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    article_artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    outlet_id TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    published_at TEXT NOT NULL,
    source_updated_at TEXT,
    bucharest_day TEXT NOT NULL,
    material_digest TEXT NOT NULL,
    extraction_digest TEXT NOT NULL,
    feed_snapshot_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    page_capture_version_id TEXT REFERENCES artifact_versions(id),
    captured_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS news_article_checks (
    article_artifact_id TEXT PRIMARY KEY REFERENCES artifacts(id),
    published_at TEXT NOT NULL,
    source_updated_at TEXT,
    captured_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS news_article_failure_attempts (
    attempt_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES news_feed_entry_event_versions(event_id),
    implementation_ref TEXT NOT NULL CHECK (length(trim(implementation_ref)) > 0),
    work_generation TEXT NOT NULL,
    dagster_run_id TEXT NOT NULL,
    retry_number INTEGER NOT NULL CHECK (retry_number >= 0),
    failure_kind TEXT NOT NULL CHECK (failure_kind IN ('deterministic', 'infrastructure')),
    failure_fingerprint TEXT NOT NULL,
    error TEXT NOT NULL CHECK (length(error) > 0),
    attempted_at TEXT NOT NULL,
    retry_at TEXT NOT NULL,
    UNIQUE (dagster_run_id, retry_number, event_id)
);

CREATE TABLE IF NOT EXISTS news_article_recovery_overrides (
    recovery_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    recovery_id TEXT NOT NULL UNIQUE,
    event_id TEXT NOT NULL REFERENCES news_feed_entry_event_versions(event_id),
    base_work_generation TEXT NOT NULL,
    expected_work_generation TEXT NOT NULL,
    requested_by TEXT NOT NULL CHECK (length(trim(requested_by)) > 0),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    requested_at TEXT NOT NULL,
    UNIQUE (event_id, base_work_generation, expected_work_generation)
);

CREATE TABLE IF NOT EXISTS daily_report_repair_requests (
    day TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    run_id TEXT UNIQUE,
    launch_state TEXT NOT NULL DEFAULT 'reserved'
        CHECK (launch_state IN ('reserved', 'launch_started'))
);

CREATE TABLE IF NOT EXISTS news_relevance_versions (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    accepted INTEGER NOT NULL CHECK (accepted IN (0, 1))
);

CREATE TABLE IF NOT EXISTS news_dlt_loads (
    load_id TEXT PRIMARY KEY,
    artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    registered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS news_feed_entry_events (
    event_id TEXT PRIMARY KEY,
    dlt_load_id TEXT NOT NULL REFERENCES news_dlt_loads(load_id),
    registry_version_id TEXT NOT NULL,
    feed_snapshot_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    feed_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    original_url TEXT NOT NULL,
    published_at TEXT NOT NULL,
    source_updated_at TEXT,
    observed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS news_feed_entry_event_versions (
    event_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS news_feed_entry_occurrences (
    event_id TEXT NOT NULL REFERENCES news_feed_entry_events(event_id),
    dlt_load_id TEXT NOT NULL REFERENCES news_dlt_loads(load_id),
    registry_version_id TEXT NOT NULL,
    feed_snapshot_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    observed_at TEXT NOT NULL,
    PRIMARY KEY (
        event_id,
        dlt_load_id,
        registry_version_id,
        feed_snapshot_version_id,
        observed_at
    )
);

INSERT OR IGNORE INTO news_feed_entry_event_versions
SELECT event_id, event_id FROM news_feed_entry_events;

INSERT OR IGNORE INTO news_feed_entry_occurrences
SELECT event_id, dlt_load_id, registry_version_id, feed_snapshot_version_id, observed_at
FROM news_feed_entry_events;

CREATE TABLE IF NOT EXISTS news_feed_entry_projection_loads (
    load_id TEXT PRIMARY KEY REFERENCES news_dlt_loads(load_id),
    projected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS news_model_calls (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    operation_key TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
    cost_usd REAL NOT NULL CHECK (cost_usd >= 0),
    latency_ms INTEGER NOT NULL CHECK (latency_ms >= 0),
    response_count INTEGER NOT NULL CHECK (response_count > 0)
);

CREATE TABLE IF NOT EXISTS news_model_attempts (
    attempt_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    operation_key TEXT NOT NULL,
    response_id TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
    cost_usd REAL NOT NULL CHECK (cost_usd >= 0),
    latency_ms INTEGER NOT NULL CHECK (latency_ms >= 0),
    status TEXT NOT NULL CHECK (status IN ('accepted', 'rejected')),
    error TEXT,
    observed_at TEXT NOT NULL,
    CHECK (
        (status = 'accepted' AND error IS NULL)
        OR (status = 'rejected' AND length(error) > 0)
    )
);

CREATE TABLE IF NOT EXISTS news_model_trace_links (
    attempt_id TEXT PRIMARY KEY REFERENCES news_model_attempts(attempt_id),
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    trace_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    project_ref TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS youtube_source_state (
    source_id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL UNIQUE,
    baseline_poll_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    last_poll_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS youtube_source_polls (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    source_id TEXT NOT NULL,
    scheduled_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('baseline', 'current', 'overlap_lost')),
    entry_count INTEGER NOT NULL CHECK (entry_count > 0),
    UNIQUE (source_id, scheduled_at)
);

CREATE TABLE IF NOT EXISTS youtube_videos (
    source_id TEXT NOT NULL,
    video_id TEXT NOT NULL CHECK (length(video_id) = 11),
    first_poll_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    state TEXT NOT NULL CHECK (
        state IN ('baseline', 'pending', 'running', 'deferred', 'quarantined', 'candidate')
    ),
    published_at TEXT NOT NULL,
    title TEXT NOT NULL CHECK (length(trim(title)) > 0),
    metadata_version_id TEXT REFERENCES artifact_versions(id),
    candidate_version_id TEXT REFERENCES artifact_versions(id),
    owner_token TEXT,
    lease_expires_at TEXT,
    retry_at TEXT,
    deterministic_failure_fingerprint TEXT,
    unchanged_deterministic_failures INTEGER NOT NULL DEFAULT 0
        CHECK (unchanged_deterministic_failures >= 0),
    last_error TEXT,
    PRIMARY KEY (source_id, video_id),
    CHECK (
        (state = 'running' AND owner_token IS NOT NULL AND lease_expires_at IS NOT NULL)
        OR (state != 'running' AND owner_token IS NULL AND lease_expires_at IS NULL)
    ),
    CHECK (
        (state = 'deferred' AND retry_at IS NOT NULL)
        OR (state != 'deferred' AND retry_at IS NULL)
    ),
    CHECK (
        (state = 'candidate' AND candidate_version_id IS NOT NULL)
        OR (state != 'candidate' AND candidate_version_id IS NULL)
    ),
    CHECK (state != 'quarantined' OR unchanged_deterministic_failures >= 3)
);

CREATE TABLE IF NOT EXISTS youtube_model_receipts (
    request_id TEXT NOT NULL,
    operation_key TEXT NOT NULL,
    attempt_index INTEGER NOT NULL CHECK (attempt_index >= 0),
    artifact_version_id TEXT NOT NULL UNIQUE REFERENCES artifact_versions(id),
    requested_model TEXT NOT NULL CHECK (length(trim(requested_model)) > 0),
    http_status INTEGER NOT NULL CHECK (http_status BETWEEN 100 AND 599),
    status TEXT NOT NULL CHECK (status IN ('unhandled', 'rejected', 'accepted')),
    handled_attempt_id TEXT REFERENCES news_model_attempts(attempt_id),
    error TEXT,
    received_at TEXT NOT NULL,
    handled_at TEXT,
    PRIMARY KEY (request_id, operation_key, attempt_index),
    CHECK (
        (status = 'unhandled' AND handled_attempt_id IS NULL AND error IS NULL AND handled_at IS NULL)
        OR (status = 'accepted' AND handled_attempt_id IS NOT NULL AND error IS NULL AND handled_at IS NOT NULL)
        OR (status = 'rejected' AND length(error) > 0 AND handled_at IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS youtube_clip_analyses (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    source_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    metadata_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    start_second INTEGER NOT NULL CHECK (start_second >= 0),
    duration_seconds INTEGER NOT NULL CHECK (duration_seconds > 0 AND duration_seconds <= 300),
    model TEXT NOT NULL CHECK (length(trim(model)) > 0),
    request_digest TEXT NOT NULL,
    model_attempt_id TEXT NOT NULL REFERENCES news_model_attempts(attempt_id),
    accepted_at TEXT NOT NULL,
    FOREIGN KEY (source_id, video_id) REFERENCES youtube_videos(source_id, video_id),
    UNIQUE (source_id, video_id, metadata_version_id, start_second, duration_seconds, model, request_digest)
);

CREATE TABLE IF NOT EXISTS youtube_merge_analyses (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    source_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    request_id TEXT NOT NULL UNIQUE,
    model_attempt_id TEXT NOT NULL REFERENCES news_model_attempts(attempt_id),
    accepted_at TEXT NOT NULL,
    FOREIGN KEY (source_id, video_id) REFERENCES youtube_videos(source_id, video_id)
);

CREATE TABLE IF NOT EXISTS youtube_candidates (
    candidate_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    source_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    policy TEXT NOT NULL CHECK (policy IN ('automatic', 'owner_review')),
    metadata_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    merge_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    owner_token TEXT NOT NULL CHECK (length(trim(owner_token)) > 0),
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id, video_id) REFERENCES youtube_videos(source_id, video_id),
    UNIQUE (source_id, video_id)
);

CREATE TABLE IF NOT EXISTS youtube_candidate_decisions (
    artifact_version_id TEXT PRIMARY KEY REFERENCES artifact_versions(id),
    candidate_version_id TEXT NOT NULL UNIQUE REFERENCES youtube_candidates(candidate_version_id),
    source_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('automatic_approved', 'owner_approved', 'owner_rejected')),
    actor TEXT NOT NULL CHECK (actor IN ('system', 'owner')),
    note TEXT CHECK (note IS NULL OR (length(trim(note)) > 0 AND length(note) <= 2000)),
    decided_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS youtube_publications (
    candidate_version_id TEXT PRIMARY KEY REFERENCES youtube_candidates(candidate_version_id),
    source_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    article_version_id TEXT NOT NULL UNIQUE REFERENCES artifact_versions(id),
    published_at TEXT NOT NULL,
    FOREIGN KEY (source_id, video_id) REFERENCES youtube_videos(source_id, video_id)
);

CREATE TABLE IF NOT EXISTS news_feedback (
    feedback_id TEXT PRIMARY KEY,
    report_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    target_kind TEXT NOT NULL CHECK (target_kind IN ('report', 'theme', 'group', 'article')),
    theme_id TEXT,
    group_id TEXT,
    article_version_id TEXT REFERENCES artifact_versions(id),
    rating TEXT CHECK (rating IN ('positive', 'negative')),
    note TEXT CHECK (note IS NULL OR length(note) <= 2000),
    actor TEXT NOT NULL CHECK (actor = 'owner'),
    created_at TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS news_evaluation_feedback_sources (
    evaluation_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    position INTEGER NOT NULL CHECK (position >= 0),
    feedback_id TEXT NOT NULL REFERENCES news_feedback(feedback_id),
    disposition TEXT NOT NULL CHECK (disposition IN ('represented', 'excluded')),
    PRIMARY KEY (evaluation_artifact_version_id, position),
    UNIQUE (evaluation_artifact_version_id, feedback_id)
);

CREATE TABLE IF NOT EXISTS news_feedback_score_curation (
    manifest_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    manifest_version TEXT NOT NULL CHECK (length(trim(manifest_version)) > 0),
    position INTEGER NOT NULL CHECK (position >= 0),
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

CREATE TABLE IF NOT EXISTS news_feedback_concern_score_sync_attempts (
    sync_attempt_id TEXT PRIMARY KEY,
    score_id TEXT NOT NULL REFERENCES news_feedback_score_curation(score_id),
    provider TEXT NOT NULL CHECK (provider = 'langfuse'),
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
    error TEXT,
    attempted_at TEXT NOT NULL,
    CHECK (
        (status = 'completed' AND error IS NULL)
        OR (status = 'failed' AND length(error) > 0)
    )
);

CREATE TABLE IF NOT EXISTS news_feedback_concern_score_dispositions (
    score_id TEXT NOT NULL REFERENCES news_feedback_score_curation(score_id),
    provider TEXT NOT NULL CHECK (provider = 'langfuse'),
    disposition TEXT NOT NULL CHECK (disposition = 'permanently_unresolved'),
    reason TEXT NOT NULL CHECK (reason = 'provider_migration'),
    source_provider TEXT NOT NULL CHECK (source_provider = 'langsmith'),
    recorded_at TEXT NOT NULL CHECK (length(trim(recorded_at)) > 0),
    PRIMARY KEY (score_id, provider)
);

CREATE TABLE IF NOT EXISTS news_evaluation_projections (
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    manifest_artifact_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    projection_kind TEXT NOT NULL CHECK (projection_kind IN ('dataset', 'experiment')),
    dataset_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    experiment_id TEXT,
    experiment_name TEXT,
    experiment_url TEXT,
    implementation_ref TEXT NOT NULL,
    completed_at TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS news_relevance_experiments (
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
    completed_at TEXT NOT NULL,
    PRIMARY KEY (provider, manifest_artifact_version_id, policy_digest, implementation_ref),
    CHECK (length(trim(policy_digest)) > 0),
    CHECK (length(trim(implementation_ref)) > 0),
    CHECK (length(trim(policy_id)) > 0)
);

CREATE TABLE IF NOT EXISTS news_theme_experiments (
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
    completed_at TEXT NOT NULL,
    PRIMARY KEY (provider, manifest_artifact_version_id, policy_digest, implementation_ref),
    CHECK (length(trim(policy_digest)) > 0),
    CHECK (length(trim(implementation_ref)) > 0),
    CHECK (length(trim(policy_id)) > 0)
);

CREATE TABLE IF NOT EXISTS news_feedback_sync_attempts (
    sync_attempt_id TEXT PRIMARY KEY,
    feedback_id TEXT NOT NULL REFERENCES news_feedback(feedback_id),

    model_attempt_id TEXT NOT NULL REFERENCES news_model_attempts(attempt_id),
    provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
    score_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
    error TEXT,
    attempted_at TEXT NOT NULL,
    CHECK (
        (status = 'completed' AND error IS NULL)
        OR (status = 'failed' AND length(error) > 0)
    )
);

CREATE TABLE IF NOT EXISTS news_feedback_sync_dispositions (
    feedback_id TEXT NOT NULL REFERENCES news_feedback(feedback_id),
    provider TEXT NOT NULL CHECK (provider = 'langfuse'),
    disposition TEXT NOT NULL CHECK (disposition = 'permanently_unresolved'),
    reason TEXT NOT NULL CHECK (reason = 'provider_migration'),
    source_provider TEXT NOT NULL CHECK (source_provider = 'langsmith'),
    recorded_at TEXT NOT NULL CHECK (length(trim(recorded_at)) > 0),
    PRIMARY KEY (feedback_id, provider)
);

CREATE TRIGGER IF NOT EXISTS artifacts_reject_conflicting_inserts
BEFORE INSERT ON artifacts
WHEN EXISTS (
    SELECT 1
    FROM artifacts AS existing
    WHERE existing.id = NEW.id
      AND (
          existing.kind IS NOT NEW.kind
          OR existing.authority_class IS NOT NEW.authority_class
      )
)
BEGIN
    SELECT RAISE(ABORT, 'artifacts identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS artifact_versions_reject_updates
BEFORE UPDATE ON artifact_versions
BEGIN
    SELECT RAISE(ABORT, 'artifact_versions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS artifact_versions_reject_deletes
BEFORE DELETE ON artifact_versions
BEGIN
    SELECT RAISE(ABORT, 'artifact_versions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS artifact_versions_reject_conflicting_inserts
BEFORE INSERT ON artifact_versions
WHEN EXISTS (
    SELECT 1
    FROM artifact_versions AS existing
    WHERE (
        existing.id = NEW.id
        OR (
            existing.artifact_id = NEW.artifact_id
            AND existing.content_digest = NEW.content_digest
        )
    )
    AND (
        existing.id IS NOT NEW.id
        OR existing.artifact_id IS NOT NEW.artifact_id
        OR existing.schema_version IS NOT NEW.schema_version
        OR existing.content_digest IS NOT NEW.content_digest
    )
)
BEGIN
    SELECT RAISE(ABORT, 'artifact_versions identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS artifact_files_reject_updates
BEFORE UPDATE ON artifact_files
BEGIN
    SELECT RAISE(ABORT, 'artifact_files are immutable');
END;

CREATE TRIGGER IF NOT EXISTS artifact_files_reject_deletes
BEFORE DELETE ON artifact_files
BEGIN
    SELECT RAISE(ABORT, 'artifact_files are immutable');
END;

CREATE TRIGGER IF NOT EXISTS artifact_files_reject_conflicting_inserts
BEFORE INSERT ON artifact_files
WHEN EXISTS (
    SELECT 1
    FROM artifact_files AS existing
    WHERE (existing.id = NEW.id OR existing.r2_key = NEW.r2_key)
    AND (
        existing.id IS NOT NEW.id
        OR existing.artifact_version_id IS NOT NEW.artifact_version_id
        OR existing.r2_key IS NOT NEW.r2_key
        OR existing.media_type IS NOT NEW.media_type
        OR existing.content_digest IS NOT NEW.content_digest
        OR existing.byte_size IS NOT NEW.byte_size
        OR existing.row_count IS NOT NEW.row_count
        OR existing.schema_fingerprint IS NOT NEW.schema_fingerprint
    )
)
BEGIN
    SELECT RAISE(ABORT, 'artifact_files identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS run_inputs_reject_updates
BEFORE UPDATE ON run_inputs
BEGIN
    SELECT RAISE(ABORT, 'run_inputs are immutable');
END;

CREATE TRIGGER IF NOT EXISTS run_inputs_reject_deletes
BEFORE DELETE ON run_inputs
BEGIN
    SELECT RAISE(ABORT, 'run_inputs are immutable');
END;

CREATE TRIGGER IF NOT EXISTS run_inputs_reject_conflicting_inserts
BEFORE INSERT ON run_inputs
WHEN EXISTS (
    SELECT 1
    FROM run_inputs AS existing
    WHERE existing.run_id = NEW.run_id
      AND existing.position = NEW.position
      AND (
          existing.artifact_version_id IS NOT NEW.artifact_version_id
          OR existing.role IS NOT NEW.role
          OR existing.locator_json IS NOT NEW.locator_json
          OR existing.selected_content_digest IS NOT NEW.selected_content_digest
          OR existing.selection_method IS NOT NEW.selection_method
          OR existing.retrieval_metadata_json IS NOT NEW.retrieval_metadata_json
      )
)
BEGIN
    SELECT RAISE(ABORT, 'run_inputs identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS run_outputs_reject_updates
BEFORE UPDATE ON run_outputs
BEGIN
    SELECT RAISE(ABORT, 'run_outputs are immutable');
END;

CREATE TRIGGER IF NOT EXISTS run_outputs_reject_deletes
BEFORE DELETE ON run_outputs
BEGIN
    SELECT RAISE(ABORT, 'run_outputs are immutable');
END;

CREATE TRIGGER IF NOT EXISTS run_outputs_reject_conflicting_inserts
BEFORE INSERT ON run_outputs
WHEN EXISTS (
    SELECT 1
    FROM run_outputs AS existing
    WHERE existing.run_id = NEW.run_id
      AND existing.position = NEW.position
      AND (
          existing.artifact_version_id IS NOT NEW.artifact_version_id
          OR existing.role IS NOT NEW.role
      )
)
BEGIN
    SELECT RAISE(ABORT, 'run_outputs identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS runs_reject_idempotency_conflicts
BEFORE INSERT ON runs
WHEN EXISTS (
    SELECT 1
    FROM runs AS existing
    WHERE (existing.id = NEW.id OR existing.idempotency_key = NEW.idempotency_key)
    AND (
        existing.id IS NOT NEW.id
        OR existing.operation_key IS NOT NEW.operation_key
        OR existing.executor_kind IS NOT NEW.executor_kind
        OR existing.implementation_ref IS NOT NEW.implementation_ref
        OR existing.parameters_json IS NOT NEW.parameters_json
        OR existing.actor IS NOT NEW.actor
        OR existing.idempotency_key IS NOT NEW.idempotency_key
    )
)
BEGIN
    SELECT RAISE(ABORT, 'runs idempotency conflict');
END;

CREATE TRIGGER IF NOT EXISTS artifacts_require_owned_current_version_on_update
BEFORE UPDATE OF current_version_id ON artifacts
WHEN NEW.current_version_id IS NOT NULL
 AND NOT EXISTS (
     SELECT 1
     FROM artifact_versions
     WHERE id = NEW.current_version_id AND artifact_id = NEW.id
 )
BEGIN
    SELECT RAISE(ABORT, 'artifact current version is not owned by artifact');
END;

CREATE TRIGGER IF NOT EXISTS artifacts_require_current_run_output_on_insert
BEFORE INSERT ON artifacts
WHEN NEW.current_run_id IS NOT NULL
 AND NOT EXISTS (
     SELECT 1
     FROM run_outputs
     WHERE run_id = NEW.current_run_id
       AND artifact_version_id = NEW.current_version_id
 )
BEGIN
    SELECT RAISE(ABORT, 'artifact current run did not output current version');
END;

CREATE TRIGGER IF NOT EXISTS artifacts_require_current_run_output_on_update
BEFORE UPDATE OF current_version_id, current_run_id ON artifacts
WHEN NEW.current_run_id IS NOT NULL
 AND NOT EXISTS (
     SELECT 1
     FROM run_outputs
     WHERE run_id = NEW.current_run_id
       AND artifact_version_id = NEW.current_version_id
 )
BEGIN
    SELECT RAISE(ABORT, 'artifact current run did not output current version');
END;

CREATE TRIGGER IF NOT EXISTS news_feed_observations_reject_updates
BEFORE UPDATE ON news_feed_observations
BEGIN
    SELECT RAISE(ABORT, 'news_feed_observations are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feed_observations_reject_deletes
BEFORE DELETE ON news_feed_observations
BEGIN
    SELECT RAISE(ABORT, 'news_feed_observations are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_article_aliases_reject_updates
BEFORE UPDATE ON news_article_aliases
BEGIN
    SELECT RAISE(ABORT, 'news_article_aliases are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_article_aliases_reject_deletes
BEFORE DELETE ON news_article_aliases
BEGIN
    SELECT RAISE(ABORT, 'news_article_aliases are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_article_aliases_reject_conflicting_inserts
BEFORE INSERT ON news_article_aliases
WHEN EXISTS (
    SELECT 1
    FROM news_article_aliases AS existing
    WHERE existing.alias_key = NEW.alias_key
      AND (
          existing.article_artifact_id IS NOT NEW.article_artifact_id
          OR existing.alias_type IS NOT NEW.alias_type
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news article alias identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_article_versions_reject_updates
BEFORE UPDATE ON news_article_versions
BEGIN
    SELECT RAISE(ABORT, 'news_article_versions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_article_versions_reject_deletes
BEFORE DELETE ON news_article_versions
BEGIN
    SELECT RAISE(ABORT, 'news_article_versions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_article_failure_attempts_reject_updates
BEFORE UPDATE ON news_article_failure_attempts
BEGIN
    SELECT RAISE(ABORT, 'news_article_failure_attempts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_article_failure_attempts_reject_deletes
BEFORE DELETE ON news_article_failure_attempts
BEGIN
    SELECT RAISE(ABORT, 'news_article_failure_attempts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_article_recovery_overrides_reject_updates
BEFORE UPDATE ON news_article_recovery_overrides
BEGIN
    SELECT RAISE(ABORT, 'news_article_recovery_overrides are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_article_recovery_overrides_reject_deletes
BEFORE DELETE ON news_article_recovery_overrides
BEGIN
    SELECT RAISE(ABORT, 'news_article_recovery_overrides are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_article_recovery_overrides_reject_conflicting_inserts
BEFORE INSERT ON news_article_recovery_overrides
WHEN EXISTS (
    SELECT 1 FROM news_article_recovery_overrides AS existing
    WHERE (
        existing.recovery_id = NEW.recovery_id
        OR (
            existing.event_id = NEW.event_id
            AND existing.base_work_generation = NEW.base_work_generation
            AND existing.expected_work_generation = NEW.expected_work_generation
        )
    )
      AND (
          existing.recovery_id IS NOT NEW.recovery_id
          OR existing.event_id IS NOT NEW.event_id
          OR existing.base_work_generation IS NOT NEW.base_work_generation
          OR existing.expected_work_generation IS NOT NEW.expected_work_generation
          OR existing.requested_by IS NOT NEW.requested_by
          OR existing.reason IS NOT NEW.reason
          OR existing.requested_at IS NOT NEW.requested_at
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news_article_recovery_overrides identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_article_failure_attempts_reject_conflicting_inserts
BEFORE INSERT ON news_article_failure_attempts
WHEN EXISTS (
    SELECT 1
    FROM news_article_failure_attempts AS existing
    WHERE (
          existing.attempt_id = NEW.attempt_id
          OR (
              existing.dagster_run_id = NEW.dagster_run_id
              AND existing.retry_number = NEW.retry_number
              AND existing.event_id = NEW.event_id
          )
      )
      AND (
          existing.attempt_id IS NOT NEW.attempt_id
          OR existing.implementation_ref IS NOT NEW.implementation_ref
          OR existing.work_generation IS NOT NEW.work_generation
          OR existing.dagster_run_id IS NOT NEW.dagster_run_id
          OR existing.retry_number IS NOT NEW.retry_number
          OR existing.failure_kind IS NOT NEW.failure_kind
          OR existing.failure_fingerprint IS NOT NEW.failure_fingerprint
          OR existing.error IS NOT NEW.error
          OR existing.attempted_at IS NOT NEW.attempted_at
          OR existing.retry_at IS NOT NEW.retry_at
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news article failure attempt identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_dlt_loads_reject_updates
BEFORE UPDATE ON news_dlt_loads
BEGIN
    SELECT RAISE(ABORT, 'news_dlt_loads are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_dlt_loads_reject_deletes
BEFORE DELETE ON news_dlt_loads
BEGIN
    SELECT RAISE(ABORT, 'news_dlt_loads are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feed_entry_events_reject_updates
BEFORE UPDATE ON news_feed_entry_events
BEGIN
    SELECT RAISE(ABORT, 'news_feed_entry_events are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feed_entry_events_reject_deletes
BEFORE DELETE ON news_feed_entry_events
BEGIN
    SELECT RAISE(ABORT, 'news_feed_entry_events are immutable');
END;

DROP TRIGGER IF EXISTS news_feed_entry_events_reject_conflicting_inserts;

CREATE TRIGGER news_feed_entry_events_reject_conflicting_inserts
BEFORE INSERT ON news_feed_entry_events
WHEN EXISTS (
    SELECT 1
    FROM news_feed_entry_events AS existing
    WHERE existing.event_id = NEW.event_id
      AND (
          existing.feed_id IS NOT NEW.feed_id
          OR existing.source_id IS NOT NEW.source_id
          OR existing.original_url IS NOT NEW.original_url
          OR existing.published_at IS NOT NEW.published_at
          OR existing.source_updated_at IS NOT NEW.source_updated_at
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news_feed_entry_events identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_feed_entry_event_versions_reject_conflicting_updates
BEFORE UPDATE ON news_feed_entry_event_versions
WHEN OLD.version_id IS NOT OLD.event_id
 AND OLD.version_id IS NOT NEW.version_id
BEGIN
    SELECT RAISE(ABORT, 'news_feed_entry_event_versions identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_feed_entry_event_versions_reject_deletes
BEFORE DELETE ON news_feed_entry_event_versions
BEGIN
    SELECT RAISE(ABORT, 'news_feed_entry_event_versions are immutable after resolution');
END;

CREATE TRIGGER IF NOT EXISTS news_feed_entry_occurrences_reject_updates
BEFORE UPDATE ON news_feed_entry_occurrences
BEGIN
    SELECT RAISE(ABORT, 'news_feed_entry_occurrences are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feed_entry_occurrences_reject_deletes
BEFORE DELETE ON news_feed_entry_occurrences
BEGIN
    SELECT RAISE(ABORT, 'news_feed_entry_occurrences are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feed_entry_projection_loads_reject_updates
BEFORE UPDATE ON news_feed_entry_projection_loads
BEGIN
    SELECT RAISE(ABORT, 'news_feed_entry_projection_loads are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feed_entry_projection_loads_reject_deletes
BEFORE DELETE ON news_feed_entry_projection_loads
BEGIN
    SELECT RAISE(ABORT, 'news_feed_entry_projection_loads are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feed_entry_projection_loads_reject_conflicting_inserts
BEFORE INSERT ON news_feed_entry_projection_loads
WHEN EXISTS (
    SELECT 1
    FROM news_feed_entry_projection_loads AS existing
    WHERE existing.load_id = NEW.load_id
      AND existing.projected_at IS NOT NEW.projected_at
)
BEGIN
    SELECT RAISE(ABORT, 'news_feed_entry_projection_loads identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_model_calls_reject_updates
BEFORE UPDATE ON news_model_calls
BEGIN
    SELECT RAISE(ABORT, 'news_model_calls are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_model_calls_reject_deletes
BEFORE DELETE ON news_model_calls
BEGIN
    SELECT RAISE(ABORT, 'news_model_calls are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_model_attempts_reject_updates
BEFORE UPDATE ON news_model_attempts
BEGIN
    SELECT RAISE(ABORT, 'news_model_attempts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_model_attempts_reject_deletes
BEFORE DELETE ON news_model_attempts
BEGIN
    SELECT RAISE(ABORT, 'news_model_attempts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_model_attempts_reject_conflicting_inserts
BEFORE INSERT ON news_model_attempts
WHEN EXISTS (
    SELECT 1
    FROM news_model_attempts AS existing
    WHERE existing.attempt_id = NEW.attempt_id
      AND (
          existing.request_id IS NOT NEW.request_id
          OR existing.operation_key IS NOT NEW.operation_key
          OR existing.response_id IS NOT NEW.response_id
          OR existing.model IS NOT NEW.model
          OR existing.input_tokens IS NOT NEW.input_tokens
          OR existing.output_tokens IS NOT NEW.output_tokens
          OR existing.cost_usd IS NOT NEW.cost_usd
          OR existing.latency_ms IS NOT NEW.latency_ms
          OR existing.status IS NOT NEW.status
          OR existing.error IS NOT NEW.error
          OR existing.observed_at IS NOT NEW.observed_at
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news_model_attempts identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_model_trace_links_reject_updates
BEFORE UPDATE ON news_model_trace_links
BEGIN
    SELECT RAISE(ABORT, 'news_model_trace_links are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_model_trace_links_reject_deletes
BEFORE DELETE ON news_model_trace_links
BEGIN
    SELECT RAISE(ABORT, 'news_model_trace_links are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_model_trace_links_reject_conflicting_inserts
BEFORE INSERT ON news_model_trace_links
WHEN EXISTS (
    SELECT 1
    FROM news_model_trace_links AS existing
    WHERE existing.attempt_id = NEW.attempt_id
      AND (
          existing.provider IS NOT NEW.provider
          OR existing.trace_id IS NOT NEW.trace_id
          OR existing.observation_id IS NOT NEW.observation_id
          OR existing.project_ref IS NOT NEW.project_ref
          OR existing.recorded_at IS NOT NEW.recorded_at
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news_model_trace_links identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS youtube_model_receipts_reject_conflicting_inserts
BEFORE INSERT ON youtube_model_receipts
WHEN EXISTS (
    SELECT 1 FROM youtube_model_receipts AS existing
    WHERE existing.request_id = NEW.request_id
      AND existing.operation_key = NEW.operation_key
      AND existing.attempt_index = NEW.attempt_index
      AND (
          existing.artifact_version_id IS NOT NEW.artifact_version_id
          OR existing.requested_model IS NOT NEW.requested_model
          OR existing.http_status IS NOT NEW.http_status
          OR existing.received_at IS NOT NEW.received_at
      )
)
BEGIN
    SELECT RAISE(ABORT, 'youtube model receipt identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS youtube_source_polls_reject_updates
BEFORE UPDATE ON youtube_source_polls BEGIN
    SELECT RAISE(ABORT, 'youtube_source_polls are immutable');
END;

CREATE TRIGGER IF NOT EXISTS youtube_source_polls_reject_deletes
BEFORE DELETE ON youtube_source_polls BEGIN
    SELECT RAISE(ABORT, 'youtube_source_polls are immutable');
END;

CREATE TRIGGER IF NOT EXISTS youtube_clip_analyses_reject_updates
BEFORE UPDATE ON youtube_clip_analyses BEGIN
    SELECT RAISE(ABORT, 'youtube_clip_analyses are immutable');
END;

CREATE TRIGGER IF NOT EXISTS youtube_clip_analyses_reject_deletes
BEFORE DELETE ON youtube_clip_analyses BEGIN
    SELECT RAISE(ABORT, 'youtube_clip_analyses are immutable');
END;

CREATE TRIGGER IF NOT EXISTS youtube_merge_analyses_reject_updates
BEFORE UPDATE ON youtube_merge_analyses BEGIN
    SELECT RAISE(ABORT, 'youtube_merge_analyses are immutable');
END;

CREATE TRIGGER IF NOT EXISTS youtube_merge_analyses_reject_deletes
BEFORE DELETE ON youtube_merge_analyses BEGIN
    SELECT RAISE(ABORT, 'youtube_merge_analyses are immutable');
END;

CREATE TRIGGER IF NOT EXISTS youtube_candidates_require_lease
BEFORE INSERT ON youtube_candidates
WHEN NOT EXISTS (
    SELECT 1 FROM youtube_videos AS video
    WHERE video.source_id = NEW.source_id
      AND video.video_id = NEW.video_id
      AND video.state = 'running'
      AND video.owner_token = NEW.owner_token
      AND video.lease_expires_at > NEW.created_at
)
BEGIN
    SELECT RAISE(ABORT, 'YouTube candidate lease was lost');
END;

CREATE TRIGGER IF NOT EXISTS youtube_candidates_reject_updates
BEFORE UPDATE ON youtube_candidates BEGIN
    SELECT RAISE(ABORT, 'youtube_candidates are immutable');
END;

CREATE TRIGGER IF NOT EXISTS youtube_candidates_reject_deletes
BEFORE DELETE ON youtube_candidates BEGIN
    SELECT RAISE(ABORT, 'youtube_candidates are immutable');
END;

CREATE TRIGGER IF NOT EXISTS youtube_candidate_decisions_require_candidate_policy
BEFORE INSERT ON youtube_candidate_decisions
WHEN NOT EXISTS (
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
)
BEGIN
    SELECT RAISE(ABORT, 'YouTube candidate decision violates source policy');
END;

CREATE TRIGGER IF NOT EXISTS youtube_candidate_decisions_reject_updates
BEFORE UPDATE ON youtube_candidate_decisions BEGIN
    SELECT RAISE(ABORT, 'youtube_candidate_decisions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS youtube_candidate_decisions_reject_deletes
BEFORE DELETE ON youtube_candidate_decisions BEGIN
    SELECT RAISE(ABORT, 'youtube_candidate_decisions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS youtube_candidate_decisions_reject_conflicts
BEFORE INSERT ON youtube_candidate_decisions
WHEN EXISTS (
    SELECT 1 FROM youtube_candidate_decisions AS existing
    WHERE existing.candidate_version_id = NEW.candidate_version_id
      AND (existing.source_id IS NOT NEW.source_id
          OR existing.kind IS NOT NEW.kind
          OR existing.actor IS NOT NEW.actor
          OR existing.note IS NOT NEW.note)
)
BEGIN
    SELECT RAISE(ABORT, 'youtube candidate decision conflict');
END;

CREATE TRIGGER IF NOT EXISTS youtube_publications_require_approval
BEFORE INSERT ON youtube_publications
WHEN NOT EXISTS (
    SELECT 1
    FROM youtube_candidates AS candidate
    JOIN youtube_candidate_decisions AS decision
      ON decision.candidate_version_id = candidate.candidate_version_id
    WHERE candidate.candidate_version_id = NEW.candidate_version_id
      AND candidate.source_id = NEW.source_id
      AND candidate.video_id = NEW.video_id
      AND decision.kind IN ('automatic_approved', 'owner_approved')
)
BEGIN
    SELECT RAISE(ABORT, 'YouTube publication requires exact approval');
END;

CREATE TRIGGER IF NOT EXISTS youtube_publications_reject_updates
BEFORE UPDATE ON youtube_publications BEGIN
    SELECT RAISE(ABORT, 'youtube_publications are immutable');
END;

CREATE TRIGGER IF NOT EXISTS youtube_publications_reject_deletes
BEFORE DELETE ON youtube_publications BEGIN
    SELECT RAISE(ABORT, 'youtube_publications are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_reject_updates
BEFORE UPDATE ON news_feedback
BEGIN
    SELECT RAISE(ABORT, 'news_feedback is immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_reject_deletes
BEFORE DELETE ON news_feedback
BEGIN
    SELECT RAISE(ABORT, 'news_feedback is immutable');
END;

DROP TRIGGER IF EXISTS news_feedback_reject_conflicting_inserts;

CREATE TRIGGER news_feedback_reject_conflicting_inserts
BEFORE INSERT ON news_feedback
WHEN EXISTS (
    SELECT 1
    FROM news_feedback AS existing
    WHERE existing.feedback_id = NEW.feedback_id
      AND (
          existing.report_version_id IS NOT NEW.report_version_id
          OR existing.target_kind IS NOT NEW.target_kind
          OR existing.theme_id IS NOT NEW.theme_id
          OR existing.group_id IS NOT NEW.group_id
          OR existing.article_version_id IS NOT NEW.article_version_id
          OR existing.rating IS NOT NEW.rating
          OR existing.note IS NOT NEW.note
          OR existing.actor IS NOT NEW.actor
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news_feedback identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_evaluation_feedback_sources_reject_updates
BEFORE UPDATE ON news_evaluation_feedback_sources
BEGIN
    SELECT RAISE(ABORT, 'news_evaluation_feedback_sources are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_evaluation_feedback_sources_reject_deletes
BEFORE DELETE ON news_evaluation_feedback_sources
BEGIN
    SELECT RAISE(ABORT, 'news_evaluation_feedback_sources are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_evaluation_feedback_sources_reject_conflicting_inserts
BEFORE INSERT ON news_evaluation_feedback_sources
WHEN EXISTS (
    SELECT 1
    FROM news_evaluation_feedback_sources AS existing
    WHERE existing.evaluation_artifact_version_id = NEW.evaluation_artifact_version_id
      AND (existing.position = NEW.position OR existing.feedback_id = NEW.feedback_id)
      AND (
          existing.position IS NOT NEW.position
          OR existing.feedback_id IS NOT NEW.feedback_id
          OR existing.disposition IS NOT NEW.disposition
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news_evaluation_feedback_sources identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_score_curation_reject_updates
BEFORE UPDATE ON news_feedback_score_curation
BEGIN
    SELECT RAISE(ABORT, 'news_feedback_score_curation is immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_score_curation_reject_deletes
BEFORE DELETE ON news_feedback_score_curation
BEGIN
    SELECT RAISE(ABORT, 'news_feedback_score_curation is immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_score_curation_reject_feedback_mismatch
BEFORE INSERT ON news_feedback_score_curation
WHEN NOT EXISTS (
    SELECT 1 FROM news_feedback AS feedback
    WHERE feedback.feedback_id = NEW.feedback_id
      AND feedback.report_version_id = NEW.report_version_id
)
BEGIN
    SELECT RAISE(ABORT, 'news_feedback_score_curation feedback identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_score_curation_reject_conflicting_inserts
BEFORE INSERT ON news_feedback_score_curation
WHEN EXISTS (
    SELECT 1 FROM news_feedback_score_curation AS existing
    WHERE (
        (existing.manifest_artifact_version_id = NEW.manifest_artifact_version_id
         AND existing.position = NEW.position)
        OR (NEW.score_id IS NOT NULL AND existing.score_id = NEW.score_id)
      )
      AND (
          existing.manifest_version IS NOT NEW.manifest_version
          OR existing.feedback_id IS NOT NEW.feedback_id
          OR existing.concern IS NOT NEW.concern
          OR existing.exclusion_reason IS NOT NEW.exclusion_reason
          OR existing.decision IS NOT NEW.decision
          OR existing.report_version_id IS NOT NEW.report_version_id
          OR existing.model_output_version_id IS NOT NEW.model_output_version_id
          OR existing.model_attempt_id IS NOT NEW.model_attempt_id
          OR existing.polarity IS NOT NEW.polarity
          OR existing.rationale IS NOT NEW.rationale
          OR existing.score_id IS NOT NEW.score_id
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news_feedback_score_curation identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_concern_score_sync_attempts_reject_updates
BEFORE UPDATE ON news_feedback_concern_score_sync_attempts
BEGIN
    SELECT RAISE(ABORT, 'news_feedback_concern_score_sync_attempts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_concern_score_sync_attempts_reject_deletes
BEFORE DELETE ON news_feedback_concern_score_sync_attempts
BEGIN
    SELECT RAISE(ABORT, 'news_feedback_concern_score_sync_attempts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_concern_score_dispositions_reject_updates
BEFORE UPDATE ON news_feedback_concern_score_dispositions
BEGIN
    SELECT RAISE(ABORT, 'news_feedback_concern_score_dispositions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_concern_score_dispositions_reject_deletes
BEFORE DELETE ON news_feedback_concern_score_dispositions
BEGIN
    SELECT RAISE(ABORT, 'news_feedback_concern_score_dispositions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_evaluation_projections_reject_updates
BEFORE UPDATE ON news_evaluation_projections
BEGIN
    SELECT RAISE(ABORT, 'news_evaluation_projections are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_evaluation_projections_reject_deletes
BEFORE DELETE ON news_evaluation_projections
BEGIN
    SELECT RAISE(ABORT, 'news_evaluation_projections are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_evaluation_projections_reject_conflicting_inserts
BEFORE INSERT ON news_evaluation_projections
WHEN EXISTS (
    SELECT 1
    FROM news_evaluation_projections AS existing
    WHERE existing.provider = NEW.provider
      AND existing.manifest_artifact_version_id = NEW.manifest_artifact_version_id
      AND existing.projection_kind = NEW.projection_kind
      AND existing.implementation_ref = NEW.implementation_ref
      AND (
          existing.dataset_id IS NOT NEW.dataset_id
          OR existing.dataset_name IS NOT NEW.dataset_name
          OR existing.experiment_id IS NOT NEW.experiment_id
          OR existing.experiment_name IS NOT NEW.experiment_name
          OR existing.experiment_url IS NOT NEW.experiment_url
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news_evaluation_projections identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_relevance_experiments_reject_updates
BEFORE UPDATE ON news_relevance_experiments
BEGIN
    SELECT RAISE(ABORT, 'news_relevance_experiments are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_relevance_experiments_reject_deletes
BEFORE DELETE ON news_relevance_experiments
BEGIN
    SELECT RAISE(ABORT, 'news_relevance_experiments are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_relevance_experiments_reject_conflicting_inserts
BEFORE INSERT ON news_relevance_experiments
WHEN EXISTS (
    SELECT 1
    FROM news_relevance_experiments AS existing
    WHERE existing.provider = NEW.provider
      AND existing.manifest_artifact_version_id = NEW.manifest_artifact_version_id
      AND existing.policy_digest = NEW.policy_digest
      AND existing.implementation_ref = NEW.implementation_ref
      AND (
          existing.policy_id IS NOT NEW.policy_id
          OR existing.dataset_id IS NOT NEW.dataset_id
          OR existing.dataset_name IS NOT NEW.dataset_name
          OR existing.experiment_id IS NOT NEW.experiment_id
          OR existing.experiment_name IS NOT NEW.experiment_name
          OR existing.experiment_url IS NOT NEW.experiment_url
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news_relevance_experiments identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_theme_experiments_reject_updates
BEFORE UPDATE ON news_theme_experiments
BEGIN
    SELECT RAISE(ABORT, 'news_theme_experiments are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_theme_experiments_reject_deletes
BEFORE DELETE ON news_theme_experiments
BEGIN
    SELECT RAISE(ABORT, 'news_theme_experiments are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_theme_experiments_reject_conflicting_inserts
BEFORE INSERT ON news_theme_experiments
WHEN EXISTS (
    SELECT 1
    FROM news_theme_experiments AS existing
    WHERE existing.provider = NEW.provider
      AND existing.manifest_artifact_version_id = NEW.manifest_artifact_version_id
      AND existing.policy_digest = NEW.policy_digest
      AND existing.implementation_ref = NEW.implementation_ref
      AND (
          existing.policy_id IS NOT NEW.policy_id
          OR existing.dataset_id IS NOT NEW.dataset_id
          OR existing.dataset_name IS NOT NEW.dataset_name
          OR existing.experiment_id IS NOT NEW.experiment_id
          OR existing.experiment_name IS NOT NEW.experiment_name
          OR existing.experiment_url IS NOT NEW.experiment_url
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news_theme_experiments identity conflict');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_sync_attempts_reject_updates
BEFORE UPDATE ON news_feedback_sync_attempts
BEGIN

    SELECT RAISE(ABORT, 'news_feedback_sync_attempts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_sync_attempts_reject_deletes
BEFORE DELETE ON news_feedback_sync_attempts
BEGIN
    SELECT RAISE(ABORT, 'news_feedback_sync_attempts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_sync_attempts_reject_conflicting_inserts
BEFORE INSERT ON news_feedback_sync_attempts
WHEN EXISTS (
    SELECT 1
    FROM news_feedback_sync_attempts AS existing
    WHERE existing.sync_attempt_id = NEW.sync_attempt_id
      AND (
          existing.feedback_id IS NOT NEW.feedback_id
          OR existing.model_attempt_id IS NOT NEW.model_attempt_id
          OR existing.provider IS NOT NEW.provider
          OR existing.score_id IS NOT NEW.score_id
          OR existing.status IS NOT NEW.status
          OR existing.error IS NOT NEW.error
          OR existing.attempted_at IS NOT NEW.attempted_at
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news_feedback_sync_attempts identity conflict');
END;


CREATE TRIGGER IF NOT EXISTS news_feedback_sync_dispositions_reject_updates
BEFORE UPDATE ON news_feedback_sync_dispositions
BEGIN
    SELECT RAISE(ABORT, 'news_feedback_sync_dispositions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_sync_dispositions_reject_deletes
BEFORE DELETE ON news_feedback_sync_dispositions
BEGIN
    SELECT RAISE(ABORT, 'news_feedback_sync_dispositions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS news_feedback_sync_dispositions_reject_conflicting_inserts
BEFORE INSERT ON news_feedback_sync_dispositions
WHEN EXISTS (
    SELECT 1
    FROM news_feedback_sync_dispositions AS existing
    WHERE existing.feedback_id = NEW.feedback_id
      AND existing.provider = NEW.provider
      AND (
          existing.disposition IS NOT NEW.disposition
          OR existing.reason IS NOT NEW.reason
          OR existing.source_provider IS NOT NEW.source_provider
      )
)
BEGIN
    SELECT RAISE(ABORT, 'news_feedback_sync_dispositions identity conflict');
END;


CREATE INDEX IF NOT EXISTS artifact_versions_by_run
    ON artifact_versions(produced_by_run_id);

CREATE INDEX IF NOT EXISTS artifacts_by_kind_current_version
    ON artifacts(kind, current_version_id);

CREATE INDEX IF NOT EXISTS artifacts_by_current_version
    ON artifacts(current_version_id);

CREATE INDEX IF NOT EXISTS artifact_files_by_version
    ON artifact_files(artifact_version_id);

CREATE INDEX IF NOT EXISTS run_inputs_by_version
    ON run_inputs(artifact_version_id);

CREATE INDEX IF NOT EXISTS run_outputs_by_version
    ON run_outputs(artifact_version_id);

CREATE INDEX IF NOT EXISTS news_feed_entry_events_by_publication
    ON news_feed_entry_events(published_at, feed_id, event_id);

CREATE INDEX IF NOT EXISTS news_article_versions_by_day
    ON news_article_versions(bucharest_day, artifact_version_id);

CREATE INDEX IF NOT EXISTS news_article_failure_attempts_by_ref_event_generation_time
    ON news_article_failure_attempts(
        implementation_ref,
        event_id,
        work_generation,
        attempted_at,
        attempt_id
    );

CREATE INDEX IF NOT EXISTS news_article_recovery_overrides_by_event_generation_sequence
    ON news_article_recovery_overrides(
        event_id, base_work_generation, recovery_sequence DESC
    );

CREATE INDEX IF NOT EXISTS news_feedback_by_report_target_time
    ON news_feedback(
        report_version_id,
        target_kind,
        theme_id,
        group_id,
        article_version_id,
        created_at DESC
    );

CREATE INDEX IF NOT EXISTS news_feedback_score_curation_by_manifest_decision
    ON news_feedback_score_curation(manifest_artifact_version_id, decision, position);

CREATE INDEX IF NOT EXISTS news_feedback_concern_score_sync_attempts_by_score_status
    ON news_feedback_concern_score_sync_attempts(provider, score_id, status);

CREATE INDEX IF NOT EXISTS news_feedback_sync_attempts_by_pair_status
    ON news_feedback_sync_attempts(provider, feedback_id, model_attempt_id, status);

CREATE INDEX IF NOT EXISTS news_feedback_sync_attempts_by_time
    ON news_feedback_sync_attempts(attempted_at);

CREATE INDEX IF NOT EXISTS news_feedback_sync_dispositions_by_provider
    ON news_feedback_sync_dispositions(provider, disposition, feedback_id);

CREATE INDEX IF NOT EXISTS youtube_videos_by_state_time
    ON youtube_videos(source_id, state, retry_at, lease_expires_at, published_at);

CREATE INDEX IF NOT EXISTS youtube_model_receipts_by_request_status
    ON youtube_model_receipts(request_id, operation_key, status, attempt_index);
