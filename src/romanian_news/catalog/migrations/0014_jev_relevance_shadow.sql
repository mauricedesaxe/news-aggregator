CREATE TABLE news_jev_shadow_claims (
    shadow_id TEXT PRIMARY KEY CHECK (shadow_id ~ '^[0-9a-f]{64}$'),
    article_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    incumbent_version_id TEXT NOT NULL REFERENCES artifact_versions(id),
    article_content_digest TEXT NOT NULL,
    incumbent_content_digest TEXT NOT NULL,
    state_digest TEXT NOT NULL,
    question_digest TEXT NOT NULL,
    execution_policy_digest TEXT NOT NULL,
    incumbent_request_id TEXT NOT NULL,
    incumbent_policy_digest TEXT NOT NULL,
    incumbent_accepted BOOLEAN NOT NULL,
    state_length BIGINT NOT NULL CHECK (state_length > 0),
    execution_ref TEXT NOT NULL CHECK (length(trim(execution_ref)) > 0),
    claimed_at TIMESTAMPTZ NOT NULL,
    UNIQUE (incumbent_version_id, state_digest, question_digest, execution_policy_digest, execution_ref)
);

CREATE TABLE news_jev_shadow_receipts (
    shadow_id TEXT PRIMARY KEY REFERENCES news_jev_shadow_claims(shadow_id),
    fallback_reason TEXT NOT NULL CHECK (
        fallback_reason IN ('none', 'over_guard', 'provider_rejected', 'provider_failed')
    ),
    jev_request_id TEXT,
    provider_request_id TEXT,
    model TEXT,
    probability NUMERIC CHECK (probability BETWEEN 0 AND 1),
    jev_accepted BOOLEAN,
    attempts JSONB NOT NULL,
    input_tokens BIGINT CHECK (input_tokens >= 0),
    output_tokens BIGINT CHECK (output_tokens >= 0),
    estimated_cost_usd NUMERIC CHECK (estimated_cost_usd >= 0),
    latency_ms BIGINT NOT NULL CHECK (latency_ms >= 0),
    accounting_complete BOOLEAN NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL,
    CHECK (
        (fallback_reason = 'none' AND jev_request_id IS NOT NULL AND jev_accepted IS NOT NULL
            AND probability IS NOT NULL)
        OR (fallback_reason <> 'none' AND jev_accepted IS NULL AND probability IS NULL)
    )
);

CREATE TRIGGER news_jev_shadow_claims_reject_updates
BEFORE UPDATE ON news_jev_shadow_claims
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_jev_shadow_claims_reject_deletes
BEFORE DELETE ON news_jev_shadow_claims
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_jev_shadow_receipts_reject_updates
BEFORE UPDATE ON news_jev_shadow_receipts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE TRIGGER news_jev_shadow_receipts_reject_deletes
BEFORE DELETE ON news_jev_shadow_receipts
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();
