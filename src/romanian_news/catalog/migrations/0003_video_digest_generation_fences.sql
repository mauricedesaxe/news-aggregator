ALTER TABLE video_digest_generation_requests
    ADD CONSTRAINT video_digest_generation_requests_provider_receipt_unique
    UNIQUE (provider_receipt_id);

ALTER TABLE video_digest_slots
    ADD COLUMN terminal_lease_owner_token TEXT,
    ADD COLUMN terminal_lease_expires_at TIMESTAMPTZ,
    ADD COLUMN terminal_claim_count BIGINT CHECK (terminal_claim_count > 0),
    ADD CONSTRAINT video_digest_slots_terminal_fence_complete CHECK (
        (terminal_lease_owner_token IS NULL
         AND terminal_lease_expires_at IS NULL
         AND terminal_claim_count IS NULL)
        OR (terminal_lease_owner_token IS NOT NULL
            AND length(trim(terminal_lease_owner_token)) > 0
            AND terminal_lease_expires_at IS NOT NULL
            AND terminal_claim_count IS NOT NULL
            AND stage = 'failed')
    );
