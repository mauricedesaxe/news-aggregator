ALTER TABLE video_digest_editions
    ADD COLUMN selection_digest TEXT
    CHECK (selection_digest IS NULL OR selection_digest ~ '^[0-9a-f]{64}$');

DO $$
DECLARE identity_constraint TEXT;
BEGIN
    SELECT conname INTO identity_constraint
    FROM pg_constraint
    WHERE conrelid = 'video_digest_editions'::regclass
      AND contype = 'u'
      AND pg_get_constraintdef(oid) =
          'UNIQUE (daily_report_version_id, policy_bundle_version_id)';
    IF identity_constraint IS NULL THEN
        RAISE EXCEPTION 'Existing video digest edition identity constraint is missing';
    END IF;
    EXECUTE format(
        'ALTER TABLE video_digest_editions DROP CONSTRAINT %I', identity_constraint
    );
END $$;

CREATE UNIQUE INDEX video_digest_legacy_edition_identity
    ON video_digest_editions (daily_report_version_id, policy_bundle_version_id)
    WHERE selection_digest IS NULL;

CREATE UNIQUE INDEX video_digest_selected_edition_identity
    ON video_digest_editions (
        daily_report_version_id, policy_bundle_version_id, selection_digest
    ) WHERE selection_digest IS NOT NULL;

CREATE TABLE video_digest_slot_sources (
    slot_id TEXT PRIMARY KEY REFERENCES video_digest_slots(slot_id),
    source_record JSONB,
    selection JSONB,
    selection_digest TEXT CHECK (
        selection_digest IS NULL OR selection_digest ~ '^[0-9a-f]{64}$'
    ),
    observed_at TIMESTAMPTZ NOT NULL,
    selected_at TIMESTAMPTZ,
    CHECK ((selection IS NULL AND selection_digest IS NULL AND selected_at IS NULL)
        OR (selection IS NOT NULL AND selection_digest IS NOT NULL
            AND selected_at IS NOT NULL))
);

CREATE INDEX video_digest_slot_sources_selection_lookup
    ON video_digest_slots (bucharest_day, scheduled_at)
    WHERE stage IN (
        'claimed', 'planning', 'generating', 'assembling',
        'subtitling', 'publishing', 'published'
    );

CREATE FUNCTION protect_video_digest_slot_selection() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF (NEW.slot_id, NEW.source_record, NEW.observed_at)
       IS DISTINCT FROM (OLD.slot_id, OLD.source_record, OLD.observed_at) THEN
        RAISE EXCEPTION 'video digest slot source is immutable' USING ERRCODE = '23000';
    END IF;
    IF OLD.selection IS NOT NULL AND
       (NEW.selection, NEW.selection_digest, NEW.selected_at)
       IS DISTINCT FROM (OLD.selection, OLD.selection_digest, OLD.selected_at) THEN
        RAISE EXCEPTION 'video digest slot selection is immutable' USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER video_digest_slot_sources_protect_selection
BEFORE UPDATE ON video_digest_slot_sources
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_slot_selection();

CREATE TRIGGER video_digest_slot_sources_reject_deletes
BEFORE DELETE ON video_digest_slot_sources
FOR EACH ROW EXECUTE FUNCTION reject_immutable_catalog_change();

CREATE FUNCTION protect_video_digest_edition_selection() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.selection_digest IS DISTINCT FROM OLD.selection_digest THEN
        RAISE EXCEPTION 'video digest edition selection is immutable' USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER video_digest_editions_protect_selection
BEFORE UPDATE ON video_digest_editions
FOR EACH ROW EXECUTE FUNCTION protect_video_digest_edition_selection();
