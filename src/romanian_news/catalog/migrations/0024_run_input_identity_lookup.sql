CREATE FUNCTION reject_run_input_identity_conflict() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM run_inputs AS existing
        WHERE existing.run_id = NEW.run_id
          AND existing.position = NEW.position
          AND ROW(
              existing.artifact_version_id,
              existing.role,
              existing.locator_json,
              existing.selected_content_digest,
              existing.selection_method,
              existing.retrieval_metadata_json
          ) IS DISTINCT FROM ROW(
              NEW.artifact_version_id,
              NEW.role,
              NEW.locator_json,
              NEW.selected_content_digest,
              NEW.selection_method,
              NEW.retrieval_metadata_json
          )
    ) THEN
        RAISE EXCEPTION 'run_inputs identity conflict' USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER run_inputs_reject_conflicting_inserts ON run_inputs;

CREATE TRIGGER run_inputs_reject_conflicting_inserts
BEFORE INSERT ON run_inputs
FOR EACH ROW EXECUTE FUNCTION reject_run_input_identity_conflict();
