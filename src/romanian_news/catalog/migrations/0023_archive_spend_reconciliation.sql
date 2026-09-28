ALTER TABLE news_archive_model_reservations
    ADD COLUMN reconciled_at TIMESTAMPTZ,
    ADD COLUMN reconciled_by TEXT,
    ADD COLUMN reconciliation_reason TEXT,
    ADD COLUMN billing_reference TEXT,
    ADD CONSTRAINT news_archive_spend_reconciliation_complete CHECK (
        (reconciled_at IS NULL AND reconciled_by IS NULL
            AND reconciliation_reason IS NULL AND billing_reference IS NULL)
        OR
        (reconciled_at IS NOT NULL AND reconciled_by IS NOT NULL
            AND length(btrim(reconciled_by)) > 0
            AND reconciliation_reason IS NOT NULL
            AND length(btrim(reconciliation_reason)) > 0
            AND billing_reference IS NOT NULL
            AND length(btrim(billing_reference)) > 0
            AND actual_usd IS NOT NULL AND settled_at IS NOT NULL)
    );

CREATE FUNCTION protect_archive_spend_reservation_update() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.actual_usd IS NOT NULL THEN
        RAISE EXCEPTION 'settled archive spend reservations are immutable';
    END IF;
    IF (NEW.reservation_id, NEW.day, NEW.operation_key, NEW.request_id,
        NEW.reserved_usd, NEW.created_at) IS DISTINCT FROM
       (OLD.reservation_id, OLD.day, OLD.operation_key, OLD.request_id,
        OLD.reserved_usd, OLD.created_at) THEN
        RAISE EXCEPTION 'archive spend reservation identity is immutable';
    END IF;
    IF NEW.actual_usd IS NULL OR NEW.settled_at IS NULL THEN
        RAISE EXCEPTION 'archive spend reservations can only be settled';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER news_archive_model_reservations_protect_updates
BEFORE UPDATE ON news_archive_model_reservations
FOR EACH ROW EXECUTE FUNCTION protect_archive_spend_reservation_update();
