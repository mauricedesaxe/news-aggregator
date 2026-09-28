CREATE TABLE news_archive_model_reservations (
    reservation_id UUID PRIMARY KEY,
    day DATE NOT NULL,
    operation_key TEXT NOT NULL,
    request_id TEXT NOT NULL,
    reserved_usd NUMERIC(18, 10) NOT NULL CHECK (reserved_usd > 0),
    actual_usd NUMERIC(18, 10) CHECK (actual_usd >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    settled_at TIMESTAMPTZ,
    CHECK ((actual_usd IS NULL) = (settled_at IS NULL))
);

CREATE INDEX news_archive_model_reservations_day
    ON news_archive_model_reservations(day);

CREATE TRIGGER news_archive_model_reservations_reject_deletes
BEFORE DELETE ON news_archive_model_reservations
FOR EACH ROW EXECUTE FUNCTION reject_archive_sitemap_mutation();
