CREATE TABLE daily_report_repair_requests (
    day DATE PRIMARY KEY,
    request_id UUID NOT NULL UNIQUE,
    run_id TEXT UNIQUE,
    launch_state TEXT NOT NULL DEFAULT 'reserved'
        CHECK (launch_state IN ('reserved', 'launch_started'))
);
