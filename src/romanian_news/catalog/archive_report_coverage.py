"""Measured source coverage for one historical Bucharest publication day."""

from datetime import date

from romanian_news.catalog_transport import catalog_query
from romanian_news.daily import bucharest_day_window
from romanian_news.reports import RetrospectiveCoverage


def read_retrospective_coverage(day: date) -> RetrospectiveCoverage | None:
    """Count date-verified pages and distinct captured articles for a day."""
    start, end = bucharest_day_window(day)
    rows = catalog_query(
        """
        SELECT * FROM (
        WITH verified AS (
            SELECT DISTINCT outlet_id, canonical_url
            FROM news_archive_page_checks
            WHERE status = 'accepted' AND published_at >= %s AND published_at < %s
        ), captured AS (
            SELECT observation.outlet_id, capture.discovered_url,
                   MIN(capture.fetched_at) AS first_fetched_at,
                   MAX(capture.fetched_at) AS last_fetched_at
            FROM news_archive_article_captures capture
            JOIN news_archive_sitemap_observations observation
              ON observation.id = capture.observation_id
            WHERE capture.published_at >= %s AND capture.published_at < %s
            GROUP BY observation.outlet_id, capture.discovered_url
        )
        SELECT (SELECT COUNT(*) FROM verified) AS verified_page_count,
               COUNT(*) AS captured_article_count,
               ARRAY_AGG(DISTINCT captured.outlet_id ORDER BY captured.outlet_id)
                   AS included_outlets,
               MIN(captured.first_fetched_at) AS capture_started_at,
               MAX(captured.last_fetched_at) AS capture_ended_at
        FROM captured
        ) coverage
        """,
        [start, end, start, end],
    )
    if len(rows) != 1:
        raise ValueError(f"Expected one archive coverage row for {day.isoformat()}")
    row = rows[0]
    captured_count = int(row["captured_article_count"])
    if captured_count == 0:
        return None
    verified_count = int(row["verified_page_count"])
    if captured_count > verified_count:
        raise ValueError(f"Archive captures exceed verified pages for {day.isoformat()}")
    return RetrospectiveCoverage.model_validate(
        {
            "capture_started_at": row["capture_started_at"],
            "capture_ended_at": row["capture_ended_at"],
            "included_outlets": tuple(row["included_outlets"]),
            "discovered_url_count": verified_count,
            "verified_page_count": verified_count,
            "captured_article_count": captured_count,
            "coverage_note": (
                "Counts include only discovered URLs whose publication date was verified "
                "from a fetched page. Undated sitemap entries cannot be assigned to a day."
            ),
        },
        strict=False,
    )
