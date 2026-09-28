"""Read the publisher archive discovery inventory for the status archive."""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from urllib.parse import parse_qs, urlsplit

from romanian_news.catalog_transport import catalog_query
from romanian_news.daily import bucharest_day_window


@dataclass(frozen=True)
class ArchiveDiscoveryMonth:
    outlet_id: str
    month: date
    sitemap_count: int
    url_entries: int
    accepted_pages: int = 0
    rejected_pages: int = 0
    retryable_pages: int = 0
    captured_articles: int = 0


@dataclass(frozen=True)
class ArchiveDailyReport:
    day: date
    version_id: str


@dataclass(frozen=True)
class ArchiveDayEvidence:
    day: date
    verified_pages: int
    captured_articles: int
    captured_outlets: tuple[str, ...]


def list_archive_day_evidence(start: date, end: date) -> tuple[ArchiveDayEvidence, ...]:
    start_at = bucharest_day_window(start)[0]
    end_at = bucharest_day_window(end)[1]
    rows = catalog_query(
        """
        SELECT * FROM (
        WITH latest_checks AS (
            SELECT DISTINCT ON (outlet_id, canonical_url)
                   outlet_id, canonical_url, status, published_at
            FROM news_archive_page_checks
            WHERE outlet_id IN ('hotnews', 'digi24')
            ORDER BY outlet_id, canonical_url, fetched_at DESC, id DESC
        ), evidence AS (
            SELECT (published_at AT TIME ZONE 'Europe/Bucharest')::date AS day,
                   outlet_id, 'verified' AS kind, canonical_url AS url
            FROM latest_checks
            WHERE status = 'accepted' AND published_at >= %s AND published_at < %s
            UNION ALL
            SELECT (capture.published_at AT TIME ZONE 'Europe/Bucharest')::date AS day,
                   observation.outlet_id, 'captured' AS kind, capture.discovered_url AS url
            FROM news_archive_article_captures capture
            JOIN news_archive_sitemap_observations observation
              ON observation.id = capture.observation_id
            WHERE observation.outlet_id IN ('hotnews', 'digi24')
              AND capture.published_at >= %s AND capture.published_at < %s
        )
        SELECT day, outlet_id, kind, COUNT(DISTINCT url) AS article_count
        FROM evidence
        GROUP BY day, outlet_id, kind
        ORDER BY day
        ) day_evidence
        """,
        [start_at, end_at, start_at, end_at],
    )
    verified: dict[date, int] = defaultdict(int)
    captured: dict[date, int] = defaultdict(int)
    outlets: dict[date, set[str]] = defaultdict(set)
    for row in rows:
        day = date.fromisoformat(str(row["day"]))
        count = int(row["article_count"])
        if row["kind"] == "verified":
            verified[day] += count
        else:
            captured[day] += count
            if count:
                outlets[day].add(str(row["outlet_id"]))
    return tuple(
        ArchiveDayEvidence(day, verified[day], captured[day], tuple(sorted(outlets[day])))
        for day in sorted(verified.keys() | captured.keys())
    )


def list_archive_daily_reports(start: date, end: date) -> tuple[ArchiveDailyReport, ...]:
    rows = catalog_query(
        """
        SELECT artifact.id, artifact.current_version_id
        FROM artifacts artifact
        WHERE artifact.kind = 'news_daily_report'
          AND artifact.id BETWEEN %s AND %s
          AND artifact.current_version_id IS NOT NULL
        ORDER BY artifact.id DESC
        """,
        [f"news:daily:{start.isoformat()}", f"news:daily:{end.isoformat()}"],
    )
    return tuple(
        ArchiveDailyReport(
            day=date.fromisoformat(str(row["id"]).removeprefix("news:daily:")),
            version_id=str(row["current_version_id"]),
        )
        for row in rows
    )


def _sitemap_month(outlet_id: str, sitemap_url: str) -> date | None:
    if outlet_id == "hotnews":
        query = parse_qs(urlsplit(sitemap_url).query)
        year, month = query.get("yyyy"), query.get("mm")
        if year and month:
            return date(int(year[0]), int(month[0]), 1)
    if outlet_id == "digi24":
        match = re.search(r"sitemap-articles-(\d{4})-(\d{2})\.xml$", sitemap_url)
        if match:
            return date(int(match[1]), int(match[2]), 1)
    return None


def list_archive_discovery_months() -> tuple[ArchiveDiscoveryMonth, ...]:
    rows = catalog_query(
        """
        SELECT DISTINCT ON (outlet_id, sitemap_url)
               outlet_id, sitemap_url, entry_count
        FROM news_archive_sitemap_observations
        WHERE outlet_id IN ('hotnews', 'digi24')
        ORDER BY outlet_id, sitemap_url, fetched_at DESC, id DESC
        """,
        [],
    )
    totals: dict[tuple[str, date], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        outlet_id = str(row["outlet_id"])
        month = _sitemap_month(outlet_id, str(row["sitemap_url"]))
        if month is None:
            continue
        count = totals[outlet_id, month]
        count[0] += 1
        count[1] += int(row["entry_count"])
    checked_rows = catalog_query(
        """
        SELECT observation.outlet_id, observation.sitemap_url, latest.status,
               COUNT(*) AS page_count
        FROM (
            SELECT DISTINCT ON (check_record.outlet_id, check_record.canonical_url)
                   check_record.observation_id, check_record.status
            FROM news_archive_page_checks check_record
            WHERE check_record.outlet_id IN ('hotnews', 'digi24')
            ORDER BY check_record.outlet_id, check_record.canonical_url,
                     check_record.fetched_at DESC, check_record.id DESC
        ) latest
        JOIN news_archive_sitemap_observations observation
          ON observation.id = latest.observation_id
        GROUP BY observation.outlet_id, observation.sitemap_url, latest.status
        """,
        [],
    )
    checks: dict[tuple[str, date], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in checked_rows:
        outlet_id = str(row["outlet_id"])
        month = _sitemap_month(outlet_id, str(row["sitemap_url"]))
        if month is not None:
            checks[outlet_id, month][str(row["status"])] += int(row["page_count"])
    captured_rows = catalog_query(
        """
        SELECT DISTINCT observation.outlet_id, observation.sitemap_url,
               capture.discovered_url
        FROM news_archive_article_captures capture
        JOIN news_archive_sitemap_observations observation
          ON observation.id = capture.observation_id
        WHERE observation.outlet_id IN ('hotnews', 'digi24')
        """,
        [],
    )
    captured: dict[tuple[str, date], set[str]] = defaultdict(set)
    for row in captured_rows:
        outlet_id = str(row["outlet_id"])
        month = _sitemap_month(outlet_id, str(row["sitemap_url"]))
        if month is not None:
            captured[outlet_id, month].add(str(row["discovered_url"]))
    return tuple(
        ArchiveDiscoveryMonth(
            outlet_id=outlet_id,
            month=month,
            sitemap_count=counts[0],
            url_entries=counts[1],
            accepted_pages=checks[outlet_id, month]["accepted"],
            rejected_pages=checks[outlet_id, month]["rejected"],
            retryable_pages=checks[outlet_id, month]["retryable"],
            captured_articles=len(captured[outlet_id, month]),
        )
        for (outlet_id, month), counts in sorted(
            totals.items(), key=lambda item: (item[0][1], item[0][0]), reverse=True
        )
    )
