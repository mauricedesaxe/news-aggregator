"""Read the publisher archive discovery inventory for the status archive."""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from urllib.parse import parse_qs, urlsplit

from romanian_news.catalog_transport import catalog_query


@dataclass(frozen=True)
class ArchiveDiscoveryMonth:
    outlet_id: str
    month: date
    sitemap_count: int
    url_entries: int


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
    return tuple(
        ArchiveDiscoveryMonth(
            outlet_id=outlet_id,
            month=month,
            sitemap_count=counts[0],
            url_entries=counts[1],
        )
        for (outlet_id, month), counts in sorted(
            totals.items(), key=lambda item: (item[0][1], item[0][0]), reverse=True
        )
    )
