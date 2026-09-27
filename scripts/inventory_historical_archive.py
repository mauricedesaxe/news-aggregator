"""Print durable archive discovery coverage without inferring publication dates."""

import csv
import sys

from romanian_news.catalog_transport import catalog_query


def main() -> None:
    rows = catalog_query(
        """
        SELECT observation.outlet_id,
               observation.sitemap_url,
               count(DISTINCT observation.id) AS observations,
               count(entry.canonical_url) AS entries,
               count(DISTINCT entry.canonical_url) AS unique_urls,
               min(observation.fetched_at) AS first_fetched_at,
               max(observation.fetched_at) AS last_fetched_at
        FROM news_archive_sitemap_observations observation
        LEFT JOIN news_archive_sitemap_entries entry
          ON entry.observation_id = observation.id
        GROUP BY observation.outlet_id, observation.sitemap_url
        ORDER BY observation.outlet_id, observation.sitemap_url
        """,
        [],
    )
    writer = csv.writer(sys.stdout)
    writer.writerow(
        (
            "outlet",
            "sitemap_url",
            "observations",
            "entries",
            "unique_urls",
            "first_fetched_at",
            "last_fetched_at",
        )
    )
    for row in rows:
        writer.writerow(
            (
                row["outlet_id"],
                row["sitemap_url"],
                row["observations"],
                row["entries"],
                row["unique_urls"],
                row["first_fetched_at"],
                row["last_fetched_at"],
            )
        )


if __name__ == "__main__":
    main()
