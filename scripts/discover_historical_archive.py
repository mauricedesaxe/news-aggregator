"""Record bounded historical sitemap discoveries without capturing article text."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from romanian_news.archive.discovery import discover_sitemaps
from romanian_news.catalog.schema import ensure_news_catalog_schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outlet", choices=("hotnews", "digi24"), required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    args = parser.parse_args()
    ensure_news_catalog_schema()
    observations = discover_sitemaps(
        args.outlet, args.start, args.end, delay_seconds=args.delay_seconds
    )
    sys.stdout.write(
        json.dumps(
            [
                {
                    "observation_id": item.id,
                    "outlet": item.outlet_id,
                    "sitemap_url": item.sitemap_url,
                    "fetched_at": item.fetched_at.isoformat(),
                    "unique_urls": len(item.entries),
                }
                for item in observations
            ],
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
