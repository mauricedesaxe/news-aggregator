"""Choose the oldest verified month with uncaptured publisher articles."""

import argparse
import json
import sys
from datetime import date

from romanian_news.archive.capture_batch import pending_archive_articles
from romanian_news.archive.windows import month_windows


def next_capture_config(outlet_id: str, start: date, end: date) -> dict[str, object] | None:
    for month_start, month_end in month_windows(start, end):
        if pending_archive_articles(outlet_id, month_start, month_end, 1):
            return {
                "ops": {
                    "archive_article_capture": {
                        "config": {
                            "outlet": outlet_id,
                            "start": month_start.isoformat(),
                            "end": month_end.isoformat(),
                            "limit": 50,
                        }
                    }
                }
            }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outlet", choices=("hotnews", "digi24"), required=True)
    args = parser.parse_args()
    config = next_capture_config(
        args.outlet,
        date(2025, 9, 27),
        date(2026, 9, 26),
    )
    sys.stdout.write(json.dumps(config, sort_keys=True) if config else "complete")
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
