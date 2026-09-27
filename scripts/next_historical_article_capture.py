"""Choose the oldest verified month with uncaptured publisher articles."""

import argparse
import json
import sys
from datetime import date

from romanian_news.archive.capture_batch import next_capture_window


def next_capture_config(outlet_id: str, start: date, end: date) -> dict[str, object] | None:
    window = next_capture_window(outlet_id, start, end)
    if window is None:
        return None
    month_start, month_end = window
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
