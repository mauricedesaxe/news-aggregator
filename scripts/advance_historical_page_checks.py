"""Advance one publisher's oldest unchecked month in the one-year archive."""

import argparse
import json
import sys

from romanian_news.archive.backfill import ARCHIVE_END, ARCHIVE_START, advance_page_checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outlet", choices=("hotnews", "digi24"), required=True)
    args = parser.parse_args()
    result = advance_page_checks(
        args.outlet,
        ARCHIVE_START,
        ARCHIVE_END,
    )
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
