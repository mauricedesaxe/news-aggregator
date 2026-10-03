#!/usr/bin/env bash
set -euo pipefail

report_dir=${1:-coverage-results}
mkdir -p "$report_dir"

uv run --no-sync coverage run \
  --data-file="$report_dir/.coverage" \
  --source=romanian_news \
  --omit='*/tests/*' \
  -m pytest -q --junitxml="$report_dir/pytest.xml"
uv run --no-sync coverage json \
  --data-file="$report_dir/.coverage" \
  -o "$report_dir/coverage.json"
uv run --no-sync python - "$report_dir" <<'PY'
import json
import sys
from pathlib import Path
from xml.etree import ElementTree

report_dir = Path(sys.argv[1])
coverage = json.loads((report_dir / "coverage.json").read_text())["totals"]
suite = ElementTree.parse(report_dir / "pytest.xml").getroot().find("testsuite")
assert suite is not None
print(
    f"{suite.attrib['tests']} collected, {suite.attrib['skipped']} skipped, "
    f"{coverage['percent_covered']:.4f}% line coverage "
    f"({coverage['covered_lines']}/{coverage['num_statements']})"
)
PY
