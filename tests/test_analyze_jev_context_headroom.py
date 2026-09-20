from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts/analyze_jev_context_headroom.py"
SPEC = importlib.util.spec_from_file_location("analyze_jev_context_headroom", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def test_analyzer_reports_provider_counted_headroom_for_all_concerns(tmp_path: Path) -> None:
    artifact_directory = ROOT / "artifacts/jev-aspect-v1"
    artifacts = tuple(sorted(artifact_directory.glob("news-*-live-v1.json")))

    report = analyzer.analyze(ROOT / "src/romanian_news/jev_context_limits.json", artifacts)
    by_concern = {item["concern"]: item for item in report["concerns"]}

    assert set(by_concern) == {"grouping", "ranking", "tier", "confidence", "daily_theme"}
    assert {concern: item["request_statistics"]["max"] for concern, item in by_concern.items()} == {
        "grouping": 5359,
        "ranking": 18069,
        "tier": 14369,
        "confidence": 2869,
        "daily_theme": 847,
    }
    assert by_concern["ranking"]["request_statistics"]["max_utilization_percent"] == 56.465625
    assert by_concern["ranking"]["request_statistics"]["minimum_headroom_tokens"] == 13931
    assert by_concern["ranking"]["request_statistics"]["thresholds"][0]["count_at_or_above"] == 1
    assert all(
        item["request_statistics"]["thresholds"][-1]["count_at_or_above"] == 0
        for item in by_concern.values()
    )

    json_output = tmp_path / "headroom.json"
    markdown_output = tmp_path / "headroom.md"
    analyzer.write_reports(report, json_output, markdown_output)
    first = (json_output.read_bytes(), markdown_output.read_bytes())
    analyzer.write_reports(report, json_output, markdown_output)

    assert (json_output.read_bytes(), markdown_output.read_bytes()) == first
    assert "Pairwise benchmark requests" in markdown_output.read_text()
