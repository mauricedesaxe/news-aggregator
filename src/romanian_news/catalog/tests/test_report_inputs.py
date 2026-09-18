from datetime import UTC, date, datetime

import pytest

from romanian_news.catalog import report_inputs


def test_current_report_record_reads_head_and_input_timestamp_atomically(monkeypatch) -> None:
    day = date(2026, 9, 15)
    timestamp = datetime(2026, 9, 15, 7, 41, tzinfo=UTC)
    calls: list[tuple[str, list[object]]] = []

    def query(statement: str, parameters: list[object]):
        calls.append((statement, parameters))
        return [
            {
                "version_id": "a" * 64,
                "run_id": "c" * 64,
                "content_digest": "b" * 64,
                "r2_key": "news/reports/current.json",
                "input_time": timestamp.isoformat(),
            }
        ]

    monkeypatch.setattr(report_inputs, "catalog_query", query)

    result = report_inputs.read_current_daily_report_record(day)

    assert result is not None
    assert result.version_id == "a" * 64
    assert result.run_id == "c" * 64
    assert result.input_time == timestamp
    assert len(calls) == 1
    assert "report.current_version_id" in calls[0][0]
    assert "MAX(input_version.created_at)" in calls[0][0]
    assert "input.role != 'prior_output'" in calls[0][0]
    assert calls[0][1] == ["news:daily:2026-09-15"]


def test_current_input_versions_use_catalog_metadata_without_r2(monkeypatch) -> None:
    day = date(2026, 9, 15)
    rows = [
        {"role": "themes", "version_id": "1" * 64},
        {"role": "assessments", "version_id": "2" * 64},
        {"role": "cluster_set", "version_id": "3" * 64},
        {"role": "summary", "version_id": "5" * 64},
        {"role": "summary", "version_id": "4" * 64},
        {"role": "sentiment", "version_id": "7" * 64},
        {"role": "sentiment", "version_id": "6" * 64},
    ]
    calls: list[tuple[str, list[object]]] = []

    def query(statement: str, parameters: list[object]):
        calls.append((statement, parameters))
        return rows

    monkeypatch.setattr(report_inputs, "catalog_query", query)
    monkeypatch.setattr(
        "romanian_news.storage.read_verified_r2_object",
        lambda *_args: pytest.fail("freshness metadata must not read R2"),
    )

    result = report_inputs.read_current_daily_report_input_versions(day)

    assert result is not None
    assert result.summaries == ("4" * 64, "5" * 64)
    assert result.sentiments == ("6" * 64, "7" * 64)
    assert len(calls) == 1
    assert calls[0][0].lstrip().startswith("SELECT")
    assert "input.artifact_version_id = theme.version_id" in calls[0][0]


def test_current_input_versions_wait_until_each_summary_has_sentiment(monkeypatch) -> None:
    monkeypatch.setattr(
        report_inputs,
        "catalog_query",
        lambda *_args: [
            {"role": "themes", "version_id": "1" * 64},
            {"role": "assessments", "version_id": "2" * 64},
            {"role": "cluster_set", "version_id": "3" * 64},
            {"role": "summary", "version_id": "4" * 64},
        ],
    )

    assert report_inputs.read_current_daily_report_input_versions(date(2026, 9, 15)) is None


def test_recorded_inputs_are_read_by_captured_run_id(monkeypatch) -> None:
    day = date(2026, 9, 15)
    run_id = "a" * 64
    calls: list[tuple[str, list[object]]] = []

    def query(statement: str, parameters: list[object]):
        calls.append((statement, parameters))
        return [
            {"role": "themes", "version_id": "1" * 64},
            {"role": "assessments", "version_id": "2" * 64},
            {"role": "cluster_set", "version_id": "3" * 64},
        ]

    monkeypatch.setattr(report_inputs, "catalog_query", query)

    result = report_inputs.read_daily_report_run_input_versions(day, run_id)

    assert result is not None
    assert calls[0][1] == [run_id]
    assert "WHERE input.run_id = %s" in calls[0][0]
