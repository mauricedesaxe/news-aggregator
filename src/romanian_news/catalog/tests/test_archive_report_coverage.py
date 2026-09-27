from datetime import UTC, date, datetime

import pytest

from romanian_news.catalog import archive_report_coverage


def test_retrospective_coverage_uses_verified_publication_day(monkeypatch) -> None:
    def query(sql, parameters):
        assert sql.lstrip().startswith("SELECT")
        assert (
            parameters
            == [
                datetime(2025, 9, 26, 21, tzinfo=UTC),
                datetime(2025, 9, 27, 21, tzinfo=UTC),
            ]
            * 2
        )
        return [
            {
                "verified_page_count": 12,
                "captured_article_count": 9,
                "included_outlets": ["digi24", "hotnews"],
                "capture_started_at": "2026-09-27T10:00:00+00:00",
                "capture_ended_at": "2026-09-27T11:00:00+00:00",
            }
        ]

    monkeypatch.setattr(archive_report_coverage, "catalog_query", query)
    coverage = archive_report_coverage.read_retrospective_coverage(date(2025, 9, 27))

    assert coverage is not None
    assert coverage.included_outlets == ("digi24", "hotnews")
    assert coverage.discovered_url_count == coverage.verified_page_count == 12
    assert coverage.captured_article_count == 9
    assert "Undated sitemap entries" in coverage.coverage_note


def test_retrospective_coverage_rejects_more_captures_than_verified(monkeypatch) -> None:
    monkeypatch.setattr(
        archive_report_coverage,
        "catalog_query",
        lambda *_: [
            {
                "verified_page_count": 1,
                "captured_article_count": 2,
                "included_outlets": ["hotnews"],
                "capture_started_at": "2026-09-27T10:00:00+00:00",
                "capture_ended_at": "2026-09-27T11:00:00+00:00",
            }
        ],
    )
    with pytest.raises(ValueError, match="exceed verified pages"):
        archive_report_coverage.read_retrospective_coverage(date(2025, 9, 27))
