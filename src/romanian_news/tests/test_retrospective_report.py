from datetime import UTC, date, datetime

import pytest

from romanian_news import retrospective_report
from romanian_news.catalog.reports import NewsDailyReportPublication
from romanian_news.reports import DailyReport, DailyReportOutput, RetrospectiveCoverage

DAY = date(2025, 9, 27)


def test_retrospective_publication_requires_captured_articles(monkeypatch) -> None:
    monkeypatch.setattr(retrospective_report, "read_retrospective_coverage", lambda _day: None)
    monkeypatch.setattr(
        retrospective_report,
        "read_daily_report_input_cached",
        lambda _day: pytest.fail("report inputs should not load without captured articles"),
    )
    with pytest.raises(ValueError, match="No captured publisher articles"):
        retrospective_report.publish_retrospective_daily_report(DAY, "git:test")


def test_retrospective_publication_passes_version_four_to_catalog(monkeypatch) -> None:
    coverage = RetrospectiveCoverage(
        capture_started_at=datetime(2026, 9, 27, 10, tzinfo=UTC),
        capture_ended_at=datetime(2026, 9, 27, 11, tzinfo=UTC),
        included_outlets=("hotnews",),
        discovered_url_count=2,
        verified_page_count=2,
        captured_article_count=2,
        coverage_note="Only verified captured pages are counted.",
    )
    live = DailyReport(day=DAY, accepted_article_count=1, theme_count=0, group_count=0, sections=())
    base = DailyReportOutput.model_construct(
        request_id="a" * 64,
        report=live,
        themes=None,
        assessments=None,
        cluster_set=None,
        summaries=(),
        sentiments=(),
        content_digest="b" * 64,
        content=b"{}",
    )
    publication = NewsDailyReportPublication(
        status="published",
        run_id="c" * 64,
        version_id="d" * 64,
        uploaded_objects=1,
        reused_objects=0,
    )
    monkeypatch.setattr(retrospective_report, "read_retrospective_coverage", lambda _day: coverage)
    monkeypatch.setattr(retrospective_report, "read_daily_report_input_cached", lambda _day: None)
    monkeypatch.setattr(retrospective_report, "build_daily_report_from_input", lambda _value: base)

    def publish(output, implementation_ref):
        assert output.report.schema_version == 4
        assert output.report.retrospective == coverage
        assert implementation_ref == "git:test"
        return publication

    monkeypatch.setattr(retrospective_report, "publish_daily_report", publish)
    assert retrospective_report.publish_retrospective_daily_report(DAY, "git:test") == publication
