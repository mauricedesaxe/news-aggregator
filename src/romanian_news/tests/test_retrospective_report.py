from datetime import date

import pytest

from romanian_news import retrospective_report

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
