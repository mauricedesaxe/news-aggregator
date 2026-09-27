from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from romanian_news.reports import RetrospectiveCoverage
from romanian_news.worker import retrospective_analysis

DAY = date(2025, 9, 27)


def _coverage(outlets: tuple[str, ...]) -> RetrospectiveCoverage:
    return RetrospectiveCoverage(
        capture_started_at=datetime(2026, 9, 27, 10, tzinfo=UTC),
        capture_ended_at=datetime(2026, 9, 27, 11, tzinfo=UTC),
        included_outlets=outlets,
        discovered_url_count=50,
        verified_page_count=50,
        captured_article_count=45,
        coverage_note="Verified publisher pages only.",
    )


def test_pilot_runs_analysis_in_report_input_order(monkeypatch) -> None:
    monkeypatch.setattr(
        retrospective_analysis,
        "read_retrospective_coverage",
        lambda _day: _coverage(("digi24", "hotnews")),
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "read_daily_article_references",
        lambda _day: SimpleNamespace(values=(object(),) * 45),
    )
    called = []
    for name in (
        "materialize_relevance",
        "materialize_embeddings",
        "materialize_clusters",
        "materialize_group_summaries",
        "materialize_group_sentiment",
        "materialize_daily_themes",
        "materialize_subject_assessments",
    ):
        monkeypatch.setattr(
            retrospective_analysis,
            name,
            lambda _day, _ref, stage=name: called.append(stage),
        )

    def publish(_day, _ref):
        called.append("publish")
        return SimpleNamespace(version_id="a" * 64)

    monkeypatch.setattr(retrospective_analysis, "publish_retrospective_daily_report", publish)
    assert retrospective_analysis.analyze_retrospective_day(DAY, "git:test") == "a" * 64
    assert called == [
        "materialize_relevance",
        "materialize_embeddings",
        "materialize_clusters",
        "materialize_group_summaries",
        "materialize_group_sentiment",
        "materialize_daily_themes",
        "materialize_subject_assessments",
        "publish",
    ]


def test_pilot_rejects_sparse_coverage_before_model_work(monkeypatch) -> None:
    monkeypatch.setattr(
        retrospective_analysis,
        "read_retrospective_coverage",
        lambda _day: _coverage(("hotnews",)),
    )
    monkeypatch.setattr(
        retrospective_analysis,
        "read_daily_article_references",
        lambda _day: pytest.fail("article inputs should not load for one outlet"),
    )
    with pytest.raises(ValueError, match="two outlets"):
        retrospective_analysis.analyze_retrospective_day(DAY, "git:test")
