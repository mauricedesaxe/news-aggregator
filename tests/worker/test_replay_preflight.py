from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace

from romanian_news.archive import replay_preflight
from romanian_news.catalog.archive_model_spend import ArchiveSpend
from romanian_news.reports import RetrospectiveCoverage


def test_preflight_reads_exact_inputs_without_provider_work(monkeypatch) -> None:
    day = date(2025, 9, 27)
    coverage = RetrospectiveCoverage(
        capture_started_at=datetime(2026, 9, 27, 10, tzinfo=UTC),
        capture_ended_at=datetime(2026, 9, 27, 11, tzinfo=UTC),
        included_outlets=("digi24", "hotnews"),
        discovered_url_count=2,
        verified_page_count=2,
        captured_article_count=2,
        coverage_note="Measured",
    )
    monkeypatch.setattr(replay_preflight, "read_retrospective_coverage", lambda _day: coverage)
    monkeypatch.setattr(
        replay_preflight,
        "read_daily_article_references",
        lambda _day: SimpleNamespace(
            values=(
                SimpleNamespace(version_id="article-a"),
                SimpleNamespace(version_id="article-b"),
            )
        ),
    )
    monkeypatch.setattr(
        replay_preflight, "read_pending_relevance_references", lambda **_kwargs: (object(),)
    )
    monkeypatch.setattr(replay_preflight, "read_pending_embedding_references", lambda **_kwargs: ())
    monkeypatch.setattr(
        replay_preflight,
        "read_pending_group_analysis_references",
        lambda _days: (SimpleNamespace(summary_needed=True, sentiment_needed=False),),
    )
    monkeypatch.setattr(
        replay_preflight,
        "read_archive_spend",
        lambda _day: ArchiveSpend(Decimal("0.1"), Decimal(0), 4),
    )

    result = replay_preflight.read_replay_preflight(day)

    assert result.article_version_ids == ("article-a", "article-b")
    assert (result.pending_relevance, result.pending_embeddings) == (1, 0)
    assert (result.pending_summaries, result.pending_sentiment) == (1, 0)
    assert result.estimated_remaining_usd == Decimal("0.0082")
    assert result.source_ready and result.paid_work_admissible
