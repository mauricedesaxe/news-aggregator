from datetime import UTC, datetime

import pytest

from romanian_news.archive.articles import ArchiveArticleCapture, capture_archive_article
from romanian_news.feeds.registry import feed_registry


def _page() -> bytes:
    return (
        "<html><head>"
        '<meta property="og:title" content="Anunț despre un proiect public">'
        '<meta property="article:published_time" content="2025-09-19T10:00:00+03:00">'
        '<meta property="article:modified_time" content="2025-09-20T12:00:00+03:00">'
        "</head><body><article class='article-single-content'>"
        "<h1>Anunț despre un proiect public</h1>"
        f"<p>{'Instituția a publicat detalii despre proiectul local. ' * 6}</p>"
        "</article></body></html>"
    ).encode()


def _capture() -> ArchiveArticleCapture:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    return capture_archive_article(
        observation_id="a" * 64,
        discovered_url="https://hotnews.ro/proiect-public-1",
        final_url="https://hotnews.ro/proiect-public-1",
        html=_page(),
        fetched_at=datetime(2026, 9, 27, 10, 0, tzinfo=UTC),
        feed=feed,
    )


def test_archive_capture_keeps_actual_fetch_time_without_html() -> None:
    first = _capture()
    replay = _capture()

    assert first == replay
    assert first.article.bucharest_day.isoformat() == "2025-09-19"
    assert first.article.source_updated_at is not None
    assert first.article.source_updated_at.year == 2025
    assert first.fetched_at.year == 2026
    assert first.publication_evidence == "article:published_time"
    assert "<html" not in first.model_dump_json()


def test_archive_capture_rejects_fetch_before_publication() -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    with pytest.raises(ValueError, match="predates article publication"):
        capture_archive_article(
            observation_id="a" * 64,
            discovered_url="https://hotnews.ro/proiect-public-1",
            final_url="https://hotnews.ro/proiect-public-1",
            html=_page(),
            fetched_at=datetime(2025, 9, 18, 10, 0, tzinfo=UTC),
            feed=feed,
        )


def test_archive_capture_rejects_naive_fetch_time() -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    with pytest.raises(ValueError, match="UTC offset"):
        capture_archive_article(
            observation_id="a" * 64,
            discovered_url="https://hotnews.ro/proiect-public-1",
            final_url="https://hotnews.ro/proiect-public-1",
            html=_page(),
            fetched_at=datetime(2026, 9, 27, 10, 0),
            feed=feed,
        )
