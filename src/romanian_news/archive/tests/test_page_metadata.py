from datetime import UTC, datetime

from romanian_news.archive.page_metadata import extract_archive_page_metadata


def test_page_publication_is_distinct_from_later_modification() -> None:
    content = b"""<html><head>
      <meta property="og:title" content="A news story">
      <meta property="article:published_time" content="2025-09-19T10:00:00+03:00">
      <meta property="article:modified_time" content="2026-09-27T11:00:00+03:00">
      <script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2025-09-19T10:00:00+03:00"}</script>
    </head></html>"""

    result = extract_archive_page_metadata(content)

    assert result.published_at == datetime(2025, 9, 19, 7, tzinfo=UTC)
    assert result.modified_at == datetime(2026, 9, 27, 8, tzinfo=UTC)
    assert result.publication_evidence == "article:published_time"
    assert result.rejection is None


def test_conflicting_page_publication_days_are_rejected() -> None:
    content = b"""<html><head>
      <meta property="og:title" content="A news story">
      <meta property="article:published_time" content="2025-09-19T10:00:00+03:00">
      <script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2025-09-20T10:00:00+03:00"}</script>
    </head></html>"""

    result = extract_archive_page_metadata(content)

    assert result.published_at is None
    assert result.rejection == "conflicting_publication_dates"


def test_unrelated_jsonld_date_does_not_date_an_article() -> None:
    content = b"""<html><head><title>A news story</title>
      <script type="application/ld+json">{"@type":"WebSite","datePublished":"2025-09-19"}</script>
    </head></html>"""

    result = extract_archive_page_metadata(content)

    assert result.published_at is None
    assert result.rejection == "missing_publication_date"


def test_date_only_metadata_does_not_invent_a_publication_time() -> None:
    content = b"""<html><head>
      <meta property="og:title" content="A news story">
      <script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2025-09-19"}</script>
    </head></html>"""

    result = extract_archive_page_metadata(content)

    assert result.published_at is None
    assert result.rejection == "missing_publication_date"
