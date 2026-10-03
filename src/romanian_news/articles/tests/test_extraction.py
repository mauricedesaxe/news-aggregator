from datetime import datetime
from pathlib import Path

import pytest
from pydantic import HttpUrl

from romanian_news.articles.extraction import extract_archive_article, extract_article
from romanian_news.feeds.models import FeedEntry
from romanian_news.feeds.registry import feed_registry

_FIXTURES = Path(__file__).parent / "fixtures"


def test_material_fingerprint_ignores_metadata_only_changes() -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    first = extract_article(
        _entry("2026-08-31T09:00:00+00:00"),
        feed,
        html=_article_html("Primul paragraf are suficient text pentru o știre reală. " * 3),
    )
    second = extract_article(
        _entry("2026-08-31T10:00:00+00:00"),
        feed,
        html=_article_html("Primul paragraf are suficient text pentru o știre reală. " * 3),
    )

    assert first.article_id == second.article_id
    assert first.material_digest == second.material_digest
    assert first.source_updated_at != second.source_updated_at


def test_material_fingerprint_changes_with_editorial_content() -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    first = extract_article(
        _entry("2026-08-31T09:00:00+00:00"),
        feed,
        html=_article_html("Primul paragraf are suficient text pentru o știre reală. " * 3),
    )
    second = extract_article(
        _entry("2026-08-31T10:00:00+00:00"),
        feed,
        html=_article_html("Textul corectat schimbă afirmația publicată în articol. " * 3),
    )

    assert first.material_digest != second.material_digest


def test_selector_drift_falls_back_to_feed_summary() -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    entry = _entry("2026-08-31T09:00:00+00:00").model_copy(
        update={"summary": "Rezumatul din flux păstrează conținutul articolului. " * 3}
    )

    article = extract_article(
        entry,
        feed,
        html=b"<html><body><main>Structura paginii s-a schimbat.</main></body></html>",
    )

    assert article.body == ("Rezumatul din flux păstrează conținutul articolului. " * 3).strip()


def test_long_title_preserves_an_article_with_a_short_summary() -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    entry = _entry("2026-08-31T09:00:00+00:00").model_copy(
        update={
            "title": "Titlul lung identifică integral evenimentul important relatat de publicație",
            "summary": "Rezumat scurt.",
        }
    )

    article = extract_article(entry, feed, html=None)

    assert article.body == "Rezumat scurt."


def test_long_title_preserves_an_article_without_a_summary() -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    title = "Titlul lung identifică integral evenimentul important relatat de publicație"
    entry = _entry("2026-08-31T09:00:00+00:00").model_copy(update={"title": title, "summary": ""})

    article = extract_article(entry, feed, html=None)

    assert article.body == title


def test_digi24_extraction_preserves_romanian_encoding() -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "digi24")
    entry = FeedEntry(
        feed_id="digi24",
        source_id="synthetic-article",
        url=HttpUrl("https://digi24.ro/stiri/synthetic-article"),
        title="Articol sintetic pentru verificarea diacriticelor românești",
        summary="",
        feed_content=None,
        published_at=datetime.fromisoformat("2026-09-05T12:00:00+00:00"),
        source_updated_at=None,
        author="Synthetic author",
    )

    article = extract_article(
        entry,
        feed,
        html=(_FIXTURES / "digi24_synthetic.html").read_bytes(),
    )

    assert article.title == "„Un titlu cu diacritice românești”"
    assert "articol sintetic despre un proiect public imaginar" in article.body
    assert "caracterelor ă, â, î, ș și ț" in article.body


def test_archive_extraction_uses_full_page_when_selected_fragment_has_no_text() -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    paragraphs = "".join(
        f"Paragraph {index} describes a public project with a concrete update and context." "<br>"
        for index in range(12)
    )
    content = (
        "<html><head><title>Acme project update</title>"
        '<meta property="og:title" content="Acme project update">'
        '<meta property="article:published_time" content="2025-09-27T10:00:00+03:00">'
        "</head><body><main><article><h1>Acme project update</h1>"
        '<div class="article-single-content"><p class="wp-block-paragraph">'
        f"{paragraphs}</p></div></article></main></body></html>"
    ).encode()

    article = extract_archive_article(content, "https://hotnews.ro/acme-project-123", feed)

    assert article.bucharest_day.isoformat() == "2025-09-27"
    assert "Paragraph 11" in article.body


def test_archive_extraction_rejects_missing_date_and_short_page() -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    content = _article_html("Short.")
    with pytest.raises(ValueError, match="verified publication date"):
        extract_archive_article(content, "https://hotnews.ro/articol-test-123", feed)

    dated = content.replace(
        b"<title>",
        b'<meta property="og:title" content="Titlul articolului">'
        b'<meta property="article:published_time" content="2025-09-19T10:00:00+03:00">'
        b"<title>",
    )
    with pytest.raises(ValueError, match="content is too short"):
        extract_archive_article(
            dated,
            "https://hotnews.ro/articol-test-123",
            feed,
        )


def _entry(updated_at: str) -> FeedEntry:
    return FeedEntry(
        feed_id="hotnews",
        source_id="post-123",
        url=HttpUrl("https://hotnews.ro/articol-test-123"),
        title="Titlul articolului",
        summary="Rezumatul articolului este disponibil în flux.",
        feed_content=None,
        published_at=datetime.fromisoformat("2026-08-31T08:00:00+00:00"),
        source_updated_at=datetime.fromisoformat(updated_at),
        author="Autor",
    )


def _article_html(body: str) -> bytes:
    return f"""
        <html lang="ro">
          <head>
            <title>Titlul articolului</title>
            <link rel="canonical" href="https://hotnews.ro/articol-test-123">
          </head>
          <body>
            <nav>Nu intră în articol.</nav>
            <article class="article-single-content">
              <h1>Titlul articolului</h1><p>{body}</p>
            </article>
          </body>
        </html>
    """.encode()
