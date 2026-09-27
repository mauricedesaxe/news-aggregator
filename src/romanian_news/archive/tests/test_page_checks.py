from urllib.robotparser import RobotFileParser

import requests

from romanian_news.archive import page_checks


def test_page_check_uses_page_date_and_replays_by_content(monkeypatch) -> None:
    candidate = page_checks.ArchivePageCandidate(
        observation_id="a" * 64,
        outlet_id="hotnews",
        canonical_url="https://hotnews.ro/story-1",
    )
    content = b"""<html><head>
      <meta property="og:title" content="A news story">
      <meta property="article:published_time" content="2025-09-19T10:00:00+03:00">
      <meta property="article:modified_time" content="2026-09-27T10:00:00+03:00">
    </head></html>"""
    monkeypatch.setattr(
        page_checks,
        "_fetch",
        lambda _session, _url: ("https://hotnews.ro/story-1", content),
    )
    robots = RobotFileParser()
    robots.parse(["User-agent: *", "Allow: /"])

    first = page_checks.check_page(candidate, requests.Session(), robots, ("hotnews.ro",))
    replay = page_checks.check_page(candidate, requests.Session(), robots, ("hotnews.ro",))

    assert first.id == replay.id
    assert first.status == "accepted"
    assert first.metadata is not None
    assert first.metadata.published_at is not None
    assert first.metadata.published_at.year == 2025
    assert first.metadata.modified_at is not None
    assert first.metadata.modified_at.year == 2026


def test_page_check_rejects_a_homepage_redirect(monkeypatch) -> None:
    candidate = page_checks.ArchivePageCandidate(
        observation_id="b" * 64,
        outlet_id="digi24",
        canonical_url="https://www.digi24.ro/story-1",
    )
    monkeypatch.setattr(
        page_checks,
        "_fetch",
        lambda _session, _url: ("https://www.digi24.ro/", b"<html></html>"),
    )
    robots = RobotFileParser()
    robots.parse(["User-agent: *", "Allow: /"])

    result = page_checks.check_page(candidate, requests.Session(), robots, ("digi24.ro",))

    assert result.status == "rejected"
    assert result.rejection == "redirected_to_homepage"
    assert result.metadata is None
