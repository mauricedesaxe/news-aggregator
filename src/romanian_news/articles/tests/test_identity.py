import pytest

from romanian_news.articles.extraction import article_id, normalize_article_url


def test_article_identity_ignores_tracking_and_url_decoration() -> None:
    first = normalize_article_url(
        "http://WWW.digi24.ro/stiri/example/?utm_source=rss&b=2&a=1#fragment",
        ("digi24.ro",),
    )
    second = normalize_article_url(
        "https://digi24.ro/stiri/example?a=1&b=2",
        ("digi24.ro",),
    )

    assert first == second == "https://digi24.ro/stiri/example?a=1&b=2"
    assert article_id("digi24", first) == article_id("digi24", second)


def test_rfi_article_identity_ignores_audio_attribution() -> None:
    decorated = normalize_article_url(
        "https://www.rfi.fr/ro/rom%C3%A2nia/exemplu?GJlXzGgqHj",
        ("rfi.fr",),
    )
    bare = normalize_article_url(
        "https://www.rfi.fr/ro/rom%C3%A2nia/exemplu",
        ("rfi.fr",),
    )

    assert decorated == bare == "https://rfi.fr/ro/rom%C3%A2nia/exemplu"
    assert article_id("rfi-romania", decorated) == article_id("rfi-romania", bare)


def test_article_identity_rejects_an_unregistered_canonical_host() -> None:
    with pytest.raises(ValueError, match="host is not registered"):
        normalize_article_url("https://tracking.example/article", ("digi24.ro",))
