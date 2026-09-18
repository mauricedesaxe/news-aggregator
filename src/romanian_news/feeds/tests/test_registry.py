from romanian_news.feeds.models import FeedSpec
from romanian_news.feeds.registry import feed_registry


def _feed(feed_id: str) -> FeedSpec:
    return {feed.id: feed for feed in feed_registry().feeds}[feed_id]


def test_registry_has_only_unique_verified_feeds() -> None:
    registry = feed_registry()

    assert len(registry.feeds) == 42
    assert registry.feeds[0].id == "adevarul"
    assert registry.feeds[-1].id == "ziarul-financiar"

    rfi_feeds = {feed.id: feed for feed in registry.feeds if feed.outlet_id == "rfi-romania"}
    assert {
        feed_id: (feed.outlet_id, feed.category, str(feed.url))
        for feed_id, feed in rfi_feeds.items()
    } == {
        "rfi-romania": ("rfi-romania", "general", "https://www.rfi.fr/ro/rss"),
        "rfi-romania-economia-reala": (
            "rfi-romania",
            "economic",
            "https://www.rfi.fr/ro/podcasturi/economia-real%C4%83/podcast",
        ),
    }

    special_arad_feeds = {
        feed.id: feed for feed in registry.feeds if feed.outlet_id == "special-arad"
    }
    assert {
        feed_id: (feed.outlet_id, feed.outlet_name, feed.category, str(feed.url))
        for feed_id, feed in special_arad_feeds.items()
    } == {
        "special-arad": (
            "special-arad",
            "Special Arad",
            "regional",
            "https://specialarad.ro/articole/stiri-arad/feed/",
        ),
        "special-arad-investigatii": (
            "special-arad",
            "Special Arad",
            "regional",
            "https://specialarad.ro/articole/investigatii/feed/",
        ),
        "special-arad-reportaj": (
            "special-arad",
            "Special Arad",
            "regional",
            "https://specialarad.ro/articole/reportaj/feed/",
        ),
    }


def test_registry_configures_dcnews() -> None:
    dcnews = _feed("dcnews")

    assert dcnews.id == "dcnews"
    assert dcnews.outlet_id == "dcnews"
    assert dcnews.outlet_name == "DCNews"
    assert dcnews.category == "general"
    assert str(dcnews.url) == "https://www.dcnews.ro/rss"
    assert dcnews.article_hosts == ("dcnews.ro",)
    assert dcnews.article_xpath is None
    assert dcnews.excluded_article_path_prefixes == ()


def test_registry_configures_gazeta_de_sud() -> None:
    gazeta_de_sud = _feed("gazeta-de-sud")

    assert gazeta_de_sud.id == "gazeta-de-sud"
    assert gazeta_de_sud.outlet_id == "gazeta-de-sud"
    assert gazeta_de_sud.outlet_name == "Gazeta de Sud"
    assert gazeta_de_sud.category == "regional"
    assert str(gazeta_de_sud.url) == "https://www.gds.ro/Local/feed/"
    assert gazeta_de_sud.article_hosts == ("gds.ro",)
    assert gazeta_de_sud.article_xpath == (
        "//div[contains(concat(' ', normalize-space(@class), ' '), ' entry-content ')]"
    )
    assert gazeta_de_sud.excluded_article_path_prefixes == ()


def test_registry_configures_info_sud_est() -> None:
    info_sud_est = _feed("info-sud-est")

    assert info_sud_est.id == "info-sud-est"
    assert info_sud_est.outlet_id == "info-sud-est"
    assert info_sud_est.outlet_name == "Info Sud-Est"
    assert info_sud_est.category == "regional"
    assert str(info_sud_est.url) == "https://www.info-sud-est.ro/feed/"
    assert info_sud_est.article_hosts == ("info-sud-est.ro",)
    assert info_sud_est.article_xpath is None
    assert info_sud_est.excluded_article_path_prefixes == ()


def test_registry_configures_monitorul_botosani() -> None:
    monitorul_botosani = _feed("monitorul-botosani")

    assert monitorul_botosani.id == "monitorul-botosani"
    assert monitorul_botosani.outlet_id == "monitorul-botosani"
    assert monitorul_botosani.outlet_name == "Monitorul de Botoșani"
    assert monitorul_botosani.category == "regional"
    assert str(monitorul_botosani.url) == "https://www.monitorulbt.ro/category/prima-pagina/feed/"
    assert monitorul_botosani.article_hosts == ("monitorulbt.ro",)
    assert monitorul_botosani.article_xpath == (
        "//div[contains(concat(' ', normalize-space(@class), ' '), ' tdb_title ')]/parent::div[1]"
    )
    assert monitorul_botosani.excluded_article_path_prefixes == ("/national/",)


def test_registry_configures_observator_news() -> None:
    observator = _feed("observator-news")

    assert observator.id == "observator-news"
    assert observator.outlet_id == "observator-news"
    assert observator.category == "general"
    assert str(observator.url) == "https://observatornews.ro/rss"
    assert observator.article_hosts == ("observatornews.ro",)


def test_registry_configures_observatorul_prahovean() -> None:
    observatorul_prahovean = _feed("observatorul-prahovean")

    assert observatorul_prahovean.id == "observatorul-prahovean"
    assert observatorul_prahovean.outlet_id == "observatorul-prahovean"
    assert observatorul_prahovean.outlet_name == "Observatorul Prahovean"
    assert observatorul_prahovean.category == "regional"
    assert str(observatorul_prahovean.url) == "https://www.observatorulph.ro/feed"
    assert observatorul_prahovean.article_hosts == ("observatorulph.ro",)
    assert observatorul_prahovean.article_xpath is None
    assert observatorul_prahovean.excluded_article_path_prefixes == (
        "/advertorial/",
        "/national/",
        "/international/",
    )


def test_registry_configures_realitatea() -> None:
    realitatea = _feed("realitatea")

    assert realitatea.id == "realitatea"
    assert realitatea.outlet_id == "realitatea"
    assert realitatea.outlet_name == "Realitatea.net"
    assert realitatea.category == "general"
    assert str(realitatea.url) == "https://www.realitatea.net/access/share/feeds/rss/homepage.xml"
    assert realitatea.article_hosts == ("realitatea.net",)
    assert realitatea.article_xpath == (
        "(//article)[1]//div[contains(concat(' ', normalize-space(@class), ' '), ' prose ')][1]"
    )
    assert realitatea.excluded_article_path_prefixes == ()


def test_registry_configures_romania_tv() -> None:
    romania_tv = _feed("romania-tv")

    assert romania_tv.id == "romania-tv"
    assert romania_tv.outlet_id == "romania-tv"
    assert romania_tv.outlet_name == "România TV"
    assert romania_tv.category == "general"
    assert str(romania_tv.url) == "https://www.romaniatv.net/feed"
    assert romania_tv.article_hosts == ("romaniatv.net",)
    assert romania_tv.article_xpath is None
    assert romania_tv.excluded_article_path_prefixes == ()


def test_registry_configures_news_ro() -> None:
    news_ro = _feed("news-ro")

    assert news_ro.outlet_id == "news-ro"
    assert news_ro.outlet_name == "News.ro"
    assert news_ro.category == "general"
    assert str(news_ro.url) == "https://www.news.ro/rss"
    assert news_ro.article_hosts == ("news.ro",)
    assert news_ro.article_xpath is None
    assert news_ro.excluded_article_path_prefixes == ()


def test_registry_scopes_newscenter_extraction_to_article_body() -> None:
    assert _feed("newscenter").article_xpath == (
        "//article[contains(@class, 'type-post')]//div[contains(@class, 'entry-content')]"
    )


def test_excluded_article_path_prefixes_reject_matching_article_urls() -> None:
    monitorul_botosani = _feed("monitorul-botosani")

    assert monitorul_botosani.accepts_article_url("https://www.monitorulbt.ro/national/x") is False
    assert monitorul_botosani.accepts_article_url("https://www.monitorulbt.ro/politica/x") is True

    observatorul_prahovean = _feed("observatorul-prahovean")
    assert (
        observatorul_prahovean.accepts_article_url("https://www.observatorulph.ro/advertorial/x")
        is False
    )
    assert (
        observatorul_prahovean.accepts_article_url("https://www.observatorulph.ro/stiri/x") is True
    )
