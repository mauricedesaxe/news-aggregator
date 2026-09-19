from datetime import datetime

from romanian_news.feeds.parsing import parse_feed
from romanian_news.feeds.registry import feed_registry


def test_parses_rss_and_rejects_entries_without_dates() -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "digi24")
    content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>Digi24</title>
      <item>
        <guid>article-1</guid><title>Guvernul anunta masuri economice</title>
        <link>https://www.digi24.ro/stiri/article-1</link>
        <pubDate>Mon, 31 Aug 2026 12:13:04 +0300</pubDate>
        <description><![CDATA[Un rezumat.]]></description>
      </item>
      <item>
        <guid>article-2</guid><title>Articol fara data</title>
        <link>https://www.digi24.ro/stiri/article-2</link>
      </item>
    </channel></rss>"""

    entries, rejected = parse_feed(feed, content)

    assert rejected == 1
    assert len(entries) == 1
    assert entries[0].source_id == "article-1"
    assert entries[0].published_at == datetime.fromisoformat("2026-08-31T09:13:04+00:00")


def test_parses_atom_content_and_update_time() -> None:
    feed = next(value for value in feed_registry().feeds if value.id == "wall-street")
    content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Wall-Street.ro</title>
      <entry>
        <id>rd-333073</id><title>BNR mentine dobanda</title>
        <link href="https://www.wall-street.ro/articol/Economie/333073"/>
        <published>2026-08-31T10:00:00+03:00</published>
        <updated>2026-08-31T11:00:00+03:00</updated>
        <content type="html">&lt;p&gt;Continut economic relevant.&lt;/p&gt;</content>
      </entry>
    </feed>"""

    entries, rejected = parse_feed(feed, content)

    assert rejected == 0
    assert entries[0].feed_content == "<p>Continut economic relevant.</p>"
    assert entries[0].source_updated_at == datetime.fromisoformat("2026-08-31T08:00:00+00:00")
