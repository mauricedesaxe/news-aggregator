from datetime import UTC, datetime

from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog.video_digest_monitor import read_due_video_incidents


def test_video_incident_queries_match_the_postgres_schema(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    assert read_due_video_incidents(datetime(2026, 9, 25, tzinfo=UTC)) == ()
