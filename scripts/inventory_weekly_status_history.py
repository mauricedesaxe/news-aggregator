"""Print production evidence coverage for historical weekly status reads."""

import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from romanian_news.catalog_transport import catalog_query

FIRST_WEEK = date(2024, 8, 26)

SOURCES = {
    "daily_reports": """
        SELECT split_part(id, ':', 3)::date AS day, count(*) AS count
        FROM artifacts
        WHERE kind = 'news_daily_report' AND current_version_id IS NOT NULL
          AND id ~ '^news:daily:[0-9]{4}-[0-9]{2}-[0-9]{2}$'
        GROUP BY day
    """,
    "weekly_reads": """
        SELECT split_part(id, ':', 3)::date AS day, count(*) AS count
        FROM artifacts
        WHERE kind = 'news_weekly_read' AND current_version_id IS NOT NULL
          AND id ~ '^news:weekly_status:[0-9]{4}-[0-9]{2}-[0-9]{2}$'
        GROUP BY day
    """,
    "feed_observations": """
        SELECT (scheduled_slot AT TIME ZONE 'Europe/Bucharest')::date AS day,
               count(*) AS count
        FROM news_feed_observations
        WHERE status != 'failed'
        GROUP BY day
    """,
    "articles": """
        SELECT metadata.bucharest_day AS day,
               count(DISTINCT metadata.article_artifact_id) AS count
        FROM news_article_versions metadata
        JOIN artifact_files file ON file.artifact_version_id = metadata.artifact_version_id
        GROUP BY day
    """,
    "feed_entries": """
        SELECT (published_at AT TIME ZONE 'Europe/Bucharest')::date AS day,
               count(*) AS count
        FROM news_feed_entry_events
        WHERE observed_at <= published_at + interval '14 days'
        GROUP BY day
    """,
}

PRE_REPORT_SOURCES = {
    "articles": """
        SELECT metadata.bucharest_day AS day,
               count(DISTINCT metadata.article_artifact_id) AS total,
               count(DISTINCT metadata.article_artifact_id) FILTER (
                   WHERE (metadata.captured_at AT TIME ZONE 'Europe/Bucharest')::date
                         <= metadata.bucharest_day + 1
               ) AS timely
        FROM news_article_versions metadata
        JOIN artifact_files file ON file.artifact_version_id = metadata.artifact_version_id
        GROUP BY day
    """,
    "feed_entries": """
        SELECT (published_at AT TIME ZONE 'Europe/Bucharest')::date AS day,
               count(*) AS total,
               count(*) FILTER (
                   WHERE (observed_at AT TIME ZONE 'Europe/Bucharest')::date
                         <= (published_at AT TIME ZONE 'Europe/Bucharest')::date + 1
               ) AS timely
        FROM news_feed_entry_events
        GROUP BY day
    """,
}


def main() -> None:
    today = datetime.now(ZoneInfo("Europe/Bucharest")).date()
    last_week = today - timedelta(days=today.weekday() + 7)
    totals: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))
    for name, query in SOURCES.items():
        for row in catalog_query(query, []):
            day = row["day"]
            if not isinstance(day, date) or not isinstance(row["count"], int):
                raise TypeError(f"Unexpected {name} inventory row")
            week = day - timedelta(days=day.weekday())
            if FIRST_WEEK <= week <= last_week:
                totals[name][week] += row["count"]

    sys.stdout.write("week,daily_reports,weekly_reads,feed_observations,articles,feed_entries\n")
    week = FIRST_WEEK
    while week <= last_week:
        sys.stdout.write(
            f"{week.isoformat()}," + ",".join(str(totals[name][week]) for name in SOURCES) + "\n"
        )
        week += timedelta(days=7)

    first_report_week = min(
        (week for week, count in totals["daily_reports"].items() if count),
        default=last_week + timedelta(days=7),
    )
    earlier: dict[date, dict[str, tuple[int, int]]] = defaultdict(dict)
    for name, query in PRE_REPORT_SOURCES.items():
        for row in catalog_query(query, []):
            day = row["day"]
            if not isinstance(day, date):
                raise TypeError(f"Unexpected {name} source date")
            if FIRST_WEEK <= day < first_report_week:
                earlier[day][name] = (int(row["total"]), int(row["timely"]))

    sys.stdout.write(
        "pre_report_day,articles,articles_within_one_day,feed_entries,entries_within_one_day\n"
    )
    for day, sources in sorted(earlier.items()):
        articles, timely_articles = sources.get("articles", (0, 0))
        entries, timely_entries = sources.get("feed_entries", (0, 0))
        sys.stdout.write(
            f"{day.isoformat()},{articles},{timely_articles},{entries},{timely_entries}\n"
        )


if __name__ == "__main__":
    main()
