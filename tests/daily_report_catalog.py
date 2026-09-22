from __future__ import annotations

import hashlib
from datetime import date

from romanian_news.reports import (
    DailyReport,
    DailyReportSection,
    ReportArticle,
    ReportEvent,
    ReportSubjectCitation,
)
from tests.postgres_catalog import PostgresCatalog

REPORT_VERSION = "a" * 64
RUN_ID = "7" * 64
INPUT_VERSION = "3" * 64
INPUT_DIGEST = "9" * 64
FILE_ID = "1" * 64
GROUP_ID = "c" * 64
ARTICLE_VERSION = "d" * 64
THEME_ID = "f" * 64
CAPTURED_AT = "2026-09-20T06:00:00+00:00"


def daily_report(day: date) -> DailyReport:
    event = ReportEvent(
        group_id=GROUP_ID,
        title_ro="The public budget enters debate",
        summary_ro="The government published the draft, and reactions differ between sources.",
        key_points_ro=("The estimated deficit remains the central issue.",),
        disagreements_ro=("Sources assess the effect of taxes differently.",),
        uncertainty_ro="Budget execution could change the estimate.",
        sentiment_label="mixed",
        sentiment_score=-0.1,
        sentiment_rationale_ro="The tone combines caution with moderate expectations.",
        articles=(
            ReportArticle(
                article_version_id=ARTICLE_VERSION,
                outlet_id="presa-exemplu",
                title="Analiză economică a proiectului",
                canonical_url="https://example.com/analiza",
                sentiment_label="neutral",
                sentiment_score=0.0,
            ),
        ),
    )
    return DailyReport(
        day=day,
        accepted_article_count=1,
        theme_count=1,
        group_count=1,
        sections=(
            DailyReportSection(
                theme_id=THEME_ID,
                title="Budget policy",
                summary="The draft budget and reactions form the subject of the day.",
                tier="main",
                semantic_rank=1,
                consequence_rationale="Budget decisions with direct national effects.",
                citations=(
                    ReportSubjectCitation(
                        article_version_id=ARTICLE_VERSION,
                        evidence_quote="Guvernul a publicat proiectul.",
                    ),
                ),
                events=(event,),
            ),
        ),
    )


def seed_daily_report(catalog: PostgresCatalog, report: DailyReport) -> bytes:
    day = report.day
    payload = report.model_dump_json().encode()
    digest = hashlib.sha256(payload).hexdigest()
    r2_key = f"news/reports/daily/{day.isoformat()}/{digest}.json"
    catalog.execute(
        "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            RUN_ID,
            "news.report_daily",
            "python",
            "reader-e2e",
            f'{{"day": "{day.isoformat()}"}}',
            "chartly",
            "completed",
            f"news.report_daily:{RUN_ID}",
            None,
            CAPTURED_AT,
            CAPTURED_AT,
        ),
    )
    catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state,"
        " visibility, current_version_id, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (
            f"news:daily:{day.isoformat()}",
            "news_daily_report",
            "Daily report",
            "derived",
            "current",
            "private",
            None,
            CAPTURED_AT,
        ),
    )
    catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state,"
        " visibility, current_version_id, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (
            f"news:themes:{day.isoformat()}",
            "news_daily_themes",
            "Daily themes",
            "derived",
            "current",
            "private",
            None,
            CAPTURED_AT,
        ),
    )
    catalog.execute(
        "INSERT INTO artifact_versions VALUES (%s, %s, %s, %s, %s, %s)",
        (REPORT_VERSION, f"news:daily:{day.isoformat()}", 3, digest, None, CAPTURED_AT),
    )
    catalog.execute(
        "INSERT INTO artifact_versions VALUES (%s, %s, %s, %s, %s, %s)",
        (INPUT_VERSION, f"news:themes:{day.isoformat()}", 1, INPUT_DIGEST, None, CAPTURED_AT),
    )
    catalog.execute(
        "INSERT INTO artifact_files VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (FILE_ID, REPORT_VERSION, r2_key, "application/json", digest, len(payload), None, None),
    )
    catalog.execute(
        "INSERT INTO run_inputs VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (RUN_ID, 0, INPUT_VERSION, "themes", None, INPUT_DIGEST, "whole_file", None),
    )
    catalog.execute(
        "INSERT INTO run_outputs VALUES (%s, %s, %s, %s)",
        (RUN_ID, 0, REPORT_VERSION, "output"),
    )
    catalog.execute(
        "UPDATE artifacts SET current_version_id = %s, current_run_id = %s WHERE id = %s",
        (REPORT_VERSION, RUN_ID, f"news:daily:{day.isoformat()}"),
    )
    catalog.execute(
        "UPDATE artifacts SET current_version_id = %s WHERE id = %s",
        (INPUT_VERSION, f"news:themes:{day.isoformat()}"),
    )
    return payload
