import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Literal
from uuid import UUID

import pytest

from romanian_news.feedback import (
    ArticleFeedbackTarget,
    DailyReportVersionNotFound,
    GroupFeedbackTarget,
    NewsFeedbackCommand,
    ReportFeedbackTarget,
    ThemeFeedbackTarget,
    list_daily_reports,
    read_daily_report_version,
    read_latest_news_feedback,
    resolve_current_daily_report_version,
    submit_news_feedback,
)
from romanian_news.reports import (
    ArchivedDailyReport,
    ArchivedDailyReportSection,
    DailyReport,
    DailyReportSection,
    ReportArticle,
    ReportEvent,
    ReportSubjectCitation,
)
from romanian_news.storage import ResearchObjectIntegrityError

REPORT_V1 = "a" * 64
REPORT_V2 = "b" * 64
GROUP_ID = "c" * 64
ARTICLE_ID = "d" * 64
OTHER_ID = "e" * 64
THEME_ID = "f" * 64


def test_submit_news_feedback_records_all_four_target_scopes(monkeypatch) -> None:
    with closing(_feedback_database()) as connection:
        _patch_database(monkeypatch, connection)
        monkeypatch.setattr(
            "romanian_news.feedback.read_daily_report_version", lambda _version: _report()
        )
        commands = (
            NewsFeedbackCommand(
                feedback_id=UUID("00000000-0000-4000-8000-000000000001"),
                target=ReportFeedbackTarget(report_version_id=REPORT_V1),
                rating="positive",
            ),
            NewsFeedbackCommand(
                feedback_id=UUID("00000000-0000-4000-8000-000000000002"),
                target=ThemeFeedbackTarget(
                    report_version_id=REPORT_V1,
                    theme_id=THEME_ID,
                ),
                rating="negative",
                note="Summary omitted the policy response.",
            ),
            NewsFeedbackCommand(
                feedback_id=UUID("00000000-0000-4000-8000-000000000004"),
                target=GroupFeedbackTarget(
                    report_version_id=REPORT_V1,
                    group_id=GROUP_ID,
                ),
                rating="positive",
            ),
            NewsFeedbackCommand(
                feedback_id=UUID("00000000-0000-4000-8000-000000000003"),
                target=ArticleFeedbackTarget(
                    report_version_id=REPORT_V1,
                    group_id=GROUP_ID,
                    article_version_id=ARTICLE_ID,
                ),
                rating="positive",
            ),
        )

        events = tuple(submit_news_feedback(command) for command in commands)

        assert tuple(event.target.kind for event in events) == (
            "report",
            "theme",
            "group",
            "article",
        )
        assert connection.execute("SELECT count(*) FROM news_feedback").fetchone()[0] == 4
        assert all(event.actor == "owner" for event in events)


def test_submit_news_feedback_converges_on_exact_retry_and_rejects_conflict(monkeypatch) -> None:
    with closing(_feedback_database()) as connection:
        _patch_database(monkeypatch, connection)
        monkeypatch.setattr(
            "romanian_news.feedback.read_daily_report_version", lambda _version: _report()
        )
        command = NewsFeedbackCommand(
            feedback_id=UUID("00000000-0000-4000-8000-000000000006"),
            target=ReportFeedbackTarget(report_version_id=REPORT_V1),
            rating="positive",
            note="Useful overview.",
        )

        first = submit_news_feedback(command)
        retry = submit_news_feedback(command)

        assert retry == first
        assert connection.execute("SELECT count(*) FROM news_feedback").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError, match="identity conflict"):
            submit_news_feedback(command.model_copy(update={"rating": "negative"}))


def test_submit_news_feedback_rejects_targets_outside_the_exact_report(monkeypatch) -> None:
    with closing(_feedback_database()) as connection:
        _patch_database(monkeypatch, connection)
        monkeypatch.setattr(
            "romanian_news.feedback.read_daily_report_version", lambda _version: _report()
        )
        missing_group = NewsFeedbackCommand(
            feedback_id=UUID("00000000-0000-4000-8000-000000000004"),
            target=GroupFeedbackTarget(report_version_id=REPORT_V1, group_id=OTHER_ID),
            rating="negative",
        )
        wrong_group_article = NewsFeedbackCommand(
            feedback_id=UUID("00000000-0000-4000-8000-000000000005"),
            target=ArticleFeedbackTarget(
                report_version_id=REPORT_V1,
                group_id=GROUP_ID,
                article_version_id=OTHER_ID,
            ),
            rating="negative",
        )

        with pytest.raises(ValueError, match="group is not in report"):
            submit_news_feedback(missing_group)
        with pytest.raises(ValueError, match="article is not in report group"):
            submit_news_feedback(wrong_group_article)

        assert connection.execute("SELECT count(*) FROM news_feedback").fetchone()[0] == 0


def test_archived_daily_report_version_resolves_to_current_version(monkeypatch) -> None:
    with closing(_feedback_database()) as connection:
        connection.execute(
            "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
            (
                REPORT_V2,
                "news:daily:2026-08-31",
                1,
                "3" * 64,
                None,
                "2026-08-31T10:00:00+00:00",
            ),
        )
        connection.execute(
            "UPDATE artifacts SET current_version_id = ? WHERE id = ?",
            [REPORT_V2, "news:daily:2026-08-31"],
        )
        _patch_database(monkeypatch, connection)

        assert resolve_current_daily_report_version(REPORT_V1) == REPORT_V2


def test_current_daily_report_version_resolves_to_itself(monkeypatch) -> None:
    with closing(_feedback_database()) as connection:
        connection.execute(
            "UPDATE artifacts SET current_version_id = ? WHERE id = ?",
            [REPORT_V1, "news:daily:2026-08-31"],
        )
        _patch_database(monkeypatch, connection)

        assert resolve_current_daily_report_version(REPORT_V1) == REPORT_V1


@pytest.mark.parametrize("report_version_id", [OTHER_ID, ARTICLE_ID, REPORT_V1])
def test_current_daily_report_version_rejects_invalid_versions(
    monkeypatch,
    report_version_id: str,
) -> None:
    with closing(_feedback_database()) as connection:
        _patch_database(monkeypatch, connection)

        with pytest.raises(DailyReportVersionNotFound):
            resolve_current_daily_report_version(report_version_id)


def test_exact_old_report_version_remains_readable_after_head_advances(monkeypatch) -> None:
    report = _archived_report()

    def query(sql, params=None):
        if "artifact.current_version_id" in sql:
            return [
                {
                    "artifact_id": "news:daily:2026-08-31",
                    "report_version_id": REPORT_V2,
                }
            ]
        assert params == [REPORT_V1]
        return [
            {
                "kind": "news_daily_report",
                "version_digest": "f" * 64,
                "file_digest": "f" * 64,
                "r2_key": "news/reports/daily/2026-08-31/old.json",
            }
        ]

    monkeypatch.setattr("romanian_news.catalog.feedback.catalog_query", query)
    monkeypatch.setattr(
        "romanian_news.feedback.read_verified_r2_object",
        lambda _key, _digest: report.model_dump_json().encode(),
    )

    summaries = list_daily_reports()
    old_report = read_daily_report_version(REPORT_V1)

    assert summaries[0].report_version_id == REPORT_V2
    assert old_report == report


def test_exact_report_read_distinguishes_missing_and_corrupt_versions(monkeypatch) -> None:
    monkeypatch.setattr("romanian_news.catalog.feedback.catalog_query", lambda _sql, _params: [])
    with pytest.raises(DailyReportVersionNotFound):
        read_daily_report_version(REPORT_V1)

    monkeypatch.setattr(
        "romanian_news.catalog.feedback.catalog_query",
        lambda _sql, _params: [
            {
                "kind": "news_daily_report",
                "version_digest": "a" * 64,
                "file_digest": "b" * 64,
                "r2_key": "news/reports/daily/2026-08-31/report.json",
            }
        ],
    )
    with pytest.raises(ResearchObjectIntegrityError, match="digests disagree"):
        read_daily_report_version(REPORT_V1)


def test_exact_report_read_rejects_invalid_stored_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        "romanian_news.catalog.feedback.catalog_query",
        lambda _sql, _params: [
            {
                "kind": "news_daily_report",
                "version_digest": "a" * 64,
                "file_digest": "a" * 64,
                "r2_key": "news/reports/daily/2026-08-31/report.json",
            }
        ],
    )
    monkeypatch.setattr(
        "romanian_news.feedback.read_verified_r2_object",
        lambda _key, _digest: b"{}",
    )

    with pytest.raises(ResearchObjectIntegrityError, match="payload is invalid"):
        read_daily_report_version(REPORT_V1)


def test_read_latest_news_feedback_selects_one_event_per_exact_target(monkeypatch) -> None:
    with closing(_feedback_database()) as connection:
        connection.executemany(
            "INSERT INTO news_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "00000000-0000-4000-8000-000000000010",
                    REPORT_V1,
                    "group",
                    None,
                    GROUP_ID,
                    None,
                    "positive",
                    None,
                    "owner",
                    "2026-08-31T09:00:00+00:00",
                ),
                (
                    "00000000-0000-4000-8000-000000000011",
                    REPORT_V1,
                    "group",
                    None,
                    GROUP_ID,
                    None,
                    "negative",
                    "Changed after review.",
                    "owner",
                    "2026-08-31T10:00:00+00:00",
                ),
                (
                    "00000000-0000-4000-8000-000000000012",
                    REPORT_V1,
                    "article",
                    None,
                    GROUP_ID,
                    ARTICLE_ID,
                    "positive",
                    None,
                    "owner",
                    "2026-08-31T09:30:00+00:00",
                ),
            ],
        )
        _patch_database(monkeypatch, connection)

        latest = read_latest_news_feedback(REPORT_V1)

        assert len(latest) == 2
        by_kind = {event.target.kind: event for event in latest}
        assert by_kind["group"].rating == "negative"
        assert by_kind["group"].note == "Changed after review."
        assert by_kind["article"].rating == "positive"


@pytest.mark.parametrize(
    ("rating", "note"),
    [
        ("positive", None),
        ("negative", None),
        (None, "The report needs a source for this claim."),
    ],
)
def test_feedback_command_accepts_a_rating_or_a_note(
    rating: Literal["positive", "negative"] | None,
    note: str | None,
) -> None:
    command = NewsFeedbackCommand(
        feedback_id=UUID("00000000-0000-4000-8000-000000000020"),
        target=ReportFeedbackTarget(report_version_id=REPORT_V1),
        rating=rating,
        note=note,
    )

    assert command.rating == rating
    assert command.note == note


@pytest.mark.parametrize("note", [None, "   "])
def test_feedback_command_rejects_an_event_without_a_rating_or_note(
    note: str | None,
) -> None:
    with pytest.raises(ValueError):
        NewsFeedbackCommand(
            feedback_id=UUID("00000000-0000-4000-8000-000000000021"),
            target=ReportFeedbackTarget(report_version_id=REPORT_V1),
            rating=None,
            note=note,
        )


def test_feedback_models_reject_invalid_commands() -> None:
    with pytest.raises(ValueError):
        NewsFeedbackCommand.model_validate(
            {
                "feedback_id": "not-a-uuid",
                "target": {"kind": "report", "report_version_id": REPORT_V1},
                "rating": "neutral",
                "note": "x" * 2001,
            }
        )


def _feedback_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    schema = Path(__file__).parents[3] / "tests" / "fixtures" / "sqlite_catalog.sql"
    connection.executescript(schema.read_text())
    connection.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "news:daily:2026-08-31",
            "news_daily_report",
            "Daily report",
            "derived",
            "current",
            "private",
            None,
            "2026-08-31T09:00:00+00:00",
        ),
    )
    connection.execute(
        "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
        (REPORT_V1, "news:daily:2026-08-31", 1, "1" * 64, None, "2026-08-31T09:00:00+00:00"),
    )
    connection.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, visibility, current_version_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "news:article:test",
            "news_article",
            "Article",
            "source",
            "current",
            "private",
            None,
            "2026-08-31T09:00:00+00:00",
        ),
    )
    connection.execute(
        "INSERT INTO artifact_versions VALUES (?, ?, ?, ?, ?, ?)",
        (ARTICLE_ID, "news:article:test", 1, "2" * 64, None, "2026-08-31T09:00:00+00:00"),
    )
    return connection


def _patch_database(monkeypatch, connection: sqlite3.Connection) -> None:
    def query(sql, params=None):
        cursor = connection.execute(sql.replace("%s", "?"), params or [])
        return [dict(row) for row in cursor.fetchall()]

    def batch(statements):
        with connection:
            for sql, params in statements:
                connection.execute(sql.replace("%s", "?"), params)

    monkeypatch.setattr("romanian_news.catalog.feedback.catalog_query", query)
    monkeypatch.setattr("romanian_news.catalog.feedback.catalog_batch", batch)


def _report() -> DailyReport:
    event = ReportEvent(
        group_id=GROUP_ID,
        title_ro="Titlu",
        summary_ro="Rezumat",
        key_points_ro=("Punct",),
        disagreements_ro=(),
        uncertainty_ro="Nicio incertitudine",
        sentiment_label="neutral",
        sentiment_score=0.0,
        sentiment_rationale_ro="Relatare neutră",
        articles=(
            ReportArticle(
                article_version_id=ARTICLE_ID,
                outlet_id="test",
                title="Articol",
                canonical_url="https://example.test/article",
                sentiment_label="neutral",
                sentiment_score=0.0,
            ),
        ),
    )
    return DailyReport(
        day=date(2026, 8, 31),
        accepted_article_count=1,
        theme_count=1,
        group_count=1,
        sections=(
            DailyReportSection(
                theme_id=THEME_ID,
                title="Politica bugetară",
                summary="Proiectul de buget și reacțiile formează tema zilei.",
                tier="main",
                semantic_rank=1,
                consequence_rationale="Decizii bugetare cu efect național direct.",
                citations=(
                    ReportSubjectCitation(
                        article_version_id=ARTICLE_ID,
                        evidence_quote="Guvernul a publicat proiectul.",
                    ),
                ),
                events=(event,),
            ),
        ),
    )


def _archived_report() -> ArchivedDailyReport:
    event = _report().sections[0].events[0]
    return ArchivedDailyReport(
        day=date(2026, 8, 31),
        accepted_article_count=1,
        group_count=1,
        sections=(ArchivedDailyReportSection(**event.model_dump()),),
    )
