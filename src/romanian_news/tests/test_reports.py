import json
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from romanian_news import reports as reports_module
from romanian_news import themes as themes_module
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.attempts import ModelCall
from romanian_news.analysis.groups.models import (
    ArticleSentiment,
    GroupSentiment,
    GroupSummary,
    SentimentAssessment,
)
from romanian_news.catalog.reports import _report_run_statements
from romanian_news.groups import DailyClusterSet, NewsGroup
from romanian_news.reports import (
    ArchivedDailyReport,
    ArchivedDailyReportSection,
    DailyReport,
    DailyReportConstruction,
    DailyReportInput,
    DailyReportOutput,
    ReportArticleSource,
    ReportEvent,
    WeeklyReportInput,
    build_daily_report_from_construction,
    build_weekly_report,
)
from romanian_news.subject_assessments import (
    PRODUCTION_SUBJECT_ASSESSMENT_POLICY,
    DailySubjectAssessmentInput,
    DailySubjectAssessmentSet,
    ModelSubjectAssessmentConstruction,
    SubjectAssessment,
    SubjectAssessmentEvidence,
    subject_assessment_policy_digest,
    subject_assessment_request_id,
)
from romanian_news.themes import (
    LEGACY_THEME_POLICY,
    DailyTheme,
    DailyThemeInput,
    DailyThemeSet,
    ModelThemeConstruction,
    ThemeGroupInput,
    ThemeModelAttemptEvidence,
    ThemeModelMessage,
    daily_theme_id,
    daily_theme_request_id,
    theme_policy_digest,
)


@pytest.fixture(autouse=True)
def _isolate_input_cache(monkeypatch):
    """Keep the module-level input cache out of staleness unit tests."""
    monkeypatch.setattr(
        "romanian_news.catalog.artifacts.current_artifact_references",
        lambda _artifact_ids: (),
    )
    reports_module.clear_daily_report_input_cache()
    yield
    reports_module.clear_daily_report_input_cache()


def test_weekly_report_uses_seven_exact_daily_versions(monkeypatch) -> None:
    week_start = date(2026, 8, 24)
    daily = {
        week_start + timedelta(days=offset): (
            _reference(str(offset) * 64),
            DailyReport(
                day=week_start + timedelta(days=offset),
                accepted_article_count=offset + 1,
                theme_count=0,
                group_count=1,
                sections=(),
            ),
        )
        for offset in range(7)
    }
    monkeypatch.setattr("romanian_news.reports._read_daily_report", daily.__getitem__)

    output = build_weekly_report(week_start)
    replay = build_weekly_report(week_start)

    assert output.content == replay.content
    assert output.request_id == replay.request_id
    assert output.report.week_end == date(2026, 8, 30)
    assert output.report.accepted_article_count == 28
    assert output.report.group_count == 7
    assert tuple(day.daily_report_version_id for day in output.report.days) == tuple(
        reference.version_id for reference, _report in daily.values()
    )


def test_weekly_report_reconstructs_a_mixed_current_and_archived_week(monkeypatch) -> None:
    week_start = date(2026, 8, 24)
    daily = {}
    for offset in range(7):
        day = week_start + timedelta(days=offset)
        report = (
            ArchivedDailyReport(
                day=day,
                accepted_article_count=2,
                group_count=1,
                sections=(),
            )
            if offset == 0
            else DailyReport(
                day=day,
                accepted_article_count=1,
                theme_count=0,
                group_count=1,
                sections=(),
            )
        )
        daily[day] = (_reference(f"{offset + 1:064x}"), report)
    monkeypatch.setattr("romanian_news.reports._read_daily_report", daily.__getitem__)

    output = build_weekly_report(week_start)

    assert isinstance(output.report.days[0].report, ArchivedDailyReport)
    assert isinstance(output.report.days[1].report, DailyReport)
    assert output.report.accepted_article_count == 8


def test_daily_report_section_accepts_no_material_uncertainty() -> None:
    section = ReportEvent.model_validate(
        {
            "group_id": "a" * 64,
            "title_ro": "Title",
            "summary_ro": "Summary",
            "key_points_ro": (),
            "disagreements_ro": (),
            "uncertainty_ro": None,
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
            "sentiment_rationale_ro": "Neutral account",
            "articles": (),
        }
    )

    assert section.uncertainty_ro is None


def test_weekly_report_requires_a_monday_and_all_seven_days(monkeypatch) -> None:
    with pytest.raises(ValueError, match="Monday"):
        build_weekly_report(date(2026, 8, 25))

    monkeypatch.setattr(
        "romanian_news.reports._read_daily_report",
        lambda day: (
            _reference("a" * 64),
            DailyReport(
                day=day, accepted_article_count=0, theme_count=0, group_count=0, sections=()
            ),
        )
        if day < date(2026, 8, 27)
        else (_ for _ in ()).throw(ValueError("missing daily report")),
    )

    with pytest.raises(ValueError, match="missing daily report"):
        build_weekly_report(date(2026, 8, 24))


def _reference(version_id: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"news:daily:{version_id[0]}",
        version_id=version_id,
        content_digest="f" * 64,
        r2_key=f"news/reports/{version_id}.json",
    )


def test_daily_report_uses_assessment_order_independent_of_theme_order(
    monkeypatch,
) -> None:
    specs = (
        ("a", "Local traffic restriction", 8, "narrow", "strong", "strong"),
        ("b", "Routine national policy", 1, "routine", "strong", "none"),
        ("f", "Romania loses PNRR funds", 1, "major", "strong", "strong"),
    )

    first = _build_ranked_report(monkeypatch, specs)
    second = _build_ranked_report(monkeypatch, tuple(reversed(specs)))

    expected = [
        "Romania loses PNRR funds",
        "Routine national policy",
        "Local traffic restriction",
    ]
    assert [section.title for section in _fresh_report(first).sections] == expected
    assert [section.title for section in _fresh_report(second).sections] == expected


def test_daily_report_uses_each_assessment_rank_exactly(
    monkeypatch,
) -> None:
    output = _build_ranked_report(
        monkeypatch,
        (
            ("c", "One strong dimension with many articles", 12, "routine", "strong", "none"),
            ("b", "Both strong dimensions with one article", 1, "routine", "strong", "strong"),
            (
                "f",
                "One major member in a small group",
                2,
                ("routine", "major"),
                ("strong", "strong"),
                ("none", "none"),
            ),
            ("a", "One strong dimension with one article", 1, "routine", "strong", "none"),
        ),
    )

    assert [section.title for section in _fresh_report(output).sections] == [
        "One major member in a small group",
        "Both strong dimensions with one article",
        "One strong dimension with many articles",
        "One strong dimension with one article",
    ]


def test_daily_report_keeps_two_event_boundaries_inside_one_theme(monkeypatch) -> None:
    output = _build_ranked_report(
        monkeypatch,
        (
            ("a", "Coalition talks", 1, "major", "strong", "strong"),
            ("b", "Prime minister search", 1, "major", "strong", "strong"),
        ),
        combine_themes=True,
    )

    report = _fresh_report(output)
    assert report.theme_count == 1
    assert report.group_count == 2
    assert [event.group_id for event in report.sections[0].events] == [
        "a" * 64,
        "b" * 64,
    ]
    assert [
        tuple(article.article_version_id for article in event.articles)
        for event in report.sections[0].events
    ] == [(f"{1:064x}",), (f"{2:064x}",)]


def test_daily_report_preserves_prefixed_source_titles(monkeypatch) -> None:
    output = _build_ranked_report(
        monkeypatch,
        (("a", "Romania loses PNRR funds", 1, "major", "strong", "strong"),),
    )

    assert _fresh_report(output).sections[0].events[0].articles[0].title == (
        "BREAKING: LIVE: Source title " + "0" * 63 + "1"
    )


def test_daily_report_records_assessment_in_output_and_request_identity(monkeypatch) -> None:
    output = _build_ranked_report(
        monkeypatch,
        (("a", "Romania loses PNRR funds", 1, "major", "strong", "strong"),),
    )
    without_assessment = reports_module._sha256(
        reports_module._canonical_json(
            {
                "theme_version_id": output.themes.version_id,
                "cluster_set_version_id": output.cluster_set.version_id,
                "operation": "news.publish_daily",
                "report_schema": DailyReport.model_json_schema(),
                "sentiment_version_ids": [value.version_id for value in output.sentiments],
                "summary_version_ids": [value.version_id for value in output.summaries],
            }
        )
    )

    assert output.assessments.artifact_id == "news:subject-assessments"
    assert output.request_id != without_assessment


def test_daily_report_publication_records_assessment_lineage(monkeypatch) -> None:
    output = _build_ranked_report(
        monkeypatch,
        (("a", "Romania loses PNRR funds", 1, "major", "strong", "strong"),),
    )

    statements = _report_run_statements(
        "9" * 64,
        "git:test",
        output,
        status="publishing",
        prior_output=None,
    )

    config = json.loads(str(statements[0][1][4]))
    assert config == {"day": output.report.day.isoformat()}
    inputs = [params for sql, params in statements if "INTO run_inputs" in sql]
    assert sum(params[3] == "assessments" for params in inputs) == 1
    assert all(params[3] != "relevance" for params in inputs)


def test_archived_daily_report_section_keeps_permissive_generated_text() -> None:
    section = ArchivedDailyReportSection.model_validate(
        {
            "group_id": "a" * 64,
            "title_ro": "x" * 101,
            "summary_ro": "x" * 501,
            "key_points_ro": (),
            "disagreements_ro": ("x" * 300,),
            "uncertainty_ro": None,
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
            "sentiment_rationale_ro": "Archived rationale",
            "articles": (),
        }
    )

    assert len(section.title_ro) == 101
    assert len(section.summary_ro) == 501
    assert section.key_points_ro == ()


def test_recorded_daily_report_input_reads_immutable_run_lineage(monkeypatch) -> None:
    day = date(2026, 9, 3)
    roles = ("themes", "assessments", "cluster_set", "summary", "sentiment")
    rows = [
        {
            "parameters_json": {"day": day.isoformat()},
            "position": position,
            "role": role,
            "artifact_id": f"artifact:{role}",
            "version_id": str(position + 1) * 64,
            "content_digest": "a" * 64,
            "r2_key": f"objects/{role}.json",
        }
        for position, role in enumerate(roles)
    ]
    monkeypatch.setattr("romanian_news.catalog.report_inputs.catalog_query", lambda *_args: rows)

    result = reports_module.read_recorded_daily_report_input(day)

    assert result.themes.artifact_id == "artifact:themes"
    assert result.assessments.artifact_id == "artifact:assessments"
    assert result.cluster_set.artifact_id == "artifact:cluster_set"
    assert tuple(value.artifact_id for value in result.summaries) == ("artifact:summary",)
    assert tuple(value.artifact_id for value in result.sentiments) == ("artifact:sentiment",)


def test_stale_daily_report_discovery_includes_the_current_day(monkeypatch) -> None:
    days = tuple(date(2026, 9, value) for value in (1, 3, 4, 5))
    with closing(_report_head_catalog()) as connection:
        for day in (date(2026, 7, 31), *days):
            connection.execute(
                "INSERT INTO artifacts VALUES (?, 'news_clusters', ?, NULL)",
                [f"news:clusters:{day.isoformat()}", day.isoformat()],
            )
        fresh_run = reports_module.report_run_id(
            reports_module.daily_report_request_id(_daily_input(days[1])), "test"
        )
        connection.execute(
            "INSERT INTO artifacts VALUES (?, 'news_daily_report', ?, ?)",
            [f"news:daily:{days[1].isoformat()}", days[1].isoformat(), fresh_run],
        )
        monkeypatch.setattr(
            "romanian_news.catalog.report_inputs.catalog_query", _sqlite_query(connection)
        )
        monkeypatch.setattr(
            reports_module,
            "read_daily_report_input",
            lambda day: (_ for _ in ()).throw(reports_module.ReportInputsUnavailable("unready"))
            if day == days[2]
            else _daily_input(day),
        )

        early_morning = reports_module.stale_daily_report_days(
            reports_module.datetime.fromisoformat("2026-09-05T06:40:00+03:00"),
            "test",
            days[0],
        )
        later = reports_module.stale_daily_report_days(
            reports_module.datetime.fromisoformat("2026-09-05T09:30:00+03:00"),
            "test",
            days[0],
        )

    assert early_morning == (days[0], days[3])
    assert later == (days[0], days[3])


def test_stale_daily_report_discovery_skips_a_day_without_themes(monkeypatch) -> None:
    from romanian_news.catalog.themes import DailyThemeUnavailable

    day = date(2026, 9, 4)
    with closing(_report_head_catalog()) as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?, 'news_clusters', ?, NULL)",
            [f"news:clusters:{day.isoformat()}", day.isoformat()],
        )
        monkeypatch.setattr(
            "romanian_news.catalog.report_inputs.catalog_query", _sqlite_query(connection)
        )
        monkeypatch.setattr(
            "romanian_news.catalog.themes.read_daily_theme_reference",
            lambda _day: (_ for _ in ()).throw(DailyThemeUnavailable("not ready")),
        )

        assert (
            reports_module.stale_daily_report_days(
                reports_module.datetime.fromisoformat("2026-09-05T10:30:00+03:00"),
                "test",
                day,
            )
            == ()
        )


def test_stale_daily_report_discovery_returns_no_fresh_days(monkeypatch) -> None:
    day = date(2026, 9, 4)
    current_run = reports_module.report_run_id(
        reports_module.daily_report_request_id(_daily_input(day)), "test"
    )
    with closing(_report_head_catalog()) as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?, 'news_clusters', ?, NULL)",
            [f"news:clusters:{day.isoformat()}", day.isoformat()],
        )
        connection.execute(
            "INSERT INTO artifacts VALUES (?, 'news_daily_report', ?, ?)",
            [f"news:daily:{day.isoformat()}", day.isoformat(), current_run],
        )
        monkeypatch.setattr(
            "romanian_news.catalog.report_inputs.catalog_query", _sqlite_query(connection)
        )
        monkeypatch.setattr(reports_module, "read_daily_report_input", _daily_input)

        assert (
            reports_module.stale_daily_report_days(
                reports_module.datetime.fromisoformat("2026-09-05T10:30:00+03:00"),
                "test",
                day,
            )
            == ()
        )


def test_stale_weekly_report_discovery_returns_complete_stale_weeks(
    monkeypatch,
) -> None:
    first_week = date(2026, 8, 31)
    second_week = date(2026, 9, 7)
    with closing(_report_head_catalog()) as connection:
        for offset in range(-7, 14):
            day = first_week + timedelta(days=offset)
            connection.execute(
                "INSERT INTO artifacts VALUES (?, 'news_daily_report', ?, 'daily-run')",
                [f"news:daily:{day.isoformat()}", day.isoformat()],
            )
        first_input = _weekly_input(first_week)
        first_run = reports_module.report_run_id(
            reports_module.weekly_report_request_id(first_input), "test"
        )
        connection.execute(
            "INSERT INTO artifacts VALUES (?, 'news_weekly_report', ?, ?)",
            [f"news:weekly:{first_week.isoformat()}", first_week.isoformat(), first_run],
        )
        monkeypatch.setattr(
            "romanian_news.catalog.report_inputs.catalog_query", _sqlite_query(connection)
        )
        monkeypatch.setattr(reports_module, "read_weekly_report_input", _weekly_input)

        stale = reports_module.stale_weekly_report_weeks(
            reports_module.datetime.fromisoformat("2026-09-21T08:00:00+03:00"),
            "test",
            first_week,
        )

    assert stale == (second_week,)


def test_stale_daily_report_discovery_skips_unreadable_ready_input(monkeypatch) -> None:
    day = date(2026, 9, 4)
    with closing(_report_head_catalog()) as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?, 'news_clusters', ?, 'run-1')",
            [f"news:clusters:{day.isoformat()}", day.isoformat()],
        )
        monkeypatch.setattr(
            "romanian_news.catalog.report_inputs.catalog_query", _sqlite_query(connection)
        )

        def _read(day):
            raise ValueError("invalid provenance")

        monkeypatch.setattr(reports_module, "read_daily_report_input", _read)
        monkeypatch.setattr(
            reports_module,
            "daily_report_request_id",
            lambda _value: "request-1",
        )
        monkeypatch.setattr(
            reports_module,
            "report_run_id",
            lambda _request_id, _implementation: "run-1",
        )

        assert (
            reports_module.stale_daily_report_days(
                reports_module.datetime.fromisoformat("2026-09-05T10:30:00+03:00"),
                "test",
                day,
            )
            == ()
        )


def _report_head_catalog() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE artifacts (id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
        "current_version_id TEXT, current_run_id TEXT)"
    )
    return connection


def _sqlite_query(connection: sqlite3.Connection):
    def query(sql, params=None):
        return [
            dict(row) for row in connection.execute(sql.replace("%s", "?"), params or []).fetchall()
        ]

    return query


def _daily_input(day: date) -> DailyReportInput:
    return DailyReportInput(
        day=day,
        themes=_reference("9" * 64),
        assessments=_reference("8" * 64),
        cluster_set=_reference(f"{day.day:064x}"),
        summaries=(),
        sentiments=(),
    )


def _weekly_input(week_start: date) -> WeeklyReportInput:
    return WeeklyReportInput(
        week_start=week_start,
        policy="test",
        daily_reports=tuple(
            _reference(f"{(week_start + timedelta(days=offset)).toordinal():064x}")
            for offset in range(7)
        ),
    )


def _fresh_report(output: DailyReportOutput) -> DailyReport:
    report = output.report
    if not isinstance(report, DailyReport):
        raise AssertionError("Expected a freshly constructed daily report")
    return report


def _ranked_report_members(specs):
    groups = []
    article_ids = []
    article_metadata = {}
    group_ranks = {}
    summary_references = []
    summary_values = {}
    sentiment_references = []
    sentiment_values = {}
    themes = []
    next_article = 1
    for index, (
        group_character,
        title,
        article_count,
        magnitude,
        _political,
        _economic,
    ) in enumerate(specs):
        member_ids = tuple(
            f"{position:064x}" for position in range(next_article, next_article + article_count)
        )
        next_article += article_count
        group = NewsGroup(id=group_character * 64, article_version_ids=member_ids)
        groups.append(group)
        article_ids.extend(member_ids)
        magnitude_values = magnitude if isinstance(magnitude, tuple) else (magnitude,)
        group_ranks[group.id] = -max(
            {"narrow": 0, "routine": 1, "major": 2}[item] for item in magnitude_values
        )
        for _member_index, article_id in enumerate(member_ids):
            article_metadata[article_id] = ReportArticleSource(
                outlet_id="test",
                title=f"BREAKING: LIVE: Source title {article_id}",
                canonical_url=f"https://example.test/{article_id}",
            )
        summary_reference = _analysis_reference(f"summary-{index}", f"{index + 10:064x}")
        summary_references.append(summary_reference)
        summary_values[summary_reference.artifact_id] = GroupSummary(
            title_ro=title,
            summary_ro=f"Summary for {title}.",
            key_points_ro=(f"Key point for {title}.",),
            disagreements_ro=(),
            uncertainty_ro=None,
            cited_article_version_ids=member_ids,
        )
        sentiment_reference = _analysis_reference(f"sentiment-{index}", f"{index + 20:064x}")
        sentiment_references.append(sentiment_reference)
        sentiment_values[sentiment_reference.artifact_id] = GroupSentiment(
            overall=SentimentAssessment(
                label="neutral", score=0.0, confidence=0.9, rationale_ro="Neutral"
            ),
            articles=tuple(
                ArticleSentiment(
                    article_version_id=article_id,
                    label="neutral",
                    score=0.0,
                    confidence=0.9,
                    rationale_ro="Neutral",
                    evidence_quote="Evidence",
                )
                for article_id in member_ids
            ),
        )
        themes.append(
            DailyTheme(
                id=f"{index + 40:064x}",
                title=title,
                summary=f"Theme summary for {title}.",
                group_ids=(group.id,),
                article_version_ids=member_ids,
            )
        )
    return (
        groups,
        article_ids,
        article_metadata,
        group_ranks,
        summary_references,
        summary_values,
        sentiment_references,
        sentiment_values,
        themes,
    )


def _ranked_assessment_set(
    day,
    themes,
    groups,
    summary_references,
    relevance_by_article,
    article_group,
    group_ranks,
    theme_reference,
    theme_set,
):
    summary_by_group = dict(zip((group.id for group in groups), summary_references, strict=True))
    ordered_themes = sorted(
        themes,
        key=lambda theme: (
            min(group_ranks[group_id] for group_id in theme.group_ids),
            theme.title,
        ),
    )
    assessments = tuple(
        SubjectAssessment(
            theme_id=theme.id,
            group_ids=theme.group_ids,
            article_version_ids=theme.article_version_ids,
            tier="main",
            semantic_rank=rank,
            rationale=f"Consequence rationale for {theme.title}.",
            evidence=(
                SubjectAssessmentEvidence(
                    group_id=article_group[theme.article_version_ids[0]],
                    article=_analysis_reference(
                        f"article-{theme.article_version_ids[0]}",
                        theme.article_version_ids[0],
                    ),
                    relevance=relevance_by_article[theme.article_version_ids[0]],
                    summary=summary_by_group[article_group[theme.article_version_ids[0]]],
                    evidence_quote="Evidence",
                ),
            ),
        )
        for rank, theme in enumerate(ordered_themes, start=1)
    )
    assessment_reference = _analysis_reference("subject-assessments", "8" * 64)
    assessment_set = DailySubjectAssessmentSet.model_construct(
        day=day,
        request_id=subject_assessment_request_id(
            DailySubjectAssessmentInput.model_construct(
                day=day,
                themes=theme_reference,
                theme_set=theme_set,
                summaries=tuple(
                    SimpleNamespace(reference=reference) for reference in summary_references
                ),
                relevance=tuple(relevance_by_article.values()),
            )
        ),
        policy=PRODUCTION_SUBJECT_ASSESSMENT_POLICY,
        policy_digest=subject_assessment_policy_digest(PRODUCTION_SUBJECT_ASSESSMENT_POLICY),
        themes=theme_reference,
        summary_inputs=tuple(summary_references),
        relevance_inputs=tuple(relevance_by_article.values()),
        construction=ModelSubjectAssessmentConstruction.model_construct(),
        subject_ids=tuple(theme.id for theme in themes),
        assessments=assessments,
    )
    return assessment_reference, assessment_set


def _build_ranked_report(monkeypatch, specs, *, combine_themes: bool = False):
    day = date(2026, 8, 31)
    (
        groups,
        article_ids,
        article_metadata,
        group_ranks,
        summary_references,
        summary_values,
        sentiment_references,
        sentiment_values,
        themes,
    ) = _ranked_report_members(specs)
    if combine_themes:
        themes = [
            DailyTheme(
                id="4" * 64,
                title="Shared theme",
                summary="The events belong to one theme.",
                group_ids=tuple(group.id for group in groups),
                article_version_ids=tuple(article_ids),
            )
        ]
    cluster_reference = _analysis_reference("clusters", "c" * 64)
    theme_reference = _analysis_reference("themes", "9" * 64)
    cluster_set = DailyClusterSet(
        day=day,
        algorithm="complete-link-cosine-v1",
        threshold=0.78,
        embedding_model="test/model",
        article_version_ids=tuple(article_ids),
        relevance_version_ids=tuple(f"{index + 100:064x}" for index in range(len(article_ids))),
        embedding_version_ids=tuple(f"{index + 200:064x}" for index in range(len(article_ids))),
        merges=(),
        groups=tuple(groups),
    )
    policy_digest = theme_policy_digest(LEGACY_THEME_POLICY)
    themes = [
        theme.model_copy(update={"id": daily_theme_id(day, theme.group_ids, policy_digest)})
        for theme in themes
    ]
    theme_input = DailyThemeInput(
        day=day,
        cluster_set=cluster_reference,
        groups=tuple(
            ThemeGroupInput(
                group=group,
                summary=summary_reference,
                value=summary_values[summary_reference.artifact_id],
            )
            for group, summary_reference in zip(groups, summary_references, strict=True)
        ),
    )
    theme_set = DailyThemeSet(
        day=day,
        request_id=daily_theme_request_id(theme_input, LEGACY_THEME_POLICY),
        policy=LEGACY_THEME_POLICY,
        policy_digest=policy_digest,
        cluster_set=cluster_reference,
        groups=tuple(groups),
        summary_inputs=tuple(summary_references),
        construction=_theme_construction(tuple(themes)),
        themes=tuple(themes),
    )
    relevance_references = tuple(
        _analysis_reference(f"relevance-{index}", version_id)
        for index, version_id in enumerate(cluster_set.relevance_version_ids)
    )
    relevance_by_article = dict(zip(article_ids, relevance_references, strict=True))
    article_group = {
        article_id: group.id for group in groups for article_id in group.article_version_ids
    }
    assessment_reference, assessment_set = _ranked_assessment_set(
        day,
        themes,
        groups,
        summary_references,
        relevance_by_article,
        article_group,
        group_ranks,
        theme_reference,
        theme_set,
    )
    inputs = DailyReportInput(
        day=day,
        themes=theme_reference,
        assessments=assessment_reference,
        cluster_set=cluster_reference,
        summaries=tuple(summary_references),
        sentiments=tuple(sentiment_references),
    )
    return build_daily_report_from_construction(
        DailyReportConstruction(
            inputs=inputs,
            theme_set=theme_set,
            assessment_set=assessment_set,
            cluster_set=cluster_set,
            articles=article_metadata,
            summaries=summary_values,
            sentiments=sentiment_values,
        )
    )


def _theme_construction(themes: tuple[DailyTheme, ...]) -> ModelThemeConstruction:
    messages = (
        ThemeModelMessage(role="system", content="System"),
        ThemeModelMessage(role="user", content="Input"),
    )
    response_content = json.dumps(
        {
            "themes": [
                {
                    "title": theme.title,
                    "summary": theme.summary,
                    "group_ids": theme.group_ids,
                }
                for theme in themes
            ]
        },
        ensure_ascii=False,
    )
    return ModelThemeConstruction(
        messages=messages,
        input_digest=themes_module._sha256(
            themes_module._canonical_json([message.model_dump(mode="json") for message in messages])
        ),
        response_schema_digest=themes_module._response_schema_digest(),
        call=ModelCall(
            response_id="response",
            model="test/model",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        ),
        attempts=(
            ThemeModelAttemptEvidence(
                attempt_id="6" * 64,
                response_id="response",
                status="accepted",
                error=None,
                response_content=response_content,
                response_content_digest=themes_module._sha256(response_content.encode()),
                provider_response={
                    "id": "response",
                    "choices": [{"message": {"content": response_content}}],
                },
            ),
        ),
    )


def _analysis_reference(name: str, version_id: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"news:{name}",
        version_id=version_id,
        content_digest="f" * 64,
        r2_key=f"news/{name}.json",
    )


def _member_value(value: object, index: int) -> object:
    return value[index] if isinstance(value, tuple) else value
