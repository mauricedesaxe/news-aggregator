from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime

import pytest
from pydantic import HttpUrl

from romanian_news.analysis.attempts import ModelCall
from romanian_news.analysis.groups.models import (
    ArticleSentiment,
    GroupSentiment,
    GroupSummary,
    SentimentAssessment,
)
from romanian_news.analysis.groups.sentiment import sentiment_request_id
from romanian_news.analysis.groups.summary import summary_request_id
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file, artifact_statements
from romanian_news.groups import DailyClusterSet, NewsGroup
from romanian_news.identity import canonical_json, sha256
from romanian_news.reports import RetrospectiveDailyReport, parse_daily_report
from romanian_news.subject_assessments import (
    DailySubjectAssessmentSet,
    ModelSubjectAssessmentConstruction,
    SubjectAssessment,
    SubjectAssessmentAttemptEvidence,
    SubjectAssessmentEvidence,
    SubjectAssessmentMessage,
    SubjectAssessmentPolicy,
    _request_identity,
    subject_assessment_policy_digest,
)
from romanian_news.themes import (
    DailyTheme,
    DailyThemeSet,
    ModelThemeConstruction,
    ThemeModelAttemptEvidence,
    ThemeModelMessage,
    ThemePolicy,
    _daily_theme_request_identity,
    _response_schema_digest,
    daily_theme_id,
    theme_policy_digest,
)
from romanian_news.worker import retrospective_analysis
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog
from tests.worker.conftest import FakeR2Client

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)

DAY = date(2026, 8, 31)
CAPTURED_AT = "2026-08-31T06:00:00+00:00"
PUBLISHED_AT = datetime(2026, 8, 31, 10, tzinfo=UTC)
FETCHED_AT = datetime(2026, 8, 31, 12, tzinfo=UTC)
THEME_TITLE = "Regional transport investment"


def _seed_input_artifact(
    catalog: PostgresCatalog, fake_r2: FakeR2Client, file: ArtifactFile
) -> ArtifactReference:
    statements = artifact_statements(file, CAPTURED_AT, None)
    statements.append(
        (
            "UPDATE artifacts SET current_version_id = %s WHERE id = %s",
            [file.version_id, file.artifact_id],
        )
    )
    catalog.batch(statements)
    fake_r2.objects[file.r2_key] = file.content
    return ArtifactReference(
        artifact_id=file.artifact_id,
        version_id=file.version_id,
        content_digest=file.content_digest,
        r2_key=file.r2_key,
    )


def _seed_foreign_version(catalog: PostgresCatalog, artifact_id: str, version_id: str) -> None:
    catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
        "visibility, created_at) VALUES (%s, 'news_input', 'Seeded input', 'derived', "
        "'current', 'private', %s) ON CONFLICT DO NOTHING",
        (artifact_id, CAPTURED_AT),
    )
    catalog.execute(
        "INSERT INTO artifact_versions VALUES (%s, %s, 1, %s, NULL, %s) ON CONFLICT DO NOTHING",
        (version_id, artifact_id, sha256(artifact_id.encode()), CAPTURED_AT),
    )


def _seed_article(catalog: PostgresCatalog, fake_r2: FakeR2Client, day: date) -> ArtifactReference:
    article = ExtractedArticle(
        article_id=sha256(b"retro-article"),
        outlet_id="presa-exemplu",
        canonical_url=HttpUrl("https://example.com/analiza"),
        title="Analiza proiectului de transport",
        body="Instituția a publicat detalii despre proiectul local. " * 6,
        author=None,
        published_at=PUBLISHED_AT,
        source_updated_at=None,
        bucharest_day=day,
        material_digest=sha256(b"retro-material"),
        extraction_digest=sha256(b"retro-extraction"),
    )
    file = artifact_file(
        artifact_id=f"news:article:{article.article_id}",
        artifact_kind="news_article",
        title=article.title,
        content=canonical_json(article.model_dump(mode="json")),
        r2_key=f"news/articles/{article.article_id}/{sha256(canonical_json(article.model_dump(mode='json')))}.json",
        media_type="application/json",
    )
    reference = _seed_input_artifact(catalog, fake_r2, file)
    feed_snapshot_version = sha256(b"retro-feed-snapshot")
    _seed_foreign_version(
        catalog, f"news:feed-snapshot:{feed_snapshot_version}", feed_snapshot_version
    )
    catalog.execute(
        "INSERT INTO news_article_versions "
        "(artifact_version_id, article_artifact_id, outlet_id, canonical_url, "
        "published_at, source_updated_at, bucharest_day, material_digest, "
        "extraction_digest, feed_snapshot_version_id, page_capture_version_id, "
        "captured_at, archive_capture_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, %s, NULL)",
        (
            reference.version_id,
            reference.artifact_id,
            article.outlet_id,
            str(article.canonical_url),
            PUBLISHED_AT,
            None,
            day,
            article.material_digest,
            article.extraction_digest,
            feed_snapshot_version,
            FETCHED_AT,
        ),
    )
    return reference


def _seed_cluster_set(
    catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    day: date,
    group: NewsGroup,
    article_version_id: str,
) -> ArtifactReference:
    cluster_set = DailyClusterSet(
        day=day,
        algorithm="synthetic-test",
        threshold=0.5,
        embedding_model="synthetic-test",
        article_version_ids=(article_version_id,),
        relevance_version_ids=(sha256(b"retro-relevance"),),
        embedding_version_ids=(sha256(b"retro-embedding"),),
        merges=(),
        groups=(group,),
    )
    content = canonical_json(cluster_set.model_dump(mode="json"))
    file = artifact_file(
        artifact_id=f"news:clusters:{day.isoformat()}",
        artifact_kind="news_clusters",
        title=f"Cluster set {day.isoformat()}",
        content=content,
        r2_key=f"news/clusters/{day.isoformat()}/{sha256(content)}.json",
        media_type="application/json",
    )
    return _seed_input_artifact(catalog, fake_r2, file)


def _seed_group_summary(
    catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    group: NewsGroup,
    article_version_id: str,
) -> ArtifactReference:
    summary = GroupSummary(
        title_ro="Investiția regională de transport",
        summary_ro="Operatorul de transport propune materiale noi pentru linia regională.",
        key_points_ro=("Modernizarea depoului rămâne centrală.",),
        disagreements_ro=(),
        cited_article_version_ids=(article_version_id,),
    )
    content = canonical_json({"group_id": group.id, "summary": summary.model_dump(mode="json")})
    file = artifact_file(
        artifact_id=f"news:summary:{summary_request_id(group)}",
        artifact_kind="news_group_summary",
        title="Group summary",
        content=content,
        r2_key=f"news/derived/summaries/{group.id}/{sha256(content)}.json",
        media_type="application/json",
    )
    return _seed_input_artifact(catalog, fake_r2, file)


def _seed_group_sentiment(
    catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    group: NewsGroup,
    article_version_id: str,
) -> ArtifactReference:
    sentiment = GroupSentiment(
        overall=SentimentAssessment(
            label="neutral",
            score=0.0,
            confidence=0.8,
            rationale_ro="Tonul combină prudența cu așteptări moderate.",
        ),
        articles=(
            ArticleSentiment(
                article_version_id=article_version_id,
                label="neutral",
                score=0.0,
                confidence=0.8,
                rationale_ro="Articolul descrie fapte fără evaluare explicită.",
                evidence_quote="Instituția a publicat detalii despre proiectul local.",
            ),
        ),
    )
    content = canonical_json({"group_id": group.id, "sentiment": sentiment.model_dump(mode="json")})
    file = artifact_file(
        artifact_id=f"news:sentiment:{sentiment_request_id(group)}",
        artifact_kind="news_sentiment",
        title="Group sentiment",
        content=content,
        r2_key=f"news/derived/sentiments/{group.id}/{sha256(content)}.json",
        media_type="application/json",
    )
    return _seed_input_artifact(catalog, fake_r2, file)


def _theme_policy() -> ThemePolicy:
    return ThemePolicy(
        policy_id="daily-reader-themes-v1",
        model="recorded/model",
        prompt_digest=sha256(b"retro-theme-prompt"),
        input_policy="ordered-group-summaries-and-article-ids-v1",
        assignment_policy="exact-partition-ordered-v1",
        correction_policy="complete-json-once-v1",
        temperature=0,
        max_tokens=4000,
    )


def _seed_theme_set(
    catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    day: date,
    group: NewsGroup,
    article_version_id: str,
    cluster_reference: ArtifactReference,
    summary_reference: ArtifactReference,
) -> tuple[ArtifactReference, str]:
    policy = _theme_policy()
    policy_digest = theme_policy_digest(policy)
    theme = DailyTheme(
        id=daily_theme_id(day, (group.id,), policy_digest),
        title=THEME_TITLE,
        summary="A synthetic transport proposal frames the publication day.",
        group_ids=(group.id,),
        article_version_ids=(article_version_id,),
    )
    messages = (
        ThemeModelMessage(role="system", content="Recorded theme request."),
        ThemeModelMessage(role="user", content="Recorded complete day input."),
    )
    response_content = canonical_json({"themes": [theme.model_dump(mode="json")]}).decode()
    construction = ModelThemeConstruction(
        messages=messages,
        input_digest=sha256(
            canonical_json([message.model_dump(mode="json") for message in messages])
        ),
        response_schema_digest=_response_schema_digest(),
        call=ModelCall(
            response_id="retro-theme-response",
            model="recorded/model",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        ),
        attempts=(
            ThemeModelAttemptEvidence(
                attempt_id=sha256(b"retro-theme-attempt"),
                response_id="retro-theme-response",
                status="accepted",
                error=None,
                response_content=response_content,
                response_content_digest=sha256(response_content.encode()),
                provider_response={
                    "id": "retro-theme-response",
                    "choices": [{"message": {"content": response_content}}],
                },
            ),
        ),
    )
    theme_set = DailyThemeSet(
        day=day,
        request_id=_daily_theme_request_identity(
            day, cluster_reference, ((group, summary_reference),), policy
        ),
        policy=policy,
        policy_digest=policy_digest,
        cluster_set=cluster_reference,
        groups=(group,),
        summary_inputs=(summary_reference,),
        construction=construction,
        themes=(theme,),
    )
    content = canonical_json(theme_set.model_dump(mode="json"))
    file = artifact_file(
        artifact_id=f"news:themes:{day.isoformat()}",
        artifact_kind="news_daily_themes",
        title=f"Daily themes {day.isoformat()}",
        content=content,
        r2_key=f"news/themes/{day.isoformat()}/{sha256(content)}.json",
        media_type="application/json",
    )
    return _seed_input_artifact(catalog, fake_r2, file), theme.id


def _seed_assessment_set(
    catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    day: date,
    group: NewsGroup,
    article_reference: ArtifactReference,
    theme_reference: ArtifactReference,
    summary_reference: ArtifactReference,
    theme_id: str,
) -> ArtifactReference:
    policy = SubjectAssessmentPolicy(
        policy_id="daily-subject-consequence-v1",
        model="google/gemini-3.8-flash",
        prompt_digest=sha256(b"retro-assessment-prompt"),
        input_policy="ordered-schema-v3-themes-summaries-relevance-v1",
        tier_policy="ordered-three-tier-complete-partition-v1",
        correction_policy="complete-json-once-v1",
        reasoning_effort="low",
        temperature=0,
        max_tokens=4000,
    )
    relevance_reference = ArtifactReference(
        artifact_id=f"news:relevance:{sha256(b'retro-relevance')}",
        version_id=sha256(b"retro-relevance"),
        content_digest=sha256(b"retro-relevance-digest"),
        r2_key="news/relevance/retro-seed.json",
    )
    assessment = SubjectAssessment(
        theme_id=theme_id,
        group_ids=(group.id,),
        article_version_ids=(article_reference.version_id,),
        tier="main",
        semantic_rank=1,
        rationale="Deciziile de transport cu efecte regionale directe.",
        evidence=(
            SubjectAssessmentEvidence(
                group_id=group.id,
                article=article_reference,
                relevance=relevance_reference,
                summary=summary_reference,
                evidence_quote="Instituția a publicat detalii despre proiectul local.",
            ),
        ),
    )
    construction = ModelSubjectAssessmentConstruction(
        request_id=sha256(b"retro-assessment-model-request"),
        messages=(
            SubjectAssessmentMessage(role="system", content="Recorded assessment request."),
            SubjectAssessmentMessage(role="user", content="Recorded assessment day input."),
        ),
        input_digest=sha256(b"retro-assessment-input"),
        response_schema_digest=sha256(b"retro-assessment-schema"),
        call=ModelCall(
            response_id="retro-assessment-response",
            model="google/gemini-3.8-flash",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        ),
        attempts=(
            SubjectAssessmentAttemptEvidence(
                attempt_id=sha256(b"retro-assessment-attempt"),
                response_id="retro-assessment-response",
                status="accepted",
                error=None,
                response_content="{}",
                response_content_digest=sha256(b"{}"),
                provider_response={},
            ),
        ),
    )
    assessment_set = DailySubjectAssessmentSet(
        day=day,
        request_id=_request_identity(
            day, theme_reference, (summary_reference,), (relevance_reference,), policy
        ),
        policy=policy,
        policy_digest=subject_assessment_policy_digest(policy),
        themes=theme_reference,
        summary_inputs=(summary_reference,),
        relevance_inputs=(relevance_reference,),
        construction=construction,
        subject_ids=(theme_id,),
        assessments=(assessment,),
    )
    content = canonical_json(assessment_set.model_dump(mode="json"))
    file = artifact_file(
        artifact_id=f"news:subject-assessments:{day.isoformat()}",
        artifact_kind="news_daily_subject_assessments",
        title=f"Daily subject assessments {day.isoformat()}",
        content=content,
        r2_key=f"news/subject-assessments/{day.isoformat()}/{sha256(content)}.json",
        media_type="application/json",
    )
    return _seed_input_artifact(catalog, fake_r2, file)


def _seed_archive_coverage(catalog: PostgresCatalog, day: date) -> None:
    for outlet in ("digi24", "hotnews"):
        observation_id = sha256(f"retro-observation-{outlet}".encode())
        url = f"https://{outlet}.ro/coverage-story"
        catalog.batch(
            [
                (
                    "INSERT INTO news_archive_sitemap_observations VALUES "
                    "(%s, %s, %s, %s, %s, %s)",
                    [
                        observation_id,
                        outlet,
                        f"https://{outlet}.ro/sitemap.xml",
                        sha256(f"retro-sitemap-{outlet}".encode()),
                        FETCHED_AT,
                        1,
                    ],
                ),
                (
                    "INSERT INTO news_archive_sitemap_entries VALUES (%s, %s, %s)",
                    [observation_id, url, url],
                ),
                (
                    "INSERT INTO news_archive_page_checks "
                    "(id, observation_id, outlet_id, canonical_url, final_url, fetched_at, "
                    "page_sha256, title, published_at, status) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    [
                        sha256(f"retro-check-{outlet}".encode()),
                        observation_id,
                        outlet,
                        url,
                        url,
                        FETCHED_AT,
                        sha256(f"retro-page-{outlet}".encode()),
                        "Coverage story",
                        PUBLISHED_AT,
                        "accepted",
                    ],
                ),
            ]
        )
        capture_version = sha256(f"retro-capture-{outlet}".encode())
        _seed_foreign_version(catalog, f"news:archive-capture:seed-{outlet}", capture_version)
        catalog.execute(
            "INSERT INTO news_archive_article_captures "
            "(id, observation_id, discovered_url, final_url, capture_artifact_version_id, "
            "page_sha256, fetched_at, published_at, modified_at, publication_evidence) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                sha256(f"retro-capture-row-{outlet}".encode()),
                observation_id,
                url,
                url,
                capture_version,
                sha256(f"retro-page-{outlet}".encode()),
                FETCHED_AT,
                PUBLISHED_AT,
                None,
                "article:published_time",
            ),
        )


def _seed_retrospective_inputs(catalog: PostgresCatalog, fake_r2: FakeR2Client, day: date) -> None:
    article_reference = _seed_article(catalog, fake_r2, day)
    group = NewsGroup(
        id=sha256(b"retro-group"), article_version_ids=(article_reference.version_id,)
    )
    cluster_reference = _seed_cluster_set(
        catalog, fake_r2, day, group, article_reference.version_id
    )
    summary_reference = _seed_group_summary(catalog, fake_r2, group, article_reference.version_id)
    _seed_group_sentiment(catalog, fake_r2, group, article_reference.version_id)
    theme_reference, theme_id = _seed_theme_set(
        catalog,
        fake_r2,
        day,
        group,
        article_reference.version_id,
        cluster_reference,
        summary_reference,
    )
    _seed_assessment_set(
        catalog,
        fake_r2,
        day,
        group,
        article_reference,
        theme_reference,
        summary_reference,
        theme_id,
    )
    _seed_archive_coverage(catalog, day)


def _stub_model_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "materialize_relevance",
        "materialize_embeddings",
        "materialize_clusters",
        "materialize_group_summaries",
        "materialize_group_sentiment",
        "materialize_daily_themes",
        "materialize_subject_assessments",
    ):
        monkeypatch.setattr(retrospective_analysis, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        retrospective_analysis, "read_pending_relevance_references", lambda **_kwargs: ()
    )


def test_retrospective_pilot_publishes_a_real_daily_report_end_to_end(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_retrospective_inputs(postgres_catalog, fake_r2, DAY)
    _stub_model_layers(monkeypatch)

    result = retrospective_analysis.retrospective_analysis_pilot.execute_in_process(
        run_config={"ops": {"retrospective_analysis": {"config": {"day": DAY.isoformat()}}}}
    )

    assert result.success
    head = postgres_catalog.execute(
        "SELECT current_version_id, current_run_id FROM artifacts "
        "WHERE id = %s AND kind = 'news_daily_report'",
        (f"news:daily:{DAY.isoformat()}",),
    ).fetchone()
    assert head is not None
    version_id = head["current_version_id"]
    run_rows = postgres_catalog.execute(
        "SELECT status FROM runs WHERE operation_key = 'news.publish_daily'"
    ).fetchall()
    assert [row["status"] for row in run_rows] == ["completed"]
    assert head["current_run_id"] is not None
    file_row = postgres_catalog.execute(
        "SELECT r2_key, content_digest FROM artifact_files WHERE artifact_version_id = %s",
        (version_id,),
    ).fetchone()
    assert file_row is not None
    stored = fake_r2.objects[file_row["r2_key"]]
    assert hashlib.sha256(stored).hexdigest() == file_row["content_digest"]

    report = parse_daily_report(stored)

    assert isinstance(report, RetrospectiveDailyReport)
    assert report.day == DAY
    assert report.accepted_article_count == 1
    assert report.sections[0].title == THEME_TITLE
    assert report.retrospective.included_outlets == ("digi24", "hotnews")
    assert report.retrospective.captured_article_count == 2
    assert report.retrospective.verified_page_count == 2
    assert report.retrospective.capture_started_at == FETCHED_AT
    assert report.retrospective.capture_ended_at == FETCHED_AT


def test_second_analysis_with_unchanged_coverage_reuses_the_published_version(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_retrospective_inputs(postgres_catalog, fake_r2, DAY)
    _stub_model_layers(monkeypatch)

    first = retrospective_analysis.analyze_retrospective_day(DAY, "git:retro-e2e")
    second = retrospective_analysis.analyze_retrospective_day(DAY, "git:retro-e2e")

    assert first is not None
    assert second == first
    head = postgres_catalog.execute(
        "SELECT current_version_id FROM artifacts " "WHERE id = %s AND kind = 'news_daily_report'",
        (f"news:daily:{DAY.isoformat()}",),
    ).fetchone()
    assert head is not None
    assert head["current_version_id"] == first
    version_count = postgres_catalog.execute(
        "SELECT count(*) AS count FROM artifact_versions version "
        "JOIN artifacts artifact ON artifact.id = version.artifact_id "
        "WHERE artifact.id = %s",
        (f"news:daily:{DAY.isoformat()}",),
    ).fetchone()
    assert version_count is not None
    assert version_count["count"] == 1
    run_rows = postgres_catalog.execute(
        "SELECT status FROM runs WHERE operation_key = 'news.publish_daily'"
    ).fetchall()
    assert [row["status"] for row in run_rows] == ["completed"]
    published_keys = sorted(
        key for key in fake_r2.objects if key.startswith(f"news/reports/daily/{DAY.isoformat()}/")
    )
    assert len(published_keys) == 1
