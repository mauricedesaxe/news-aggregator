from __future__ import annotations

import io
from datetime import date

import pytest
from botocore.exceptions import ClientError

from romanian_news import storage
from romanian_news.analysis.groups.sentiment import sentiment_request_id
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file
from romanian_news.groups import DailyClusterSet, NewsGroup
from romanian_news.identity import sha256
from romanian_news.reports import ReportInputsUnavailable, read_daily_report_input
from romanian_news.subject_assessments import (
    DailySubjectAssessmentInput,
    construct_daily_subject_assessments,
)
from romanian_news.themes import DailyThemeInput, SparseDailyThemeSet, construct_daily_themes
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog, postgres_catalog_fixture

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)

postgres_catalog = postgres_catalog_fixture("artifact_reference")

DAY = date(2026, 9, 4)
CAPTURED_AT = "2026-09-04T06:00:00+00:00"
SENTIMENT_ARTIFACT_ID = f"news:sentiment:{sentiment_request_id(NewsGroup(id='1' * 64, article_version_ids=('2' * 64,)))}"


class FakeR2Client:
    """In-memory stand-in for the R2 S3 client at the storage network boundary."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str]) -> None:
        self.objects[Key] = Body


@pytest.fixture
def fake_r2(monkeypatch: pytest.MonkeyPatch) -> FakeR2Client:
    client = FakeR2Client()
    monkeypatch.setattr(storage, "_r2_client", lambda: client)
    return client


def test_read_daily_report_input_carries_exact_current_references(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
) -> None:
    references = _seed_report_input_day(postgres_catalog, fake_r2)
    sentiment = _seed_current_file(
        postgres_catalog,
        artifact_id=SENTIMENT_ARTIFACT_ID,
        kind="news_sentiment",
        content=b"{}",
    )

    value = read_daily_report_input(DAY)

    assert value.day == DAY
    assert value.themes == references["themes"]
    assert value.assessments == references["assessments"]
    assert value.cluster_set == references["cluster"]
    assert value.summaries == ()
    assert value.sentiments == (sentiment,)


def test_read_daily_report_input_reports_missing_current_analysis_artifact(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
) -> None:
    _seed_report_input_day(postgres_catalog, fake_r2)

    with pytest.raises(
        ReportInputsUnavailable,
        match=f"Required analysis artifact is unavailable: {SENTIMENT_ARTIFACT_ID}",
    ):
        read_daily_report_input(DAY)


def test_read_daily_report_input_rejects_artifact_with_more_than_one_current_file(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
) -> None:
    _seed_report_input_day(postgres_catalog, fake_r2)
    _seed_current_file(
        postgres_catalog,
        artifact_id=SENTIMENT_ARTIFACT_ID,
        kind="news_sentiment",
        content=b"{}",
        duplicate_file=True,
    )

    with pytest.raises(
        ValueError,
        match=f"Artifact has more than one current file: {SENTIMENT_ARTIFACT_ID}",
    ):
        read_daily_report_input(DAY)


def _seed_report_input_day(
    catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
) -> dict[str, ArtifactReference]:
    cluster = artifact_file(
        artifact_id=f"news:clusters:{DAY.isoformat()}",
        artifact_kind="news_clusters",
        title=f"Romanian news clusters for {DAY.isoformat()}",
        content=_cluster_payload(),
        r2_key=f"news/clusters/{DAY.isoformat()}.json",
        media_type="application/json",
    )
    theme_output = construct_daily_themes(
        DailyThemeInput(
            day=DAY,
            cluster_set=_reference(cluster),
            groups=(),
        )
    )
    themes = artifact_file(
        artifact_id=f"news:themes:{DAY.isoformat()}",
        artifact_kind="news_daily_themes",
        title=f"Romanian news themes for {DAY.isoformat()}",
        content=theme_output.content,
        r2_key=f"news/themes/{DAY.isoformat()}.json",
        media_type="application/json",
    )
    sparse_theme_set = theme_output.theme_set
    assert isinstance(sparse_theme_set, SparseDailyThemeSet)
    assessment_output = construct_daily_subject_assessments(
        DailySubjectAssessmentInput(
            day=DAY,
            themes=_reference(themes),
            theme_set=sparse_theme_set,
            summaries=(),
            relevance=(),
            evidence=(),
        )
    )
    assessments = artifact_file(
        artifact_id=f"news:subject-assessments:{DAY.isoformat()}",
        artifact_kind="news_daily_subject_assessments",
        title=f"Romanian news subject assessments for {DAY.isoformat()}",
        content=assessment_output.content,
        r2_key=f"news/subject-assessments/{DAY.isoformat()}.json",
        media_type="application/json",
    )
    fake_r2.objects[cluster.r2_key] = cluster.content
    fake_r2.objects[themes.r2_key] = themes.content
    fake_r2.objects[assessments.r2_key] = assessments.content
    _seed_current_file(
        catalog,
        artifact_id=themes.artifact_id,
        kind="news_daily_themes",
        content=themes.content,
        r2_key=themes.r2_key,
    )
    _seed_current_file(
        catalog,
        artifact_id=assessments.artifact_id,
        kind="news_daily_subject_assessments",
        content=assessments.content,
        r2_key=assessments.r2_key,
    )
    return {
        "themes": _reference(themes),
        "assessments": _reference(assessments),
        "cluster": _reference(cluster),
    }


def _cluster_payload() -> bytes:
    cluster_set = DailyClusterSet(
        day=DAY,
        algorithm="synthetic",
        threshold=0.72,
        embedding_model="synthetic",
        article_version_ids=("2" * 64,),
        relevance_version_ids=("3" * 64,),
        embedding_version_ids=("4" * 64,),
        merges=(),
        groups=(NewsGroup(id="1" * 64, article_version_ids=("2" * 64,)),),
    )
    return cluster_set.model_dump_json().encode()


def _seed_current_file(
    catalog: PostgresCatalog,
    *,
    artifact_id: str,
    kind: str,
    content: bytes,
    r2_key: str | None = None,
    duplicate_file: bool = False,
) -> ArtifactReference:
    key = r2_key or f"news/test/{artifact_id}.json"
    file = artifact_file(
        artifact_id=artifact_id,
        artifact_kind=kind,
        title=f"Seeded {kind}",
        content=content,
        r2_key=key,
        media_type="application/json",
    )
    catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
        "visibility, created_at) VALUES "
        "(%s, %s, %s, 'derived', 'current', 'private', %s) ON CONFLICT DO NOTHING",
        (artifact_id, kind, file.title, CAPTURED_AT),
    )
    catalog.execute(
        "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
        "created_at) VALUES (%s, %s, 1, %s, %s) ON CONFLICT DO NOTHING",
        (file.version_id, artifact_id, file.content_digest, CAPTURED_AT),
    )
    _seed_artifact_file(catalog, file.version_id, file.r2_key, file.content_digest, len(content))
    if duplicate_file:
        _seed_artifact_file(
            catalog, file.version_id, f"{file.r2_key}.duplicate", file.content_digest, len(content)
        )
    catalog.execute(
        "UPDATE artifacts SET current_version_id = %s WHERE id = %s",
        (file.version_id, artifact_id),
    )
    return _reference(file)


def _seed_artifact_file(
    catalog: PostgresCatalog,
    version_id: str,
    r2_key: str,
    content_digest: str,
    byte_size: int,
) -> None:
    catalog.execute(
        "INSERT INTO artifact_files (id, artifact_version_id, r2_key, media_type, "
        "content_digest, byte_size) VALUES (%s, %s, %s, 'application/json', %s, %s) "
        "ON CONFLICT DO NOTHING",
        (sha256(f"{version_id}\0{r2_key}".encode()), version_id, r2_key, content_digest, byte_size),
    )


def _reference(file: ArtifactFile) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=file.artifact_id,
        version_id=file.version_id,
        content_digest=file.content_digest,
        r2_key=file.r2_key,
    )
