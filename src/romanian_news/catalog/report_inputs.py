from datetime import date, datetime

from romanian_news import NewsModel, Sha256
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog_transport import catalog_query


class DailyReportParameters(NewsModel):
    day: date


class RecordedDailyReportCatalogInput(NewsModel):
    day: date
    themes: ArtifactReference
    assessments: ArtifactReference
    cluster_set: ArtifactReference
    summaries: tuple[ArtifactReference, ...]
    sentiments: tuple[ArtifactReference, ...]


class ReportDayHead(NewsModel):
    day: date
    current_run_id: str | None


class ReportArticleCatalogRecord(NewsModel):
    reference: ArtifactReference
    outlet_id: str
    canonical_url: str


class CurrentDailyReportRecord(NewsModel):
    day: date
    version_id: Sha256
    run_id: Sha256
    content_digest: Sha256
    r2_key: str
    input_time: datetime


class ExactDailyReportFile(NewsModel):
    version_id: Sha256
    content_digest: Sha256
    r2_key: str


def read_exact_daily_report_file(version_id: Sha256) -> ExactDailyReportFile:
    rows = catalog_query(
        """SELECT version.id AS version_id, file.content_digest, file.r2_key
           FROM artifact_versions version
           JOIN artifacts artifact ON artifact.id = version.artifact_id
           JOIN artifact_files file ON file.artifact_version_id = version.id
           WHERE version.id = %s AND artifact.kind = 'news_daily_report'""",
        [version_id],
    )
    if len(rows) != 1:
        raise ValueError(f"Exact daily report file is unavailable: {version_id}")
    return ExactDailyReportFile.model_validate(rows[0], strict=False)


class DailyReportInputVersions(NewsModel):
    day: date
    themes: Sha256
    assessments: Sha256
    cluster_set: Sha256
    summaries: tuple[Sha256, ...]
    sentiments: tuple[Sha256, ...]


def read_current_daily_report_record(day: date) -> CurrentDailyReportRecord | None:
    """Read one report head and its newest recorded input time in one catalog snapshot."""
    rows = catalog_query(
        """SELECT report.current_version_id AS version_id, report.current_run_id AS run_id,
                  file.content_digest, file.r2_key, MAX(input_version.created_at) AS input_time
           FROM artifacts report
           JOIN artifact_files file ON file.artifact_version_id = report.current_version_id
           JOIN run_inputs input
             ON input.run_id = report.current_run_id AND input.role != 'prior_output'
           JOIN artifact_versions input_version ON input_version.id = input.artifact_version_id
           WHERE report.id = %s AND report.kind = 'news_daily_report'
           GROUP BY report.current_version_id, report.current_run_id,
                    file.content_digest, file.r2_key""",
        [f"news:daily:{day.isoformat()}"],
    )
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError(f"Daily report has more than one current file: {day.isoformat()}")
    return CurrentDailyReportRecord.model_validate({"day": day, **rows[0]}, strict=False)


def read_current_daily_report_input_versions(day: date) -> DailyReportInputVersions | None:
    """Read the ready report input versions without loading artifact files."""
    rows = catalog_query(
        """SELECT candidate.role, candidate.version_id
           FROM (
           WITH theme AS (
             SELECT current_version_id AS version_id, current_run_id AS run_id
             FROM artifacts WHERE id = %s AND kind = 'news_daily_themes'
               AND current_version_id IS NOT NULL
           ), assessment AS (
             SELECT artifact.current_version_id AS version_id
             FROM artifacts artifact
             JOIN theme ON TRUE
             WHERE artifact.id = %s AND artifact.kind = 'news_daily_subject_assessments'
               AND artifact.current_version_id IS NOT NULL
               AND EXISTS (
                 SELECT 1 FROM run_inputs input
                 WHERE input.run_id = artifact.current_run_id AND input.role = 'themes'
                   AND input.artifact_version_id = theme.version_id
               )
           ), theme_inputs AS (
             SELECT input.role, input.artifact_version_id AS version_id
             FROM theme JOIN run_inputs input ON input.run_id = theme.run_id
             WHERE input.role IN ('cluster_set', 'summary')
           ), sentiments AS (
             SELECT sentiment.current_version_id AS version_id
             FROM artifacts sentiment
             JOIN artifact_versions version ON version.id = sentiment.current_version_id
             JOIN run_inputs input ON input.run_id = version.produced_by_run_id
               AND input.role = 'cluster_set'
             JOIN theme_inputs cluster ON cluster.role = 'cluster_set'
               AND cluster.version_id = input.artifact_version_id
             WHERE sentiment.kind = 'news_sentiment'
           )
           SELECT 'themes' AS role, version_id FROM theme
           UNION ALL SELECT 'assessments', version_id FROM assessment
           UNION ALL SELECT role, version_id FROM theme_inputs
           UNION ALL SELECT 'sentiment', version_id FROM sentiments
           ) candidate
           ORDER BY candidate.role, candidate.version_id""",
        [
            f"news:themes:{day.isoformat()}",
            f"news:subject-assessments:{day.isoformat()}",
        ],
    )
    return _input_versions(day, rows, require_matching_analysis=True)


def read_daily_report_run_input_versions(
    day: date, run_id: Sha256
) -> DailyReportInputVersions | None:
    """Read the exact report inputs recorded by one captured report run."""
    rows = catalog_query(
        """SELECT input.role, input.artifact_version_id AS version_id
           FROM run_inputs input
           WHERE input.run_id = %s AND input.role != 'prior_output'
           ORDER BY input.role, input.artifact_version_id""",
        [run_id],
    )
    return _input_versions(day, rows, require_matching_analysis=False)


def _input_versions(
    day: date,
    rows: list[dict[str, object]],
    *,
    require_matching_analysis: bool,
) -> DailyReportInputVersions | None:
    versions: dict[str, list[Sha256]] = {
        "themes": [],
        "assessments": [],
        "cluster_set": [],
        "summary": [],
        "sentiment": [],
    }
    for row in rows:
        role = str(row["role"])
        if role not in versions:
            raise ValueError(f"Daily report recorded an unexpected input role: {role}")
        versions[role].append(str(row["version_id"]))
    if (
        len(versions["themes"]) != 1
        or len(versions["assessments"]) != 1
        or len(versions["cluster_set"]) != 1
    ):
        return None
    if require_matching_analysis and len(versions["summary"]) != len(versions["sentiment"]):
        return None
    return DailyReportInputVersions(
        day=day,
        themes=versions["themes"][0],
        assessments=versions["assessments"][0],
        cluster_set=versions["cluster_set"][0],
        summaries=tuple(sorted(versions["summary"])),
        sentiments=tuple(sorted(versions["sentiment"])),
    )


def read_recorded_daily_report_input(day: date) -> RecordedDailyReportCatalogInput | None:
    rows = catalog_query(
        """SELECT run.parameters_json, input.position, input.role, version.artifact_id,
                  version.id AS version_id, file.content_digest, file.r2_key
           FROM artifacts report JOIN runs run ON run.id = report.current_run_id
           JOIN run_inputs input ON input.run_id = report.current_run_id AND input.role != 'prior_output'
           JOIN artifact_versions version ON version.id = input.artifact_version_id
           JOIN artifact_files file ON file.artifact_version_id = version.id
           WHERE report.id = %s ORDER BY input.position""",
        [f"news:daily:{day.isoformat()}"],
    )
    if not rows:
        return None
    parameters = DailyReportParameters.model_validate(rows[0]["parameters_json"], strict=False)
    recorded_day = parameters.day
    references: dict[str, list[ArtifactReference]] = {
        "themes": [],
        "assessments": [],
        "cluster_set": [],
        "summary": [],
        "sentiment": [],
    }
    for row in rows:
        role = str(row["role"])
        if role not in references:
            raise ValueError(f"Daily report recorded an unexpected input role: {role}")
        references[role].append(_reference(row))
    if (
        len(references["themes"]) != 1
        or len(references["assessments"]) != 1
        or len(references["cluster_set"]) != 1
    ):
        raise ValueError("Daily report must record one theme set, assessment set, and cluster set")
    return RecordedDailyReportCatalogInput(
        day=recorded_day,
        themes=references["themes"][0],
        assessments=references["assessments"][0],
        cluster_set=references["cluster_set"][0],
        summaries=tuple(references["summary"]),
        sentiments=tuple(references["sentiment"]),
    )


def read_report_day_heads(not_before: date, through: date) -> tuple[ReportDayHead, ...]:
    rows = catalog_query(
        """SELECT substr(cluster.id, length('news:clusters:') + 1) AS day, report.current_run_id
           FROM artifacts cluster LEFT JOIN artifacts report
             ON report.id = 'news:daily:' || substr(cluster.id, length('news:clusters:') + 1)
           WHERE cluster.kind = 'news_clusters' AND cluster.current_version_id IS NOT NULL
             AND substr(cluster.id, length('news:clusters:') + 1) >= %s
             AND substr(cluster.id, length('news:clusters:') + 1) <= %s ORDER BY day""",
        [not_before.isoformat(), through.isoformat()],
    )
    return tuple(
        ReportDayHead(
            day=date.fromisoformat(str(row["day"])),
            current_run_id=None if row["current_run_id"] is None else str(row["current_run_id"]),
        )
        for row in rows
    )


def read_report_articles(
    version_ids: tuple[Sha256, ...],
) -> dict[Sha256, ReportArticleCatalogRecord]:
    result: dict[Sha256, ReportArticleCatalogRecord] = {}
    for offset in range(0, len(version_ids), 50):
        values = version_ids[offset : offset + 50]
        placeholders = ", ".join("%s" for _ in values)
        rows = catalog_query(
            f"""SELECT version.artifact_id, metadata.artifact_version_id AS version_id,
                       metadata.outlet_id, metadata.canonical_url, file.r2_key, file.content_digest
                FROM news_article_versions metadata
                JOIN artifact_versions version ON version.id = metadata.artifact_version_id
                JOIN artifact_files file ON file.artifact_version_id = metadata.artifact_version_id
                WHERE metadata.artifact_version_id IN ({placeholders})""",
            list(values),
        )
        for row in rows:
            reference = _reference(row)
            result[reference.version_id] = ReportArticleCatalogRecord(
                reference=reference,
                outlet_id=str(row["outlet_id"]),
                canonical_url=str(row["canonical_url"]),
            )
    return result


def _reference(row: dict[str, object]) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=str(row["artifact_id"]),
        version_id=str(row["version_id"]),
        content_digest=str(row["content_digest"]),
        r2_key=str(row["r2_key"]),
    )
