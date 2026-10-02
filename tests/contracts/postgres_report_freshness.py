from __future__ import annotations

import hashlib
from datetime import date

import pytest

from romanian_news.catalog.report_inputs import read_current_daily_report_input_versions
from romanian_news.current_report import read_daily_report_freshness
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog, postgres_catalog_fixture

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_DSN is None, reason="NEWS_TEST_POSTGRES_DSN is required"
)

postgres_catalog = postgres_catalog_fixture("report_freshness")
DAY = date(2026, 10, 1)
CAPTURED_AT = "2026-10-01T06:00:00+00:00"


def _id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _seed_daily_publication_lineage(
    postgres_catalog: PostgresCatalog, *, with_report: bool
) -> dict[str, dict[str, str]]:
    """Seed one day's theme/assessment/sentiment lineage, optionally with a report run."""
    day = DAY.isoformat()
    run_names = ("themes", "assessments", "sentiment_old", "sentiment_current", "sentiment_update")
    if with_report:
        run_names = (*run_names, "report")
    runs = {name: _id(f"run:{name}") for name in run_names}
    artifact_ids = {
        "cluster": f"news:clusters:{day}",
        "themes": f"news:themes:{day}",
        "assessments": f"news:subject-assessments:{day}",
        "summary_one": "news:summary:one",
        "sentiment_one": "news:sentiment:one",
        "sentiment_two": "news:sentiment:two",
    }
    kinds = {
        "cluster": "news_daily_clusters",
        "themes": "news_daily_themes",
        "assessments": "news_daily_subject_assessments",
        "summary_one": "news_summary",
        "sentiment_one": "news_sentiment",
        "sentiment_two": "news_sentiment",
    }
    if with_report:
        artifact_ids["report"] = f"news:daily:{day}"
        artifact_ids["summary_two"] = "news:summary:two"
        kinds["report"] = "news_daily_report"
        kinds["summary_two"] = "news_summary"
    version_labels = (
        "cluster_old",
        "cluster",
        "themes",
        "assessments",
        "summary_one",
        "sentiment_one",
        "sentiment_two",
    )
    if with_report:
        version_labels = (*version_labels, "summary_two", "report")
    versions = {name: _id(f"version:{name}") for name in version_labels}
    digests = {name: _id(f"content:{name}") for name in version_labels}
    version_artifacts = {
        "cluster_old": "cluster",
        **{name: name for name in version_labels if name != "cluster_old"},
    }
    produced_by = {
        "themes": "themes",
        "assessments": "assessments",
        "sentiment_one": "sentiment_old",
        "sentiment_two": "sentiment_current",
    }
    if with_report:
        produced_by["report"] = "report"
    statements: list[tuple[str, list[object]]] = []
    for name, run_id in runs.items():
        statements.append(
            (
                "INSERT INTO runs (id, operation_key, executor_kind, implementation_ref, "
                "parameters_json, actor, status, idempotency_key, started_at) "
                "VALUES (%s, %s, 'python', 'test', '{}'::jsonb, 'test', 'completed', %s, %s)",
                [run_id, f"test.{name}", f"test:{name}", CAPTURED_AT],
            )
        )
    for name, artifact_id in artifact_ids.items():
        statements.append(
            (
                "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
                "visibility, created_at) VALUES (%s, %s, %s, 'derived', 'current', 'private', %s)",
                [artifact_id, kinds[name], name, CAPTURED_AT],
            )
        )
    for name in version_labels:
        statements.append(
            (
                "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
                "produced_by_run_id, created_at) VALUES (%s, %s, 1, %s, %s, %s)",
                [
                    versions[name],
                    artifact_ids[version_artifacts[name]],
                    digests[name],
                    runs[produced_by[name]] if name in produced_by else None,
                    CAPTURED_AT,
                ],
            )
        )

    theme_inputs: tuple[tuple[str, str], ...] = (
        ("cluster", "cluster_set"),
        ("summary_one", "summary"),
    )
    if with_report:
        theme_inputs = (*theme_inputs, ("summary_two", "summary"))
    inputs: dict[str, tuple[tuple[str, str], ...]] = {
        "themes": theme_inputs,
        "assessments": (("themes", "themes"),),
        "sentiment_old": (("cluster_old", "cluster_set"),),
        "sentiment_current": (("cluster", "cluster_set"),),
    }
    if with_report:
        inputs["report"] = (
            ("themes", "themes"),
            ("assessments", "assessments"),
            ("cluster", "cluster_set"),
            ("summary_one", "summary"),
            ("summary_two", "summary"),
            ("sentiment_one", "sentiment"),
            ("sentiment_two", "sentiment"),
        )
    for run_name, references in inputs.items():
        for position, (version_name, role) in enumerate(references):
            statements.append(
                (
                    "INSERT INTO run_inputs (run_id, position, artifact_version_id, role, "
                    "selected_content_digest, selection_method) "
                    "VALUES (%s, %s, %s, %s, %s, 'whole_file')",
                    [runs[run_name], position, versions[version_name], role, digests[version_name]],
                )
            )
    for version_name, run_name in produced_by.items():
        statements.append(
            (
                "INSERT INTO run_outputs (run_id, position, artifact_version_id, role) "
                "VALUES (%s, 0, %s, 'output')",
                [runs[run_name], versions[version_name]],
            )
        )
    if with_report:
        statements.append(
            (
                "INSERT INTO artifact_files (id, artifact_version_id, r2_key, media_type, "
                "content_digest, byte_size) VALUES (%s, %s, %s, 'application/json', %s, 0)",
                [
                    _id("report_file"),
                    versions["report"],
                    f"news/reports/daily/{day}/report.json",
                    digests["report"],
                ],
            )
        )
    for name in artifact_ids:
        statements.append(
            (
                "UPDATE artifacts SET current_version_id = %s, current_run_id = %s WHERE id = %s",
                [
                    versions[name],
                    runs[produced_by[name]] if name in produced_by else None,
                    artifact_ids[name],
                ],
            )
        )
    postgres_catalog.batch(statements)
    return {"runs": runs, "versions": versions, "artifact_ids": artifact_ids}


def test_reused_sentiment_lineage_still_marks_current_report_fresh(
    postgres_catalog: PostgresCatalog,
) -> None:
    seeded = _seed_daily_publication_lineage(postgres_catalog, with_report=True)
    versions = seeded["versions"]

    current = read_current_daily_report_input_versions(DAY)
    assert current is not None
    assert current.sentiments == tuple(
        sorted((versions["sentiment_one"], versions["sentiment_two"]))
    )
    assert read_daily_report_freshness(DAY).kind == "fresh"

    updated_version = _id("version:sentiment_one_updated")
    postgres_catalog.batch(
        [
            (
                "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
                "produced_by_run_id, created_at) VALUES (%s, %s, 1, %s, %s, %s)",
                [
                    updated_version,
                    seeded["artifact_ids"]["sentiment_one"],
                    _id("content:sentiment_one_updated"),
                    seeded["runs"]["sentiment_update"],
                    CAPTURED_AT,
                ],
            ),
            (
                "INSERT INTO run_outputs (run_id, position, artifact_version_id, role) "
                "VALUES (%s, 0, %s, 'output')",
                [seeded["runs"]["sentiment_update"], updated_version],
            ),
            (
                "UPDATE artifacts SET current_version_id = %s, current_run_id = %s WHERE id = %s",
                [
                    updated_version,
                    seeded["runs"]["sentiment_update"],
                    seeded["artifact_ids"]["sentiment_one"],
                ],
            ),
        ]
    )
    assert read_daily_report_freshness(DAY).kind == "stale"


def test_sentiment_lineage_is_read_before_any_report_is_captured(
    postgres_catalog: PostgresCatalog,
) -> None:
    seeded = _seed_daily_publication_lineage(postgres_catalog, with_report=False)
    versions = seeded["versions"]

    current = read_current_daily_report_input_versions(DAY)

    assert current is not None
    assert current.sentiments == (versions["sentiment_two"],)
    assert read_daily_report_freshness(DAY).kind == "missing"
