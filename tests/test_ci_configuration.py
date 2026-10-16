import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
DAGSTER_CLOUD_ACTION_SHA = "a5c409dd1635efc77ca2a3925288864f1b35f73e"
PRODUCTION_PYTHON_WORKFLOWS = (
    "archive-article-backfill.yml",
    "archive-article-capture.yml",
    "archive-page-backfill.yml",
    "archive-page-metadata.yml",
    "archive-sitemap-discovery.yml",
    "archive-sitemap-inventory.yml",
    "catalog-schema-activation.yml",
    "historical-replay-preflight.yml",
)


def test_ci_covers_the_standalone_project() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    for command in (
        "uv sync --frozen --group test",
        "ruff check .",
        "ruff format --check .",
        "basedpyright --level error",
        "uv run --no-sync vulture",
        "uv run --no-sync xenon",
        "uv run --no-sync pytest",
        "uv run --no-sync dg check defs",
        "uv build",
        "tests/contracts/postgres_*.py",
        "tests/worker/postgres_operations_contract.py",
        "tests/reader/postgres_reader_app_e2e.py",
        "docker build -f deploy/reader/Dockerfile",
        "http://127.0.0.1:8080/livez",
    ):
        assert command in workflow

    assert "image: postgres:15" in workflow
    assert "NEWS_TEST_POSTGRES_DSN" in workflow


def test_ci_pull_requests_receive_no_production_secrets() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert "secrets." not in workflow
    assert "DAGSTER_CLOUD_API_TOKEN" not in workflow
    assert "CLOUDFLARE_API_TOKEN" not in workflow


def test_dagster_cloud_actions_pin_the_release_commit() -> None:
    workflow = (ROOT / ".github/workflows/dagster-plus-deploy.yml").read_text()
    revisions = re.findall(r"dagster-io/dagster-cloud-action/[^@]+@([0-9a-f]{40})", workflow)

    assert revisions
    assert set(revisions) == {DAGSTER_CLOUD_ACTION_SHA}


def test_production_workflows_share_the_pinned_python_setup() -> None:
    action = (ROOT / ".github/actions/setup-production-python/action.yml").read_text()

    for expected in (
        "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
        'python-version: "3.12"',
        "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9",
        'version: "0.12.5"',
        "uv sync --frozen --no-default-groups",
    ):
        assert expected in action

    for name in PRODUCTION_PYTHON_WORKFLOWS:
        workflow = (ROOT / ".github/workflows" / name).read_text()
        assert workflow.count("uses: ./.github/actions/setup-production-python") == 1
        assert workflow.index("uses: actions/checkout@") < workflow.index(
            "uses: ./.github/actions/setup-production-python"
        )
        assert "actions/setup-python@" not in workflow
        assert "astral-sh/setup-uv@" not in workflow
        assert "uv sync --frozen --no-default-groups" not in workflow
