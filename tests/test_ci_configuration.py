import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
DAGSTER_CLOUD_ACTION_SHA = "a5c409dd1635efc77ca2a3925288864f1b35f73e"


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
