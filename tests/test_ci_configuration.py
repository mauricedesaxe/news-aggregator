from pathlib import Path

ROOT = Path(__file__).parents[1]


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
