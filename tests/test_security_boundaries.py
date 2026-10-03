from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_docker_build_excludes_private_data_and_drops_root() -> None:
    excluded = set((ROOT / ".dockerignore").read_text().splitlines())
    dockerfile = (ROOT / "deploy/reader/Dockerfile").read_text()

    assert {".env", ".env.*", "data/"} <= excluded
    assert "USER app" in dockerfile


def test_public_production_logs_exclude_remote_error_payloads() -> None:
    workflow = (ROOT / ".github/workflows/romanian-news-production.yml").read_text()

    assert "logsForRun" not in workflow
    assert "print(response.text)" not in workflow
    assert "print(event)" not in workflow
