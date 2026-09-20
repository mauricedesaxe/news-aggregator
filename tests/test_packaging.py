import os
import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).parents[1]


def test_wheel_contains_runtime_assets_without_tests(tmp_path: Path) -> None:
    distributions = tmp_path / "dist"
    subprocess.run(
        ("uv", "build", "--wheel", "--no-build-logs", "--out-dir", str(distributions)),
        cwd=ROOT,
        check=True,
    )
    wheel = next(distributions.glob("*.whl"))

    with ZipFile(wheel) as archive:
        names = set(archive.namelist())

    assert "romanian_news/worker/definitions.py" in names
    assert "romanian_news/reader/app.py" in names
    assert "romanian_news/reader/static/htmx.min.js" in names
    assert "romanian_news/catalog/migrations/0001_initial.sql" in names
    assert "romanian_news/catalog/migrations/0002_video_digest.sql" in names
    assert "romanian_news/catalog/migrations/0003_video_digest_generation_fences.sql" in names
    assert "romanian_news/catalog/migrations/0004_video_digest_publication_evidence.sql" in names
    assert not any("/tests/" in name for name in names)
    assert not any("debt" in name or "budget" in name for name in names)

    installed = tmp_path / "installed"
    subprocess.run(
        (
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(installed),
            "--no-deps",
            str(wheel),
        ),
        check=True,
    )
    imported = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import romanian_news; "
                "import romanian_news.reader.app; "
                "import romanian_news.worker.definitions; "
                "print(romanian_news.__file__)"
            ),
        ),
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(installed)},
        check=True,
        capture_output=True,
        text=True,
    )

    assert imported.stdout.strip() == str(installed / "romanian_news/__init__.py")


def test_source_distribution_excludes_tests_and_recorded_fixtures(tmp_path: Path) -> None:
    distributions = tmp_path / "dist"
    subprocess.run(
        ("uv", "build", "--sdist", "--no-build-logs", "--out-dir", str(distributions)),
        cwd=ROOT,
        check=True,
    )
    source_distribution = next(distributions.glob("*.tar.gz"))

    with tarfile.open(source_distribution) as archive:
        names = set(archive.getnames())

    assert not any("/tests/" in name for name in names)
    assert not any("fixtures" in name for name in names)


def test_reader_deployment_uses_the_standalone_package() -> None:
    dockerfile = (ROOT / "deploy/reader/Dockerfile").read_text()
    railway = (ROOT / "railway.toml").read_text()

    assert "COPY src/ ./src/" in dockerfile
    assert "romanian_news.reader.app:create_app" in dockerfile
    assert "chartly" not in dockerfile
    assert "USER app" in dockerfile
    assert 'dockerfilePath = "deploy/reader/Dockerfile"' in railway
    assert 'healthcheckPath = "/readyz"' in railway


def test_docker_context_excludes_local_and_recorded_data() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text().splitlines()

    assert {".env", ".env.*", "data/", "dist/", "tests/", "src/**/tests/"} <= set(dockerignore)
