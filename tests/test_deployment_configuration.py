from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_runtime_configuration_has_no_owner_infrastructure_defaults() -> None:
    config = (ROOT / "src/romanian_news/config.py").read_text()
    workflows = "\n".join(path.read_text() for path in (ROOT / ".github/workflows").glob("*.yml"))

    assert "leetsoftware" not in config
    assert "chartly-research" not in config
    assert "leetsoftware" not in workflows
    assert "mauricedesaxe" not in workflows
    assert 'os.getenv("NEWS_R2_BUCKET", "")' in config
    assert 'os.getenv("DAGSTER_CLOUD_GRAPHQL_URL")' in config


def test_deployment_workflows_require_operator_owned_secrets() -> None:
    deployment = (ROOT / ".github/workflows/dagster-plus-deploy.yml").read_text()
    production = (ROOT / ".github/workflows/romanian-news-production.yml").read_text()

    for variable in (
        "DAGSTER_CLOUD_ORGANIZATION",
        "DAGSTER_CLOUD_URL",
        "DAGSTER_CLOUD_ENV",
        "DAGSTER_CLOUD_DEPLOYMENT",
    ):
        assert f"secrets.{variable}" in deployment or f"secrets.{variable}" in production
    assert "secrets.DAGSTER_CLOUD_GRAPHQL_URL" in production
    assert "secrets.DAGSTER_CLOUD_LOCATION" in production
    assert "secrets.NEWS_POSTGRES_DSN" in production


def test_deploy_requires_push_and_operator_secrets() -> None:
    workflow = (ROOT / ".github/workflows/dagster-plus-deploy.yml").read_text()
    validation, deploy = workflow.split("\n  deploy:", maxsplit=1)

    assert "DAGSTER_CLOUD_API_TOKEN" not in validation
    assert "GITHUB_TOKEN" not in validation
    assert "github.event_name == 'push'" in deploy
    assert "environment: production" in deploy
    assert "secrets.DAGSTER_CLOUD_API_TOKEN" in deploy
    assert "secrets.DAGSTER_CLOUD_DEPLOYMENT" in deploy


def test_public_production_logs_exclude_remote_error_payloads() -> None:
    workflow = (ROOT / ".github/workflows/romanian-news-production.yml").read_text()

    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert "environment: production" in workflow
    assert "logsForRun" not in workflow
    assert "print(response.text)" not in workflow
    assert "print(event)" not in workflow
    assert "stack" not in workflow
