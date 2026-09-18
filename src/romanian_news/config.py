import os
from pathlib import Path

from dotenv import load_dotenv
from psycopg import ProgrammingError
from psycopg.conninfo import make_conninfo

load_dotenv()


def _parse_postgres_dsn(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    try:
        return make_conninfo(value)
    except ProgrammingError:
        raise ValueError("PostgreSQL DSN is invalid") from None


BETTERSTACK_INGESTING_HOST: str | None = os.getenv("BETTERSTACK_INGESTING_HOST")
BETTERSTACK_SOURCE_TOKEN: str | None = os.getenv("BETTERSTACK_SOURCE_TOKEN")
BETTERSTACK_REPORT_HEARTBEAT_URL: str | None = os.getenv("BETTERSTACK_REPORT_HEARTBEAT_URL")
BETTERSTACK_MORNING_REPORT_HEARTBEAT_URL: str | None = os.getenv(
    "BETTERSTACK_MORNING_REPORT_HEARTBEAT_URL"
)
BETTERSTACK_RESEARCH_TRIGGER_HEARTBEAT_URL: str | None = os.getenv(
    "BETTERSTACK_RESEARCH_TRIGGER_HEARTBEAT_URL"
)
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
TYPESAFE_API_KEY: str | None = os.getenv("TYPESAFE_API_KEY")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY: str | None = os.getenv("YOUTUBE_API_KEY")
LANGFUSE_PUBLIC_KEY: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY: str | None = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL: str | None = os.getenv("LANGFUSE_BASE_URL")
LANGFUSE_PROJECT_ID: str | None = os.getenv("LANGFUSE_PROJECT_ID")
CLOUDFLARE_ACCOUNT_ID: str | None = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN: str | None = os.getenv("CLOUDFLARE_API_TOKEN")
NEWS_WORKSPACE: Path = Path(os.getenv("NEWS_WORKSPACE", "/tmp/romanian-news"))
NEWS_R2_BUCKET: str = os.getenv("NEWS_R2_BUCKET", "")
NEWS_POSTGRES_DSN: str | None = _parse_postgres_dsn(os.getenv("NEWS_POSTGRES_DSN"))
NEWS_TEST_POSTGRES_DSN: str | None = _parse_postgres_dsn(os.getenv("NEWS_TEST_POSTGRES_DSN"))
IMPLEMENTATION_REF: str = (
    os.getenv("IMPLEMENTATION_REF") or os.getenv("DAGSTER_CLOUD_GIT_SHA") or "local-working-tree"
)
DAGSTER_CLOUD_GRAPHQL_URL: str | None = os.getenv("DAGSTER_CLOUD_GRAPHQL_URL")
DAGSTER_CLOUD_API_TOKEN: str | None = os.getenv("DAGSTER_CLOUD_API_TOKEN")

APP_PASSWORD: str | None = os.getenv("APP_PASSWORD")
SESSION_SECRET: str | None = os.getenv("SESSION_SECRET")
# Force the Secure flag on the session cookie regardless of request scheme,
# which derives from the spoofable X-Forwarded-Proto behind the proxy. Set
# COOKIE_SECURE=true in the Railway dashboard; leave unset locally so the
# cookie still works over plain http in development.
COOKIE_SECURE: bool = os.getenv("COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes")
TRUST_PROXY_HEADERS: bool = os.getenv("TRUST_PROXY_HEADERS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
