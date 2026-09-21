# Romanian News Aggregator

A research-driven Romanian news pipeline and reader. The project acquires
registered feeds and selected YouTube sources, extracts articles, evaluates
relevance, groups related reporting, and publishes daily and weekly reports.

The Dagster worker and authenticated FastHTML reader share one installable
Python package. PostgreSQL stores catalog identity and lineage. Cloudflare R2
stores immutable artifacts.

## Development

Python 3.12 and `uv` are required.

```bash
uv sync --group test
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Runtime credentials are supplied through environment variables. No production
credentials or private monorepo code are included in this repository.

## Configuration

The pipeline requires `NEWS_POSTGRES_DSN`, `NEWS_R2_BUCKET`,
`CLOUDFLARE_ACCOUNT_ID`, and `CLOUDFLARE_API_TOKEN`. Model-assisted and source
workflows additionally use the relevant OpenRouter, Gemini, Fal, YouTube,
Langfuse, and Better Stack variables defined in `src/romanian_news/config.py`.
Fal H3 generation reads `FAL_KEY`; `FAL_AI_API_KEY` remains an accepted runtime
alias for the provider's existing secret name.

The reader requires `APP_PASSWORD` and `SESSION_SECRET`. Production deployments
should also set `COOKIE_SECURE=true`. Set `TRUST_PROXY_HEADERS=true` only behind
a proxy such as Railway that overwrites `X-Real-IP` with the connecting client.

GitHub deployment workflows read Dagster organization, URL, environment,
deployment, GraphQL URL, location, API token, and PostgreSQL DSN from repository
secrets so deployment configuration remains outside the public source tree.

## License

MIT. See `LICENSE`. The bundled htmx notice is in `THIRD_PARTY_NOTICES.md`.
