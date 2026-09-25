# Video digest production verification

This runbook is the release gate for the video digest. Do not start
`scheduled_video_digest` until every production check below passes. PostgreSQL is the workflow
source of truth; Dagster retries must resume its recorded state rather than replay completed work.

## Current status

Verification on 2026-09-25 established these production facts:

| Boundary | Result | Evidence |
| --- | --- | --- |
| Authenticated reader | Pass | `/` returns a `303` to `/login`; `/livez` and `/readyz` return `200` at `https://news.alexlazar.dev` |
| Reader deployment | Pass | Railway deployed repository commit `abdb3e6` successfully |
| Public media origin | Pass | `romanian-news-public-media` is served at `https://news-media.alexlazar.dev` with active TLS 1.2 or newer |
| R2 custom-domain probe | Pass | The permanent smoke object returns exact bytes, `Content-Length`, `Accept-Ranges`, and a valid `206` byte range |
| Dagster deployment | Pass | The code location deployed successfully from the merged H3 reference registry commit |
| Execution ownership | Decided | Dagster owns all background work; the unrelated Railway worker remains untouched |
| Schedule ownership | Safe, inactive | `scheduled_video_digest` is registered `STOPPED` and excluded from production activation |
| Subtitle timing | English sample passed, production gate open | The [approved-video QA](video-digest-english-timing-qa.md) produced 12/12 valid cues with per-story fallback; an exact generated edition still needs reader playback review |
| Incident delivery | Configuration pending | Terminal failure and deadline payloads are wired to an incoming webhook. The stopped five-minute monitor also reads durable publication, budget, and success-gap state. Configure and verify `BETTERSTACK_VIDEO_INCIDENT_WEBHOOK_URL` before activation |
| Video heartbeat | Configuration pending | A successful publication pings `BETTERSTACK_VIDEO_DIGEST_HEARTBEAT_URL`; create the Better Stack heartbeat before activation |

The Railway service named `romanian-news-worker` is unrelated. Its observed deployments target
`mauricedesaxe/chartly`, not this repository. Video work executes only in Dagster.

For incident delivery, configure a Better Stack incoming webhook that creates on
`incident.status=alert`, extracts its alert ID from `incident.id`, title from `incident.title`,
and cause from `incident.description`. Store its URL in Dagster as
`BETTERSTACK_VIDEO_INCIDENT_WEBHOOK_URL`. The stable alert ID prevents duplicate incidents when
Dagster retries. Configure a separate successful-publication heartbeat URL in
`BETTERSTACK_VIDEO_DIGEST_HEARTBEAT_URL`. Test both with a non-production payload before starting
the video schedule. See the [Better Stack incoming webhook guide](https://betterstack.com/docs/uptime/incoming-webhooks/).

`scheduled_video_digest_incident_monitor` is also registered `STOPPED`. Once incident delivery is
verified, start it alongside the video schedule. Every five minutes it reports durable slot failures,
unpublished editions 60 minutes after their slot, publication integrity conflicts, daily or monthly
reservations at 80% of their budget, and a 24-hour gap after selected source changes without a
successful publication. A stalled Fal queue is reported after 20 minutes in one observed queue
state. Map `incident.id` as the Better Stack alert ID so repeated polls update the same incident.

The production reader has `NEWS_PUBLIC_MEDIA_R2_BUCKET=romanian-news-public-media` and
`NEWS_PUBLIC_MEDIA_BASE_URL=https://news-media.alexlazar.dev`. The probe object is
`verification/production-smoke-v1.txt`; its body SHA-256 is
`9767a285d5f6e4d8ce4c2ea2f1afdcc3b6073c92d93e235035bfcfcf61a46716`.

## Activation prerequisites

1. Install production planning, generation, assembly, subtitle timing, and publication adapters
   through `runtime_factory`.
2. Configure `NEWS_POSTGRES_DSN`, private R2 credentials, `NEWS_PUBLIC_MEDIA_R2_BUCKET`, and the bare
   HTTPS `NEWS_PUBLIC_MEDIA_BASE_URL` in both the worker and reader as required.
3. Apply all catalog migrations to production PostgreSQL and run the upgrade contracts against a
   disposable PostgreSQL 15 database restored from the pre-video schema.
4. Deploy the Dagster location from the same commit as the reader and confirm definitions load.
5. Probe one immutable public video and subtitle through the custom domain. Require exact bytes,
   `Content-Length`, `Accept-Ranges: bytes`, and a successful byte-range response.
6. Run one deterministic slot manually. Stop the worker after each durable stage, restart it with a
   new process, and confirm each paid or public side effect occurred once.
7. Use the [approved English feasibility video](https://media.alexlazar.dev/h3-daily-news/2026-09-18/daily-news-report-subtitled-93bfad318192.mp4)
   from [issue #11](https://github.com/mauricedesaxe/news-aggregator/issues/11), whose SHA-256 is
   `93bfad318192125019bbe48a99b1efeb6deb000e99c036aecd631c21020dd382`, with its exact
   approved English screenplay. Run all three subtitle strategies through the production `base.en`
   timing adapter. Record word and cue coverage, match rate, and fallback rate. Require a valid VTT
   before marking subtitle timing verified.
   Confirm reader playback, transcript order, clean-video fallback after subtitle exhaustion, and
   feedback persistence for that edition.
8. Start `scheduled_video_digest` only after all preceding checks pass.

## Verification commands

Run from a clean checkout at the deployed commit:

```bash
uv sync --frozen --group test
uv run pytest -q --ignore=tests/test_packaging.py
NEWS_TEST_POSTGRES_DSN=postgresql://postgres:postgres@127.0.0.1:5432/postgres \
  uv run pytest -q tests/contracts/postgres_*.py \
  tests/worker/postgres_operations_contract.py tests/reader/postgres_reader_app_e2e.py
uv run pytest -q tests/test_packaging.py
uv run ruff check .
uv run ruff format --check .
uv run basedpyright --level error
uv run vulture
uv run xenon --max-absolute C --max-modules B --max-average A src/romanian_news
uv run dg check defs
bd lint
bd preflight
```

Inspect production variable names without printing their values. The required public-media names must
be present in both relevant services before probing R2. Never place credentials in command history,
logs, screenshots, or issue comments.

## Measured limits

- Schedule: `08:00`, `13:00`, and `20:00` Europe/Bucharest.
- Absolute slot deadline: 90 minutes.
- Lease duration: 10 minutes; recovery increments the lease fence.
- Dagster retry budget: 95 retries, with a minimum 60-second delay.
- Assembly: at most three durable attempts.
- Subtitles: three ordered strategies; exhaustion publishes the clean video.
- Publication: at most five durable attempts.
- Generated candidates: private, seven-day retention class.
- Accepted clips and published media: permanent retention classes.

## Rollback

Stop `scheduled_video_digest` first. Do not delete catalog rows or immutable R2 objects. Roll the
Dagster location and reader back to the last known-good shared commit, then inspect the affected
slot in PostgreSQL. A later deployment must resume from that durable state with a fresh owner token.
If public verification failed, leave the edition unpublished and preserve the publication attempt
evidence for diagnosis.
