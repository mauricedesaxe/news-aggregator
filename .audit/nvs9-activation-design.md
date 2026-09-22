# news-nvs.9 activation design — session handoff 2026-09-22

## Shipped

- PR #15 (Jev migration evaluation suite) repaired, rebased, and merged to `main`
  at `fe373c5`. Full gate passed locally: 1108 tests incl. 141 PostgreSQL
  contracts, packaging, Ruff, basedpyright, vulture, xenon, `dg check defs`.
- Production public-media boundary live: `romanian-news-public-media` bucket,
  origin `https://news-media.alexlazar.dev`, smoke object verified byte-exact.

## Current branch

`feat/30-production-video-workflow` (from `origin/main` = `fe373c5`).

## Architecture trace (completed this session)

Three read-only explorations covered the full video-digest path. Condensed map:

```
scheduled_video_digest [STOPPED, 08/13/20 Europe/Bucharest]
  → video_digest job → orchestrate_video_digest op
      runtime_factory()          ← None; op fails non-retryably today
      VideoDigestRuntime.run()
        source resolution        ← missing (no slot→report decision)
        CatalogPort adapter      ← missing (functions exist, no composition)
        DomainPorts
          planning               ← missing adapter (prepare_paid_generation exists)
          generation             ← missing adapter (generate_next_candidate exists)
          assembly               ← missing adapter; see mismatch below
          subtitles              ← missing adapter + missing timing provider
          publication            ← R2PublicationPort implemented
```

### Confirmed blocking gaps (in dependency order)

1. `runtime_factory` unset — `worker/video_digest.py:39`. Every run fails at
   lines 65–77 with "Video digest production adapters are not configured."
2. No `VideoDigestRuntime` implementation, no source resolver (which report
   version, empty/unchanged classification, schema-v3 eligibility).
3. No `CatalogPort` adapter composing schedule_slot / read_slot_resume_state /
   skip_slot / claim_slot / reacquire_slot / renew_slot / fail_slot_deadline.
4. No planning/generation port adapters around the existing preflight and
   generation APIs.
5. No production `H3ReferencePack` (only synthetic test fixtures; no R2 assets,
   registry, or resolution rule).
6. **Assembly mismatch**: `assemble_edition()` (media.py:502) bypasses
   `record_assembly_attempt()` and calls `checkpoint_assembled_video()`
   directly; ignores `AssembleAction.attempt_index`. Needs a per-attempt
   boundary or refactor.
7. **Subtitle mismatch**: `produce_subtitles()` (media.py:599) runs all three
   strategies in-process and never calls `record_subtitle_attempt()`. Needs
   one-strategy-per-action execution. No `SubtitleTimingProvider` exists.
8. No ffmpeg/ffprobe guarantee in the Dagster Python-executable deployment;
   no custom worker image; `FileNotFoundError` not converted to typed media
   failure evidence (`accept_candidate` catches MediaValidationError,
   ResearchObjectIntegrityError, TimeoutExpired only).
9. No incident transport (`IncidentAlert` is metadata only).
10. No real-process restart integration test (existing convergence test is
    in-memory against `_Catalog`).

### Strongest existing seams (do not rebuild)

- `run_video_digest()` orchestration reducer + PostgreSQL resume projection
  (`read_slot_resume_state`), lease fencing, 90-min deadline, 64-action bound.
- Fal boundary: durable request-before-submit, receipt-first persistence,
  fail-closed ambiguous window, trusted-host enforcement, spend admission.
- `R2PublicationPort`: intent-upload-commit, exact-byte adoption, public
  verification with ranges, five durable attempts.
- Reader projection + feedback lineage (migrations 0002–0010).

## Chosen design direction

Typed production composition root at
`src/romanian_news/worker/video_digest_runtime.py`:

- `ProductionVideoDigestRuntime` implementing the `VideoDigestRuntime` protocol
- Source resolver slot→exact DailyReport version (schema-v3 only, typed
  skip reasons for missing/empty/unchanged)
- Concrete `CatalogPort` over existing catalog functions
- Planning/generation adapters over existing preflight/generation APIs
- Assembly/subtitle adapters requiring the per-attempt refactor above
- `R2PublicationPort.from_environment()` installed
- `build_video_digest_runtime()` env-validated factory

`worker/video_digest.py` binds the factory; `definitions.py` stays
registration-only. Schedule stays STOPPED until the real Dagster runtime smoke
test passes.

## Rubric for the design arena (framed, not yet run)

1. Restart convergence: every adapter re-derives state from PostgreSQL, zero
   side-effect replay except documented immutable-upload adoption.
2. Fail-closed: missing config/binaries/credentials become typed terminal
   evidence, never partial work.
3. Smallest surface: fewer new public types; composition root owns wiring.
4. Per-attempt durability for assembly and subtitles.
5. Testable at `run_video_digest()` seam with real PostgreSQL + fake ports.

## External blockers (not code)

- GitHub-hosted Actions runners: account-level allocation/billing outage
  (`runner_id: 0`, zero steps). Local gates are the authoritative check.
- Dagster+ deploy workflow cannot run until runners return.

## Verification gate (run before any merge)

```
uv sync --frozen --group test
NEWS_TEST_POSTGRES_DSN=postgresql://postgres:postgres@127.0.0.1:5432/postgres \
  uv run --no-sync pytest -q
uv run --no-sync pytest -q tests/test_packaging.py
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
uv run --no-sync basedpyright --level error
uv run --no-sync vulture
uv run --no-sync xenon --max-absolute C --max-modules B --max-average A src/romanian_news
uv run --no-sync dg check defs
```

PostgreSQL 15 runs locally via `pg_ctlcluster 15 main start` (systemd absent).

## Environment notes

- Beads embedded Dolt restored from remote; database name is `news_restored`
  (`.beads/metadata.json` on this branch). `bd dolt push` publishes tracker
  state.
- Railway production: project `bots` `8c3c26f6-54f2-42ab-862f-dfebaa15be11`,
  env `daabd92f-8bb3-4dc4-b177-fc71059b2beb`, reader `574caf0d-fcec-4ecf-b07e-17bf5d2cec1e`.
  Prefix commands with `RAILWAY_CALLER=skill:use-railway@1.5.1` and a stable
  `RAILWAY_AGENT_SESSION`. Never print secret values.
- Next steps: run the design arena, implement the composition root + per-attempt
  media refactor, add the real-process restart smoke test, then re-verify and
  only then consider schedule activation.
