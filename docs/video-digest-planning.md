# Video digest planning boundary

## Problem

Planning must turn one immutable daily report and one immutable policy bundle into a complete screenplay without starting paid video generation early. The current catalog freezes the first checkpointed plan, so drafts and rewrites cannot use the canonical plan tables.

## Caller usage

The future Dagster workflow calls one resumable preparation operation:

```python
prepared = prepare_paid_generation(lease, report, policy)
for story in prepared.plan.stories:
    generate_story(prepared.authorization, story)
```

`prepare_paid_generation` returns only after PostgreSQL contains the accepted plan, every independent story verification, and the edition verification manifest.

## Shape

- `VideoDigestPolicyBundle` versions all planning and verification behavior.
- `PlanningAttempt` records each complete draft, its verification results, and its disposition as immutable evidence.
- `VerifiedDigestPlan` contains the accepted screenplay and exact report subjects in report order.
- `GenerationAuthorization` binds the accepted plan to its ordered verification evidence.
- Draft attempts use an append-only ledger. They never enter canonical story tables.
- The catalog parses canonical attempt and plan artifacts and rejects any mismatch between their bytes, provider evidence, accepted screenplay, or relational plan projection.
- Fenced, replay-safe checkpoints persist the accepted plan and each story verification before the final edition manifest authorizes generation.
- PostgreSQL rejects generation requests without the edition authorization.
- Editions created before this planning contract remain explicitly legacy: already-submitted work may settle, but they cannot start new paid work without complete original evidence.
- The Fal boundary accepts `GenerationAuthorization`, so application code cannot submit an unverified story accidentally.

Validation stays in pure planning and verification functions. OpenRouter payload parsing, R2 publication, PostgreSQL rows, and Dagster contexts remain at their respective boundaries.

## Synthesis decision

The convergent preflight design is the base because it preserves the existing immutable plan state machine. The rejected checkpoint-each-draft design contributed exact replay semantics and the database generation barrier. Drafts remain discoverable through an append-only attempt ledger rather than canonical story rows.

## Tradeoffs accepted

- We accept private orphaned content-addressed R2 objects after a catalog failure in exchange for retry-safe R2-first publication.
- We accept visible intermediate checkpoints for resumable recovery; the manifest barrier keeps every partial edition unauthorized for paid generation.
- We retain detailed attempt evidence in immutable artifacts and index only the identities PostgreSQL needs to enforce bounds and readiness.

## Alternatives considered

- Checkpointing each draft as the edition plan lost because the first checkpoint is immutable and already advances the slot toward generation.
- Letting Dagster sequence story verification lost because another caller or retry could submit paid work early.
- Adding a second draft edition state machine lost because an attempt ledger preserves auditability with a smaller catalog change.

## Open risks

- Later editions that depend on prior published coverage need that comparison baseline included in an immutable input identity.
- Synchronous planning model calls retain a small crash window after provider completion and before attempt recording. This can repeat an unpaid planning call but cannot authorize Fal generation.
