# Video digest orchestration

## Caller usage

Dagster owns one operation and one job. The schedule records the deterministic slot before it
requests a run. A production runtime supplies the source decision, catalog adapter, and domain
ports:

```python
outcome, alert = run_video_digest(
    VideoDigestRunRequest(slot=slot, owner_token=run_id, source=source),
    catalog,
    DomainPorts(
        planning=planning,
        generation=generation,
        assembly=assembly,
        subtitles=subtitles,
        publication=publication,
    ),
)
```

PostgreSQL remains the durable source of truth. Dagster supplies the owner token and invokes the
runner. It does not retain workflow progress or choose attempt indexes.

## Typed state and actions

`read_slot_resume_state()` returns one discriminated state for a known slot. Each active state
contains its fenced lease and only the data needed for its next action. `next_action()` maps that
state to one of `PlanAction`, `GenerateAction`, `AssembleAction`, `SubtitleAction`, or
`PublishAction`.

The runner performs this sequence for each action:

1. Read the current catalog state.
2. Reacquire the active slot through a row lock.
3. Reload the catalog state after acquisition.
4. Check the slot's absolute 90-minute deadline.
5. Renew the acquired lease fence.
6. Execute one action through its domain port.
7. Reload the catalog state.

An unexpired lease held by another owner returns a typed busy result with an absolute retry time.
An unexpired same-owner lease replays without changing its fence. Any expired lease can be
recovered, including by the same owner, and recovery increments `claim_count` before work resumes.
The incremented count fences every lease value loaded before recovery.

The runner returns a bounded retry time when a port reports that external work is still pending,
durable state does not advance, another owner holds the lease, or the action bound is reached.
The Dagster operation maps that outcome to `RetryRequested` with at least a one-minute delay and
a budget of 95 attempts. The retry budget therefore reaches the 90-minute catalog deadline even
when a port asks to poll sooner. The step cannot succeed while the catalog slot remains active,
and a terminal slot failure fails the Dagster step without another retry. A restart follows the
same sequence from the catalog state and does not replay completed actions.

Deadline terminalization records deterministic failure evidence, the `deadline` failure reason,
and the exact terminal lease fence in one catalog transaction. Reloading the failed slot retains
the reason, so alert selection remains `deadline` after a process restart.

PostgreSQL stores assembly and subtitle attempts in append-only ledgers. Assembly attempts are
contiguous and stop after three attempts or the first success. Subtitle attempts must use
`whole-edition-v1`, `per-story-v1`, and `per-story-without-vad-v1` in that order. Failure of all
three subtitle strategies records the failed subtitle outcome and advances the clean video to
publication without an incident.

## Schedule ownership

`scheduled_video_digest` runs at `0 8,13,20 * * *` in `Europe/Bucharest`. The slot ID determines
the run key as `video-digest:{slot_id}`. The schedule records each slot before checking Dagster
for a queued or active run of `video_digest`. If one exists, it records the new slot as
`overlapping_run` and does not request another run.

The job contains one operation. Its executor allows one concurrent operation, and schedule
evaluation suppresses concurrent runs of the same job. Run tags contain only the slot ID, slot
name, and scheduled time.

## Publication boundary

`PublicationPort` receives one `PublicationHandoff`. The handoff contains the exact assembled
video artifact metadata and either the exact subtitle artifact metadata or the durable failed
subtitle state. The orchestration layer does not upload public R2 objects, retry publication five
times, verify public objects, or deliver incidents. `news-nvs.7` supplies those adapters and owns
their policies.

Alert selection is pure. Skips and nonterminal retries return `NoAlert`. Subtitle exhaustion also
returns `NoAlert` because it proceeds with the clean video. A terminal eligible-slot failure or
deadline returns an `IncidentAlert` whose ID is the digest of the slot ID and alert category.

## Rejected alternatives

- A generic workflow ledger would duplicate slot stage and attempt state already held in the
  catalog.
- A queue table would duplicate Dagster run state and the existing slot lease.
- Publication and alert ledgers belong with the `news-nvs.7` transport policy, not this runner.
- A Dagster graph with one operation per stage would make Dagster another workflow owner and
  weaken catalog-driven recovery.
- Retrying assembly or subtitles inside one process would lose the durable attempt bound after a
  crash.

## Why the schedule is stopped

The repository does not yet contain every production adapter. Screenplay preparation and Fal H3
generation exist, but the production subtitle timing, public publication, and incident transport
adapters are incomplete. The schedule is registered as `STOPPED`, and the production activation
workflow excludes it. `news-nvs.7` can start it only after those ports have production adapters.
