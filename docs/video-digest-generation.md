# Resumable H3 generation

The generation workflow reduces durable catalog state to one next action. It processes the
first incomplete story and reloads PostgreSQL after every provider side effect. Dagster does
not choose story positions or attempt indexes.

## Caller contract

`generate_next_candidate()` accepts a fenced slot lease, the exact verified plan, immutable
reference assets, and the production generation policy. It returns either a candidate for the
media checks in `news-nvs.5`, a completed edition, or a typed terminal failure.

## Approved H3 references

An operator imports approved host video and voice audio files with an explicit approval reference:

```sh
uv run python -m romanian_news.video_digest.references import \
  --video /path/to/approved-host.mp4 \
  --audio /path/to/approved-voice.wav \
  --approval-ref issue-11-approved-assets
```

The command prints a content-derived pack ID. It stores media in private R2, records each exact
artifact version and ordered role in PostgreSQL, and stores an immutable manifest containing the
approval reference. `verify <pack-id>` re-reads and hashes every object. Set
`NEWS_H3_REFERENCE_PACK_ID` to that exact ID in the worker environment. The production generation
port verifies the pack and Fal credentials before request admission, and rejects a pack whose media
references differ from an earlier request in the same edition.
The approved feasibility video is provenance for this workflow, not itself a reference pack.

The workflow uses the existing request checkpoints. A new catalog read projection exposes the
ordered stories, attempts, receipts, responses, and candidates needed to resume. PostgreSQL
still enforces every transition selected by the reducer.

## Request admission

One transaction admits a request and its spend reservation. The transaction:

1. Locks the slot and verifies the exact generation authorization.
2. Requires every lower story position to be accepted.
3. Requires attempt zero to fail before attempt one.
4. Checks the slot deadline using PostgreSQL time.
5. Checks the story, edition, Bucharest-day, and calendar-month limits.
6. Inserts the immutable request and four reservation rows.

The reservation amount remains charged after an unknown terminal cost. This is conservative,
but it prevents missing provider billing data from becoming a zero-dollar attempt. A single
transaction advisory lock serializes budget admission across editions.

The production safeguards are $7 per story, $7 times the mandatory story count per edition,
$150 per Bucharest day, and $1,000 per calendar month. Each request binds the estimate and the
policy artifact used for admission.

## Submission recovery

Fal assigns the queue request ID after accepting a submission. Its public queue API does not
document a caller idempotency key or a lookup by caller identity. PostgreSQL therefore cannot
close the crash window between Fal accepting the request and the receipt checkpoint.

The catalog distinguishes a request created by the current call from an existing pending
request. Only the creator may submit. If a later process finds a pending request without a Fal
receipt, it records an ambiguous-submission failure with unknown cost and does not submit
again. This trades automatic recovery for the no-duplicate-paid-submission invariant.
Documented Fal ingress failures (502, 503, and 504) and connection timeouts before submission
permit retry. Responses that may follow acceptance, including 408, 409, and 429, are ambiguous
and fail closed.

After the receipt reaches immutable R2 and PostgreSQL, every restart polls the stored Fal
request ID. The workflow never submits another request while one is active.

## Media boundary

Fal completion publishes the raw response and candidate bytes, then returns `CandidateReady`.
It does not mark the story accepted. `news-nvs.5` owns deterministic media checks and calls the
existing acceptance or failure checkpoint. A failed first candidate permits one regeneration;
a failed second candidate terminates the story and edition.

## Rejected designs

- A replacement catalog facade would duplicate the existing typed checkpoint API.
- A mutable budget balance would require reconciliation; immutable reservations are enough for
  conservative safeguards.
- Application-only ordering and budget checks would race across workers.
- Retrying an ordinary Fal submission after a timeout could create a second paid request.
- Media probing inside generation would couple this stage to `news-nvs.5`.
