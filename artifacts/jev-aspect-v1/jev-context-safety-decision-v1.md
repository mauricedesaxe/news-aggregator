# Jev context-safety decision

This decision covers context routing only. It does not approve a production migration.

The local guard uses `len(state) <= 70,000`. The value comes from one deterministic ASCII probe: 70,000 characters was accepted at 30,489 input tokens, while 80,000 characters returned HTTP 400 `max_tokens_exceeded`. This is not a formal tokenizer bound.

| Role | Status | Affected production requests | Provider requests | Max chars | Character headroom | Observed token headroom | Estimated token headroom | Max naive clip | Reviewed routing delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| relevance | conditional | 0/496 | 0/496 | 24181 | 45819 | N/A | N/A | 0 | N/A |
| grouping | safe | 0/21139 | 0/21139 | 24753 | 45247 | 26641 | 11876 | 0 | 0.000000 |
| ranking | conditional | 190/8367 | 190/8367 | 140493 | -70493 | 13931 | -28927 | 70493 | 0.000000 |
| tier | unsafe | 394/456 | 197/228 | 110825 | -40825 | 17631 | -1736 | 40825 | 0.000000 |
| confidence | safe | 0/324 | 0/324 | 62656 | 7344 | 29131 | 3232 | 0 | 0.000000 |
| daily_theme | safe | 0/8367 | 0/8367 | 2165 | 67835 | 31153 | 30752 | 0 | 0.000000 |

## Decision

- `safe` means context-safe only with the local guard and full-state incumbent fallback.
- `conditional` means coverage or calibration is incomplete, or sampled requests need fallback.
- `unsafe` means Jev is not the production default. Tier always stays on the incumbent.
- No role clips, chunks, summarizes, or drops evidence. A local refusal or provider rejection routes the unchanged full state to the incumbent.
- Do not retry an unchanged `max_tokens_exceeded` request.

## Truncation evidence

The 70,000-character guard affects no reviewed request, so it changes zero reviewed routing decisions for grouping, ranking, tier, confidence, and daily theme. Relevance has no reviewed headroom sample. Quality on affected production ranking and tier requests is not estimable because the reviewed corpus has no matching over-guard states.

Naive clipping would remove as many as 70,493 ranking characters and 40,825 tier characters. Character loss is not a measure of semantic evidence loss, so clipping is rejected.

## Tier batching

One exact tier state used 11482 input tokens in two separate requests and 5841 in one batched request. The measured reduction was 49.1291%. Both answer IDs returned. The probe did not evaluate quality equivalence, and batching does not reduce state size.
