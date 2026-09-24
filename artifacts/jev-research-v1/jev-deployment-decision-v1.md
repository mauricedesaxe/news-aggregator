# Jev deployment decision

This artifact records prompt-class dispositions. It does not change production behavior.

| Product | Aspect | Prompt class | Disposition | Implementation | Basis |
| --- | --- | --- | --- | --- | --- |
| news | relevance | direct_binary_classification | guarded_shadow_candidate | news-gh9 | Jev alone passed the registered News relevance quality gate; production relevance states were below the local guard, but reviewed context headroom and affected-production quality remain unavailable. |
| job_finder | relevance | multi_criterion_binary_classification | retain_incumbent | N/A | Every tested target failed the registered direct-suite gate; the proposed atomic architecture is untested. |
| news | grouping | pairwise_relation_judgment | research_only | N/A | Pair judgments do not establish a complete partition. |
| news | ranking | pairwise_relation_judgment | research_only | N/A | Pair precedence does not establish a coherent global order, and 190 production requests exceed the guard. |
| news | tier | deterministic_composition_of_binary_questions | retain_incumbent | N/A | Composed exact accuracy is weak, complete assessment generation is untested, and production context is unsafe. |
| news | confidence | direct_binary_classification | require_more_evidence | N/A | Only four immutable labeled cases exist. |
| news | daily_theme | pairwise_relation_judgment | research_only | N/A | Pair judgments do not establish transitivity, partition validity, or report usefulness. |
| news | sentiment | four_class_classification | retain_incumbent | N/A | No immutable human-reviewed article or group sentiment labels exist. |
| cross_product | structured_extraction | structured_extraction | retain_incumbent | N/A | No comparable labeled structured-extraction evaluation was completed. |
| cross_product | prose_generation | prose_generation | retain_incumbent | N/A | No comparable human-reviewed prose quality evaluation was completed. |

## Guarded rollout

- Ticket: `news-gh9` (`gh-9`)
- Mode: `shadow_only_incumbent_authoritative`
- Preflight: send to Jev only when len(state) <= 70000
- Fallback: use incumbent with unchanged full state on guard breach or provider rejection
- Promotion gate: A separately reviewed immutable sample preserves every positive control, has recall 1.0 and precision >= 0.80, with no unresolved accounting gaps.
- Rollback: configuration-only return to the incumbent, exercised before promotion
- Evidence is never clipped or chunked, and an unchanged rejected request is never retried.

## Source

- `artifacts/jev-research-v1/jev-research-snapshot-v1.json`: `8a889bef56d63c071803fe476041a0c5c4414bafc10e7bc6e474d059208e779f`
