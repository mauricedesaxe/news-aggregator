# Jev production context inventory

This read-only inventory rebuilds proposed single-question states from captured exact daily report versions. It stores no article text, state text, provider output, or label.

| Concern | Scope | Requests | p50 chars | p95 chars | p99 chars | Max chars | >=50% | >=75% | >=90% | >=100% | Estimated max tokens | Estimated >=32k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| relevance | accepted cluster articles only | 496 | 2556 | 11219 | 16256 | 24181 | 0 (0.0000%) | 0 (0.0000%) | 0 (0.0000%) | 0 (0.0000%) | null | null |
| grouping | all unordered accepted cluster article pairs | 21139 | 6263 | 15024 | 19868 | 24753 | 0 (0.0000%) | 0 (0.0000%) | 0 (0.0000%) | 0 (0.0000%) | 20124 | 0 |
| ranking | all unordered cluster group pairs with exact relevance evidence | 8367 | 11128 | 40688 | 82651 | 140493 | 162 (1.9362%) | 2 (0.0239%) | 1 (0.0120%) | 0 (0.0000%) | 60927 | 164 |
| tier | two registered questions per DailyReport v3 subject | 456 | 80178 | 110825 | 110825 | 110825 | 324 (71.0526%) | 0 (0.0000%) | 0 (0.0000%) | 0 (0.0000%) | 33736 | 98 |
| confidence | each cluster group | 324 | 3898 | 14867 | 42215 | 62656 | 0 (0.0000%) | 0 (0.0000%) | 0 (0.0000%) | 0 (0.0000%) | 28768 | 0 |
| daily_theme | all unordered cluster group pairs | 8367 | 1629 | 1863 | 1957 | 2165 | 0 (0.0000%) | 0 (0.0000%) | 0 (0.0000%) | 0 (0.0000%) | 1248 | 0 |

## Evidence gaps

- [relevance/accepted-only] Relevance covers accepted cluster articles only because broader rejected day articles cannot be pinned from report lineage.
- [all/observed-estimate-not-bound] Estimated input tokens use the maximum reviewed input_tokens/state_characters ratio. This is a conservative observed estimate, not a formal tokenizer bound.
- [relevance/no-reviewed-calibration] Relevance has no reviewed character calibration, so estimates are null.
