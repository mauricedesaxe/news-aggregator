# Split subject-assessment experiment analysis

- Execution: `split-assessment-v1` (live)
- Identity: `66d68602c8ab05d1cc3d23e9149be6ac054eeb9a106a463d57d953e9ea6b97f8`
- Execution complete: yes
- Promotion status: **fail**

## Architecture summary

| Architecture | Terminal | Completed | Failed | Unavailable | Quality | Tier | Ranking | Controls | Regressions | Structural | Wall p50/p95 ms | Whole cost USD | Accounting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `incumbent` | 3 | 3 | 0 | 0 | 95/114 | 56/75 | 39/39 | 29/39 | 0 | 3/3 | 11065.0/11594.2 | 0.066349500 | complete known provider cost |
| `candidate` | 3 | 3 | 0 | 0 | 72/114 | 39/75 | 33/39 | 15/39 | 32 | 3/3 | 35923.0/36487.3 | 0.116908638 | lower bound; accounting is incomplete |

## Paired trials

| Trial | Incumbent status | Incumbent quality | Candidate status | Candidate quality | Pass delta | Tier delta | Ranking delta | Control regressions | Wall ms I/C |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `split-assessment-v1:trial-001` | completed | all 32/38; tier 19/25; rank 13/13; controls 10/13 | completed | all 23/38; tier 13/25; rank 10/13; controls 5/13 | -9 | -6 | -3 | 6 | 11065/35754 |
| `split-assessment-v1:trial-002` | completed | all 31/38; tier 18/25; rank 13/13; controls 10/13 | completed | all 26/38; tier 13/25; rank 13/13; controls 5/13 | -5 | -5 | +0 | 6 | 6527/35923 |
| `split-assessment-v1:trial-003` | completed | all 32/38; tier 19/25; rank 13/13; controls 9/13 | completed | all 23/38; tier 13/25; rank 10/13; controls 5/13 | -9 | -6 | -3 | 5 | 11653/36550 |

## Quality diagnostics

| Architecture | Per-tier recall (main/worth/excluded) | Ranking inversions | Ranking unavailable | Merged anchor pairs | Semantic ties | Merged tier resolutions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `incumbent` | 29/39 (0.744)/20/27 (0.741)/7/9 (0.778) | 0 | 0 | 0 | 0 | 0 |
| `candidate` | 15/39 (0.385)/21/27 (0.778)/3/9 (0.333) | 6 | 0 | 0 | 0 | 0 |

## Requests and cost

| Architecture | Requests | Attempts | Accepted/rejected/retryable/terminal | Retries | Corrections | Failures | Attempt latency ms | Tier/ranking/incumbent cost USD | Unknown in-flight |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `incumbent` | 3 | 5 | 3/2/0/0 | 0 | 2 | 2 | 29200 | 0/0.066349500/0 | 0 |
| `candidate` | 120 | 120 | 120/0/0/0 | 0 | 0 | 0 | 143931 | 0.070453488/0.04645515/0 | 1 |

## Promotion gates

- **pass** `zero_invalid_or_incomplete_candidate_assessments`. candidate completed and structurally valid in 3/3 trials
- **fail** `zero_control_case_regressions`. candidate control regressions: 17
- **fail** `candidate_tier_and_ranking_passes_at_least_incumbent`. candidate versus incumbent aggregate tier/ranking passes: 39/56 and 33/39
- **fail** `no_omitted_provider_attempt_or_accounting_gap`. lower bound; accounting is incomplete
- **fail** `complete_run_latency_and_cost_include_failed_and_unavailable_arms`. execution or in-flight latency evidence is incomplete

## Uncertainty

Intervals describe run-to-run uncertainty across only three frozen pairs; they are not evidence of population-level statistical power. Negative latency and cost deltas favor the candidate.

- `quality_pass_delta_per_trial`: mean -7.666666666667; 95% interval [-9.0, -5.866666666667]
- `tier_pass_delta_per_trial`: mean -5.666666666667; 95% interval [-6.0, -5.216666666667]
- `ranking_pass_delta_per_trial`: mean -2.0; 95% interval [-3.0, -0.65]
- `wall_latency_delta_ms_per_trial`: mean 26327.333333333332; 95% interval [24734.066666666666, 28421.216666666664]
- `known_cost_delta_usd_per_trial`: mean 0.005686672; 95% interval [-0.00160812175, 0.014204147]

## Accounting

lower bound; accounting is incomplete
