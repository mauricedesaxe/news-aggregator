# Jev research snapshot

This snapshot preserves the completed cross-product evidence. It does not approve a blanket production migration.

## Model identity and architecture

| Model | News provider | Job provider | News cost basis | Job cost basis | Comparative roles |
| --- | --- | --- | --- | --- | --- |
| jev-1.13.0 | typesafe | jev | estimated-input-rate | rate_card | job_relevance, news_relevance, grouping, ranking, tier, confidence, daily_theme |
| google/gemini-2.5-flash | openrouter | openrouter | provider-reported | provider_reported | job_relevance, news_relevance, grouping, ranking, tier, confidence, daily_theme |
| google/gemini-3.8-flash | openrouter | openrouter | provider-reported | provider_reported | job_relevance, news_relevance, grouping, ranking, tier, confidence, daily_theme |
| openai/gpt-4.1-mini | openai | N/A | production model-call accounting | N/A | none |

Jev documents a 64,000-token request budget and a 32,000-token state-plus-longest-question budget. The local 70,000-character guard is empirical and is not a tokenizer bound.

## Corpus and task comparability

| Product | Concern | Relationship | Directly tested | Complete output tested | Limitation |
| --- | --- | --- | --- | --- | --- |
| job_finder | relevance | current_six_criterion_policy | yes | yes | The proposed atomic architecture was not evaluated. |
| news | relevance | shared_monolithic_prompt_model_isolation | yes | no | The benchmark isolates one shared relevance question; trials reuse 174 cases. |
| news | grouping | derived_binary_decomposition | no | no | Pair labels do not establish a complete partition. |
| news | ranking | derived_binary_decomposition | no | no | Pair precedence does not establish a coherent global order. |
| news | tier | derived_binary_decomposition | no | no | Two binary questions do not evaluate complete assessment generation. |
| news | confidence | derived_binary_decomposition | no | no | Only four evidence-sufficiency labels exist. |
| news | daily_theme | derived_binary_decomposition | no | no | Pair labels do not establish transitivity, partition validity, or report usefulness. |
| news | sentiment | not_evaluable | no | no | No immutable human-reviewed article or group sentiment labels exist. |

## Product relevance

| Product | Jev status | Cases | Trials | Gate result |
| --- | --- | ---: | ---: | --- |
| news | eligible | 174 | 3 | pass |
| job_finder | not eligible | 146 | 3 | fail |

### News relevance trials

| Model | Correct by trial | Precision by trial (95% exact CI) | Recall by trial (95% exact CI) | Requests | Tokens in/out | Cost | Median trial p95 | Eligible Pareto |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| typesafe-jev | 146/174, 147/174, 146/174 | 81.46% [74.33%, 87.31%], 82.00% [74.90%, 87.79%], 81.46% [74.33%, 87.31%] | 100.00% [97.05%, 100.00%], 100.00% [97.05%, 100.00%], 100.00% [97.05%, 100.00%] | 522 | 1095921/11484 | $0.046028682 | 824.7 | yes |
| openrouter-gemini-2.5-flash | 142/174, 141/174, 140/174 | 80.95% [73.66%, 86.95%], 80.82% [73.49%, 86.86%], 80.69% [73.31%, 86.77%] | 96.75% [91.88%, 99.11%], 95.93% [90.77%, 98.67%], 95.12% [89.68%, 98.19%] | 522 | 865962/2610 | $0.266313600 | 857 | no |
| openrouter-gemini-3.8-flash | 153/174, 152/174, 152/174 | 90.48% [83.95%, 94.98%], 90.40% [83.83%, 94.94%], 90.40% [83.83%, 94.94%] | 92.68% [86.56%, 96.60%], 91.87% [85.56%, 96.03%], 91.87% [85.56%, 96.03%] | 522 | 884232/368266 | $2.044171500 | 5485.85 | no |

### Job relevance trials

| Model | Suite | Correct by trial | FP by trial (95% exact CI) | FN by trial (95% exact CI) | Requests | Tokens in/out | Cost | Aggregate p95 | Gate |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| jev-1.13.0 | direct | 106/121, 104/121, 106/121 | 10 [7.63%, 26.48%], 11 [8.76%, 28.27%], 10 [7.63%, 26.48%] | 5 [2.96%, 19.62%], 6 [4.03%, 21.88%], 5 [2.96%, 19.62%] | 1530 | 2885171/36612 | $0.121177182 | 4300.90 | fail |
| jev-1.13.0 | ats | 23/25, 23/25, 23/25 | 1 [0.16%, 30.23%], 1 [0.16%, 30.23%], 1 [0.16%, 30.23%] | 1 [0.28%, 48.25%], 1 [0.28%, 48.25%], 1 [0.28%, 48.25%] | 330 | 564960/7821 | $0.023728320 | 3969.80 | pass |
| google-gemini-2.5-flash | direct | 106/121, 106/121, 104/121 | 10 [7.63%, 26.48%], 10 [7.63%, 26.48%], 12 [9.92%, 30.03%] | 5 [2.96%, 19.62%], 5 [2.96%, 19.62%], 5 [2.96%, 19.62%] | 1538 | 2666834/7690 | $0.805769530 | 4640.00 | fail |
| google-gemini-2.5-flash | ats | 23/25, 24/25, 24/25 | 1 [0.16%, 30.23%], 0 [0.00%, 20.59%], 0 [0.00%, 20.59%] | 1 [0.28%, 48.25%], 1 [0.28%, 48.25%], 1 [0.28%, 48.25%] | 326 | 497434/1630 | $0.148778650 | 4117.10 | pass |
| google-gemini-3.8-flash | direct | 104/121, 105/121, 106/121 | 10 [7.63%, 26.48%], 10 [7.63%, 26.48%], 9 [6.53%, 24.66%] | 7 [5.18%, 24.07%], 6 [4.03%, 21.88%], 6 [4.03%, 21.88%] | 1569 | 2760678/602624 | $4.330348500 | 32881.20 | fail |
| google-gemini-3.8-flash | ats | 24/25, 24/25, 24/25 | 1 [0.16%, 30.23%], 1 [0.16%, 30.23%], 1 [0.16%, 30.23%] | 0 [0.00%, 33.63%], 0 [0.00%, 33.63%], 0 [0.00%, 33.63%] | 339 | 525600/121346 | $0.849247500 | 28802.60 | pass |

## News aspects

| Concern | Evidence | Context | Deployment conclusion | Limitation |
| --- | --- | --- | --- | --- |
| grouping | derived_task_only | safe | not_established | Pair judgments do not establish a valid complete partition. |
| ranking | derived_task_only | conditional | not_established | Pairwise precedence does not establish a coherent global order. |
| tier | derived_task_only | unsafe | not_established | Composed tier classification does not evaluate complete assessment generation. |
| confidence | descriptive_only | safe | not_established | Four cases support descriptive evidence only, not a deployment winner. |
| daily_theme | derived_task_only | safe | not_established | Pair judgments do not establish transitivity or report usefulness. |
| sentiment | not_evaluable | N/A | retain incumbent | No immutable human-reviewed article or group sentiment class labels exist. |

### Derived aspect model evidence

| Concern | Model | Correct by trial (95% exact CI) | Failures | Tokens in/out | Cost | Cost/1,000 scored units | Median trial p50/p95 | Stability | Pareto |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grouping | typesafe-jev | 21/23 [71.96%, 98.93%], 21/23 [71.96%, 98.93%], 20/23 [66.41%, 97.22%] | 0 | 191712/1518 | $0.008051904 | $0.116694261 | 646/839.8 | 22/23 | yes |
| grouping | openrouter-gemini-2.5-flash | 21/23 [71.96%, 98.93%], 21/23 [71.96%, 98.93%], 21/23 [71.96%, 98.93%] | 0 | 154449/345 | $0.045756210 | $0.663133478 | 670/906.5 | 23/23 | yes |
| grouping | openrouter-gemini-3.8-flash | 19/23 [61.22%, 95.05%], 19/23 [61.22%, 95.05%], 20/23 [66.41%, 97.22%] | 0 | 156864/29875 | $0.229679250 | $3.328684783 | 1658/4695.3 | 22/23 | no |
| ranking | typesafe-jev | 21/28 [55.13%, 89.31%], 22/28 [59.05%, 91.70%], 20/28 [51.33%, 86.78%] | 0 | 664227/2100 | $0.027897534 | $0.332113500 | 816.5/1057.3 | 26/28 | yes |
| ranking | openrouter-gemini-2.5-flash | 12/28 [24.46%, 62.82%], 12/28 [24.46%, 62.82%], 12/28 [24.46%, 62.82%] | 0 | 574677/420 | $0.157518780 | $1.875223571 | 678.5/1174.2 | 28/28 | no |
| ranking | openrouter-gemini-3.8-flash | 27/28 [81.65%, 99.91%], 27/28 [81.65%, 99.91%], 25/28 [71.77%, 97.73%] | 0 | 577617/48280 | $0.520667250 | $6.198419643 | 2467.5/4863.3 | 24/28 | yes |
| tier | typesafe-jev | 23/50 [31.81%, 60.68%], 23/50 [31.81%, 60.68%], 23/50 [31.81%, 60.68%] | 0 | 2154102/3525 | $0.090472284 | $0.603148560 | 1015.5/1083.1 | 50/50 | yes |
| tier | openrouter-gemini-2.5-flash | 25/50 [35.53%, 64.47%], 25/50 [35.53%, 64.47%], 25/50 [35.53%, 64.47%] | 0 | 2107539/750 | $0.390069390 | $2.600462600 | 769.5/1420.35 | 50/50 | yes |
| tier | openrouter-gemini-3.8-flash | 30/50 [45.18%, 73.59%], 27/50 [39.32%, 68.19%], 29/50 [43.21%, 71.81%] | 0 | 2112789/107792 | $1.663521150 | $11.090141000 | 3238/5005.5 | 47/50 | yes |
| confidence | typesafe-jev | 4/4 [39.76%, 100.00%], 4/4 [39.76%, 100.00%], 4/4 [39.76%, 100.00%] | 0 | 25329/312 | $0.001063818 | $0.088651500 | 625/745.6 | 4/4 | yes |
| confidence | openrouter-gemini-2.5-flash | 3/4 [19.41%, 99.37%], 3/4 [19.41%, 99.37%], 3/4 [19.41%, 99.37%] | 0 | 19530/60 | $0.005543520 | $0.461960000 | 562.5/844.35 | 4/4 | no |
| confidence | openrouter-gemini-3.8-flash | 4/4 [39.76%, 100.00%], 4/4 [39.76%, 100.00%], 4/4 [39.76%, 100.00%] | 0 | 19950/9718 | $0.051405000 | $4.283750000 | 3492/3816.65 | 4/4 | no |
| daily_theme | typesafe-jev | 20/27 [53.72%, 88.89%], 20/27 [53.72%, 88.89%], 20/27 [53.72%, 88.89%] | 0 | 60717/1782 | $0.002550114 | $0.031482889 | 622/694.8 | 27/27 | yes |
| daily_theme | openrouter-gemini-2.5-flash | 21/27 [57.74%, 91.38%], 20/27 [53.72%, 88.89%], 20/27 [53.72%, 88.89%] | 0 | 40665/405 | $0.013212000 | $0.163111111 | 551/916.8 | 24/27 | no |
| daily_theme | openrouter-gemini-3.8-flash | 17/27 [42.37%, 80.60%], 17/27 [42.37%, 80.60%], 17/27 [42.37%, 80.60%] | 0 | 43500/25933 | $0.129873750 | $1.603379630 | 1926/3125.4 | 27/27 | no |

### Paired significance and recommendations

| Concern | Paired comparisons | p < 0.05 | Minimum p | Recommended use |
| --- | ---: | ---: | ---: | --- |
| news/relevance | 9 | 1 | 0.035698138 | Jev is eligible for guarded relevance use under the registered quality gate; retain full-state incumbent fallback above the context guard or on rejection. |
| job_finder/relevance | 18 | 0 | 0.774414062 | Retain the current Job relevance implementation; no target passed the registered direct-suite gate, and the proposed atomic architecture is untested. |
| news/grouping | 9 | 0 | 0.500000000 | Use only as guarded pairwise evidence or in shadow evaluation; complete partition replacement is not established. |
| news/ranking | 9 | 6 | 0.000244141 | Use only as guarded pairwise evidence with incumbent fallback; coherent global ordering and over-guard quality are not established. |
| news/tier | 9 | 0 | 0.118469238 | Retain the incumbent; composed quality is weak and production context is unsafe. |
| news/confidence | 9 | 0 | 1.000000000 | Retain the incumbent until more than four immutable labeled cases exist. |
| news/daily_theme | 9 | 0 | 0.125000000 | Do not replace complete theme generation from pair judgments; transitivity, partition validity, and report usefulness are untested. |
| news/sentiment | 0 | 0 | N/A | Retain the current sentiment implementation. Collect blinded human labels before comparing or replacing its classification component. |

## Context routing

The operational guard is `len(state) <= 70,000`. It is not a formal tokenizer bound. Over-guard or provider-rejected requests retain the full state and use the incumbent; evidence is never clipped or chunked.

| Role | Status | Affected question requests | Affected provider requests |
| --- | --- | ---: | ---: |
| relevance | conditional | 0/496 | 0/496 |
| grouping | safe | 0/21139 | 0/21139 |
| ranking | conditional | 190/8367 | 190/8367 |
| tier | unsafe | 394/456 | 197/228 |
| confidence | safe | 0/324 | 0/324 |
| daily_theme | safe | 0/8367 | 0/8367 |

## Current News cost baseline

- Exact immutable reports: 7
- Attributed cost: $2.032867430
- Unique physical cost: $2.032867430
- Accounting gaps: 0

## Research backlog

- `job_atomic_policy`: Run frozen atomic questions plus deterministic composition on disjoint labeled direct and ATS cases. Completion: A preregistered target passes every direct and ATS error gate in every trial.
- `news_sentiment_labels`: Collect blinded, adjudicated four-class labels tied to immutable article and group versions. Completion: A frozen disjoint test set has enough labels to report per-class recall and exact intervals.
- `news_complete_structures`: Evaluate grouping partitions, coherent global rankings, and complete tier assessments rather than pairwise surrogates. Completion: Registered end-to-end structure metrics pass on disjoint immutable reports.
- `over_guard_routing_quality`: Label production-shaped ranking and tier cases above the 70,000-character guard and compare fallback outcomes. Completion: Quality deltas and exact intervals exist for the 190 ranking and 197 tier provider-request strata represented by the inventory.
- `tier_batch_quality`: Compare separate and batched tier answers on a preregistered multi-case sample. Completion: Equivalence bounds are reported; the existing one-sample 49.1291% input reduction remains transport evidence only.
- `confidence_and_theme_coverage`: Expand confidence labels and evaluate daily-theme transitivity, partition validity, and report usefulness. Completion: Confidence exceeds four cases and daily themes pass registered end-to-end usefulness metrics.

## Sources

- `artifacts/jev-research-v1/imports/manifest.json`: `60d3295d43b075a06dc888b08cc22032f26af06f5cba9a23d4b92439ea2ce07f`
- `artifacts/jev-research-v1/imports/job-current-policy-v1-preregistration.json.gz`: `aa6a5aca52d1798725437f08f3f85053c7d84144a8cfa3c2bf9629d70016ddbd`
- `artifacts/jev-research-v1/imports/job-current-policy-v1-raw.json.gz`: `e2190975349d12897a9b180d91b8d600484fe6676ceb7bf63b2acc279d39e84d`
- `artifacts/jev-research-v1/imports/job-current-policy-v1-report.json.gz`: `10b77f3110d853302cc2ab2bea0e03219341dafdad925455dc26b63ed91a3836`
- `artifacts/jev-research-v1/imports/job-current-policy-v1-report.md.gz`: `2c25715bd854508174a67cec8ae016469bbd8dfc2fe572d7127242f0b3d77d54`
- `artifacts/jev-research-v1/imports/news-relevance-v11-raw.json.gz`: `8d76c3b01b6cec5726c70159c82f5093ec3cfca36811d10f700b19adea985605`
- `artifacts/jev-research-v1/imports/news-relevance-v11-report.json.gz`: `2a2afa034d98806d478918921626ddef264cb7a9e35494eed263f76230e50cfd`
- `artifacts/jev-research-v1/imports/news-relevance-v11-report.md.gz`: `9289221a722a2291b000e5ad9cf54c95a811f61005300e28851febf6af1df45a`
- `src/romanian_news/binary_relevance_preregistration.json`: `580b0710c3118175ae843bf0612da1d943b8721205bac7e10262ee335aee645a`
- `src/romanian_news/jev_aspect_evaluation_protocol.json`: `4df85e8d3395f28da8c9e06df70aa78f2fe373529e2064daaca951db4e77aedb`
- `artifacts/jev-aspect-v1/news-grouping-live-v1-report.json`: `d3c159d5a2d36a7497b8fa0757c8b3ac80ca1a87c612f21e6186cf67e238ab06`
- `artifacts/jev-aspect-v1/news-ranking-live-v1-report.json`: `e6a434efe3a98c4f15eb7d8e1e466403b55c6103f864ea3d093cc9191f0452b3`
- `artifacts/jev-aspect-v1/news-tier-live-v1-report.json`: `0b5299071c9576933b2c8d31e7482e800e03807d4a3f59b0c5c344ffab6cf755`
- `artifacts/jev-aspect-v1/news-tier-live-v1-composed.json`: `5fe540fea32975fcca6a3dfa603ed2b7007bf853d0dcf1b130528b73a19bb0c9`
- `artifacts/jev-aspect-v1/news-confidence-live-v1-descriptive.json`: `b7afdde51e6aa7072bebafcf07eb9932452561572bd250611d1451b70ad2f6d0`
- `artifacts/jev-aspect-v1/news-daily-theme-live-v1-report.json`: `ae4fc99d1db978b92d90b948eebe2ee427258977befee428c52ef85511db4c5f`
- `artifacts/jev-aspect-v1/news-daily-theme-live-v1-constraints.json`: `87005952429f4c7a5557f11b5b949c29e0caa29509b65cc72f3590d6a92c2371`
- `src/romanian_news/sentiment_evaluation_result.json`: `298adf5c391b50ed3e845a81d4f5bf8cd5d674e451a18ca5e813509df3478baf`
- `artifacts/jev-aspect-v1/jev-context-safety-decision-v1.json`: `400d154709c17c63659512d4ad077bb035d4cb93f708501a435fb54a45ee0549`
- `artifacts/jev-research-v1/news-report-model-cost-v1.json`: `6c257bb6cf25c13af2a39d6f5cc3a253a6d3e848a4d77ade9582f09b18aa7c39`
