# tier binary benchmark analysis

- Identity: `7dcadb5b33c63acd20cb89dc6803d1bbc07f1d7db2abcde81d4e7dcdf50b1215`
- Source artifact: `083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a`
- Shape: 3 targets x 3 trials x 25 cases
- Limitation: Composed tier classification does not evaluate complete assessment generation.

## Trial results

| Target | Trial | Completed | Failed | Correct | Accuracy (95% exact CI) | Attempts | Cost USD | p50/p95 ms |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-001` | 50 | 0 | 25 | 0.5000 (0.3553-0.6447) | 50 | 0.172220800 | 787.0/1420.3 |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-002` | 50 | 0 | 25 | 0.5000 (0.3553-0.6447) | 50 | 0.138427060 | 769.5/1091.3 |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-003` | 50 | 0 | 25 | 0.5000 (0.3553-0.6447) | 50 | 0.079421530 | 764.5/1896.5 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-001` | 50 | 0 | 30 | 0.6000 (0.4518-0.7359) | 50 | 0.635071500 | 3159.5/4711.2 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-002` | 50 | 0 | 27 | 0.5400 (0.3932-0.6819) | 50 | 0.399390300 | 3587.5/5634.6 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-003` | 50 | 0 | 29 | 0.5800 (0.4321-0.7181) | 50 | 0.629059350 | 3238.0/5005.5 |
| `typesafe-jev` | `jev-aspect-v1:trial-001` | 50 | 0 | 23 | 0.4600 (0.3181-0.6068) | 50 | 0.030157428 | 1001.0/1083.1 |
| `typesafe-jev` | `jev-aspect-v1:trial-002` | 50 | 0 | 23 | 0.4600 (0.3181-0.6068) | 50 | 0.030157428 | 1015.5/1080.7 |
| `typesafe-jev` | `jev-aspect-v1:trial-003` | 50 | 0 | 23 | 0.4600 (0.3181-0.6068) | 50 | 0.030157428 | 1024.5/1100.1 |

## Paired comparisons

- `jev-aspect-v1:trial-001` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 50 paired, 0 unavailable, exact McNemar p=0.404873.
- `jev-aspect-v1:trial-001` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 50 paired, 0 unavailable, exact McNemar p=0.774414.
- `jev-aspect-v1:trial-001` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 50 paired, 0 unavailable, exact McNemar p=0.118469.
- `jev-aspect-v1:trial-002` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 50 paired, 0 unavailable, exact McNemar p=0.831812.
- `jev-aspect-v1:trial-002` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 50 paired, 0 unavailable, exact McNemar p=0.774414.
- `jev-aspect-v1:trial-002` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 50 paired, 0 unavailable, exact McNemar p=0.42395.
- `jev-aspect-v1:trial-003` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 50 paired, 0 unavailable, exact McNemar p=0.541256.
- `jev-aspect-v1:trial-003` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 50 paired, 0 unavailable, exact McNemar p=0.774414.
- `jev-aspect-v1:trial-003` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 50 paired, 0 unavailable, exact McNemar p=0.210114.

## Stability

- `openrouter-gemini-2.5-flash`: 1.0000 unanimous, 0 adjacent-trial flips, 0 unavailable.
- `openrouter-gemini-3.8-flash`: 0.9400 unanimous, 5 adjacent-trial flips, 0 unavailable.
- `typesafe-jev`: 1.0000 unanimous, 0 adjacent-trial flips, 0 unavailable.

## Pareto

- Non-dominated targets: `openrouter-gemini-2.5-flash`, `openrouter-gemini-3.8-flash`, `typesafe-jev`.
