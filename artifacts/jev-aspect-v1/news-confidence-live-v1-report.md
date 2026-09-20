# confidence binary benchmark analysis

- Identity: `8b3b14b58d68d35364ee6c7063f64efbaaa2185a9de8230ab066feaa74262edc`
- Source artifact: `083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a`
- Shape: 3 targets x 3 trials x 4 cases
- Limitation: Four cases support descriptive evidence only, not a deployment winner.

## Trial results

| Target | Trial | Completed | Failed | Correct | Accuracy (95% exact CI) | Attempts | Cost USD | p50/p95 ms |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-001` | 4 | 0 | 3 | 0.7500 (0.1941-0.9937) | 4 | 0.002003000 | 562.5/844.3 |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-002` | 4 | 0 | 3 | 0.7500 (0.1941-0.9937) | 4 | 0.002003000 | 522.5/682.6 |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-003` | 4 | 0 | 3 | 0.7500 (0.1941-0.9937) | 4 | 0.001537520 | 630.0/1562.6 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-001` | 4 | 0 | 4 | 1.0000 (0.3976-1.0000) | 4 | 0.015768750 | 3107.0/3663.1 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-002` | 4 | 0 | 4 | 1.0000 (0.3976-1.0000) | 4 | 0.017842500 | 3696.5/4222.0 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-003` | 4 | 0 | 4 | 1.0000 (0.3976-1.0000) | 4 | 0.017793750 | 3492.0/3816.6 |
| `typesafe-jev` | `jev-aspect-v1:trial-001` | 4 | 0 | 4 | 1.0000 (0.3976-1.0000) | 4 | 0.000354606 | 690.0/749.2 |
| `typesafe-jev` | `jev-aspect-v1:trial-002` | 4 | 0 | 4 | 1.0000 (0.3976-1.0000) | 4 | 0.000354606 | 625.0/659.0 |
| `typesafe-jev` | `jev-aspect-v1:trial-003` | 4 | 0 | 4 | 1.0000 (0.3976-1.0000) | 4 | 0.000354606 | 619.0/745.6 |

## Paired comparisons

- `jev-aspect-v1:trial-001` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 4 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-001` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 4 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-001` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 4 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-002` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 4 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-002` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 4 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-002` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 4 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-003` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 4 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-003` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 4 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-003` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 4 paired, 0 unavailable, exact McNemar p=1.

## Stability

- `openrouter-gemini-2.5-flash`: 1.0000 unanimous, 0 adjacent-trial flips, 0 unavailable.
- `openrouter-gemini-3.8-flash`: 1.0000 unanimous, 0 adjacent-trial flips, 0 unavailable.
- `typesafe-jev`: 1.0000 unanimous, 0 adjacent-trial flips, 0 unavailable.

## Pareto

- Non-dominated targets: `typesafe-jev`.
