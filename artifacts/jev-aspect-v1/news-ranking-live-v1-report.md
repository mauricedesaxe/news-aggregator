# ranking binary benchmark analysis

- Identity: `0148359e774e6b000f62f8993daa179e56d7877bd84873f4b5c3437effd38f2c`
- Source artifact: `083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a`
- Shape: 3 targets x 3 trials x 28 cases
- Limitation: Pairwise precedence does not establish a coherent global order.

## Trial results

| Target | Trial | Completed | Failed | Correct | Accuracy (95% exact CI) | Attempts | Cost USD | p50/p95 ms |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-001` | 28 | 0 | 12 | 0.4286 (0.2446-0.6282) | 28 | 0.057817700 | 678.5/1227.4 |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-002` | 28 | 0 | 12 | 0.4286 (0.2446-0.6282) | 28 | 0.054137870 | 686.5/1071.8 |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-003` | 28 | 0 | 12 | 0.4286 (0.2446-0.6282) | 28 | 0.045563210 | 677.0/1174.2 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-001` | 28 | 0 | 27 | 0.9643 (0.8165-0.9991) | 28 | 0.204284250 | 2468.0/4863.3 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-002` | 28 | 0 | 27 | 0.9643 (0.8165-0.9991) | 28 | 0.161730300 | 2467.5/5418.9 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-003` | 28 | 0 | 25 | 0.8929 (0.7177-0.9773) | 28 | 0.154652700 | 2392.5/4673.8 |
| `typesafe-jev` | `jev-aspect-v1:trial-001` | 28 | 0 | 21 | 0.7500 (0.5513-0.8931) | 28 | 0.009299178 | 831.5/1048.9 |
| `typesafe-jev` | `jev-aspect-v1:trial-002` | 28 | 0 | 22 | 0.7857 (0.5905-0.9170) | 28 | 0.009299178 | 816.5/1057.3 |
| `typesafe-jev` | `jev-aspect-v1:trial-003` | 28 | 0 | 20 | 0.7143 (0.5133-0.8678) | 28 | 0.009299178 | 799.5/1095.6 |

## Paired comparisons

- `jev-aspect-v1:trial-001` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 28 paired, 0 unavailable, exact McNemar p=0.000274658.
- `jev-aspect-v1:trial-001` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 28 paired, 0 unavailable, exact McNemar p=0.0224609.
- `jev-aspect-v1:trial-001` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 28 paired, 0 unavailable, exact McNemar p=0.0703125.
- `jev-aspect-v1:trial-002` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 28 paired, 0 unavailable, exact McNemar p=0.000274658.
- `jev-aspect-v1:trial-002` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 28 paired, 0 unavailable, exact McNemar p=0.0129395.
- `jev-aspect-v1:trial-002` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 28 paired, 0 unavailable, exact McNemar p=0.125.
- `jev-aspect-v1:trial-003` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 28 paired, 0 unavailable, exact McNemar p=0.000244141.
- `jev-aspect-v1:trial-003` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 28 paired, 0 unavailable, exact McNemar p=0.0385742.
- `jev-aspect-v1:trial-003` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 28 paired, 0 unavailable, exact McNemar p=0.125.

## Stability

- `openrouter-gemini-2.5-flash`: 1.0000 unanimous, 0 adjacent-trial flips, 0 unavailable.
- `openrouter-gemini-3.8-flash`: 0.8571 unanimous, 4 adjacent-trial flips, 0 unavailable.
- `typesafe-jev`: 0.9286 unanimous, 3 adjacent-trial flips, 0 unavailable.

## Pareto

- Non-dominated targets: `openrouter-gemini-3.8-flash`, `typesafe-jev`.
