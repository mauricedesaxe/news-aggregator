# daily_theme binary benchmark analysis

- Identity: `d2a768664653003e58ad42c65dc4e2264e0110e1522520ce8ddd60b5d75e97c6`
- Source artifact: `083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a`
- Shape: 3 targets x 3 trials x 6 cases
- Limitation: Pair judgments do not establish transitivity or report usefulness.

## Trial results

| Target | Trial | Completed | Failed | Correct | Accuracy (95% exact CI) | Attempts | Cost USD | p50/p95 ms |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-001` | 27 | 0 | 21 | 0.7778 (0.5774-0.9138) | 27 | 0.004404000 | 551.0/916.8 |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-002` | 27 | 0 | 20 | 0.7407 (0.5372-0.8889) | 27 | 0.004404000 | 526.0/979.3 |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-003` | 27 | 0 | 20 | 0.7407 (0.5372-0.8889) | 27 | 0.004404000 | 561.0/902.3 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-001` | 27 | 0 | 17 | 0.6296 (0.4237-0.8060) | 27 | 0.042378750 | 1847.0/3527.7 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-002` | 27 | 0 | 17 | 0.6296 (0.4237-0.8060) | 27 | 0.042836250 | 1926.0/3125.4 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-003` | 27 | 0 | 17 | 0.6296 (0.4237-0.8060) | 27 | 0.044658750 | 2057.0/2693.2 |
| `typesafe-jev` | `jev-aspect-v1:trial-001` | 27 | 0 | 20 | 0.7407 (0.5372-0.8889) | 27 | 0.000850038 | 622.0/694.8 |
| `typesafe-jev` | `jev-aspect-v1:trial-002` | 27 | 0 | 20 | 0.7407 (0.5372-0.8889) | 27 | 0.000850038 | 628.0/704.3 |
| `typesafe-jev` | `jev-aspect-v1:trial-003` | 27 | 0 | 20 | 0.7407 (0.5372-0.8889) | 27 | 0.000850038 | 612.0/692.7 |

## Paired comparisons

- `jev-aspect-v1:trial-001` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 27 paired, 0 unavailable, exact McNemar p=0.125.
- `jev-aspect-v1:trial-001` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 27 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-001` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 27 paired, 0 unavailable, exact McNemar p=0.25.
- `jev-aspect-v1:trial-002` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 27 paired, 0 unavailable, exact McNemar p=0.25.
- `jev-aspect-v1:trial-002` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 27 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-002` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 27 paired, 0 unavailable, exact McNemar p=0.25.
- `jev-aspect-v1:trial-003` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 27 paired, 0 unavailable, exact McNemar p=0.25.
- `jev-aspect-v1:trial-003` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 27 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-003` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 27 paired, 0 unavailable, exact McNemar p=0.25.

## Stability

- `openrouter-gemini-2.5-flash`: 0.8889 unanimous, 3 adjacent-trial flips, 0 unavailable.
- `openrouter-gemini-3.8-flash`: 1.0000 unanimous, 0 adjacent-trial flips, 0 unavailable.
- `typesafe-jev`: 1.0000 unanimous, 0 adjacent-trial flips, 0 unavailable.

## Pareto

- Non-dominated targets: `typesafe-jev`.
