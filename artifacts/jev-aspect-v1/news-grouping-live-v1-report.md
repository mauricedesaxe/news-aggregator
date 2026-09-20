# grouping binary benchmark analysis

- Identity: `2e2dc1f4f17b355c2da914a564517e6a9bec6541ba4e5bdf55004704f61e9936`
- Source artifact: `083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a`
- Shape: 3 targets x 3 trials x 23 cases
- Limitation: Pair judgments do not establish a valid complete partition.

## Trial results

| Target | Trial | Completed | Failed | Correct | Accuracy (95% exact CI) | Attempts | Cost USD | p50/p95 ms |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-001` | 23 | 0 | 21 | 0.9130 (0.7196-0.9893) | 23 | 0.015732400 | 703.0/1027.5 |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-002` | 23 | 0 | 21 | 0.9130 (0.7196-0.9893) | 23 | 0.015732400 | 670.0/882.8 |
| `openrouter-gemini-2.5-flash` | `jev-aspect-v1:trial-003` | 23 | 0 | 21 | 0.9130 (0.7196-0.9893) | 23 | 0.014291410 | 663.0/906.5 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-001` | 23 | 0 | 19 | 0.8261 (0.6122-0.9505) | 23 | 0.075197250 | 1588.0/4695.3 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-002` | 23 | 0 | 19 | 0.8261 (0.6122-0.9505) | 23 | 0.077792250 | 1658.0/4927.2 |
| `openrouter-gemini-3.8-flash` | `jev-aspect-v1:trial-003` | 23 | 0 | 20 | 0.8696 (0.6641-0.9722) | 23 | 0.076689750 | 1837.0/4008.0 |
| `typesafe-jev` | `jev-aspect-v1:trial-001` | 23 | 0 | 21 | 0.9130 (0.7196-0.9893) | 23 | 0.002683968 | 646.0/800.7 |
| `typesafe-jev` | `jev-aspect-v1:trial-002` | 23 | 0 | 21 | 0.9130 (0.7196-0.9893) | 23 | 0.002683968 | 641.0/839.8 |
| `typesafe-jev` | `jev-aspect-v1:trial-003` | 23 | 0 | 20 | 0.8696 (0.6641-0.9722) | 23 | 0.002683968 | 658.0/858.2 |

## Paired comparisons

- `jev-aspect-v1:trial-001` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 23 paired, 0 unavailable, exact McNemar p=0.5.
- `jev-aspect-v1:trial-001` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 23 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-001` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 23 paired, 0 unavailable, exact McNemar p=0.5.
- `jev-aspect-v1:trial-002` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 23 paired, 0 unavailable, exact McNemar p=0.5.
- `jev-aspect-v1:trial-002` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 23 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-002` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 23 paired, 0 unavailable, exact McNemar p=0.5.
- `jev-aspect-v1:trial-003` `openrouter-gemini-2.5-flash` vs `openrouter-gemini-3.8-flash`: 23 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-003` `openrouter-gemini-2.5-flash` vs `typesafe-jev`: 23 paired, 0 unavailable, exact McNemar p=1.
- `jev-aspect-v1:trial-003` `openrouter-gemini-3.8-flash` vs `typesafe-jev`: 23 paired, 0 unavailable, exact McNemar p=1.

## Stability

- `openrouter-gemini-2.5-flash`: 1.0000 unanimous, 0 adjacent-trial flips, 0 unavailable.
- `openrouter-gemini-3.8-flash`: 0.9565 unanimous, 1 adjacent-trial flips, 0 unavailable.
- `typesafe-jev`: 0.9565 unanimous, 1 adjacent-trial flips, 0 unavailable.

## Pareto

- Non-dominated targets: `openrouter-gemini-2.5-flash`, `typesafe-jev`.
