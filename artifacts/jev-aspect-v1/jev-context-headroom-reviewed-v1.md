# Jev context headroom on reviewed requests

TypeSafe reports token use after each accepted request. The table deduplicates identical requests across the three trials and compares them with the documented 32,000-token single-question limit.

| Concern | Requests | p50 | p95 | p99 | Max | Max use | Minimum headroom | >=50% | >=75% | >=90% | >=100% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| confidence | 4 | 1710 | 2869 | 2869 | 2869 | 8.965625% | 29131 | 0 | 0 | 0 | 0 |
| daily_theme | 27 | 740 | 847 | 847 | 847 | 2.646875% | 31153 | 0 | 0 | 0 | 0 |
| grouping | 23 | 2312 | 5244 | 5359 | 5359 | 16.746875% | 26641 | 0 | 0 | 0 | 0 |
| ranking | 28 | 7169 | 14351 | 18069 | 18069 | 56.465625% | 13931 | 1 | 0 | 0 | 0 |
| tier | 50 | 14361 | 14367 | 14369 | 14369 | 44.903125% | 17631 | 0 | 0 | 0 | 0 |

## Limits of this evidence

- The reviewed v11 artifacts are not a recent-production-day sample.
- Provider token counts exist only for requests that TypeSafe accepted.
- Pairwise benchmark requests do not establish full-pipeline context fit.
