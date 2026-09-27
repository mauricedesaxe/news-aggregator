# Weekly status history coverage

Production catalog inventory on 27 September 2026 covered 108 completed Monday-to-Sunday weeks, from 26 August 2024 through 14 September 2026. The [weekly inventory run](https://github.com/mauricedesaxe/news-aggregator/actions/runs/36302341955) and [source timing run](https://github.com/mauricedesaxe/news-aggregator/actions/runs/36302934102) provide the underlying counts. The inventory workflow can be rerun as new data arrives.

The 99 weeks from 26 August 2024 through 13 July 2026 have no saved daily reports, published weekly reads, successful feed observations, article files dated to those weeks, or feed entries observed within 14 days of publication. The remaining weeks are:

| Week starting | Daily reports | Weekly reads | Feed observations | Article files | Feed entries observed within 14 days |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 Jul 2026 | 0 | 0 | 0 | 2 | 0 |
| 27 Jul 2026 | 0 | 0 | 0 | 3 | 0 |
| 3 Aug 2026 | 0 | 0 | 0 | 2 | 0 |
| 10 Aug 2026 | 0 | 0 | 0 | 6 | 0 |
| 17 Aug 2026 | 0 | 0 | 0 | 11 | 34 |
| 24 Aug 2026 | 0 | 0 | 0 | 412 | 2,269 |
| 31 Aug 2026 | 7 | 1 | 5,131 | 7,625 | 95,813 |
| 7 Sep 2026 | 6 | 1 | 5,804 | 11,134 | 12,461 |
| 14 Sep 2026 | 7 | 1 | 6,838 | 12,723 | 14,316 |

The oldest saved daily report is 31 August 2026. All three completed weeks with at least five daily reports already have published weekly reads, so there is no eligible unpublished week for the existing backfill job.

Older publication dates do not establish contemporaneous coverage. The timing audit found no pre-31 August article file captured within one calendar day of its article date, and no successful feed observation before 31 August. For the week of 24 August, feed entries observed within one calendar day of publication occur on 30 August only. Reconstructing five daily reports from that partial retrospective corpus would imply coverage we cannot verify. The earliest defensible weekly read is therefore the week of 31 August 2026. No historical reconstruction or model calls were run, so this audit incurred no model cost.

The archive lists published weeks only and states that a missing week has no saved read. A longer history will require independently retained, date-bound source material for at least five days of each earlier week, followed by the same citation and coverage checks used for current reads.
