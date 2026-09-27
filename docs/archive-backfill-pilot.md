# Historical article backfill: first pilot

The target is one year of visible article reports and weekly status pages. Historical
articles need their own source record: a sitemap observation is not a feed observation,
and a sitemap `lastmod` is not an article publication date. The live article table and
downstream daily analysis can be reused once that provenance is represented honestly.

## Seven-day source check

On 2026-09-27, `scripts/probe_historical_archive.py` sampled three article URLs per
outlet per day for 2025-09-15 through 2025-09-21. It checked robots rules, fetched the
article page, required a page publication timestamp on the target Bucharest day, and
used the existing article extractor. The script saved URL and metadata evidence, not
page text.

| Outlet | Sitemap entries in the selected window | Pages sampled | Correct publication day | Usable extraction |
| --- | ---: | ---: | ---: | ---: |
| HotNews | 643 | 21 | 21 | 21 |
| Digi24 | 799 | 21 | 21 | 21 |

These are small deterministic samples, not estimates of a month's usable yield.
HotNews exposes daily sitemaps. Digi24 exposes a monthly sitemap; its 799 entries
were selected by `lastmod` in that week. Articles published during the week and edited
later can be absent from this slice. A full Digi24 month scan must fetch candidates
and assign the day from page publication metadata.

The sample included HotNews general news, economy, sports, and family sections, plus
Digi24 domestic and foreign news. Editorial relevance still needs a separate gate.
The check made no model calls, so it does not establish backfill model cost.

## Publication boundary

The current [Digi24 terms](https://www.digi24.ro/termeni-si-conditii) restrict copying
and republication, and the [HotNews terms](https://hotnews.ro/termeni-si-conditii-de-utilizare-ale-site-ului-hotnews-ro-1642230)
limit commercial content use without written consent. Bulk retention of full article
text or republication must be resolved before a historical capture run for these
outlets. Source metadata and URL discovery can proceed without publishing articles.

The first infrastructure slice will record immutable sitemap observations and
candidate URLs with source, retrieval time, and date-hint provenance. Later capture
must verify the page publication timestamp, deduplicate by canonical URL, enforce
source-specific retention, and keep retrospective reports in preview until their
coverage and disclosure checks pass.
