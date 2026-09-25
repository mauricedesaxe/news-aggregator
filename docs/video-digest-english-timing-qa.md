# English subtitle timing on the approved video

Measured 2026-09-25 against the [approved 44.352-second video](https://media.alexlazar.dev/h3-daily-news/2026-09-18/daily-news-report-subtitled-93bfad318192.mp4). Its downloaded bytes matched SHA-256
`93bfad318192125019bbe48a99b1efeb6deb000e99c036aecd631c21020dd382`
from [issue #11](https://github.com/mauricedesaxe/news-aggregator/issues/11). The
English dialogue and manual cues came from the approved Chartly prototype at
`mauricedesaxe/chartly@9ee232c`, in `news_video_h3_report_assembly.py` and its
three imported H3 story scripts. The exact dialogue, source identity, QA
windows, and expected 95-word, 12-cue coverage are recorded in
[`english_reference_93bfad318192.json`](../tests/fixtures/video_digest/english_reference_93bfad318192.json).

The test used the Dagster video base image built from
`deploy/dagster-video/Dockerfile`, the repository's locked Python dependencies,
and the production `FasterWhisperSubtitleTimingProvider` with `base.en` on CPU
in `int8` mode. The updated adapter was mounted into the QA image. The three
story windows were 0–15, 15–30, and 30–44.352 seconds. Those cuts follow the
prototype's 15-second story design; the exact clip-duration manifest was not
available, so this is a representative timing check rather than an exact
production-edition replay.

| Strategy | Exact word matches | Valid cues | Result |
| --- | ---: | ---: | --- |
| Whole edition with VAD | 72/95 (75.8%) | 0/12 | Rejected because a cue crossed the approximate story window |
| Per story with VAD | 71/95 (74.7%) | 12/12 | Valid complete WebVTT, SHA-256 `b826e1d67189594bb47cbf65412ab6d7271d40ba6c63c9488d5304a00b85de77` |
| Per story without VAD | 71/95 (74.7%) | 12/12 | Valid complete WebVTT, SHA-256 `13a639db71e426943aa08823e05cabcab4833e0fa934c3ed31b0660ccf65c735` |

The first strategy fell back to the second. One of three attempted strategies
needed fallback; no clean-video fallback was needed. All 12 generated cues
cover the approved English screenplay in order. The speech recognizer writes
some spoken numbers as digits and mishears a name, which accounts for part of
the exact-word difference. The WebVTT keeps the approved words, not the
recognizer's text. The adapter requires at least 60% matched words and matched
first and last words in each story before timing a cue.

Before the adapter change, all three strategies failed with zero valid cues.
The first per-story attempt rejected a zero-duration Whisper word; the other
failures came from adjacent timestamp anchors with no gap for an unmatched
word. The adapter now drops unusable zero-duration anchors and can borrow the
minimum number of milliseconds from the next valid anchor for an unmatched
word. It still rejects a transcript that lacks enough matches or cannot fit
positive word spans.

**Decision:** The English reference passes the technical subtitle timing gate
through the per-story fallback. Keep `scheduled_video_digest` stopped until the
other production prerequisites pass and an exact generated edition is reviewed
in the reader with its approved transcript and subtitle track. This one sample
does not establish a population fallback rate.
