# Jev production relevance shadow, 2026-09-25

A local worker using the production catalog, immutable R2 inputs, and a dedicated TypeSafe key ran the guarded Jev shadow once for the 2026-09-25 Bucharest partition. Gemini remained authoritative. The exact claim identities, incumbent and Jev verdicts, provider attempts, usage, costs, and timestamps remain in the immutable production catalog. The canonical 83-row sample SHA-256 is `1a86d2c36abd1dcab53a8316dd5f9de55febc955d723fb211786f0ca10c12d2b`.

The worker observed 307 current articles and 83 completed incumbent decisions at its start. It paired all 83. The other 224 had no matching incumbent decision and were not sent to Jev. Every pair has a terminal receipt and complete accounting. There were no provider failures or over-guard requests. The longest state was 13,396 characters, below the 70,000-character guard. Usage was 138,026 input tokens and 1,826 output tokens, with an estimated cost of $0.005797092.

| Incumbent | Jev | Cases |
| --- | --- | ---: |
| Rejected | Rejected | 42 |
| Rejected | Accepted | 21 |
| Accepted | Rejected | 0 |
| Accepted | Accepted | 20 |

Jev retained all 20 incumbent accepted cases. The 21 Jev-only accepted cases need independent article-level labels before production precision and positive-control gates can be judged. Incumbent agreement is not a substitute for ground truth. The Dagster shadow flag remains off, and no production relevance authority changed.
