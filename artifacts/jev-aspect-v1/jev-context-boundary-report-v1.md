# Jev context-boundary probe

Jev 1.13.0 rejects oversized single-question requests with HTTP 400 `max_tokens_exceeded`. It did not silently truncate any accepted probe state.

## Method

- Provider counting evidence: successful TypeSafe `usage.input_tokens` values.
- State construction: deterministic ASCII filler with `BOUNDARY_MARKER_PRESENT` at the final bytes.
- Execution: one attempt per preregistered case, no retries.
- Spend: $0.003117828 across all nine requests, below each protocol's separate $0.05 ceiling.
- Accuracy: not evaluated. Marker probability is only secondary truncation evidence.

| Protocol | State characters | Outcome | HTTP | Input tokens | Marker probability |
| --- | ---: | --- | ---: | ---: | ---: |
| v3 | 40,000 | accepted | 200 | 17,562 | 0.99 |
| v3 | 60,000 | accepted | 200 | 26,183 | 0.99 |
| v3 | 70,000 | accepted | 200 | 30,489 | 0.99 |
| v2 | 80,000 | rejected: `max_tokens_exceeded` | 400 | unavailable | unavailable |
| v2 | 100,000 | rejected: `max_tokens_exceeded` | 400 | unavailable | unavailable |
| v2 | 110,000 | rejected: `max_tokens_exceeded` | 400 | unavailable | unavailable |
| v1 | 120,000 | rejected: `max_tokens_exceeded` | 400 | unavailable | unavailable |
| v1 | 150,000 | rejected: `max_tokens_exceeded` | 400 | unavailable | unavailable |
| v1 | 180,000 | rejected: `max_tokens_exceeded` | 400 | unavailable | unavailable |

## Conclusion

The observed acceptance boundary lies between these exact deterministic states: 70,000 characters produced 30,489 input tokens and was accepted; 80,000 characters was rejected before usage accounting. The accepted 70,000-character state retained its final marker with probability 0.99. The rejected response exposed no token count, so this probe does not claim an exact tokenizer threshold in characters.

Production requests must therefore be preflighted or reduced before calling Jev. Retrying an unchanged `max_tokens_exceeded` request cannot succeed.
