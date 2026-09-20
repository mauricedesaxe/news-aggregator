# Jev 1.13.0 context limits

Retrieved on 2026-09-19.

## Documented limits

TypeSafe documents a 64,000-token request budget for Jev 1.13.0. This budget covers the shared `state` plus all questions. A second 32,000-token limit covers the `state` plus the single longest question. The evaluated News requests each contain one question, so the 32,000-token limit is the relevant ceiling.

The primitives guide describes 32,000 tokens as roughly 150,000 English characters. This is an approximation, not a tokenizer contract.

The API endpoint is `POST https://api.typesafe.ai/v1/systemone`. A successful response reports `usage.input_tokens` and `usage.output_tokens`.

## Unknowns

The public documentation does not name the tokenizer or provide an exact local counter. It does not state a separate output-token ceiling. It also does not state whether an over-limit request is rejected or silently truncated. The generic API error table has no context-overflow entry.

Provider-reported `usage.input_tokens` is therefore the only exact counting evidence currently available for completed requests. Any boundary test must record the response status, error body, usage fields, and latency without treating synthetic answers as quality evidence.

## Sources

- [Models](https://docs.typesafe.ai/models) documents the model identity, aliases, endpoint family, and both context budgets.
- [API reference](https://docs.typesafe.ai/api) documents the endpoint, request shape, response usage fields, and generic errors.
- [Primitives](https://docs.typesafe.ai/primitives) documents the shared state and question budget and the approximate English-character equivalent.

The machine-readable copy is `src/romanian_news/jev_context_limits.json`.
