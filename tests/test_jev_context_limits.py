from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
LIMITS_PATH = ROOT / "src/romanian_news/jev_context_limits.json"
NOTE_PATH = ROOT / "artifacts/jev-aspect-v1/jev-context-limits.md"


def test_jev_context_limit_registry_records_the_single_question_ceiling() -> None:
    limits = json.loads(LIMITS_PATH.read_text())

    assert limits["requested_model"] == "jev-1.13.0"
    assert limits["endpoint"] == "POST https://api.typesafe.ai/v1/systemone"
    assert limits["documented_limits"] == {
        "request_tokens": 64_000,
        "state_plus_longest_question_tokens": 32_000,
        "single_question_effective_input_tokens": 32_000,
        "approximate_english_characters_for_32000_tokens": 150_000,
    }
    assert limits["counting"]["tokenizer"] is None
    assert limits["counting"]["local_exact_counter"] is None
    assert limits["output_limit_tokens"] is None
    assert limits["over_limit_behavior"] is None
    assert {source["url"] for source in limits["sources"]} == {
        "https://docs.typesafe.ai/models",
        "https://docs.typesafe.ai/api",
        "https://docs.typesafe.ai/primitives",
    }


def test_jev_context_limit_note_links_every_registered_source() -> None:
    limits = json.loads(LIMITS_PATH.read_text())
    note = NOTE_PATH.read_text()

    for source in limits["sources"]:
        assert source["url"] in note
