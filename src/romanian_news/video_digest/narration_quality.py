"""Check that generated clip audio follows the approved English screenplay."""

from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

MIN_APPROVED_WORD_COVERAGE = 0.8
MAX_EXTRA_WORD_RUN = 3
NARRATION_CHECK_TIMEOUT_SECONDS = 120
_WORD_OR_NUMBER = re.compile(r"\d+(?:\.\d+)?|\.\d+|[^\W\d_]+(?:['’][^\W\d_]+)?")
_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


class NarrationCheckError(RuntimeError):
    pass


def _normalized(word: str) -> str:
    normalized = "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKD", word)
        if character.isalnum()
    )
    return {"bushoy": "busoi", "buschoy": "busoi"}.get(normalized, normalized)


def _number_words(tokens: list[str], start: int) -> tuple[str, int] | None:
    first = tokens[start]
    if first not in _UNITS and first not in _TENS:
        return None
    whole = _UNITS.get(first, _TENS.get(first, 0))
    position = start + 1
    if first in _TENS and position < len(tokens) and tokens[position] in _UNITS:
        whole += _UNITS[tokens[position]]
        position += 1
    if position >= len(tokens) or tokens[position] != "point":
        return str(whole), position
    position += 1
    fraction = ""
    while position < len(tokens):
        word = tokens[position]
        if word in _UNITS:
            fraction += str(_UNITS[word])
        elif word in _TENS:
            value = _TENS[word]
            if position + 1 < len(tokens) and tokens[position + 1] in _UNITS:
                value += _UNITS[tokens[position + 1]]
                position += 1
            fraction += str(value)
        else:
            break
        position += 1
    if not fraction:
        return str(whole), position - 1
    return f"{whole}.{fraction}", position


def _canonical_words(text: str) -> tuple[str, ...]:
    tokens = _WORD_OR_NUMBER.findall(text.casefold())
    words: list[str] = []
    position = 0
    while position < len(tokens):
        token = tokens[position]
        if (
            token.isdigit()
            and position + 1 < len(tokens)
            and re.fullmatch(r"\.\d+", tokens[position + 1])
        ):
            words.append(f"#{token}{tokens[position + 1]}")
            position += 2
            continue
        number = _number_words(tokens, position)
        if number is not None:
            value, position = number
            words.append(f"#{value}")
            continue
        words.append(f"#{token}" if token[0].isdigit() else _normalized(token))
        position += 1
    return tuple(filter(None, words))


def _numeric_facts_match(approved: tuple[str, ...], spoken: tuple[str, ...]) -> bool:
    return tuple(word for word in approved if word.startswith("#")) == tuple(
        word for word in spoken if word.startswith("#")
    )


def narration_matches(approved_text: str, spoken_words: tuple[str, ...]) -> bool:
    approved = _canonical_words(approved_text)
    spoken = _canonical_words(" ".join(spoken_words))
    if not approved or not spoken:
        return False
    if not _numeric_facts_match(approved, spoken):
        return False
    opcodes = SequenceMatcher(None, approved, spoken, autojunk=False).get_opcodes()
    matched = sum(i2 - i1 for tag, i1, i2, _j1, _j2 in opcodes if tag == "equal")
    if matched / len(approved) < MIN_APPROVED_WORD_COVERAGE:
        return False
    if not any(tag == "equal" and i1 <= 1 and j1 <= 1 for tag, i1, _i2, j1, _j2 in opcodes):
        return False
    if not any(
        tag == "equal" and i2 >= len(approved) - 1 and j2 >= len(spoken) - 1
        for tag, _i1, i2, _j1, j2 in opcodes
    ):
        return False
    return all(j2 - j1 <= MAX_EXTRA_WORD_RUN for tag, _i1, _i2, j1, j2 in opcodes if tag != "equal")


@lru_cache(maxsize=1)
def _whisper_model() -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel("base.en", device="cpu", compute_type="int8")


def clip_narration_matches(path: Path, approved_text: str) -> bool:
    result = _run_narration_check(path, approved_text)
    if result not in {"match", "mismatch"}:
        raise NarrationCheckError("English narration check returned an invalid result")
    return result == "match"


def clip_narration_tail_cutoff_ms(path: Path, approved_text: str) -> int | None:
    result = _run_narration_check(path, approved_text, tail_cutoff=True)
    if result == "mismatch":
        return None
    try:
        cutoff = int(result)
    except ValueError as error:
        raise NarrationCheckError("English narration cutoff was invalid") from error
    if cutoff <= 0:
        raise NarrationCheckError("English narration cutoff was invalid")
    return cutoff


def _run_narration_check(path: Path, approved_text: str, *, tail_cutoff: bool = False) -> str:
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "romanian_news.video_digest.narration_quality",
                *(["--tail-cutoff"] if tail_cutoff else []),
                str(path),
            ],
            input=approved_text,
            capture_output=True,
            text=True,
            timeout=NARRATION_CHECK_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise NarrationCheckError("English narration check failed") from error
    return result.stdout.strip()


def narration_tail_cutoff_ms(
    approved_text: str, words: tuple[tuple[str, float, float], ...]
) -> int | None:
    spoken = tuple(word[0] for word in words)
    if not spoken or narration_matches(approved_text, spoken):
        return None
    approved_tokens = _canonical_words(approved_text)
    if not approved_tokens:
        return None
    final_token = approved_tokens[-1]
    for count in range(1, len(words)):
        current = _canonical_words(words[count - 1][0])
        if not current or current[-1] != final_token:
            continue
        if narration_matches(approved_text, spoken[:count]):
            last_end = words[count - 1][2]
            next_start = words[count][1]
            return round(max(last_end, min(next_start, last_end + 0.15)) * 1000)
    return None


def _transcribed_narration_matches(path: Path, approved_text: str) -> bool:
    segments, _info = _whisper_model().transcribe(
        str(path), language="en", word_timestamps=True, vad_filter=True
    )
    spoken = tuple(word.word for segment in segments for word in segment.words or ())
    return narration_matches(approved_text, spoken)


def _transcribed_tail_cutoff_ms(path: Path, approved_text: str) -> int | None:
    segments, _info = _whisper_model().transcribe(
        str(path), language="en", word_timestamps=True, vad_filter=True
    )
    words = tuple(
        (word.word, word.start, word.end) for segment in segments for word in segment.words or ()
    )
    return narration_tail_cutoff_ms(approved_text, words)


if __name__ == "__main__":
    if sys.argv[1] == "--tail-cutoff":
        cutoff = _transcribed_tail_cutoff_ms(Path(sys.argv[2]), sys.stdin.read())
        sys.stdout.write(f"{cutoff if cutoff is not None else 'mismatch'}\n")
    else:
        sys.stdout.write(
            "match\n"
            if _transcribed_narration_matches(Path(sys.argv[1]), sys.stdin.read())
            else "mismatch\n"
        )
