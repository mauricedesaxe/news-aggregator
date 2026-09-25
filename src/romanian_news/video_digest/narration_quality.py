"""Check that generated clip audio follows the approved English screenplay."""

from __future__ import annotations

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


class NarrationCheckError(RuntimeError):
    pass


def _normalized(word: str) -> str:
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKC", word)
        if character.isalnum()
    )


def narration_matches(approved_text: str, spoken_words: tuple[str, ...]) -> bool:
    approved = tuple(filter(None, (_normalized(word) for word in approved_text.split())))
    spoken = tuple(filter(None, (_normalized(word) for word in spoken_words)))
    if not approved or not spoken:
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
    try:
        result = subprocess.run(
            [sys.executable, "-m", "romanian_news.video_digest.narration_quality", str(path)],
            input=approved_text,
            capture_output=True,
            text=True,
            timeout=NARRATION_CHECK_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise NarrationCheckError("English narration check failed") from error
    if result.stdout.strip() not in {"match", "mismatch"}:
        raise NarrationCheckError("English narration check returned an invalid result")
    return result.stdout.strip() == "match"


def _transcribed_narration_matches(path: Path, approved_text: str) -> bool:
    segments, _info = _whisper_model().transcribe(
        str(path), language="en", word_timestamps=True, vad_filter=True
    )
    spoken = tuple(word.word for segment in segments for word in segment.words or ())
    return narration_matches(approved_text, spoken)


if __name__ == "__main__":
    print(
        "match"
        if _transcribed_narration_matches(Path(sys.argv[1]), sys.stdin.read())
        else "mismatch"
    )
