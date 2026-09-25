from romanian_news.video_digest.narration_quality import (
    narration_matches,
    narration_tail_cutoff_ms,
)

APPROVED = (
    "The market opened lower today while investors awaited the central bank decision "
    "on interest rates and the latest employment figures from across the country"
)


def test_approved_english_line_allows_small_transcription_errors() -> None:
    spoken = APPROVED.replace("investors", "traders").split()
    assert narration_matches(APPROVED, tuple(spoken))


def test_spoken_visual_instruction_cannot_pass() -> None:
    spoken = "Tangled ticker tape fills the booth " + APPROVED
    assert not narration_matches(APPROVED, tuple(spoken.split()))


def test_spoken_instruction_in_the_middle_cannot_pass() -> None:
    spoken = APPROVED.replace(
        "central bank", "central bank say exactly once in English central bank"
    )
    assert not narration_matches(APPROVED, tuple(spoken.split()))


def test_truncated_ending_cannot_pass() -> None:
    assert not narration_matches(APPROVED, tuple(APPROVED.split()[:14]))


def test_one_missed_boundary_word_does_not_reject_faithful_audio() -> None:
    words = APPROVED.split()
    assert narration_matches(APPROVED, tuple(words[1:]))
    assert narration_matches(APPROVED, tuple(words[:-1]))


def test_empty_transcript_cannot_pass() -> None:
    assert not narration_matches(APPROVED, ())


def test_tail_cutoff_keeps_the_complete_approved_line() -> None:
    approved_words = APPROVED.split()
    spoken_words = approved_words + "Read the full sentence once at a measured pace".split()
    timed = tuple((word, index * 0.3, index * 0.3 + 0.2) for index, word in enumerate(spoken_words))

    assert not narration_matches(APPROVED, tuple(spoken_words))
    assert narration_tail_cutoff_ms(APPROVED, timed) == round((len(approved_words) - 1) * 300 + 300)
    assert narration_tail_cutoff_ms(APPROVED, timed[: len(approved_words)]) is None
    assert (
        narration_tail_cutoff_ms(
            APPROVED, timed[: len(approved_words) - 1] + timed[len(approved_words) :]
        )
        is None
    )


def test_tail_cutoff_accepts_documented_name_transcription_without_prompt_leak() -> None:
    approved = (
        "Low Danube water levels prevent restarting Cernavodă nuclear reactors for ten "
        "days, delaying any reopening until mid-October, according to Cristian Bușoi."
    )
    spoken = (
        "Low Danube water levels prevent restarting Cernavoda nuclear reactors for 10 "
        "days, delaying any reopening until mid October, according to Christian Bushoy. "
        "Read the full sentence once at a measured pace."
    ).split()
    timed = tuple((word, index * 0.3, index * 0.3 + 0.2) for index, word in enumerate(spoken))

    assert narration_matches(approved, tuple(spoken[: spoken.index("Read")]))
    assert not narration_matches(approved, tuple(spoken))
    assert narration_tail_cutoff_ms(approved, timed) is not None


NUMERIC_APPROVED = (
    "The National Bank of Romania established the official reference exchange rate "
    "at five point two seven seven zero lei per euro, as the continental currency "
    "slipped zero point eighteen bani yesterday."
)
NUMERIC_TRANSCRIPT = (
    "The National Bank of Romania established the official reference exchange rate "
    "at 5 .2770 per euro as the continental currency slipped 0 .18 Bonnie yesterday."
)


def test_spoken_decimal_matches_equivalent_english_number_words() -> None:
    assert narration_matches(NUMERIC_APPROVED, tuple(NUMERIC_TRANSCRIPT.split()))


def test_wrong_decimal_cannot_pass_as_a_transcription_variant() -> None:
    transcript = NUMERIC_TRANSCRIPT.replace(".2770", ".2870").replace(".18", ".19")
    assert not narration_matches(NUMERIC_APPROVED, tuple(transcript.split()))


def test_numeric_transcript_with_extra_intro_or_repeated_phrase_fails() -> None:
    intro = "The Department's yards a fliver's news play gold coffins "
    assert not narration_matches(NUMERIC_APPROVED, tuple((intro + NUMERIC_TRANSCRIPT).split()))
    repeated = NUMERIC_TRANSCRIPT.replace(
        "Bonnie yesterday",
        "Bonnie as the continental currency slipped 0 .18 Bonnie yesterday",
    )
    assert not narration_matches(NUMERIC_APPROVED, tuple(repeated.split()))
