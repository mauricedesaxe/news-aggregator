from romanian_news.video_digest.narration_quality import narration_matches

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
