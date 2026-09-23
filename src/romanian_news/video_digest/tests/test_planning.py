from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Literal

import pytest
from pydantic import ValidationError

from romanian_news.reports import (
    DailyReport,
    DailyReportSection,
    ReportArticle,
    ReportEvent,
    ReportSubjectCitation,
)
from romanian_news.video_digest.planning import (
    PlanningAttempt,
    PlanningFailure,
    ScreenplayPlan,
    ScreenplayStory,
    StoryVerificationEvidence,
    VideoDigestPolicyBundle,
    accept_planning_attempt,
    authorize_generation,
    create_screenplay_plan,
    policy_bundle_digest,
    record_planning_attempt,
    screenplay_plan_digest,
    screenplay_story_digest,
    verification_evidence_digest,
)

REPORT_VERSION = "a" * 64
POLICY_VERSION = "b" * 64
ARTICLE_ONE = "1" * 64
ARTICLE_TWO = "2" * 64
ARTICLE_THREE = "3" * 64
MAIN_ONE = "4" * 64
MAIN_TWO = "5" * 64
WORTH_KNOWING = "6" * 64
EXCLUDED = "7" * 64


def _policy(**changes: object) -> VideoDigestPolicyBundle:
    values: dict[str, object] = {
        "policy_id": "video-digest-production-v1",
        "planning_model": "openrouter/planner",
        "verification_model": "openrouter/verifier",
        "planning_prompt_digest": "8" * 64,
        "verification_prompt_digest": "9" * 64,
    }
    return VideoDigestPolicyBundle.model_validate(values | changes)


def _event(article_version_id: str) -> ReportEvent:
    return ReportEvent(
        group_id=article_version_id,
        title_ro="Titlu",
        summary_ro="Rezumat",
        key_points_ro=(),
        disagreements_ro=(),
        sentiment_label="neutral",
        sentiment_score=0.0,
        sentiment_rationale_ro="Fara abatere",
        articles=(
            ReportArticle(
                article_version_id=article_version_id,
                outlet_id="example",
                title="Articol",
                canonical_url="https://example.com/article",
                sentiment_label="neutral",
                sentiment_score=0.0,
            ),
        ),
    )


def _section(
    theme_id: str,
    tier: Literal["main", "worth_knowing", "excluded"],
    rank: int,
    citation_ids: tuple[str, ...],
) -> DailyReportSection:
    return DailyReportSection(
        theme_id=theme_id,
        title=f"Subiect {rank}",
        summary="Rezumatul subiectului",
        events=(_event(citation_ids[0]),),
        tier=tier,
        semantic_rank=rank,
        consequence_rationale="Consecinta este materiala.",
        citations=tuple(
            ReportSubjectCitation(
                article_version_id=article_id,
                evidence_quote=f"Dovada {index}",
            )
            for index, article_id in enumerate(citation_ids)
        ),
    )


def _report() -> DailyReport:
    sections = (
        _section(MAIN_ONE, "main", 1, (ARTICLE_ONE, ARTICLE_TWO)),
        _section(WORTH_KNOWING, "worth_knowing", 2, (ARTICLE_THREE,)),
        _section(MAIN_TWO, "main", 3, (ARTICLE_THREE,)),
        _section(EXCLUDED, "excluded", 4, (ARTICLE_TWO,)),
    )
    return DailyReport(
        day=date(2026, 9, 20),
        accepted_article_count=3,
        theme_count=4,
        group_count=4,
        sections=sections,
    )


def _narration(words: int = 30) -> str:
    return " ".join(f"cuvant{index}" for index in range(words))


def _story(
    subject_id: str,
    citations: tuple[str, ...],
    *,
    words: int = 30,
    duration_ms: int = 15_000,
) -> ScreenplayStory:
    return ScreenplayStory(
        report_subject_id=subject_id,
        title="Titlu video",
        citation_article_version_ids=citations,
        narration=_narration(words),
        visual_direction="Gazdele explica intr-un cadru activ.",
        requested_duration_ms=duration_ms,
    )


def _stories() -> tuple[ScreenplayStory, ...]:
    return (
        _story(MAIN_ONE, (ARTICLE_ONE, ARTICLE_TWO)),
        _story(MAIN_TWO, (ARTICLE_THREE,)),
    )


def _evidence(
    story: ScreenplayStory,
    policy: VideoDigestPolicyBundle,
    *,
    status: Literal["accepted", "rejected"] = "accepted",
) -> StoryVerificationEvidence:
    failures = (
        ()
        if status == "accepted"
        else (PlanningFailure(code="unsupported_claim", message="A claim lacks support"),)
    )
    return StoryVerificationEvidence(
        story_screenplay_digest=screenplay_story_digest(story),
        daily_report_version_id=REPORT_VERSION,
        policy_bundle_digest=policy_bundle_digest(policy),
        citation_article_version_ids=story.citation_article_version_ids,
        status=status,
        failures=failures,
    )


def _attempt(
    plan: ScreenplayPlan,
    policy: VideoDigestPolicyBundle,
) -> PlanningAttempt:
    evidence = tuple(_evidence(story, policy) for story in plan.stories)
    return record_planning_attempt(0, plan, evidence, "accepted")


def test_policy_is_frozen_strict_and_content_addressed() -> None:
    policy = _policy()

    assert policy_bundle_digest(policy) == policy_bundle_digest(_policy())
    assert policy_bundle_digest(policy) != policy_bundle_digest(
        _policy(planning_prompt_digest="0" * 64)
    )
    with pytest.raises(ValidationError):
        policy.__setattr__("story_duration_ms", 14_000)
    with pytest.raises(ValidationError):
        VideoDigestPolicyBundle.model_validate(policy.model_dump() | {"story_duration_ms": "15000"})


def test_complete_plan_contains_each_main_subject_once_in_report_order() -> None:
    policy = _policy()

    plan = create_screenplay_plan(_report(), REPORT_VERSION, POLICY_VERSION, policy, _stories())

    assert tuple(story.report_subject_id for story in plan.stories) == (MAIN_ONE, MAIN_TWO)
    assert WORTH_KNOWING not in {story.report_subject_id for story in plan.stories}
    assert EXCLUDED not in {story.report_subject_id for story in plan.stories}


def test_plan_identity_is_stable_and_sensitive_to_report_and_policy() -> None:
    report = _report()
    stories = _stories()
    policy = _policy()
    plan = create_screenplay_plan(report, REPORT_VERSION, POLICY_VERSION, policy, stories)

    assert screenplay_plan_digest(plan) == screenplay_plan_digest(
        create_screenplay_plan(report, REPORT_VERSION, POLICY_VERSION, policy, stories)
    )
    assert screenplay_plan_digest(plan) != screenplay_plan_digest(
        create_screenplay_plan(report, "0" * 64, POLICY_VERSION, policy, stories)
    )
    assert screenplay_plan_digest(plan) != screenplay_plan_digest(
        create_screenplay_plan(
            report,
            REPORT_VERSION,
            POLICY_VERSION,
            _policy(planning_prompt_digest="0" * 64),
            stories,
        )
    )


@pytest.mark.parametrize(
    "stories",
    (
        (_stories()[0],),
        (_stories()[1], _stories()[0]),
        (_stories()[0], _stories()[0], _stories()[1]),
        (*_stories(), _story(WORTH_KNOWING, (ARTICLE_THREE,))),
    ),
)
def test_plan_rejects_missing_reordered_duplicate_and_extra_subjects(
    stories: tuple[ScreenplayStory, ...],
) -> None:
    with pytest.raises(ValueError, match="exactly cover main report subjects in report order"):
        create_screenplay_plan(_report(), REPORT_VERSION, POLICY_VERSION, _policy(), stories)


def test_plan_requires_exact_report_citation_membership() -> None:
    wrong = (
        _story(MAIN_ONE, (ARTICLE_ONE,)),
        _story(MAIN_TWO, (ARTICLE_THREE,)),
    )

    with pytest.raises(ValueError, match="citations must exactly match"):
        create_screenplay_plan(_report(), REPORT_VERSION, POLICY_VERSION, _policy(), wrong)


@pytest.mark.parametrize("words", (24, 36))
def test_plan_enforces_about_thirty_spoken_words(words: int) -> None:
    stories = (_story(MAIN_ONE, (ARTICLE_ONE, ARTICLE_TWO), words=words), _stories()[1])

    with pytest.raises(ValueError, match="25 to 35 spoken words"):
        create_screenplay_plan(_report(), REPORT_VERSION, POLICY_VERSION, _policy(), stories)


@pytest.mark.parametrize("words", (25, 30, 35))
def test_plan_accepts_the_production_spoken_word_range(words: int) -> None:
    stories = (_story(MAIN_ONE, (ARTICLE_ONE, ARTICLE_TWO), words=words), _stories()[1])

    assert (
        create_screenplay_plan(
            _report(), REPORT_VERSION, POLICY_VERSION, _policy(), stories
        ).stories
        == stories
    )


def test_plan_requires_exact_production_story_duration() -> None:
    stories = (
        _story(MAIN_ONE, (ARTICLE_ONE, ARTICLE_TWO), duration_ms=14_999),
        _stories()[1],
    )

    with pytest.raises(ValueError, match="exactly 15000 ms"):
        create_screenplay_plan(_report(), REPORT_VERSION, POLICY_VERSION, _policy(), stories)


@pytest.mark.parametrize("attempt_index", (-1, 3))
def test_planning_attempt_rejects_out_of_bounds_index(attempt_index: int) -> None:
    policy = _policy()
    plan = create_screenplay_plan(_report(), REPORT_VERSION, POLICY_VERSION, policy, _stories())
    evidence = tuple(_evidence(story, policy) for story in plan.stories)

    with pytest.raises(ValidationError):
        record_planning_attempt(attempt_index, plan, evidence, "accepted")


@pytest.mark.parametrize(
    "change",
    (
        lambda evidence: evidence.model_copy(update={"story_screenplay_digest": "0" * 64}),
        lambda evidence: evidence.model_copy(update={"daily_report_version_id": "0" * 64}),
        lambda evidence: evidence.model_copy(update={"policy_bundle_digest": "0" * 64}),
    ),
)
def test_accepted_evidence_is_bound_to_story_report_and_policy(
    change: Callable[[StoryVerificationEvidence], StoryVerificationEvidence],
) -> None:
    policy = _policy()
    plan = create_screenplay_plan(_report(), REPORT_VERSION, POLICY_VERSION, policy, _stories())
    evidence = tuple(_evidence(story, policy) for story in plan.stories)

    with pytest.raises(ValueError, match="evidence does not match"):
        record_planning_attempt(0, plan, (change(evidence[0]), evidence[1]), "accepted")


def test_rejected_attempt_records_structured_failures_but_cannot_be_verified() -> None:
    policy = _policy()
    plan = create_screenplay_plan(_report(), REPORT_VERSION, POLICY_VERSION, policy, _stories())
    rejected = record_planning_attempt(
        0,
        plan,
        (_evidence(plan.stories[0], policy, status="rejected"), _evidence(plan.stories[1], policy)),
        "rejected",
        failures=(PlanningFailure(code="story_rejected", message="Story 0 failed verification"),),
    )

    assert rejected.failures[0].code == "story_rejected"
    with pytest.raises(ValueError, match="accepted planning attempt"):
        accept_planning_attempt(_report(), policy, rejected)


def test_accepted_attempt_cannot_be_rebased_onto_a_different_policy() -> None:
    policy = _policy()
    plan = create_screenplay_plan(_report(), REPORT_VERSION, POLICY_VERSION, policy, _stories())
    attempt = _attempt(plan, policy)

    with pytest.raises(ValueError, match="does not match the supplied report and policy"):
        accept_planning_attempt(
            _report(),
            _policy(planning_prompt_digest="0" * 64),
            attempt,
        )


def test_authorization_binds_plan_to_ordered_accepted_evidence() -> None:
    policy = _policy()
    plan = create_screenplay_plan(_report(), REPORT_VERSION, POLICY_VERSION, policy, _stories())
    verified = accept_planning_attempt(_report(), policy, _attempt(plan, policy))

    authorization = authorize_generation(verified)

    assert authorization.edition_id == plan.edition_id
    assert authorization.policy_bundle_version_id == POLICY_VERSION
    assert authorization.screenplay_plan_digest == verified.screenplay_plan_digest
    assert authorization.ordered_verification_evidence_digests == tuple(
        verification_evidence_digest(item) for item in verified.accepted_evidence
    )
    reversed_verified = verified.model_copy(
        update={"accepted_evidence": tuple(reversed(verified.accepted_evidence))}
    )
    with pytest.raises(ValueError, match="ordered accepted evidence"):
        authorize_generation(reversed_verified)
