from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.catalog.artifacts import canonical_json, sha256
from romanian_news.reports import DailyReport
from romanian_news.video_digest.models import EditionIdField, edition_id

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PlanningAttemptIndex = Annotated[int, Field(ge=0, le=2)]


class VideoDigestPolicyBundle(NewsModel):
    policy_id: NonEmptyText
    planning_model: NonEmptyText
    verification_model: NonEmptyText
    planning_prompt_digest: Sha256
    verification_prompt_digest: Sha256
    coverage_policy: Literal[
        "main-sections-exact-report-order-v1",
        "selected-main-sections-exact-report-order-v2",
    ] = "main-sections-exact-report-order-v1"
    citation_policy: Literal["exact-report-subject-citations-v1"] = (
        "exact-report-subject-citations-v1"
    )
    narration_policy: Literal["target-30-tolerance-5-v1"] = "target-30-tolerance-5-v1"
    duration_policy: Literal["fixed-15000ms-v1"] = "fixed-15000ms-v1"
    planning_attempt_policy: Literal["initial-plus-two-rewrites-v1"] = (
        "initial-plus-two-rewrites-v1"
    )
    target_spoken_words: Literal[30] = 30
    spoken_word_tolerance: Literal[5] = 5
    story_duration_ms: Literal[15_000] = 15_000
    maximum_planning_attempt_index: Literal[2] = 2


def policy_bundle_digest(policy: VideoDigestPolicyBundle) -> Sha256:
    return sha256(canonical_json(policy.model_dump(mode="json")))


class ScreenplayStory(NewsModel):
    report_subject_id: Sha256
    title: NonEmptyText
    citation_article_version_ids: Annotated[tuple[Sha256, ...], Field(min_length=1)]
    narration: NonEmptyText
    visual_direction: NonEmptyText
    requested_duration_ms: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def require_unique_citations(self) -> ScreenplayStory:
        if len(set(self.citation_article_version_ids)) != len(self.citation_article_version_ids):
            raise ValueError("Story citations must be unique")
        return self


def screenplay_story_digest(story: ScreenplayStory) -> Sha256:
    return sha256(canonical_json(story.model_dump(mode="json")))


class ScreenplayPlan(NewsModel):
    edition_id: EditionIdField
    daily_report_version_id: Sha256
    policy_bundle_version_id: Sha256
    policy_bundle_digest: Sha256
    selection_digest: Sha256 | None = None
    stories: Annotated[tuple[ScreenplayStory, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_unique_subjects(self) -> ScreenplayPlan:
        expected_edition_id = edition_id(
            self.daily_report_version_id,
            self.policy_bundle_version_id,
            self.selection_digest,
        )
        if self.edition_id != expected_edition_id:
            raise ValueError("Screenplay edition ID does not match its immutable inputs")
        subject_ids = tuple(story.report_subject_id for story in self.stories)
        if len(set(subject_ids)) != len(subject_ids):
            raise ValueError("Screenplay report subjects must be unique")
        return self


def screenplay_plan_digest(plan: ScreenplayPlan) -> Sha256:
    return sha256(canonical_json(plan.model_dump(mode="json")))


class PlanningFailure(NewsModel):
    code: NonEmptyText
    message: NonEmptyText


class StoryVerificationEvidence(NewsModel):
    story_screenplay_digest: Sha256
    daily_report_version_id: Sha256
    policy_bundle_digest: Sha256
    citation_article_version_ids: Annotated[tuple[Sha256, ...], Field(min_length=1)]
    status: Literal["accepted", "rejected"]
    failures: tuple[PlanningFailure, ...] = ()

    @model_validator(mode="after")
    def require_consistent_disposition(self) -> StoryVerificationEvidence:
        if self.status == "accepted" and self.failures:
            raise ValueError("Accepted story verification cannot contain failures")
        if self.status == "rejected" and not self.failures:
            raise ValueError("Rejected story verification requires a structured failure")
        if len(set(self.citation_article_version_ids)) != len(self.citation_article_version_ids):
            raise ValueError("Verification citations must be unique")
        return self


def verification_evidence_digest(evidence: StoryVerificationEvidence) -> Sha256:
    return sha256(canonical_json(evidence.model_dump(mode="json")))


class PlanningAttempt(NewsModel):
    attempt_index: PlanningAttemptIndex
    plan: ScreenplayPlan
    story_evidence: Annotated[tuple[StoryVerificationEvidence, ...], Field(min_length=1)]
    disposition: Literal["accepted", "rejected"]
    failures: tuple[PlanningFailure, ...] = ()

    @model_validator(mode="after")
    def require_complete_evidence(self) -> PlanningAttempt:
        if len(self.story_evidence) != len(self.plan.stories):
            raise ValueError("Planning attempt must contain evidence for every story")
        for story, evidence in zip(self.plan.stories, self.story_evidence, strict=True):
            if not _evidence_matches_story(evidence, story, self.plan):
                raise ValueError("Story verification evidence does not match its exact inputs")
        all_accepted = all(evidence.status == "accepted" for evidence in self.story_evidence)
        if self.disposition == "accepted" and (not all_accepted or self.failures):
            raise ValueError("Accepted planning attempt requires only accepted story evidence")
        if self.disposition == "rejected" and all_accepted and not self.failures:
            raise ValueError("Rejected planning attempt requires a structured failure")
        return self


class VerifiedDigestPlan(NewsModel):
    screenplay_plan_digest: Sha256
    plan: ScreenplayPlan
    report_subject_ids: Annotated[tuple[Sha256, ...], Field(min_length=1)]
    accepted_evidence: Annotated[tuple[StoryVerificationEvidence, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_ordered_accepted_evidence(self) -> VerifiedDigestPlan:
        if self.screenplay_plan_digest != screenplay_plan_digest(self.plan):
            raise ValueError("Verified screenplay plan digest does not match its content")
        story_subject_ids = tuple(story.report_subject_id for story in self.plan.stories)
        if self.report_subject_ids != story_subject_ids:
            raise ValueError("Verified plan subjects do not match screenplay order")
        if len(self.accepted_evidence) != len(self.plan.stories):
            raise ValueError("Verified plan requires evidence for every story")
        for story, evidence in zip(self.plan.stories, self.accepted_evidence, strict=True):
            if evidence.status != "accepted" or not _evidence_matches_story(
                evidence, story, self.plan
            ):
                raise ValueError("Verified plan requires ordered accepted evidence")
        return self


class GenerationAuthorization(NewsModel):
    edition_id: EditionIdField
    daily_report_version_id: Sha256
    policy_bundle_version_id: Sha256
    policy_bundle_digest: Sha256
    screenplay_plan_digest: Sha256
    ordered_verification_evidence_digests: Annotated[tuple[Sha256, ...], Field(min_length=1)]


def create_screenplay_plan(
    report: DailyReport,
    daily_report_version_id: Sha256,
    policy_bundle_version_id: Sha256,
    policy: VideoDigestPolicyBundle,
    stories: tuple[ScreenplayStory, ...],
    *,
    selected_subject_ids: tuple[Sha256, ...] | None = None,
    selection_digest: Sha256 | None = None,
) -> ScreenplayPlan:
    if (selected_subject_ids is None) != (selection_digest is None):
        raise ValueError("Selected subjects and selection digest must be supplied together")
    if (
        selected_subject_ids is not None
        and policy.coverage_policy != "selected-main-sections-exact-report-order-v2"
    ):
        raise ValueError("Selected subjects require the selected-main coverage policy")
    main_sections = tuple(section for section in report.sections if section.tier == "main")
    if selected_subject_ids is not None:
        selected = set(selected_subject_ids)
        if selected_subject_ids != tuple(
            section.theme_id for section in main_sections if section.theme_id in selected
        ):
            raise ValueError("Selected subjects must be main sections in report order")
        main_sections = tuple(section for section in main_sections if section.theme_id in selected)
    expected_subject_ids = tuple(section.theme_id for section in main_sections)
    actual_subject_ids = tuple(story.report_subject_id for story in stories)
    if actual_subject_ids != expected_subject_ids:
        raise ValueError(
            "Screenplay stories must exactly cover main report subjects in report order"
        )

    minimum_words = policy.target_spoken_words - policy.spoken_word_tolerance
    maximum_words = policy.target_spoken_words + policy.spoken_word_tolerance
    for section, story in zip(main_sections, stories, strict=True):
        expected_citations = tuple(item.article_version_id for item in section.citations)
        if story.citation_article_version_ids != expected_citations:
            raise ValueError("Story citations must exactly match the report subject citations")
        spoken_words = len(story.narration.split())
        if not minimum_words <= spoken_words <= maximum_words:
            raise ValueError(
                f"Story narration must contain {minimum_words} to {maximum_words} spoken words"
            )
        if story.requested_duration_ms != policy.story_duration_ms:
            raise ValueError(
                f"Story requested duration must be exactly {policy.story_duration_ms} ms"
            )

    return ScreenplayPlan(
        edition_id=edition_id(daily_report_version_id, policy_bundle_version_id, selection_digest),
        daily_report_version_id=daily_report_version_id,
        policy_bundle_version_id=policy_bundle_version_id,
        policy_bundle_digest=policy_bundle_digest(policy),
        selection_digest=selection_digest,
        stories=stories,
    )


def record_planning_attempt(
    attempt_index: int,
    plan: ScreenplayPlan,
    story_evidence: tuple[StoryVerificationEvidence, ...],
    disposition: Literal["accepted", "rejected"],
    *,
    failures: tuple[PlanningFailure, ...] = (),
) -> PlanningAttempt:
    return PlanningAttempt(
        attempt_index=attempt_index,
        plan=plan,
        story_evidence=story_evidence,
        disposition=disposition,
        failures=failures,
    )


def accept_planning_attempt(
    report: DailyReport,
    policy: VideoDigestPolicyBundle,
    attempt: PlanningAttempt,
    *,
    selected_subject_ids: tuple[Sha256, ...] | None = None,
    selection_digest: Sha256 | None = None,
) -> VerifiedDigestPlan:
    attempt = PlanningAttempt.model_validate(attempt.model_dump())
    if attempt.disposition != "accepted":
        raise ValueError("Generation requires an accepted planning attempt")
    checked_plan = create_screenplay_plan(
        report,
        attempt.plan.daily_report_version_id,
        attempt.plan.policy_bundle_version_id,
        policy,
        attempt.plan.stories,
        selected_subject_ids=selected_subject_ids,
        selection_digest=selection_digest,
    )
    if checked_plan != attempt.plan:
        raise ValueError("Planning attempt does not match the supplied report and policy")
    return VerifiedDigestPlan(
        screenplay_plan_digest=screenplay_plan_digest(checked_plan),
        plan=checked_plan,
        report_subject_ids=tuple(story.report_subject_id for story in checked_plan.stories),
        accepted_evidence=attempt.story_evidence,
    )


def authorize_generation(verified: VerifiedDigestPlan) -> GenerationAuthorization:
    checked = VerifiedDigestPlan.model_validate(verified.model_dump())
    return GenerationAuthorization(
        edition_id=checked.plan.edition_id,
        daily_report_version_id=checked.plan.daily_report_version_id,
        policy_bundle_version_id=checked.plan.policy_bundle_version_id,
        policy_bundle_digest=checked.plan.policy_bundle_digest,
        screenplay_plan_digest=checked.screenplay_plan_digest,
        ordered_verification_evidence_digests=tuple(
            verification_evidence_digest(evidence) for evidence in checked.accepted_evidence
        ),
    )


def _evidence_matches_story(
    evidence: StoryVerificationEvidence,
    story: ScreenplayStory,
    plan: ScreenplayPlan,
) -> bool:
    return (
        evidence.story_screenplay_digest == screenplay_story_digest(story)
        and evidence.daily_report_version_id == plan.daily_report_version_id
        and evidence.policy_bundle_digest == plan.policy_bundle_digest
        and evidence.citation_article_version_ids == story.citation_article_version_ids
    )
