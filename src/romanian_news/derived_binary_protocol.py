from __future__ import annotations

import hashlib
from decimal import Decimal
from types import MappingProxyType

from romanian_news import Sha256
from romanian_news.analysis.binary_evaluation import BinaryQuestion
from romanian_news.binary_benchmark import BenchmarkId, BinaryBenchmarkDefinition

GROUPING_BINARY_QUESTION = BinaryQuestion(
    question_id="same_news_event",
    instructions=(
        "Estimate the probability that Article A and Article B describe the same concrete news "
        "event. Judge the event, not broad topic similarity."
    ),
    true_criteria=(
        "Both articles report the same occurrence, decision, announcement, transaction, "
        "publication, or direct update to that event. Different wording, outlets, detail, or "
        "publication times do not make the event different."
    ),
    false_criteria=(
        "The articles only share a person, organization, place, policy area, or continuing story, "
        "but report distinct occurrences, decisions, announcements, or developments."
    ),
    threshold=Decimal("0.5"),
)

RANKING_BINARY_QUESTION = BinaryQuestion(
    question_id="a_should_rank_above_b",
    instructions=(
        "Estimate the probability that Event A should appear above Event B in a concise Romanian "
        "national news report for a general reader."
    ),
    true_criteria=(
        "Event A has the stronger current Romanian political or economic consequence after "
        "considering reach, magnitude, certainty, realized effects, public-finance impact, and "
        "national or broad-population relevance."
    ),
    false_criteria=(
        "Event B has the stronger current Romanian political or economic consequence, or the "
        "evidence does not justify ranking Event A above Event B."
    ),
    threshold=Decimal("0.5"),
)

TIER_MAIN_BINARY_QUESTION = BinaryQuestion(
    question_id="main_subject",
    instructions=(
        "Estimate the probability that this subject belongs in the main section of today's concise "
        "Romanian national news report."
    ),
    true_criteria=(
        "The subject is among the day's strongest current Romanian political or economic stories, "
        "with broad or national consequence, material magnitude, or exceptional reader importance."
    ),
    false_criteria=(
        "The subject is useful but secondary, too narrow, weakly supported, or not important enough "
        "for the main section."
    ),
    threshold=Decimal("0.5"),
)

TIER_WORTH_KNOWING_BINARY_QUESTION = BinaryQuestion(
    question_id="worth_knowing_if_not_main",
    instructions=(
        "Assuming this subject is not in the main section, estimate the probability that it still "
        "belongs in a worth-knowing section rather than being excluded."
    ),
    true_criteria=(
        "The subject has a concrete, current and supported Romanian political or economic consequence "
        "that is useful to a general reader despite being secondary."
    ),
    false_criteria=(
        "The subject is redundant, trivial, too narrow, weakly supported, stale, or lacks enough "
        "Romanian consequence to include."
    ),
    threshold=Decimal("0.5"),
)

CONFIDENCE_BINARY_QUESTION = BinaryQuestion(
    question_id="evidence_sufficient",
    instructions=(
        "Estimate the probability that the supplied source evidence is sufficient to present the "
        "key claim without an uncertainty disclosure."
    ),
    true_criteria=(
        "The key claim is directly supported by specific, credible and mutually consistent source "
        "evidence with enough attribution and corroboration for the stated certainty."
    ),
    false_criteria=(
        "The key claim depends on weak attribution, a single uncertain assertion, material source "
        "disagreement, missing corroboration, or evidence that does not support the stated certainty."
    ),
    threshold=Decimal("0.5"),
)

DAILY_THEME_BINARY_QUESTION = BinaryQuestion(
    question_id="same_reader_subject",
    instructions=(
        "Estimate the probability that Event A and Event B belong under one reader subject in the "
        "same day's Romanian news report."
    ),
    true_criteria=(
        "The events are distinct updates that a reader would naturally understand as one coherent "
        "subject because they share the same underlying development, decision, consequence, or "
        "directly connected chain of events."
    ),
    false_criteria=(
        "The events only share a broad topic, actor, institution, place, or policy area and should be "
        "presented as separate reader subjects."
    ),
    threshold=Decimal("0.5"),
)

TIER_COMPOSITION = (
    "main when main_subject >= 0.5; otherwise worth_knowing when "
    "worth_knowing_if_not_main >= 0.5; otherwise excluded"
)
TIER_COMPOSITION_DIGEST: Sha256 = hashlib.sha256(TIER_COMPOSITION.encode()).hexdigest()


DERIVED_BINARY_CONCERNS = (
    BinaryBenchmarkDefinition(
        benchmark="grouping",
        case_count=23,
        scored_unit_count=23,
        questions=(GROUPING_BINARY_QUESTION,),
        calls_per_model_trial=23,
        state_format="two-anchor-articles-v1",
        metrics=("accuracy", "same_group_recall", "different_group_preservation"),
        limitation="Pair judgments do not establish a valid complete partition.",
    ),
    BinaryBenchmarkDefinition(
        benchmark="ranking",
        case_count=28,
        scored_unit_count=28,
        questions=(RANKING_BINARY_QUESTION,),
        calls_per_model_trial=28,
        state_format="two-groups-with-relevance-evidence-v1",
        metrics=("accuracy", "inversion_count"),
        limitation="Pairwise precedence does not establish a coherent global order.",
    ),
    BinaryBenchmarkDefinition(
        benchmark="tier",
        case_count=25,
        scored_unit_count=25,
        questions=(TIER_MAIN_BINARY_QUESTION, TIER_WORTH_KNOWING_BINARY_QUESTION),
        calls_per_model_trial=50,
        state_format="subject-with-report-context-v1",
        metrics=("accuracy", "confusion_matrix", "per_tier_recall"),
        limitation="Composed tier classification does not evaluate complete assessment generation.",
        composition_digest=TIER_COMPOSITION_DIGEST,
    ),
    BinaryBenchmarkDefinition(
        benchmark="confidence",
        case_count=4,
        scored_unit_count=4,
        questions=(CONFIDENCE_BINARY_QUESTION,),
        calls_per_model_trial=4,
        state_format="source-evidence-without-disclosure-v1",
        metrics=("accuracy", "error_ids"),
        limitation="Four cases support descriptive evidence only, not a deployment winner.",
    ),
    BinaryBenchmarkDefinition(
        benchmark="daily_theme",
        case_count=6,
        scored_unit_count=27,
        questions=(DAILY_THEME_BINARY_QUESTION,),
        calls_per_model_trial=27,
        state_format="two-groups-with-day-context-v1",
        metrics=("must_link_recall", "must_separate_preservation"),
        limitation="Pair judgments do not establish transitivity or report usefulness.",
    ),
)

DERIVED_BINARY_BENCHMARKS: MappingProxyType[BenchmarkId, BinaryBenchmarkDefinition] = (
    MappingProxyType({definition.benchmark: definition for definition in DERIVED_BINARY_CONCERNS})
)
