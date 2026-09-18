from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

from romanian_news import Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.catalog import evaluations as evaluation_catalog
from romanian_news.evaluation import (
    ExcludedConcernDisposition,
    ExecutableConcernDisposition,
    FeedbackReview,
    NewsEvaluationDataset,
    NewsEvaluationManifest,
    ReportEvaluationSpec,
    TierConcernDisposition,
)
from romanian_news.evaluation_curation import (
    RankingSpec,
    build_ranking_case,
    evaluation_case_spec,
    read_evaluation_artifact,
    read_evaluation_report_bundle,
)
from romanian_news.feedback import GroupFeedbackTarget

DATASET_VERSION = "news-evaluation-2026-09-11-v9"
REVIEWED_AT = datetime.fromisoformat("2026-09-11T08:30:00+00:00")
ISSUE_URL = "https://github.com/mauricedesaxe/chartly/issues/208"
DEFAULT_OUTPUT_PATH = Path("data/news/evaluations/news-evaluation-2026-09-11-v9-manifest.json")
V8_MANIFEST_VERSION_ID: Sha256 = "8102a1c71b83dacc896ff04f9490f8e13729754e52f17ef07760a969f8a07549"
V9_MANIFEST_VERSION_ID: Sha256 = "a9109efa64a6e2da258ef535b03f40a74a191c1fb52c8bec01f188dd688fd807"
V10_MANIFEST_VERSION_ID: Sha256 = "f64390550214eab0645fd96c382b38cfe1a8bc5f54d2b8f3b95b63f8d30e6178"
V10_DATASET_VERSION = "news-evaluation-2026-09-11-v10"
RESOLUTION_DATASET_VERSION = "news-evaluation-2026-09-11-v11"
RESOLUTION_REVIEWED_AT = datetime.fromisoformat("2026-09-11T10:20:00+00:00")
RESOLUTION_OUTPUT_PATH = Path("data/news/evaluations/news-evaluation-2026-09-11-v11-manifest.json")
REPORT_2026_09_02: Sha256 = "d66f0afd82c0dc2eea8a84ef5da082429f96e87ab266746374253aac1aacbbb4"
REPORT_2026_09_03: Sha256 = "f2e276ffa470a5b80ebf5fbf9fa1f098934c95ce0fcbf0f172eac9b15d93543c"

STB_FEEDBACK_ID = UUID("ebdc6e3f-7fa7-4e63-a296-1159bcb45f90")
STB_TIER_CASE_ID = "sep10-subject-13-tier"
BET_CHANGES_FEEDBACK_ID = UUID("099eabde-d07c-4740-ba53-42e5ee0afdc5")
BET_CHANGES_TIER_CASE_ID = "sep10-subject-18-tier"

SEPTEMBER_2_GROUPS = {
    "questioned_pnl_criticism": (
        "12cda1d4373890e575292f6b72219b3a1d43f7a4f19236fe3cc5d87e6561e354"
    ),
    "government_collapse": ("2f1c45ed3835695bdc49ac56c76425b0b033c5af346495cfc3fb931b38667ce2"),
}
SEPTEMBER_3_GROUPS = {
    "deficit_moratorium": ("8177f6bb36492538fdb0efbbacd37781e5922e39137ef04bd1696cfb3d16c559"),
    "opinion_led_commentary": ("1a6c0fe83840f2ad292a476cbf07e07be49d22d9c9bfd5403a9eaaf458c2bba6"),
    "irrelevant_climate_control": (
        "d3c5758cca9f46667b865e5449fec96db921c498cd3251c94b541c3cff4de926"
    ),
}

QUESTIONED_PNL_CRITICISM_FEEDBACK_ID = UUID("2e080f78-746b-42ab-9fe9-e67879c7df02")
DEFICIT_MORATORIUM_FEEDBACK_ID = UUID("2cef857b-3ca8-4978-807d-070e858fbbb7")

RANKING_SPECS = (
    RankingSpec(
        "sep02-government-collapse-ranks-above-questioned-pnl-criticism",
        str(QUESTIONED_PNL_CRITICISM_FEEDBACK_ID),
        REPORT_2026_09_02,
        SEPTEMBER_2_GROUPS["government_collapse"],
        SEPTEMBER_2_GROUPS["questioned_pnl_criticism"],
    ),
    RankingSpec(
        "sep03-deficit-moratorium-ranks-above-opinion-led-commentary",
        str(DEFICIT_MORATORIUM_FEEDBACK_ID),
        REPORT_2026_09_03,
        SEPTEMBER_3_GROUPS["deficit_moratorium"],
        SEPTEMBER_3_GROUPS["opinion_led_commentary"],
    ),
    RankingSpec(
        "sep03-deficit-moratorium-stays-above-irrelevant-climate-control",
        str(DEFICIT_MORATORIUM_FEEDBACK_ID),
        REPORT_2026_09_03,
        SEPTEMBER_3_GROUPS["deficit_moratorium"],
        SEPTEMBER_3_GROUPS["irrelevant_climate_control"],
        True,
    ),
)

QUESTIONED_PNL_CRITICISM_REVIEW = FeedbackReview(
    feedback_id=QUESTIONED_PNL_CRITICISM_FEEDBACK_ID,
    target=GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_02,
        group_id=SEPTEMBER_2_GROUPS["questioned_pnl_criticism"],
    ),
    concerns=(
        ExecutableConcernDisposition(
            concern="ranking",
            case_ids=("sep02-government-collapse-ranks-above-questioned-pnl-criticism",),
            rationale=(
                "The owner questions a single-article PNL criticism holding first place, so "
                "reviewed substantive government subjects must rank above it."
            ),
        ),
    ),
)
DEFICIT_MORATORIUM_REVIEW = FeedbackReview(
    feedback_id=DEFICIT_MORATORIUM_FEEDBACK_ID,
    target=GroupFeedbackTarget(
        report_version_id=REPORT_2026_09_03,
        group_id=SEPTEMBER_3_GROUPS["deficit_moratorium"],
    ),
    concerns=(
        ExecutableConcernDisposition(
            concern="ranking",
            case_ids=(
                "sep03-deficit-moratorium-ranks-above-opinion-led-commentary",
                "sep03-deficit-moratorium-stays-above-irrelevant-climate-control",
            ),
            rationale=(
                "The reviewed deficit-moratorium subject should have ranked higher, above "
                "opinion-led commentary, and must stay above reviewed irrelevant subjects."
            ),
        ),
    ),
)
NEW_FEEDBACK_REVIEWS = (
    QUESTIONED_PNL_CRITICISM_REVIEW,
    DEFICIT_MORATORIUM_REVIEW,
)


def _pinned_prior(
    version_id: Sha256, expected_version: str
) -> tuple[ArtifactReference, NewsEvaluationManifest]:
    reference = evaluation_catalog.read_news_evaluation_artifact_references((version_id,))[
        version_id
    ]
    manifest = NewsEvaluationManifest.model_validate_json(
        read_evaluation_artifact(reference), strict=True
    )
    if manifest.version != expected_version:
        raise ValueError(f"Pinned prior manifest has unexpected version: {manifest.version}")
    return reference, manifest


def _require_reviewed_frozen_reports(prior: NewsEvaluationManifest | NewsEvaluationDataset) -> None:
    report_version_ids = {spec.report.version_id for spec in prior.reports}
    if not {REPORT_2026_09_02, REPORT_2026_09_03} <= report_version_ids:
        raise ValueError("Pinned prior release must freeze the reviewed September 2 and 3 reports")


@dataclass(frozen=True)
class TierConflictResolution:
    """One reviewed merge rule that retires a standalone tier judgment."""

    feedback_id: UUID
    tier_case_id: str
    standalone_note: str
    merged_subject: str


TIER_CONFLICT_RESOLUTIONS = (
    TierConflictResolution(
        feedback_id=BET_CHANGES_FEEDBACK_ID,
        tier_case_id=BET_CHANGES_TIER_CASE_ID,
        standalone_note="The standalone subject is useful financial context",
        merged_subject=(
            "once grouped with the BET movement and BVB decline into the reviewed "
            "financial-signals subject, the subject keeps their reviewed main tier"
        ),
    ),
)


def _resolve_tier_conflicts(review: FeedbackReview) -> FeedbackReview:
    """Retire standalone tier judgments that the reviewed grouping merges supersede."""
    resolutions = tuple(
        item for item in TIER_CONFLICT_RESOLUTIONS if item.feedback_id == review.feedback_id
    )
    if not resolutions:
        return review
    concerns = []
    resolved = False
    for disposition in review.concerns:
        if not isinstance(disposition, TierConcernDisposition):
            concerns.append(disposition)
            continue
        matched = next(
            (item for item in resolutions if disposition.case_ids == (item.tier_case_id,)),
            None,
        )
        if matched is None:
            concerns.append(disposition)
            continue
        if disposition.judgment != "worth_knowing":
            raise ValueError(
                "Reviewed merge rule expects the retired standalone tier judgment: "
                f"{matched.tier_case_id}"
            )
        resolved = True
        concerns.append(
            ExcludedConcernDisposition(
                concern="tier",
                reason="conditional",
                rationale=f"{matched.standalone_note}; {matched.merged_subject}.",
            )
        )
    if not resolved:
        raise ValueError(
            "Pinned release does not carry the reviewed standalone tier judgment: "
            f"{resolutions[0].tier_case_id}"
        )
    return review.model_copy(update={"concerns": tuple(concerns)})


def build_resolved_tier_evaluation_manifest() -> NewsEvaluationManifest:
    """Resolve the reviewed financial-signals tier conflict onto the pinned v10 release."""
    prior_reference, prior = _pinned_prior(V10_MANIFEST_VERSION_ID, V10_DATASET_VERSION)
    _require_reviewed_frozen_reports(prior)
    resolved_case_ids = {item.tier_case_id for item in TIER_CONFLICT_RESOLUTIONS}
    cases = tuple(case for case in prior.cases if case.case_id not in resolved_case_ids)
    if len(cases) != len(prior.cases) - len(resolved_case_ids):
        raise ValueError("Pinned release does not carry the reviewed tier cases")
    reviews = tuple(_resolve_tier_conflicts(review) for review in prior.feedback_reviews)
    manifest = NewsEvaluationManifest(
        version=RESOLUTION_DATASET_VERSION,
        reviewed_at=RESOLUTION_REVIEWED_AT,
        issue_url=ISSUE_URL,
        prior_manifest=prior_reference,
        source_feedback_ids=prior.source_feedback_ids,
        reports=prior.reports,
        cases=cases,
        feedback_reviews=reviews,
    )
    return NewsEvaluationManifest.model_validate_json(manifest.model_dump_json(), strict=True)


def build_national_consequence_evaluation_manifest() -> NewsEvaluationManifest:
    """Freeze the reviewed September 2 and 3 ranking comparisons onto the V8 release."""
    prior_reference, prior_manifest = _pinned_prior(
        V8_MANIFEST_VERSION_ID, "news-evaluation-2026-09-10-v8"
    )
    prior = evaluation_catalog.hydrate_news_evaluation_manifest(prior_manifest)
    _require_reviewed_frozen_reports(prior)
    bundles = {
        report_version_id: read_evaluation_report_bundle(report_version_id)
        for report_version_id in (REPORT_2026_09_02, REPORT_2026_09_03)
    }
    known_groups = {
        group_id for bundle in bundles.values() for group_id in bundle.snapshot.report_group_ids
    }
    anchors = (
        *SEPTEMBER_2_GROUPS.values(),
        *SEPTEMBER_3_GROUPS.values(),
    )
    if not set(anchors) <= known_groups:
        raise ValueError("Reviewed ranking anchors are absent from their frozen reports")

    dataset = NewsEvaluationDataset(
        version=DATASET_VERSION,
        reviewed_at=REVIEWED_AT,
        issue_url=ISSUE_URL,
        source_feedback_ids=(
            *prior.source_feedback_ids,
            QUESTIONED_PNL_CRITICISM_FEEDBACK_ID,
            DEFICIT_MORATORIUM_FEEDBACK_ID,
        ),
        reports=prior.reports,
        cases=(
            *prior.cases,
            *(build_ranking_case(spec, bundles[spec.report_version_id]) for spec in RANKING_SPECS),
        ),
        feedback_reviews=(*prior.feedback_reviews, *NEW_FEEDBACK_REVIEWS),
    )
    manifest = NewsEvaluationManifest(
        version=dataset.version,
        reviewed_at=dataset.reviewed_at,
        issue_url=dataset.issue_url,
        prior_manifest=prior_reference,
        source_feedback_ids=dataset.source_feedback_ids,
        reports=tuple(
            ReportEvaluationSpec(
                report=report.report,
                themes=report.themes,
                cluster_set=report.cluster_set,
            )
            for report in dataset.reports
        ),
        cases=tuple(evaluation_case_spec(case) for case in dataset.cases),
        feedback_reviews=dataset.feedback_reviews,
    )
    return NewsEvaluationManifest.model_validate_json(manifest.model_dump_json(), strict=True)


def write_national_consequence_evaluation_manifest(
    path: Path = RESOLUTION_OUTPUT_PATH,
) -> NewsEvaluationManifest:
    """Write deterministic reference-only JSON without publishing it."""
    manifest = build_resolved_tier_evaluation_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--output", type=Path, default=RESOLUTION_OUTPUT_PATH)
    arguments = parser.parse_args(argv)
    output: object = getattr(arguments, "output", None)
    if not isinstance(output, Path):
        raise TypeError("Evaluation output must be a path")
    _ = write_national_consequence_evaluation_manifest(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
