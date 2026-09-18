from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from romanian_news import Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.catalog import evaluations as evaluation_catalog
from romanian_news.evaluation import (
    ConfidenceConcernDisposition,
    ContextConcernDisposition,
    DailyThemeJudgment,
    ExcludedConcernDisposition,
    ExecutableConcernDisposition,
    ExploratoryConcernDisposition,
    FeedbackConcernDisposition,
    FeedbackReview,
    NewsEvaluationCase,
    NewsEvaluationDataset,
    NewsEvaluationManifest,
    OverallReportConcernDisposition,
    RelevanceConcernDisposition,
    ReportEvaluationSnapshot,
    ReportEvaluationSpec,
    SupersededConcernDisposition,
    ThemeEvaluationDayCase,
    TierConcernDisposition,
    UnreviewedTheme,
)
from romanian_news.evaluation_curation import (
    ConfidenceSpec,
    RankingSpec,
    RelevanceSpec,
    TierSpec,
    build_confidence_case,
    build_ranking_case,
    build_relevance_cases,
    build_theme_day_cases,
    build_tier_case,
    evaluation_case_spec,
    read_evaluation_artifact,
    read_evaluation_report_bundle,
)
from romanian_news.feedback import NewsFeedbackEvent, ReportFeedbackTarget, ThemeFeedbackTarget

DATASET_VERSION = "news-evaluation-2026-09-10-v7"
REVIEWED_AT = datetime.fromisoformat("2026-09-10T10:54:35.937122+00:00")
ISSUE_URL = "https://github.com/mauricedesaxe/chartly/issues/370"
DEFAULT_OUTPUT_PATH = Path("data/news/evaluations/news-evaluation-2026-09-10-v7-manifest.json")
REPORT_VERSION_ID: Sha256 = "c1369f9a24202a111dfc07e29a7194c73cb22dd491958093932a5d3b9b270d72"
PRIOR_MANIFEST_REFERENCE = ArtifactReference(
    artifact_id="news:evaluation-manifest:news-evaluation-2026-09-07-v6.3",
    version_id="84dc7bc276ee25fae607872a408cf8dbd0b95f45e0aedbc2aee59b301c039f30",
    content_digest="56ec12d369b60b55bfa03c6199ca00d1bdccc6aeb801de65f89a95520ad45f11",
    r2_key=(
        "news/evaluations/manifests/news-evaluation-2026-09-07-v6.3/"
        "56ec12d369b60b55bfa03c6199ca00d1bdccc6aeb801de65f89a95520ad45f11.json"
    ),
)
REPORT_REFERENCE = ArtifactReference(
    artifact_id="news:daily:2026-09-10",
    version_id=REPORT_VERSION_ID,
    content_digest="4682275f72c6f3ac1e6e1d50a72aaa5175a82f256cf1217275d69f0c0b0fe30e",
    r2_key=(
        "news/reports/daily/2026-09-10/"
        "4682275f72c6f3ac1e6e1d50a72aaa5175a82f256cf1217275d69f0c0b0fe30e.json"
    ),
)
THEMES_REFERENCE = ArtifactReference(
    artifact_id="news:themes:2026-09-10",
    version_id="1937abb01ab206d63712a0e232c20217e89caafae508592c1f43aae90b9b4907",
    content_digest="5247242a52e4df7a128330aecc00a64c84144678574a1b1410d05b0d262cd99c",
    r2_key=(
        "news/derived/themes/2026-09-10/"
        "5247242a52e4df7a128330aecc00a64c84144678574a1b1410d05b0d262cd99c.json"
    ),
)
CLUSTER_REFERENCE = ArtifactReference(
    artifact_id="news:clusters:2026-09-10",
    version_id="8990bb875216d480395cd31c6ea1b5938f7027ef0a2a2eb1e096d8cf399b9612",
    content_digest="c16542cb0a7aa5791e01bde994c301099fd129a526f29f4ef70584d287b782af",
    r2_key=(
        "news/clusters/2026-09-10/"
        "c16542cb0a7aa5791e01bde994c301099fd129a526f29f4ef70584d287b782af.json"
    ),
)

GROUP_IDS: dict[int, Sha256] = {
    1: "2f753553920f1fa3e83844ba034017cfa78876209d404b9aedc0cab902c65435",
    2: "66ef905c0071681d47d643395c239719522695746714ed3b0c7d0b66da94d604",
    3: "6cda0f91399d00136ffd5c17c9920ced1369399e19bfcbb58bbb8bd7084a3753",
    4: "6ff7f531f9adf2485ce61c77c71af37b32b2a3148900da7c8eff136099a09f9e",
    5: "74e178cefeebb5b78d430f3bcd8d1d5ddd35b5d274f6bad75b4949b944c39bb3",
    6: "8822607cf653251d8467841d76b808980497ab074d541b3666ebdbeec7994af5",
    7: "bc6084265043065c2712bcd4c771136920ebeff1cfe20dab521857d543b0b975",
    8: "bebd70a8c104fe91382c5d9ab2b59a5d5a6159fa4ca8325a3e01c856d1431009",
    9: "e914cddbd0779016a1a3f27d41eb1da4cb1ada9dacb10d6a0b2357c2e9eba8d2",
    10: "0a36d0918760f3f8e625393abb875936dfdbc52db6d732b531bee8c51d33070c",
    11: "15c68235640519f5dc9e052e8e50ce36e4da9a0642db0b081646324b2b137f78",
    12: "2a503e9c4fb893b9b8f661e72a8070da6730d1a1ed864f2cf2f6fa0ecb1e83fa",
    13: "2c876749d928c2fd12a4c70128187694844999545bbf505b8303177c43f3045a",
    14: "4f03df02aee948f7762b6e03d198487d9d41df0f73c14860022167b6055acbab",
    15: "631f89ef42f55a3f369be69b44f52b4612de3130c7ace0ffdc6fb85e80a107f4",
    16: "6e797ec3180b58e3ef74843f12c5b34da718d644b41f22aec78ded24e080ccef",
    17: "6ea2781246603d2a8088f7f19ef368414c9261a9cdd7bd2a35a5306bcf127311",
    18: "92dec2a38bdb39cb00d897095bca151a8d12278f451994034c00e36d77a970dc",
    19: "9a1b3bc42c932259b5784e29b084d74706876b438af45e70d7f11596f8333c29",
    20: "d246705f2645024aeeb9f30c37f2681b5c73a208f4d02454ccfa08db306f7e60",
    21: "f8007939d98f943416ede68429e027926b328e54cc83ec1187345b4c2ef8377f",
    22: "fa78ad8beb9d7c37021f3fbbf543d7b47467468583bec1a470a46fa9dc573188",
    23: "fb0ffdf538a319686187e72f7ae62d2d3f087e56d4e2e0bd8b0d27941486594b",
    24: "37aebe0473f1e5d9239f098b309d6af4d35949c58084d72342d1ef5895a71398",
    25: "4b3ed43f0fc3a6b53c5f950d5cced7e9eaf3253076001b8def7bfa66def9a21c",
    26: "567e6b2761fb08a3db8a62efc3dd7d11b62dc26eee5c9a255c3947e3456f49a6",
    27: "b21aad4160edfabb987fc1b92b170283bde11b47e49e5adc5d18d95436471839",
    28: "b8fb51d161d612803b3b4787099a235c528f7c02e846af12eb59bc443f9ecd7d",
    29: "be905d44a73ea5c65604a421c1a83536800b0c7e370d0bf02276ec416bec7d32",
}
THEME_IDS: dict[int, Sha256] = {
    1: "7e1b1922151b5b30c65b3e858f14f52bc553388a869da87c01f230c53c6e1bf9",
    2: "dada69d9ba1725f57684484b103f42894ba213c9651eb2616f63176fbcacfa78",
    3: "5175e5bcae84f9041c809fdee99d446a5669644961b2c7d46ac130395aed498a",
    4: "0b8ef6f51c5c9ae80759ec9938683282e43665884dd07bc35e8091fd049d79dd",
    5: "6d403ed62c3628aeff90a8590f8e0cff97ac5644d37dc09ce56afa2446586f2d",
    6: "eb915d0408a70e92def91f9ca8fb52e29c92d77330fdab3dcee9a4ba7b71a2a9",
    7: "ae853e078e170d3cf0956078b3f3815d9fe7a8c0b453a8205aac19a78cd2fba8",
    8: "87ec780ea596f0c3b59e6a43f9b69477117a73581d878cc3c4624b5471e0bc21",
    9: "a033164e80b75c1969b59c2b2f63364de6717bda8dcb814c69a2a8b7304f62ef",
    10: "1e977ec5b30e6ff8b03c7cda65d6e311680c9e2d6c9feff891a92a1b3fe79e06",
    11: "f89e07c71db420f3a1d19d4c8166bb135c12dd0f71ce8fe9042cfded93e010e6",
    12: "9905287b60f029bd25872cbcd3dde8e4169ce64f7e4010e44cb120624c53d968",
    13: "2a5ba6dd2787ec1f02e9f136c86c0f0ec8d621b29885e76c1556fcbe3b85726c",
    14: "6828233aea8a5c5fe39e087d46c885f0fb67a6428b0bde8c55da55eb74777633",
    15: "e96e1de1b36cef0a17e034628e8fbdb87367cc239148bf8fad32245a2dc10bbb",
    16: "5a7ac58c376c970a0ad86f7a44e5549fc5899bccb45c2c32e6fdef7b44e12e11",
    17: "9dae7ca2a639894a9a051b5870b83b53dabbe65141031904728a3410709daa74",
    18: "bbf5ad7a1cbd5bb56d6d1b7b4742f52a6f5914726da2c52a4f1515f84833b04e",
    19: "8bef8c1c383ffeb5d24bef2b8cf8031a464aef426d9344d73ec180aebcd8ad4d",
    20: "981fb8af0223fc7ed076cbc339d5c111ffba3e64d5802bf5d4c4979833165d3c",
    21: "f595199ed100203e69d3b6951677e127cc2d105be604d8c11112fa3270c64e96",
    22: "cc0572d2f989e8c986b57b1bbf736380a43b3bf79b1cc55d74f666b6e3fbfd6b",
    23: "e5ccc9821b2dfc0117bed303f61d07036f6e6c4d0cbaf181edabd59aba35fb37",
    24: "fc48d613ba54a369e2d39af390a292e2bea88e85c6b6b57d07d44e8c4f9166b8",
    25: "27d3127ca16a8a383e51c71a43b849e95b8b71762d97015074e2c19d49be4092",
    26: "55fd425f94e3cd3e63d528cdfda9955f774154ba3deb27de93474a3782ef21e6",
    27: "72983a5c1e6c10d6047a433179a3faf31bee7254d1d66b342ae0d79a734bf35a",
    28: "cc56cdc0f79b70422a40f38ab9ee2f79ee813f6aa8cf431c5057d9ab71b8677d",
    29: "1433c5e7cd386f449ec5c01e29196e0aff34d53fac739a8772785e705e320c82",
}

CURRENT_FEEDBACK_IDS: dict[int, str] = {
    1: "4f08b504-ea65-4baa-93f0-9aec04d69753",
    2: "351ede8c-9398-4051-afd3-8ac9a3ef3f55",
    3: "37399f0e-f030-45f5-b5cb-fb32c0f5af3c",
    4: "5fcfb650-9c77-4604-9f04-656a6e31a6f9",
    5: "12712b04-127f-4260-83bf-dd923ffc2213",
    6: "b2f92241-5a4a-4425-be07-5f646b42ddbc",
    7: "736f39d8-4b30-4da9-820e-bda2c29f5917",
    8: "a44faaff-9135-43fb-b749-c7e71914b804",
    9: "2b82f2e5-1bf6-48bf-bf64-caaf953fd6fb",
    10: "e2f3019c-423f-4ea8-a66b-ce22d6678164",
    11: "0c071aa1-9cb0-4ac4-bf78-fa36f3114349",
    12: "44a396e9-8613-4001-9b5f-384b613cafb5",
    13: "ebdc6e3f-7fa7-4e63-a296-1159bcb45f90",
    14: "b19a8101-5c35-4e31-b88e-3e5a7c9261c4",
    15: "650de0b3-3307-4f60-a24d-3c452f46ac87",
    16: "f844f25c-02db-4726-9913-13e0045a608c",
    18: "099eabde-d07c-4740-ba53-42e5ee0afdc5",
    19: "9ee26633-222a-4445-b934-90532b6c0551",
    20: "aba588b6-5411-45aa-8457-7e0bde6f7408",
    21: "33c1dccc-e4d6-4ec5-9775-7c5a53fd952b",
    22: "fe4c9127-8c7b-4528-942a-07ed1a9d58be",
    23: "cf08cc52-5f32-47a2-8de4-e3d687383960",
    24: "a395ade1-9f1d-4a3c-87dd-17fe3fcf3328",
    25: "31ee14fc-e827-4c82-a3ab-f23afb8739b6",
    26: "6c92cf32-87cb-43c0-aee3-0428699df6f2",
    27: "7dab7cb3-b7ce-4e9a-b7d0-1cfddabeb122",
    28: "1e5226d4-f060-428f-a5b6-7713821cb565",
    29: "c2619b7c-fb8b-4dfa-8ebe-3012b5e8f5b9",
}

_RELEVANCE_ARTICLE_COUNTS = {1: 2, 9: 3, 25: 2}
_FIRM_RELEVANCE = {
    1: True,
    4: True,
    5: True,
    6: True,
    7: True,
    8: True,
    9: True,
    10: True,
    11: True,
    12: True,
    13: True,
    14: True,
    15: True,
    18: True,
    19: True,
    20: True,
    23: False,
    24: True,
    25: True,
    26: True,
    27: True,
    28: True,
    29: True,
}
RELEVANCE_SPECS = tuple(
    RelevanceSpec(
        feedback_id=CURRENT_FEEDBACK_IDS[position],
        report_version_id=REPORT_VERSION_ID,
        group_id=GROUP_IDS[position],
        expected_accepted=expected,
        control=expected,
        case_ids=tuple(
            f"sep10-subject-{position:02d}-article-{article_position}-relevance"
            for article_position in range(1, _RELEVANCE_ARTICLE_COUNTS.get(position, 1) + 1)
        ),
    )
    for position, expected in _FIRM_RELEVANCE.items()
)

_TIER_JUDGMENTS: dict[int, Literal["main", "worth_knowing", "excluded"]] = {
    1: "worth_knowing",
    2: "worth_knowing",
    3: "worth_knowing",
    4: "main",
    5: "main",
    6: "main",
    7: "main",
    8: "main",
    9: "main",
    10: "worth_knowing",
    11: "worth_knowing",
    12: "worth_knowing",
    13: "worth_knowing",
    14: "main",
    15: "main",
    16: "excluded",
    18: "worth_knowing",
    19: "worth_knowing",
    20: "worth_knowing",
    22: "excluded",
    23: "excluded",
    24: "main",
    25: "main",
    26: "main",
    27: "worth_knowing",
    28: "main",
    29: "main",
}
TIER_SPECS = tuple(
    TierSpec(
        case_id=f"sep10-subject-{position:02d}-tier",
        feedback_id=CURRENT_FEEDBACK_IDS[position],
        report_version_id=REPORT_VERSION_ID,
        group_id=GROUP_IDS[position],
        expected_tier=tier,
        control=tier == "main",
    )
    for position, tier in _TIER_JUDGMENTS.items()
)

CONFIDENCE_SPECS = tuple(
    ConfidenceSpec(
        case_id=f"sep10-subject-{position:02d}-confidence",
        feedback_id=CURRENT_FEEDBACK_IDS[position],
        report_version_id=REPORT_VERSION_ID,
        group_id=GROUP_IDS[position],
        expected_sufficient=False,
    )
    for position in (16, 21, 22, 24)
)

_RELEVANCE_CASE_IDS = {spec.feedback_id: spec.case_ids for spec in RELEVANCE_SPECS}
_TIER_CASE_IDS = {spec.feedback_id: (spec.case_id,) for spec in TIER_SPECS}
_CONFIDENCE_CASE_IDS = {spec.feedback_id: (spec.case_id,) for spec in CONFIDENCE_SPECS}

RANKING_SPECS = (
    RankingSpec(
        "sep10-pnrr-ranks-above-research-funding",
        "b2f92241-5a4a-4425-be07-5f646b42ddbc",
        REPORT_VERSION_ID,
        GROUP_IDS[6],
        GROUP_IDS[1],
    ),
    RankingSpec(
        "sep10-pnrr-ranks-above-aur-afd",
        "b2f92241-5a4a-4425-be07-5f646b42ddbc",
        REPORT_VERSION_ID,
        GROUP_IDS[6],
        GROUP_IDS[2],
    ),
    RankingSpec(
        "sep10-pnrr-ranks-above-cyber-readiness",
        "b2f92241-5a4a-4425-be07-5f646b42ddbc",
        REPORT_VERSION_ID,
        GROUP_IDS[6],
        GROUP_IDS[3],
    ),
    RankingSpec(
        "sep10-pnrr-ranks-above-metrorex",
        "b2f92241-5a4a-4425-be07-5f646b42ddbc",
        REPORT_VERSION_ID,
        GROUP_IDS[6],
        GROUP_IDS[4],
    ),
    RankingSpec(
        "sep10-pnrr-ranks-above-hydropower-law",
        "b2f92241-5a4a-4425-be07-5f646b42ddbc",
        REPORT_VERSION_ID,
        GROUP_IDS[6],
        GROUP_IDS[5],
    ),
    RankingSpec(
        "sep10-pisa-ranks-above-aur-afd",
        "2b82f2e5-1bf6-48bf-bf64-caaf953fd6fb",
        REPORT_VERSION_ID,
        GROUP_IDS[9],
        GROUP_IDS[2],
    ),
    RankingSpec(
        "sep10-pisa-ranks-above-cyber-readiness",
        "2b82f2e5-1bf6-48bf-bf64-caaf953fd6fb",
        REPORT_VERSION_ID,
        GROUP_IDS[9],
        GROUP_IDS[3],
    ),
    RankingSpec(
        "sep10-job-losses-rank-above-aur-afd",
        "650de0b3-3307-4f60-a24d-3c452f46ac87",
        REPORT_VERSION_ID,
        GROUP_IDS[15],
        GROUP_IDS[2],
    ),
    RankingSpec(
        "sep10-prime-minister-ranks-above-aur-afd",
        "a395ade1-9f1d-4a3c-87dd-17fe3fcf3328",
        REPORT_VERSION_ID,
        GROUP_IDS[24],
        GROUP_IDS[2],
    ),
    RankingSpec(
        "sep10-fidelis-ranks-above-aur-afd",
        "31ee14fc-e827-4c82-a3ab-f23afb8739b6",
        REPORT_VERSION_ID,
        GROUP_IDS[25],
        GROUP_IDS[2],
    ),
    RankingSpec(
        "sep10-rates-rank-above-aur-afd",
        "6c92cf32-87cb-43c0-aee3-0428699df6f2",
        REPORT_VERSION_ID,
        GROUP_IDS[26],
        GROUP_IDS[2],
    ),
    RankingSpec(
        "sep10-bvb-ranks-above-aur-afd",
        "1e5226d4-f060-428f-a5b6-7713821cb565",
        REPORT_VERSION_ID,
        GROUP_IDS[28],
        GROUP_IDS[2],
    ),
    RankingSpec(
        "sep10-euro-ranks-above-aur-afd",
        "c2619b7c-fb8b-4dfa-8ebe-3012b5e8f5b9",
        REPORT_VERSION_ID,
        GROUP_IDS[29],
        GROUP_IDS[2],
    ),
)

THEME_JUDGMENTS = (
    DailyThemeJudgment(
        judgment_id="sep10-metrorex-and-stb-share-public-transport-theme",
        feedback_ids=(UUID("ebdc6e3f-7fa7-4e63-a296-1159bcb45f90"),),
        report_version_id=REPORT_VERSION_ID,
        left_group_id=GROUP_IDS[4],
        right_group_id=GROUP_IDS[13],
        expected_same_theme=True,
        rationale="Metrorex and STB describe related financial stress in Bucharest public transport.",
    ),
)


def build_september_10_evaluation_manifest() -> NewsEvaluationManifest:
    """Build v7 from the exact v6.3 release and September 10 production report."""
    prior_manifest = NewsEvaluationManifest.model_validate_json(
        read_evaluation_artifact(PRIOR_MANIFEST_REFERENCE), strict=True
    )
    if prior_manifest.version != "news-evaluation-2026-09-07-v6.3":
        raise ValueError(f"Pinned prior manifest has unexpected version: {prior_manifest.version}")
    prior_dataset = evaluation_catalog.hydrate_news_evaluation_manifest(prior_manifest)
    bundle = read_evaluation_report_bundle(REPORT_VERSION_ID)
    _require_exact_report(bundle.snapshot)
    feedback = evaluation_catalog.read_news_evaluation_feedback(SOURCE_FEEDBACK_IDS)
    require_exact_september_10_feedback(feedback)

    theme_cases = build_theme_day_cases(
        (*prior_dataset.reports, bundle.snapshot),
        THEME_JUDGMENTS,
        frozenset(SOURCE_FEEDBACK_IDS),
        {REPORT_VERSION_ID: THEMES_REFERENCE},
    )
    dataset = NewsEvaluationDataset(
        version=DATASET_VERSION,
        reviewed_at=REVIEWED_AT,
        issue_url=ISSUE_URL,
        source_feedback_ids=SOURCE_FEEDBACK_IDS,
        reports=(*prior_dataset.reports, bundle.snapshot),
        cases=(
            *_without_inherited_feedback_provenance(prior_dataset.cases),
            *(case for spec in RELEVANCE_SPECS for case in build_relevance_cases(spec, bundle)),
            *(build_ranking_case(spec, bundle) for spec in RANKING_SPECS),
            *(build_tier_case(spec, bundle) for spec in TIER_SPECS),
            *(build_confidence_case(spec, bundle) for spec in CONFIDENCE_SPECS),
            *theme_cases,
        ),
        feedback_reviews=FEEDBACK_REVIEWS,
        unreviewed_themes=UNREVIEWED_THEMES,
    )
    manifest = NewsEvaluationManifest(
        version=dataset.version,
        reviewed_at=dataset.reviewed_at,
        issue_url=dataset.issue_url,
        prior_manifest=PRIOR_MANIFEST_REFERENCE,
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
        unreviewed_themes=dataset.unreviewed_themes,
    )
    return NewsEvaluationManifest.model_validate_json(manifest.model_dump_json(), strict=True)


def write_september_10_evaluation_manifest(
    path: Path = DEFAULT_OUTPUT_PATH,
) -> NewsEvaluationManifest:
    """Write deterministic reference-only JSON without publishing it."""
    manifest = build_september_10_evaluation_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _without_inherited_feedback_provenance(
    cases: tuple[NewsEvaluationCase, ...],
) -> tuple[NewsEvaluationCase, ...]:
    stripped: list[NewsEvaluationCase] = []
    for case in cases:
        provenance = case.provenance.model_copy(update={"feedback_ids": ()})
        if isinstance(case, ThemeEvaluationDayCase):
            stripped.append(
                case.model_copy(
                    update={
                        "provenance": provenance,
                        "expectations": tuple(
                            expectation.model_copy(update={"feedback_ids": ()})
                            for expectation in case.expectations
                        ),
                    }
                )
            )
        else:
            stripped.append(case.model_copy(update={"provenance": provenance}))
    return tuple(stripped)


def _require_exact_report(snapshot: ReportEvaluationSnapshot) -> None:
    if (
        snapshot.report != REPORT_REFERENCE
        or snapshot.themes != THEMES_REFERENCE
        or snapshot.cluster_set != CLUSTER_REFERENCE
    ):
        raise ValueError("September 10 report lineage differs from the reviewed release")
    if tuple(snapshot.report_group_ids) != tuple(GROUP_IDS.values()):
        raise ValueError("September 10 subject order or group identity changed")


def require_exact_september_10_feedback(feedback: tuple[NewsFeedbackEvent, ...]) -> None:
    """Verify the exact reviewed events, targets, and expansion order."""
    rows = {row.feedback_id: row for row in feedback}
    if len(feedback) != 32 or set(rows) != set(SOURCE_FEEDBACK_IDS):
        raise ValueError("PostgreSQL did not return the exact reviewed 32-event source")
    reviews = {review.feedback_id: review for review in FEEDBACK_REVIEWS}
    for feedback_id, review in reviews.items():
        if rows[feedback_id].target != review.target:
            raise ValueError(f"Reviewed feedback target changed: {feedback_id}")
        disposition = review.concerns[0]
        if disposition.kind != "superseded":
            continue
        replacement = rows[disposition.superseded_by_feedback_id]
        if replacement.target != review.target:
            raise ValueError(f"Superseding feedback changed target: {feedback_id}")
        if (replacement.created_at, str(replacement.feedback_id)) <= (
            rows[feedback_id].created_at,
            str(feedback_id),
        ):
            raise ValueError(f"Superseding feedback is not newer: {feedback_id}")


def _executable(
    concern: Literal["ranking", "grouping"],
    case_ids: tuple[str, ...],
    rationale: str,
) -> ExecutableConcernDisposition:
    return ExecutableConcernDisposition(
        concern=concern,
        case_ids=case_ids,
        rationale=rationale,
    )


def _relevance(
    feedback_id: str,
    judgment: Literal["relevant", "irrelevant"],
    rationale: str,
) -> RelevanceConcernDisposition:
    return RelevanceConcernDisposition(
        judgment=judgment,
        case_ids=_RELEVANCE_CASE_IDS[feedback_id],
        rationale=rationale,
    )


def _tier(
    feedback_id: str,
    judgment: Literal["main", "worth_knowing", "excluded"],
    rationale: str,
) -> TierConcernDisposition:
    return TierConcernDisposition(
        judgment=judgment,
        case_ids=_TIER_CASE_IDS[feedback_id],
        rationale=rationale,
    )


def _confidence(
    feedback_id: str,
    judgment: Literal["supported", "insufficient"],
    rationale: str,
) -> ConfidenceConcernDisposition:
    return ConfidenceConcernDisposition(
        judgment=judgment,
        case_ids=_CONFIDENCE_CASE_IDS[feedback_id],
        rationale=rationale,
    )


def _context(
    judgment: Literal["sufficient", "needed"], rationale: str
) -> ContextConcernDisposition:
    return ContextConcernDisposition(judgment=judgment, rationale=rationale)


def _research(
    topics: tuple[
        Literal[
            "coverage_review",
            "entity_context",
            "external_corroboration",
            "financial_data",
            "historical_data",
            "investigative_report",
        ],
        ...,
    ],
    rationale: str,
) -> ExploratoryConcernDisposition:
    return ExploratoryConcernDisposition(topics=topics, rationale=rationale)


def _excluded(
    concern: Literal[
        "relevance",
        "ranking",
        "grouping",
        "tier",
        "confidence",
        "context",
        "research",
        "overall_report",
    ],
    reason: Literal["ambiguous", "conditional", "unsupported"],
    rationale: str,
) -> ExcludedConcernDisposition:
    return ExcludedConcernDisposition(concern=concern, reason=reason, rationale=rationale)


def _theme_review(
    feedback_id: str,
    position: int,
    concerns: tuple[FeedbackConcernDisposition, ...],
) -> FeedbackReview:
    return FeedbackReview(
        feedback_id=UUID(feedback_id),
        target=ThemeFeedbackTarget(
            report_version_id=REPORT_VERSION_ID,
            theme_id=THEME_IDS[position],
        ),
        concerns=concerns,
    )


def _superseded_theme_review(
    feedback_id: str,
    replacement_id: str,
    position: int,
) -> FeedbackReview:
    return _theme_review(
        feedback_id,
        position,
        (
            SupersededConcernDisposition(
                superseded_by_feedback_id=UUID(replacement_id),
            ),
        ),
    )


FEEDBACK_REVIEWS = (
    _theme_review(
        "4f08b504-ea65-4baa-93f0-9aec04d69753",
        1,
        (
            _relevance(
                "4f08b504-ea65-4baa-93f0-9aec04d69753",
                "relevant",
                "The research funding subject is useful and relevant.",
            ),
            _tier(
                "4f08b504-ea65-4baa-93f0-9aec04d69753",
                "worth_knowing",
                "The subject is useful but not among the day's highest-impact news.",
            ),
        ),
    ),
    FeedbackReview(
        feedback_id=UUID("4f87f587-61f0-4259-881e-ab836a459977"),
        target=ReportFeedbackTarget(report_version_id=REPORT_VERSION_ID),
        concerns=(
            OverallReportConcernDisposition(
                judgment="needs_tiering",
                rationale="The flat list does not contain 29 equally high-impact subjects.",
            ),
        ),
    ),
    _theme_review(
        "351ede8c-9398-4051-afd3-8ac9a3ef3f55",
        2,
        (
            _tier(
                "351ede8c-9398-4051-afd3-8ac9a3ef3f55",
                "worth_knowing",
                "The alliance is a low-impact side subject.",
            ),
            _excluded(
                "relevance",
                "conditional",
                "The review calls the subject borderline irrelevant rather than making a firm relevance judgment.",
            ),
        ),
    ),
    _theme_review(
        "37399f0e-f030-45f5-b5cb-fb32c0f5af3c",
        3,
        (
            _tier(
                "37399f0e-f030-45f5-b5cb-fb32c0f5af3c",
                "worth_knowing",
                "Cyber preparedness is interesting but not high impact.",
            ),
        ),
    ),
    _theme_review(
        "5fcfb650-9c77-4604-9f04-656a6e31a6f9",
        4,
        (
            _relevance(
                "5fcfb650-9c77-4604-9f04-656a6e31a6f9",
                "relevant",
                "Metrorex disruption is relevant to Bucharest and signals national conditions.",
            ),
            _tier(
                "5fcfb650-9c77-4604-9f04-656a6e31a6f9",
                "main",
                "Potential Metrorex salary and service disruption has high impact.",
            ),
            _context("needed", "The report should explain Metrorex's role and scale."),
            _research(
                ("entity_context",),
                "Reusable entity context could explain Metrorex without changing this judgment.",
            ),
        ),
    ),
    _theme_review(
        "12712b04-127f-4260-83bf-dd923ffc2213",
        5,
        (
            _relevance(
                "12712b04-127f-4260-83bf-dd923ffc2213",
                "relevant",
                "The hydropower law is useful and relevant.",
            ),
            _tier(
                "12712b04-127f-4260-83bf-dd923ffc2213",
                "main",
                "The review treats the subject as nationally relevant.",
            ),
        ),
    ),
    _theme_review(
        "b2f92241-5a4a-4425-be07-5f646b42ddbc",
        6,
        (
            _relevance(
                "b2f92241-5a4a-4425-be07-5f646b42ddbc",
                "relevant",
                "Low PNRR completion is nationally relevant.",
            ),
            _executable(
                "ranking",
                (
                    "sep10-pnrr-ranks-above-research-funding",
                    "sep10-pnrr-ranks-above-aur-afd",
                    "sep10-pnrr-ranks-above-cyber-readiness",
                    "sep10-pnrr-ranks-above-metrorex",
                    "sep10-pnrr-ranks-above-hydropower-law",
                ),
                "PNRR completion should rank above every earlier report subject.",
            ),
            _tier(
                "b2f92241-5a4a-4425-be07-5f646b42ddbc",
                "main",
                "Low PNRR completion has high national economic impact.",
            ),
            _research(
                ("investigative_report",),
                "The consequences of incomplete projects deserve later research.",
            ),
        ),
    ),
    _theme_review(
        "736f39d8-4b30-4da9-820e-bda2c29f5917",
        7,
        (
            _relevance(
                "736f39d8-4b30-4da9-820e-bda2c29f5917",
                "relevant",
                "The BET movement is nationally relevant economic news.",
            ),
            _tier(
                "736f39d8-4b30-4da9-820e-bda2c29f5917",
                "main",
                "The review identifies national economic impact.",
            ),
            _research(
                ("financial_data", "historical_data"),
                "Market data and history could corroborate the article.",
            ),
        ),
    ),
    _theme_review(
        "a44faaff-9135-43fb-b749-c7e71914b804",
        8,
        (
            _relevance(
                "a44faaff-9135-43fb-b749-c7e71914b804",
                "relevant",
                "The derivatives launch is relevant to Romania's financial market.",
            ),
            _tier(
                "a44faaff-9135-43fb-b749-c7e71914b804",
                "main",
                "A new derivatives market has national financial impact.",
            ),
            _context("needed", "The report should identify the instruments and launch timing."),
            _research(
                ("investigative_report",), "The missing market details deserve later research."
            ),
        ),
    ),
    _theme_review(
        "2b82f2e5-1bf6-48bf-bf64-caaf953fd6fb",
        9,
        (
            _relevance(
                "2b82f2e5-1bf6-48bf-bf64-caaf953fd6fb",
                "relevant",
                "PISA results affect Romania's workforce and growth trajectory.",
            ),
            _executable(
                "ranking",
                ("sep10-pisa-ranks-above-aur-afd", "sep10-pisa-ranks-above-cyber-readiness"),
                "PISA results should rank above subjects 2 and 3.",
            ),
            _tier(
                "2b82f2e5-1bf6-48bf-bf64-caaf953fd6fb",
                "main",
                "The review identifies major national socioeconomic impact.",
            ),
        ),
    ),
    _superseded_theme_review(
        "380ab371-7eb0-4c71-ad1d-4651a117e086",
        "e2f3019c-423f-4ea8-a66b-ce22d6678164",
        10,
    ),
    _theme_review(
        "e2f3019c-423f-4ea8-a66b-ce22d6678164",
        10,
        (
            _relevance(
                "e2f3019c-423f-4ea8-a66b-ce22d6678164",
                "relevant",
                "Commute time is a relevant infrastructure signal.",
            ),
            _tier(
                "e2f3019c-423f-4ea8-a66b-ce22d6678164",
                "worth_knowing",
                "The current fact is useful but not immediate high-impact news.",
            ),
            _research(
                ("historical_data", "investigative_report"),
                "A long-term commute trend could become a separate useful investigation.",
            ),
        ),
    ),
    _theme_review(
        "0c071aa1-9cb0-4ac4-bf78-fa36f3114349",
        11,
        (
            _relevance(
                "0c071aa1-9cb0-4ac4-bf78-fa36f3114349",
                "relevant",
                "Electric-car registrations are relevant economic data.",
            ),
            _tier(
                "0c071aa1-9cb0-4ac4-bf78-fa36f3114349",
                "worth_knowing",
                "One month's registrations are a useful side signal.",
            ),
            _research(
                ("historical_data", "financial_data"),
                "Longer registration and spending series could explain the change.",
            ),
        ),
    ),
    _theme_review(
        "44a396e9-8613-4001-9b5f-384b613cafb5",
        12,
        (
            _relevance(
                "44a396e9-8613-4001-9b5f-384b613cafb5",
                "relevant",
                "Dacia supplier risk is relevant regional economic news.",
            ),
            _tier(
                "44a396e9-8613-4001-9b5f-384b613cafb5",
                "worth_knowing",
                "The subject is useful regional context rather than a main subject.",
            ),
        ),
    ),
    _theme_review(
        "ebdc6e3f-7fa7-4e63-a296-1159bcb45f90",
        13,
        (
            _relevance(
                "ebdc6e3f-7fa7-4e63-a296-1159bcb45f90",
                "relevant",
                "STB insolvency is relevant to Bucharest public transport.",
            ),
            _executable(
                "grouping",
                ("sep10-metrorex-and-stb-share-public-transport-theme",),
                "STB and Metrorex belong in one broader public-transport subject.",
            ),
            _tier(
                "ebdc6e3f-7fa7-4e63-a296-1159bcb45f90",
                "worth_knowing",
                "The standalone STB event is a regional signal.",
            ),
        ),
    ),
    _theme_review(
        "b19a8101-5c35-4e31-b88e-3e5a7c9261c4",
        14,
        (
            _relevance(
                "b19a8101-5c35-4e31-b88e-3e5a7c9261c4",
                "relevant",
                "Housing costs across cities are nationally relevant.",
            ),
            _tier(
                "b19a8101-5c35-4e31-b88e-3e5a7c9261c4",
                "main",
                "The review identifies a Romanian housing crisis.",
            ),
            _research(
                ("financial_data", "historical_data"),
                "City comparisons and change over time could strengthen the subject.",
            ),
        ),
    ),
    _theme_review(
        "650de0b3-3307-4f60-a24d-3c452f46ac87",
        15,
        (
            _relevance(
                "650de0b3-3307-4f60-a24d-3c452f46ac87",
                "relevant",
                "The loss of 64,000 jobs is nationally relevant.",
            ),
            _executable(
                "ranking",
                ("sep10-job-losses-rank-above-aur-afd",),
                "Job losses should rank near the top and above the alliance subject.",
            ),
            _tier(
                "650de0b3-3307-4f60-a24d-3c452f46ac87",
                "main",
                "The review identifies this as the day's highest-impact subject.",
            ),
            _research(
                ("financial_data", "historical_data", "investigative_report"),
                "Industry, workforce-share, and historical comparisons deserve later research.",
            ),
        ),
    ),
    _theme_review(
        "f844f25c-02db-4726-9913-13e0045a608c",
        16,
        (
            _excluded(
                "relevance",
                "conditional",
                "The opinion framing is irrelevant, but the underlying deployment could be relevant if verified.",
            ),
            _tier(
                "f844f25c-02db-4726-9913-13e0045a608c",
                "excluded",
                "The opinion-framed subject should not appear in the report.",
            ),
            _confidence(
                "f844f25c-02db-4726-9913-13e0045a608c",
                "insufficient",
                "The report does not establish whether the claimed deployment occurred.",
            ),
            _research(
                ("external_corroboration", "investigative_report"),
                "The underlying event needs corroboration and factual reframing.",
            ),
        ),
    ),
    _theme_review(
        "099eabde-d07c-4740-ba53-42e5ee0afdc5",
        18,
        (
            _relevance(
                "099eabde-d07c-4740-ba53-42e5ee0afdc5",
                "relevant",
                "The market changes are relevant but not high impact.",
            ),
            _tier(
                "099eabde-d07c-4740-ba53-42e5ee0afdc5",
                "worth_knowing",
                "The subject is useful financial context.",
            ),
            _context("needed", "The companies' national significance is unclear."),
            _research(("entity_context",), "Entity context could explain the affected companies."),
        ),
    ),
    _theme_review(
        "9ee26633-222a-4445-b934-90532b6c0551",
        19,
        (
            _relevance(
                "9ee26633-222a-4445-b934-90532b6c0551",
                "relevant",
                "Tourism spending shows a national economic trend.",
            ),
            _tier(
                "9ee26633-222a-4445-b934-90532b6c0551",
                "worth_knowing",
                "The subject is useful but not among the main high-impact events.",
            ),
            _research(
                ("historical_data",),
                "A ten-year tourism series could explain the current observation.",
            ),
        ),
    ),
    _theme_review(
        "aba588b6-5411-45aa-8457-7e0bde6f7408",
        20,
        (
            _relevance(
                "aba588b6-5411-45aa-8457-7e0bde6f7408",
                "relevant",
                "Battery investment is relevant economic news.",
            ),
            _tier(
                "aba588b6-5411-45aa-8457-7e0bde6f7408",
                "worth_knowing",
                "The subject is interesting but not a main high-impact event.",
            ),
            _research(
                ("entity_context",), "Entity context could establish the company's importance."
            ),
        ),
    ),
    _superseded_theme_review(
        "b485f4d4-0405-4828-9302-0e55d43758aa",
        "33c1dccc-e4d6-4ec5-9775-7c5a53fd952b",
        21,
    ),
    _theme_review(
        "33c1dccc-e4d6-4ec5-9775-7c5a53fd952b",
        21,
        (
            _excluded(
                "relevance",
                "conditional",
                "The project matters only if company and regional scale support national impact.",
            ),
            _excluded(
                "tier",
                "conditional",
                "The subject should be excluded unless company and regional scale establish enough impact.",
            ),
            _confidence(
                "33c1dccc-e4d6-4ec5-9775-7c5a53fd952b",
                "insufficient",
                "The report lacks company and regional scale data.",
            ),
            _research(
                ("entity_context", "financial_data"),
                "Public company and regional data could resolve the impact judgment.",
            ),
        ),
    ),
    _theme_review(
        "fe4c9127-8c7b-4528-942a-07ed1a9d58be",
        22,
        (
            _excluded(
                "relevance",
                "conditional",
                "The appointment is important only if the influence claim is corroborated.",
            ),
            _tier(
                "fe4c9127-8c7b-4528-942a-07ed1a9d58be",
                "excluded",
                "The uncorroborated influence framing should not appear in the report.",
            ),
            _confidence(
                "fe4c9127-8c7b-4528-942a-07ed1a9d58be",
                "insufficient",
                "The report does not establish the appointee's alleged affiliations.",
            ),
            _research(
                ("entity_context", "external_corroboration", "investigative_report"),
                "The claim and related appointments need independent evidence.",
            ),
        ),
    ),
    _theme_review(
        "cf08cc52-5f32-47a2-8de4-e3d687383960",
        23,
        (
            _relevance(
                "cf08cc52-5f32-47a2-8de4-e3d687383960",
                "irrelevant",
                "The castle restoration does not affect national political, economic, or social conditions.",
            ),
            _tier(
                "cf08cc52-5f32-47a2-8de4-e3d687383960",
                "excluded",
                "The castle subject should not appear in this report.",
            ),
        ),
    ),
    _theme_review(
        "a395ade1-9f1d-4a3c-87dd-17fe3fcf3328",
        24,
        (
            _relevance(
                "a395ade1-9f1d-4a3c-87dd-17fe3fcf3328",
                "relevant",
                "A prime-minister proposal amid an interim government is nationally relevant.",
            ),
            _executable(
                "ranking",
                ("sep10-prime-minister-ranks-above-aur-afd",),
                "The proposal should rank far above the alliance subject.",
            ),
            _tier(
                "a395ade1-9f1d-4a3c-87dd-17fe3fcf3328",
                "main",
                "The proposal is a main political development.",
            ),
            _confidence(
                "a395ade1-9f1d-4a3c-87dd-17fe3fcf3328",
                "insufficient",
                "One article does not fully establish party support.",
            ),
            _research(
                ("coverage_review", "entity_context", "external_corroboration"),
                "Filtered coverage, the nominee, and party support need review.",
            ),
        ),
    ),
    _superseded_theme_review(
        "7adad8d4-623a-4892-8cf2-b3cdb79c3479",
        "31ee14fc-e827-4c82-a3ab-f23afb8739b6",
        25,
    ),
    _theme_review(
        "31ee14fc-e827-4c82-a3ab-f23afb8739b6",
        25,
        (
            _relevance(
                "31ee14fc-e827-4c82-a3ab-f23afb8739b6",
                "relevant",
                "Fidelis euro demand is a national economic signal.",
            ),
            _executable(
                "ranking",
                ("sep10-fidelis-ranks-above-aur-afd",),
                "Fidelis should rank above low-impact side subjects.",
            ),
            _tier(
                "31ee14fc-e827-4c82-a3ab-f23afb8739b6",
                "main",
                "The review treats euro preference as a strong economic signal.",
            ),
        ),
    ),
    _theme_review(
        "6c92cf32-87cb-43c0-aee3-0428699df6f2",
        26,
        (
            _relevance(
                "6c92cf32-87cb-43c0-aee3-0428699df6f2",
                "relevant",
                "Interest-rate data is relevant to national economics.",
            ),
            _executable(
                "ranking",
                ("sep10-rates-rank-above-aur-afd",),
                "Rates should rank above low-impact side subjects.",
            ),
            _tier(
                "6c92cf32-87cb-43c0-aee3-0428699df6f2",
                "main",
                "The subject is useful national financial news.",
            ),
            _research(
                ("financial_data", "historical_data"),
                "Market data and history could add useful context.",
            ),
        ),
    ),
    _theme_review(
        "7dab7cb3-b7ce-4e9a-b7d0-1cfddabeb122",
        27,
        (
            _relevance(
                "7dab7cb3-b7ce-4e9a-b7d0-1cfddabeb122",
                "relevant",
                "NIS2 sanctions are relevant side news.",
            ),
            _tier(
                "7dab7cb3-b7ce-4e9a-b7d0-1cfddabeb122",
                "worth_knowing",
                "The subject is useful but low impact and correctly ranked low.",
            ),
        ),
    ),
    _theme_review(
        "1e5226d4-f060-428f-a5b6-7713821cb565",
        28,
        (
            _relevance(
                "1e5226d4-f060-428f-a5b6-7713821cb565",
                "relevant",
                "BVB trading data is relevant national financial news.",
            ),
            _executable(
                "ranking",
                ("sep10-bvb-ranks-above-aur-afd",),
                "BVB movement should rank above low-impact side subjects.",
            ),
            _excluded(
                "grouping",
                "ambiguous",
                "The note relates this signal to other financial news but does not require one reader subject.",
            ),
            _tier(
                "1e5226d4-f060-428f-a5b6-7713821cb565",
                "main",
                "The subject belongs with the report's financial signals.",
            ),
            _research(
                ("financial_data", "historical_data"),
                "A chart and historical market data could add context.",
            ),
        ),
    ),
    _theme_review(
        "c2619b7c-fb8b-4dfa-8ebe-3012b5e8f5b9",
        29,
        (
            _relevance(
                "c2619b7c-fb8b-4dfa-8ebe-3012b5e8f5b9",
                "relevant",
                "The EUR/RON move is relevant national financial news.",
            ),
            _executable(
                "ranking",
                ("sep10-euro-ranks-above-aur-afd",),
                "The currency move should rank above low-impact side subjects.",
            ),
            _excluded(
                "grouping",
                "ambiguous",
                "The note relates this signal to other financial news but does not require one reader subject.",
            ),
            _tier(
                "c2619b7c-fb8b-4dfa-8ebe-3012b5e8f5b9",
                "main",
                "The subject belongs with the report's financial signals.",
            ),
            _research(
                ("financial_data", "historical_data"),
                "A chart and historical currency data could add context.",
            ),
        ),
    ),
)
SOURCE_FEEDBACK_IDS = tuple(review.feedback_id for review in FEEDBACK_REVIEWS)

UNREVIEWED_THEMES = (
    UnreviewedTheme(
        subject_position=17,
        target=ThemeFeedbackTarget(
            report_version_id=REPORT_VERSION_ID,
            theme_id=THEME_IDS[17],
        ),
    ),
)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args(argv)
    output: object = getattr(arguments, "output", None)
    if not isinstance(output, Path):
        raise TypeError("Evaluation output must be a path")
    _ = write_september_10_evaluation_manifest(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
