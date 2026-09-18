from __future__ import annotations

from datetime import date, datetime

from romanian_news import Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.catalog import evaluations as evaluation_catalog
from romanian_news.evaluation import (
    EvaluationProvenance,
    NewsEvaluationDataset,
    NewsEvaluationManifest,
    ReportEvaluationSnapshot,
    ReportEvaluationSpec,
    ThemeEvaluationDayCase,
    ThemePairExpectation,
    ThemeReportExpectation,
)
from romanian_news.evaluation_curation import (
    evaluation_case_spec,
    read_evaluation_artifact,
    read_evaluation_report_bundle,
)
from romanian_news.themes import read_recorded_daily_theme_input

DATASET_VERSION = "news-evaluation-2026-09-10-v8"
REVIEWED_AT = datetime.fromisoformat("2026-09-10T12:00:00+00:00")
ISSUE_URL = "https://github.com/mauricedesaxe/chartly/issues/371"
PRIOR_MANIFEST_REFERENCE = ArtifactReference(
    artifact_id="news:evaluation-manifest:news-evaluation-2026-09-10-v7",
    version_id="b471873b7a17a15d6076536d0222fd85829c8470ea7e8ee7b68ff641be0488cd",
    content_digest="da798efac820d1d63cc1cb56f620b51ef04a8cf10025e5a72cd8ba191c66ce31",
    r2_key=(
        "news/evaluations/manifests/news-evaluation-2026-09-10-v7/"
        "da798efac820d1d63cc1cb56f620b51ef04a8cf10025e5a72cd8ba191c66ce31.json"
    ),
)
SEPTEMBER_8_REPORT_VERSION_ID: Sha256 = (
    "75e907b5ae99760b9f72745b9260a0d536c760ff1c8ea3ad19dd38018fbdb428"
)
SEPTEMBER_10_REPORT_VERSION_ID: Sha256 = (
    "c1369f9a24202a111dfc07e29a7194c73cb22dd491958093932a5d3b9b270d72"
)

SEPTEMBER_8_GROUPS = {
    "tax_outlook": "b8147faaa8ee5e9cd009003a5b208c7c39ab48dacdeafaee7cc9dc4e078df22c",
    "vat_fraud": "a7e92442f40a47e9b654b7390e3442ceaa68449a5215ca879014cbf15c2c18ee",
    "contraction": "47a6a5ec5bdd37ad114bd511f5f80eac440055af40fb3e960917075eb67cabe8",
    "public_debt": "578863f12c690da505cbe5bfc115ff2ecf4015b81796df9ecb9bf0fa1f4864b1",
    "fuel_prices": "f10d1d735b314c9a7c948d8e2b846a56b8ecda71dde0d83669b8f911a307eaa6",
    "electricity_prices": "a9601fa366c29cf1b248d8b8d5e44c9e17da778426b5420996938316537440a7",
    "aur": "f46c50130b2d02ba7e13acc53a55df4316c7574aff2574a87acc6d98041ded17",
    "senate_hearing": "f6893f8e09f9f96971c0ab2b4728ac1d3e4c945b8c8239b8c95a3b0632a8cca6",
    "prime_minister": "d93b28747521a994e5618b53dd9a8c5fc3946fee8ee75e8166d856887985ffb5",
    "premier_role": "cca58bf51edea1b71185ce269f01078c59cbbb69a5a1cf8303335a12cd687310",
    "labor_cost": "241181c7926704cd527616ee3a5a0de0198f218d7e610904f3aa98473bd5aef8",
    "eu_defense": "52eb6f05722741f47b1f0f0ad96fbf8a298f03c41e742ccdd0b6b6a934aefe6b",
    "us_partnership": "7661bedb0be725b44909fcc1eede882a44e033422c517eea94cd527f17f331bc",
    "roads": "faa5af6cf7fec4705411d5a27a407035c91519148d0725800a22054200c71fec",
}
SEPTEMBER_10_GROUPS = {
    "aur_afd": "66ef905c0071681d47d643395c239719522695746714ed3b0c7d0b66da94d604",
    "metrorex": "6ff7f531f9adf2485ce61c77c71af37b32b2a3148900da7c8eff136099a09f9e",
    "bet_fatigue": "bc6084265043065c2712bcd4c771136920ebeff1cfe20dab521857d543b0b975",
    "commute": "0a36d0918760f3f8e625393abb875936dfdbc52db6d732b531bee8c51d33070c",
    "stb": "2c876749d928c2fd12a4c70128187694844999545bbf505b8303177c43f3045a",
    "bet_changes": "92dec2a38bdb39cb00d897095bca151a8d12278f451994034c00e36d77a970dc",
    "tvr": "fa78ad8beb9d7c37021f3fbbf543d7b47467468583bec1a470a46fa9dc573188",
    "prime_minister": "37aebe0473f1e5d9239f098b309d6af4d35949c58084d72342d1ef5895a71398",
    "deposit_rates": "567e6b2761fb08a3db8a62efc3dd7d11b62dc26eee5c9a255c3947e3456f49a6",
    "bvb_decline": "b8fb51d161d612803b3b4787099a235c528f7c02e846af12eb59bc443f9ecd7d",
    "euro": "be905d44a73ea5c65604a421c1a83536800b0c7e370d0bf02276ec416bec7d32",
}


def build_reader_subject_evaluation_manifest() -> NewsEvaluationManifest:
    """Freeze the paired September 8 and 10 reader-subject workload."""
    prior_manifest = NewsEvaluationManifest.model_validate_json(
        read_evaluation_artifact(PRIOR_MANIFEST_REFERENCE), strict=True
    )
    prior = evaluation_catalog.hydrate_news_evaluation_manifest(prior_manifest)
    september_8 = read_evaluation_report_bundle(SEPTEMBER_8_REPORT_VERSION_ID).snapshot
    september_10_case = next(
        case
        for case in prior.cases
        if case.concern == "daily_theme"
        and case.source_report.version_id == SEPTEMBER_10_REPORT_VERSION_ID
    )
    fresh_cases = (
        _september_8_case(september_8),
        _fresh_september_10_case(september_10_case),
    )
    inherited_cases = tuple(
        case
        for case in prior.cases
        if not (
            case.concern == "daily_theme"
            and case.source_report.version_id == SEPTEMBER_10_REPORT_VERSION_ID
        )
    )
    dataset = NewsEvaluationDataset(
        version=DATASET_VERSION,
        reviewed_at=REVIEWED_AT,
        issue_url=ISSUE_URL,
        source_feedback_ids=prior.source_feedback_ids,
        reports=(*prior.reports, september_8),
        cases=(*inherited_cases, *fresh_cases),
        archived_daily_theme_judgments=prior.archived_daily_theme_judgments,
        excluded_feedback=prior.excluded_feedback,
        feedback_reviews=prior.feedback_reviews,
        unreviewed_themes=prior.unreviewed_themes,
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
        archived_daily_theme_judgments=dataset.archived_daily_theme_judgments,
        excluded_feedback=dataset.excluded_feedback,
        feedback_reviews=dataset.feedback_reviews,
        unreviewed_themes=dataset.unreviewed_themes,
    )
    return NewsEvaluationManifest.model_validate_json(manifest.model_dump_json(), strict=True)


def _september_8_case(snapshot: ReportEvaluationSnapshot) -> ThemeEvaluationDayCase:
    value = read_recorded_daily_theme_input(date(2026, 9, 8))
    if value.cluster_set != snapshot.cluster_set:
        raise ValueError("September 8 theme inputs do not match the frozen report")
    return ThemeEvaluationDayCase(
        case_id="sep08-reader-subject-relationships",
        control=True,
        provenance=EvaluationProvenance(feedback_ids=(), report=snapshot.report),
        source_report=snapshot.report,
        input=value,
        expectations=(
            _pair(
                "sep08-energy-costs-link",
                SEPTEMBER_8_GROUPS,
                "fuel_prices",
                "electricity_prices",
                True,
            ),
            _pair("sep08-tax-policy-link", SEPTEMBER_8_GROUPS, "tax_outlook", "vat_fraud", True),
            _pair(
                "sep08-economic-pressure-link",
                SEPTEMBER_8_GROUPS,
                "contraction",
                "public_debt",
                True,
            ),
            _pair(
                "sep08-defense-posture-link",
                SEPTEMBER_8_GROUPS,
                "eu_defense",
                "us_partnership",
                True,
            ),
            _pair(
                "sep08-premier-role-link",
                SEPTEMBER_8_GROUPS,
                "prime_minister",
                "premier_role",
                True,
            ),
            _pair(
                "sep08-politics-economy-separate", SEPTEMBER_8_GROUPS, "aur", "contraction", False
            ),
            _pair(
                "sep08-governance-energy-separate",
                SEPTEMBER_8_GROUPS,
                "senate_hearing",
                "electricity_prices",
                False,
            ),
            _pair(
                "sep08-labor-debt-separate", SEPTEMBER_8_GROUPS, "labor_cost", "public_debt", False
            ),
            _pair("sep08-roads-politics-separate", SEPTEMBER_8_GROUPS, "roads", "aur", False),
        ),
        report_expectation=ThemeReportExpectation(),
    )


def _fresh_september_10_case(case: ThemeEvaluationDayCase) -> ThemeEvaluationDayCase:
    return ThemeEvaluationDayCase(
        case_id="sep10-reader-subject-relationships",
        control=False,
        provenance=case.provenance,
        source_report=case.source_report,
        input=case.input,
        expectations=(
            *case.expectations,
            _pair("sep10-bet-index-link", SEPTEMBER_10_GROUPS, "bet_fatigue", "bet_changes", True),
            _pair("sep10-bvb-market-link", SEPTEMBER_10_GROUPS, "bet_changes", "bvb_decline", True),
            _pair(
                "sep10-politics-currency-separate", SEPTEMBER_10_GROUPS, "aur_afd", "euro", False
            ),
            _pair(
                "sep10-transport-scope-separate", SEPTEMBER_10_GROUPS, "metrorex", "commute", False
            ),
            _pair(
                "sep10-politics-rates-separate",
                SEPTEMBER_10_GROUPS,
                "prime_minister",
                "deposit_rates",
                False,
            ),
            _pair("sep10-media-market-separate", SEPTEMBER_10_GROUPS, "tvr", "bvb_decline", False),
        ),
        report_expectation=ThemeReportExpectation(),
    )


def _pair(
    case_id: str,
    groups: dict[str, Sha256],
    left: str,
    right: str,
    expected_same_theme: bool,
) -> ThemePairExpectation:
    relationship = "one specific reader subject" if expected_same_theme else "unrelated subjects"
    return ThemePairExpectation(
        case_id=case_id,
        feedback_ids=(),
        left_group_id=groups[left],
        right_group_id=groups[right],
        expected_same_theme=expected_same_theme,
        control=False,
        rationale=f"The reviewed groups represent {relationship}.",
    )
