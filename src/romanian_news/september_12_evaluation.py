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
    DailyThemeJudgment,
    ExcludedConcernDisposition,
    ExecutableConcernDisposition,
    ExploratoryConcernDisposition,
    FeedbackConcernDisposition,
    FeedbackReview,
    NewsEvaluationDataset,
    NewsEvaluationManifest,
    OverallReportConcernDisposition,
    ReportEvaluationSnapshot,
    ReportEvaluationSpec,
    SupersededConcernDisposition,
    TierConcernDisposition,
)
from romanian_news.evaluation_curation import (
    TierSpec,
    build_theme_day_cases,
    build_tier_case,
    evaluation_case_spec,
    read_evaluation_artifact,
    read_evaluation_report_bundle,
)
from romanian_news.feedback import (
    NewsFeedbackEvent,
    ReportFeedbackTarget,
    ThemeFeedbackTarget,
)

DATASET_VERSION = "news-evaluation-2026-09-12-v12"
REVIEWED_AT = datetime.fromisoformat("2026-09-12T13:40:59.760989+00:00")
ISSUE_URL = "https://github.com/mauricedesaxe/chartly/issues/392"
DEFAULT_OUTPUT_PATH = Path("data/news/evaluations/news-evaluation-2026-09-12-v12-manifest.json")
REPORT_VERSION_ID: Sha256 = "5929257f0bf6554d80fcb3cd4f27a61a3785d2dd0c764aa3006ad2017946685d"
PRIOR_MANIFEST_REFERENCE = ArtifactReference(
    artifact_id="news:evaluation-manifest:news-evaluation-2026-09-11-v11",
    version_id="083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a",
    content_digest="19859f4edb7a7b7f28d02fe22c79e3afd2fdca71d50fec3dc8140a6ee35d1caf",
    r2_key=(
        "news/evaluations/manifests/news-evaluation-2026-09-11-v11/"
        "19859f4edb7a7b7f28d02fe22c79e3afd2fdca71d50fec3dc8140a6ee35d1caf.json"
    ),
)
REPORT_REFERENCE = ArtifactReference(
    artifact_id="news:daily:2026-09-12",
    version_id=REPORT_VERSION_ID,
    content_digest="132a00f67e4c6681808b4473f303e9519e339c8cb26e90b3cd84b5166770e3f4",
    r2_key=(
        "news/reports/daily/2026-09-12/"
        "132a00f67e4c6681808b4473f303e9519e339c8cb26e90b3cd84b5166770e3f4.json"
    ),
)
THEMES_REFERENCE = ArtifactReference(
    artifact_id="news:themes:2026-09-12",
    version_id="8addf453385edabedd5d8cfa390fe79457a7bd7499629eca502cee0ab9f4a3de",
    content_digest="834f4a91389aafeaa7df4c7c1053667d6e827d75c7d45e1ea2cfabb6fee0f973",
    r2_key=(
        "news/derived/themes/2026-09-12/"
        "834f4a91389aafeaa7df4c7c1053667d6e827d75c7d45e1ea2cfabb6fee0f973.json"
    ),
)
CLUSTER_REFERENCE = ArtifactReference(
    artifact_id="news:clusters:2026-09-12",
    version_id="0d6f568b37fa2092d252b61b54b80db8aa73211c0f8940695de29da8b33a644e",
    content_digest="4adfae206ebabe8fd714f512b2d628edee1aa099de238cc55373351dabd09309",
    r2_key=(
        "news/clusters/2026-09-12/"
        "4adfae206ebabe8fd714f512b2d628edee1aa099de238cc55373351dabd09309.json"
    ),
)

MAIN_TIER_FEEDBACK_IDS: dict[int, str] = {
    1: "d2019784-7d3f-4803-952c-1f40411a40f9",
    2: "134a7f64-c053-485a-8417-3b99591d11b0",
    4: "139398fd-cef6-43d4-a3ec-ce6119967e4c",
    5: "f9164813-7673-4129-b2aa-546edaa9762f",
    6: "99d6391c-19ff-49a9-9707-78c54598bf77",
    7: "5134001d-38e4-4847-acb9-f0ebbca95783",
    8: "5600bd39-6988-40f3-a294-f37a232bcbce",
    9: "c8bef42e-1869-4637-9e5a-bbf8840750bc",
}
MAIN_SUBJECT_FEEDBACK_IDS: dict[int, str] = {
    12: "c7ae76eb-088a-42c9-acca-a38b72f8f5e1",
    13: "42736350-e762-4fac-925c-20ea096f26dc",
    14: "3fabeb8b-fc8a-429f-bbbb-ae82275ff7f8",
    15: "71bfad89-75ce-4807-bd80-f03f8988e041",
}
WORTH_KNOWING_FEEDBACK_IDS: dict[int, str] = {
    17: "7577642e-1161-4449-b384-df5d7f21ba7f",
    18: "5dd6bd0f-fc6f-40ea-8f68-c994a7e0c06c",
    19: "04fb2df4-cc3a-4820-8670-e16c8124deb5",
}
INFLATION_MERGE_FEEDBACK_ID = "fa0d49b7-11e4-46d9-b989-ae2cf2a358d4"
REPORT_FINAL_FEEDBACK_ID = "a5dcaba6-9a0a-474a-9bcb-b1ec39394c82"

MERGE_WISH_RATIONALE = (
    "The owner asks for the subject to join the main subjects without naming a target subject."
)
MAIN_SUBJECT_RATIONALE = "The owner asks for the subject to join the main subjects."
PLACED_MAIN_RATIONALE = (
    "The owner praises the subject, summary, events, and rank, so the reviewed tier stays main."
)
PLACED_WORTH_KNOWING_RATIONALE = "The owner accepts or praises the side-show placement, so the reviewed tier stays worth_knowing."
STATS_RESEARCH_RATIONALE = (
    "The owner asks for backend research with historical and financial context, tracked in #373."
)
BVB_RESEARCH_RATIONALE = (
    "The owner asks for a European market comparison on the BVB subject, tracked in #373."
)

THEME_IDS: dict[int, Sha256] = {
    1: "5cde58f2b8bf5e6e97dff8b5664b0e010eab2eeb76dc229fe49faa12179e4c37",
    2: "562051050abc8431715771f1f01705c551490e07d8d01504f69a8fac4b3eac92",
    3: "97f6a068cc67c84b602a407110eba5d5fe00094552931da233eb5fe4ff4dddbf",
    4: "fb8acab8187ed0c8b822ada039cd9cba358c2692bd4c63d4ba3310bfd41bec25",
    5: "ee0d5c84cda6b3a214b4a1477e1e9eab305eef03973b7f99ee9081a740feeff0",
    6: "0a85c99a39361a653280e8c80afe7d7663d0a7792e1b37c30a4ae7eb33598502",
    7: "d7887388fc24dbbfcf994e1fe0a7dd4f17a956817335ac79903d237cc1a523bc",
    8: "c204ee4aa0b3a8ed1ec17ce4bd4ae7c1bdd8e6ebac8d235480690a131657aed2",
    9: "0bfa7f9f7af5a3c08f2aeaa993dd4d6341b25c2b009cb80800a0d3eeacac35ef",
    10: "9722e8fb87d1d691d9c1b073b5937ce05f86f0cb777f9b256cda72a09b4b8aef",
    11: "ec5a264074ac8109f5a69250c62ebbfe1fcf24b578107484be3322625eb0f0ad",
    12: "9e7a9a31cc56bcdcce0ca42d34287b29658163aa72741c9916546f75d740c8a3",
    13: "d4fec3cf87ae6c135cacd1dbdf3c700cb3e56b1ba8329f544b6a3900c236920c",
    14: "d4be79f21926012a1b12645f9dcf759dc42c0a4107d8a0dc4ab46a04e13afd73",
    15: "082b6455bee4419602714f9ab1128cb5adf603fd5f33c17bbc7d996fb17a9fd8",
    16: "c2a55c41d1df1435c1e60be85e145ce4479e3b2fee6c52e686ef7158682e42be",
    17: "1a7732b19a9d4de13eb26863c8ca8e7ccab7003f99a29a3f9144999ffe3e5367",
    18: "364e53ea2cb7dfe72883b4526fa4c61b66bbb3cd12334a910947d99e5072f662",
    19: "c740382a8db2880c0c8fa7aaa5796b4ef6e940d3d51f2e2df47fcbdd1f09187d",
    20: "61a3a3a59a220303596136cc13a679663c5341e53309887dedd4f45c5605b51c",
    21: "c911a0544fe64dc4df47b7d84c43f36b68d76d3438310060f8ffc05c683296ad",
    22: "2ef9b297e693723ff235a54f4e4478af736989dcc312b490b4b297e0294d7a95",
    23: "2ff0795c0643d570c2fee44edc89ff6aa4992cb22ffcd17371cef1ec497e0d17",
    24: "8ff1237c9a9a95ebb49371de1a7f5e68ec3292238879aa036981aa571cc35462",
    25: "d77f45f193cee41f0ec8af7ef372eee92d8557a330afbfccd480e4c2892c0b49",
    26: "0252fa889393fa6ff95f32f812e8481da5a6672f7931238ec43093b12e84f7bf",
    27: "67a437626b876762e34689694da1167f4b5edd01f07e94926e9731f86ed9df73",
    28: "a67751c31745165fc3b81f4677c8f68be2242e66de4491d0d51afac54813c84e",
    29: "f5018797e80b4ef9b8060c45efc78e884bd286639eabcf7d825224e365753c4d",
    30: "c495129eb301ae6d041a9748c82f6a82f2a65a1ab13368878ce2cf9550f19304",
    31: "597b9c1e7efe2cbf9b391bed26f854ffb197339f903836063d2a95c17e9d84fa",
    32: "8cca9d7de5c1bc3e50faf3455ed0be1f4e385dea5ceeb23e602b8b08bb85284a",
    33: "f8fd07e92fe9b5391f6ee23bd55fc69df7a227ac3b08c54222165acdb2539e24",
    34: "8df5a698ff97653bca374d063e984acd45260a4974ed2393654a8dbf59cd5dc8",
    35: "aed095e59a2f5ad30f24b78ba72ab902ed8da9a882ded5bb167b98d46ca505b8",
    36: "e26d000faea5e6b234c85daceea5dbe0e9682474b8cc07b17d13b5f0faa04fd7",
    37: "2bbbb9335e6fd394ff5f9ae4ebc222b2a9ba671a1a10059eede1914c9f22cd55",
    38: "1b797fc69cc8d872a0a6ee9ba358ae02fa5fa85e3f5b95b8b84cc26ebeb0a451",
    39: "18a9ee7204e84e476eb42e83bcb04123be0d02cbfc6dafb7d1cf95e7d1e55f11",
    40: "27b9e9799839cc98ae77fedde67c6c8e5803e1d70458ee35f44d658460378095",
    41: "f95cc8459bcbeff6b362935450fb541c8ed53e04423dc1d11b46fd2fd521d4f1",
    42: "7ca0e7ba01e1ea986a6e9003b0d54e9cf45dcb4141bb4892bdbb304052a7cac3",
}

GROUP_IDS: dict[int, Sha256] = {
    1: "6e3b8c7b7b4739371654909b27079af564ce709a101125b93cb30cb725ae6921",
    2: "c14ece816419e009fc179d1ffab383234527cc758046198b10078fe2910763c7",
    3: "fa587e8ce2086371900ab014835ccda20d57bfc896df46e7f9646a387cc27192",
    4: "22aa91bc181b2fdda16f0b19700a6fea36aada69211c13c70a86e1e32e4ff33d",
    5: "0d01fa0a3fea75fa164ed83a883d66832477689f3e6eeb2cc165f22dce568a7a",
    6: "b95929a7f39a748804480ba62b3ec31fa8fbfed70385f767ba55fc5f6e8709ca",
    7: "ddb22cb1cb18e3ae7085633c0452403a0026ceb1568bd6de15237e489cd6d146",
    8: "25e568885813e290b68c3e26f3ef02782a9008273d1bd54de1ef41a4fc073e3b",
    9: "d7125d2b1e5aee243767d0f1e2edc208189923bfa829e5025625767d3b92ee5d",
    10: "19014124209d157d303d96edf50bbeb07b4db6eb4f90e1edbc8e0b7d8e0d045e",
    11: "aac8138dbfdf617e859ca545540de009564965c8af6463d4e52d27326833c87a",
    12: "8243379c2c27d870e41ccd1e24afc472a39933b1ea05f23d44f9e1af4c31c817",
    13: "c8a99d659d6ed7a44e29196d19995584991db25d2b3897a2edc2d2a043543d31",
    14: "98d78aa81555615c0a79772a273d23628e65eae05965ba374f59b9c9c4e09795",
    15: "a1c79579082c1b0a0c84d61580b8bb88c9397338740da53aa0937ddb9e53bbd8",
    16: "282a5c7e090fa17b9ea2a1d901cee2767453adaa218ef3079c2f310896c46594",
    17: "72978de3dd83e3534ccac46b51b25e9d245c254b6579af193be5c349bf797077",
    18: "ede7ad829c09f2e16bd56a8a14c221e1b5349db643541e3255c171c0c405640a",
    19: "daa236d1d2f190a3bd0471211116aba2436ea355c22d90304fec6edc68816945",
    20: "aad934363aa1f760370ae265abcbfeabf5feb9142042b58b547375393e9bb804",
    21: "ffd472d1cac10ec3a4c25363d86b638359f2290cf5cd97b0ae20ba7ab2657073",
    22: "5de8a34fc45e061c3270c692e73d7e2cbe4a5cbe84ccd1fae5ad72d854b491be",
    23: "5c58809d3ad3f725ae4478790edd687cc2ec287f37913daf64462018f8a57e68",
    24: "156e5394f795f414939b9ecd6ffaf1513786ff5eb4643f81c081cbbde817a427",
    25: "132cd92b5e656de5493bad4ceaa0b796f92387c0788553bcc6ee1a8dfc39109c",
    26: "ca3c66c456a30678058dbecf82e9cd0bcc67b6aa18d673df7904fd0deb6121e6",
    27: "e663394ce76bd8ce5794f6178d5dc9fc9b7a5ab4b542aa64860ce2b188a4ad6a",
    28: "aa8221b697b77fcc582c22be498905d38c4350c33fa81458b69c110c73ff781b",
    29: "de927c708f92e42d7585e8c0a7705990bf1e500d66796c65e569ecd7c6bddb89",
    30: "3196b9885786e5f4c0af59c11cb6eb11610d4283085527b5f973a276459d91ac",
    31: "47baa3fa04e83955a340187e1aaca6abbd5d64f163bddd58685c451ba32b7a3e",
    32: "64b376c144c4ca3e97ef950dae36118a83f98df85c89d4fc7ad38ca19b1b7f11",
    33: "6c3e0c42eb3aa7ed71eaa97219cd8a08cf95443a5dd845542089bf7e64b2c218",
    34: "66805cefcc36ee71a7667333ce2be670dab5b3fcb87ffaadde8e068673d6317c",
    35: "10c8997b26664a9912b82425a232004058b135185666db035c209078c0c51f03",
    36: "9043253d4877c76da6eafa50ce43e63b7b24ee765f29678989624fd0f9f5ada5",
    37: "ced0e0f32103c2aff6dec6ea7b0ddae150bd6dc03d7cde8d162c7be532f07722",
    38: "fd388bbe63588dfa277c6ceb86ad77f1b979bee4936d1264b2d18a56c10e63e7",
    39: "aac56db10a71a0eefd4ccd83e40432d861ac37bbd6ea1812dd3cf428e23dd443",
    40: "a160164307681777d4b95957ab00a6282251fd1d41df75ba160cf9327dc45559",
    41: "23119f1747a73dd22e94d9445df660e074096f5d2839d82d05bc3f39ecbf303a",
    42: "a9d7a3046a8c2ea35bd326014a059ad29050525d86180fee947544c0c770e6ec",
}

GROUP_ORDER: tuple[Sha256, ...] = (
    "6e3b8c7b7b4739371654909b27079af564ce709a101125b93cb30cb725ae6921",
    "7b93bacf994a6e5881b8e04422c6dbba7979a8b3a74d5820889fa64411ff7483",
    "c9e15b1cdedc80bdbe4c47f8d40d7dc373dd88a110753de95801d3eef00f698e",
    "2354539c2ba9e8af7c1b42910bbf16dd11e27e7d72b91ea307615f9756a58077",
    "9edf1c338d325b0465524b1c2e04b45ca68251ef561c91190497a10b5c021cc6",
    "b3d82d197df45476a795549c7f035e05ab856c8a0ac47809db63ce7e19003cb0",
    "e7a802200d4636fdcc35b4748e7e9d4514fe0c0ad6c0053e1db0fdd7a1a66499",
    "c14ece816419e009fc179d1ffab383234527cc758046198b10078fe2910763c7",
    "e045dbfa0858707d80625536f214e22c25e52b0c05d4e97fcd0e8a5deca3d0a2",
    "d49feaa3f62335c9a3b5e5f96d2c859f45b060948ec6364abeea95fffff4487e",
    "38d811e2ce71d5ae614aaf25445c943b89f0471df93f351ef62e15590cc6b0df",
    "fa587e8ce2086371900ab014835ccda20d57bfc896df46e7f9646a387cc27192",
    "5fd592687477b639fa8a3e35e6676804d391e0c05409c517cef6220ea1ccff46",
    "22aa91bc181b2fdda16f0b19700a6fea36aada69211c13c70a86e1e32e4ff33d",
    "632d9dd7feedbb9855057e584f3496d049d0c58ae8b318f676657f96414dff30",
    "0d01fa0a3fea75fa164ed83a883d66832477689f3e6eeb2cc165f22dce568a7a",
    "b95929a7f39a748804480ba62b3ec31fa8fbfed70385f767ba55fc5f6e8709ca",
    "ddb22cb1cb18e3ae7085633c0452403a0026ceb1568bd6de15237e489cd6d146",
    "25e568885813e290b68c3e26f3ef02782a9008273d1bd54de1ef41a4fc073e3b",
    "d7125d2b1e5aee243767d0f1e2edc208189923bfa829e5025625767d3b92ee5d",
    "df72b27d6f6669db2f46d2f8e8a6cf3b04a0f47d28560812fcee804ea368f3f3",
    "19014124209d157d303d96edf50bbeb07b4db6eb4f90e1edbc8e0b7d8e0d045e",
    "aac8138dbfdf617e859ca545540de009564965c8af6463d4e52d27326833c87a",
    "8243379c2c27d870e41ccd1e24afc472a39933b1ea05f23d44f9e1af4c31c817",
    "c8a99d659d6ed7a44e29196d19995584991db25d2b3897a2edc2d2a043543d31",
    "98d78aa81555615c0a79772a273d23628e65eae05965ba374f59b9c9c4e09795",
    "a1c79579082c1b0a0c84d61580b8bb88c9397338740da53aa0937ddb9e53bbd8",
    "282a5c7e090fa17b9ea2a1d901cee2767453adaa218ef3079c2f310896c46594",
    "a2b2d9e31997dbad1c65b349e1aac61623d6bf040bfe9de5c57e74506f8ede9b",
    "72978de3dd83e3534ccac46b51b25e9d245c254b6579af193be5c349bf797077",
    "ede7ad829c09f2e16bd56a8a14c221e1b5349db643541e3255c171c0c405640a",
    "2d026e11bc0fd398a18029839ddbccb62c5926ab4698faca7cdb43f888dd449e",
    "daa236d1d2f190a3bd0471211116aba2436ea355c22d90304fec6edc68816945",
    "aad934363aa1f760370ae265abcbfeabf5feb9142042b58b547375393e9bb804",
    "ffd472d1cac10ec3a4c25363d86b638359f2290cf5cd97b0ae20ba7ab2657073",
    "5de8a34fc45e061c3270c692e73d7e2cbe4a5cbe84ccd1fae5ad72d854b491be",
    "5c58809d3ad3f725ae4478790edd687cc2ec287f37913daf64462018f8a57e68",
    "156e5394f795f414939b9ecd6ffaf1513786ff5eb4643f81c081cbbde817a427",
    "132cd92b5e656de5493bad4ceaa0b796f92387c0788553bcc6ee1a8dfc39109c",
    "ca3c66c456a30678058dbecf82e9cd0bcc67b6aa18d673df7904fd0deb6121e6",
    "e663394ce76bd8ce5794f6178d5dc9fc9b7a5ab4b542aa64860ce2b188a4ad6a",
    "aa8221b697b77fcc582c22be498905d38c4350c33fa81458b69c110c73ff781b",
    "840e67f1aa1b4ce32af5e8fcefb1d9e66ddb06917fac1e1b9cb930127c648b95",
    "de927c708f92e42d7585e8c0a7705990bf1e500d66796c65e569ecd7c6bddb89",
    "3196b9885786e5f4c0af59c11cb6eb11610d4283085527b5f973a276459d91ac",
    "47baa3fa04e83955a340187e1aaca6abbd5d64f163bddd58685c451ba32b7a3e",
    "64b376c144c4ca3e97ef950dae36118a83f98df85c89d4fc7ad38ca19b1b7f11",
    "6c3e0c42eb3aa7ed71eaa97219cd8a08cf95443a5dd845542089bf7e64b2c218",
    "66805cefcc36ee71a7667333ce2be670dab5b3fcb87ffaadde8e068673d6317c",
    "10c8997b26664a9912b82425a232004058b135185666db035c209078c0c51f03",
    "9043253d4877c76da6eafa50ce43e63b7b24ee765f29678989624fd0f9f5ada5",
    "ced0e0f32103c2aff6dec6ea7b0ddae150bd6dc03d7cde8d162c7be532f07722",
    "fe5474c515a6e621a4c64e8ccba994b2a34803a6f91591c56afb37cd98648fef",
    "fd388bbe63588dfa277c6ceb86ad77f1b979bee4936d1264b2d18a56c10e63e7",
    "aac56db10a71a0eefd4ccd83e40432d861ac37bbd6ea1812dd3cf428e23dd443",
    "a160164307681777d4b95957ab00a6282251fd1d41df75ba160cf9327dc45559",
    "23119f1747a73dd22e94d9445df660e074096f5d2839d82d05bc3f39ecbf303a",
    "a9d7a3046a8c2ea35bd326014a059ad29050525d86180fee947544c0c770e6ec",
)

TIER_SPECS = (
    *(
        TierSpec(
            case_id=f"sep12-subject-{position:02d}-tier",
            feedback_id=MAIN_TIER_FEEDBACK_IDS[position],
            report_version_id=REPORT_VERSION_ID,
            group_id=GROUP_IDS[position],
            expected_tier="main",
            control=True,
        )
        for position in (1, 2, 4, 5, 6, 7, 8, 9)
    ),
    *(
        TierSpec(
            case_id=f"sep12-subject-{position:02d}-tier",
            feedback_id=MAIN_SUBJECT_FEEDBACK_IDS[position],
            report_version_id=REPORT_VERSION_ID,
            group_id=GROUP_IDS[position],
            expected_tier="main",
            control=False,
        )
        for position in (12, 13, 14, 15)
    ),
    *(
        TierSpec(
            case_id=f"sep12-subject-{position:02d}-tier",
            feedback_id=WORTH_KNOWING_FEEDBACK_IDS[position],
            report_version_id=REPORT_VERSION_ID,
            group_id=GROUP_IDS[position],
            expected_tier="worth_knowing",
            control=True,
        )
        for position in (17, 18, 19)
    ),
)

INFLATION_MERGE_JUDGMENT_ID = "sep12-inflation-shares-economic-subject"
THEME_JUDGMENTS = (
    DailyThemeJudgment(
        judgment_id=INFLATION_MERGE_JUDGMENT_ID,
        feedback_ids=(UUID(INFLATION_MERGE_FEEDBACK_ID),),
        report_version_id=REPORT_VERSION_ID,
        left_group_id=GROUP_IDS[3],
        right_group_id=GROUP_IDS[2],
        expected_same_theme=True,
        rationale="The owner asked for the inflation subject to be part of subject 2.",
    ),
)


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


def _superseded_report_review(
    feedback_id: str,
    replacement_id: str,
) -> FeedbackReview:
    return FeedbackReview(
        feedback_id=UUID(feedback_id),
        target=ReportFeedbackTarget(report_version_id=REPORT_VERSION_ID),
        concerns=(
            SupersededConcernDisposition(
                superseded_by_feedback_id=UUID(replacement_id),
            ),
        ),
    )


def _tier_concern(
    position: int,
    judgment: Literal["main", "worth_knowing"],
    rationale: str,
) -> TierConcernDisposition:
    return TierConcernDisposition(
        judgment=judgment,
        case_ids=(f"sep12-subject-{position:02d}-tier",),
        rationale=rationale,
    )


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


FEEDBACK_REVIEWS = (
    _theme_review(
        MAIN_TIER_FEEDBACK_IDS[1],
        1,
        (_tier_concern(1, "main", PLACED_MAIN_RATIONALE),),
    ),
    _theme_review(
        MAIN_TIER_FEEDBACK_IDS[2],
        2,
        (_tier_concern(2, "main", PLACED_MAIN_RATIONALE),),
    ),
    _theme_review(
        INFLATION_MERGE_FEEDBACK_ID,
        3,
        (
            ExecutableConcernDisposition(
                concern="grouping",
                case_ids=(INFLATION_MERGE_JUDGMENT_ID,),
                rationale=(
                    "The owner asks for the inflation subject to be part of subject 2 "
                    "(economic challenges)."
                ),
            ),
        ),
    ),
    _theme_review(
        MAIN_TIER_FEEDBACK_IDS[4],
        4,
        (_tier_concern(4, "main", PLACED_MAIN_RATIONALE),),
    ),
    _theme_review(
        MAIN_TIER_FEEDBACK_IDS[5],
        5,
        (
            _tier_concern(5, "main", PLACED_MAIN_RATIONALE),
            _research(("financial_data", "external_corroboration"), BVB_RESEARCH_RATIONALE),
        ),
    ),
    _theme_review(
        MAIN_TIER_FEEDBACK_IDS[6],
        6,
        (_tier_concern(6, "main", PLACED_MAIN_RATIONALE),),
    ),
    _theme_review(
        MAIN_TIER_FEEDBACK_IDS[7],
        7,
        (_tier_concern(7, "main", PLACED_MAIN_RATIONALE),),
    ),
    _theme_review(
        MAIN_TIER_FEEDBACK_IDS[8],
        8,
        (
            _tier_concern(8, "main", PLACED_MAIN_RATIONALE),
            _excluded("grouping", "ambiguous", MERGE_WISH_RATIONALE),
        ),
    ),
    _theme_review(
        MAIN_TIER_FEEDBACK_IDS[9],
        9,
        (
            _tier_concern(9, "main", PLACED_MAIN_RATIONALE),
            _excluded("grouping", "ambiguous", MERGE_WISH_RATIONALE),
        ),
    ),
    _theme_review(
        MAIN_SUBJECT_FEEDBACK_IDS[12],
        12,
        (_tier_concern(12, "main", MAIN_SUBJECT_RATIONALE),),
    ),
    _superseded_theme_review(
        "cd53196c-8e09-4514-a8e2-c532e520aa23",
        "42736350-e762-4fac-925c-20ea096f26dc",
        13,
    ),
    _superseded_theme_review(
        "f6f2dc24-7c24-4224-b9d5-5a7cdb452823",
        "3fabeb8b-fc8a-429f-bbbb-ae82275ff7f8",
        14,
    ),
    _theme_review(
        MAIN_SUBJECT_FEEDBACK_IDS[14],
        14,
        (
            _tier_concern(14, "main", MAIN_SUBJECT_RATIONALE),
            _research(("financial_data", "historical_data"), STATS_RESEARCH_RATIONALE),
        ),
    ),
    _theme_review(
        MAIN_SUBJECT_FEEDBACK_IDS[13],
        13,
        (
            _tier_concern(13, "main", MAIN_SUBJECT_RATIONALE),
            _research(("financial_data", "historical_data"), STATS_RESEARCH_RATIONALE),
        ),
    ),
    _theme_review(
        MAIN_SUBJECT_FEEDBACK_IDS[15],
        15,
        (
            _tier_concern(15, "main", MAIN_SUBJECT_RATIONALE),
            _research(("financial_data", "historical_data"), STATS_RESEARCH_RATIONALE),
        ),
    ),
    _superseded_theme_review(
        "76360535-c261-4790-b7a2-d5f01c1c3b4f",
        "7577642e-1161-4449-b384-df5d7f21ba7f",
        17,
    ),
    _theme_review(
        WORTH_KNOWING_FEEDBACK_IDS[17],
        17,
        (_tier_concern(17, "worth_knowing", PLACED_WORTH_KNOWING_RATIONALE),),
    ),
    _superseded_theme_review(
        "a25348bf-71e0-4c71-a13c-d3747eeaedc8",
        "5dd6bd0f-fc6f-40ea-8f68-c994a7e0c06c",
        18,
    ),
    _superseded_theme_review(
        "dd235eef-a59c-456e-a3d3-3ee87a358062",
        "04fb2df4-cc3a-4820-8670-e16c8124deb5",
        19,
    ),
    _theme_review(
        WORTH_KNOWING_FEEDBACK_IDS[18],
        18,
        (_tier_concern(18, "worth_knowing", PLACED_WORTH_KNOWING_RATIONALE),),
    ),
    _theme_review(
        WORTH_KNOWING_FEEDBACK_IDS[19],
        19,
        (_tier_concern(19, "worth_knowing", PLACED_WORTH_KNOWING_RATIONALE),),
    ),
    _superseded_report_review(
        "a98866cc-09d8-473d-8019-1f4709a25985",
        "a5dcaba6-9a0a-474a-9bcb-b1ec39394c82",
    ),
    _superseded_report_review(
        "a70b81a1-16dc-46e1-bd50-9b143b09a403",
        "a5dcaba6-9a0a-474a-9bcb-b1ec39394c82",
    ),
    FeedbackReview(
        feedback_id=UUID(REPORT_FINAL_FEEDBACK_ID),
        target=ReportFeedbackTarget(report_version_id=REPORT_VERSION_ID),
        concerns=(
            OverallReportConcernDisposition(
                judgment="useful",
                rationale=(
                    "The owner calls it the best report so far and would change little "
                    "beyond exploring backend investigations."
                ),
            ),
            _excluded(
                "research",
                "unsupported",
                "The owner asks why the report is dominated by September 11 articles; "
                "the observation needs investigation before it becomes a judgment.",
            ),
        ),
    ),
)
SOURCE_FEEDBACK_IDS = tuple(review.feedback_id for review in FEEDBACK_REVIEWS)


def build_september_12_evaluation_manifest() -> NewsEvaluationManifest:
    """Freeze the reviewed September 12 grouping feedback onto the pinned v11 release."""
    prior_manifest = NewsEvaluationManifest.model_validate_json(
        read_evaluation_artifact(PRIOR_MANIFEST_REFERENCE), strict=True
    )
    if prior_manifest.version != "news-evaluation-2026-09-11-v11":
        raise ValueError(f"Pinned prior manifest has unexpected version: {prior_manifest.version}")
    prior = evaluation_catalog.hydrate_news_evaluation_manifest(prior_manifest)
    bundle = read_evaluation_report_bundle(REPORT_VERSION_ID)
    _require_exact_report(bundle.snapshot)
    feedback = evaluation_catalog.read_news_evaluation_feedback(SOURCE_FEEDBACK_IDS)
    require_exact_september_12_feedback(feedback)

    theme_cases = build_theme_day_cases(
        (*prior.reports, bundle.snapshot),
        THEME_JUDGMENTS,
        frozenset(SOURCE_FEEDBACK_IDS),
        {REPORT_VERSION_ID: THEMES_REFERENCE},
    )
    dataset = NewsEvaluationDataset(
        version=DATASET_VERSION,
        reviewed_at=REVIEWED_AT,
        issue_url=ISSUE_URL,
        source_feedback_ids=(*prior.source_feedback_ids, *SOURCE_FEEDBACK_IDS),
        reports=(*prior.reports, bundle.snapshot),
        cases=(
            *prior.cases,
            *(build_tier_case(spec, bundle) for spec in TIER_SPECS),
            *theme_cases,
        ),
        feedback_reviews=(*prior.feedback_reviews, *FEEDBACK_REVIEWS),
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
    )
    return NewsEvaluationManifest.model_validate_json(manifest.model_dump_json(), strict=True)


def write_september_12_evaluation_manifest(
    path: Path = DEFAULT_OUTPUT_PATH,
) -> NewsEvaluationManifest:
    """Write deterministic reference-only JSON without publishing it."""
    manifest = build_september_12_evaluation_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _require_exact_report(snapshot: ReportEvaluationSnapshot) -> None:
    if (
        snapshot.report != REPORT_REFERENCE
        or snapshot.themes != THEMES_REFERENCE
        or snapshot.cluster_set != CLUSTER_REFERENCE
    ):
        raise ValueError("September 12 report lineage differs from the reviewed release")
    if tuple(snapshot.report_group_ids) != GROUP_ORDER:
        raise ValueError("September 12 subject order or group identity changed")


def require_exact_september_12_feedback(feedback: tuple[NewsFeedbackEvent, ...]) -> None:
    """Verify the exact reviewed events, targets, and expansion order."""
    rows = {row.feedback_id: row for row in feedback}
    if len(feedback) != 24 or set(rows) != set(SOURCE_FEEDBACK_IDS):
        raise ValueError("PostgreSQL did not return the exact reviewed 24-event source")
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


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args(argv)
    output: object = getattr(arguments, "output", None)
    if not isinstance(output, Path):
        raise TypeError("Evaluation output must be a path")
    _ = write_september_12_evaluation_manifest(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
