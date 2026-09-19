from datetime import date

from romanian_news import daily
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.embeddings import embedding_request_id
from romanian_news.analysis.relevance_v3 import production_relevance_v3_request_id
from romanian_news.catalog import artifacts as catalog_artifacts
from romanian_news.catalog import daily as catalog_daily
from romanian_news.daily import DailyArtifactReferences

DAY = date(2099, 9, 2)


def test_daily_relevance_references_target_the_production_v3_identity(monkeypatch) -> None:
    article = _reference("news:article", "a")
    relevance_id = f"news:relevance:{production_relevance_v3_request_id(article)}"
    requested: list[str] = []

    def current_references(artifact_ids):
        requested.extend(artifact_ids)
        return tuple(_reference(value, "b") for value in sorted(artifact_ids))

    monkeypatch.setattr(catalog_daily, "read_daily_article_references", lambda _day: (article,))
    monkeypatch.setattr(catalog_artifacts, "current_artifact_references", current_references)

    assert daily.read_daily_relevance_references(DAY) == DailyArtifactReferences(
        day=DAY, values=(_reference(relevance_id, "b"),)
    )
    assert requested == [relevance_id]


def test_daily_embedding_references_require_accepted_relevance(monkeypatch) -> None:
    first = _reference("news:article:first", "a")
    second = _reference("news:article:second", "c")
    first_relevance = _reference(f"news:relevance:{production_relevance_v3_request_id(first)}", "b")
    second_relevance = _reference(
        f"news:relevance:{production_relevance_v3_request_id(second)}", "d"
    )
    embedding = _reference(f"news:embedding:{embedding_request_id(first, first_relevance)}", "e")
    known = {
        first_relevance.artifact_id: first_relevance,
        second_relevance.artifact_id: second_relevance,
        embedding.artifact_id: embedding,
    }
    monkeypatch.setattr(
        catalog_daily, "read_daily_article_references", lambda _day: (first, second)
    )
    monkeypatch.setattr(
        catalog_artifacts,
        "current_artifact_references",
        lambda artifact_ids: tuple(
            known[value] for value in sorted(artifact_ids) if value in known
        ),
    )
    monkeypatch.setattr(
        catalog_daily,
        "relevance_is_accepted",
        lambda version_id: version_id == first_relevance.version_id,
    )

    references = daily.read_daily_embedding_references(DAY)

    assert references == DailyArtifactReferences(day=DAY, values=(embedding,))


def _reference(artifact_id: str, digest_character: str) -> ArtifactReference:
    digest = digest_character * 64
    return ArtifactReference(
        artifact_id=artifact_id,
        version_id=digest,
        content_digest=digest,
        r2_key=f"news/{digest_character}.json",
    )
