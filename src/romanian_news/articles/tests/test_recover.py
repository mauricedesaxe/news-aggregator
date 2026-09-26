from __future__ import annotations

import json
from datetime import datetime

import pytest

from romanian_news.articles import recover
from romanian_news.articles.acquisition import ArticleRecoveryReceipt, ArticleRecoveryStatus

EVENT_ID = "a" * 64


@pytest.fixture(autouse=True)
def _offline_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recover, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(recover, "feed_registry", lambda: object())


def test_status_prints_the_recovery_state(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        recover,
        "inspect_article_recovery_event",
        lambda event_id, _registry: ArticleRecoveryStatus(
            event_id=event_id,
            work_generation="b" * 64,
            quarantined=True,
            deterministic_fingerprint="c" * 64,
        ),
    )

    assert recover.main(["status", EVENT_ID]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "event_id": EVENT_ID,
        "work_generation": "b" * 64,
        "quarantined": True,
        "deterministic_fingerprint": "c" * 64,
    }


def test_release_prints_the_receipt_with_a_utc_timestamp(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    captured: dict[str, object] = {}

    def release(event_ids, _registry, **kwargs):
        captured.update(event_ids=event_ids, **kwargs)
        return ArticleRecoveryReceipt(recovery_ids=("d" * 64,), released_event_ids=tuple(event_ids))

    monkeypatch.setattr(recover, "recover_quarantined_article_events", release)

    assert (
        recover.main(
            [
                "release",
                EVENT_ID,
                "--expected-generation",
                "b" * 64,
                "--requested-by",
                "operator",
                "--reason",
                "Parser fixed",
                "--requested-at",
                "2026-09-24T15:00:00+03:00",
            ]
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["recovery_ids"] == ["d" * 64]
    assert printed["released_event_ids"] == [EVENT_ID]
    assert printed["requested_at"] == "2026-09-24T12:00:00+00:00"
    assert captured["event_ids"] == (EVENT_ID,)
    assert captured["expected_work_generations"] == {EVENT_ID: "b" * 64}
    assert captured["requested_at"] == datetime.fromisoformat("2026-09-24T15:00:00+03:00")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["status", "not-hex"], "event_id must be a lowercase SHA-256 digest"),
        (["release", EVENT_ID, "--expected-generation", "zz"], "expected generation"),
        (
            [
                "release",
                EVENT_ID,
                "--expected-generation",
                "b" * 64,
                "--requested-by",
                "operator",
                "--reason",
                "still quarantined elsewhere",
            ],
            "generation changed",
        ),
    ],
)
def test_invalid_input_exits_with_an_error(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], message: str
) -> None:
    monkeypatch.setattr(
        recover,
        "recover_quarantined_article_events",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("generation changed")),
    )
    with pytest.raises(SystemExit) as raised:
        recover.main(argv)
    assert raised.value.code == 2
