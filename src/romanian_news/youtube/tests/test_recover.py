from __future__ import annotations

import json
import sys

import pytest

from romanian_news.youtube import recover


def test_release_prints_replay_identity_before_database_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover",
            "release",
            "recorder-youtube",
            "abcdefghijk",
            "--expected-generation",
            "1",
            "--requested-by",
            "operator",
            "--reason",
            "Source payload fixed",
        ],
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        identity = json.loads(capsys.readouterr().err)
        assert identity["request_id"]
        assert identity["requested_at"]
        raise RuntimeError("Database reply lost")

    monkeypatch.setattr(recover, "release_quarantined_youtube_video", fail)
    with pytest.raises(RuntimeError, match="Database reply lost"):
        recover.main()
