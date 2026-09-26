from __future__ import annotations

from datetime import date

from romanian_news.catalog import weekly_status
from romanian_news.weekly_status import WeeklyStatusOutput, WeeklyStatusRead


def test_retry_reuses_saved_output_even_if_model_text_differs(monkeypatch) -> None:
    saved_version = "b" * 64
    batches = []
    monkeypatch.setattr(weekly_status, "current_artifact_file", lambda _artifact: None)
    monkeypatch.setattr(weekly_status, "run_status", lambda _run: "completed")
    monkeypatch.setattr(
        weekly_status,
        "catalog_query",
        lambda _statement, _parameters: [{"artifact_version_id": saved_version}],
    )
    monkeypatch.setattr(
        weekly_status, "catalog_batch", lambda statements: batches.append(statements)
    )
    output = WeeklyStatusOutput.model_construct(
        request_id="a" * 64,
        read=WeeklyStatusRead.model_construct(week_start=date(2026, 9, 14)),
        content_digest="c" * 64,
        content=b"new model wording",
    )

    result = weekly_status.publish_weekly_status(output, "implementation")

    assert result.version_id == saved_version
    assert result.status == "published"
    assert batches[0][0][1][0] == saved_version
