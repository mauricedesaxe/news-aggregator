from datetime import date

from scripts import next_historical_article_capture as advance


def test_capture_plan_skips_months_without_verified_articles(monkeypatch) -> None:
    checked = []

    def pending(outlet, start, end, limit):
        checked.append((outlet, start, end, limit))
        return (object(),) if start.month == 10 else ()

    monkeypatch.setattr(advance, "pending_archive_articles", pending)

    result = advance.next_capture_config("digi24", date(2025, 9, 27), date(2025, 10, 31))

    assert [row[1].month for row in checked] == [9, 10]
    assert result == {
        "ops": {
            "archive_article_capture": {
                "config": {
                    "outlet": "digi24",
                    "start": "2025-10-01",
                    "end": "2025-10-31",
                    "limit": 50,
                }
            }
        }
    }
