from datetime import date

from scripts import advance_historical_page_checks as advance


def test_month_windows_keep_partial_first_and_last_months() -> None:
    assert advance.month_windows(date(2025, 9, 27), date(2025, 11, 2)) == (
        (date(2025, 9, 27), date(2025, 9, 30)),
        (date(2025, 10, 1), date(2025, 10, 31)),
        (date(2025, 11, 1), date(2025, 11, 2)),
    )


def test_advance_checks_oldest_month_with_pending_pages(monkeypatch) -> None:
    queried = []

    def pending(outlet, start, end, limit):
        queried.append((outlet, start, end, limit))
        return (object(),) if start.month == 10 else ()

    def check(outlet, start, end, *, limit):
        assert (outlet, start, end, limit) == (
            "hotnews",
            date(2025, 10, 1),
            date(2025, 10, 31),
            100,
        )
        return ()

    monkeypatch.setattr(advance, "pending_page_candidates", pending)
    monkeypatch.setattr(advance, "check_archive_pages", check)

    result = advance.advance_page_checks("hotnews", date(2025, 9, 27), date(2025, 10, 31))

    assert [item[1].month for item in queried] == [9, 10]
    assert result == {
        "outlet": "hotnews",
        "start": "2025-10-01",
        "end": "2025-10-31",
        "checked": 0,
        "statuses": {},
    }
