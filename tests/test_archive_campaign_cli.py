from romanian_news.archive.campaign import ARCHIVE_END, ARCHIVE_START
from scripts import next_historical_article_capture as advance


def test_capture_planner_uses_active_archive_campaign(monkeypatch, capsys) -> None:
    calls = []

    def next_config(outlet, start, end):
        calls.append((outlet, start, end))
        return None

    monkeypatch.setattr(advance, "next_capture_config", next_config)
    monkeypatch.setattr("sys.argv", ["next_historical_article_capture", "--outlet", "hotnews"])

    advance.main()

    assert calls == [("hotnews", ARCHIVE_START, ARCHIVE_END)]
    assert capsys.readouterr().out == "complete\n"
