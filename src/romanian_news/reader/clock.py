from datetime import date, datetime

from romanian_news import BUCHAREST


def bucharest_today() -> date:
    return datetime.now(BUCHAREST).date()
