import pytest

from romanian_news.config import _parse_postgres_dsn


def test_postgres_dsn_parser_rejects_invalid_values_without_echoing_them() -> None:
    assert _parse_postgres_dsn(None) is None
    assert _parse_postgres_dsn("  ") is None
    assert _parse_postgres_dsn("postgresql:///chartly") == "postgresql:///chartly"

    secret = "not a dsn password=must-not-appear"
    with pytest.raises(ValueError, match="PostgreSQL DSN is invalid") as failure:
        _parse_postgres_dsn(secret)

    assert secret not in str(failure.value)
