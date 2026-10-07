import pytest

from romanian_news.config import _parse_optional_bool, _parse_postgres_dsn


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("  ", False),
        ("0", False),
        ("false", False),
        ("no", False),
        ("on", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        (" TRUE ", True),
        ("\tYeS\n", True),
    ],
)
def test_optional_bool_parser_accepts_only_trimmed_true_values(
    value: str | None,
    expected: bool,
) -> None:
    assert _parse_optional_bool(value) is expected


def test_postgres_dsn_parser_rejects_invalid_values_without_echoing_them() -> None:
    assert _parse_postgres_dsn(None) is None
    assert _parse_postgres_dsn("  ") is None
    assert _parse_postgres_dsn("postgresql:///chartly") == "postgresql:///chartly"

    secret = "not a dsn password=must-not-appear"
    with pytest.raises(ValueError, match="PostgreSQL DSN is invalid") as failure:
        _parse_postgres_dsn(secret)

    assert secret not in str(failure.value)
