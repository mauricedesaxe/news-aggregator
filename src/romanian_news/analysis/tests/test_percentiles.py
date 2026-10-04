from decimal import Decimal

from romanian_news.analysis.percentiles import (
    decimal_linear_percentile,
    float_linear_percentile,
)


def test_float_linear_percentile_preserves_interpolation_contract() -> None:
    assert float_linear_percentile((), 0.50) == 0.0
    assert type(float_linear_percentile((7,), 0.50)) is float
    assert float_linear_percentile((40, 10, 30, 20), 0.50) == 25.0
    assert float_linear_percentile((40, 10, 30, 20), 0.95) == 38.5
    assert float_linear_percentile((40, 10, 30, 20), 0) == 10.0
    assert float_linear_percentile((40, 10, 30, 20), 1) == 40.0


def test_decimal_linear_percentile_preserves_interpolation_scale() -> None:
    assert decimal_linear_percentile((), Decimal("0.50")).as_tuple() == Decimal(0).as_tuple()
    assert decimal_linear_percentile((7,), Decimal("0.50")).as_tuple() == Decimal(7).as_tuple()
    assert (
        decimal_linear_percentile((40, 10, 30, 20), Decimal("0.50")).as_tuple()
        == Decimal("25.00").as_tuple()
    )
    assert (
        decimal_linear_percentile((40, 10, 30, 20), Decimal("0.95")).as_tuple()
        == Decimal("38.50").as_tuple()
    )
    assert (
        decimal_linear_percentile((40, 10, 30, 20), Decimal("0.5")).as_tuple()
        == Decimal("25.0").as_tuple()
    )
    assert decimal_linear_percentile((40, 10, 30, 20), Decimal(0)) == Decimal(10)
    assert decimal_linear_percentile((40, 10, 30, 20), Decimal(1)) == Decimal(40)
