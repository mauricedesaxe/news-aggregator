from __future__ import annotations

import math
from decimal import Decimal


def float_linear_percentile(values: tuple[int, ...], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def decimal_linear_percentile(values: tuple[int, ...], quantile: Decimal) -> Decimal:
    if not values:
        return Decimal(0)
    ordered = sorted(values)
    position = Decimal(len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return Decimal(ordered[lower])
    return Decimal(ordered[lower]) + Decimal(ordered[upper] - ordered[lower]) * (
        position - Decimal(lower)
    )
