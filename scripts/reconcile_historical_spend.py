"""Reconcile one archive model reservation after checking provider billing."""

import argparse
import json
import sys
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from uuid import UUID

from romanian_news.catalog.archive_model_spend import reconcile_archive_spend


def _decimal(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("Cost must be a decimal amount in USD") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reservation_id", type=UUID)
    parser.add_argument("actual_usd", type=_decimal)
    parser.add_argument("--operator", required=True, help="Operator login for the audit record")
    parser.add_argument("--reason", required=True, help="Why the measured charge is correct")
    parser.add_argument(
        "--billing-reference", required=True, help="Provider billing record checked by the operator"
    )
    arguments = parser.parse_args()
    result = reconcile_archive_spend(
        arguments.reservation_id,
        arguments.actual_usd,
        operator=arguments.operator,
        reason=arguments.reason,
        billing_reference=arguments.billing_reference,
    )
    sys.stdout.write(json.dumps(asdict(result), default=str, indent=2) + "\n")


if __name__ == "__main__":
    main()
