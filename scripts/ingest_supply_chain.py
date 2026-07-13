"""Turns the amirmotefaker supply-chain dataset into quality_rejection and
stockout exception contexts.

Real signals used: Inspection results == "Fail" (36 of 100 rows) for
quality_rejection, and Stock levels < Order quantities for stockout. Only 100
rows total in this dataset — small, but it
has a real Supplier name column, so unlike DataCo these rows can genuinely be
addressed to a supplier.

Run with: python scripts/ingest_supply_chain.py
Output: data/exceptions_pool_quality_rejection.json, data/exceptions_pool_stockout.json
"""
import json
from pathlib import Path

import pandas as pd

RAW_FILE = Path(__file__).parent.parent / "data" / "raw" / "supply_chain" / "supply_chain_data.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data"


def _supplier_id(name: str) -> str:
    # "Supplier 3" -> "supplier-3", matching the kebab-case id style used
    # elsewhere in this project's mock/synthetic data.
    return name.lower().replace(" ", "-")


def main() -> None:
    df = pd.read_csv(RAW_FILE)

    failed_inspection = df[df["Inspection results"] == "Fail"].copy()
    quality_contexts = [
        {
            "exception_type": "quality_rejection",
            "supplier_id": _supplier_id(row["Supplier name"]),
            "product_ids": [row["SKU"]],
            "defect_rate": float(row["Defect rates"]),
            "source": "supply_chain_amirmotefaker",
        }
        for _, row in failed_inspection.iterrows()
    ]
    (OUTPUT_DIR / "exceptions_pool_quality_rejection.json").write_text(json.dumps(quality_contexts, indent=2))

    understocked = df[df["Stock levels"] < df["Order quantities"]].copy()
    stockout_contexts = [
        {
            "exception_type": "stockout",
            "supplier_id": _supplier_id(row["Supplier name"]),
            "product_ids": [row["SKU"]],
            "stock_levels": int(row["Stock levels"]),
            "order_quantities": int(row["Order quantities"]),
            "source": "supply_chain_amirmotefaker",
        }
        for _, row in understocked.iterrows()
    ]
    (OUTPUT_DIR / "exceptions_pool_stockout.json").write_text(json.dumps(stockout_contexts, indent=2))

    print(f"Supply chain: {len(failed_inspection)} failed inspections -> quality_rejection contexts")
    print(f"Supply chain: {len(understocked)} understocked rows -> stockout contexts")


if __name__ == "__main__":
    main()
