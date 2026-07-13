"""Turns real DataCo PAYMENT_REVIEW orders into price_dispute exception contexts.

Real signal used: Order Status == "PAYMENT_REVIEW" — 1,893 rows out of 180,519
total. This is the ONLY thing
DataCo is used for: it has no supplier/seller entity at all (checked directly —
no column represents "who shipped this"), so it is deliberately NOT used for
late_shipment even though it has a Late_delivery_risk column and 98,977 "Late
delivery" rows — using it for that would mean fabricating a supplier_id.

File is latin-1 encoded (not UTF-8) — confirmed by trial when this was first
explored; pandas needs the encoding passed explicitly or it raises a
UnicodeDecodeError.

Run with: python scripts/ingest_dataco.py
Output: data/exceptions_pool_price_dispute.json
"""
import json
from pathlib import Path

import pandas as pd

RAW_FILE = Path(__file__).parent.parent / "data" / "raw" / "dataco" / "DataCoSupplyChainDataset.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "exceptions_pool_price_dispute.json"
MAX_ROWS = 500


def main() -> None:
    df = pd.read_csv(RAW_FILE, encoding="latin-1")

    disputed = df[df["Order Status"] == "PAYMENT_REVIEW"].copy()

    contexts = [
        {
            "exception_type": "price_dispute",
            "supplier_id": None,  # no real supplier entity in this dataset — flagged, not fabricated
            "order_ids": [str(row["Order Id"])],
            "revenue_at_risk_usd": round(float(row["Order Item Total"]), 2),
            "discount_rate": float(row["Order Item Discount Rate"]),
            "source": "dataco",
        }
        for _, row in disputed.head(MAX_ROWS).iterrows()
    ]

    OUTPUT_PATH.write_text(json.dumps(contexts, indent=2))
    print(f"DataCo: {len(disputed)} PAYMENT_REVIEW orders found, wrote {len(contexts)} contexts to {OUTPUT_PATH}")
    print("NOTE: supplier_id is None for every row — DataCo has no supplier entity. "
          "build_initial_dataset.py must assign a placeholder supplier for training "
          "purposes, or these rows should be used supplier-agnostic (internal-facing "
          "price-dispute drafts only, never a supplier-addressed email).")


if __name__ == "__main__":
    main()
