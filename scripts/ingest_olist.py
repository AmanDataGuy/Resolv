"""Turns real Olist late-delivery orders into late_shipment exception contexts.

Real signal used: order_status == "delivered" AND order_delivered_customer_date
> order_estimated_delivery_date — 7,826 rows out of 99,441 total orders. Capped at
800 to stay a manageable dataset while clearing the 500-example floor recommended
for narrow fine-tuning.

The promised/delivered dates and customer_id are carried through deliberately: a
realistic customer complaint says "it was due on the 14th and turned up on the 20th",
so the downstream generator (scripts/gen_complaint_cases.py) needs the real dates, not
just the day count. These same dates are what the harness later checks a claim against.

Olist has a real seller_id (a genuine marketplace entity) — this is why Olist,
not DataCo, is the late_shipment source: DataCo has no supplier/seller column at
all, so using it here would mean fabricating a supplier_id, teaching a fine-tuned
model a fake join.

Run with: python scripts/ingest_olist.py
Output: data/exceptions_pool_late_shipment.json
"""
import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "olist"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "pools" / "exceptions_pool_late_shipment.json"
MAX_ROWS = 800


def main() -> None:
    orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv", parse_dates=[
        "order_estimated_delivery_date", "order_delivered_customer_date"
    ])
    items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")

    late = orders[
        (orders["order_status"] == "delivered")
        & (orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"])
    ].copy()
    late["delay_days"] = (
        late["order_delivered_customer_date"] - late["order_estimated_delivery_date"]
    ).dt.days

    per_order = (
        items.groupby(["order_id", "seller_id"])
        .agg(revenue_at_risk_usd=("price", "sum"), freight_value=("freight_value", "sum"))
        .reset_index()
    )
    per_order["revenue_at_risk_usd"] = per_order["revenue_at_risk_usd"] + per_order["freight_value"]

    merged = late.merge(per_order, on="order_id", how="inner")

    contexts = [
        {
            "exception_type": "late_shipment",
            "supplier_id": row["seller_id"],
            "order_ids": [row["order_id"]],
            "customer_id": row["customer_id"],
            "delay_days": int(row["delay_days"]),
            "revenue_at_risk_usd": round(float(row["revenue_at_risk_usd"]), 2),
            # real dates — a complaint quotes them, and the harness verifies against them
            "promised_date": row["order_estimated_delivery_date"].date().isoformat(),
            "delivered_date": row["order_delivered_customer_date"].date().isoformat(),
            "source": "olist",
        }
        for _, row in merged.head(MAX_ROWS).iterrows()
    ]

    OUTPUT_PATH.write_text(json.dumps(contexts, indent=2))
    print(f"Olist: {len(late)} late-delivered orders found, wrote {len(contexts)} contexts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
