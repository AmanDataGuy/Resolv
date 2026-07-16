"""Builds the extractor's dataset AND the demo database, in one pass, from real Olist data.

WHY. The first fine-tune failed because the task was too easy (the answer was already in the
input) — see study/finetuning_report.md sec 12. This dataset fixes that: the model must read a
messy customer message and recover WHICH order and WHAT KIND of problem. Ground truth is exact
because every message is generated FROM a known real Olist order, so the reward stays verifiable.

FIVE REAL SITUATIONS (every claim type is backed by a real Olist order_status — nothing invented):

    situation      Olist source                         claim_type        verifier truth
    late           delivered, actual > estimated        late_delivery     true
    on_time        delivered, actual <= estimated       late_delivery     FALSE (customer wrong)
    never_arrived  status == shipped (never delivered)  never_arrived     true
    canceled       status == canceled                   order_canceled    true
    unavailable    status == unavailable                item_unavailable  true

Including on_time is the point: it's the only way the verifier can ever answer FALSE, which is what
gives the reward variance (the saturation fix, applied at the data level this time).

TWO OUTPUTS:
    data/db/orders.json             — the demo database the harness verifies claims against.
    data/datasets/complaint_cases.json — the messy messages + exact answer keys.

Scored on (schemas.CustomerClaim): order_id + claim_type. NOT the amount/dates — customers
misremember those, so the harness looks them up from orders.json. The model reads; the harness
knows the numbers.

HARD CASES ARE DELIBERATE: spelled-out/typo'd order numbers, a red-herring second order number, a
confidently-wrong amount, and a share of hard cases with NO order number (answer must be None).

Resumable (caches every message). Run: python scripts/gen_complaint_cases.py
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
from litellm import completion
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
RAW = DATA_DIR / "raw" / "olist"
ORDERS_OUT = DATA_DIR / "db" / "orders.json"
CASES_OUT = DATA_DIR / "datasets" / "complaint_cases.json"
CACHE_FILE = DATA_DIR / "cache" / "complaint_cache.json"

SEED = 7
# how many orders to sample per situation (real Olist has far more of each)
SAMPLES = {"late": 180, "on_time": 100, "never_arrived": 100, "canceled": 80, "unavailable": 80}
CLAIM_BY_SITUATION = {
    "late": "late_delivery",
    "on_time": "late_delivery",  # customer wrongly believes it was late -> verifier will say false
    "never_arrived": "never_arrived",
    "canceled": "order_canceled",
    "unavailable": "item_unavailable",
}

# Groq: 30 req/min/key. Rotate + back off (same as build_initial_dataset). Offline high-volume
# work stays on Groq's free tier; the Gemini key is reserved for the low-volume live path.
_GROQ_KEYS = [
    k for k in (os.environ.get("GROQ_API_KEY"), os.environ.get("GROQ_API_KEY_2"), os.environ.get("GROQ_API_KEY_3")) if k
]
_key_i = 0

SYSTEM = (
    "You write realistic customer-support chat messages. Output ONLY the message the customer "
    "would type — no preamble, no quotes, no explanation, no signature."
)

# what the customer is upset about, per situation
_SITUATION_GRIEVANCE = {
    "late": "Their order arrived LATER than promised. They want to know why and what you'll do.",
    "on_time": (
        "They are ANGRY and convinced their order arrived LATE and demand a refund — even though "
        "it actually arrived on or before the promised date. Write them insisting it was late."
    ),
    "never_arrived": "Their order shipped but has NEVER ARRIVED. They are still waiting and frustrated.",
    "canceled": "Their order was CANCELED without a clear reason. They are confused and want it resolved.",
    "unavailable": "They were told the item is UNAVAILABLE / out of stock after ordering. They want a fix.",
}

_STYLE = {
    "easy": "Annoyed but coherent. State the order number plainly. Normal punctuation.",
    "medium": "Ramble a little. Bury the order number mid-sentence. Mention one irrelevant detail.",
    "hard": (
        "Genuinely messy: spell the order number out digit-by-digit or fumble it, mention a "
        "DIFFERENT unrelated order number as a red herring, add an unrelated grievance, lowercase "
        "with sloppy punctuation."
    ),
}
_MIX = ["easy"] * 3 + ["medium"] * 4 + ["hard"] * 3  # 30/40/30


def _difficulty(i: int) -> str:
    return _MIX[i % len(_MIX)]


def _omit_order(i: int, difficulty: str) -> bool:
    """Every 3rd hard case drops the order number — teaches the model to answer 'unknown'
    rather than invent an order id."""
    return difficulty == "hard" and i % 3 == 0


def build_orders() -> list[dict]:
    """Sample real Olist orders across the five situations into demo-DB records with friendly IDs."""
    orders = pd.read_csv(
        RAW / "olist_orders_dataset.csv",
        parse_dates=["order_estimated_delivery_date", "order_delivered_customer_date"],
    )
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    amount = (items.groupby("order_id")["price"].sum() + items.groupby("order_id")["freight_value"].sum())

    delivered = orders[orders["order_status"] == "delivered"]
    est, act = "order_estimated_delivery_date", "order_delivered_customer_date"
    pools = {
        "late": delivered[delivered[act] > delivered[est]],
        "on_time": delivered[delivered[act] <= delivered[est]],
        "never_arrived": orders[orders["order_status"] == "shipped"],
        "canceled": orders[orders["order_status"] == "canceled"],
        "unavailable": orders[orders["order_status"] == "unavailable"],
    }

    records, idx = [], 1000
    for situation, n in SAMPLES.items():
        df = pools[situation]
        df = df[df["order_id"].isin(amount.index)]  # need a real amount to quote
        df = df.sample(min(n, len(df)), random_state=SEED)
        for _, r in df.iterrows():
            delivered_date = r[act].date().isoformat() if pd.notna(r[act]) else None
            days_late = int((r[act] - r[est]).days) if situation == "late" and pd.notna(r[act]) else 0
            records.append(
                {
                    "order_id": f"ORD-{idx}",
                    "olist_order_id": r["order_id"],
                    "customer_id": r["customer_id"],
                    "amount_usd": round(float(amount[r["order_id"]]), 2),
                    "promised_date": r[est].date().isoformat(),
                    "delivered_date": delivered_date,
                    "days_late": days_late,
                    "situation": situation,
                    "status": r["order_status"],
                }
            )
            idx += 1
    return records


def _prompt(order: dict, difficulty: str, omit: bool) -> str:
    situation = order["situation"]
    lines = [
        _SITUATION_GRIEVANCE[situation],
        "",
        "The real facts (write like a person would — imprecise and emotional, not a report):",
        f"- Order number: {order['order_id']}",
        f"- Order value: ${order['amount_usd']:.2f}",
        f"- Promised delivery: {order['promised_date']}",
    ]
    if order["delivered_date"]:
        lines.append(f"- Arrived: {order['delivered_date']}")
    lines += ["", f"Style: {_STYLE[difficulty]}"]
    if omit:
        lines.append("IMPORTANT: do NOT mention the order number — this customer doesn't have it to hand.")
    lines.append("Write 2-5 sentences, as typed into a website support chat.")
    return "\n".join(lines)


def _generate(prompt: str) -> str:
    global _key_i
    for _ in range(len(_GROQ_KEYS) * 3):
        try:
            resp = completion(
                model="groq/llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                api_key=_GROQ_KEYS[_key_i % len(_GROQ_KEYS)],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" not in str(e).lower() and "429" not in str(e):
                raise
            _key_i += 1
            time.sleep(2)
    raise RuntimeError("Groq rate limit: all keys exhausted — rerun later, the cache resumes.")


def main() -> None:
    if not (RAW / "olist_orders_dataset.csv").exists():
        raise SystemExit(f"{RAW} not found — the raw Olist CSVs are required.")

    orders = build_orders()
    ORDERS_OUT.write_text(json.dumps(orders, indent=2))
    dist = {s: sum(1 for o in orders if o["situation"] == s) for s in SAMPLES}
    print(f"wrote {len(orders)} order records -> {ORDERS_OUT}  {dist}")

    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    cases = []
    for i, order in enumerate(tqdm(orders, desc="Generating complaints")):
        difficulty = _difficulty(i)
        omit = _omit_order(i, difficulty)
        key = order["order_id"]
        if key not in cache:
            cache[key] = _generate(_prompt(order, difficulty, omit))
            CACHE_FILE.write_text(json.dumps(cache, indent=2))
        cases.append(
            {
                "case_id": f"case-{i:04d}",
                "channel": "chat",
                "difficulty": difficulty,
                "message": cache[key],
                "ground_truth": {
                    "order_id": None if omit else order["order_id"],
                    "claim_type": CLAIM_BY_SITUATION[order["situation"]],
                    "stated_amount_usd": None,
                },
            }
        )

    CASES_OUT.write_text(json.dumps(cases, indent=2))
    claims = {c: sum(1 for x in cases if x["ground_truth"]["claim_type"] == c) for c in set(CLAIM_BY_SITUATION.values())}
    diff = {d: sum(1 for x in cases if x["difficulty"] == d) for d in ("easy", "medium", "hard")}
    unknown = sum(1 for x in cases if x["ground_truth"]["order_id"] is None)
    print(f"wrote {len(cases)} cases -> {CASES_OUT}")
    print(f"  claim types: {claims}")
    print(f"  difficulty:  {diff}")
    print(f"  no-order-id (answer must be 'unknown'): {unknown}")


if __name__ == "__main__":
    main()
