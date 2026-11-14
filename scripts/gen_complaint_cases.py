"""
# scripts/gen_complaint_cases.py

**What:** builds the demo database (`data/db/orders.json`) and the extractor dataset
(`data/datasets/complaint_cases.json`) in one pass from real Olist orders, with an LLM writing one messy
customer message per order.
**Why:** every message is generated from a known real order, so the answer key (order_id + claim_type) is exact.
Four situations map to three claim types: late and on_time (both `late_delivery`; on_time is the customer being
wrong, the only way the verifier can say FALSE), never_arrived, canceled. Hard cases are deliberate (spelled-out or
typo'd numbers, a red-herring order number, no order number at all). Generator and resolv/harness/validity.py derive truth
independently and must agree before any case is written. Resumable: every message is cached on its prompt.
**Used by:** run by hand: `python scripts/gen_complaint_cases.py`.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from litellm import completion
from tqdm import tqdm

load_dotenv()  # this script doesn't import config, so nothing else loads .env for it

DATA_DIR = Path(__file__).parent.parent / "data"
RAW = DATA_DIR / "raw" / "olist"
ORDERS_OUT = DATA_DIR / "db" / "orders.json"
CASES_OUT = DATA_DIR / "datasets" / "complaint_cases.json"
CACHE_FILE = DATA_DIR / "cache" / "complaint_cache.json"

SEED = 7

# Only orders promised on/after this date. A uniform two-year sample put 86% of orders outside the 90-day claim
# window (an always-deny agent would score 86%); 2018-05-01 puts about half inside it, so neither "always deny"
# nor "always allow" beats chance. Every situation's pool still exceeds what SAMPLES asks for.
MIN_PROMISED = "2018-05-01"

# Orders sampled per situation. `unavailable` was dropped: Olist has only 6 such orders, too few to train or measure.
SAMPLES = {"late": 180, "on_time": 100, "never_arrived": 100, "canceled": 80}
CLAIM_BY_SITUATION = {
    "late": "late_delivery",
    "on_time": "late_delivery",  # customer wrongly believes it was late -> verifier will say false
    "never_arrived": "never_arrived",
    "canceled": "order_canceled",
}

# One Groq key; on a 429 the generator waits and retries.
_GROQ_KEY = os.environ.get("GROQ_API_KEY")

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


# The difficulty of case i, cycling through the 30/40/30 mix.
def _difficulty(i: int) -> str:
    return _MIX[i % len(_MIX)]


# Every 3rd hard case drops the order number, so the model learns to answer "unknown" instead of inventing one.
def _omit_order(i: int, difficulty: str) -> bool:
    return difficulty == "hard" and i % 3 == 0


# Sample real Olist orders per situation into demo-DB records with friendly ORD-#### ids.
def build_orders() -> list[dict]:
    orders = pd.read_csv(
        RAW / "olist_orders_dataset.csv",
        parse_dates=["order_estimated_delivery_date", "order_delivered_customer_date"],
    )
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    amount = (items.groupby("order_id")["price"].sum() + items.groupby("order_id")["freight_value"].sum())

    est, act = "order_estimated_delivery_date", "order_delivered_customer_date"

    # Bucket on dates, not timestamps: validity.py compares date strings, and same-day "late by hours" orders were
    # once mislabelled late. main() asserts generator and verifier agree.
    orders["_est_d"] = orders[est].dt.normalize()
    orders["_act_d"] = orders[act].dt.normalize()
    orders = orders[orders["_est_d"] >= MIN_PROMISED]  # recent orders only — see MIN_PROMISED

    delivered = orders[orders["order_status"] == "delivered"]
    pools = {
        "late": delivered[delivered["_act_d"] > delivered["_est_d"]],
        "on_time": delivered[delivered["_act_d"] <= delivered["_est_d"]],
        "never_arrived": orders[orders["order_status"] == "shipped"],
        "canceled": orders[orders["order_status"] == "canceled"],
    }

    records, idx = [], 1000
    for situation, n in SAMPLES.items():
        df = pools[situation]
        df = df[df["order_id"].isin(amount.index)]  # need a real amount to quote
        df = df.sample(min(n, len(df)), random_state=SEED)
        for _, r in df.iterrows():
            delivered_date = r[act].date().isoformat() if pd.notna(r[act]) else None
            # from the normalised dates, so late always means days_late >= 1
            days_late = int((r["_act_d"] - r["_est_d"]).days) if situation == "late" else 0
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


# Stop the run unless every order's situation label matches what resolv/harness/validity.py independently derives.
def _assert_verifier_agrees(orders: list[dict]) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))  # run as a script; repo root isn't on the path
    from resolv.harness.validity import verify_claim
    from resolv.schemas import CustomerClaim

    disagreed = []
    for o in orders:
        claim = CustomerClaim(order_id=o["order_id"], claim_type=CLAIM_BY_SITUATION[o["situation"]])
        expected = o["situation"] != "on_time"  # on_time is FALSE by design
        if verify_claim(claim).claim_true is not expected:
            disagreed.append(o["order_id"])

    if disagreed:
        raise SystemExit(
            f"generator/verifier disagree on {len(disagreed)}/{len(orders)} orders "
            f"(e.g. {disagreed[:5]}) — the answer key is wrong, refusing to write cases."
        )
    print(f"  generator/verifier agree on all {len(orders)} orders")


# The generation prompt for one order: the grievance, the real facts, and the style for its difficulty.
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


# One customer message from Groq, waiting and retrying on rate limits.
# The model is the one the committed dataset was generated with (since retired from Groq's catalog).
def _generate(prompt: str) -> str:
    if not _GROQ_KEY:
        raise RuntimeError("No GROQ_API_KEY found in the environment or .env — nothing to generate with.")

    for _ in range(6):
        try:
            resp = completion(
                model="groq/llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                api_key=_GROQ_KEY,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" not in str(e).lower() and "429" not in str(e):
                raise
            time.sleep(10)
    raise RuntimeError("Groq rate limit persisted — rerun later, the cache resumes.")


# Write the order DB, check it against the verifier, then generate (or reuse cached) messages and write the cases.
def main() -> None:
    if not (RAW / "olist_orders_dataset.csv").exists():
        raise SystemExit(f"{RAW} not found — the raw Olist CSVs are required.")

    orders = build_orders()
    ORDERS_OUT.write_text(json.dumps(orders, indent=2))
    dist = {s: sum(1 for o in orders if o["situation"] == s) for s in SAMPLES}
    print(f"wrote {len(orders)} order records -> {ORDERS_OUT}  {dist}")
    _assert_verifier_agrees(orders)  # gate: a wrong answer key must stop the run, not flow downstream

    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    cases = []
    for i, order in enumerate(tqdm(orders, desc="Generating complaints")):
        difficulty = _difficulty(i)
        omit = _omit_order(i, difficulty)

        # cached on the prompt, not order_id: ORD-1000 is a sample position and can become a different order
        prompt = _prompt(order, difficulty, omit)
        key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        if key not in cache:
            cache[key] = _generate(prompt)
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
