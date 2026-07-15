"""Generates the complaint dataset: messy customer messages built from REAL late Olist
deliveries, each paired with an exact answer key.

WHY GENERATE. No public dataset pairs a messy customer complaint with the verified order
it refers to — that label is proprietary support correspondence, the same wall we hit for
the resolution emails. But we can work backwards. Every case starts from a REAL Olist late
delivery (real order, real promised/delivered dates, real value), so **the answer is known
before the message exists**. An LLM then writes what an annoyed customer would type about
that specific order. Result: a genuinely hard extraction task with exact ground truth —
which is what makes the RLVR reward verifiable instead of judged.

Only the conversational wrapper is synthetic; every fact underneath is real Olist data.

WHAT THE EXTRACTOR IS SCORED ON (schemas.CustomerClaim): which ORDER and what KIND of
claim. Deliberately not the amount or dates — customers misremember those, and the harness
looks them up from data/orders.json instead. The model reads; the harness knows the numbers.

HARD CASES ARE THE POINT. The previous fine-tune failed because the task was too easy (the
answer was already in the input, so the reward saturated). Here difficulty is injected on
purpose: order numbers spelled out digit-by-digit, typo'd, buried mid-sentence, a red-herring
second order number, a confidently-wrong amount, and — for a share of hard cases — no order
number at all, where the correct answer is None. A model that hallucinates an order id on
those is wrong, and the reward says so.

Two outputs:
  data/orders.json          — real Olist orders as a lookup table with friendly IDs.
                              THIS is the record the harness verifies a claim against.
  data/complaint_cases.json — the messy messages + answer keys (easy/medium/hard).

Prereq: run scripts/ingest_olist.py first (it now emits promised_date/delivered_date).
Resumable: every generated message is cached, so an interrupted run costs nothing to redo.

Run: python scripts/gen_complaint_cases.py
"""
import json
import os
import time
from pathlib import Path

from litellm import completion
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
POOL_FILE = DATA_DIR / "pools" / "exceptions_pool_late_shipment.json"
ORDERS_OUT = DATA_DIR / "db" / "orders.json"          # the demo DB the harness verifies against
CASES_OUT = DATA_DIR / "datasets" / "complaint_cases.json"
CACHE_FILE = DATA_DIR / "cache" / "complaint_cache.json"

MAX_CASES = 200  # start small: eyeball quality before spending on 800

# Groq is 30 req/min per key — rotate across the configured keys and back off, same as
# scripts/build_initial_dataset.py. Kept on Groq (not Gemini) because this is high-volume
# offline generation and the Gemini key is quota-limited for the live path.
_GROQ_KEYS = [
    k for k in (os.environ.get("GROQ_API_KEY"), os.environ.get("GROQ_API_KEY_2"), os.environ.get("GROQ_API_KEY_3")) if k
]
_key_i = 0

SYSTEM = (
    "You write realistic customer-support chat messages. Output ONLY the message the "
    "customer would type — no preamble, no quotes, no explanation, no signature."
)

_STYLE = {
    "easy": "Annoyed but coherent. State the order number plainly. Normal punctuation.",
    "medium": (
        "Ramble a little. Bury the order number mid-sentence. Misremember the amount by a "
        "few dollars. Mention one irrelevant detail (the packaging, the courier)."
    ),
    "hard": (
        "Genuinely messy: spell the order number out digit-by-digit or fumble it, mention a "
        "DIFFERENT unrelated order number as a red herring, confidently quote a wrong amount, "
        "add an unrelated grievance, and use lowercase with sloppy punctuation."
    ),
}

# 30 / 40 / 30 split — enough hard cases to leave the model real room to improve.
_MIX = ["easy"] * 3 + ["medium"] * 4 + ["hard"] * 3


def _difficulty(i: int) -> str:
    return _MIX[i % len(_MIX)]


def _omit_order(i: int, difficulty: str) -> bool:
    """Every 3rd hard case drops the order number entirely — teaches the model that the
    right answer is sometimes 'I don't know' rather than an invented order id."""
    return difficulty == "hard" and i % 3 == 0


def _build_orders(pool: list[dict]) -> list[dict]:
    """Real Olist rows -> order records with a friendly, human-speakable ID.

    Olist's real order ids are 32-char hashes; no customer would ever read one out. The
    friendly id is what appears in the complaint, and the hash is retained so every record
    traces back to the real row.
    """
    orders = []
    for i, ctx in enumerate(pool[:MAX_CASES]):
        orders.append(
            {
                "order_id": f"ORD-{1000 + i}",
                "olist_order_id": ctx["order_ids"][0],
                "customer_id": ctx.get("customer_id"),
                "seller_id": ctx.get("supplier_id"),
                "amount_usd": ctx["revenue_at_risk_usd"],
                "promised_date": ctx["promised_date"],
                "delivered_date": ctx["delivered_date"],
                "days_late": ctx["delay_days"],
                "status": "delivered_late",
            }
        )
    return orders


def _prompt(order: dict, difficulty: str, omit: bool) -> str:
    lines = [
        "Write a message from an annoyed customer about a LATE DELIVERY.",
        "",
        "The real facts (write like a person would — imprecise and emotional, not a report):",
        f"- Order number: {order['order_id']}",
        f"- Order value: ${order['amount_usd']:.2f}",
        f"- Promised delivery: {order['promised_date']}",
        f"- Actually arrived: {order['delivered_date']} ({order['days_late']} days late)",
        "",
        f"Style: {_STYLE[difficulty]}",
    ]
    if omit:
        lines.append(
            "IMPORTANT: do NOT mention the order number anywhere — this customer doesn't have it to hand."
        )
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
    if not POOL_FILE.exists():
        raise SystemExit(f"{POOL_FILE} not found — run scripts/ingest_olist.py first.")
    pool = json.loads(POOL_FILE.read_text())
    if pool and "promised_date" not in pool[0]:
        raise SystemExit(
            "Pool has no promised_date — re-run scripts/ingest_olist.py (it now emits the real dates)."
        )

    orders = _build_orders(pool)
    ORDERS_OUT.write_text(json.dumps(orders, indent=2))
    print(f"wrote {len(orders)} order records -> {ORDERS_OUT}")

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
                # the answer key: which order, what kind of claim. None when the customer
                # never gave an order number — the model must say so, not invent one.
                "ground_truth": {
                    "order_id": None if omit else order["order_id"],
                    "claim_type": "late_delivery",
                    "stated_amount_usd": None,
                },
            }
        )

    CASES_OUT.write_text(json.dumps(cases, indent=2))
    counts = {d: sum(1 for c in cases if c["difficulty"] == d) for d in ("easy", "medium", "hard")}
    unknown = sum(1 for c in cases if c["ground_truth"]["order_id"] is None)
    print(f"wrote {len(cases)} cases -> {CASES_OUT}")
    print(f"  difficulty: {counts}")
    print(f"  no-order-id (answer must be 'unknown'): {unknown}")


if __name__ == "__main__":
    main()
