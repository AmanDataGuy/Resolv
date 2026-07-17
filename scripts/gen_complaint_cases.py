"""Builds the extractor's dataset AND the demo database, in one pass, from real Olist data.

WHY. The first fine-tune failed because the task was too easy (the answer was already in the
input) — see study/finetuning_report.md sec 12. This dataset fixes that: the model must read a
messy customer message and recover WHICH order and WHAT KIND of problem. Ground truth is exact
because every message is generated FROM a known real Olist order, so the reward stays verifiable.

FOUR REAL SITUATIONS -> THREE CLAIM TYPES (each backed by a real Olist order_status — nothing
invented):

    situation      Olist source                          claim_type      verifier truth
    late           delivered, actual DATE > estimated    late_delivery   true
    on_time        delivered, actual DATE <= estimated   late_delivery   FALSE (customer wrong)
    never_arrived  status == shipped (never delivered)   never_arrived   true
    canceled       status == canceled                    order_canceled  true

Including on_time is the point: it's the only way the verifier can ever answer FALSE, which is what
gives the reward variance (the saturation fix, applied at the data level this time).

Note DATE, not timestamp — see build_orders(). And `unavailable` was dropped; Olist only has 6
such orders, which is not a class, it's a rounding error. See SAMPLES.

The generator and harness/validity.py derive ground truth by two independent paths. main() asserts
they agree on every order before writing any cases — see _assert_verifier_agrees().

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

# Only sample orders promised on/after this date.
#
# WHY THIS EXISTS. Olist spans 2016-09 to 2018-11. harness/policy.py pins the demo clock at
# 2018-10-20 and denies claims on orders older than CLAIM_WINDOW_DAYS (90). Sampling uniformly
# across two years left only 14% of orders inside that window — so 86% of eval tasks would be
# correctly resolved by "deny", and an agent that denied everything unconditionally would score
# 86%. That's a benchmark measuring nothing: the same failure as a saturated reward, arrived at
# from the data side instead of the reward side.
#
# A real support queue holds recent orders anyway — the uniform sample was the artificial part.
# This date was picked by sweeping it against the in-window fraction (2018-04-01 -> 33%,
# 2018-06-01 -> 65%, 2018-06-15 -> 80%). 2018-05-01 lands 52%, near a coin flip, so neither
# "always deny" nor "always allow" beats chance and the window rule fires often without
# dominating. Pools after the filter (late 2275, on_time 35016, shipped 419, canceled 214) all
# still exceed what SAMPLES asks for — verified, no situation comes up short.
MIN_PROMISED = "2018-05-01"

# How many orders to sample per situation (real Olist has far more of each).
#
# `unavailable` is GONE. It asked for 80 and Olist only has 6 orders with that status, so it
# was a 6-example class: too few to train on, too few to measure (a per-class accuracy over
# n=6 moves in 17-point steps, and pass^k over 6 tasks reports nothing). Four claim types with
# one of them a rounding error is worse than three real ones.
SAMPLES = {"late": 180, "on_time": 100, "never_arrived": 100, "canceled": 80}
CLAIM_BY_SITUATION = {
    "late": "late_delivery",
    "on_time": "late_delivery",  # customer wrongly believes it was late -> verifier will say false
    "never_arrived": "never_arrived",
    "canceled": "order_canceled",
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

    est, act = "order_estimated_delivery_date", "order_delivered_customer_date"

    # Bucket on DATES, not timestamps. Olist parses these as datetimes, but a record stores
    # `.date()` and harness/validity.py compares those date STRINGS. Splitting on the raw
    # timestamp put "promised the 13th 00:00, arrived the 13th 18:04" in the late pool — late
    # by hours, same date, so the verifier called it on-time. That silently mislabelled 20 of
    # 180 late orders. Truncate first, then compare, and generator + verifier agree by
    # construction — asserted in main() so it can never drift back.
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
            # From the normalised dates, for the same reason the pools are: days_late must
            # agree with what a date-string comparison sees, so late => days_late >= 1 always.
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


def _assert_verifier_agrees(orders: list[dict]) -> None:
    """Every order's `situation` label must match what harness/validity.py independently says.

    These are two separate code paths to the same truth: the generator labels from the Olist
    CSV columns, the verifier re-derives it from the written record. They must never disagree.
    When they did (the timestamp/date bug in build_orders), 20 orders carried a `late` label
    the verifier scored FALSE — harmless for extractor training, but it would have silently
    handed the agent eval the wrong answer key on 8% of tasks, and nothing would have flagged
    it. A wrong number that looks right is worse than no number.

    Run on every regeneration, so the bug cannot creep back in quietly.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))  # run as a script; repo root isn't on the path
    from harness.validity import verify_claim
    from schemas import CustomerClaim

    disagreed = []
    for o in orders:
        claim = CustomerClaim(order_id=o["order_id"], claim_type=CLAIM_BY_SITUATION[o["situation"]])
        # on_time is the one situation where the claim is FALSE by design — that's the whole
        # point of including it (it's the only way the verifier can ever answer no).
        expected = o["situation"] != "on_time"
        if verify_claim(claim).claim_true is not expected:
            disagreed.append(o["order_id"])

    if disagreed:
        raise SystemExit(
            f"generator/verifier disagree on {len(disagreed)}/{len(orders)} orders "
            f"(e.g. {disagreed[:5]}) — the answer key is wrong, refusing to write cases."
        )
    print(f"  generator/verifier agree on all {len(orders)} orders")


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
    # Check explicitly rather than letting range(0) fall through to the raise below. With no
    # keys loaded that loop runs zero times and reports "all keys exhausted" — which sent me
    # looking at Groq's rate limits when the real cause was that .env had never been read.
    # An error message that names the wrong cause is worse than no error message.
    if not _GROQ_KEYS:
        raise RuntimeError("No GROQ_API_KEY* found in the environment or .env — nothing to generate with.")

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
    _assert_verifier_agrees(orders)  # gate: a wrong answer key must stop the run, not flow downstream

    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    cases = []
    for i, order in enumerate(tqdm(orders, desc="Generating complaints")):
        difficulty = _difficulty(i)
        omit = _omit_order(i, difficulty)

        # Cache on the PROMPT, not on order_id. "ORD-1000" is a position in the sample, not an
        # identity: change the pools and ORD-1000 becomes a different Olist order, so an
        # order_id-keyed cache would hand the new order the old one's message — quoting a
        # different amount, different dates, even a different order number in the text. The
        # ground truth would be wrong and nothing would say so. Hashing the prompt means a hit
        # is only ever a message generated from exactly these facts.
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
