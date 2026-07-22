"""Offline eval for the extractor — the one component whose job is purely reading.

    python -m eval.extractor                     # held-out split (92 cases)
    python -m eval.extractor --split all         # all 460
    python -m eval.extractor --save-baseline     # pin these numbers as the regression floor
    python -m eval.extractor --baseline          # compare against the pinned run and gate on it

WHY THIS IS SEPARATE FROM THE SWEEP. eval/runner.py grades the whole pipeline end to end, which
is the number that matters but also the number that hides things: the harness absorbs extraction
errors, so a bad read usually costs a wasted tool turn rather than a wrong refund, and pass^k
barely moves. That's good system design and bad measurement — the component with the most
headroom is the one the headline metric is least sensitive to. Grading it alone is the only way
to see it.

WHY EXACT MATCH AND NOT A SIMILARITY SCORE. "ORD-1007" and "ORD-1070" are equally wrong, and a
metric that scores them 0.87 similar is describing string edit distance, not the thing we care
about. The downstream consumer is a dict lookup — it matches or it doesn't. The eval should have
the same opinion the database does. (This is also exactly the RLVR reward from
scripts/train_extractor.py, which is the point: one definition of "got the facts right", used in
training, in production, and here.)

THE METRIC THAT ISN'T ACCURACY. `hallucination_rate` is measured on the 46 cases whose ground
truth order_id is None — messages where the customer never gave a usable number. There, inventing
a plausible "ORD-1204" is worse than any wrong-order error, because it sends a confident lookup
into a real record belonging to someone else. Overall accuracy cannot see this: those cases are
10% of the pool, so a model that hallucinates on every one of them still scores 0.9.

SPLIT CONSTANTS ARE DUPLICATED FROM scripts/train_extractor.py ON PURPOSE. That script has to run
standalone in a Kaggle notebook where this repo isn't installed, so it cannot import from here,
and importing IT from here would pull torch and trl onto a CPU box. Two constants and four lines
of shuffling is the cheaper duplication. They must stay in sync — tests/test_extractor_eval.py
asserts the split size, so a drift fails CI rather than silently comparing different held-out sets.
"""
import argparse
import asyncio
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("LITELLM_LOG", "ERROR")

from tqdm import tqdm

from agents.loop import extract
from agents.runner_utils import tokens_split
from config import LLM_PROVIDER, MODEL
from eval.metrics import mcnemar, standard_error, wilson
from eval.runner import USD_PER_MTOK_IN, USD_PER_MTOK_OUT

CASES = Path(__file__).parent.parent / "data" / "datasets" / "complaint_cases.json"
OUT_DIR = Path(__file__).parent.parent / "data" / "eval"
BASELINE = Path(__file__).parent / "baselines" / "extractor.json"

SEED = 0
TEST_FRACTION = 0.2

# Extraction is one independent call per case with no shared state, so unlike the sweep (which is
# pinned to one worker by the rate-limit backoff maths) this can fan out. Modest anyway: the point
# is to finish in two minutes, not to find the provider's ceiling.
WORKERS = 4


def load_cases(split: str = "test") -> list[dict]:
    """The graded cases. `test` is the same held-out fifth the fine-tune never trained on.

    Same seed, same shuffle, same fraction as scripts/train_extractor.py — so the both-right
    number here is directly comparable to the 0.489 (base) / 0.707 (tuned) Qwen figures, rather
    than being a different number that merely looks like one.
    """
    cases = sorted(json.loads(CASES.read_text(encoding="utf-8")), key=lambda c: c["case_id"])
    if split == "all":
        return cases
    shuffled = list(cases)
    random.Random(SEED).shuffle(shuffled)
    return shuffled[: int(len(shuffled) * TEST_FRACTION)]


def _norm(order_id) -> str:
    """None and the string 'none' are the same answer: 'the customer gave no usable number'."""
    return (order_id or "none").strip().lower()


def _one(case: dict) -> dict:
    """Extract one case and score it. Never raises — a crashed call is a wrong answer, recorded.

    Swallowing the exception into a row rather than letting it kill the run is the same rule
    eval/runner.py follows: dropping failed cases would quietly raise the score by removing
    exactly the messages that went worst.
    """
    truth_order = _norm(case["ground_truth"]["order_id"])
    truth_claim = case["ground_truth"]["claim_type"]

    started = time.perf_counter()
    before = tokens_split()
    try:
        claim = asyncio.run(extract(case["message"])) or {}
        got_order, got_claim, error = _norm(claim.get("order_id")), claim.get("claim_type") or "", None
    except Exception as e:
        got_order, got_claim, error = "", "", f"{type(e).__name__}: {e}"
    prompt, completion = (a - b for a, b in zip(tokens_split(), before))

    order_ok = got_order == truth_order
    claim_ok = got_claim == truth_claim
    return {
        "case_id": case["case_id"],
        "difficulty": case["difficulty"],
        "truth_order": truth_order,
        "truth_claim": truth_claim,
        "got_order": got_order,
        "got_claim": got_claim,
        "order_ok": order_ok,
        "claim_ok": claim_ok,
        "both_ok": order_ok and claim_ok,
        # Only defined where the customer gave no number. None elsewhere so the rate divides by
        # the cases that actually offered the chance to invent one.
        "hallucinated": (got_order not in ("none", "")) if truth_order == "none" else None,
        "latency_s": round(time.perf_counter() - started, 2),
        "usd": round(prompt / 1e6 * USD_PER_MTOK_IN + completion / 1e6 * USD_PER_MTOK_OUT, 6),
        **({"error": error} if error else {}),
    }


def score(rows: list[dict]) -> dict:
    """Aggregate. Every rate carries the count it was computed from — a percentage without its
    denominator is not a result, and the per-difficulty slices get small fast."""
    n = len(rows)
    if not n:
        return {"n": 0}

    both = sum(r["both_ok"] for r in rows)
    asked = [r for r in rows if r["hallucinated"] is not None]

    by_difficulty = {}
    for level in sorted({r["difficulty"] for r in rows}):
        rs = [r for r in rows if r["difficulty"] == level]
        by_difficulty[level] = {"n": len(rs), "both": round(sum(r["both_ok"] for r in rs) / len(rs), 4)}

    # Which claim types get mistaken for which. An overall claim-type accuracy of 0.87 can be one
    # type at 0.99 and another at 0.40; the aggregate is the only number that hides that, and the
    # weak type is the one worth fixing.
    confusion = {t: dict(Counter(r["got_claim"] or "(none)" for r in rows if r["truth_claim"] == t))
                 for t in sorted({r["truth_claim"] for r in rows})}

    return {
        "n": n,
        "order_acc": round(sum(r["order_ok"] for r in rows) / n, 4),
        "claim_acc": round(sum(r["claim_ok"] for r in rows) / n, 4),
        "both_acc": round(both / n, 4),
        # Wilson, not p +/- z*SE: both_acc sits high enough that the normal approximation starts
        # reporting bounds above 1.0.
        "both_ci95": tuple(round(x, 4) for x in wilson(both, n)),
        "both_se": round(standard_error(both / n, n), 4),
        "hallucination_rate": round(sum(r["hallucinated"] for r in asked) / len(asked), 4) if asked else None,
        "abstention_n": len(asked),
        "by_difficulty": by_difficulty,
        "claim_confusion": confusion,
        "errors": sum(1 for r in rows if r.get("error")),
        "p50_latency_s": round(sorted(r["latency_s"] for r in rows)[n // 2], 2),
        "usd_total": round(sum(r["usd"] for r in rows), 4),
    }


def compare(rows: list[dict], baseline: dict) -> dict:
    """Paired comparison against the pinned run. Returns the McNemar verdict and a pass/fail.

    PAIRED, because the same messages were scored both times: the question is not "are the two
    averages different" but "which cases changed, and in which direction". b and c are recoverable
    only because both runs stored per-case rows — the exact thing the last fine-tune could not do.

    Two failure conditions, deliberately different in kind:
      - a SIGNIFICANT drop in both-right accuracy (McNemar) — the model got worse
      - ANY rise in hallucination, significant or not — the model got less safe, and a safety
        regression does not get to hide behind a small sample.
    """
    was = {r["case_id"]: r for r in baseline["rows"]}
    paired = [(was[r["case_id"]], r) for r in rows if r["case_id"] in was]

    b = sum(1 for old, new in paired if old["both_ok"] and not new["both_ok"])   # regressions
    c = sum(1 for old, new in paired if not old["both_ok"] and new["both_ok"])   # improvements
    chi2, significant = mcnemar(b, c)

    now, before = score(rows), baseline["score"]
    hallucination_delta = (now["hallucination_rate"] or 0) - (before["hallucination_rate"] or 0)

    return {
        "paired_cases": len(paired),
        "regressed": b,
        "improved": c,
        "chi2": round(chi2, 3),
        "significant": significant,
        "both_acc_delta": round(now["both_acc"] - before["both_acc"], 4),
        "hallucination_delta": round(hallucination_delta, 4),
        # A significant change is only a FAILURE if it went the wrong way — a significant
        # improvement must not fail the build.
        "failed": bool((significant and b > c) or hallucination_delta > 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade the extractor on its own, against known truth.")
    ap.add_argument("--split", choices=["test", "all"], default="test",
                    help="test = the held-out fifth the fine-tune never saw (comparable numbers)")
    ap.add_argument("--limit", type=int, default=0, help="first N cases only — a smoke run")
    ap.add_argument("--out", default="extractor.jsonl")
    ap.add_argument("--save-baseline", action="store_true", help="pin this run as the regression floor")
    ap.add_argument("--baseline", action="store_true", help="compare against the pinned run; exit 1 on regression")
    args = ap.parse_args()

    cases = load_cases(args.split)[: args.limit or None]
    print(f"provider={LLM_PROVIDER}  model={MODEL}")
    print(f"{len(cases)} cases ({args.split} split), {WORKERS} workers")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(tqdm(pool.map(_one, cases), total=len(cases), unit="case", desc="extract"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / args.out
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    card = score(rows)
    print(f"\nextractor scorecard over {card['n']} cases -> {out.name}\n")
    for key, value in card.items():
        if key not in ("by_difficulty", "claim_confusion"):
            print(f"  {key:<20}  {value}")
    print("\n  by difficulty:")
    for level, s in card["by_difficulty"].items():
        print(f"    {level:<10} n={s['n']:<4} both={s['both']}")
    print("\n  claim type -> what it predicted:")
    for truth, preds in card["claim_confusion"].items():
        print(f"    {truth:<16} {preds}")

    if card["hallucination_rate"]:
        print(f"\n  !! invented an order number on {card['hallucination_rate']:.1%} of the "
              f"{card['abstention_n']} cases that gave none — a confident lookup into")
        print("     someone else's record is worse than any wrong-order error.")

    if args.save_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(
            json.dumps({"model": MODEL, "split": args.split, "score": card, "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\n  baseline pinned -> {BASELINE}")

    if args.baseline:
        if not BASELINE.exists():
            raise SystemExit(f"no baseline at {BASELINE}. Run --save-baseline first.")
        verdict = compare(rows, json.loads(BASELINE.read_text(encoding="utf-8")))
        print("\n  vs baseline:")
        for key, value in verdict.items():
            print(f"    {key:<22}  {value}")
        if verdict["failed"]:
            raise SystemExit("\n  REGRESSION — the extractor got worse or less safe. Build fails.")
        print("\n  no regression.")


if __name__ == "__main__":
    main()
