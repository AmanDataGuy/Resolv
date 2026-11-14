"""
# evals/extractor.py

**What:** offline eval of the extractor alone (message -> order_id + claim_type) on the held-out fifth of the case
file, exact match, with a McNemar regression gate against a same-model pinned run.
**Why:** the harness absorbs extraction errors, so the end-to-end sweep barely sees them; grading the component
alone is the only way to measure it. Exact match because the consumer is a dict lookup. `hallucination_rate` is
measured only on cases with no order id, where inventing one sends a lookup into someone else's record. The split
constants are duplicated from scripts/train_extractor.py (it runs standalone on Kaggle); a test keeps them in sync.
**Used by:** `python -m evals.extractor`, inference/serve_bench.py (`_norm`, `load_cases`, `score`), tests.
"""
import os

os.environ.setdefault("RESOLV_ENV", "eval")
os.environ.setdefault("LITELLM_LOG", "ERROR")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from tqdm import tqdm  # noqa: E402

from resolv import config  # noqa: E402
from evals import provenance  # noqa: E402
from evals.metrics import mcnemar, standard_error, wilson  # noqa: E402

CASES = Path(__file__).parent.parent / "data" / "datasets" / "complaint_cases.json"
BASELINE = Path(__file__).parent / "baselines" / "extractor.json"
SEED = 0
TEST_FRACTION = 0.2
# one independent call per case, so this can fan out a little
WORKERS = 4


# the graded cases: "test" is the held-out fifth the fine-tune never saw, "all" is every case
def load_cases(split: str = "test") -> list[dict]:
    if not CASES.exists():
        raise FileNotFoundError(f"{CASES} is missing; the extractor eval needs the committed case file.")
    cases = sorted(json.loads(CASES.read_text(encoding="utf-8")), key=lambda c: c["case_id"])
    if split == "all":
        return cases
    shuffled = list(cases)
    random.Random(SEED).shuffle(shuffled)
    return shuffled[: int(len(shuffled) * TEST_FRACTION)]


# None and "none" are the same answer: the customer gave no usable number
def _norm(order_id) -> str:
    return (order_id or "none").strip().lower()


# extract one case and score it; a crashed call is a wrong answer, recorded, never dropped
def _one(case: dict) -> dict:
    from resolv.agents.loop import extract
    from resolv.agents.llm import new_meter, served_model, tokens_split

    truth_order = _norm(case["ground_truth"]["order_id"])
    truth_claim = case["ground_truth"]["claim_type"]

    # extract and read the meter in the same task context
    async def drive():
        new_meter()
        try:
            claim, err = (await extract(case["message"])) or {}, None
        except Exception as e:
            claim, err = {}, e
        return claim, err, *tokens_split(), served_model()

    started = time.perf_counter()
    claim, err, prompt, completion, served = asyncio.run(drive())
    got_order = "" if err else _norm(claim.get("order_id"))
    got_claim = "" if err else (claim.get("claim_type") or "")
    order_ok, claim_ok = got_order == truth_order, got_claim == truth_claim
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
        "hallucinated": (got_order not in ("none", "")) if truth_order == "none" else None,
        "latency_s": round(time.perf_counter() - started, 2),
        "error_kind": provenance.error_kind(err) if err else None,
        **({"error": f"{type(err).__name__}: {err}"} if err else {}),
        **provenance.usage_fields(config.EXTRACTOR_MODEL, prompt, completion, served),
    }


# aggregate; every rate carries the count it was computed from
def score(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    both = sum(r["both_ok"] for r in rows)
    asked = [r for r in rows if r["hallucinated"] is not None]
    by_difficulty = {}
    for level in sorted({r["difficulty"] for r in rows}):
        rs = [r for r in rows if r["difficulty"] == level]
        by_difficulty[level] = {"n": len(rs), "both": round(sum(r["both_ok"] for r in rs) / len(rs), 4)}
    confusion = {t: dict(Counter(r["got_claim"] or "(none)" for r in rows if r["truth_claim"] == t))
                 for t in sorted({r["truth_claim"] for r in rows})}
    return {
        "n": n,
        "order_acc": round(sum(r["order_ok"] for r in rows) / n, 4),
        "claim_acc": round(sum(r["claim_ok"] for r in rows) / n, 4),
        "both_acc": round(both / n, 4),
        "both_ci95": tuple(round(x, 4) for x in wilson(both, n)),
        "both_se": round(standard_error(both / n, n), 4),
        "hallucination_rate": round(sum(r["hallucinated"] for r in asked) / len(asked), 4) if asked else None,
        "abstention_n": len(asked),
        "by_difficulty": by_difficulty,
        "claim_confusion": confusion,
        "errors": sum(1 for r in rows if r.get("error")),
        "p50_latency_s": round(sorted(r["latency_s"] for r in rows)[n // 2], 2),
        "usd_total": round(sum(r.get("usd") or 0.0 for r in rows), 4),
    }


# paired comparison with the pinned run: fails on a significant both-right drop or any hallucination rise
def compare(rows: list[dict], baseline: dict) -> dict:
    was = {r["case_id"]: r for r in baseline["rows"]}
    paired = [(was[r["case_id"]], r) for r in rows if r["case_id"] in was]
    b = sum(1 for old, new in paired if old["both_ok"] and not new["both_ok"])
    c = sum(1 for old, new in paired if not old["both_ok"] and new["both_ok"])
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
        "failed": bool((significant and b > c) or hallucination_delta > 0),
    }


# CLI: grade, write rows and card, optionally pin or gate against a same-model baseline
def main() -> None:
    ap = argparse.ArgumentParser(description="Grade the extractor on its own, against known truth.")
    ap.add_argument("--split", choices=["test", "all"], default="test")
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (a smoke run)")
    ap.add_argument("--out", default="extractor.jsonl")
    ap.add_argument("--save-baseline", nargs="?", const=str(BASELINE.with_name("extractor_ref.json")), default="",
                    help="pin this run (bare flag writes extractor_ref.json, never the historical extractor.json)")
    ap.add_argument("--baseline", nargs="?", const=str(BASELINE), default="",
                    help="same-model pinned run to gate against; exit 1 on regression")
    args = ap.parse_args()

    cases = load_cases(args.split)[: args.limit or None]
    run = provenance.manifest("extractor")
    print(f"run_id={run['run_id']}  extractor={config.EXTRACTOR_MODEL}  {len(cases)} cases ({args.split})")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = [{**r, **run} for r in tqdm(pool.map(_one, cases), total=len(cases), unit="case", desc="extract")]

    out = provenance.RESULTS / args.out
    provenance.write_jsonl(out, rows)
    card = score(rows)
    provenance.write_json(out.with_name(out.stem + "_card.json"), {"manifest": run, "card": card})
    print(f"\nextractor scorecard over {card['n']} cases -> {out}\n")
    for key, value in card.items():
        print(f"  {key:<20}  {value}")

    if args.save_baseline:
        provenance.write_json(Path(args.save_baseline), {"model": config.EXTRACTOR_MODEL, "split": args.split,
                                                         "manifest": run, "score": card, "rows": rows})
        print(f"\n  baseline pinned -> {args.save_baseline}")
    if args.baseline:
        if not Path(args.baseline).exists():
            raise SystemExit(f"no baseline at {args.baseline}. Pin one with --save-baseline first.")
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        if base.get("model") != config.EXTRACTOR_MODEL:
            raise SystemExit(f"baseline model {base.get('model')} != EXTRACTOR_MODEL {config.EXTRACTOR_MODEL}")
        verdict = compare(rows, base)
        print(f"\n  vs baseline: {verdict}")
        if verdict["failed"]:
            raise SystemExit("\n  REGRESSION: the extractor got worse or less safe. Build fails.")


if __name__ == "__main__":
    main()
