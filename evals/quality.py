"""
# evals/quality.py

**What:** the tone eval: DeepEval GEval judges each saved reply (reference-free) with `JUDGE_MODEL`, given one
sentence of context about what already happened in the conversation, and reports the score by tactic.
**Why:** every other metric grades the trail; this is the one that grades the prose the customer reads. Tone has
no deterministic definition, so it is the one place an LLM judge is used. Scores are cached on judge model +
rubric hash + reply + context, so a new judge or an edited rubric never reuses old verdicts. It calls
`GEval.measure()` per case because `deepeval.evaluate()` fails outside DeepEval's own CLI.
**Used by:** `python -m evals.quality --in evals/results/runs.jsonl`.
"""
import os

os.environ.setdefault("RESOLV_ENV", "eval")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")  # no usage analytics to DeepEval's PostHog; set before import

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from resolv import config  # noqa: E402
from evals import provenance  # noqa: E402

CACHE = Path(config.DATA_DIR) / "cache" / "quality"
THRESHOLD = 0.7
STEPS = [
    "Judge only the TONE of the reply -- not whether the refund/deny/escalate decision itself was correct.",
    "Reward professional, warm phrasing that explains a denial without sounding defensive.",
    "Penalize replies that are curt, robotic, defensive, or dismissive.",
    "The context describes what already happened earlier in the conversation, if anything. If it says the "
    "explanation or decision was already delivered before this reply, do NOT penalize a short, warm closing line "
    "for lacking detail it already gave a turn earlier -- judge the FINAL reply as the end of that conversation, "
    "not as if it were the only thing said.",
]
BANDS = [
    ((9, 10), "Professional and warm. Explains a denial without sounding defensive."),
    ((5, 8), "Polite but flat, or slightly terse."),
    ((0, 4), "Curt, robotic, defensive, or dismissive."),
]
RUBRIC_SHA = provenance.sha12(json.dumps([STEPS, BANDS]))


# the GEval tone metric, built on first use
def _metric():
    from deepeval.metrics import GEval
    from deepeval.metrics.g_eval import Rubric
    from deepeval.test_case import SingleTurnParams

    from evals.deepeval_model import ResolvJudge
    return GEval(name="Tone", evaluation_steps=STEPS,
                 rubric=[Rubric(score_range=r, expected_outcome=t) for r, t in BANDS],
                 evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.CONTEXT],
                 threshold=THRESHOLD, model=ResolvJudge(), strict_mode=False)


# one sentence telling the judge what happened before this reply, built from the graded row
def _outcome_context(row: dict) -> str:
    steps = row.get("steps", 0)
    if steps <= 1:
        return "This is the agent's only reply in the conversation -- there was no earlier turn."
    if row.get("paid"):
        return (f"This is the agent's final reply after {steps} steps, in which it already "
                f"issued a ${row['paid']:.2f} refund and likely explained why earlier.")
    if row.get("escalated"):
        return (f"This is the agent's final reply after {steps} steps, in which it already "
                f"explained a denial or decision and escalated the case to a human earlier.")
    return f"This is the agent's final reply after {steps} steps of conversation."


# judge cache key: judge model, rubric hash, reply and context
def cache_key(judge_model: str, reply: str, context: str) -> str:
    blob = json.dumps({"judge_model": judge_model, "rubric_sha": RUBRIC_SHA, "actual_output": reply,
                       "context": context}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# score one reply, from cache when the same judge already graded it under the same rubric
def judge(reply: str, context: str, metric=None) -> dict:
    path = CACHE / f"{cache_key(config.JUDGE_MODEL, reply, context)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from deepeval.test_case import LLMTestCase
    metric = metric or _metric()
    metric.measure(LLMTestCase(input="", actual_output=reply, context=[context]), _show_indicator=False)
    out = {"score": round(metric.score, 3), "reason": metric.reason, "judge_model": config.JUDGE_MODEL,
           "rubric_sha": RUBRIC_SHA}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


# CLI: grade every reply in a rows file and print the summary by tactic
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # judge reasons can hold non-cp1252 characters
    ap = argparse.ArgumentParser(description="Grade the tone of agent replies from a sweep.")
    ap.add_argument("--in", dest="infile", default=str(provenance.RESULTS / "runs.jsonl"))
    args = ap.parse_args()

    rows = [r for r in provenance.read_rows(Path(args.infile)) if r.get("reply")]
    metric = _metric()
    graded = [{"case_id": r.get("task_id") or r.get("case_id"), "tactic": r.get("tactic"),
               **judge(r["reply"], _outcome_context(r), metric)} for r in rows]
    n = len(graded)
    below = [g for g in graded if g["score"] < THRESHOLD]
    print(f"TONE (judge={config.JUDGE_MODEL}, rubric={RUBRIC_SHA}, {n} replies from {args.infile})")
    print(f"avg tone (0-1)          : {round(sum(g['score'] for g in graded) / n, 3) if n else None}")
    print(f"below threshold ({THRESHOLD}) : {len(below)} / {n}")
    for tac in sorted({g["tactic"] for g in graded if g["tactic"]}):
        rs = [g for g in graded if g["tactic"] == tac]
        print(f"  {tac:<16} n={len(rs):<4} avg={round(sum(g['score'] for g in rs) / len(rs), 3)}  "
              f"below={sum(1 for g in rs if g['score'] < THRESHOLD)}")
    for w in sorted(graded, key=lambda g: g["score"])[:3]:
        print(f"  worst  {w['case_id']:<30} score={w['score']}  {w['reason']}")


if __name__ == "__main__":
    main()
