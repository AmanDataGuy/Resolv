"""
Application-quality eval: TONE.

    python -m eval.quality                       # grade eval/runner.py's last sweep
    python -m eval.quality --in ablation.jsonl    # grade any JSONL with a "reply" field

Every other metric in this project grades the TRAIL -- did the right money move, in the
right order (eval/metrics.py, eval/runner.py). None of them grade the PROSE the customer
actually reads. A technically correct denial delivered curtly is a real failure nothing
else here measures. This is the one metric that judges HOW it was said, reference-free
on purpose -- it must never be satisfiable by simply being more correct, or it would just
duplicate reply_grounded_rate.

NO NEW AGENT CALLS. It reads the `reply` field straight out of an existing sweep's JSONL,
so the resolution cost that produced those replies is sunk already -- only the judge
call is new spend. Same GEval-style rubric-as-prompt idea as the reference
eval_application.py's Style metric, but as a plain complete() call (this project's
existing litellm/Groq/Gemini judge) rather than adding a DeepEval + OpenAI dependency.
"""
import argparse
import json
from pathlib import Path

from agents.runner_utils import complete
from config import MODEL

OUT_DIR = Path(__file__).parent.parent / "data" / "eval"
THRESHOLD = 7  # out of 10

RUBRIC = """Rate the TONE of this customer-support reply, 0-10. Judge tone only -- not
whether the underlying decision (refund/deny/escalate) was correct.

  9-10: professional and warm. Explains a denial without sounding defensive.
  5-8:  polite but flat, or slightly terse.
  0-4:  curt, robotic, defensive, or dismissive.

Reply: {reply}

Answer with ONLY a single integer 0-10, nothing else."""


def judge(reply: str) -> int:
    resp = complete(
        model=MODEL,
        messages=[{"role": "user", "content": RUBRIC.format(reply=reply)}],
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or "0").strip()
    try:
        return max(0, min(10, int(text.split()[0])))
    except ValueError:
        return 0  # a judge that won't answer with a number is a failed grade, not a crash


def score(rows: list[dict]) -> dict:
    graded = [
        {"case_id": r.get("task_id") or r.get("case_id"), "tone": judge(r["reply"])}
        for r in rows
        if r.get("reply")
    ]
    n = len(graded)
    avg = sum(g["tone"] for g in graded) / n if n else 0.0
    below = [g for g in graded if g["tone"] < THRESHOLD]
    worst = sorted(graded, key=lambda g: g["tone"])[:3]
    return {"n": n, "avg_tone": round(avg, 2), "below_threshold": len(below), "worst": worst}


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade the tone of agent replies from a sweep.")
    ap.add_argument("--in", dest="infile", default="runs.jsonl")
    args = ap.parse_args()

    path = OUT_DIR / args.infile
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    card = score(rows)
    print(f"TONE  (judge={MODEL}, {card['n']} replies from {path.name})")
    print("=" * 60)
    print(f"avg tone               : {card['avg_tone']} / 10")
    print(f"below threshold ({THRESHOLD}) : {card['below_threshold']} / {card['n']}")
    if card["worst"]:
        print("worst 3:")
        for w in card["worst"]:
            print(f"  {w['case_id']:<30} tone={w['tone']}")


if __name__ == "__main__":
    main()
