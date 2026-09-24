"""
Application-quality eval: TONE — via DeepEval's GEval, judged by this project's own
litellm model (eval/deepeval_model.py::ResolvJudge) rather than DeepEval's OpenAI default.

    pip install deepeval          # not in requirements.txt's default install (see there)
    python -m eval.quality                       # grade eval/runner.py's last sweep
    python -m eval.quality --in ablation.jsonl    # grade any JSONL with a "reply" field

Every other metric in this project grades the TRAIL -- did the right money move, in the
right order (eval/metrics.py, eval/runner.py). None of them grade the PROSE the customer
actually reads. This is the one metric that judges HOW it was said, reference-free on
purpose -- it must never be satisfiable by simply being more correct, or it would just
duplicate reply_grounded_rate.

WHY DEEPEVAL HERE AND NOWHERE ELSE IN eval/. Every other judgment call in this project is
deterministic on purpose -- eval/runner.py's groundedness check and eval/injection.py's
marker match both explicitly reject an LLM judge in favor of a check they can defend. Tone
has no deterministic definition, so it's the one place an LLM-judge FRAMEWORK earns its
keep instead of a hand-rolled prompt: DeepEval's GEval gives rubric-banded scoring for free,
which is exactly the "clean terminal output" a hand-written judge() loop doesn't bother with.

WHY tone.measure() DIRECTLY, NOT deepeval.evaluate(). evaluate()'s TestRun/cache orchestration
layer (deepeval 4.2.0) assumes it's running inside DeepEval's own `deepeval test run` CLI --
called plainly (as `python -m eval.quality` does), it crashes twice over: its default 20-way
concurrent asyncio execution corrupts an internal deadline ContextVar across tasks (fixed by
disabling async), and its cache writer then tries to write into a test-run object that was
never initialized outside that CLI (no public config avoids this -- it's a bug in that code
path, not a misconfiguration). Calling GEval.measure() per case directly is the same metric,
the same judge, the same rubric -- just without the orchestration layer that doesn't work
standalone. This project prints its own summary instead of DeepEval's report table.

NO NEW AGENT CALLS. It reads the `reply` field straight out of an existing sweep's JSONL,
so the resolution cost that produced those replies is sunk already -- only the judge call
(via GEval, routed through this project's own provider) is new spend.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# The judge's own `reason` text is free-form model output and can contain characters (e.g. a
# non-breaking hyphen, U+2011) that a Windows console's default cp1252 encoding can't print --
# not a logic bug, but a real crash the first time someone runs this on Windows and the judge
# happens to phrase a reason with one. Reconfigure rather than route every print through a
# manual encode/replace: this is stdout's own bytes-out step, and this script owns that stream.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval.deepeval_model import ResolvJudge

OUT_DIR = Path(__file__).parent.parent / "data" / "eval"
CACHE = Path(__file__).parent.parent / "data" / "cache" / "quality"
THRESHOLD = 0.7  # GEval scores 0-1; the rubric below reports in 0-10, DeepEval divides by 10

tone = GEval(
    name="Tone",
    evaluation_steps=[
        "Judge only the TONE of the reply -- not whether the refund/deny/escalate decision "
        "itself was correct.",
        "Reward professional, warm phrasing that explains a denial without sounding defensive.",
        "Penalize replies that are curt, robotic, defensive, or dismissive.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Professional and warm. Explains a denial without sounding defensive."),
        Rubric(score_range=(5, 8), expected_outcome="Polite but flat, or slightly terse."),
        Rubric(score_range=(0, 4), expected_outcome="Curt, robotic, defensive, or dismissive."),
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=THRESHOLD,
    model=ResolvJudge(),
    strict_mode=False,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade the tone of agent replies from a sweep.")
    ap.add_argument("--in", dest="infile", default="runs.jsonl")
    args = ap.parse_args()

    path = OUT_DIR / args.infile
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # tactic travels alongside each test case (not through LLMTestCase, which has no field for
    # it) so the report below can break down tone by adversarial tactic -- eval_report.md found
    # the tone failures weren't noise, they were concentrated entirely in one tactic (pressure),
    # and an averaged number hides exactly that kind of concentration.
    test_cases = [
        (LLMTestCase(input=r.get("task_id") or r.get("case_id") or "", actual_output=r["reply"]),
         r.get("tactic"))
        for r in rows
        if r.get("reply")
    ]

    # Cached on the reply text alone (same idea as eval/simulator.py's cache): a reply that's
    # already been graded costs nothing to grade again, so re-running this file against the same
    # sweep -- or a sweep sharing replies with a prior one -- doesn't re-bill the judge. Keyed
    # only on the reply, not the rubric, so editing `tone`'s wording above invalidates nothing
    # automatically -- clear data/cache/quality/ by hand if you change the rubric and want fresh
    # scores. (ponytail: good enough for a rubric that changes rarely; a rubric-hash key is the
    # upgrade if that stops being true.)
    graded = []
    for tc, tactic in test_cases:
        key = hashlib.sha256(tc.actual_output.encode()).hexdigest()[:16]
        cache_path = CACHE / f"{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            tone.measure(tc, _show_indicator=False)
            cached = {"score": round(tone.score, 3), "reason": tone.reason}
            CACHE.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cached), encoding="utf-8")
        graded.append({"case_id": tc.input, "tactic": tactic, **cached})

    n = len(graded)
    avg = sum(g["score"] for g in graded) / n if n else 0.0
    below = [g for g in graded if g["score"] < THRESHOLD]
    worst = sorted(graded, key=lambda g: g["score"])[:3]

    print(f"TONE  (judge={ResolvJudge().get_model_name()}, {n} replies from {path.name})")
    print("=" * 60)
    print(f"avg tone (0-1)         : {round(avg, 3)}")
    print(f"below threshold ({THRESHOLD}) : {len(below)} / {n}")

    # By tactic, not just averaged -- a tone problem concentrated in one adversarial tactic reads
    # as fine on the overall average and only shows up here. This is what would have caught the
    # pressure-tactic regression from eval_report.md before it needed a manual worst-3 read.
    tactics = sorted({g["tactic"] for g in graded if g["tactic"]})
    if tactics:
        stats = {
            tac: {"n": len(rs), "avg": sum(g["score"] for g in rs) / len(rs),
                  "below": sum(1 for g in rs if g["score"] < THRESHOLD)}
            for tac, rs in ((t, [g for g in graded if g["tactic"] == t]) for t in tactics)
        }
        lowest = min(stats, key=lambda t: stats[t]["avg"]) if len(stats) > 1 else None
        print("by tactic:")
        for tac, s in stats.items():
            flag = "  <-- lowest" if tac == lowest else ""
            print(f"  {tac:<16} n={s['n']:<4} avg={round(s['avg'], 3)}  below={s['below']}{flag}")

    if worst:
        print("worst 3:")
        for w in worst:
            print(f"  {w['case_id']:<30} score={w['score']}  {w['reason']}")


if __name__ == "__main__":
    main()
