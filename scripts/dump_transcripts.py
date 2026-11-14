"""
# scripts/dump_transcripts.py

**What:** writes one markdown file per sampled eval run (task, tactic, expected outcome, compact trail, transcript,
invariant violations) so a person can read and code the failures.
**Why:** error analysis starts from reading runs, not from aggregate numbers. The sample is fixed (seed 0): every
failure in the reference sweep, 30 random passes, every harness-off unauthorized refund from the ablation, then a
random top-up from the historical ksweep failures until there are at least 100 files.
**Used by:** `python scripts/dump_transcripts.py`; coding lives in evals/error_analysis/coding.csv and
docs/eval/ERROR_ANALYSIS.md. The output folder is gitignored because it only repeats committed rows.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals import provenance  # noqa: E402

RESULTS = provenance.RESULTS
OUT = Path(__file__).resolve().parents[1] / "evals" / "error_analysis" / "transcripts"
TARGET = 100


# the stable id a coded row points at: source, task and repeat
def run_key(source: str, row: dict) -> str:
    return f"{source}-{row['task_id']}-r{row['repeat']}"


# one run rendered as markdown
def render(source: str, row: dict) -> str:
    lines = [
        f"# {run_key(source, row)}", "",
        f"- source: `{source}` (sweep run `{row.get('run_id', 'historical, no manifest')}`)",
        f"- task: `{row['task_id']}`, tactic `{row['tactic']}`, difficulty `{row['difficulty']}`",
        f"- expected: **{row['expected']}** ${row['expected_amount']}; paid ${row['paid']}; "
        f"escalated {row['escalated']}; resolved **{row['resolved']}**",
        f"- unauthorized {row['unauthorized']}; over_blocked {row['over_blocked']}; "
        f"reply_grounded {row['reply_grounded']}; error {row.get('error_kind') or (row.get('error') or '')[:80] or None}",
        f"- invariant_violations: {row.get('invariant_violations', 'not recorded')}",
        f"- rules_hit: {row['rules_hit']}", "", "## Trail", "",
    ]
    for t in row.get("trail") or []:
        lines.append(f"- `{t['tool']}` {t.get('order_id')} {json.dumps(t.get('args'))} -> "
                     f"{t.get('action')} / {t.get('rule_id')} ok={t.get('ok')}")
    if not row.get("trail"):
        lines.append("_not recorded in this file_")
    lines += ["", "## Transcript", ""]
    for turn in row.get("transcript") or []:
        lines.append(f"**{turn['role']}:** {turn['content']}\n")
    lines += ["## Final reply", "", row.get("reply") or "_none_", ""]
    return "\n".join(lines)


# the fixed sample as (source, row) pairs
def sample() -> list[tuple[str, dict]]:
    rng = random.Random(0)
    ref = provenance.read_rows(RESULTS / "ref_gpt_oss_120b.jsonl")
    passes = [r for r in ref if r["resolved"]]
    picked = [("ref", r) for r in ref if not r["resolved"]]  # every failure, over-blocks included
    picked += [("ref", r) for r in rng.sample(passes, 30)]
    ablation = provenance.read_rows(RESULTS / "ablation_gpt_oss_120b.jsonl")
    picked += [("ablation_off", r) for r in ablation if r["arm"] == "off" and r["unauthorized"]]
    history = [r for r in provenance.read_rows(RESULTS / "historical" / "ksweep.jsonl") if not r["resolved"]]
    picked += [("historical_ksweep", r) for r in rng.sample(history, max(0, TARGET - len(picked)))]
    return picked


# write the sample and print one line per file
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for source, row in sample():
        path = OUT / f"{run_key(source, row)}.md"
        path.write_text(render(source, row), encoding="utf-8")
        print(path.name)


if __name__ == "__main__":
    main()
