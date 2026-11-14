"""
# scripts/regrade_cards.py

**What:** re-scores the committed reference-sweep and harness-ablation rows with today's grader and rewrites their
cards, recording every number that moved under a `regrade` key in the card.
**Why:** error analysis found two grader bugs: a refund the policy rejected could still count as resolved, and the
groundedness check missed "initiated" / "submitted" as words that claim a refund happened. Rows are never edited by
hand; the fix is applied in memory and only the card is rewritten, with the existing `evals.metrics` code.
`unauthorized` is kept as recorded: the policy a run was graded against is the one it ran under.
**Used by:** `python scripts/regrade_cards.py`, once per grader fix; docs/eval/ERROR_ANALYSIS.md cites its output.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals import provenance  # noqa: E402
from evals.metrics import cluster_bootstrap, design_effect, pass_k_stat, scorecard  # noqa: E402
from resolv.harness.guardrails import reply_grounded  # noqa: E402

RESULTS = provenance.RESULTS
REASON = ("grader fixes from docs/eval/ERROR_ANALYSIS.md: a run with an unauthorized refund or an invariant "
          "violation is not resolved; 'initiated' and 'submitted' count as claiming a refund happened")


# one row re-graded with today's grader, from the facts the row already holds
def regrade(row: dict) -> dict:
    refunded = any(r["tool"] == "issue_refund" and r["ok"] for r in row.get("trail") or [])
    resolved = row["resolved"] and not row["unauthorized"] and not row.get("invariant_violations")
    return {**row, "resolved": resolved, "recovered": None if row["recovered"] is None else resolved,
            "reply_grounded": reply_grounded(row.get("reply") or "", float(row["paid"] or 0), refunded)}


# the keys whose value moved, as {key: [before, after]}
def changes(before: dict, after: dict) -> dict:
    after = json.loads(json.dumps(after))  # tuples become lists, as they are in the stored card
    return {k: [before.get(k), v] for k, v in after.items() if k != "regrade" and before.get(k) != v}


# the new card with its regrade record; an already-regraded card with nothing new to change is kept as it is
def merge(old: dict, card: dict) -> dict:
    moved = changes(old, card)
    return {**old, **card, "regrade": {"reason": REASON, "changed": moved}} if moved else old


# the partial reference card: scorecard at k=3 plus pass^1..5, the design effect and spend over every row
def ref_card(rows: list[dict], scored: set[str]) -> dict:
    runs = [r for r in rows if r["task_id"] in scored and not r.get("error")]
    card = scorecard(runs, k=3)
    for k in range(1, 6):
        groups = {}
        for r in runs:
            groups.setdefault(r["task_id"], []).append(r)
        card[f"pass^{k}"] = round(sum(pass_k_stat(k)(g)[0] for g in groups.values()) / len(groups), 4)
        card[f"pass^{k}_ci95_clustered"] = cluster_bootstrap(runs, pass_k_stat(k))
    card["design_effect_resolved"] = design_effect(runs)
    card["usd_total_all_rows"] = round(sum(r.get("usd") or 0.0 for r in rows), 4)
    return card


# rewrite both cards and print what moved
def main() -> None:
    path = RESULTS / "ref_gpt_oss_120b_partial_card.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = [regrade(r) for r in provenance.read_rows(RESULTS / "ref_gpt_oss_120b.jsonl")]
    card = ref_card(rows, set(doc["coverage"]["scored_task_ids"]))
    old = doc["card"]
    doc["card"] = merge(old, card)
    provenance.write_json(path, doc)
    print(path.name, doc["card"].get("regrade"))

    path = RESULTS / "ablation_gpt_oss_120b_card.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = [regrade(r) for r in provenance.read_rows(RESULTS / "ablation_gpt_oss_120b.jsonl")]
    for arm, old in doc["cards"].items():
        card = scorecard([r for r in rows if r["arm"] == arm], k=1)
        doc["cards"][arm] = merge(old, card)
        print(path.name, arm, doc["cards"][arm].get("regrade"))
    provenance.write_json(path, doc)


if __name__ == "__main__":
    main()
