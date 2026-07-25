"""Online eval — grading the LIVE system, one real request at a time.

    python -m eval.monitor              # live scorecard over data/telemetry/live.jsonl

WHY THIS IS DIFFERENT FROM THE OFFLINE SWEEP, AND WHY THAT DIFFERENCE IS THE WHOLE POINT.
eval/runner.py grades SAVED cases against a known answer key — it can say "the correct refund was
$47.09, did the agent pay exactly that?" Production has no answer key: a real customer's message
does not arrive with the right outcome stapled to it. So online eval can only measure the signals
that need **no label**:

  - is anything leaking?      replay check_refund() over the live trail — the same safety net as
                              offline, and it needs no ground truth, only the policy engine.
  - did the reply match       groundedness: the prose vs the audit trail. No label needed — the
    what actually happened?   trail IS the truth.
  - did it read before        tool discipline: lookup_order before issue_refund, from trail order.
    it refunded?
  - how fast, how much?       latency + cost, measured directly.

It CANNOT measure resolve_rate or pass^k live, because those need the answer this system does not
have in production. That gap is exactly what separates offline (labelled, pre-deploy) from online
(unlabelled, live), made concrete rather than asserted.

record() appends one telemetry row per request — append-only, same discipline as harness/audit.py.
main() aggregates the file into a live scorecard. Langfuse (eval/observability.py) is a dashboard
layer over the same events; THIS file is the always-on, dependency-free foundation.
"""
import argparse
import json
import time
from pathlib import Path

from eval.metrics import _percentile
from eval.runner import USD_PER_MTOK_IN, USD_PER_MTOK_OUT, _looked_up_first, _reply_grounded
from harness.policy import check_refund

TELEMETRY = Path(__file__).parent.parent / "data" / "telemetry" / "live.jsonl"


def _action(trail: list[dict]) -> str:
    """What the system did, read from the trail: a successful refund, an escalation, or a denial."""
    if any(r["tool"] == "issue_refund" and r["ok"] for r in trail):
        return "refund"
    if any(r["tool"] == "escalate_to_human" for r in trail):
        return "escalate"
    return "deny"


def _unauthorized(trail: list[dict]) -> bool:
    """Replay each successful refund against real policy using only the records that preceded it.

    The one safety metric that survives the absence of a label: it asks the policy engine directly,
    not the answer key. Mirrors the replay in eval/runner.py::_grade — duplicated deliberately (six
    stable lines) rather than coupling the live path to the offline grader's task-shaped signature.
    """
    for i, r in enumerate(trail):
        if r["tool"] != "issue_refund" or not r["ok"]:
            continue
        verdict = check_refund(r["order_id"], r["args"]["claim_type"], r["args"]["amount_usd"], trail[:i])
        if verdict.action != "allow":
            return True
    return False


def record(case_id: str, message: str, result: dict,
           latency_s: float, prompt_tokens: int, completion_tokens: int) -> dict:
    """Compute the label-free metrics for one live request and append a telemetry row.

    Returned as well as written so the caller (api/main.py) can hand the same dict to the Langfuse
    layer without recomputing anything.
    """
    trail = result["trail"]
    refunds = [r for r in trail if r["tool"] == "issue_refund" and r["ok"]]
    paid = round(sum(r["args"]["amount_usd"] for r in refunds), 2)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "case_id": case_id,
        "action": _action(trail),
        "paid": paid,
        "unauthorized": _unauthorized(trail),
        "reply_grounded": _reply_grounded(result.get("reply", ""), paid, bool(refunds)),
        "looked_up_first": _looked_up_first(trail),
        "steps": result.get("steps", 0),
        "latency_s": round(latency_s, 2),
        "tokens": prompt_tokens + completion_tokens,
        "usd": round(prompt_tokens / 1e6 * USD_PER_MTOK_IN + completion_tokens / 1e6 * USD_PER_MTOK_OUT, 6),
        "rules_hit": [r["rule_id"] for r in trail],
    }
    TELEMETRY.parent.mkdir(parents=True, exist_ok=True)
    with open(TELEMETRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def read(path: Path = TELEMETRY) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scorecard(rows: list[dict]) -> dict:
    """The live board. Same label-free signals as each row, aggregated across the window."""
    n = len(rows)
    if not n:
        return {"requests": 0}

    def rate(field: str) -> float | None:
        asked = [r[field] for r in rows if r.get(field) is not None]
        return round(sum(asked) / len(asked), 4) if asked else None

    actions: dict[str, int] = {}
    for r in rows:
        actions[r["action"]] = actions.get(r["action"], 0) + 1

    total_usd = sum(r.get("usd") or 0.0 for r in rows)
    return {
        "requests": n,
        # The one that must stay 0 in production, exactly as in the sweep.
        "unauthorized_rate": round(sum(bool(r.get("unauthorized")) for r in rows) / n, 4),
        "reply_grounded_rate": rate("reply_grounded"),
        "lookup_before_refund_rate": rate("looked_up_first"),
        "escalation_rate": round(actions.get("escalate", 0) / n, 4),
        "actions": actions,
        "p50_latency_s": _percentile([r["latency_s"] for r in rows if r.get("latency_s")], 50),
        "p95_latency_s": _percentile([r["latency_s"] for r in rows if r.get("latency_s")], 95),
        "usd_total": round(total_usd, 4),
        "usd_per_request": round(total_usd / n, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Live scorecard over the online telemetry log.")
    ap.add_argument("--file", default=str(TELEMETRY), help="telemetry JSONL to aggregate")
    args = ap.parse_args()

    rows = read(Path(args.file))
    card = scorecard(rows)
    print(f"online scorecard over {card['requests']} live requests\n")
    for key, value in card.items():
        if key != "actions":
            print(f"  {key:<26}  {value}")
    print(f"  {'actions':<26}  {card.get('actions', {})}")

    if card.get("unauthorized_rate"):
        raise SystemExit("\n  !! LIVE UNAUTHORIZED ACTION — a production refund violated policy.")


if __name__ == "__main__":
    main()
