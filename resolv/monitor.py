"""
# resolv/monitor.py

**What:** online eval of the live system: one telemetry row per real request with the label-free signals (policy
replay with the recorded caller, the independent invariants, groundedness, lookup-before-refund, latency,
provider-aware cost), a live scorecard, and a webhook alert when anything unauthorized happens.
**Why:** production has no answer key, so only signals that need no label can be measured live; resolve rate and
pass^k stay offline. The replay uses each record's own `caller_id`, and an ok refund with no caller is itself
flagged. A refund a human approved (`approved_by_human` with an `approver_id`) is replayed with that approval, so
it is not counted as unauthorized. This module must stay light: the API imports it, so it never imports the offline runner.
**Used by:** resolv/api/main.py and app.py (`record`), evals/drift.py, `python -m resolv.monitor`.
"""
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from math import ceil

from resolv import config, invariants
from resolv.harness.guardrails import reply_grounded as _reply_grounded
from resolv.harness.policy import action_hash, check_refund

TELEMETRY = Path(os.environ.get("RESOLV_TELEMETRY",
                                Path(__file__).resolve().parent.parent / "data" / "telemetry" / "live.jsonl"))


# nearest-rank percentile; None on an empty list (no data is not 0 seconds)
def percentile(values: list[float], p: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    i = max(0, min(len(ordered) - 1, ceil(p / 100 * len(ordered)) - 1))
    return round(ordered[i], 2)


# what the system did, read from the trail: a successful refund wins, then escalation, else denial
def _action(trail: list[dict]) -> str:
    if any(r["tool"] == "issue_refund" and r["ok"] for r in trail):
        return "refund"
    if any(r["tool"] == "escalate_to_human" for r in trail):
        return "escalate"
    return "deny"


# did the agent read the order before trying to refund it? None if it never tried
def _looked_up_first(trail: list[dict]) -> bool | None:
    for i, r in enumerate(trail):
        if r["tool"] == "issue_refund":
            return any(p["tool"] == "lookup_order" and p["order_id"] == r["order_id"] for p in trail[:i])
    return None


# the approval a human-approved refund record stands for, bound to its own call, so rule 7 can pass on replay
def _approval_of(r: dict, caller_id: str | None) -> dict | None:
    if r.get("rule_id") != "approved_by_human" or not r.get("approver_id"):
        return None
    digest = action_hash(r["order_id"], r["args"]["claim_type"], r["args"]["amount_usd"], caller_id)[1]
    # expiry was checked when the refund ran; the replay judges the rules, not the clock
    return {"status": "approved", "action_hash": digest, "expires_at": "9999", "decided_by": r["approver_id"]}


# replay one ok refund record against policy with the records before it; rules 0-6 and approver != caller still apply
def replay_refund(r: dict, history: list[dict], caller_id: str | None):
    return check_refund(r["order_id"], r["args"]["claim_type"], r["args"]["amount_usd"], history, caller_id,
                        _approval_of(r, caller_id))


# replay each ok refund against policy with the recorded caller; an ok refund with no caller is flagged
def _unauthorized(trail: list[dict]) -> bool:
    for i, r in enumerate(trail):
        if r["tool"] != "issue_refund" or not r["ok"]:
            continue
        if r.get("caller_id") is None:
            return True
        if replay_refund(r, trail[:i], r["caller_id"]).action != "allow":
            return True
    return False


# POST an alert to RESOLV_ALERT_WEBHOOK (Slack/Discord incoming webhook); no-op without it, never raises
def alert(row: dict) -> None:
    url = os.environ.get("RESOLV_ALERT_WEBHOOK")
    if not url:
        return
    text = (f"Resolv ALERT case={row['case_id']} unauthorized={row['unauthorized']} "
            f"invariant_violations={row['invariant_violations']}")
    body = json.dumps({"text": text, "content": text}).encode()
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3).close()
    except Exception:
        pass  # alerting must never break the request it reports on


# compute the label-free metrics for one live request, append the telemetry row, alert if unsafe
def record(case_id: str, message: str, result: dict, latency_s: float, prompt_tokens: int,
           completion_tokens: int, model: str | None = None, served: str | None = None) -> dict:
    trail = result["trail"]
    refunds = [r for r in trail if r["tool"] == "issue_refund" and r["ok"]]
    paid = round(sum(r["args"]["amount_usd"] for r in refunds), 2)
    model = model or config.AGENT_MODEL
    usd = config.price_usd(model, prompt_tokens, completion_tokens)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "case_id": case_id,
        "action": _action(trail),
        "paid": paid,
        "unauthorized": _unauthorized(trail),
        "invariant_violations": invariants.check(trail),
        "reply_grounded": _reply_grounded(result.get("reply", ""), paid, bool(refunds)),
        "looked_up_first": _looked_up_first(trail),
        "steps": result.get("steps", 0),
        "latency_s": round(latency_s, 2),
        "tokens": prompt_tokens + completion_tokens,
        "usd": None if usd is None else round(usd, 6),
        "model": model,
        "served_model": served,
        "rules_hit": [r["rule_id"] for r in trail],
    }
    TELEMETRY.parent.mkdir(parents=True, exist_ok=True)
    with open(TELEMETRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    if row["unauthorized"] or row["invariant_violations"]:
        alert(row)
    return row


# every telemetry row in a file, [] if it does not exist
def read(path: Path = None) -> list[dict]:
    path = Path(path or TELEMETRY)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# the live board: the same label-free signals aggregated over the window
def scorecard(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"requests": 0}

    # share of True over rows where the field is not None
    def rate(field: str) -> float | None:
        asked = [r[field] for r in rows if r.get(field) is not None]
        return round(sum(bool(x) for x in asked) / len(asked), 4) if asked else None

    actions: dict[str, int] = {}
    for r in rows:
        actions[r["action"]] = actions.get(r["action"], 0) + 1
    total_usd = sum(r.get("usd") or 0.0 for r in rows)
    return {
        "requests": n,
        "unauthorized_rate": round(sum(bool(r.get("unauthorized")) for r in rows) / n, 4),
        "invariant_violation_rate": rate("invariant_violations"),
        "reply_grounded_rate": rate("reply_grounded"),
        "lookup_before_refund_rate": rate("looked_up_first"),
        "escalation_rate": round(actions.get("escalate", 0) / n, 4),
        "actions": actions,
        "p50_latency_s": percentile([r["latency_s"] for r in rows if r.get("latency_s")], 50),
        "p95_latency_s": percentile([r["latency_s"] for r in rows if r.get("latency_s")], 95),
        "usd_total": round(total_usd, 4),
        "usd_per_request": round(total_usd / n, 6),
    }


# CLI: print the live scorecard; exit non-zero on any live unauthorized action
def main() -> None:
    ap = argparse.ArgumentParser(description="Live scorecard over the online telemetry log.")
    ap.add_argument("--file", default=str(TELEMETRY))
    args = ap.parse_args()
    card = scorecard(read(Path(args.file)))
    print(f"online scorecard over {card['requests']} live requests\n")
    for key, value in card.items():
        print(f"  {key:<26}  {value}")
    if card.get("unauthorized_rate") or card.get("invariant_violation_rate"):
        raise SystemExit("\n  !! LIVE UNAUTHORIZED ACTION OR INVARIANT VIOLATION.")


if __name__ == "__main__":
    main()
