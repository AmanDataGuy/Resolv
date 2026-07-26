"""Phase 3 — live drift detection. Is production still behaving like the pinned baseline?

    python -m eval.drift                # compare live telemetry against the offline baseline

Offline eval (`eval/runner.py`) proves the system safe on a labelled test set ONCE, before deploy.
Drift detection asks the question offline eval can't: *is live traffic still behaving like that test
set, or has the world moved underneath it?* A model can pass every offline test and then rot in
production as customer language, order mix, or an upstream model version shifts. No labels are needed
to catch this — you compare **distributions** (what the system does now) against the known-good
baseline (what it did when it passed).

TWO KINDS OF SIGNAL, because they fail in different ways:

  1. **Action mix** (refund / escalate / deny) — no "correct" direction; it should just stay close to
     baseline. Measured with **PSI (Population Stability Index)**, the standard drift statistic:
        PSI < 0.10  stable  |  0.10-0.25  moderate shift  |  > 0.25  significant shift.
     A jump from mostly-refund to mostly-escalate is exactly the kind of silent behaviour change PSI
     is built to surface.

  2. **Safety rates** (unauthorized, lookup-before-refund, groundedness) — here the acceptable
     direction IS known: unauthorized may never rise above 0, lookup/groundedness may never fall far.
     So these get one-directional threshold alarms, not PSI.

The baseline is the SAME pinned `agent.json` the offline regression gate uses — one source of truth
for "known good", offline and online. Drift reads the live log written by `eval/monitor.py`; it adds
nothing to the request path.
"""
import argparse
import json
from math import log
from pathlib import Path

from eval import monitor
from eval.runner import AGENT_BASELINE

# Below this many live requests, distributions are too noisy to judge — report, don't alarm.
MIN_REQUESTS = 30
# PSI bands (industry-standard). We alarm at the "significant" boundary.
PSI_SIGNIFICANT = 0.25
# Safety-rate tolerances: unauthorized is zero-tolerance; the others may sag this far below baseline
# before it counts as drift (matches the sweep gate's spirit — small wobble is noise, a cliff isn't).
RATE_TOLERANCE = 0.05


def _dist(labels: list[str]) -> dict[str, float]:
    """Category -> share of the whole. The empirical distribution of a categorical signal."""
    n = len(labels)
    out: dict[str, float] = {}
    for x in labels:
        out[x] = out.get(x, 0.0) + 1.0 / n
    return out


def psi(live: dict[str, float], base: dict[str, float]) -> float:
    """Population Stability Index between two categorical distributions.

    Sum over every category of (live% - base%) * ln(live% / base%). A tiny epsilon replaces a 0% so
    the log is defined — a category that appears live but never in baseline (or vice versa) is exactly
    the drift we want the number to spike on, not a division error.
    """
    eps = 1e-6
    keys = set(live) | set(base)
    total = 0.0
    for k in keys:
        lv = max(live.get(k, 0.0), eps)
        bv = max(base.get(k, 0.0), eps)
        total += (lv - bv) * log(lv / bv)
    return round(total, 4)


def _offline_action_dist(baseline: dict) -> dict[str, float]:
    """The baseline action mix, derived from the pinned sweep's per-case rows.

    agent.json stores rows with `escalated`/`paid`, not a single `action` string, so we fold them
    into the same three buckets `monitor._action` produces for live rows — apples to apples.
    """
    actions = []
    for r in baseline["rows"]:
        if r.get("escalated"):
            actions.append("escalate")
        elif (r.get("paid") or 0) > 0:
            actions.append("refund")
        else:
            actions.append("deny")
    return _dist(actions)


def report(rows: list[dict], baseline: dict) -> dict:
    """Compare live telemetry against the baseline; return the drift verdict.

    `drift=True` means at least one alarm fired: a significant action-mix PSI, any unauthorized
    action live, or a safety rate that fell more than RATE_TOLERANCE below baseline.
    """
    n = len(rows)
    if n < MIN_REQUESTS:
        return {"requests": n, "verdict": "insufficient_data",
                "note": f"need >= {MIN_REQUESTS} live requests to judge drift"}

    live = monitor.scorecard(rows)
    card = baseline["card"]
    live_actions = _dist([r["action"] for r in rows])
    action_psi = psi(live_actions, _offline_action_dist(baseline))

    alarms = []
    if action_psi > PSI_SIGNIFICANT:
        alarms.append(f"action-mix PSI {action_psi} > {PSI_SIGNIFICANT} (significant shift)")
    if live["unauthorized_rate"] > 0:
        alarms.append(f"unauthorized_rate {live['unauthorized_rate']} > 0 (zero-tolerance)")
    for key in ("lookup_before_refund_rate", "reply_grounded_rate"):
        base_val, live_val = card.get(key), live.get(key)
        if base_val is not None and live_val is not None and base_val - live_val > RATE_TOLERANCE:
            alarms.append(f"{key} fell {round(base_val - live_val, 4)} below baseline "
                          f"({live_val} vs {base_val})")

    return {
        "requests": n,
        "verdict": "DRIFT" if alarms else "stable",
        "drift": bool(alarms),
        "action_psi": action_psi,
        "live_action_mix": {k: round(v, 3) for k, v in live_actions.items()},
        "baseline_action_mix": {k: round(v, 3) for k, v in _offline_action_dist(baseline).items()},
        "unauthorized_rate": live["unauthorized_rate"],
        "lookup_before_refund_rate": live.get("lookup_before_refund_rate"),
        "reply_grounded_rate": live.get("reply_grounded_rate"),
        "alarms": alarms,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Live drift vs the pinned offline baseline.")
    ap.add_argument("--file", default=str(monitor.TELEMETRY), help="telemetry JSONL to check")
    args = ap.parse_args()

    baseline = json.loads(Path(AGENT_BASELINE).read_text(encoding="utf-8"))
    out = report(monitor.read(Path(args.file)), baseline)
    print(json.dumps(out, indent=2))
    if out.get("drift"):
        raise SystemExit("\n  DRIFT DETECTED — live behaviour diverged from the baseline.")


if __name__ == "__main__":
    main()
