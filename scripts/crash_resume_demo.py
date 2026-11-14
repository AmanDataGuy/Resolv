"""
# scripts/crash_resume_demo.py

**What:** kills a resolution mid-flight and resumes it. A child process resolves a refund-owed case with a scripted
fake model (lookup, refund, reply; `RESOLV_FAKE_LLM=1` for the extractor, so no key is needed). The parent waits
until N steps are journaled, kills the child with `os.kill` (SIGKILL on POSIX, TerminateProcess on Windows), runs
the same case again, and asserts exactly one settled refund, one ok refund record and an intact audit chain.
**Why:** durable execution is only a claim until a real kill shows it: journaled steps must replay without
re-calling the model, and a refund must not be paid twice even when the crash lands next to it.
**Used by:** run by hand: `python scripts/crash_resume_demo.py [--kill-after 3] [--latency 1.0]`.
"""
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# A refund-owed order: (order_id, owner, amount) for an in-window late delivery, paid at its cap.
def pick_order() -> tuple[str, str, float]:
    from datetime import date

    from resolv.config import AUTO_APPROVE_MAX_USD, CLAIM_WINDOW_DAYS, REFUND_CAP_FRACTION
    from resolv.harness.policy import NOW
    from resolv.harness.validity import get_order

    for i in range(1000, 1460):
        o = get_order(f"ORD-{i}")
        if o and o["situation"] == "late" and (NOW - date.fromisoformat(o["promised_date"])).days <= CLAIM_WINDOW_DAYS:
            cap = round(o["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
            return o["order_id"], o["customer_id"], min(cap, AUTO_APPROVE_MAX_USD)
    raise SystemExit("no in-window late order in data/db")


# The scripted model: what it does next depends only on how many tool results it has seen, so a resume continues.
def scripted_model(order_id: str, amount: float, calls: list):
    import litellm

    # one fake model call: sleep like a provider, then lookup -> refund -> plain-text reply
    async def fake(**kwargs):
        calls.append(1)
        await asyncio.sleep(float(os.environ.get("FAKE_LLM_LATENCY_S", "0")))
        seen = sum(1 for m in kwargs["messages"] if m.get("role") == "tool")
        plan = [("lookup_order", {"order_id": order_id}),
                ("issue_refund", {"order_id": order_id, "claim_type": "late_delivery", "amount_usd": amount})]
        if seen < len(plan):
            name, args = plan[seen]
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"tc{seen}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
        else:
            message = {"role": "assistant", "content": f"I've issued a refund of ${amount:.2f} for the late delivery."}
        return litellm.ModelResponse(model="fake", choices=[{"index": 0, "finish_reason": "stop", "message": message}])
    return fake


# Child process: resolve (or resume) the case and print how many model calls it really made.
def child(case_id: str, order_id: str, owner: str, amount: float) -> None:
    from resolv.agents import loop

    calls: list = []
    loop.acomplete = scripted_model(order_id, amount, calls)
    result = asyncio.run(loop.run_case(case_id, f"My order {order_id} arrived late.", caller_id=owner))
    print(json.dumps({"llm_calls": len(calls), "steps": result["steps"], "reply": result["reply"]}))


# Parent: start the child, kill it after N journaled steps, resume, and check the invariants.
def main() -> None:
    ap = argparse.ArgumentParser(description="Kill a resolution mid-flight, resume it, check nothing paid twice.")
    ap.add_argument("--kill-after", type=int, default=3, help="journaled steps before the kill (1 = after extract)")
    ap.add_argument("--latency", type=float, default=1.0, help="seconds per fake model call")
    ap.add_argument("--child", nargs=4, metavar=("CASE", "ORDER", "OWNER", "AMOUNT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return child(*args.child[:3], float(args.child[3]))

    tmp = tempfile.mkdtemp(prefix="resolv-crash-")
    os.environ.update(RESOLV_ENV="dev", RESOLV_DB=os.path.join(tmp, "resolv.db"),
                      RESOLV_PSP_DB=os.path.join(tmp, "psp.db"), RESOLV_DURABLE="1", RESOLV_FAKE_LLM="1",
                      FAKE_LLM_LATENCY_S=str(args.latency))
    from resolv.harness import audit, ledger

    order_id, owner, amount = pick_order()
    case_id = f"crash-{int(time.time())}"
    cmd = [sys.executable, __file__, "--child", case_id, order_id, owner, str(amount)]

    proc = subprocess.Popen(cmd, cwd=ROOT, env=os.environ.copy())
    while len(ledger.steps(case_id)) < args.kill_after and proc.poll() is None:
        time.sleep(0.05)
    if proc.poll() is None:
        os.kill(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))  # Windows: TerminateProcess, just as abrupt
        proc.wait()
        print(f"killed pid {proc.pid} after {len(ledger.steps(case_id))} journaled steps")
    else:
        print("child finished before the kill point; raise --latency or lower --kill-after")

    resumed = subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, check=True)
    print("resumed:", resumed.stdout.strip().splitlines()[-1])

    settled = [r for r in ledger.refunds(("settled",)) if r["order_id"] == order_id]
    paid = [r for r in audit.read(case_id) if r["tool"] == "issue_refund" and r["ok"]]
    chain = audit.verify_chain()
    assert len(settled) == 1, f"expected one settled refund, got {len(settled)}"
    assert len(paid) == 1, f"expected one ok refund record, got {len(paid)}"
    assert chain == (True, None), f"audit chain broken at {chain}"
    print(f"crash-resume OK: one settled refund of ${settled[0]['amount_cents'] / 100:.2f} on {order_id}, "
          f"{len(ledger.steps(case_id))} steps journaled, chain intact ({tmp})")


if __name__ == "__main__":
    main()
