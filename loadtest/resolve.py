"""
# loadtest/resolve.py

**What:** an open-loop load test of the API: one POST /token up front, then POST /resolve at a fixed arrival
rate for a fixed duration (plus a light GET /health probe), with a cap on requests in flight. Writes one JSONL row
per request and a summary (throughput, p50/p95/p99 latency, error rate) per endpoint.
**Why:** open loop means requests arrive on schedule whether or not earlier ones finished, so a slow server shows
up as queueing and errors, not as a politely lower request rate (the closed-loop blind spot). asyncio + httpx
instead of k6: no new binary. For a capacity number the server must run with the fake LLM (RESOLV_FAKE_LLM=1),
otherwise you measure the provider, not the service. See loadtest/README.md.
**Used by:** people, from the command line; tests/test_loadtest_summary.py (percentile, summarize).
"""
import argparse
import asyncio
import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

HEALTH_PER_SECOND = 2
MESSAGES = [
    "My order {order} arrived a week late. I want a refund for the delay.",
    "Order {order} never showed up. Please refund me.",
    "I canceled {order} but I was still charged. Refund it please.",
    "Hi, {order} was late again, this is the third time. What can you do?",
]


# The p-th percentile (0-100) by linear interpolation between closest ranks; None for no data.
def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


# Per-endpoint summary of request rows: count, errors, error rate, throughput and latency percentiles.
def summarize(rows: list[dict], duration_s: float) -> dict:
    out = {}
    for endpoint in sorted({r["endpoint"] for r in rows}):
        mine = [r for r in rows if r["endpoint"] == endpoint]
        ok = [r["latency_s"] for r in mine if r["ok"]]
        errors = len(mine) - len(ok)
        out[endpoint] = {
            "requests": len(mine),
            "errors": errors,
            "error_rate": round(errors / len(mine), 4),
            "ok_per_s": round(len(ok) / duration_s, 3) if duration_s > 0 else None,
            "p50_s": _round(percentile(ok, 50)),
            "p95_s": _round(percentile(ok, 95)),
            "p99_s": _round(percentile(ok, 99)),
            "status_counts": _counts(r["status"] for r in mine),
        }
    return out


# Round a latency to milliseconds, keeping None.
def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 3)


# {status: count} with string keys (JSON-friendly); a failed connection counts as status 0.
def _counts(statuses) -> dict:
    counts: dict[str, int] = {}
    for s in statuses:
        counts[str(s)] = counts.get(str(s), 0) + 1
    return dict(sorted(counts.items()))


# Time one request and return its row; any transport error becomes status 0, never an exception.
async def _timed(client: httpx.AsyncClient, endpoint: str, method: str, path: str, **kwargs) -> dict:
    started = time.perf_counter()
    status, error = 0, None
    try:
        resp = await client.request(method, path, **kwargs)
        status = resp.status_code
    except httpx.HTTPError as e:
        error = f"{type(e).__name__}: {e}"[:200]
    return {"endpoint": endpoint, "t_start": round(started, 4), "latency_s": round(time.perf_counter() - started, 4),
            "status": status, "ok": 200 <= status < 300, "error": error}


# Exchange the order proof for a customer token once; the run reuses it (tokens live an hour).
async def get_token(client: httpx.AsyncClient, order: str, proof: str) -> str:
    resp = await client.post("/token", json={"order_id": order, "proof": proof})
    resp.raise_for_status()
    return resp.json()["token"]


# Fire `make()` at `rate` per second for `duration` seconds, at most `limit` in flight; returns all rows.
async def open_loop(make, rate: float, duration: float, limit: int) -> list[dict]:
    gate = asyncio.Semaphore(limit)
    tasks = []

    # One scheduled request; it waits for a slot, and that wait counts in its latency (as queueing would).
    async def one():
        started = time.perf_counter()
        async with gate:
            row = await make()
        row["latency_s"] = round(time.perf_counter() - started, 4)
        return row

    begin = time.perf_counter()
    for i in range(int(rate * duration)):
        await asyncio.sleep(max(0.0, begin + i / rate - time.perf_counter()))
        tasks.append(asyncio.create_task(one()))
    return list(await asyncio.gather(*tasks))


# Run the test and write requests.jsonl + summary.json; returns the summary.
async def run(args) -> dict:
    out_dir = Path(args.out or f"evals/results/loadtest_{'fake' if args.fake_llm else 'live'}_"
                               f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(base_url=args.base, timeout=args.timeout) as client:
        token = await get_token(client, args.order, args.proof)

        # One /resolve call; with --ip-header each call claims a random client IP (see README, rate limits).
        def resolve():
            headers = {"Idempotency-Key": uuid.uuid4().hex}
            if args.ip_header:
                headers[args.ip_header] = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
            body = {"message": random.choice(MESSAGES).format(order=args.order), "token": token}
            return _timed(client, "resolve", "POST", "/resolve", json=body, headers=headers)

        # One /health probe: the event loop's responsiveness under load.
        def health():
            return _timed(client, "health", "GET", "/health")

        started = time.perf_counter()
        groups = await asyncio.gather(
            open_loop(resolve, args.rate, args.duration, args.concurrency),
            open_loop(health, HEALTH_PER_SECOND, args.duration, HEALTH_PER_SECOND * 2),
        )
        elapsed = time.perf_counter() - started

    rows = groups[0] + groups[1]
    with (out_dir / "requests.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {"mode": "fake-llm" if args.fake_llm else "live", "base": args.base, "rate_per_s": args.rate,
               "duration_s": args.duration, "concurrency": args.concurrency, "elapsed_s": round(elapsed, 2),
               "endpoints": summarize(rows, elapsed)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# Command-line flags.
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Open-loop load test of POST /resolve.")
    p.add_argument("--base", default="http://localhost:8000", help="API base URL")
    p.add_argument("--order", required=True, help="a demo order id, e.g. ORD-1187")
    p.add_argument("--proof", required=True, help="that order's proof code (resolv.api.auth.order_proof, same ORDER_PROOF_SECRET)")
    p.add_argument("--rate", type=float, default=2.0, help="new /resolve requests per second (arrival rate)")
    p.add_argument("--duration", type=float, default=60.0, help="seconds to keep sending")
    p.add_argument("--concurrency", type=int, default=50, help="max /resolve requests in flight")
    p.add_argument("--timeout", type=float, default=120.0, help="per-request timeout in seconds")
    p.add_argument("--ip-header", default=None,
                   help="send a random client IP in this header (server's TRUSTED_CLIENT_IP_HEADER) to spread rate limits")
    p.add_argument("--fake-llm", action="store_true",
                   help="label the run as a capacity run; the SERVER must be started with RESOLV_FAKE_LLM=1")
    p.add_argument("--out", default=None, help="output folder (default evals/results/loadtest_<mode>_<time>)")
    return p.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(parse_args())), indent=2))
