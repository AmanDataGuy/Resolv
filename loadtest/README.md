# Load test: `POST /resolve`

`loadtest/resolve.py` measures where the **service** saturates. It is open loop: requests arrive at
`--rate` per second whether or not earlier ones finished, capped at `--concurrency` in flight. A
slow server shows up as queueing and errors instead of a quietly lower request rate. A light
`GET /health` probe (2/s) runs alongside, so you can see whether the event loop stays responsive.

No measured results yet. The numbers in this file are flags, not findings.

## Output

One folder per run (default `evals/results/loadtest_<fake|live>_<UTC time>/`):

- `requests.jsonl`: one row per request: `endpoint`, `t_start`, `latency_s` (includes time spent
  waiting for an in-flight slot), `status` (0 means the connection failed), `ok`, `error`
- `summary.json`: per endpoint: `requests`, `errors`, `error_rate`, `ok_per_s`, `p50_s`, `p95_s`,
  `p99_s`, `status_counts`. Percentiles use only successful requests. Failures count in
  `error_rate`.

## Two things the server needs

1. **A fake LLM for capacity runs.** With a real provider you measure Groq's queue, not Resolv's. Start the
   server with `RESOLV_FAKE_LLM=1` (a canned, tool-free reply, optional `FAKE_LLM_LATENCY_S` sleep;
   this hook lives in `resolv/agents/llm.py`, see "Fake-LLM mode" below), and pass `--fake-llm` so the
   run is labelled. `--fake-llm` only labels the run; the server-side variable is what changes behaviour.
2. **Rate limits.** `/resolve` allows 10 requests per minute per client IP and `/token` allows 5.
   From one machine, almost every request would be a 429. To spread load, start the server with
   `TRUSTED_CLIENT_IP_HEADER=X-Loadtest-IP` and run with `--ip-header X-Loadtest-IP`, so each
   request claims a random `10.x.x.x` address. **Never set that variable on a server reachable
   from the internet**: any client could then pick its own rate-limit key.

## Run

```bash
# server (capacity mode), in one shell
RESOLV_ENV=dev RESOLV_FAKE_LLM=1 FAKE_LLM_LATENCY_S=0.8 TRUSTED_CLIENT_IP_HEADER=X-Loadtest-IP \
  ORDER_PROOF_SECRET=loadtest-secret uvicorn resolv.api.main:app --port 8000

# the proof code for a demo order (same secret as the server)
ORDER_PROOF_SECRET=loadtest-secret python -c "from resolv.api.auth import order_proof; print(order_proof('ORD-1187'))"

# load, in another shell: step the rate up one run at a time
for rate in 2 5 10 20 40; do
  python -m loadtest.resolve --order ORD-1187 --proof <code> --rate $rate --duration 60 \
    --concurrency 400 --ip-header X-Loadtest-IP --fake-llm --out evals/results/loadtest_fake_rate$rate
done
```

`FAKE_LLM_LATENCY_S=0.8` is a placeholder. Set it to the median step latency in the reference
rows and record the value you used.

To limit CPU and memory the way a small VM would, run the API image under Docker with
`--cpus=1 --memory=1g` and sample `docker stats` every 2 s.

## Reading the result

- **Saturation knee:** the first rate where `/resolve` p95 exceeds the 15 s end-to-end SLO or
  `error_rate` exceeds 1%.
- **`/health` p95 under load:** if it climbs with the offered rate, something is blocking the
  event loop.
- **Bottleneck:** event loop (health latency rises), SQLite writer lock (`BEGIN IMMEDIATE` waits,
  503s), or CPU (from `docker stats`).
- Every `/resolve` uses the same order, so after the first refund, rule 5 (`already_refunded`)
  denies the rest. This is fine for a service-capacity test. It is not a test of refund
  throughput.

## Fake-LLM mode

Set on the **server** process:

| Variable | Effect |
|---|---|
| `RESOLV_FAKE_LLM=1` | `acomplete()` and `complete()` return a litellm `ModelResponse` with a short canned text reply and no tool calls (a JSON-schema extractor call gets an all-null claim). `run_agent_once()` returns `{"order_id": None, "claim_type": None}` (or `""` with no output key). No provider is called and no key is needed. |
| `FAKE_LLM_LATENCY_S` | seconds each fake call sleeps (default `0`), standing in for provider latency |

Each `/resolve` then costs one fake extractor call and one fake agent call, so it ends with a plain reply and no
refund. Everything else is real: auth, rate limits, the ledger writes, the step journal, routing and telemetry.
Never set `RESOLV_FAKE_LLM` on a server real customers reach.

## Live low-concurrency check

Against a real deploy with the real model: drop `--fake-llm` and `--ip-header`, and use
`--rate 0.1 --duration 400 --concurrency 4` (about 40 requests). This uses provider quota.
Demo orders only.
