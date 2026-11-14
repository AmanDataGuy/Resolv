# Scale-out design note

What breaks first when Resolv outgrows one node, and how it would be moved past. **None of this is built.**
Today Resolv is one process with one SQLite file, and the measured knee says that is enough for a demo.
Read by people sizing a deploy and by reviewers asking "what happens at 10x". Numbers come from
`loadtest/results/fake_llm_knee.json` (fake model, one uvicorn worker, a laptop).

## Where one node stands

- One worker holds **40 req/s** of `/resolve` cleanly (p95 1.619-1.644 s against 1.6 s of fake model time,
  `/health` p95 9 ms). The knee is **60 req/s** (`/health` p95 168-179 ms). At 80 req/s throughput falls to
  33.894 ok/s.
- Before the step journal moved off the event loop, the knee was 40 req/s: each SQLite commit's fsync froze
  every request. Now the next limit is CPU in one Python event loop.
- With a real model the provider's rate limit binds long before this. 40 req/s of model traffic is far past
  typical provider rate limits. The service is not the first wall. The provider quota is.

## What breaks, in order

| # | Part | Where | What goes wrong past one node | Smallest fix |
|---|---|---|---|---|
| 1 | One event loop | one uvicorn worker | CPU-bound at about 60 req/s | More workers or processes. This is safe only after rows 2-4, because each process gets its own copy of that state |
| 2 | Rate limits | `resolv/api/ratelimit.py:19`, a dict in memory | N processes allow N x 10 requests a minute per IP. The `/token` brute-force bound grows N times | A shared counter: Redis `INCR` + `EXPIRE` per key and window, or a Postgres row per key |
| 3 | Circuit breaker | `resolv/agents/llm.py:59`, a deque in memory | Each process learns about a provider outage on its own and keeps calling it | The same shared store. One failure counter and one "open until" time per provider |
| 4 | Background jobs | `_every` in `resolv/api/main.py:43` (reconciler, outbox drain) | Every process runs them. That is safe, because the work is keyed and idempotent, but it wastes work and sends duplicate PSP calls | One leader: `pg_try_advisory_lock(<job id>)` at the start of each pass, or a single cron job |
| 5 | Ledger writer | `resolv/harness/ledger.py`, one SQLite file, `BEGIN IMMEDIATE` | One file means one host, because SQLite over a network filesystem is unsafe. Writers queue on one lock. Each commit pays a full fsync | Postgres with the same tables (see below) |
| 6 | Idempotency keys | `idempotency` table, `ledger.claim_key` | Correct on one file. Across hosts it needs the same shared database | Moves with the ledger. `INSERT ... ON CONFLICT DO NOTHING` replaces read-then-insert |
| 7 | Outbox and telemetry files | `resolv/integrations/notify.py:19`, `data/telemetry` | Local files on each host. A second host's messages and metrics are invisible to the first | A real notifier (email or webhook) fed by the outbox table, and metrics sent to the trace backend instead of a file |

## The ledger on Postgres

- **Same schema.** The partial unique indexes (`one_live_refund_per_order`, `one_pending_per_action`) port
  as they are, so "one live refund per order" stays a database rule, not an app rule.
- **Serialize per order, not globally.** `BEGIN IMMEDIATE` locks the whole file today. On Postgres, start the
  refund transaction with `pg_advisory_xact_lock(hashtext(order_id))`, or take `SELECT ... FOR UPDATE` on the
  order's row. That keeps policy's read-check-write atomic for one order while other orders run in parallel.
- **The HMAC chain is the hard part.** Each record signs the previous record's mac, so appends must take
  turns. Keep one chain and serialize only the audit insert behind a global advisory lock. That is a
  short critical section, not the whole request. If that lock becomes the bottleneck, split into one chain
  per shard (for example per day or per order hash) and export every chain head.
- **Jobs and drains** use `SELECT ... FOR UPDATE SKIP LOCKED`, so two workers never take the same outbox
  row or pending refund.

## A queue, if requests outgrow the request/response model

With a real model, one `/resolve` lasts seconds, and the API process holds it the whole time. Past a few
hundred in flight, accept the complaint, write a `cases` row, return the case id, and let a worker pool
pull cases with `SKIP LOCKED`, or from SQS. The client polls or listens on a stream. The step journal
already makes a case resumable after a crash, so a worker that dies mid-case can be replaced without
paying twice.

## Why not now

There is one node and no traffic that needs more. Each fix adds a service (Redis, Postgres, a queue) that
must be run, backed up and secured. Built before there is load, it is a liability. Moving to Postgres
first would also need the restore drill in `docs/RUNBOOK.md` redone for the new store.
