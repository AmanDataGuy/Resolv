# Runbook

How to run Resolv in production, how to tell when it is unhealthy, and what to do about it.
This is a single-node, demo-grade deployment: one process, one SQLite file (`RESOLV_DB`, default
`data/resolv.db`), one payment-provider stub (`RESOLV_PSP_DB`).

## SLOs

| SLO | Target | Measured by | Current |
|---|---|---|---|
| Availability (`/health` returns 200) | 99% monthly (single machine, no failover) | external uptime probe | no probe running yet |
| End-to-end `/resolve` p95 | ≤ 15 s | `resolv_resolve_seconds` (p95 panel), `evals/latency.py`, `loadtest/resolve.py` | real model on the deploy: no run yet. Resolv's own share is small: with a fake 1.6 s model, p95 is 1.644 s at 40 req/s |
| First action p95 (first tool call) | ≤ 6 s | `evals/latency.py` | real model on the deploy: no run yet |
| `/health` p95 while `/resolve` is busy | ≤ 100 ms | `loadtest/resolve.py` health probe | 9 ms at 40 req/s, 168-179 ms at 60 req/s (fake model, one worker) |
| Sustained `/resolve` load per node | ≤ 40 req/s (two thirds of the 60 req/s knee) | `loadtest/resolve.py` | 40 req/s held twice: 38.957-38.958 ok/s, 0 errors, p95 1.619-1.644 s |
| `/resolve` errors | ≤ 5% of requests over 10 min | `resolv_errors_total` vs `resolv_resolve_seconds_count`, alert `ResolveErrorRateHigh` | no production traffic yet |
| Unauthorized refunds | 0, always | `resolv/invariants.py`, audit trail | must stay 0 |
| Refund payouts | no hour above 3x the previous day's hourly rate (and over $200) | `resolv_refund_usd_total`, alert `RefundPayoutSpike` | no production traffic yet |
| LLM spend | ≤ $10 a day | `resolv_llm_cost_usd_total`, alert `LLMCostBurnHigh` | fake model only so far |
| Escalations decided | ≤ 24 h (they expire at 24 h) | `escalations` table: `decided_at - created_at` | no production traffic yet |

The load numbers come from `loadtest/results/fake_llm_knee.json`: a fake model with 0.8 s per call, one
uvicorn worker, a laptop (i5-14450HX, NVMe disk), 60 s per rate. They measure Resolv itself, not the
provider. With a real model, provider rate limits and per-IP limits (10/min) bind long before 40 req/s.
The deploy VM is smaller than the laptop, so re-run the sweep there before trusting the capacity row.

## Dashboards

- **Prometheus and Grafana:** `GET /metrics` exposes `resolv_refund_decisions_total{decision,rule}`,
  `resolv_refund_usd_total`, `resolv_resolve_seconds`, `resolv_llm_cost_usd_total`, `resolv_refunds_frozen` and
  `resolv_errors_total{kind}` (defined in `resolv/metrics.py`). `ops/` holds the scrape config, the three alert
  rules (`alerts.yml`) and a six-panel Grafana dashboard. Local stack on a fake model:
  `docker compose -f ops/docker-compose.yml up -d --build`, then Grafana at 127.0.0.1:3000, Prometheus at
  127.0.0.1:9090, the API at 127.0.0.1:8010; `down -v` removes it. Export `ORDER_PROOF_SECRET` first if a client
  (app.py with `RESOLV_API_URL`, or curl) needs receipt proofs for `/token`.
- **`/metrics` has no auth, on purpose.** It holds only counts, rule ids and error kinds, never customer data,
  and Prometheus scrapes it from inside the private network. Do not route `/metrics` through the public proxy:
  block it there, so only the internal network can read it.
- **Langfuse:** one trace per `/resolve` (root agent span, LLM spans, tool spans), plus online
  scores from `resolv/monitor.py`.
- **Approvals queue:** `/console` (pending count, age, time to expiry).
- **Local telemetry:** `data/telemetry` rows written by `resolv/monitor.py` on every request.

## Alert to action

| Alert | First action | Then |
|---|---|---|
| `RefundPayoutSpike` (Prometheus) | Look at the "Refund decisions" panel by rule, and recent `issue_refund` rows in the audit trail | If the payouts are not explained by real traffic, freeze refunds (kill switch below) and treat it like `unauthorized > 0` |
| `ResolveErrorRateHigh` (Prometheus) | Read `resolv_errors_total` by kind: `circuit_open` is the provider, `resolve_failed` is in the logs | Follow the provider row below for `circuit_open`; for `resolve_failed` read the logged traceback |
| `LLMCostBurnHigh` (Prometheus) | Check traffic (`resolv_resolve_seconds_count`) and the model in use (`AGENT_MODEL`) | If traffic is normal, a prompt or model change raised the cost per case. Roll it back or lower the per-IP rate limit |
| `unauthorized > 0` (a refund paid that policy forbids) | **Freeze refunds**: set `REFUNDS_FROZEN=1` and restart (kill switch below) | Export the case trace from Langfuse. Run the chain check. Write a failing test that reproduces it before any fix |
| Provider errors / circuit breaker open (`/resolve` returns 503) | Check the provider status page and the breaker state in logs (`CircuitOpen`) | Wait for the breaker to half-open, or switch provider (`AGENT_MODEL` etc. to another provider's model) and restart |
| Provider daily quota exhausted | `/resolve` degrades to 503. Customers see "try again" | Switch provider or wait for the reset. Do not raise retry counts |
| `unknown` or `pending` refunds older than 10 min | Run the reconciler: `python -c "from resolv.harness import tools; print(tools.reconcile_once())"` | If a row stays `unknown` after 3 attempts it becomes `failed` and a review escalation is filed. Check the PSP for that idempotency key |
| Audit chain verification failed | Freeze refunds. Copy the DB file aside (do not edit it) | `python -c "from resolv.harness import ledger; print(ledger.verify(<exported head>))"` gives the first bad `seq`. Compare it with the last exported head |
| Disk near full | Check `data/` (DB, outbox, telemetry) | Rotate `data/outbox` and `data/telemetry`. Never delete from the ledger |
| Escalations expiring undecided | Check the console queue and the approver roster (`APPROVER_IDS`) | Expired approvals never pay. The customer must re-request |

### Kill switch

`REFUNDS_FROZEN=1` (set it and restart) stops all money. In `resolv/harness/tools.py` `issue_refund`, any
refund policy would allow becomes an escalation with `rule_id=refunds_frozen`: it is recorded in the
audit trail, an approval lands in the console, and the customer's ticket is routed to a team. Refunds policy
denies are still denied for their real reason. Approving in the console while frozen pays nothing either: it
re-escalates. The check sits after `check_refund`, not inside it, so policy stays pure and replays of past
refunds (the `unauthorized` monitor) still judge them by the rules alone. Unset it and restart to resume;
approvals filed while frozen can then be approved normally. `resolv_refunds_frozen` reads 1 while it is on.

### Shadow mode

`RESOLV_SHADOW=1` runs the agent and the policy on real complaints but moves no money: a refund policy would
allow is recorded with `rule_id=shadow_mode`, `action=escalate`, `ok=0`, and the reason "Shadow mode: would
have refunded $X (<policy rule_id>)". No refund row, no payment call; the ticket is routed to a person.
Compare these rows with what the humans decided. It is refused (RuntimeError) when `RESOLV_ENV` is `test` or
`eval`, so an eval can never score a dry run. With both switches set, the freeze wins.

### Idempotency on /resolve

Clients may send an `Idempotency-Key` header (1-255 chars) on `POST /resolve` or `POST /resolve/stream`; both
share one key table, so a key first used on the stream replays on `/resolve` and the other way round (a replay on
the stream sends only the stored `result` event). The first response is stored
in the ledger's `idempotency` table for 24 h; a repeat with the same key and body returns it without running a
new case. The same key with a different body or customer gets 422; a repeat while the first is still running
gets 409. A failed request releases its key so the client can retry.

## Key rotation

| Secret | Effect of rotating | How |
|---|---|---|
| `AUTH_SECRET` | every outstanding customer and approver token becomes invalid | Set the new value and restart. Customers re-exchange their proof codes. Re-mint approver tokens with `python -m resolv.api.auth --mint-approver <id>` |
| `ORDER_PROOF_SECRET` | every printed receipt code changes | Rotate only on compromise. Old receipts stop working |
| `AUDIT_HMAC_KEY` | new records are signed with the new key, old ones still verify | Set `AUDIT_KEY_ID` to a new id and `AUDIT_HMAC_KEY` to the new key. Move the old pair into `AUDIT_OLD_KEYS="oldid:oldsecret,..."`. Restart. Run the chain check |

All three are required (≥ 32 chars for the auth secrets) when `RESOLV_ENV=prod`. The app refuses
to start without them.

## Backup

- Online copy (safe while the app is running). The slim Python image has no `sqlite3` CLI, so use
  Python's backup API:
  `python -c "import sqlite3; s=sqlite3.connect('data/resolv.db'); d=sqlite3.connect('data/backup/resolv-$(date +%F).db'); s.backup(d); d.close()"`.
  Where the CLI exists, `sqlite3 data/resolv.db ".backup data/backup/resolv-$(date +%F).db"` does the
  same. Do the same for `RESOLV_PSP_DB`.
- Record the chain head with each backup (`python -c "from resolv.harness import ledger; print(ledger.head())"`),
  so a later restore can be checked with `ledger.verify(head)`. Keep the head outside the database.
- Use the host's volume snapshots on top of this. Check the platform documentation for how often
  they run and how long they are kept.

### Restore

1. Stop the API (this is also the refund freeze). Copy the damaged file and its `-wal`/`-shm` aside. Do not edit them.
2. Delete `resolv.db`, `resolv.db-wal` and `resolv.db-shm`, then copy the backup to `RESOLV_DB`. A
   leftover `-wal` from the old file would be replayed into the restored one.
3. `python -c "import sqlite3; print(sqlite3.connect('data/resolv.db').execute('PRAGMA integrity_check').fetchone())"` must print `('ok',)`.
4. `python -c "from resolv.harness import ledger; print(ledger.verify({'seq': <seq>, 'mac': '<mac>'}))"`
   with the head saved with that backup must print `(True, None)`. Run it with the same `AUDIT_HMAC_KEY`
   and `AUDIT_OLD_KEYS` as production, or every record fails.
5. Before you start the API again: every refund paid after the backup exists at the payment provider but
   not in the restored ledger, so policy rule 5 cannot see it and could pay it a second time. List the
   provider's refunds since the backup time and put them back in the ledger (or deny those orders by hand)
   before refunds reopen. Then start the API, run the reconciler once and send one smoke `/resolve`.

A file that is not a database at all makes `ledger.verify` raise `sqlite3.DatabaseError`. It does not
return `(False, seq)`. Treat either result as "restore from backup".

### Restore drill, 2026-11-14 (local)

There is no production disk to restore from, so the drill ran on a local ledger (`data/drill/`) seeded
through the real harness: 5 cases of `lookup_order`, `issue_refund` and `escalate_to_human`, which gave 15
signed records under a throwaway `AUDIT_HMAC_KEY`.

| Step | Result |
|---|---|
| Online backup with the Python backup API, head recorded | 81,920 bytes, head `seq` 15 |
| Edit one audit row in place (`seq` 3), then `verify(head)` | `(False, 3)`: the edited row is named |
| Overwrite the live file with junk, then `verify(head)` | raises `DatabaseError: file is not a database` |
| Restore steps 1-3 | `integrity_check` = `ok` |
| Restore step 4 | `verify(head)` = `(True, None)` |
| One new escalation on the restored ledger, then `verify()` | `(True, None)`, head `seq` 16 |

Not covered by this drill: a restore from a host volume snapshot, and the provider match in step 5 (the
drill's refunds were all denied by policy, so no money moved).

## Deploy and rollback

The deploy configuration belongs to the owner and is not described here. The application-side
rules are:

- Deploy one image. Health check `GET /health`.
- Roll back by redeploying the previous image. The SQLite schema is created with
  `CREATE TABLE IF NOT EXISTS`, so an older image reads a newer file only if no column was
  removed. Check before you roll back across a schema change.
- After any deploy or rollback, run the chain check and one `/resolve` smoke request on a demo order.
