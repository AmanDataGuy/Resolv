# Runbook

How to run Resolv in production, how to tell when it is unhealthy, and what to do about it.
This is a single-node, demo-grade deployment: one process, one SQLite file (`RESOLV_DB`, default
`data/resolv.db`), one payment-provider stub (`RESOLV_PSP_DB`).

## SLOs

| SLO | Target | Measured by | Current |
|---|---|---|---|
| Availability (`/health` returns 200) | 99% monthly (single machine, no failover) | external uptime probe | not yet measured |
| End-to-end `/resolve` p95 | ≤ 15 s | `evals/latency.py`, `loadtest/resolve.py` | not yet measured on the deploy |
| First action p95 (first tool call) | ≤ 6 s | `evals/latency.py` | not yet measured on the deploy |
| Unauthorized refunds | 0, always | `resolv/invariants.py`, audit trail | must stay 0 |
| Escalations decided | ≤ 24 h (they expire at 24 h) | `escalations` table: `decided_at - created_at` | not yet measured |

## Dashboards

- **Langfuse:** one trace per `/resolve` (root agent span, LLM spans, tool spans), plus online
  scores from `resolv/monitor.py`.
- **Approvals queue:** `/console` (pending count, age, time to expiry).
- **Local telemetry:** `data/telemetry` rows written by `resolv/monitor.py` on every request.

## Alert to action

| Alert | First action | Then |
|---|---|---|
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
approvals filed while frozen can then be approved normally.

### Shadow mode

`RESOLV_SHADOW=1` runs the agent and the policy on real complaints but moves no money: a refund policy would
allow is recorded with `rule_id=shadow_mode`, `action=escalate`, `ok=0`, and the reason "Shadow mode: would
have refunded $X (<policy rule_id>)". No refund row, no payment call; the ticket is routed to a person.
Compare these rows with what the humans decided. It is refused (RuntimeError) when `RESOLV_ENV` is `test` or
`eval`, so an eval can never score a dry run. With both switches set, the freeze wins.

### Idempotency on /resolve

Clients may send an `Idempotency-Key` header (1-255 chars) on `POST /resolve`. The first response is stored
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

- Online copy: `sqlite3 data/resolv.db ".backup data/backup/resolv-$(date +%F).db"` (safe while
  the app is running). Do the same for `RESOLV_PSP_DB`.
- Record the chain head with each backup (`python -c "from resolv.harness import ledger; print(ledger.head())"`),
  so a later restore can be checked with `ledger.verify(head)`.
- Use the host's volume snapshots on top of this. Check the platform documentation for how often
  they run and how long they are kept.
- Restore drill: not yet performed.

## Deploy and rollback

The deploy configuration belongs to the owner and is not described here. The application-side
rules are:

- Deploy one image. Health check `GET /health`.
- Roll back by redeploying the previous image. The SQLite schema is created with
  `CREATE TABLE IF NOT EXISTS`, so an older image reads a newer file only if no column was
  removed. Check before you roll back across a schema change.
- After any deploy or rollback, run the chain check and one `/resolve` smoke request on a demo order.
