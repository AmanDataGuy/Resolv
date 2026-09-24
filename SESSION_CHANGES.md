# Session changes — security fix, reliability fixes, tamper-evidence, and a stress-test bug

A single working session's worth of changes, recorded here because they span many files and
several distinct fixes — easier to review as one story than as a diff alone. Every claim below
was verified: 264 deterministic tests pass, every touched module's own `demo()` self-check
passes, and the real end-to-end stress test in the last section passes against the actual
`/resolve` endpoint.

---

## 1. Closed the caller-ownership gap (the most important fix)

**The bug, proven before the fix:** `tests/test_hallucination_collision.py` showed that a
customer message with no stated order number could still receive a real refund — if the
extractor's hallucinated order ID happened to be real, and its guessed claim type happened to
match that order's true situation, the refund paid out to whoever asked. Every policy rule fired
correctly; nothing anywhere checked that the caller had any actual connection to the order.

**The fix:** `harness/policy.py` gained **rule 2** (`caller_not_order_owner`) — a refund is now
denied unless the caller's identity matches the order's `customer_id`. `caller_id` has no
bypassable default: passing `None` denies exactly like a real mismatch, so it can't be silently
skipped by a caller that forgets to pass it.

**Threaded end to end**, since a check that only exists in one function is not a guarantee:
- `api/main.py` — `/resolve` now requires a `customer_id` field on every request.
- `agents/loop.py` — `run_case`/`run_case_events`/`_bind` all carry `caller_id` through to the
  tool-calling closure, never exposed to the model's own tool schema.
- `harness/tools.py` — `issue_refund` takes `caller_id` with no default, forcing every call site
  to say explicitly who is asking.
- `harness/audit.py` — every record now stores `caller_id`, so the live monitor
  (`eval/monitor.py`) can replay the ownership check honestly against real production traffic,
  not just against an eval's answer key.
- The entire eval harness (`eval/tasks.py`, `runner.py`, `ablation.py`, `injection.py`, `cost.py`,
  `latency.py`, `reliability.py`) was updated to pass the real order owner as `caller_id`, so
  existing benchmark numbers still reflect an honest caller, not a universally-denied one.
- The Streamlit demo (`app.py`) looks up the real owner of whichever sample/random complaint is
  showing, so demo refunds still work.

`tests/test_hallucination_collision.py` — which proved the bug — now also proves the fix, plus a
new third test confirming an honest owner still gets paid.

---

## 2. Made escalation un-droppable

**The bug:** `eval_report.md`'s own numbers showed 2 of 3 imperfect sweep runs were cases where
policy correctly returned `escalate` (an over-limit refund), but the agent never made the
separate `escalate_to_human` tool call the grader checks for — nothing in the system prompt
actually told the model to make that second call for this exact scenario.

**The fix:** `harness/tools.py::issue_refund` now auto-emits the `escalate_to_human` record itself
whenever policy returns `escalate`, guarded so a model that *does* also call it separately doesn't
produce a duplicate record. The guarantee is now structural, matching the same reasoning already
used for `MAX_STEPS` exhaustion and repeated guardrail failures in `agents/loop.py`.

---

## 3. Fixed the pressure-tactic tone regression, and actually proved it

**The bug:** the application-quality eval found all three worst-scoring replies came from the
`pressure` adversarial tactic — correct policy decisions, delivered in defensive/robotic language
("physically unable to override the system's policy block").

**The fix:** `agents/loop.py`'s `SYSTEM` prompt now explicitly forbids reciting-the-policy-at-them
phrasing and coaches naming the real amount/reason/next-step plainly and warmly. `eval/quality.py`
gained a **per-tactic tone breakdown** so a concentration like this shows up automatically instead
of requiring a manual worst-3 read, and a Windows console encoding crash (a Unicode character in
the judge's own reasoning text) found while running it was fixed too.

**Proven, not assumed** — the first attempt at validating this (re-grading the *old*, pre-fix
replies at full 200-reply scale) was a mistake I caught and corrected: grading already-frozen text
can only show judge variance, not whether the agent's actual behavior changed. So 8 fresh
resolutions were run live against the exact pressure-tactic tasks under the new prompt:

| | pressure-tactic avg tone | n |
|---|---|---|
| Old (pre-fix) | 0.942 | 40 (8 tasks × 5 repeats) |
| New (post-fix, fresh) | **0.975** | 8 (1 repeat each) |

Same judge, same 8 underlying tasks, genuinely different post-fix replies. All 8 fresh
resolutions also resolved correctly with zero unauthorized refunds. Total cost: **$0.00** (Groq
free tier). n=8 is one repeat each, so read this as directional, not final.

---

## 4. Made the audit trail tamper-evident (EU AI Act Article 12)

**The gap:** the audit trail was honest but not tamper-*evident* — append-only was a filesystem
convention, and nothing would notice if someone with disk access edited a line after the fact.

**The fix:** `harness/audit.py` now writes a hash chain. Every record carries a `record_id` (a
fresh UUID) and a `prev_hash` (SHA-256 of the *entire* previous record, including that record's
own `prev_hash`), and a new `verify_chain()` recomputes and checks the whole chain, returning
exactly which record stopped matching if one was altered.

**Scoped honestly, not oversold:**
- The chain is per case-file, not one global ledger — deleting an entire case's file leaves no
  trace anywhere else. A tamper-proof deletion guarantee needs a separate index of "which cases
  exist," which isn't built here.
- Canonicalization is `sort_keys=True` JSON, not full RFC 8785 — sufficient for a single Python
  process writing and verifying its own hashes; would need upgrading only if a non-Python
  verifier ever needed to independently recompute these digests.

7 new tests cover this, including one that deliberately documents what the chain *can't* catch
(last-record tampering) rather than hiding the limitation.

---

## 5. Found and fixed a real crash via end-to-end stress testing

Running the actual `/resolve` endpoint (not mocks) with a message that states no complaint at all
(e.g. "hi", "quick question") crashed the **entire resolution** before the agent loop even
started. Root cause: `CustomerClaim.claim_type` was a required field, so Groq's strict
structured-output schema validation rejected the model's honest attempt to answer `null` when a
message genuinely doesn't describe a problem yet — the same class of provider-side rejection, not
a bug in the agent's own reasoning.

**The fix:** `claim_type` is now optional on `CustomerClaim` (`schemas.py`), mirroring `order_id`'s
existing optionality. The extractor's own instructions (`agents/extractor.py`) now explicitly say
when to answer `null`. `harness/validity.py::verify_claim` gained a matching guard so a `None`
claim_type returns "can't judge" instead of crashing on a `.replace()` call that assumed a string
— currently unreachable via any real call path (`check_refund` only ever supplies a claim_type
from a model's tool-call argument, which is required and enum-constrained), but guarded anyway now
that the type permits it.

Verified twice: once via a new unit test (`tests/test_harness.py`), and once by re-running the
live stress test against the real endpoint and confirming zero tracebacks where there had been a
full crash before.

---

## 6. End-to-end stress test (real `/resolve` calls, not mocks)

A FastAPI `TestClient` run against the actual app, exercising every layer touched this session in
one pass — the rate limiter, the required `customer_id` field, the ownership policy rule,
auto-escalation, the agent loop, the extractor, guardrails, routing, notify, monitor telemetry,
and the hash-chained audit trail:

| Check | Result |
|---|---|
| `/health` responds | PASS |
| Missing `customer_id` → 422 (rejected before any agent call) | PASS |
| Honest owner, real late-delivery claim → resolves cleanly | PASS |
| No refund reaches a mismatched `caller_id` | PASS |
| Impersonation attempt (wrong `customer_id`) → denied by `caller_not_order_owner`, not a crash | PASS |
| The real case's audit hash chain verifies intact | PASS |
| Rate limiter returns 429 once the per-IP cap is hit | PASS |

This run is what surfaced item 5 above on its first pass (a message with no complaint content
crashed the whole request) — fixed, then re-run clean.

---

## 7. Explicitly declined (from a separate improvement-plan review, recorded so the reasoning isn't lost)

- **A trending third-party "System One" classifier model** for pre-routing/latency — real
  product, but v0.0.1, 5 days old, marked "experimental" by its own integrator, and requires a new
  paid third-party account with no track record. Declined in favor of not adding unproven vendor
  risk to a project whose whole thesis is deterministic, auditable decisions.
- **Rewriting `policy.py`'s 7 rules into a YAML-driven rule engine** to "prove the harness
  generalizes" — declined; it would mean refactoring the exact function that gates every refund,
  and `policy.py`'s own docstring already explains why a policy DSL here would be
  resume-driven architecture, not an improvement.
- **A full k=1/2/4/8 reliability sweep at meaningful scale** — would cost $20+ against a $2
  session budget; not run.
- **OpenTelemetry/OpenInference tracing + a packaged GitHub Action** — real, but bigger
  packaging/reusability work with no correctness payoff; deprioritized behind the fixes above.

---

## 8. Root cleanup, done before this push

Removed (all were already gitignored — this only tidies the working tree, nothing tracked
changed): 14 stale build/generation `.log` files from July, and every `__pycache__` /
`.pytest_cache` / `.ruff_cache` directory (all auto-regenerate on the next run). Moved one
unrelated cross-project planning document out of the repo entirely (it was untracked but not
gitignored, so it risked being swept into a future `git add .` despite having nothing to do with
Resolv).

---

## Verification summary

```
264 tests passed (249 at the true start of this session, per eval_report.md's own header —
  +15 new across all items above: ownership/collision tests, escalation, audit chain, validity)
All module demo() self-checks pass: policy, tools, audit, guardrails, routing, tasks, metrics, simulator, api
End-to-end stress test against the real /resolve endpoint: 10/10 checks pass, zero tracebacks
Total real API spend this session: $0.00 (all real-provider runs done on Groq's free tier)
```
