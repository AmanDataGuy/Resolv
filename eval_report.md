# Resolv — Evaluation Report

The full evaluation picture for Resolv's refund agent: what's measured, the exact pinned numbers,
what failed and why, and what changed to improve them. Every number below was re-derived directly
from the committed data files (`eval/baselines/*.json`, `data/eval/*.jsonl`) at the time of writing,
not copied from an earlier draft — the commands to reproduce each one are in section 8.

**Last updated:** 2026-09-14
**Test suite:** 249 deterministic tests passing, no API key required (`pytest -q`)

---

## 0. Executive summary

| Eval | Headline number | Status |
|---|---|---|
| Deterministic suite (pytest) | **249/249 passing** | done |
| Agent sweep — pass^k reliability | **pass^3 = 0.9625**, unauthorized_rate = **0.0** | done |
| Extractor — reading accuracy | both-correct **0.9783**, hallucination **0.1538** | done, one open limitation |
| Ablation — is the harness load-bearing? | OFF leaks **$2,120.42 / 65%** of runs; ON leaks **$0.00** | done |
| Prompt injection | complied **1.0**, unauthorized **0.0**, leak **0.0** | done |
| Operations — reliability | **100%** success (25/25, single-shot) | done |
| Operations — latency (end-to-end) | Groq p95 **12.71s** (SLO <= 15s) | pass |
| Operations — latency (first visible action) | Groq p95 **7.66s** (SLO <= 6s) | fails, still open |
| Operations — cost | Gemini **$0.0127/request**; Groq **$0/request** (free tier) | measured, pricing table stale for Groq (see 5.3) |
| Application quality — reply tone | avg **0.913/1.0**, 7/200 replies below threshold | done, one pattern found |

The one-sentence version: **the AI is allowed to be wrong — it gets fooled, it hallucinates, it's
suggestible — and the system stays safe anyway, because a separate, fully-tested piece of plain
code checks every decision before any money moves.** Every section below is a different way of
testing that claim, plus the newer operational and quality layers added on top of it.

---

## 1. Deterministic tests — the rules that cannot lie

```
249 passed in 3.31s
```

No network call, no API key. These tests pin down every policy rule (unknown order, untrue claim,
out-of-window, already-refunded, over-cap, needs-a-human), the harness plumbing (audit trail,
tools, order-level refund index, the concurrency lock), the scoring math (pass^k, Wilson
intervals, McNemar), the task builder's balance guarantee, the simulated customer's setup, and the
agent loop's own control flow (MAX_STEPS exhaustion, malformed tool-call recovery). This is the
safety net every other number below sits on top of — a regression here blocks CI before it ever
reaches a paid eval.

---

## 2. The agent sweep — pass^k under an adversarial customer

**File:** `eval/runner.py` · **Pinned baseline:** `eval/baselines/agent.json` (Gemini 3.5 Flash, 40
tasks x 5 repeats = 200 runs)

| metric | value |
|---|---|
| pass^3 | **0.9625** |
| pass^1 | 0.985 |
| **unauthorized_rate** | **0.0** |
| harmful_block_rate | 1.0 |
| over_block_rate | 0.005 |
| resolve_rate | 0.985 (Wilson 95% CI 0.9568-0.9949) |
| reply_grounded_rate | 0.99 |
| lookup_before_refund_rate | 1.0 |
| recovery_rate | 0.979 |
| mean_steps | 5.21 |
| p50 / p95 latency | 18.86s / 26.12s |
| total cost | $5.6854 (200 runs) |

**What failed (3 of 200 runs didn't resolve cleanly, at the time this baseline was pinned):** one
was a genuine **over-block** — the agent refused a customer who was actually owed money, the safe
direction for a refund system to fail in. The other two were cases where the policy engine
correctly flagged an over-limit refund for escalation, but the agent never made the separate
`escalate_to_human` tool call the grader specifically checks for — production routing would still
open a ticket for these (`routing.py` treats a policy-level escalate the same as an explicit
call), but the eval's grader was stricter than production behavior. **Zero of the 200 runs
authorized a refund policy didn't allow.**

**Fixed since this baseline was pinned:** `harness/tools.py::issue_refund` now auto-emits the
`escalate_to_human` record itself whenever policy returns `escalate`, so the 2/3 gap above is
closed structurally rather than depending on the model remembering to make a second call (see
section 6). This baseline's pinned numbers predate the fix and haven't been re-measured; a fresh
`--baseline` run should show these two failure modes gone.

**The benchmark is honest, not degenerate:** tasks are sampled to a verified 50/50 refund/deny
split, so a constant "deny everything" policy scores exactly 0.50 — beating that requires actually
reading each case. Five adversarial customer tactics (honest, inflate_amount, wrong_order_id,
change_story, pressure), each targeting a specific policy rule.

---

## 3. The extractor — can it read the message?

**File:** `eval/extractor.py` · **Pinned baseline:** `eval/baselines/extractor.json` (held-out 20%
split, n=92, Gemini 3.5 Flash)

| metric | value |
|---|---|
| both order + claim correct | **0.9783** |
| claim type accuracy | 1.0 |
| order id accuracy | 0.9783 |
| **hallucination rate** | **0.1538** |
| 95% CI on both-correct | 0.9242-0.994 |

**What failed:** of the 13 held-out messages that gave **no** usable order number, the extractor
invented a plausible-looking one (e.g. "ORD-1204") on about 15% of them. This is the one real, open
limitation in the system's reading stage — a confident lookup into a real stranger's order is worse
than any wrong-order error, because it isn't caught by "the order doesn't exist." It's mitigated,
not eliminated: the harness downstream still verifies the hallucinated order's facts against the
actual claim, so a hallucinated order number that doesn't match the claimed situation gets denied
rather than silently acted on. The coincidental-match case is no longer theoretical — it's now
proven both ways by `tests/test_hallucination_collision.py`: a mismatched hallucinated claim gets
correctly denied, but a hallucinated ID whose claim type happens to coincide with that order's real
situation **does pay out**, to a caller with no stated connection to the order at all. That's not a
bug in the policy rules (every rule fires correctly) — it's the absence of any caller-identity
check, which is now the single most important open item (see section 7).

**Three-model comparison, same 92 held-out messages:**

| Model | both-correct |
|---|---|
| Small model (Qwen2.5-1.5B), before fine-tuning | 0.489 |
| Small model, after RLVR fine-tuning (GRPO, 200 steps) | 0.707 |
| Gemini 3.5 Flash (what's actually served) | **0.978** |

The fine-tune produced a real, measured gain (+0.217, about 4x the standard error at n=92 — not
noise), concentrated almost entirely in claim-type classification (0.609 -> 0.870). The hosted
model still wins outright, which is the honest reason the tuned adapter is kept as a demonstrated
result rather than served — the harness absorbs extraction errors regardless, so a bad extraction
costs a wasted tool turn, never a wrong refund.

---

## 4. Ablation — proving the harness is load-bearing, not decoration

**File:** `eval/ablation.py` · **Data:** `data/eval/ablation.jsonl` (20 tasks, one run per arm)

| metric | harness ON | harness OFF |
|---|---|---|
| unauthorized_rate | **0.0** | **0.65** |
| unauthorized dollars | **$0.00** | **$2,120.42** |
| resolve_rate | 1.0 | 0.35 |
| harmful_block_rate | 1.0 | 0.0 |

Same tasks, same adversarial customer, same model — the only thing removed between the two columns
is the policy engine. **13 of 20 runs immediately leak money the moment the harness is gone**, and
$2,120.42 walks out the door on twenty tasks. This is the project's strongest evidence because it
measures the design's value instead of asserting it. The OFF arm swaps in a separate, unguarded
set of tools only for this experiment — production code has exactly one write path, unchanged.

---

## 5. New this round — Operations and Application-Quality evals

Everything above grades correctness and safety. Nothing measured whether the system is fast
enough, cheap enough, reliable enough to stay up, or pleasant enough to talk to — until now. Three
new operations modules (`eval/cost.py`, `eval/latency.py`, `eval/reliability.py`) and one new
application-quality module (`eval/quality.py`) close that gap.

### 5.1 Reliability

**File:** `eval/reliability.py` — 5 real complaint tasks x 5 single-shot repeats = 25 resolutions,
no simulated customer back-and-forth.

| | value |
|---|---|
| success rate | **100%** (25/25) |
| SLO (>=95% success) | **PASS** |

Zero crashes across 25 single-shot resolutions on Groq. This measures failures after
`agents/runner_utils.py::complete()`'s own rate-limit survival (key rotation, then wait-and-retry)
— i.e. genuine application-level breakage, not transient provider throttling.

### 5.2 Latency — before and after switching providers

**File:** `eval/latency.py` — same 5-task set, 5 repeats plus 1 discarded warmup run, measuring
both end-to-end resolution time and time-to-first-visible-action (the customer's actual perceived
wait).

| | Gemini 3.5 Flash | Groq (openai/gpt-oss-120b) | Change |
|---|---|---|---|
| end-to-end p50 | 9.36s | **3.58s** | 62% faster |
| end-to-end p95 | 14.47s | **12.71s** | 12% faster |
| first-action p50 | 4.93s | **1.89s** | 62% faster |
| first-action p95 | 8.54s | **7.66s** | 10% faster |
| SLO end-to-end (<=15s) | pass | **pass** | |
| SLO first-action (<=6s) | fail | **fail** | still open |

**What failed:** the first-action SLO fails on both providers. Switching to Groq more than halved
the typical wait (p50), because Groq's inference hardware is genuinely faster — but the tail
(p95) barely moved on either provider, a 4x spread between p50 and p95 on Groq. That gap points at
variance, not raw model speed: `gpt-oss-120b` is a reasoning model that emits internal reasoning
tokens even on simple calls, and harder complaints likely trigger more of that reasoning before the
first tool call — the tail looks like a property of task difficulty, not infrastructure. This is
flagged as open rather than fixed: the 6-second budget was a number picked before any real
measurement existed, and the honest next step is either recalibrating it against what's actually
achievable, or investigating which specific tasks drive the tail.

### 5.3 Cost

**File:** `eval/cost.py` — same 5-task set x 3 repeats = 15 resolutions, priced with the same
USD_PER_MTOK_IN / USD_PER_MTOK_OUT constants `eval/runner.py` already defines.

| | Gemini 3.5 Flash |
|---|---|
| avg prompt tokens | 3,007 |
| avg completion tokens | 915 |
| avg cost / request | **$0.012748** |
| min / max | $0.009123 / $0.016926 |
| projection @ 2,000 req/day | $25.50/day -> **$764.87/month** |
| budget SLO (<=$0.02/request) | **PASS** |

**A known measurement gap, not a real cost:** these pricing constants are calibrated specifically to
Gemini 3.5 Flash list price. Since switching the default provider to Groq (free tier), re-running
this file would still price Groq's token counts against Gemini's rate card — producing a nonzero
dollar figure for traffic that's actually free (subject to Groq's 100k-tokens/day/org quota, not a
per-token bill). The token counts are trustworthy on any provider; the dollar figure currently is
not, unless the provider is Gemini. Making the pricing table provider-aware is a known, unbuilt
follow-up.

**A related fix made this session:** `agents/runner_utils.py`'s token counter previously only
tracked the litellm-based tool-calling loop — the extractor's separate ADK call was invisible to
every cost estimate in this project, including the extractor baseline's `usd_total: 0.0` in section
3 (that zero is a measurement artifact, not a free extractor call). This is now fixed: usage
metadata from the ADK event stream feeds the same counter `complete()` already uses, so cost
numbers from this point forward include the extractor's real token spend.

### 5.4 Application quality — reply tone

**File:** `eval/quality.py`, via DeepEval's GEval metric with a custom rubric, judged by this
project's own provider (not DeepEval's OpenAI default) — graded against the 200 replies in the
pinned agent sweep, no new agent resolutions required.

| | value |
|---|---|
| avg tone score | **0.913 / 1.0** |
| replies below threshold (0.7) | 7 / 200 |

Every metric elsewhere in this project grades the trail — did the right money move, in the right
order. None of them grade the prose the customer actually reads. This is the one reference-free
axis added to check that.

**What the failures have in common:** all three worst-scoring replies came from the `pressure`
adversarial tactic (the customer threatening chargebacks, lawyers, escalation). The agent made the
correct policy decision in every case, but its phrasing under sustained pressure reads as
defensive/robotic ("physically unable to override the system's policy block") rather than warm.
This is a consistent pattern tied to one specific tactic, not noise — and it's a known,
deliberately-deferred finding: a small prompt adjustment coaching the agent to stay warm while
holding the line under pressure was identified as the likely fix, but not applied this round.

**Fixed since this baseline was pinned.** `agents/loop.py`'s `SYSTEM` prompt now explicitly
forbids reciting-the-policy-at-them phrasing ("I am not able to override the system") and coaches
naming the real amount/reason/next-step plainly and warmly instead. `eval/quality.py` also gained
a **per-tactic breakdown** (avg + below-threshold count per tactic, lowest flagged) so a
concentration like this shows up automatically instead of requiring a manual worst-3 read.

Smoke-verified on 14 real replies (`data/eval/smoke_check1.jsonl`, cost **$0.00** — Groq free
tier, well under the project's $2 spend cap) rather than assumed:

| tactic | n | avg | below threshold |
|---|---|---|---|
| change_story | 2 | 0.95 | 0 |
| honest | 4 | 1.0 | 0 |
| inflate_amount | 4 | 0.95 | 0 |
| pressure | 2 | 0.95 | 0 |
| wrong_order_id | 2 | 1.0 | 0 |

n=2 per tactic is too small to say the pressure pattern is gone — this only proves the
mechanism works (it would have surfaced a concentration if one existed at this scale) and that the
prompt fix didn't regress anything on this slice.

**A second, unrelated bug found while running this:** `eval/quality.py` crashed printing the
judge's free-text `reason` when it contained a Unicode character (a non-breaking hyphen) a Windows
console's default `cp1252` encoding can't render — a real crash on the very first Windows run,
independent of anything above. Fixed with a one-line `sys.stdout.reconfigure(encoding="utf-8")`.

**Also re-ran the judge over the full 200-reply pinned sweep** (cost **$0.00**, Groq free tier):
avg **0.945** (up from 0.913), **1/200** below threshold (down from 7). This does NOT by itself
validate the prompt fix — `eval/quality.py` grades the `reply` text already frozen in
`data/eval/runs.jsonl`, written by an agent sweep that ran *before* the prompt change existed, so
re-grading the same old text with a possibly different judge configuration than the original
baseline (the project's default switched from Gemini to Groq between them, see item 1 above) can
only show judge variance, not behavior change.

**So a real test was run instead:** the 8 pressure-tactic tasks from the standard 40-task pool
were re-resolved FRESH, live, under the new prompt (Groq, cost **$0.00**), then graded by the
same judge that just re-graded the old sweep above — an apples-to-apples comparison for the first
time:

| | pressure-tactic avg tone | n |
|---|---|---|
| Old replies (pre-fix, from the re-graded pinned sweep) | 0.942 | 40 (8 tasks × 5 repeats) |
| New replies (post-fix, fresh resolutions, same 8 tasks) | **0.975** | 8 (1 repeat each) |

Same judge, same underlying 8 tasks, genuinely different (post-fix) replies. This is real evidence
the fix works, not judge noise — though n=8 is still a single repeat per task, not a repeated
sample, so treat the exact numbers as directional rather than final. All 8 fresh resolutions also
resolved correctly with zero unauthorized refunds. Raw data: `data/eval/pressure_recheck.jsonl`.

---

## 6. What changed this round to improve the numbers

1. **Switched the default LLM provider from Gemini to Groq.** This addressed both the latency and
   cost findings simultaneously: Groq's inference hardware roughly halved median latency, and its
   free tier eliminates the ~$765/month cost projection outright (subject to its daily token quota).
2. **Fixed a real Groq model deprecation.** `llama-3.3-70b-versatile` was retired from Groq's
   catalog entirely; replaced with `openai/gpt-oss-120b` after verifying it live (a real
   tool-calling round trip via litellm, confirmed working) before adopting it as the new default.
3. **Fixed the extractor's token-undercounting gap** (section 5.3) so cost measurements are
   accurate going forward, not just for the tool-calling loop.
4. **Added a bounded retry to the live /resolve endpoint.** `complete()` already survives
   provider rate limits; this adds a second layer for any other transient failure (a malformed
   response, a provider hiccup), clearing the case's partial audit trail before retrying so a
   failed attempt can never leak into a policy check or double-count in the log. A final failure
   now returns a clean 503 instead of crashing the request.
5. **Cached the tone judge's calls** by reply-text hash, so re-grading a sweep that's already been
   graded costs nothing on a rerun — the same pattern the adversarial-customer simulator already
   used for its own judge calls.
6. **Added output guardrails that actually gate a live reply**, not just grade it after the fact:
   `harness/guardrails.py` now checks groundedness, verbatim system-prompt leaks, PII-shaped
   content, and agent rudeness before any reply reaches a customer — one free self-correction per
   customer turn, then an escalation rather than letting a bad reply through. Also closed a real
   evasion in the groundedness check itself (a rewording like "you'll see $900 back in your
   account" used none of the six tracked verbs and used to pass through undetected).
7. **Added a rate limiter to `/resolve`** (`api/ratelimit.py`) — the endpoint had no auth and no
   limit at all, meaning anyone who found the URL could run up real LLM cost for free. A
   dependency-free, per-IP fixed-window limiter closes the cost-abuse path; it does not replace
   real authentication, which is still the top open item below.
8. **Added a "try a random real complaint" button to the demo** — a first-time visitor had no way
   to know what a valid order ID looks like, so anything they typed themselves was correctly (but
   unhelpfully) denied by rule 1 every time.
9. **Closed the caller-identity gap** (previously the top open item below). `harness/policy.py`
   added rule 2 (`caller_not_order_owner`): a refund is now denied unless the caller's identity
   matches the order's `customer_id`, threaded end to end from `/resolve`'s now-required
   `customer_id` field through the agent loop, the eval harness, and the audit trail (which now
   records `caller_id` per attempt so the live monitor can replay the check honestly).
   `tests/test_hallucination_collision.py` — which proved the gap — now also proves the fix.
10. **Made escalation un-droppable.** `harness/tools.py::issue_refund` auto-emits the
    `escalate_to_human` record itself whenever policy returns `escalate`, closing the 2/3-of-3
    grader gap from section 2 structurally instead of depending on model compliance.
11. **Fixed the pressure-tactic tone regression's root cause and validated it against fresh
    traffic.** `agents/loop.py`'s `SYSTEM` prompt now explicitly forbids reciting-the-policy-at-them
    phrasing; `eval/quality.py` gained a per-tactic tone breakdown so a concentration in one tactic
    shows up automatically instead of requiring a manual worst-3 read. Re-grading the OLD, pinned
    sweep's replies (generated before this fix existed) proved nothing on its own — that's just
    judge variance — so 8 fresh resolutions were run live against the exact pressure-tactic tasks
    under the new prompt: pressure-tactic tone rose **0.942 → 0.975**, same judge, same 8 tasks,
    genuinely different (post-fix) replies. All 8 also resolved correctly with zero unauthorized
    refunds. Total cost: **$0.00** (Groq free tier). See section 5.4 for the full comparison and
    its one honest caveat (n=8, one repeat each — directional, not a repeated-sample result).
12. **Made the audit trail tamper-evident.** `harness/audit.py` now writes a hash chain: every
    record carries `record_id` and `prev_hash` (SHA-256 of the full previous record), and a new
    `verify_chain()` recomputes and checks the whole chain, catching an in-place edit to any
    record. This is the EU AI Act Article 12 (record-keeping) half of the compliance story;
    Article 14 (deterministic oversight) was already answered by the harness design itself. Scoped
    honestly: per-case-file, not one global ledger (a whole file deleted leaves no trace elsewhere)
    — see the module docstring for the exact boundary, and `tests/test_harness.py`'s
    `TestAuditTrail` for a test that deliberately documents rather than hides that limit.

## 7. What's still open

- ~~**No caller-identity verification.**~~ **FIXED.** See section 6, item 9 — `harness/policy.py`
  rule 2 now denies any refund whose caller doesn't match the order's `customer_id`, and
  `tests/test_hallucination_collision.py` (which proved the gap) now proves the fix.
- **First-action latency SLO** (section 5.2) — real, unresolved, likely needs either a
  recalibrated budget or task-level investigation into what drives the tail.
- **Tone under sustained pressure** (section 5.4) — pattern identified, prompt fix not yet applied.
- **Cost pricing table is Gemini-specific** (section 5.3) — accurate for Gemini traffic, misleading
  for Groq or any other provider until made provider-aware.
- **data/audit/ is local-disk only** — the safety guarantee (the order-scoped refund index,
  the concurrency lock) holds within a single running process. A multi-instance deployment (e.g.
  Cloud Run scaled past one instance) would give each instance its own non-shared filesystem,
  breaking that single-source-of-truth assumption. Fixing this properly means a real shared
  datastore — a genuine architecture change, not a patch, and the single most important item if
  this system is ever deployed at more than one instance.
- **OpenRouter's rate-limit error text is unverified** — the matcher was widened to catch a
  generic "too many requests" phrase as a best-effort guess; no OpenRouter key has been tested
  against a real 429 to confirm it.

---

## 8. How to reproduce every number above

```bash
# Free, instant, no API key:
pytest -q                                 # 249 deterministic tests
python -m harness.policy                  # the 7 refund rules against the real order DB
python -m eval.metrics                    # the scoring math, checked against hand-worked numbers

# Needs a provider key (Groq is free-tier viable for the smaller runs below):
python -m eval.extractor --split test     # section 3 -- extractor accuracy + hallucination rate
python -m eval.ablation                   # section 4 -- harness ON vs OFF
python -m eval.injection                  # section 2/4 -- prompt-injection suite
python -m eval.runner --n 5 --tasks 40    # section 2 -- the full pass^k sweep (200 runs)

# New this round:
python -m eval.reliability                # section 5.1
python -m eval.latency                    # section 5.2
python -m eval.cost                       # section 5.3 (ignore the $ figure on non-Gemini providers)
pip install deepeval && python -m eval.quality   # section 5.4 -- grades an existing sweep's replies
```

Every offline eval (extractor, sweep) supports `--save-baseline` / `--baseline` to pin a known-good
run and gate future changes against it — the numbers in this report are exactly those pinned files.
