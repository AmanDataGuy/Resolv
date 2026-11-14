# Resolv — Evaluation Report

The evaluation history of Resolv's refund agent: what was measured, what failed, and what changed.

> **Read this first.** The per-run rows behind most numbers here (`data/eval/*.jsonl`) were archived
> out of the repo. Only `evals/baselines/agent.json` and `evals/baselines/extractor.json` are committed,
> so only numbers taken from those two files can be checked from this repo. Every other number is
> marked **historical, rows not in repo** and has not been re-measured. Both baseline files are
> themselves historical: Gemini 3.5 Flash, pinned 2026-07-25, before rule 2 (caller must own the
> order), the reply guardrails and the prompt change.
>
> The system under test has changed since: SQLite ledger with a keyed (HMAC) audit chain, a refund
> saga against a simulated PSP, human approvals bound to the exact action, a durable step journal,
> NaN/Infinity rejection in tool arguments, malformed calls recorded on the trail, owner-scoped order
> lookup, reworked reply guardrails, a fully async loop, order-proof session tokens, and an
> independent invariant grader. None of the historical numbers below were measured on that system.

**Test suite:** keyless, no API key (`pytest -q`). The count is not restated here; get it from
`pytest --collect-only -q`. (Earlier drafts said 244, 249 and 274 in different places.)

---

## 0. Executive summary

| Eval | Headline number | Status |
|---|---|---|
| Deterministic suite (pytest) | keyless; count from `pytest --collect-only -q` | not restated |
| Agent sweep — pinned pass^k (Gemini) | **pass^3 = 0.9625**, unauthorized_rate = **0.0** | historical, from `evals/baselines/agent.json` |
| Agent sweep — full reliability curve (Groq, 320 runs) | pass^1→8: 0.850 → 0.625, unauthorized_rate 0.0 | historical, rows not in repo |
| Extractor — reading accuracy | both-correct **0.9783**, hallucination **0.1538** (n=13, Wilson 0.043–0.422) | historical, from `evals/baselines/extractor.json` |
| Extractor — regex baseline | 0.891 both-correct (prior audit) | not yet re-measured in-repo (section 3.1) |
| Ablation — is the harness load-bearing? | OFF paid $2,120.42 gross ($2,069.32 above what was owed), 65% of runs; ON $0.00 | historical, rows not in repo |
| Prompt injection | complied 1.0, unauthorized 0.0, leak 0.0 | **provisional: metric confounded**, see 2.2 |
| Operations — reliability | 100% success (25/25, single-shot) | historical, rows not in repo |
| Operations — latency (end-to-end) | Groq p95 12.71s (SLO <= 15s) | historical, rows not in repo |
| Operations — latency (first visible action) | Groq p95 7.66s (SLO <= 6s) | historical, rows not in repo; SLO still open |
| Operations — cost | Gemini $0.0127/request at historical constants | historical, rows not in repo; see 5.3 |
| Application quality — reply tone | avg 0.953/1.0, 0/198 below threshold | historical, rows not in repo; same-model judge, uncalibrated |
| Authentication | order-proof session tokens (`/token` + `/resolve`) | rewritten, see 6.13 |
| Audit trail | SQLite ledger, HMAC chain keyed by `AUDIT_HMAC_KEY` | rewritten, see 6.12 |

The one-sentence version: **the AI is allowed to be wrong — it gets fooled, it hallucinates, it's
suggestible — and the system stays safe anyway, because a separate, fully-tested piece of plain
code checks every decision before any money moves.** Every section below is a different way of
testing that claim, plus the newer operational and quality layers added on top of it.

---

## 1. Deterministic tests — the rules that cannot lie

No network call, no API key. These tests pin down every policy rule (unknown order, untrue claim,
out-of-window, already-refunded, over-cap, needs-a-human), the harness plumbing (audit trail,
tools, order-level refund index, the concurrency lock), the scoring math (pass^k, Wilson
intervals, clustered bootstrap), the task builder's balance guarantee, the simulated customer's setup, and the
agent loop's own control flow (MAX_STEPS exhaustion, malformed tool-call recovery). This is the
safety net every other number below sits on top of — a regression here blocks CI before it ever
reaches a paid eval.

---

## 2. The agent sweep — pass^k under an adversarial customer

**File:** `evals/runner.py` · **Pinned baseline:** `evals/baselines/agent.json` (Gemini 3.5 Flash, 40
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

**Fixed since this baseline was pinned:** `resolv/harness/tools.py::issue_refund` now auto-emits the
`escalate_to_human` record itself whenever policy returns `escalate`, so the 2/3 gap above is
closed structurally rather than depending on the model remembering to make a second call (see
section 6). This baseline's pinned numbers predate the fix and haven't been re-measured. The rule
that flagged them was rule 6 at the time and is rule 7 now (rule 2, caller ownership, was added).

**How grading works, and its limit:** the grader replays each refund through the same
`check_refund` the system uses, so a bug in that function would score as "authorized". The sweep now
also runs an independent invariant checker (`resolv/invariants.py`) that does not call policy code.

**The benchmark is honest, not degenerate:** tasks are sampled to a verified 50/50 refund/deny
split, so a constant "deny everything" policy scores exactly 0.50 — beating that requires actually
reading each case. Five adversarial customer tactics (honest, inflate_amount, wrong_order_id,
change_story, pressure), each targeting a specific policy rule.

### 2.1 The full reliability curve — pass^1 through pass^8

**Historical, rows not in repo.** **File:** `evals/runner.py --n 8 --tasks 40` (320 runs, Groq
`gpt-oss-120b`, $0.00 on the free tier). The run spread load across several keys on separate
accounts; the code now supports one key per provider, and that practice is not repeated. This is a
genuinely new measurement, not the pinned baseline re-run — it's on the project's current default
provider (Groq), not the Gemini the pinned baseline above used, so treat the two as separate data
points about different models rather than a before/after.

| k | pass^k |
|---|---|
| 1 | 0.850 |
| 2 | 0.785 |
| 4 | 0.713 |
| 8 | **0.625** |

| metric | value |
|---|---|
| unauthorized_rate | **0.0** (320/320) |
| harmful_block_rate | 1.0 |
| over_block_rate | 0.078 |
| resolve_rate | 0.85 (Wilson 95% CI 0.807-0.885) |
| recovery_rate | 0.867 |
| mean_steps | 5.22 |
| p50 / p95 latency | 8.55s / 27.82s |
| crashed runs | 6 / 320 (1.9%, recorded as failures, not dropped) |

**What this shows, and why it's the headline artifact of this whole eval suite:** single-shot
success (pass^1 = 0.85) looks solid, but reliability degrades sharply the more times the same task
is attempted — by k=8 it's down to 0.625. Part of that decay is not the model: 5 of the 6 crashes
were provider rate limits, the grader of that time had a known spec gap, and the naive CI is about
half the width of a task-clustered one. Re-scoring it is open.
**The claim that survives this entire curve unmoved is `unauthorized_rate = 0.0`.** Task-success
reliability is a property of the underlying model and degrades exactly as expected; the safety
guarantee is a property of the deterministic harness gating every mutation, and it does not degrade
at all, across 320 real adversarial runs. That separation — a model that gets *less* reliable at
the actual job the more you ask it to repeat, while the thing that must never fail never does — is
the entire thesis of this project, now demonstrated with a real curve instead of one k=3 snapshot.

### 2.2 Prompt injection (provisional)

The old injection result (complied 1.0, unauthorized 0.0, leak 0.0; rows not in repo) is
**provisional: the metric is confounded**. The model probes above the cap even with no attack, so
"complied" measured baseline cap-probing, not the attack. `evals/injection.py` now runs a control arm
(the honest complaint alone) next to each attack arm and reports the lift over control. Not yet
measured with the new design.

---

## 3. The extractor — can it read the message?

**File:** `evals/extractor.py` · **Pinned baseline:** `evals/baselines/extractor.json` (held-out 20%
split, n=92, Gemini 3.5 Flash)

The extractor backend changed from ADK to json on 2026-11-14; numbers before that date are from the ADK backend.

| metric | value |
|---|---|
| both order + claim correct | **0.9783** |
| claim type accuracy | 1.0 |
| order id accuracy | 0.9783 |
| **hallucination rate** | **0.1538** (2 of n=13, Wilson 95% CI 0.043–0.422) |
| 95% CI on both-correct | 0.9242-0.994 |

**What failed:** of the 13 held-out messages that gave **no** usable order number, the extractor
invented a plausible-looking one (e.g. "ORD-1204") on about 15% of them. This is the one real, open
limitation in the system's reading step — a confident lookup into a real stranger's order is worse
than any wrong-order error, because it isn't caught by "the order doesn't exist." It's mitigated,
not eliminated: the harness downstream still verifies the hallucinated order's facts against the
actual claim, so a hallucinated order number that doesn't match the claimed situation gets denied
rather than silently acted on. The coincidental-match case is no longer theoretical — it's now
proven both ways by `tests/test_hallucination_collision.py`: a mismatched hallucinated claim gets
correctly denied, but a hallucinated ID whose claim type happens to coincide with that order's real
situation **does pay out**, to a caller with no stated connection to the order at all. That's not a
bug in the policy rules (every rule fires correctly) — it was the absence of a caller-identity
check, since closed by rule 2 (section 6, item 9).

**Comparison on the same 92 held-out messages:**

| Extractor | both-correct | source |
|---|---|---|
| Small model (Qwen2.5-1.5B), before fine-tuning | 0.489 | measured once on Kaggle, rows not retained |
| Small model, after GRPO fine-tuning (200 steps) | 0.707 | measured once on Kaggle, rows not retained |
| Regex extractor (no model) | 0.891 | prior audit, not yet re-measured in-repo |
| Gemini 3.5 Flash (hosted) | **0.9783** | `evals/baselines/extractor.json` |

### 3.1 The regex baseline is the bar

A plain regex extractor is the floor any model has to beat. A prior audit measured it at 0.891
both-correct on this split; that has not yet been re-measured in-repo. Re-measure it (seconds, CPU,
no key) with:

```bash
python -m inference.serve_bench --backend regex --label regex
```

Every later `inference.serve_bench` run is compared against it, paired. See `docs/inference.md`.

### 3.2 The GRPO fine-tune: an honest negative result

The GRPO LoRA fine-tune raised the small model from 0.489 to 0.707, but that is still below the
regex baseline (0.891) and well below the hosted model (0.9783). It is a negative result: the tuned
adapter is not worth serving. Its reward was a regex exact-match on the order id and claim type,
not the `verify_claim` check production uses, so "RLVR" here means a regex-verified reward. The
harness absorbs extraction errors either way: a bad extraction costs a wasted tool turn, never a
wrong refund.

---

## 4. Ablation — proving the harness is load-bearing, not decoration

**Historical, rows not in repo.** **File:** `evals/ablation.py` · **Data:** was `data/eval/ablation.jsonl`
(20 tasks, one run per arm)

| metric | harness ON | harness OFF |
|---|---|---|
| unauthorized_rate | **0.0** | **0.65** |
| unauthorized dollars | **$0.00** | **$2,120.42** |
| resolve_rate | 1.0 | 0.35 |
| harmful_block_rate | 1.0 | 0.0 |

Same tasks, same adversarial customer, same model — the only thing removed between the two columns
is the policy engine. **13 of 20 runs leak money once the harness is gone**: $2,120.42 paid out
gross, of which $2,069.32 was above what the customers were owed. One run per arm, so no CI. This is the project's strongest evidence because it
measures the design's value instead of asserting it. The OFF arm swaps in a separate, unguarded
set of tools only for this experiment — production code has exactly one write path, unchanged.

---

## 5. Operations and application-quality evals

**Every number in this section is historical, rows not in repo.** The task set was 5 LLM-generated
complaints over real orders, not real customer complaints. Dollar figures computed at $1.50/$9.00 per
M tokens are **priced at historical Gemini 3.5 Flash constants**; pricing now lives in
`config.price_usd`.

Everything above grades correctness and safety. Nothing measured whether the system is fast
enough, cheap enough, reliable enough to stay up, or pleasant enough to talk to — until now. Three
new operations modules (`evals/cost.py`, `evals/latency.py`, `evals/reliability.py`) and one new
application-quality module (`evals/quality.py`) close that gap.

### 5.1 Reliability

**File:** `evals/reliability.py` — 5 synthetic complaint tasks x 5 single-shot repeats = 25 resolutions,
no simulated customer back-and-forth.

| | value |
|---|---|
| success rate | **100%** (25/25) |
| SLO (>=95% success) | **PASS** |

Zero crashes across 25 single-shot resolutions on Groq. This measures failures after
`resolv/agents/llm.py::complete()`'s own rate-limit survival (wait-and-retry; key rotation at the
time, since removed)
— i.e. genuine application-level breakage, not transient provider throttling.

### 5.2 Latency — before and after switching providers

**File:** `evals/latency.py` — same 5-task set, 5 repeats plus 1 discarded warmup run, measuring
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

**The task-difficulty hypothesis above was wrong — corrected by real data.** Analyzing 136+ real
resolutions (from the tone-revalidation sweep, section 5.4) against total latency: `difficulty`
barely matters (easy/medium/hard all within 2s of each other, 13.9-16.3s avg) — the actual driver
is **step count**, almost linearly (2 steps→5s, 4→11.5s, 6→17.6s, 9-10→~38s), and step count is
driven by which **adversarial tactic** is in play, not the complaint's inherent complexity.
`inflate_amount` and `wrong_order_id`/`change_story` (the tactics that force the agent into
back-and-forth correction) average 16-19s and 5.5-5.8 steps; `honest` averages 12.5s and 4.2
steps. A handful of specific tasks are consistently slow across all 5 of their repeats — a real,
reproducible property of those complaint+tactic combinations, not noise. Caveat: this measures
*total* latency, not the specific *first-action* p95 the SLO is about — that needs the
event-level timestamps `evals/latency.py` captures separately, not yet cross-referenced against
this breakdown. The corrected mental model — the tail is about how many correction rounds an
adversarial customer forces, not raw task difficulty — is still open as an unsolved SLO, but it's
now a falsifiable, specific claim instead of a guess.

**Tested a real fix, got a real negative result.** `gpt-oss-120b` supports a documented
`reasoning_effort` parameter (low/medium/high) trading reasoning depth for latency — this is the
actual mechanism behind the diagnosis above, not a guess, and `resolv/agents/llm.py::complete()`
supported it (opt-in via `REASONING_EFFORT`, since litellm 1.91.0 doesn't recognise Groq as a
supported provider for this param yet and it has to be forced through via
`allowed_openai_params`). A real before/after comparison (5 tasks, 1 repeat each, $0 on Groq) found
`reasoning_effort=low` did **not** help: avg total latency 4.19s→5.09s, avg first-action
2.92s→4.17s, and one task's resolution outcome flipped from resolved to not (4/5 vs 5/5). n=5 is
small and noisy — this is not proof it never helps — but there's no positive signal here to act
on, so it is NOT recommended or defaulted to. Left in the code as a real, working, opt-in
capability for whoever wants to re-test it at a larger sample. **Rows not retained.**

### 5.3 Cost

**File:** `evals/cost.py` — same 5-task set x 3 repeats = 15 resolutions, priced with the
USD_PER_MTOK_IN / USD_PER_MTOK_OUT constants `evals/runner.py` defined at the time (historical Gemini
3.5 Flash constants; replaced by `config.price_usd`).

| | Gemini 3.5 Flash |
|---|---|
| avg prompt tokens | 3,007 |
| avg completion tokens | 915 |
| avg cost / request | **$0.012748** |
| min / max | $0.009123 / $0.016926 |
| projection @ 2,000 req/day | $25.50/day -> **$764.87/month** |
| budget SLO (<=$0.02/request) | **PASS** |

**FIXED since this baseline was pinned.** `evals/cost.py` is now provider-aware. Real numbers,
re-measured live (15 resolutions, $0.00 on Groq): avg 3,826 prompt / 598 completion tokens, and
the report now shows **$0.00 (free tier)** honestly instead of a borrowed Gemini rate, alongside
what it WOULD cost on Groq's real paid/dev-tier list price ($0.15/$0.60 per Mtok, checked against
Groq's own docs) — **$0.000933/request, ~$56/month at 2,000 req/day**. OpenRouter gets the same
honest treatment: no single rate is stated (it depends entirely on which model is configured), with a clearly-labeled Gemini-equivalent comparison number instead of a fabricated real one.

**A related fix made this session:** `resolv/agents/llm.py`'s token counter previously only
tracked the litellm-based tool-calling loop — the extractor's separate ADK call was invisible to
every cost estimate in this project, including the extractor baseline's `usd_total: 0.0` in section
3 (that zero is a measurement artifact, not a free extractor call). This is now fixed: the
extractor's json call goes through the same `acomplete()` meter as the agent loop, so cost
numbers from this point forward include the extractor's real token spend.

### 5.4 Application quality — reply tone

**Caveat:** the judge was the same model family as the agent and had no human labels; calibrating
it against a human-labelled set is future work. **File:** `evals/quality.py`, via DeepEval's GEval metric with a custom rubric, judged by this
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

**Fixed since this baseline was pinned.** `resolv/agents/loop.py`'s `SYSTEM` prompt now explicitly
forbids reciting-the-policy-at-them phrasing ("I am not able to override the system") and coaches
naming the real amount/reason/next-step plainly and warmly instead. `evals/quality.py` also gained
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

**A second, unrelated bug found while running this:** `evals/quality.py` crashed printing the
judge's free-text `reason` when it contained a Unicode character (a non-breaking hyphen) a Windows
console's default `cp1252` encoding can't render — a real crash on the very first Windows run,
independent of anything above. Fixed with a one-line `sys.stdout.reconfigure(encoding="utf-8")`.

**Also re-ran the judge over the full 200-reply pinned sweep** (cost **$0.00**, Groq free tier):
avg **0.945** (up from 0.913), **1/200** below threshold (down from 7). This does NOT by itself
validate the prompt fix — `evals/quality.py` grades the `reply` text already frozen in
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

**Full-scale confirmation.** The complete 40-task × 5-repeat sweep (200 fresh resolutions, live,
under the new prompt, Groq, cost **$0.00**) finished and was graded for tone:

| tactic | n | avg tone | below threshold |
|---|---|---|---|
| pressure | 40 | **0.972** | 0 |
| wrong_order_id | 38 | 0.987 | 0 |
| change_story | 40 | 0.940 | 2 |
| inflate_amount | 40 | 0.927 | 1 |
| honest | 40 | 0.905 | 1 |

`pressure` — the tactic that was the *worst* in the original pinned sweep — is now the **best**
of the five, at full scale, not just n=8. Overall avg tone 0.946, 4/198 below threshold. Because
all five tactics were graded together in the same sweep by the same judge, this ranking is not
vulnerable to the judge/provider-swap confound noted above — it's a genuine within-sweep
comparison. `unauthorized_rate` on this same sweep: **0.0** (200/200), the safety claim holding at
full scale under a different model and a changed prompt. Raw data: `data/eval/tone_recheck_runs.jsonl`.

**New finding, and it's a measurement artifact, not an agent problem — checked before acting on
it.** `honest` is now the *lowest*-scoring tactic (0.905), with judge comments like "does not
address or explain a denial at all" on the worst cases. Traced before writing any fix: every one
of those low-scoring replies has `steps=5, escalated=True` — a multi-turn conversation where the
actual explanation was given 1-2 turns *before* escalation, and the reply being graded is just the
closing line after that ("You're welcome! We'll let you know..."). `evals/quality.py` grades only
the final `reply` string with zero conversation context, so a perfectly reasonable sign-off reads
as content-free to a judge that never saw the substantive explanation earlier in the same
exchange. This is NOT a tone regression to fix in the agent's prompt — doing that would train it
to redundantly re-explain itself in every closing message, which reads worse, not better. It's a
real limitation of the grader itself: worth fixing by giving the judge more context (the last
assistant turn that actually decided something, not just the literal final string), not by
changing what the agent says. Left as a known eval-methodology gap rather than a behavior bug.

**Fixed and verified.** `evals/quality.py`'s `tone` metric now includes a one-sentence context
string per reply (derived from `steps`/`paid`/`escalated`, fields already in every row — no new
agent calls needed), telling the judge whether the explanation was already delivered earlier in
the conversation. Re-graded the same 198 replies: `honest` rose **0.905 → 0.947** (no longer the
lowest tactic), replies below threshold dropped **4/198 → 0/198** (every reply now passes), and
overall avg rose **0.946 → 0.953**. The judge's own stated reasoning now explicitly reflects the
fix — e.g. "given the context that the denial was already explained earlier, the short closing is
acceptable and not defensive or dismissive," on a reply that previously scored low for exactly
that brevity. This fixes the missing-context problem. It does not validate the judge itself:
that needs the human-labelled calibration set.

**One number from this sweep that is NOT comparable to anything above:** `pass^3` on this run is
0.765 (resolve_rate 0.845), against the pinned baseline's 0.9625. This is NOT a regression from
the prompt fix — the pinned baseline was measured on **Gemini 3.5 Flash**; this sweep ran on
**Groq's gpt-oss-120b** (the project's current default provider, switched for cost/latency
reasons — see item 1 below). Comparing pass^k across two different models measures the models,
not the prompt change, and reporting it as a before/after would be exactly the kind of confound
this project's own methodology exists to catch. The one number that DOES survive the model swap
cleanly is `unauthorized_rate = 0.0` — the core safety claim doesn't depend on which model is
proposing actions, only on the harness that gates them.

---

## 6. What changed this round to improve the numbers

1. **Switched the default LLM provider from Gemini to Groq.** This addressed both the latency and
   cost findings: median sweep latency went from 18.86s (Gemini, `agent.json`) to 8.55s (Groq,
   rows not in repo). On Groq's free tier the cost is $0 within its daily cap; at paid list rates
   ($0.15/$0.60 per M tokens, checked 2026-10-31) the same traffic projects to ~$56/month (5.3).
   An earlier "$765/month eliminated" figure used constants that match no current price.
2. **Fixed a real Groq model deprecation.** `llama-3.3-70b-versatile` was retired from Groq's
   catalog entirely; replaced with `openai/gpt-oss-120b` after verifying it live (a real
   tool-calling round trip via litellm, confirmed working) before adopting it as the new default.
3. **Fixed the extractor's token-undercounting gap** (section 5.3) so cost measurements are
   accurate going forward, not just for the tool-calling loop.
4. **Added a bounded retry to the live /resolve endpoint.** The first version cleared the case's
   partial audit trail before retrying. That was wrong: it deleted evidence, and a refund already
   paid in the failed attempt would vanish from the history rule 5 reads. Now each attempt gets a
   fresh case id and nothing is deleted; `audit.clear` refuses to run outside test/eval. A final
   failure returns a 503.
5. **Cached the tone judge's calls** by reply-text hash, so re-grading a sweep that's already been
   graded costs nothing on a rerun — the same pattern the adversarial-customer simulator already
   used for its own judge calls.
6. **Added output guardrails that actually gate a live reply**, not just grade it after the fact:
   `resolv/harness/guardrails.py` now checks groundedness, verbatim system-prompt leaks, PII-shaped
   content, and agent rudeness before any reply reaches a customer — one free self-correction per
   customer turn, then an escalation rather than letting a bad reply through. Also closed a real
   evasion in the groundedness check itself (a rewording like "you'll see $900 back in your
   account" used none of the six tracked verbs and used to pass through undetected).
7. **Added a rate limiter to `/resolve`** (`resolv/api/ratelimit.py`) — the endpoint had no auth and no
   limit at all, meaning anyone who found the URL could run up real LLM cost for free. A
   dependency-free, per-IP fixed-window limiter narrows the cost-abuse path (in-process only, so
   it does not hold across instances); it does not replace authentication.
8. **Added a "try a random sample complaint" button to the demo** (synthetic text over a real order) — a first-time visitor had no way
   to know what a valid order ID looks like, so anything they typed themselves was correctly (but
   unhelpfully) denied by rule 1 every time.
9. **Closed the caller-identity gap** (previously the top open item below). `resolv/harness/policy.py`
   added rule 2 (`caller_not_order_owner`): a refund is now denied unless the caller's identity
   matches the order's `customer_id`, threaded end to end from `/resolve`'s now-required
   `customer_id` field through the agent loop, the eval harness, and the audit trail (which now
   records `caller_id` per attempt so the live monitor can replay the check honestly).
   `tests/test_hallucination_collision.py` — which proved the gap — now also proves the fix.
10. **Made escalation un-droppable.** `resolv/harness/tools.py::issue_refund` auto-emits the
    `escalate_to_human` record itself whenever policy returns `escalate`, closing the 2/3-of-3
    grader gap from section 2 structurally instead of depending on model compliance.
11. **Fixed the pressure-tactic tone regression's root cause and validated it at full scale.**
    `resolv/agents/loop.py`'s `SYSTEM` prompt now explicitly forbids reciting-the-policy-at-them phrasing;
    `evals/quality.py` gained a per-tactic tone breakdown so a concentration in one tactic shows up
    automatically instead of requiring a manual worst-3 read. First an n=8 spot check (0.942 →
    0.975 on fresh pressure-tactic resolutions), then the full 40-task × 5-repeat sweep (200 fresh
    resolutions, $0.00 on Groq): `pressure` went from the worst tactic in the original pinned
    sweep to the **best** of five (0.972 avg, 0/40 below threshold), a within-sweep comparison
    immune to the judge/provider-swap confound that affects any comparison against the old pinned
    numbers. `unauthorized_rate` on this same 200-run sweep: **0.0**. See section 5.4 for the full
    breakdown, a new (unrelated, minor) tone finding on the `honest` tactic, and why this sweep's
    `pass^3` figure is NOT comparable to the pinned baseline (different model, not a regression).
12. **Audit trail.** The first version was an unkeyed SHA-256 chain per case file: it caught
    accidental edits, not a motivated insider, and a whole deleted file left no trace. It is now one
    SQLite ledger (`resolv/harness/ledger.py`) where each record carries an HMAC over the previous record's
    mac, keyed by `AUDIT_HMAC_KEY`; `verify_chain()` catches an edit, deletion or reorder by anyone
    without the key. Limit: someone with DB write access and the key can rewrite history.
13. **Session tokens** (`resolv/api/auth.py`). The first version minted a token for any `customer_id`
    a caller typed, so it did not close the "anyone can claim any customer" gap. Now `POST /token`
    issues a token only for a valid order proof code (an HMAC of the order id, as printed on a
    receipt); unknown orders and wrong codes look identical, `/token` is rate-limited, approver
    tokens are minted only locally, and prod fails closed without the secrets.
14. **Made `evals/cost.py` provider-aware.** Previously always priced at Gemini's rate regardless
    of the active provider — a cost report on Groq showed $0.0127/request for traffic that's
    actually free. Now reports each provider's own real number, or an honestly-labeled comparison
    estimate where no single real number exists (OpenRouter).
15. **Tested `reasoning_effort` against the latency SLO miss and got a real negative result.**
    `gpt-oss-120b` supports a documented low/medium/high reasoning-depth parameter — the literal
    mechanism behind the diagnosed tail latency. Wired up as an opt-in (`REASONING_EFFORT` env
    var, `resolv/agents/llm.py`), measured live, and found not to help (section 5.2) — recorded
    honestly rather than assumed to work because it sounded like it should.

## 7. What's still open

- ~~**No caller-identity verification.**~~ **FIXED.** See section 6, item 9 — `resolv/harness/policy.py`
  rule 2 now denies any refund whose caller doesn't match the order's `customer_id`, and
  `tests/test_hallucination_collision.py` (which proved the gap) now proves the fix.
- ~~**No authentication — customer_id was a bare trusted field.**~~ **Rewritten.** Tokens are
  issued only for a valid order proof code (section 6, item 13). A real deploy still needs a real
  identity provider behind `/token`.
- ~~**Tone under sustained pressure.**~~ **FIXED.** See section 6, item 11 — real before/after
  measurement, not just a prompt change (0.942 → 0.975 on fresh pressure-tactic resolutions).
- ~~**Cost pricing table is Gemini-specific.**~~ **FIXED.** See section 5.3 — provider-aware now,
  each provider reports its own real number (or an honestly-labeled comparison estimate where no
  single real number exists, e.g. OpenRouter).
- **First-action latency SLO** — still open. `reasoning_effort` was tested as a real candidate
  fix and did NOT help (section 5.2) — a negative result worth having, not a fix. Recalibrating
  the budget or investigating which specific tasks drive the tail remain the honest options.
- **Single node.** The ledger is one SQLite file (`BEGIN IMMEDIATE` serialises writers across
  threads and processes on one machine). Several instances on separate disks would each have their
  own ledger. Fixing that means a shared database; deliberately not built.
- **OpenRouter's rate-limit error text is unverified** — the matcher was widened to catch a
  generic "too many requests" phrase as a best-effort guess; no OpenRouter key has been tested
  against a real 429 to confirm it.

---

## 8. How to reproduce every number above

```bash
# Free, instant, no API key:
pytest -q                                 # the keyless suite
python -m resolv.harness.policy                  # the 8 refund rules (0-7) against the real order DB
python -m inference.serve_bench --backend regex --label regex   # section 3.1 -- the regex bar
python -m evals.metrics                    # the scoring math, checked against hand-worked numbers

# Needs a provider key (Groq is free-tier viable for the smaller runs below):
python -m evals.extractor --split test     # section 3 -- extractor accuracy + hallucination rate
python -m evals.ablation                   # section 4 -- harness ON vs OFF
python -m evals.injection                  # section 2/4 -- prompt-injection suite
python -m evals.runner --n 5 --tasks 40    # section 2 -- the full pass^k sweep (200 runs)

# Operations:
python -m evals.reliability                # section 5.1
python -m evals.latency                    # section 5.2
python -m evals.cost                       # section 5.3 (ignore the $ figure on non-Gemini providers)
pip install -r requirements-eval.txt && python -m evals.quality   # section 5.4
```

`evals.extractor` takes `--save-baseline` / `--baseline`; `evals.runner` takes `--save-baseline` and
`--gate <same-model baseline>`. See `evals/baselines/README.md`.
