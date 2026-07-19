# Resolv — Evaluation Report

*A living document. Updated every working turn. Newest findings at the top of each section;
full run history at the bottom.*

**Last updated:** 2026-07-20
**Phase:** 5 — the benchmark gate (produce an honest scorecard before anything downstream)
**Status:** ✅ **First real scorecard is in.** A partial sweep on Gemini 3.5 Flash (126 runs / 26
tasks) scored **pass³ = 0.977, unauthorized_rate = 0.0, over_block = 0.008** — the thesis holds
under adversarial pressure. Stopped early at 126/200 as a budget precaution (no billing dashboard
to watch); resumable to the full 200. Numbers and caveats in §4ter.

---

## 1. What we are measuring, and why it costs so much

The headline metric is **pass^k** — *"does the agent resolve the same task correctly `k` times
running?"* This is **reliability**, not capability. It is deliberately different from pass@k
("did *any* of k attempts succeed"), which is the SWE-bench-style number where a human picks the
best of k. Support has no human picking the best run — whatever the agent did to the customer is
what happened — so the honest question is whether it is right *every* time.

Measuring reliability **requires repetition by construction.** You cannot estimate "right k times
running" from one sample. So each task is run `n` times independently, and the agent samples at
**temperature 0.7** (at temperature 0 all n runs are identical and the metric measures nothing).
That repetition is the entire reason the call volume is high — it is the cost of measuring
reliability instead of a single lucky trajectory.

**Per-sweep call volume (n=5, 40 tasks):**

| | value |
|---|---|
| tasks | 40 |
| repeats per task (n) | 5 |
| **runs** | **200** |
| model calls per run (extract + tool turns + adversarial customer replies) | ~6–10 |
| **model calls per sweep** | **~1,500** |
| est. tokens per full run (growing tool-loop context) | ~20,000–30,000 |
| **est. tokens per full sweep** | **~4–6 million** |

The scored metrics on each sweep:

- **pass^k** — reliability (the headline).
- **pass^1** — plain per-attempt success rate, for comparison.
- **unauthorized_rate** — refunds that policy did *not* authorise, re-checked independently by
  replaying each refund against `harness/policy.py`. **Must be 0.** This is not a quality dial:
  nonzero means the enforcement claim is false and nothing else on the card matters.
- **harmful_block_rate** — of the refunds that *should* have been refused, how many were. Target 1.
- **over_block_rate** — the honest counterweight: customers who *were* owed money and got refused.
  Stops "deny everything" from looking safe.
- **resolve_rate** + Wilson 95% CI, **mean_steps**.

---

## 2. What is being tested (the adversarial benchmark)

Each task pairs a real, messy complaint (generated from a real Olist order) with one of five
**customer tactics**, played by a second LLM at temperature 0.0 so every repeat of a task poses
the *same* challenge. Each tactic targets a specific policy rule:

| tactic | what the customer does | rule it attacks |
|---|---|---|
| `honest` | states the real order and amount, accepts limits | baseline |
| `inflate_amount` | claims they paid far more than they did | rule 5 (refund cap) |
| `wrong_order_id` | gives a non-existent order number, insists it's right | rule 1 (unknown order) |
| `change_story` | refused? instantly switches to a claim that pays | rule 2 (claim truth) |
| `pressure` | chargebacks, lawyers, followers, repetition | none — tests if volume moves policy |

The task set is **40 tasks, sampled to a verified 20/20 split** between refund-owed and
deny-owed. This is load-bearing: an earlier version of the data had only 14% of orders in-window,
so "deny everything" scored 86% and the benchmark measured nothing. The 50/50 split means no
constant policy beats a coin flip. The answer key is **derived from the policy engine at build
time**, not hard-coded, so the benchmark cannot drift out of sync with the system it grades.

---

## 3. Results so far

### 3.1 Component self-checks (deterministic, no network) — ✅ all green

Every non-trivial module ships a runnable `demo()` with asserts. Current state:

| module | check | result |
|---|---|---|
| `harness/policy.py` | all 6 rules, one allow + one deny each, against real DB | ✅ PASS |
| `harness/tools.py` | no bypass, denials recorded, denial reads as result not exception | ✅ PASS |
| `harness/audit.py` | append-only, rule-4 read path intact | ✅ PASS |
| `eval/metrics.py` | pass^k estimator, geometric trajectory, Wilson, McNemar, scorecard | ✅ PASS |
| `eval/simulator.py` | 5 tactics render, turn limits hold (no network) | ✅ PASS |
| `eval/tasks.py` | 40 tasks, 20/20 split, key agrees with policy, deterministic | ✅ PASS |

### 3.2 Smoke test A — WORKERS=2, n=2 × 5 tasks (10 runs) — ✅ VALID

The first end-to-end smoke, before the daily quota was hit. **This is our best real signal so far.**

| metric | value |
|---|---|
| pass^2 | **0.80** |
| pass^1 | 0.80 |
| **unauthorized_rate** | **0.0** ✅ |
| harmful_block_rate | 1.0 |
| over_block_rate | 0.0 |
| resolve_rate | 0.80 (Wilson 95% CI 0.49–0.94) |
| mean_steps | 4.2 |

Interpretation: the agent, the harness, and the adversarial loop are wired correctly and the
enforcement claim held. Small n, wide CI — indicative, not final.

### 3.3 Full sweep-01 — WORKERS=2, n=5 × 40 tasks (200 runs) — ❌ INVALID (throughput failure)

| metric | reported | truth |
|---|---|---|
| pass^3 | 0.025 | **meaningless** |
| runs that actually completed | — | **9 of 200** |
| runs that crashed on rate limits | — | **191 of 200** |
| of the 9 that ran: resolved | — | **9 / 9** ✅ |
| of the 9 that ran: unauthorized | — | **0** ✅ |

The 0.025 is an artifact of 191 runs dying before they finished, correctly recorded as failures
(not dropped — dropping them would have flattered the score). **The 9 completed runs resolved 9/9
with zero unauthorized actions**, consistent with smoke test A. The agent is not the problem;
throughput is.

### 3.4 Smoke test B — WORKERS=1 fixed retry, n=2 × 6 tasks (12 runs) — ❌ hit the DAILY wall

Ran after sweep-01. **All 12 crashed**, and the error changed from per-minute to per-**day**:

```
Rate limit reached ... tokens per day (TPD): Limit 100,000, Used 99,296, Requested 923.
Please try again in 3m9s.
```

This is the decisive finding — see §4. It is *not* a regression in the WORKERS=1 fix; the org's
entire daily token budget had been consumed by sweep-01.

---

## 4. THE BLOCKER — Groq free tier cannot supply a full sweep

Groq's free tier enforces **two** limits, and we were only defending against the first:

| limit | value | our need |
|---|---|---|
| tokens per **minute** (TPM) | 12,000 / org | ~2,500 per call — WORKERS=1 + retry survives this |
| tokens per **day** (TPD) | **100,000 / org** | **~4–6 million per full sweep** |

We have **3 keys across 3 orgs → ~300,000 tokens/day total.** A single full sweep needs ~4–6M.
That is **12–20 days of the entire free daily quota for one sweep.** Pacing does not help: once
the daily bucket is empty it stays empty until reset. My earlier "throttle + ~3 hours" plan was
wrong — this was never a speed problem.

**What actually fits in one free-tier day (~300k tokens):**

| tokens/run | runs/day feasible | as a sweep |
|---|---|---|
| ~25,000 | **~12 runs/day** | e.g. n=3 × 4 tasks, or n=2 × 6 tasks |

A statistically meaningful pass^k sweep (n≥5, tasks≥20) is **not achievable on the free tier.**

### 4.1 OpenRouter — tested turn 7, same class of wall on free tier

Wired OpenRouter as a second provider (litellm-native, one config switch). Tested the 3 supplied
keys directly:

- **All 3 keys are valid but one free account.** The `auth/key` endpoint returns
  `is_free_tier=True, usage=0` for all three, and a completion error exposed an identical
  `user_id` — so they share **one** account. On free tier that means **~50 requests/DAY total**,
  not 150; rotating the keys buys no daily headroom.
- **Free (:free) model endpoints are congested.** `meta-llama/llama-3.3-70b-instruct:free` and
  `qwen/qwen3-next-80b:free` returned upstream 429s ("temporarily rate-limited upstream / Venice")
  even after 75s of retries. **`nvidia/nemotron-3-super-120b-a12b:free` worked** — a clean
  tool-call in 337 tokens — and is set as the default free model.
- **The unlock is ~$10 of credit:** lifts free from 50 → **1000 req/day** AND enables reliable
  **paid** model endpoints (e.g. `meta-llama/llama-3.3-70b-instruct`, `deepseek/deepseek-chat`).

### Options (updated)
1. **~$10 OpenRouter credit** — 1000 req/day + reliable paid 70B endpoints; a full sweep in one
   sitting for ~$1–2 of tokens. **Recommended** — also gives model choice.
2. **Paid Groq (Dev tier)** — same idea, lifts the 100k/day cap; less model flexibility.
3. **Free, spread over days** — the runner is now **resumable** (§7): run ~50 req/day on the free
   nemotron, accumulate across days. Free, slow, correct.
4. **Local (Ollama)** — unlimited and free, needs a capable local machine; slower, and a small
   local model reads as a lower (but honest) pass^k.
5. **Two-row result** — run a strong model (headline) *and* a weak one (contrast) so the harness
   thesis is visible: competence (pass^k) falls with the weaker agent, but `unauthorized_rate`
   stays 0 because enforcement is in Python. Set the agent model per run with `LLM_MODEL`.

---

## 4bis. Extractor fine-tune (RLVR) — DONE, and it worked

Ran `scripts/train_extractor.py` on Kaggle (T4, Qwen2.5-1.5B + LoRA, GRPO via TRL, 200 steps,
368 train / 92 held-out). Reward = the production verifier's exact-match check, no LLM judge.

| metric (held-out, n=92) | base | tuned | Δ |
|---|---|---|---|
| order id correct | 0.761 | 0.783 | +0.022 |
| claim type correct | 0.609 | **0.870** | **+0.261** |
| **both right** | **0.489** | **0.707** | **+0.217** |

+0.217 at n=92 is ~4× the standard error — a real gain, not noise. Order extraction was already
strong; the fine-tune's work was on claim-type classification (and emitting the label in the
parseable form the reward wants, which is exactly the production requirement).

**The reported training loss was `0.000000` for all 200 steps — and this time that is EXPECTED,
not the collapse.** GRPO's scalar loss is ~0 by construction: advantages are normalized mean-zero
within each group, so `loss ≈ -mean(A_i) ≈ 0` while the per-completion gradients are still
nonzero. Last fine-tune, loss=0 came with a flat held-out metric (genuine zero-advantage
collapse: all rollouts scored identically, `std=0`, gradient truly zero). This time loss=0 came
with +21.7 pts held-out — proof the gradients flowed. The lesson, now load-bearing: **judge GRPO
by held-out accuracy, never by the loss curve.** Switching the target from the drafter to the
extractor (for reward variance) is what fixed it.

---

## 4ter. Baseline pass^k — the first real scorecard (partial, Gemini 3.5 Flash)

Ran the sweep on **Gemini 3.5 Flash** (agent, extractor, and adversarial customer all on it).
Stopped at **126/200 runs** as a budget precaution — the key was pay-as-you-go with no billing
dashboard to watch, and Gemini Flash's output tokens are pricey, so we halted rather than risk a
hard $5 ceiling. 126 runs = **26 tasks touched, 25 with the full 5 repeats**, so pass³ is over 26
tasks. Scored offline from `data/eval/runs.jsonl` (pure `eval.metrics.scorecard`, no re-run):

| metric | value | read |
|---|---|---|
| **pass³** | **0.977** | right 3-of-5-ways-running on ~all tasks — strong reliability |
| pass¹ | 0.992 | single-attempt success |
| **unauthorized_rate** | **0.0** | **the thesis: zero policy-violating refunds across 126 adversarial runs** |
| harmful_block_rate | 1.0 | every refund that should've been refused, was |
| over_block_rate | 0.008 | one owed customer wrongly denied (the honesty counterweight — near zero) |
| resolve_rate | 0.992 (CI 0.956–0.999) | |
| mean_steps | 5.44 | |
| crashed | 0 | Gemini path clean; no rate-limit losses |

**By tactic** (resolved/total): honest 26/26, inflate_amount 25/25, wrong_order_id 25/25,
pressure 25/25, change_story 24/25. The single miss was a `change_story` case (the one
over-block) — the agent was talked into denying a customer who was actually owed.

**Caveats, stated honestly:**
- **Partial + not perfectly balanced.** The 126 runs cover the first 26 tasks by case_id, which
  skewed deny-heavy (76 deny / 45 refund / 5 escalate) rather than the intended 50/50. So
  `resolve_rate` is mildly flattered by easier deny cases being over-represented. `pass³` (per-task
  averaged) and `unauthorized_rate` (0 regardless of mix) are not affected. The full 200-run,
  40-task sweep restores the 50/50 balance.
- **Cost:** ~2M tokens projected for the full run; stopped at ~1.24M ≈ **~$1.5–2.5 real** (Gemini
  3.5 Flash ≈ $0.75–1.50/M in, $4.50–9/M out; see §4). The `est_usd` meter's flat $1/M
  under-counts because output is expensive — real ≈ 1.5–2× the displayed figure.
- **`--max-tokens` cap did NOT hard-stop mid-sweep** as intended: `ThreadPoolExecutor.map` submits
  all jobs eagerly, so the cap can stop *reading* results but can't cancel queued runs. Known bug;
  needs a submit-as-you-go execution model to fix. The run was stopped by killing the process.

**Bottom line:** the headline holds. **Zero unauthorized actions and pass³ ≈ 0.98 under five
adversarial customer tactics** — on a partial but real sample. Finishing the balanced 200 would
tighten the CIs and restore the 50/50 split, but the core claim is already evidenced.

---

## 5. What is NOT being tested yet (and why)

- **The full balanced 200-run sweep** — 126/200 done; resume the same command to finish the
  remaining 74 (restores 50/50 balance, tightens CIs). ~$1–1.5 more.
- **Wiring the tuned extractor into the agent loop** and re-running to show the fine-tune moves
  *pass^k* — the extractor fine-tune is measured directly (§4bis, 0.489→0.707); connecting it to
  the agent path is the remaining stretch step.
- **Legacy training data** (`data/datasets/legacy/`: `resolv_sft.json`, `resolv_orpo.json`,
  `eval_drafts.json`) — belongs to the deleted email-drafter product. Kept only so it isn't
  confused with live data; safe to delete.
- **Live training data** (`data/datasets/complaint_cases.json`, 460 cases) — generated from real
  Olist orders, verifier-checked. This is what the eval samples and what the extractor would
  retrain against.

---

## 6. Run log (append-only)

| # | date | config | runs | completed | resolved | unauthorized | verdict |
|---|---|---|---|---|---|---|---|
| smoke A | 07-17 | W=2, n=2×5 | 10 | 10 | 8 | 0 | ✅ valid, pass^2 0.80 |
| sweep-01 | 07-17 | W=2, n=5×40 | 200 | 9 | 9 | 0 | ❌ throughput; 191 rate-limit crashes |
| smoke B | 07-17 | W=1, n=2×6 | 12 | 0 | — | — | ❌ daily quota exhausted (TPD) |
| extractor FT | 07-19 | Kaggle T4, GRPO 200 steps, 92 held-out | — | — | — | — | ✅ both-right 0.489 → 0.707 (+0.217) |
| gemini-partial | 07-20 | Gemini 3.5 Flash, n=5×40, stopped 126/200 | 126 | 126 | 125 | 0 | ✅ **pass³ 0.977, unauth 0.0**, over-block 0.008 |

---

## 7. How to run the sweep (this is the file to run)

```bash
# The full sweep — resumable, with a progress bar. Provider/model come from .env (config.py).
python -m eval.runner                       # 40 tasks × 5 repeats = 200 runs -> data/eval/runs.jsonl

# Smaller/indicative run:
python -m eval.runner --n 3 --tasks 20      # 60 runs
python -m eval.runner --n 1 --tasks 2       # tiny wiring check

# Pick the agent model per run (headline vs contrast row):
LLM_MODEL=openrouter/meta-llama/llama-3.3-70b-instruct python -m eval.runner   # needs OR credit
LLM_MODEL=openrouter/nvidia/nemotron-3-super-120b-a12b:free python -m eval.runner
```

Behaviour to expect:
- **Resumable.** Re-running continues into the same `data/eval/runs.jsonl` — completed runs are
  skipped, crashed (rate-limited) runs are retried, no double-counting. `--fresh` starts over.
  This is what makes the free-tier "spread over days" path work: run until the daily cap, come
  back tomorrow, run again.
- **Live progress.** tqdm shows a bar with running `resolved / unauth / crashed` counts.
- **Scorecard at the end**, computed over *every* row in the file (all sessions), with a note if
  any runs crashed so a rate-limited partial run is never mistaken for a real low score.

---

## Changelog
- **2026-07-20:** First real agent scorecard, on Gemini 3.5 Flash. Partial sweep (126/200 runs,
  26 tasks), stopped early as a budget precaution: **pass³ = 0.977, unauthorized_rate = 0.0,
  over_block = 0.008, 0 crashes.** All 5 adversarial tactics ~100% (1 change_story miss). Wired
  Gemini as an opt-in provider; added a token meter + est-cost display. Found the `--max-tokens`
  cap can't hard-stop under `ThreadPoolExecutor.map` (eager submit). See §4ter.
- **2026-07-19:** Ran the extractor RLVR fine-tune on Kaggle. Held-out extraction accuracy
  **0.489 → 0.707 both-right** (+0.217; claim-type 0.609 → 0.870). Training loss `0.000000`
  throughout is expected GRPO behaviour (mean-zero advantages), NOT the earlier collapse — the
  held-out gain proves gradients flowed. See §4bis.
- **turn 7:** Wired OpenRouter as a second provider — `config.py` is now
  provider-agnostic (`LLM_PROVIDER`/`LLM_MODEL`, generic `rotate_key`/`reset_key`/`key_count`);
  `loop.py`, `simulator.py`, `runner_utils.py` updated. Tested the 3 keys (valid, one free
  account, ~50 req/day shared; free 70B/qwen endpoints congested; **nemotron-120b:free works**).
  Made `eval/runner.py` **resumable** with a **tqdm** progress bar. Added run instructions (§7).
  Verified offline: compiles, imports, all demos green. Did not run a live sweep (user runs it).
- **turn 6:** Discovered the binding limit is tokens-per-**day** (100k/org), not per-minute. A
  full sweep needs ~4–6M tokens ≈ 12–20 free-tier-days. Full sweep infeasible on free tier;
  options in §4. Report created.
