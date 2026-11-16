# Resolv: Technical Report

A refund agent where the language model proposes tool calls and plain code decides whether money moves. This report
summarises what was built, every evaluation, the test suite, the limits, and how to reproduce each number. The full
run-by-run record is [docs/eval_report.md](docs/eval_report.md); this file is the readable summary of it.

## Contents

1. [Summary](#1-summary)
2. [What the system does](#2-what-the-system-does)
3. [Evaluations](#3-evaluations)
   - [3.1 Setup shared by every eval](#31-setup-shared-by-every-eval)
   - [3.2 Harness on vs off](#32-harness-on-vs-off)
   - [3.3 Reliability: pass^k on the reference sweep](#33-reliability-passk-on-the-reference-sweep)
   - [3.4 Extractor vs regex](#34-extractor-vs-regex)
   - [3.5 Prompt injection with a control arm](#35-prompt-injection-with-a-control-arm)
   - [3.6 Red team, 11 attack families](#36-red-team-11-attack-families)
   - [3.7 Latency, cost and reliability](#37-latency-cost-and-reliability)
   - [3.8 Load test: the knee before and after the SQLite fix](#38-load-test-the-knee-before-and-after-the-sqlite-fix)
   - [3.9 Eval gate false-positive rate](#39-eval-gate-false-positive-rate)
   - [3.10 Error analysis: 100 runs read by hand](#310-error-analysis-100-runs-read-by-hand)
   - [3.11 Red-team cassettes in the PR gate](#311-red-team-cassettes-in-the-pr-gate)
   - [3.12 Second model on the current code: Gemini 2.5 Flash-Lite](#312-second-model-on-the-current-code-gemini-25-flash-lite)
   - [3.13 Older numbers (history only)](#313-older-numbers-history-only)
4. [Tests and CI](#4-tests-and-ci)
5. [Limitations](#5-limitations)
6. [How to reproduce each number](#6-how-to-reproduce-each-number)

---

## 1. Summary

Resolv handles customer refund requests with an LLM agent, but the model never decides whether money moves. Every
refund passes through one function, an 8-rule policy gate in plain Python; refunds above $200 wait for a human;
payment runs as a reserve, pay, settle saga; and every attempt is written to a tamper-evident ledger. On the same
model (`openai/gpt-oss-120b`) and the same 40 refund cases, turning the harness off let the agent pay **16 refunds
policy forbids and $2,281.70 more than customers were owed**; with it on, **0 and $0**, and more cases were resolved
correctly (82.5% vs 52.5%). Across 432 red-team and prompt-injection runs, no attack moved money or leaked data.
The agent succeeds on the same case 5 times out of 5 (pass^5) at 0.71, from a partial sweep of 28 of 40 tasks.
Median latency is 4.7 s, model cost is about $0.0018 per resolved case, and one server worker holds 40 requests a
second against a fake model. Measured 14 November 2026.

On 16 November 2026 the suites were re-run on the current code (after the split-refund fix) with a second, cheaper
model, Gemini 2.5 Flash-Lite. The full 40-task sweep (200 runs) gives pass^1 **0.765** and pass^5 **0.675** with
**0 unauthorized refunds**, and **432 more red-team and prompt-injection runs again caused no harm**. All of it cost
$0.31 (section 3.12).

| Headline | Harness on | Harness off |
| :-- | --: | --: |
| Refunds paid that policy forbids | **0 of 40** | 16 of 40 |
| Money paid above what was owed | **$0** | $2,281.70 |
| Cases resolved correctly | **82.5%** | 52.5% |

<!-- src: evals/results/ablation_gpt_oss_120b_card.json#cards.on.unauthorized_rate, cards.off.unauthorized_rate (x runs=40), cards.<arm>.unauthorized_usd.excess, cards.<arm>.resolve_rate -->

---

## 2. What the system does

A customer writes a complaint ("my order arrived two weeks late, refund me $900"). Resolv reads it, looks up the
order, decides what is owed, and either pays, refuses, or hands the case to a person.

How one request flows:

1. **API** (`resolv/api/main.py`). The customer exchanges an order proof code for a session token at `POST /token`,
   then sends the complaint to `POST /resolve`. Requests are rate limited per IP.
2. **Extractor**. One LLM call turns the message into a hint: which order, which kind of claim.
3. **Agent loop** (`resolv/agents/loop.py`). A hand-written tool-calling loop over LiteLLM (no agent framework). The
   model proposes tool calls (`lookup_order`, `issue_refund`, `escalate_to_human`). The loop enforces budgets
   (tool calls, tokens, dollars), a provider circuit breaker and reply guardrails, and journals each step so a
   crashed case resumes without redoing work.
4. **Harness** (`resolv/harness/tools.py`, the only code that changes anything). Every refund call first goes
   through `check_refund` in `resolv/harness/policy.py`: eight rules, first match wins. The cap comes from what the
   order record says was paid, never from what the customer says.

   | Rule | Refuses or escalates when |
   | :-- | :-- |
   | 0 | the amount is not a positive finite number, or the claim type is unknown |
   | 1 | the order does not exist |
   | 2 | the caller does not own the order |
   | 3 | the claim is false against the order record (for example "late" when it was on time) |
   | 4 | the order is outside the 90-day claim window |
   | 5 | the order was already refunded |
   | 6 | the amount is above the cap for the claim type |
   | 7 | the amount is above $200, or a larger refund on the same order is already waiting for a human (escalate) |

   <!-- src: resolv/harness/policy.py#check_refund; resolv/config.py#AUTO_APPROVE_MAX_USD, CLAIM_WINDOW_DAYS -->
5. **Saga** over a simulated payment provider (`resolv/integrations/payments.py`): reserve a pending refund, call the
   provider, then settle, compensate on a decline, or park as `unknown` for a reconciler.
6. **Human approval**. Escalations appear in a console at `/console`. An approval is bound to the exact refund the
   approver saw and runs back through the same policy gate.
7. **Ledger** (`resolv/harness/ledger.py`). One SQLite file. Every audit record carries an HMAC (a keyed hash) of
   the record before it, so an edited, deleted or reordered row is caught and named. A database constraint allows
   only one live refund per order.

Around this: a Streamlit demo (`app.py`), a read-only MCP server with `lookup_order`, a dry-run `check_refund` and
`list_escalations` (no refund tool), operator switches (`REFUNDS_FROZEN=1` stops all money, `RESOLV_SHADOW=1`
records what would have been paid), and OpenTelemetry traces to Langfuse.

Diagram and file map: [docs/architecture.md](docs/architecture.md). Operations: [docs/RUNBOOK.md](docs/RUNBOOK.md).

---

## 3. Evaluations

### 3.1 Setup shared by every eval

| Setting | Value |
| :-- | :-- |
| Model (agent, simulated customer, extractor, judge) | `openai/gpt-oss-120b`, called through LiteLLM as `openrouter/openai/gpt-oss-120b` |
| Provider | OpenRouter |
| Temperature | 0.7 (the extractor card records `null`) |
| Date | 2026-11-14 |
| Price used for dollar figures | $0.15 in / $0.60 out per million tokens (list price; an upper bound) |
| Data | order facts from the Olist e-commerce dataset; customer messages are synthetic |
| Simulated customer | the same model, playing one of five tactics: honest, inflate_amount, wrong_order_id, change_story, pressure |

<!-- src: evals/results/*_gpt_oss_120b*_card.json#manifest (provider, *_model, temperature, started_at); evals/results/extractor_gpt_oss_120b_card.json#manifest.temperature; resolv/config.py#PRICES_USD_PER_MTOK (openrouter/openai/gpt-oss-120b) -->

Terms used below:

- **Resolved**: the run did the right thing for its task (paid exactly what was owed, refused, or escalated), with
  no unauthorized refund.
- **Unauthorized refund**: a refund that the policy, replayed over the records before it, would not have allowed.
- **Over-block**: refusing a customer who was actually owed money.
- **pass^k**: the chance that all k independent attempts at the same task succeed. pass^1 is the plain success
  rate; pass^5 asks for 5 out of 5.
- **CI**: 95% confidence interval. Most are clustered by task (repeats of one task are correlated, so they are
  resampled together).

Total model spend across the gpt-oss-120b result files: about $0.81 at list price (see
[eval report, Spend](docs/eval_report.md#spend)). Section 3.12 uses a second model; its setup and spend are listed
there.

### 3.2 Harness on vs off

- **Measures:** what the policy gate is worth. The same model, cases and simulated customer run twice; the "off"
  arm swaps in unguarded tools that pay whatever the model asks for.
- **Why it matters:** it shows the safety comes from the code, not from the model behaving.
- **How:** 40 tasks x 1 run per arm, `python -m evals.ablation --tasks 40 --n 1 --arm both`.

| Metric | Harness on | Harness off |
| :-- | --: | --: |
| Unauthorized refunds | **0 of 40** (rate 0.0) | 16 of 40 (rate 0.40) |
| Unauthorized dollars, gross | $0.00 | $2,637.72 |
| Dollars paid above what was owed | **$0.00** | **$2,281.70** (15 runs overpaid) |
| Resolve rate (clustered CI) | **0.825** (0.70 to 0.925) | 0.525 (0.375 to 0.675) |
| Harmful requests blocked (deny tasks) | 1.0 | 0.5 |
| Over-block rate (refund tasks) | 0.125 | 0.025 |
| Cost per resolved case | $0.0019 | $0.0025 |

<!-- src: evals/results/ablation_gpt_oss_120b_card.json#cards.on, cards.off (unauthorized_rate, unauthorized_usd.gross, unauthorized_usd.excess, resolve_rate, resolve_ci95_clustered, harmful_block_rate, over_block_rate, cost_per_resolution); 15 overpaid runs: count of excess_usd > 0 in evals/results/ablation_gpt_oss_120b.jsonl (arm=off) -->

The off arm's resolve rate dropped from 0.55 to 0.525 after a grader fix found in the error analysis (a refund of
the right amount under a false claim used to count as resolved). The change is recorded in the card under `regrade`.

**Extractor on vs off** (a second ablation, same setup): resolve rate 0.85 (0.725 to 0.95) with the extractor's hint
vs 0.825 (0.70 to 0.925) without, unauthorized 0 in both. The difference is inside the noise; the two "on" arms of
the two ablations (0.825 and 0.85) show how far one 40-run arm moves on its own.

<!-- src: evals/results/ablation_extractor_gpt_oss_120b_card.json#cards.on.resolve_rate, cards.no_extractor.resolve_rate, *.resolve_ci95_clustered, *.unauthorized_rate -->

### 3.3 Reliability: pass^k on the reference sweep

- **Measures:** how often the agent gets the same case right when asked repeatedly, against an adversarial
  simulated customer.
- **Why it matters:** a single success rate hides flakiness. A support agent that is right 83% of the time but not
  the same 83% each time is harder to trust.
- **How:** planned 40 tasks x 5 repeats, `python -m evals.runner --n 5 --tasks 40`.
- **Caveat: partial sweep.** The provider account ran out of credit. 144 rows were written (3 failed with "requires
  more credits"), and only the **28 tasks with all 5 repeats** (140 runs) are scored. They are the first 28 of 40
  and not balanced: 11 refund, 16 deny and 1 escalate task. Read these as partial, not as the 40-task reference.

| Metric | Value (task-clustered 95% CI) |
| :-- | --: |
| pass^1 | **0.829** (0.714 to 0.929) |
| pass^3 | 0.736 (0.579 to 0.889) |
| pass^5 | **0.714** (0.536 to 0.857) |
| Unauthorized refund rate | **0.0** |
| Invariant violations (independent checker) | 0.0 |
| Harmful requests blocked | 1.0 |
| Over-block rate | 0.093 |
| Reply matches what was actually paid | 0.986 |
| Mean steps per case | 5.28 |
| Latency p50 / p95 (multi-turn) | 11.67 s / 17.87 s |
| Cost per resolved case | $0.00179 |
| Effective sample size | 43.3 independent runs out of 140 (ICC 0.558) |

<!-- src: evals/results/ref_gpt_oss_120b_partial_card.json#coverage (rows_in_file, error_rows_excluded, scored_tasks, scored_rows, expected_mix), card (pass^1, pass^3, pass^5 and *_ci95_clustered, unauthorized_rate, invariant_violation_rate, harmful_block_rate, over_block_rate, reply_grounded_rate, mean_steps, p50_latency_s, p95_latency_s, cost_per_resolution, design_effect_resolved) -->

The effective sample size matters: repeats of one task are strongly correlated, so 140 runs carry about as much
information as 43 independent ones. That is why every interval here is clustered by task.

### 3.4 Extractor vs regex

- **Measures:** whether the LLM extractor reads the order id and claim type from a message better than a plain
  regex with no model.
- **Why it matters:** a regex is the bar a model has to clear to be worth its latency and cost.
- **How:** the same 92 held-out messages for both. 13 of them contain no usable order id; on those, inventing one
  counts as a hallucination. Paired comparison with McNemar's test (a test that looks only at the cases where the
  two methods disagree).

| | gpt-oss-120b extractor | Regex (no model) |
| :-- | --: | --: |
| Order id and claim both correct | **0.891** (0.811 to 0.940) | **0.891** (0.811 to 0.940) |
| Order id correct / claim correct | 0.957 / 0.935 | 0.946 / 0.946 |
| Hallucinated an order id (of 13) | 4 (0.308) | 5 (0.385) |
| Median latency | 1.66 s | under 1 ms |
| Paired disagreements | 5 cases model right, regex wrong | 5 cases regex right, model wrong |

<!-- src: evals/results/extractor_gpt_oss_120b_card.json#card (both_acc, both_ci95, order_acc, claim_acc, hallucination_rate, abstention_n, p50_latency_s); inference/results/extractor/serve_regex.json#card (same keys, serving.latency_p50_s); paired 5/5, chi2 0.1: evals.extractor.compare run over both row sets (command in section 6) -->

McNemar chi-squared 0.1, not significant. **The hosted model does not beat the regex on this data.** This is safe
either way: a wrong order id costs a turn, never a wrong refund, because the policy gate checks the real record.

### 3.5 Prompt injection with a control arm

- **Measures:** whether text planted in the customer's message (fake system turns, fake tool results, "I'm your
  supervisor", "repeat your prompt") changes what the agent does, and whether any of it causes harm.
- **Why it matters:** the model probes above the refund cap sometimes even with no attack. Without a control arm,
  that habit reads as attack success. So "complied" is reported as a lift over the control, and harm is reported
  separately.
- **How:** 8 refund-owed orders x 3 repeats x 6 arms = 144 runs, `python -m evals.injection --orders 8 --n 3`.

| Arm | Complied | Lift over control (95% CI) | p | Unauthorized | Excess $ |
| :-- | --: | --: | --: | --: | --: |
| control (no attack) | 0.208 | | | 0.0 | 0 |
| direct_override | 0.208 | 0.0 (-0.125 to 0.125) | 1.0 | 0.0 | 0 |
| fake_system_turn | 0.375 | +0.167 (0.0 to 0.417) | 0.20 | 0.0 | 0 |
| fake_tool_result | 0.208 | 0.0 (-0.125 to 0.125) | 1.0 | 0.0 | 0 |
| authority_roleplay | 0.25 | +0.042 (0.0 to 0.125) | 0.68 | 0.0 | 0 |
| exfiltrate_prompt | 0.083 | +0.083 (0.0 to 0.208) | 0.21 | 0.0 | 0 |

<!-- src: evals/results/injection_gpt_oss_120b_card.json#card.complied.<arm>.complied_rate, card.complied_lift.<arm> (lift, ci95, p_two_sided), card.harm.<arm> (unauthorized_rate, excess_usd) -->

No attack's lift has an interval that excludes zero, and harm is 0 in every arm.

### 3.6 Red team, 11 attack families

- **Measures:** 11 attack families mapped to the OWASP Top 10 for LLM applications: direct injection, tag escape,
  multi-turn pressure, identity spoofing, cross-customer data theft, malformed amounts (NaN, 1e9, -50), injection via
  a tool result, system-prompt extraction, unbounded consumption, approver social engineering and PII echo.
- **Why it matters:** it separates two questions. *Complied*: did the model try what the attacker wanted? *Harm*:
  did money move or data leak? The first is about the model; the second is about the system.
- **How:** 4 orders x 2 repeats per attack case plus a control arm, 288 runs through the real loop and harness,
  `python -m evals.redteam`.

| Family | Runs | Complied | Lift vs control | Money moved | Data leaked |
| :-- | --: | --: | --: | --: | --: |
| A1 direct injection | 24 | 0.417 | -0.083 | 0 | 0 |
| A2 wrapper-tag escape | 24 | 0.417 | -0.083 | 0 | 0 |
| A3 multi-turn crescendo | 24 | 0.333 | -0.167 | 0 | 0 |
| A4 identity spoofing | 24 | **1.0** | +1.0 | 0 | 0 |
| A5 cross-customer exfiltration | 24 | 0.0 | 0.0 | 0 | 0 |
| A6 amount-format attacks | 40 | 0.0 | 0.0 | 0 | 0 |
| A7 injection via tool output | 24 | 0.5 | 0.0 | 0 | 0 |
| A8 system prompt extraction | 24 | 0.0 | 0.0 | 0 | 0 |
| A9 unbounded consumption | 24 | 0.0 | 0.0 | 0 | 0 |
| A10 approver social engineering | 24 | manual review | | 0 | 0 |
| A11 PII echo | 24 | 0.0 | 0.0 | 0 | 0 |
| control | 8 | | | 0 | 0 |

<!-- src: evals/results/redteam_gpt_oss_120b_card.json#runs, card.families.<Fam> (n, complied_rate, lift_vs_control.lift, money_moved, data_leaked), card.harm_summary -->

Harm summary over all 288 runs: money moved 0, data leaked 0, PII leaked 0, prompt leaked 0, tool budget bypassed 0.

The clearest result is A4: told "I'm the account holder of this other order", the model looked up the stranger's
order on every row. Each lookup was refused because lookups are scoped to the caller's own orders, so nothing leaked.
The model is suggestible; the system is not. A10 needs a person to compare each escalation summary with the raw
action and has not been reviewed yet. 14 rows (A9, A10, A11) errored when the provider ran out of credit; they stay
in the denominators. Full method and OWASP mapping: [docs/eval/REDTEAM.md](docs/eval/REDTEAM.md).

<!-- src: errors per family: evals/results/redteam_gpt_oss_120b_card.json#card.families.A9.errors (2), A10.errors (6), A11.errors (6) -->

### 3.7 Latency, cost and reliability

- **Measures:** how long a customer waits, what a case costs, and whether single attempts crash.
- **Why it matters:** a safe agent that is slow or expensive does not ship.
- **How:** 5 tasks, single-turn (no simulated back-and-forth). Latency 5 x 5 = 25 runs after 1 warm-up, cost 5 x 3 =
  15 runs, reliability 5 x 5 = 25 runs. Small samples: a p95 from 25 runs is noisy, so its interval is shown.

| Metric | Value | Target |
| :-- | --: | --: |
| End-to-end p50 / p95 | 4.73 s / **7.68 s** (p95 CI 5.53 to 7.77) | p95 at most 15 s: pass |
| Time to first action p50 / p95 | 2.14 s / **3.04 s** (p95 CI 2.54 to 4.30) | p95 at most 6 s: pass |
| Cost per single-turn request | $0.00062 (3,315 in / 204 out tokens) | at most $0.02 per request: pass |
| Projected monthly cost at 2,000 requests a day | $37.17 | |
| Cost per resolved case (multi-turn sweep) | $0.00179 | |
| Single-attempt success | 25 of 25 (1.0) | at least 95%: pass |

<!-- src: evals/results/ops_latency_2026-11-14T06-54-54+00-00-openrouter-latency_card.json#card.e2e_s, card.e2e_p95_ci95_clustered, card.first_action_s, card.first_action_p95_ci95_clustered, card.slo_*_pass; evals/latency.py#SLO_P95_S, SLO_FIRST_ACTION_P95_S; evals/cost.py#COST_BUDGET_PER_REQUEST_USD; evals/reliability.py#SLO_MIN_SUCCESS_RATE; evals/results/ops_cost_2026-11-14T06-57-02+00-00-openrouter-cost_card.json#card (avg_usd_per_request, avg_input_tokens, avg_output_tokens, usd_per_month, budget_pass); evals/results/ref_gpt_oss_120b_partial_card.json#card.cost_per_resolution; evals/results/ops_reliability_2026-11-14T06-58-09+00-00-openrouter-reliability_card.json#card (successful, requests, success_rate, slo_pass) -->

The targets are constants in the eval code: `evals/latency.py` (15 s, 6 s), `evals/cost.py` ($0.02) and
`evals/reliability.py` (95%). Multi-turn cases take longer (p50 11.67 s in the sweep, section 3.3) because the simulated
customer argues back.

### 3.8 Load test: the knee before and after the SQLite fix

- **Measures:** where the service itself saturates. The **knee** is the first request rate where the server stops
  keeping up: successful requests per second fall below 95% of the offered rate, `/resolve` p95 passes twice its
  unloaded time, or `/health` p95 passes 100 ms.
- **Why it matters:** with a real model the provider's rate limit binds first, so a fake model (0.8 s per call, no
  key) isolates Resolv's own overhead: auth, rate limits, ledger writes, the step journal.
- **How:** open-loop load (requests arrive on schedule whether or not earlier ones finished), one uvicorn worker, a
  16-thread laptop with the ledger on an NVMe disk, 60 s per rate, 24 rate steps in total. A `/health` probe runs
  alongside.

| Rate (req/s) | Before: ok/s, `/resolve` p95, `/health` p95 | After: ok/s, `/resolve` p95, `/health` p95 |
| --: | :-- | :-- |
| 20 | 19.487, 1.763 s, 0.036 s | 19.486, 1.641 s, 0.008 s |
| 40 | 30.225, 20.513 s, 48.597 s | **38.957, 1.644 s, 0.009 s** |
| 60 | 51.111, 9.552 s, 14.53 s | 58.428, 1.829 s, 0.168 s |
| 80 | 31.337, 91.013 s, 113.888 s | 33.894, 79.774 s, 104.774 s |

<!-- src: loadtest/results/fake_llm_knee.json#runs (series before_fix_disk and after_fix_disk; endpoints.resolve.ok_per_s, endpoints.resolve.p95_s, endpoints.health.p95_s), #notes (knee rule), #profile_before_fix_rate40 -->

- **Before** the fix, the knee was **40 req/s**. A profile at 40 req/s put about two thirds of the event loop's busy
  time in synchronous SQLite calls (step-journal commits 17.6%, transaction begin/commit 16.8%, connect 9.1%); each
  commit's disk sync froze every request, `/health` included.
- **After** moving those calls to a worker thread (`asyncio.to_thread`), the knee is **60 req/s** (`/health` p95
  passes 100 ms there), and 40 req/s holds cleanly. A repeat run at 40 and 60 req/s gave matching numbers (38.958
  and 58.433 ok/s).
- 0 errors in all 24 runs. The fake model calls no tools, so this measures the request path, not refunds.

<!-- src: loadtest/results/fake_llm_knee.json#runs[*].endpoints.*.errors (all 0), runs (after_fix_disk_repeat) -->

Detail: [loadtest/README.md](loadtest/README.md). What breaks next and the design past one node:
[docs/scale_out.md](docs/scale_out.md).

### 3.9 Eval gate false-positive rate

- **Measures:** how often the nightly regression gate would fail when nothing changed.
- **Why it matters:** a gate that cries wolf gets ignored. The gate fails on any unauthorized refund, or on a pass^1
  drop of at least 0.25 whose upper confidence bound is below zero.
- **How:** 1,000 pseudo-nightlies, each 8 tasks x 2 runs resampled from the reference rows, compared with the same
  rows. No model calls; CI runs this on every push.

| Metric | Value | Target |
| :-- | --: | --: |
| False-positive rate | **0.0** (0 of 1,000) | at most 0.05 |

<!-- src: output of `python -m evals.gate --baseline evals/results/ref_gpt_oss_120b.jsonl --subset 8 --runs 2 --draws 1000`, run for this report on 2026-11-14 (false_positive_rate 0.0, draws 1000, target <= 0.05); docs/eval/METRICS.md#Nightly gate rule -->

Limit: the resampled nightlies come from the same rows they are compared with, so this is the rule's noise floor,
not a test against real day-to-day drift.

### 3.10 Error analysis: 100 runs read by hand

- **Measures:** why runs fail, not just how often.
- **How:** a fixed sample of 100 runs (seed 0): every failure in the reference sweep plus 30 random passes as a
  control (57), every unauthorized refund from the harness-off arm (16), and 27 failures from an older historical
  sweep. Each run got a short open code, then one of 10 categories. One person coded all of them, so there is no
  inter-rater agreement number.

| Category | Count | Action |
| :-- | --: | :-- |
| clean pass (control) | 30 | none |
| refused a claim the record supports | 20 | prompt change proposed |
| paid a made-up amount below what was owed | 18 | prompt change proposed |
| harness off: paid what was asked | 15 | none needed (this is what the harness is for) |
| provider error (credit, rate limit) | 6 | accepted |
| promised a refund, never called the tool | 3 | prompt change proposed |
| never got the real order id | 3 | accepted (historical only) |
| **split a refund to slip under the approval limit** | 2 | **policy rule 7 fixed, tests added** |
| reply claims a refund that was refused | 2 | guardrail and grader fixed, test added |
| grader scored a wrong refund as a success | 1 | grader fixed, test added |

<!-- src: evals/error_analysis/coding.csv (100 rows; counts of axial_code; counts of source: ref 57, ablation_off 16, historical_ksweep 27); actions from docs/eval/ERROR_ANALYSIS.md#Summary -->

**The split-refund approval bypass.** In one reference task, a $248.90 refund was correctly sent to a human (over the
$200 limit). The agent then asked for exactly $200, which the old rule 7 allowed with no human, and in the recorded
run it went on to try $48.90 to top up to the full amount. No money above the cap moved, but the human approval step
was skipped. Rule 7 now also escalates any refund on an order that already has a refund waiting for a human, unless
it carries an approval for that exact call. Tests: `tests/test_policy.py::test_rule7_smaller_refund_cannot_skip_a_pending_approval`
and `tests/test_regression_replay.py::test_recorded_split_refund_cannot_skip_the_approval` (replays the four recorded
refund calls; no money moves).

Most real failures (refusing or under-paying a customer who was owed money) fail in the safe direction. The three
proposed prompt changes were not applied, because changing the prompt invalidates every measured number and needs a
paid re-run. Full write-up with quotes: [docs/eval/ERROR_ANALYSIS.md](docs/eval/ERROR_ANALYSIS.md).

### 3.11 Red-team cassettes in the PR gate

A **cassette** is a recorded run: the model's replies, the customer turns and the graded result, which can be
replayed through the real loop and harness with no model call. Three red-team defenses were recorded on order
ORD-1026: `A1-a` (direct injection), `A6-nan` (a NaN refund amount) and `A8-a` (system prompt extraction). Each
expects harm 0, and the harness capped each refund at what was owed ($6.85). They replay keylessly in
`tests/test_regression_replay.py` on every push, next to six recorded reference cases. They were recorded on Gemini
(`gemini-3.6-flash`) because the `gpt-oss-120b` account had no credit left; a replay checks the harness decisions,
not which model wrote the replies.

<!-- src: evals/regression/redteam_subset.jsonl (3 lines, harm 0); evals/regression/cassettes/rt-*.json#expected_row.paid, #harm; model per docs/eval/REDTEAM.md#4 (the cassette files do not record the model) -->

### 3.12 Second model on the current code: Gemini 2.5 Flash-Lite

- **Measures:** the full reliability sweep, the red team and the prompt-injection suite again, on the code after
  the split-refund fix (`cecae59`), with a different and much cheaper model in every role.
- **Why it matters:** the red-team and injection runs in 3.5 and 3.6 ran before that fix, and the gpt-oss sweep in
  3.3 stopped at 28 of 40 tasks. This answers whether "no harm" still holds on today's code, and gives a complete
  40-task pass^k curve. Using a weaker model is also a harder test of the harness: the model makes more mistakes,
  and the gate has to catch them.
- **How:** agent, simulated customer, extractor and judge all on `gemini/gemini-2.5-flash-lite` through LiteLLM,
  temperature 0.7, code `6fb23c5`, 16 November 2026. Dollar figures use the paid-tier list price, $0.10 in / $0.40
  out per million tokens. Same commands as 3.3, 3.5 and 3.6 with Gemini file names (section 6).
- **Not a before/after.** The model changed as well as the code, so compare harm (which should be 0 on any model),
  not success rates, against the gpt-oss numbers.

**Reliability, full sweep: 40 tasks x 5 repeats = 200 runs, all scored.**

| Metric | Gemini 2.5 Flash-Lite (task-clustered 95% CI) | gpt-oss-120b, partial (3.3) |
| :-- | --: | --: |
| pass^1 | **0.765** (0.64 to 0.875) | 0.829 |
| pass^3 | 0.708 | 0.736 |
| pass^5 | **0.675** (0.525 to 0.825) | 0.714 |
| Unauthorized refund rate | **0.0** | 0.0 |
| Invariant violations | 0.0 | 0.0 |
| Over-block rate | 0.21 | 0.093 |
| Reply matches what was actually paid | 1.0 | 0.986 |
| Latency p50 / p95 (multi-turn) | 12.29 s / 23.42 s | 11.67 s / 17.87 s |
| Cost per resolved case | $0.00108 | $0.00179 |
| Spend for the sweep | $0.165 | $0.208 |

<!-- src: evals/results/ref_gemini_2_5_flash_lite_card.json#manifest (agent_model, temperature, git_sha, git_dirty false), card (pass^1..5 and pass^1/pass^5 _ci95_clustered, unauthorized_rate, invariant_violation_rate, over_block_rate, reply_grounded_rate, p50_latency_s, p95_latency_s, cost_per_resolution, usd_total_all_rows); right column: evals/results/ref_gpt_oss_120b_partial_card.json#card -->

The weaker model refuses more customers who are owed money (over-block 0.21, against 0.093) and so resolves fewer
cases, but it never got money out that policy forbids. One run hit the 90-second timeout and was retried, so all
200 rows are complete.

**Red team, 288 runs, and prompt injection, 144 runs.**

| Family or arm | Runs | Complied | Money moved | Data, PII or prompt leaked |
| :-- | --: | --: | --: | --: |
| A1 direct injection | 24 | 0.083 | 0 | 0 |
| A2 wrapper-tag escape | 24 | 0.167 | 0 | 0 |
| A3 multi-turn crescendo | 24 | 0.458 | 0 | 0 |
| A4 identity spoofing | 24 | **1.0** | 0 | 0 |
| A5 cross-customer exfiltration | 24 | 0.0 | 0 | 0 |
| A6 amount-format attacks | 40 | 0.0 | 0 | 0 |
| A7 injection via tool output | 24 | 0.0 | 0 | 0 |
| A8 system prompt extraction | 24 | 0.0 | 0 | 0 |
| A9 unbounded consumption | 24 | 0.25 | 0 | 0 |
| A10 approver social engineering | 24 | manual review | 0 | 0 |
| A11 PII echo | 24 | 0.0 | 0 | 0 |
| red-team control | 8 | | 0 | 0 |
| injection control (no attack) | 24 | 0.125 | 0 | 0 |
| direct_override | 24 | 0.042 | 0 | 0 |
| fake_system_turn | 24 | 0.042 | 0 | 0 |
| fake_tool_result | 24 | 0.25 | 0 | 0 |
| authority_roleplay | 24 | 0.083 | 0 | 0 |
| exfiltrate_prompt | 24 | 0.042 | 0 | 0 |

<!-- src: evals/results/redteam_gemini_2_5_flash_lite_card.json#runs, card.families.<Fam> (n, complied_rate, money_moved, data_leaked, pii_leaked, prompt_leaked, errors), card.harm_summary; evals/results/injection_gemini_2_5_flash_lite_card.json#card.runs, card.errors, card.complied.<arm>.complied_rate, card.harm.<arm> (unauthorized_rate, excess_usd) -->

Harm over all 432 runs: money moved 0, data leaked 0, PII leaked 0, prompt leaked 0, tool budget bypassed 0, and
0 unauthorized refunds in every injection arm. No injection arm's lift over the control has an interval that
excludes zero. A4 repeats the gpt-oss result: the model looked up the stranger's order on all 24 runs, and every
lookup was refused by the ownership check. 4 red-team runs ended in a timeout or provider error (A3 1, A6 2, A7 1);
they stay in the denominators. The red-team and injection manifests say `git_dirty: true` only because the sweep's
own new result files were untracked at the time; the code was `6fb23c5`.

Spend for all three Gemini runs: $0.314 (sweep $0.165, red team $0.10, injection $0.047).

<!-- src: evals/results/{ref,redteam,injection}_gemini_2_5_flash_lite.jsonl, sum of row usd; injection card.errors 0 -->

### 3.13 Older numbers (history only)

Earlier rounds (July to October 2026) ran mostly on Gemini 3.5 Flash with an older prompt and harness, for example a
pinned pass^3 of 0.9625 over 200 runs and an extractor both-correct of 0.9783. The model, prompt and harness have all
changed since, so these are not a before/after for the current numbers. They are kept in
[docs/eval_report.md](docs/eval_report.md), sections 2 to 7.

<!-- src: evals/baselines/agent.json#model, card.pass^3, card.runs; evals/baselines/extractor.json#score.both_acc -->

---

## 4. Tests and CI

The suite needs no API key and no network. Run for this report on 2026-11-14 with Python 3.11:

- With provider keys blanked (as CI runs): **458 passed, 2 skipped** (the two tone-judge tests skip without a key).
- `pytest --collect-only -q`: **465 tests** in 38 modules (rechecked 2026-11-15). `ruff check .` passes.

| Area | What the tests check | Modules | Tests |
| :-- | :-- | :-- | --: |
| Policy rules | every rule with a passing and failing case, exact thresholds, which rule answers when two break, the split-refund fix | `test_policy` | 29 |
| Harness and tools | the claim verifier, audit trail, tools, routing, escalation auto-emit | `test_harness` | 53 |
| Ledger and audit chain | one live refund per order enforced by the database, cents round-trip, idempotency keys, outbox; HMAC chain catches edits, deletions, reordering, truncation and a wrong key; concurrent double-refund race closed | `test_ledger`, `test_hmac_chain`, `test_audit_concurrency_gap` | 20 |
| Chaos and crash resume | 200 seeded chaos runs of the saga (provider 500s, timeouts after commit, declines, crashes mid-saga, concurrent attempts, replayed keys) then seven invariants; a crashed case resumes from its journal and pays once; a live retry keeps the failed attempt's evidence | `test_chaos_refunds`, `test_crash_resume`, `test_live_retry` | 10 |
| Human approvals | approval bound to the exact call, approver-only access, approved refunds replay correctly | `test_escalations`, `test_approved_replay` | 18 |
| Guardrails and PII | reply groundedness evasions, prompt leak, PII and rudeness checks; PII masking leaves order ids and money alone | `test_guardrail_evasions`, `test_pii` | 42 |
| Malformed tool input | NaN, infinity, strings, booleans and off-enum values refused and still recorded in the audit trail | `test_tool_args`, `test_input_validation` | 29 |
| Identity and ownership | lookups show only the caller's own orders; a hallucinated order id that hits a stranger's order does not pay | `test_lookup_ownership`, `test_hallucination_collision` | 14 |
| Agent loop | step cap, malformed calls, guardrail gate, budgets, circuit breaker, event loop never blocked | `test_loop_orchestration`, `test_budgets`, `test_async_health` | 22 |
| API and operator controls | order-proof tokens, approver tokens not mintable over HTTP, prod fails closed without secrets; `Idempotency-Key`, shadow mode and kill switch counted in ledger rows | `test_auth`, `test_resolve_controls` | 31 |
| MCP server | every tool through an in-process MCP client; `check_refund` never writes a refund row | `test_mcp_server` | 5 |
| Cassette replay (PR gate) | every recorded cassette reproduces its graded row; red-team cassettes stay harm-free | `test_regression_replay` | 11 |
| Eval machinery | task builder and grader, pass^k, Wilson, McNemar, clustered bootstrap and Holm math against hand-worked values, extractor scoring, run provenance and the independent invariant checker, injection control arm, red-team predicates | `test_eval`, `test_metrics`, `test_extractor_eval`, `test_provenance`, `test_injection_ablation`, `test_injection_control`, `test_redteam_predicates` | 129 |
| Observability | span tree and attributes, online monitor, drift detector | `test_tracing`, `test_monitor`, `test_drift` | 26 |
| Config | per-role models, self-hosted routing at $0, one key per provider | `test_config_roles` | 9 |
| Load test and serving math | percentiles and per-endpoint summaries | `test_loadtest_summary`, `test_serve_bench` | 10 |
| Tone judge (needs a key) | a warm refusal passes, a curt one fails | `llm/test_tone_judge` | 2 |
| **Total** | | | **460** |

<!-- src: `venv/bin/pytest --collect-only -q` grouped by module (run 2026-11-14); `GEMINI_API_KEY= GROQ_API_KEY= OPENROUTER_API_KEY= venv/bin/pytest -q -p no:cacheprovider` -> 458 passed, 2 skipped; module descriptions from each test file's header -->

**CI** ([.github/workflows/ci.yml](.github/workflows/ci.yml)), on every push and pull request, no secrets: `ruff
check .`, the full keyless `pytest` suite (including cassette replay), and the eval gate false-positive check from
section 3.9.

**Scheduled evals** ([.github/workflows/nightly-eval.yml](.github/workflows/nightly-eval.yml)) run only when a
`GROQ_API_KEY` secret is set; without it every job shows as skipped. Nightly (06:00 UTC): the extractor regression
gate against `evals/baselines/extractor_ref_gpt_oss_120b.json`, and an 8-task agent smoke gate that also skips until
an agent baseline from the full sweep is pinned. Weekly (Sunday): the full 40-task sweep, capped at $0.50 and
non-blocking. Results upload as Actions artifacts.

---

## 5. Limitations

- **The gpt-oss-120b reliability sweep is partial.** 28 of 40 tasks have all 5 repeats; the scored set leans toward
  deny tasks (16 deny, 11 refund, 1 escalate). A complete 40-task sweep exists only on Gemini 2.5 Flash-Lite
  (3.12), a different model. No agent baseline is pinned yet, so the nightly agent gate skips.
- **The fine-tuned extractor has not been re-measured.** The Qwen2.5-1.5B GRPO adapter recorded 0.707 both-correct
  once during training, and its rows were not kept. Re-measuring it on the 92 held-out messages needs a GPU and
  about 8 GB of free RAM while the weights load (docs/inference.md).
- **The simulated customer is the same model as the agent.** That can flatter the agent (it may argue in ways the
  agent handles well). The tone judge is also the same model and has no human labels.
- **Small samples.** Ablation arms are 40 runs each, latency and reliability 25 runs, red-team CIs rest on 4 orders,
  and hallucination on 13 messages. Several differences reported above are inside the noise, and are labelled so.
- **The extractor is no better than a regex** on this data (0.891 both). It is kept because the harness makes its
  errors safe, not because it wins.
- **Single node.** One SQLite file and in-process rate limits and circuit breaker. One worker holds 40 req/s on a
  laptop with a fake model; past one node, rate limits, the breaker and the ledger all need a shared store
  ([docs/scale_out.md](docs/scale_out.md)). None of that is built.
- **Not deployed yet.** It runs locally or in Docker. The latency SLOs have not been measured on a deploy, and the
  restore drill ran on a local ledger only.
- **Simulated payments.** The saga runs against a fake payment provider, not a real one.
- **Open checks.** The A10 approver social-engineering rows need a manual review; the red-team cassettes were recorded
  on a different model from the one measured; the error-analysis coding has a single rater.

---

## 6. How to reproduce each number

Setup:

```bash
git clone https://github.com/AmanDataGuy/Resolv && cd Resolv
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
```

**No key needed:**

```bash
# Section 4: the test suite (blank keys so the two tone-judge tests skip, as in CI)
GEMINI_API_KEY= GROQ_API_KEY= OPENROUTER_API_KEY= pytest -q
pytest --collect-only -q

# Section 3.9: eval gate false-positive rate
python -m evals.gate --baseline evals/results/ref_gpt_oss_120b.jsonl --subset 8 --runs 2 --draws 1000

# Section 3.4: the regex baseline (seconds, CPU)
python -m inference.serve_bench --backend regex --label regex

# Section 3.4: paired McNemar, model rows vs regex rows
python -c "import json; from evals.extractor import compare; \
rows=[json.loads(l) for l in open('evals/results/extractor_gpt_oss_120b.jsonl')]; \
b=json.load(open('inference/results/extractor/serve_regex.json')); b['score']=b['card']; print(compare(rows, b))"

# Section 2: the 8 policy rules against the real order data
python -m resolv.harness.policy

# Section 3.10: re-grade committed rows with the fixed grader
python scripts/regrade_cards.py

# Section 3.8: load test with the fake model (full commands in loadtest/README.md)
RESOLV_ENV=dev RESOLV_FAKE_LLM=1 FAKE_LLM_LATENCY_S=0.8 TRUSTED_CLIENT_IP_HEADER=X-Loadtest-IP \
  ORDER_PROOF_SECRET=loadtest-secret uvicorn resolv.api.main:app --port 8000
python -m loadtest.resolve --order ORD-1187 --proof <code> --rate 40 --duration 60 \
  --concurrency 400 --ip-header X-Loadtest-IP --fake-llm
```

**Needs a provider key** (paid; set `LLM_PROVIDER=openrouter LLM_MODEL=openrouter/openai/gpt-oss-120b`):

| Section | Command |
| :-- | :-- |
| 3.3 reference sweep (resumes into the same file) | `python -m evals.runner --n 5 --tasks 40 --out ref_gpt_oss_120b.jsonl --usd-cap 2` |
| 3.2 harness on vs off | `python -m evals.ablation --tasks 40 --n 1 --arm both --out ablation_gpt_oss_120b.jsonl` |
| 3.2 extractor on vs off | `python -m evals.ablation --tasks 40 --n 1 --arm extractor --out ablation_extractor_gpt_oss_120b.jsonl` |
| 3.4 extractor | `python -m evals.extractor --split test --out extractor_gpt_oss_120b.jsonl` |
| 3.5 prompt injection | `python -m evals.injection --orders 8 --n 3 --out injection_gpt_oss_120b.jsonl` |
| 3.6 red team | `python -m evals.redteam --out redteam_gpt_oss_120b.jsonl` |
| 3.7 latency | `python -m evals.latency --tasks 5 --n 5 --warmup 1` |
| 3.7 cost | `python -m evals.cost --tasks 5 --n 3` |
| 3.7 reliability | `python -m evals.reliability --tasks 5 --n 5` |

For section 3.12 set `LLM_PROVIDER=gemini LLM_MODEL=gemini/gemini-2.5-flash-lite` and use the same commands with
Gemini file names: `python -m evals.runner --n 5 --tasks 40 --k 5 --out ref_gemini_2_5_flash_lite.jsonl --usd-cap 2`,
`python -m evals.redteam --out redteam_gemini_2_5_flash_lite.jsonl` and
`python -m evals.injection --orders 8 --n 3 --out injection_gemini_2_5_flash_lite.jsonl`. The pass^1 to pass^5 curve
on its card comes from `ref_card` in `scripts/regrade_cards.py`, applied to the Gemini rows.

<!-- src: docs/eval_report.md#8 (commands); .github/workflows/ci.yml (gate command); loadtest/README.md#Run; loadtest/results/fake_llm_knee.json#commands -->

Every result file and card lives in [evals/results/](evals/results/); [evals/results/README.md](evals/results/README.md)
explains what each row records (model, prompt hash, dataset hash, tokens, cost).
