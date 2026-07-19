<div align="center">

# Resolv

**A refund agent that cannot break its own policy — and a reliability benchmark that proves it.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Google ADK](https://img.shields.io/badge/Google_ADK-Extractor-4285F4?style=flat-square&logo=google&logoColor=white)](https://google.github.io/adk-docs/)
[![litellm](https://img.shields.io/badge/litellm-Groq·OpenRouter-F55036?style=flat-square)](https://litellm.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-/resolve-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![eval](https://img.shields.io/badge/eval-pass%5Ek-EE4C2C?style=flat-square)](EVAL_REPORT.md)

*A customer lies about how much they paid. The agent looks up the real figure, the policy engine refuses the inflated refund, and the agent recovers with the exact amount owed — every step of it in an audit trail that the benchmark then grades.*

</div>

---

## What it does

A customer sends a messy complaint — *"my order ORD-1000 was late, I paid $900, I want it all back."* Resolv:

1. **Extracts** a typed claim (`order_id`, `claim_type`) from the free-form message — the one genuinely hard language task, and the fine-tune target.
2. **Runs an agent loop** that can look up the order and attempt a refund — but every refund is checked against a deterministic **policy engine** *before* any money moves.
3. **Routes** anything over the agent's authority to the right human team, and **emails the customer a ticket**.

The dollar math, the eligibility rules, and the allow/deny/escalate decision are never left to the language model. **The model proposes; the harness disposes.**

---

## The core idea — the model proposes, the harness disposes

The agent has no authority. It emits tool calls; [`harness/tools.py`](harness/tools.py) is the *only* write path, and it calls [`harness/policy.py`](harness/policy.py) before every mutation. Breaking that would require adding a second write path — a reviewable change to one file, not something a prompt can talk its way into.

| Decision | Why |
|---|---|
| **Policy is Python, not a prompt** | Six first-match rules over two dicts. The model's own "rules" were already a threshold table; asking a model to apply a table it can only get *wrong* is pure downside. |
| **The cap comes from the record, never the customer** | The refund cap is a fraction of what the order shows was *paid* — so an inflated claim ($900 on a $639 order) is refused by arithmetic, not by the model noticing. |
| **The audit trail IS the state** | Every attempt — allowed *or denied* — is appended to a JSONL trail. Rule 4 reads it to catch a double refund; the eval grades from it; the UI renders it. One source of truth. |
| **Denials are results, not exceptions** | A refused refund comes back as a string the agent explains to the customer, not a crash that strands them mid-conversation. |

---

## Architecture

```mermaid
flowchart TD
    Msg([Messy customer message]) --> Ext[["Extractor · LLM<br>message → CustomerClaim"]]:::llm
    Ext --> Loop[["Agent loop · LLM<br>proposes tool calls"]]:::llm
    Loop -->|lookup / refund / escalate| Tools[tools.py<br>the only write path]:::h

    subgraph HARNESS ["harness/ · deterministic · no LLM"]
        Tools --> Policy[policy.py<br>6 rules, first match]:::h
        Policy --> Audit[(audit trail<br>append-only JSONL)]:::io
    end

    Audit --> Route[routing.py<br>ticket + team]:::h
    Route --> Notify[notify.py<br>email customer · page team]:::io

    classDef llm fill:#fce7f3,stroke:#db2777,stroke-width:2px,color:#831843;
    classDef h fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a;
    classDef io fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b;
```

*The two pink nodes are the only LLM calls. Everything blue is deterministic and tested; the model never enforces its own constraints.*

---

## The benchmark — pass^k under an adversarial customer

This is the heart of the project: not "can the agent resolve a complaint," but **"can it do so reliably, against a customer who is actively trying to extract money they aren't owed."** The methodology follows τ-bench (Sierra/Princeton): an LLM-simulated user, tools that read/write real state, a written policy, and verified outcomes.

- **A non-collaborative customer** ([`eval/simulator.py`](eval/simulator.py)) plays five tactics — inflate the amount, give a wrong order number and insist, change the story after a refusal, apply pressure — each targeting a specific policy rule.
- **pass^k, not pass@k** ([`eval/metrics.py`](eval/metrics.py)): *did it get the same task right k times running?* Reliability, not luck. The agent samples at temperature 0.7 (so the n repeats are independent); the harness stays deterministic. That contrast is the thesis.
- **The answer key is derived from the policy engine** ([`eval/tasks.py`](eval/tasks.py)), not written down — so the benchmark can never drift out of sync with the system it grades. Tasks are sampled to a verified 50/50 split, so no constant policy ("deny everything") beats a coin flip.
- **Grading replays the trail against policy** ([`eval/runner.py`](eval/runner.py)): for every successful refund, `check_refund()` is re-run over the prior history to independently confirm it was allowed. `unauthorized_rate` **must be 0** — nonzero means the enforcement claim is false and nothing else on the scorecard matters.

**Headline result** — a balanced 200-run sweep (40 tasks × 5 repeats) on Gemini 3.5 Flash:

| metric | value | reading |
|---|---|---|
| **pass³** | **0.97** | reliably correct across repeats, not lucky once |
| **unauthorized_rate** | **0.0** | zero refunds outside policy in 200 adversarial runs |
| over_block_rate | 0.01 | the only 2 misses — and both erred toward *not* paying |
| resolve_rate | 0.99 | (Wilson CI 0.964–0.997) |

The tasks are a verified 50/50 refund/deny split, so "deny everything" scores 0.50 — the numbers are earned, not degenerate. And the two errors in 200 runs were both *over-blocks*: the agent denied someone owed money, never paid someone who wasn't. That asymmetry is exactly what a refund system should fail toward. Full methodology, per-tactic breakdown, and run history in **[EVAL_REPORT.md](EVAL_REPORT.md)**.

---

## Fine-tuning — RLVR on the extractor

The fine-tune target is the **extractor** ([`agents/extractor.py`](agents/extractor.py)) — the one task where a model genuinely has headroom (recovering *which* order and *what* problem from an emotional ramble, where the order number may be spelled out, typo'd, or a decoy). Its reward is [`harness/validity.py`](harness/validity.py) — the *same deterministic verifier* that gates production — so it's **RLVR, not RLHF**: no LLM judge, one definition of "got the facts right," used in training and at run time.

**Result** (Qwen2.5-1.5B, LoRA, GRPO via TRL, on Kaggle): held-out both-fields-correct extraction accuracy **0.489 → 0.707 (+0.217)** — order-id 0.761→0.783, claim-type 0.609→0.870. Judged on a held-out split, never the training set. The near-zero GRPO loss is *expected*, not a collapse: GRPO advantages are mean-zero by construction, so the loss reads ~0 while the policy still learns — which is exactly why the fine-tune is judged by held-out accuracy, not the loss curve.

Sequencing is deliberate: establish a trustworthy benchmark and a **baseline** first, then fine-tune, then judge the improvement. The reward is the deterministic verifier, so "better" means "more claims that pass the same check production enforces" — not a preference a judge model happened to hold.

---

## Tech stack

| Layer | Choice |
|---|---|
| **Extractor** | Google ADK (`LlmAgent`, `output_schema`) — structured claim extraction; the RLVR target |
| **Agent loop** | litellm tool-calling, **provider-agnostic** (Groq or OpenRouter via one config switch), temperature per run |
| **Harness** | Deterministic Python — policy, tools, audit, validity, routing (no API key needed to test) |
| **Data** | 460 orders + complaints derived from real **Olist** deliveries; ground truth known before the message exists |
| **Eval** | pass^k estimator, geometric-mean trajectory scoring, Wilson intervals, McNemar — pure math, no LLM |
| **API / UI** | FastAPI `/resolve` · Streamlit split-screen demo (customer ⟷ audit trail) |
| **Validation** | Pydantic v2 schemas ([`schemas.py`](schemas.py)) shared across agent, harness, eval, and tests |

---

## Quick start

```bash
python -m venv venv && venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env                                # add a GROQ or OPENROUTER key

# The deterministic core — no API key needed:
pytest tests/ -q
python -m harness.policy        # the 6 rules, against the real order DB
python -m eval.metrics          # the scoring math, checked against hand-worked numbers

# Needs an LLM key:
streamlit run app.py            # the split-screen demo
uvicorn api.main:app --reload   # POST /resolve  {"message": "..."}
python -m eval.runner --n 1 --tasks 2   # a tiny eval sweep (resumable, with a progress bar)
```

**Environment** (see [`.env.example`](.env.example)) — the harness and its tests run with *no* key at all; a key is only needed to drive the agent end to end.

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` (+ `_2`, `_3`) | Agent + simulator via OpenRouter; extra keys rotate on rate limits |
| `GROQ_API_KEY` (+ `_2`, `_3`) | Alternative provider; fastest, but a hard daily token cap |
| `LLM_PROVIDER` / `LLM_MODEL` | Override provider/model per run |
</content>
