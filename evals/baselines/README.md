# Pinned baselines

Baseline files are the floor a regression check compares against. Unlike per-run rows under `evals/results/`,
they are tracked in git. Baselines are named by model weights, not provider: the same `gpt-oss-120b` weights run
on Groq and OpenRouter, and the gates compare a model against a baseline of the same weights on any host
(`evals/provenance.py::weights_of`).

| File | What | Model | Pinned | Status |
|---|---|---|---|---|
| `agent_ref_gpt_oss_120b.json` | pass^k sweep: scorecard + 200 per-run rows (40 tasks × 5) | `gpt-oss-120b` | **not pinned yet** | The reference sweep stopped at 141 of 200 rows when the provider account ran out of credit (`evals/results/ref_gpt_oss_120b_partial_card.json`). Pin it once the sweep completes; until then the nightly agent gate skips, and the PR step reads `evals/results/ref_gpt_oss_120b.jsonl`, whose first 8 tasks are complete. |
| `extractor_ref_gpt_oss_120b.json` | extractor (json backend) on the held-out split (n=92) | `openrouter/openai/gpt-oss-120b` | 2026-11-14 | **Current reference.** The nightly extractor gate reads it. |
| `agent.json` | pass^k sweep: scorecard + 200 per-run rows (40 tasks × 5) | `gemini/gemini-3.5-flash` | 2026-07-25 (commit 64e48b6) | **Historical.** Before rule 2 (caller must own the order), the reply guardrails and the prompt change. Google now routes that endpoint to a newer model, so it cannot be reproduced as-is. |
| `extractor.json` | extractor on the held-out split (n=92) | `gemini/gemini-3.5-flash` | 2026-07-25 (commit 64e48b6) | **Historical.** Same model caveat. |

## Eval tiers

| Tier | Trigger | What runs | Needs a key | Pass rule |
|---|---|---|---|---|
| PR | every push and pull request (`.github/workflows/ci.yml`) | `ruff`, `pytest` (includes the recorded cassette replays and the red-team cassettes), and `python -m evals.gate` on the reference rows (the nightly rule's false-positive rate, no model calls) | no | pytest exit 0; the printed false-positive rate is recorded in `docs/eval/METRICS.md` (target ≤ 5%) |
| Nightly | 06:00 UTC (`.github/workflows/nightly-eval.yml`) | extractor gate against `extractor_ref_gpt_oss_120b.json`; 8 tasks × 2 runs gated against `agent_ref_gpt_oss_120b.json` | yes (`GROQ_API_KEY`) | exit 0 pass, 1 regression (fails), 78 inconclusive (warning) |
| Weekly | Sunday 07:00 UTC | the full 40 × 5 sweep, capped by `NIGHTLY_USD_CAP` | yes | non-blocking; compare with the reference card in `evals/results/` by hand |
| Reference | by hand, after any prompt or policy change | the full sweep, ablation, extractor, injection, red team and ops runs, with `--save-baseline` | yes | new baseline files and result rows committed with the change |

A change to `SYSTEM` or `TOOL_SCHEMAS` in `resolv/agents/loop.py` changes `prompt_sha`: re-record the cassettes
(`scripts/record_cassettes.py`) and re-run the reference sweep in the same change.

## Pin

```bash
python -m evals.extractor --split test --save-baseline evals/baselines/extractor_ref_gpt_oss_120b.json
python -m evals.runner --n 5 --tasks 40 --out ref_gpt_oss_120b.jsonl --save-baseline evals/baselines/agent_ref_gpt_oss_120b.json
```

A bare `--save-baseline` writes `agent_ref.json` / `extractor_ref.json`; pass a path named by model.

## Gate

```bash
python -m evals.extractor --split test --baseline evals/baselines/extractor_ref_gpt_oss_120b.json  # exit 1 on a significant regression
python -m evals.runner --subset 8 --n 2 --k 1 --gate evals/baselines/agent_ref_gpt_oss_120b.json   # exit 1 fail, 78 inconclusive
python -m evals.gate --baseline evals/results/ref_gpt_oss_120b.jsonl --subset 8 --runs 2           # false-positive rate, keyless
```

The runner refuses to gate against a missing file and tells you to pin one first.
