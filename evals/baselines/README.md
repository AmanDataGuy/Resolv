# Pinned baselines

Baseline files are the floor a regression check compares against. Unlike per-run rows under `evals/results/`,
they are tracked in git.

| File | What | Model | Pinned | Status |
|---|---|---|---|---|
| `agent.json` | pass^k sweep: scorecard + 200 per-run rows (40 tasks × 5) | `gemini/gemini-3.5-flash` | 2026-07-25 (commit 64e48b6) | **Historical.** Before rule 2 (caller must own the order), the reply guardrails and the prompt change. Google now routes that endpoint to a newer model, so it cannot be reproduced as-is. |
| `extractor.json` | extractor on the held-out split (n=92) | `gemini/gemini-3.5-flash` | 2026-07-25 (commit 64e48b6) | **Historical.** Same model caveat. |

New agent references go in `agent_ref_<provider>.json`, one per model, so the gate always compares a model against
itself. None is pinned yet; the nightly smoke expects `agent_ref_groq.json`.

## Pin

```bash
python -m evals.extractor --split test --save-baseline                                   # -> extractor.json
python -m evals.runner --n 5 --tasks 40 --save-baseline evals/baselines/agent_ref_groq.json
```

`--save-baseline` with no path writes `agent.json` and overwrites the historical file; pass a path.

## Gate

```bash
python -m evals.extractor --split test --baseline                                        # exit 1 on a significant regression
python -m evals.runner --subset 8 --n 2 --k 1 --gate evals/baselines/agent_ref_groq.json  # exit 1 fail, 78 inconclusive
```

The runner refuses to gate against a missing file and tells you to pin one first.
