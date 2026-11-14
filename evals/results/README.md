# evals/results

Result rows and cards written by the eval entry points. Every number published in README.md or docs/eval_report.md
must resolve to a file here that `git ls-files` lists.

## What gets committed

| File | Written by | Committed? |
|---|---|---|
| `ref_<provider>.jsonl` + `_card.json` | `python -m evals.runner --out ref_<provider>.jsonl` (reference sweeps) | yes |
| `injection*.jsonl` + `_card.json` | `python -m evals.injection` | yes, when a number is published from it |
| `ablation*.jsonl` + `_card.json` | `python -m evals.ablation` | yes, when a number is published from it |
| `extractor*.jsonl` + `_card.json` | `python -m evals.extractor` | yes, when a number is published from it |
| `ops_{latency,cost,reliability}_<run_id>.*` | `python -m evals.latency` / `cost` / `reliability` | yes, the run that backs the README ops table |
| `historical/` | copies of older local evidence | yes, files under 5 MB (gzip or a column subset otherwise) |
| smoke runs, nightly runs, anything else | | no (CI uploads nightly/weekly rows as Actions artifacts) |

Rules:
- Keep each committed file under 5 MB.
- Never commit sweep logs.
- A file that backs a published number is never edited by hand; re-run the entry point into a new file instead.

## Row provenance

Every row carries the run manifest (`run_id`, `git_sha`, `git_dirty`, `provider`, the model of each role,
`prompt_sha`, `dataset_sha`, `openings_sha`, `temperature`, `seed`, `started_at`) and its own `served_model`,
`input_tokens`, `output_tokens`, `usd` (from `config.price_usd`), `error_kind` and `finished_at`.
Agent rows also carry the compact audit `trail`, the `transcript`, and `invariant_violations` from the independent
grader in `resolv/invariants.py`.
