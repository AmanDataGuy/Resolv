# Pinned baselines for the regression gates

`extractor.json` and `agent.json` are written by `--save-baseline` and read by `--baseline`
(see [`eval/extractor.py`](../extractor.py) and [`eval/runner.py`](../runner.py)). They are the
frozen floor a CI regression check compares against — so unlike the per-run `data/eval/*.jsonl`
sweeps, these two files **are** tracked in git.

Neither exists yet: pinning one takes a paid provider run. Until a baseline is committed here,
`--baseline` exits cleanly telling you to run `--save-baseline` first.

To pin the current model as the floor:

```bash
python -m eval.extractor --split test --save-baseline      # -> extractor.json
python -m eval.runner    --n 5 --tasks 40 --save-baseline   # -> agent.json
```

Then a later run gates on it:

```bash
python -m eval.extractor --split test --baseline            # exit 1 on a significant regression
python -m eval.runner    --n 5 --tasks 40 --baseline
```
