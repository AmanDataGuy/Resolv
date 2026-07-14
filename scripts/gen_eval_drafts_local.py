"""Local draft generation — the GPU half of the fine-tune eval, run on this machine
instead of Kaggle, across EVERY training stage.

scripts/eval_finetune.py is deliberately split: generation needs a GPU, scoring
doesn't. This is the generation half. It drafts each held-out context once per stage —
base (no adapter), SFT, ORPO, GRPO — so the scorer can show each stage's contribution
in isolation, not just base-vs-final.

Memory (fits a 6 GB card): ONE copy of the 1.5B base is held; all three LoRA adapters
are attached to it as named adapters and selected with model.set_adapter(...) /
model.disable_adapter() for base. Switching an adapter is nearly free — no extra full
model copy.

Reuse: if data/eval_drafts.json already has a draft for a stage (older runs saved GRPO
as "ft_draft"), it is carried over instead of regenerated — so re-running after adding a
stage only pays for the missing stages, not all of them.

Run:  .venv-train\\Scripts\\python.exe scripts\\gen_eval_drafts_local.py
Then: venv\\Scripts\\python.exe scripts\\eval_finetune.py   (the scorer)
"""
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_finetune import DRAFTS_FILE, load_heldout  # sibling module (same scripts/ dir)

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
STAGES = ["sft", "orpo", "grpo"]  # adapter dirs under models/adapters/<stage>/latest
ADAPTERS = Path("models/adapters")


def draft(model, tok, prompt: str) -> str:
    """Greedy single draft with whatever adapter is currently active on `model`
    (do_sample=False = deterministic, so stage-to-stage diffs reflect weights, not noise)."""
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    out = model.generate(ids, max_new_tokens=220, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    heldout = load_heldout()
    print(f"{len(heldout)} held-out contexts")
    if not heldout:
        raise SystemExit("0 contexts — data/exceptions_pool_*.json missing. Nothing to generate.")

    # carry over anything already generated (older runs saved GRPO under 'ft_draft')
    prev = {}
    if DRAFTS_FILE.exists():
        for r in json.loads(DRAFTS_FILE.read_text()):
            if "ft_draft" in r and "grpo_draft" not in r:
                r["grpo_draft"] = r["ft_draft"]
            prev[r["order_id"]] = r

    stages = [s for s in STAGES if (ADAPTERS / s / "latest" / "adapter_config.json").exists()]
    print(f"stages found: base + {stages}")

    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="cuda")
    model = None
    for s in stages:  # attach each adapter as a named adapter on the single base copy
        path = str(ADAPTERS / s / "latest")
        if model is None:
            model = PeftModel.from_pretrained(base, path, adapter_name=s)
        else:
            model.load_adapter(path, adapter_name=s)
    model.eval()

    rows = []
    with torch.no_grad():
        for i, ctx in enumerate(heldout, 1):
            old = prev.get(ctx["order_id"], {})
            row = {**ctx}
            # base = adapter disabled (reuse if we already have it)
            if old.get("base_draft"):
                row["base_draft"] = old["base_draft"]
            else:
                with model.disable_adapter():
                    row["base_draft"] = draft(model, tok, ctx["prompt"])
            # each fine-tuned stage (reuse if present, else generate)
            for s in stages:
                col = f"{s}_draft"
                if old.get(col):
                    row[col] = old[col]
                else:
                    model.set_adapter(s)
                    row[col] = draft(model, tok, ctx["prompt"])
            rows.append(row)
            print(f"  {i}/{len(heldout)}", end="\r")

    DRAFTS_FILE.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} rows -> {DRAFTS_FILE}  (columns: base + {stages})")


if __name__ == "__main__":
    main()
