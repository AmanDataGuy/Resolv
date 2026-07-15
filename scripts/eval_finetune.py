"""Real evaluation harness: base vs fine-tuned drafter, on held-out contexts,
across TWO axes — the thing the 6-example Cell 7 smoke test failed to do.

Why two axes:
  - facts (deterministic): does the draft contain the exact order ID + dollar
    amount? Reuses the same exact-match idea as harness/guardrails.py. The base
    model already aces this (that's the whole finding), so it's the control axis.
  - tone (LLM judge, Groq): is it a professional supply-chain ops email? 1-5.
    This is the axis fine-tuning can actually move — a base 1.5B writes generic
    prose; fine-tuning on 190 domain emails should read more like an ops manager.
    THIS is where a real fine-tuning win (or honest null) shows up.

Two-part design, because generation needs a GPU but scoring doesn't:
  1. GENERATE (Kaggle, GPU): load base + each adapter, draft on the held-out set,
     write data/eval_drafts.json. See the Kaggle cell in this file's docstring.
  2. SCORE (anywhere, no GPU): read eval_drafts.json, score facts + tone, print
     the base-vs-fine-tuned table. That's what `main()` here does.

Held-out set: rows 50-99 of the late_shipment and price_dispute pools — training
used only the first 50 of each (MAX_PER_TYPE=50), so these are genuinely unseen.

Run scoring:  python scripts/eval_finetune.py
(expects data/eval_drafts.json produced by the Kaggle generate cell)

--- Kaggle GENERATE cell (produces eval_drafts.json) ---
    import json, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from scripts.eval_finetune import load_heldout   # or inline it

    BASE = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype="float16", device_map="cuda")
    ft = PeftModel.from_pretrained(base, "models/adapters/grpo/latest")  # or sft/orpo

    def draft(model, prompt):
        msgs = [{"role":"user","content":prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=220, do_sample=False)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    rows = []
    for ctx in load_heldout():
        p = ctx["prompt"]
        rows.append({**ctx, "base_draft": draft(base, p), "ft_draft": draft(ft, p)})
    json.dump(rows, open("data/eval_drafts.json", "w"), indent=2)
"""
import json
import os
import re
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DRAFTS_FILE = DATA_DIR / "datasets" / "legacy" / "eval_drafts.json"

# Groq is 30 req/min per key; a full run is ~200 tone calls. Rotate across the
# configured keys and back off on a rate-limit so the eval doesn't die mid-run.
_GROQ_KEYS = [
    k for k in (os.environ.get("GROQ_API_KEY"), os.environ.get("GROQ_API_KEY_2"), os.environ.get("GROQ_API_KEY_3")) if k
]
_key_i = 0


def load_heldout() -> list[dict]:
    """Held-out contexts (rows 50-99 of the two large pools) — unseen in training.
    Each has a prompt plus the order_id + amount the facts-check needs.
    """
    rows = []
    for pool in ["late_shipment", "price_dispute"]:
        path = DATA_DIR / "pools" / f"exceptions_pool_{pool}.json"
        if not path.exists():
            continue
        for ctx in json.loads(path.read_text())[50:100]:
            oid = (ctx.get("order_ids") or ["N/A"])[0]
            amt = float(ctx.get("revenue_at_risk_usd") or 0)
            rows.append({
                "exception_type": ctx["exception_type"],
                "order_id": oid,
                "amount": amt,
                "prompt": f"Exception type: {ctx['exception_type']} | Orders: {oid} | "
                          f"Revenue at risk: ${amt:.2f}\nDraft a professional resolution email.",
            })
    return rows


def score_facts(draft_text: str, order_id: str, amount: float) -> int:
    """Deterministic — 1 point each for the exact order ID and the dollar amount
    appearing. Same exact-match principle as harness/guardrails.py. Max 2.
    """
    pts = 0
    if order_id in draft_text:
        pts += 1
    if f"{amount:,.0f}" in draft_text or f"{amount:.2f}" in draft_text or f"{int(amount)}" in draft_text:
        pts += 1
    return pts


def score_tone(draft_text: str) -> int:
    """LLM-as-judge (Groq) — professional supply-chain email tone, 1-5. Explicitly
    the subjective axis the deterministic verifier can't cover; used directional,
    not as a hard gate (LLM judges carry length/position/self bias).
    """
    global _key_i
    from litellm import completion

    rubric = (
        "Rate this supply-chain exception email's professionalism and tone on a 1-5 "
        "integer scale (5=firm, clear, appropriately-toned ops-manager email; "
        "1=unprofessional/rambling/wrong tone). Reply with ONLY the integer."
    )
    keys = _GROQ_KEYS or [None]
    for _ in range(len(keys) * 5):
        try:
            resp = completion(
                model="groq/llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": rubric}, {"role": "user", "content": draft_text}],
                api_key=keys[_key_i % len(keys)],
            )
            m = re.search(r"[1-5]", resp.choices[0].message.content)
            return int(m.group()) if m else 0
        except Exception as e:
            if "rate" not in str(e).lower() and "429" not in str(e):
                raise
            _key_i += 1          # rotate to the next key
            time.sleep(2)        # respect Groq's "try again in 2s"
    raise RuntimeError("Groq rate limit: all keys exhausted during tone scoring — rerun later.")


# stage -> draft column, in pipeline order. Older files stored GRPO as "ft_draft".
_STAGE_COLS = [("BASE", "base_draft"), ("SFT", "sft_draft"), ("ORPO", "orpo_draft"), ("GRPO", "grpo_draft")]


def main() -> None:
    if not DRAFTS_FILE.exists():
        raise SystemExit(
            f"{DRAFTS_FILE} not found. Generate drafts first — locally via "
            "scripts/gen_eval_drafts_local.py, or the Kaggle cell in this file's docstring."
        )
    drafts = json.loads(DRAFTS_FILE.read_text())
    for row in drafts:  # accept legacy files that stored GRPO under "ft_draft"
        if "ft_draft" in row and "grpo_draft" not in row:
            row["grpo_draft"] = row["ft_draft"]

    # only score stages actually present in the data (base vs GRPO, or the full ladder)
    stages = [(label, col) for label, col in _STAGE_COLS if col in drafts[0]]
    agg = {label: {"facts": 0, "tone": 0} for label, _ in stages}
    for row in drafts:
        for label, col in stages:
            agg[label]["facts"] += score_facts(row[col], row["order_id"], row["amount"])
            agg[label]["tone"] += score_tone(row[col])

    n = len(drafts)
    print(f"Evaluated {n} held-out contexts\n")
    print(f"{'':12} {'facts (/2)':>12} {'tone (/5)':>12} {'d-facts':>10} {'d-tone':>10}")
    base_f = agg["BASE"]["facts"] / n
    base_t = agg["BASE"]["tone"] / n
    for label, _ in stages:
        f, t = agg[label]["facts"] / n, agg[label]["tone"] / n
        df = "" if label == "BASE" else f"{f - base_f:+.2f}"
        dt = "" if label == "BASE" else f"{t - base_t:+.2f}"
        print(f"{label:12} {f:>12.2f} {t:>12.2f} {df:>10} {dt:>10}")
    print("\n(d = change vs BASE. Facts is the axis the RLVR reward optimized; tone is the LLM-judge axis.)")


if __name__ == "__main__":
    main()
