"""
# scripts/train_extractor.py

**What:** GRPO (RLVR) LoRA fine-tune of Qwen2.5-1.5B-Instruct as the claim extractor, with the
held-out accuracy measured before and after training and a paired McNemar test on the delta.
**Why:** recovering {order_id, claim_type} from a messy message is genuinely hard, so the reward
has variance for GRPO to learn from. The reward is verifiable (exact match against ground truth,
the same check resolv/harness/validity.py makes), not judged. Per-case rows are saved so the paired test
can be recomputed later. Constants are duplicated from evals/extractor.py on purpose: this file runs
standalone on Kaggle, where the repo is not installed. The output is the one-line format
`order=<ORD-####|none> type=<...>`, which only the `line` arm of inference/serve_bench.py serves.
**Used by:** the owner, by hand. Kaggle cell:
    !pip install -q "transformers==4.56.1" "trl==0.21.0" peft datasets accelerate
    !pip uninstall -y torchvision torchao        # remove the mismatched preinstalls
    # then run this file. Needs data/datasets/complaint_cases.json.

Output: models/adapters/extractor/latest
"""
import glob
import json
import random
import re
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = "models/adapters/extractor/latest"
TEST_FRACTION = 0.2   # held out of training; the accuracy we print is measured only on these
SEED = 0


# complaint_cases.json on Kaggle (/kaggle/input/...) or in the repo, checked in that order.
def _find_cases() -> Path:
    for hit in glob.glob("/kaggle/input/**/complaint_cases.json", recursive=True):
        return Path(hit)
    return Path("data/datasets/complaint_cases.json")


CASES = _find_cases()
# Must mirror schemas.ClaimType exactly, or the fine-tune learns a label the verifier rejects.
# Hardcoded because this script runs standalone without the repo installed.
CLAIM_TYPES = ["late_delivery", "never_arrived", "order_canceled"]

INSTRUCTION = (
    "Extract the order number and the claim type from this customer message.\n"
    "Reply on ONE line exactly as: order=<ORD-#### or none> type=<one of: "
    + ", ".join(CLAIM_TYPES) + ">\n\nMessage:\n"
)


# Every case as {prompt, order_id_gt, claim_type_gt}; a null order becomes "none".
def _all_rows() -> list[dict]:
    rows = []
    for c in json.loads(CASES.read_text()):
        gt = c["ground_truth"]
        rows.append(
            {
                "prompt": INSTRUCTION + c["message"],
                "order_id_gt": gt["order_id"] or "none",   # None -> "none" so the match is simple
                "claim_type_gt": gt["claim_type"],
            }
        )
    return rows


# Deterministic (train, test) split; test rows never reach training.
def _split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(SEED).shuffle(shuffled)
    n_test = int(len(shuffled) * TEST_FRACTION)
    return shuffled[n_test:], shuffled[:n_test]  # (train, test)


# First ORD-#### the model emits, normalised; "none" if it emits none.
def _parse_order(text: str) -> str:
    m = re.search(r"ORD[-\s]?(\d{3,5})", text, re.IGNORECASE)
    return f"ORD-{m.group(1)}" if m else "none"


# First known claim type in the text, or "".
def _parse_type(text: str) -> str:
    t = text.lower()
    for ct in CLAIM_TYPES:
        if ct in t:
            return ct
    return ""


# +0.5 for the exact order id (including a correct "none"), +0.5 for the claim type.
def extraction_reward(completions, order_id_gt, claim_type_gt, **kwargs) -> list[float]:
    rewards = []
    for text, oid, ct in zip(completions, order_id_gt, claim_type_gt):
        r = 0.0
        if _parse_order(text).lower() == oid.lower():
            r += 0.5
        if _parse_type(text) == ct:
            r += 0.5
        rewards.append(r)
    return rewards


# McNemar (copy of evals/metrics.py::mcnemar); b = base right/tuned wrong, c = the reverse.
def _mcnemar(b: int, c: int) -> tuple[float, bool]:
    if b + c == 0:
        return (0.0, False)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return (chi2, chi2 > 3.841)  # 1 df, alpha=0.05


# Held-out accuracy with greedy decoding; returns per-case both_ok so the paired test is possible.
def measure(model, tok, rows: list[dict]) -> dict:
    model.eval()
    order_ok = type_ok = both_ok = 0
    per_case = []
    for r in rows:
        enc = tok(r["prompt"], return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        o = _parse_order(text).lower() == r["order_id_gt"].lower()
        t = _parse_type(text) == r["claim_type_gt"]
        order_ok += o
        type_ok += t
        both_ok += o and t
        per_case.append(o and t)
    n = len(rows)
    return {"order": order_ok / n, "type": type_ok / n, "both": both_ok / n, "n": n, "per_case": per_case}


# Measure base, train GRPO on the train split, measure tuned, test the paired delta, save rows.
def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype="float16", device_map="cuda")

    train_rows, test_rows = _split(_all_rows())
    print(f"cases: {CASES}  |  {len(train_rows)} train / {len(test_rows)} held-out")

    # Baseline BEFORE any training — the model has never seen these messages and isn't tuned yet.
    base = measure(model, tok, test_rows)
    print(f"BASE   order={base['order']:.3f}  type={base['type']:.3f}  both={base['both']:.3f}")

    config = GRPOConfig(
        output_dir=OUT,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,          # G rollouts scored per prompt
        max_completion_length=40,   # the answer is one short line
        learning_rate=1e-5,
        max_steps=200,
        logging_steps=10,
        fp16=True,
        report_to="none",
    )
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[extraction_reward],
        args=config,
        train_dataset=Dataset.from_list(train_rows),   # train split only; test rows held out
        processing_class=tok,
        peft_config=peft_config,   # LoRA -> saves an adapter (matches OUT), fits a T4
    )
    trainer.train()
    trainer.save_model(OUT)

    # The same held-out messages, now through the tuned model. The delta is the result.
    tuned = measure(trainer.model, tok, test_rows)
    print(f"TUNED  order={tuned['order']:.3f}  type={tuned['type']:.3f}  both={tuned['both']:.3f}")
    print(f"Δ both-right = {tuned['both'] - base['both']:+.3f}   (n={base['n']} held-out messages)")

    # Paired: both measure() loops ran over test_rows in the same order, so per_case[i] is the same message.
    b = sum(1 for was, now in zip(base["per_case"], tuned["per_case"]) if was and not now)
    c = sum(1 for was, now in zip(base["per_case"], tuned["per_case"]) if not was and now)
    chi2, significant = _mcnemar(b, c)
    print(f"McNemar: b(regressed)={b}  c(improved)={c}  chi2={chi2:.3f}  significant@0.05={significant}")

    # Per-case rows for both runs, so the numbers can be re-checked without a second GPU run.
    compare_path = Path("data/eval/extractor_finetune_compare.jsonl")
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    with open(compare_path, "w") as f:
        for i, r in enumerate(test_rows):
            f.write(json.dumps({
                "order_id_gt": r["order_id_gt"], "claim_type_gt": r["claim_type_gt"],
                "base_both_ok": base["per_case"][i], "tuned_both_ok": tuned["per_case"][i],
            }) + "\n")
    print(f"per-case base/tuned rows -> {compare_path}")
    print(f"extractor adapter saved to {OUT}")


if __name__ == "__main__":
    main()
