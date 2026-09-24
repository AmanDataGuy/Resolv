"""GRPO / RLVR training for the CLAIM EXTRACTOR — the second, harder fine-tune.

WHY THIS ONE HAS HEADROOM. The first fine-tune (train_grpo.py) trained the drafter, which was
handed the facts and only had to echo them — the reward saturated (study/finetuning_report.md
sec 6.4, 12). This one trains the extractor to recover {order_id, claim_type} from a MESSY
customer message, where the answer is genuinely hard to find. So the reward has real variance,
which is what GRPO needs to learn.

THE REWARD IS VERIFIABLE, NOT JUDGED. Each rollout's extracted order id + claim type are checked
by exact match against the known ground truth — the same check harness/validity.py performs in
production. No LLM judge. That reuse is the whole thesis.

Runs on Kaggle (GPU). Kaggle cell:
    !pip install -q "transformers==4.56.1" "trl==0.21.0" peft datasets accelerate
    !pip uninstall -y torchvision torchao        # remove the mismatched preinstalls (see the report)
    # then paste this file's body, or upload it and run it. Needs data/datasets/complaint_cases.json.

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


def _find_cases() -> Path:
    """complaint_cases.json, wherever it lives. On Kaggle it's under /kaggle/input/<dataset>/;
    locally it's in the repo. Checked in that order so the same file runs in both places."""
    for hit in glob.glob("/kaggle/input/**/complaint_cases.json", recursive=True):
        return Path(hit)
    return Path("data/datasets/complaint_cases.json")


CASES = _find_cases()
# Must mirror schemas.ClaimType exactly — the training prompt has to offer the same claim types
# the harness accepts, or the fine-tune learns a label the verifier will always reject. Hardcoded
# (not imported from schemas) because this script is meant to run standalone in a Kaggle/Colab
# notebook where the repo isn't installed. item_unavailable was dropped; keep these three in sync.
CLAIM_TYPES = ["late_delivery", "never_arrived", "order_canceled"]

INSTRUCTION = (
    "Extract the order number and the claim type from this customer message.\n"
    "Reply on ONE line exactly as: order=<ORD-#### or none> type=<one of: "
    + ", ".join(CLAIM_TYPES) + ">\n\nMessage:\n"
)


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


def _split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic train/test split. The test rows are held out of training entirely, so the
    accuracy we print is on messages the model never saw — the only kind of number worth a CV."""
    shuffled = list(rows)
    random.Random(SEED).shuffle(shuffled)
    n_test = int(len(shuffled) * TEST_FRACTION)
    return shuffled[n_test:], shuffled[:n_test]  # (train, test)


def _parse_order(text: str) -> str:
    """First ORD-#### the model emits, normalised; 'none' if it says none / emits nothing."""
    m = re.search(r"ORD[-\s]?(\d{3,5})", text, re.IGNORECASE)
    return f"ORD-{m.group(1)}" if m else "none"


def _parse_type(text: str) -> str:
    t = text.lower()
    for ct in CLAIM_TYPES:
        if ct in t:
            return ct
    return ""


def extraction_reward(completions, order_id_gt, claim_type_gt, **kwargs) -> list[float]:
    """+0.5 for the exact order id (incl. correctly saying 'none'), +0.5 for the claim type.
    Identical logic to harness/validity.py's inputs — deterministic, verifiable, no judge."""
    rewards = []
    for text, oid, ct in zip(completions, order_id_gt, claim_type_gt):
        r = 0.0
        if _parse_order(text).lower() == oid.lower():
            r += 0.5
        if _parse_type(text) == ct:
            r += 0.5
        rewards.append(r)
    return rewards


def _mcnemar(b: int, c: int) -> tuple[float, bool]:
    """Duplicated from eval/metrics.py::mcnemar() on purpose — this script has to run standalone
    in a Kaggle notebook where the repo isn't installed (same reasoning eval/extractor.py gives
    for duplicating the split constants). b = base-right/tuned-wrong, c = the reverse."""
    if b + c == 0:
        return (0.0, False)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return (chi2, chi2 > 3.841)  # 1 df, alpha=0.05


def measure(model, tok, rows: list[dict]) -> dict:
    """Extraction accuracy on held-out rows — order id, claim type, and both-right. Greedy decode,
    the same parse the reward uses. This is THE number: run it on the base model and again on the
    tuned one, on messages neither was trained on, and the delta is what RLVR bought.

    Returns per-case `both_ok` alongside the aggregate, not just the aggregate. A prior fine-tune
    on this project logged only aggregates and the resulting "+0.016" claim needed a McNemar test
    it could never get, because b and c (which specific cases flipped) were unrecoverable — see
    eval/runner.py's docstring for the full story. Returning rows here is what lets main() compute
    a real paired test the same run, rather than repeating that mistake in a second script.
    """
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

    # Paired, not just a delta: b = base got it right and tuned lost it, c = the reverse. Both
    # loops ran over test_rows IN THE SAME ORDER, so per_case[i] in each list refers to the same
    # held-out message — that pairing is what makes McNemar valid here rather than two unrelated
    # accuracy numbers.
    b = sum(1 for was, now in zip(base["per_case"], tuned["per_case"]) if was and not now)
    c = sum(1 for was, now in zip(base["per_case"], tuned["per_case"]) if not was and now)
    chi2, significant = _mcnemar(b, c)
    print(f"McNemar: b(regressed)={b}  c(improved)={c}  chi2={chi2:.3f}  significant@0.05={significant}")

    # Save per-case rows for BOTH runs — not just the printed aggregate. If this run's numbers
    # ever need re-checking later (a different rubric, a disputed claim), the rows make that
    # possible without a second GPU run; the aggregate alone would not.
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
