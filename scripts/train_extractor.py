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
import json
import re
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = "models/adapters/extractor/latest"
CASES = Path("data/datasets/complaint_cases.json")
CLAIM_TYPES = ["late_delivery", "never_arrived", "order_canceled", "item_unavailable"]

INSTRUCTION = (
    "Extract the order number and the claim type from this customer message.\n"
    "Reply on ONE line exactly as: order=<ORD-#### or none> type=<one of: "
    + ", ".join(CLAIM_TYPES) + ">\n\nMessage:\n"
)


def _load_prompts() -> Dataset:
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
    return Dataset.from_list(rows)


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


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype="float16", device_map="cuda")

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
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[extraction_reward],
        args=config,
        train_dataset=_load_prompts(),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(OUT)
    print(f"extractor adapter saved to {OUT}")


if __name__ == "__main__":
    main()
