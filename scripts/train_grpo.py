"""GRPO / RLVR stage — the capstone. Uses TRL's GRPOTrainer with a deterministic
verifier reward, NOT an LLM judge. This is what makes it real RLVR: the reward is
the exact same fact-check as harness/guardrails.py (order ID present, dollar amount
present), computed on each generated rollout at train time.

Why TRL not LlamaFactory: LlamaFactory's RL stages expect a reward *model*. We have
a reward *function* (the verifier), and TRL's GRPOTrainer takes exactly that via
reward_funcs=[callable]. So GRPO is this ~60-line script, not a YAML config.

Runs on Kaggle (where SFT+ORPO already ran and the adapters live). Starts from the
SFT+ORPO adapter and optimizes it against the verifier reward.

Kaggle cell:
    !pip install -q trl "transformers==4.56.1"
    # then run this file, or paste its body into a cell
"""
import json
import re
from pathlib import Path

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
START_ADAPTER = "models/adapters/orpo/latest"  # build on SFT+ORPO
OUT = "models/adapters/grpo/latest"
DATA_DIR = Path("data")


def _load_prompts() -> Dataset:
    """Build GRPO prompts from resolv_sft.json (the same file already on Kaggle) —
    its `instruction` field already contains "Orders: X" and "Revenue at risk: $Y",
    so parse those out for the reward's facts. Avoids needing the raw pool files
    uploaded separately.
    """
    rows = []
    data = json.loads((DATA_DIR / "resolv_sft.json").read_text())
    for ex in data:
        instr = ex["instruction"]
        oid_m = re.search(r"Orders:\s*([A-Za-z0-9\-]+)", instr)
        amt_m = re.search(r"\$([\d,]+\.?\d*)", instr)
        if not oid_m or not amt_m:
            continue
        rows.append({
            "prompt": instr + "\nDraft a professional resolution email.",
            "order_id": oid_m.group(1),
            "amount": float(amt_m.group(1).replace(",", "")),
        })
    return Dataset.from_list(rows)


def verifier_reward(completions, order_id, amount, **kwargs) -> list[float]:
    """Deterministic reward in [0,1] — the SAME check harness/guardrails.py runs in
    production. Half credit for the order ID appearing, half for the dollar amount.
    No LLM judgment anywhere: this is why it's RLVR, not RLHF.
    """
    rewards = []
    for text, oid, amt in zip(completions, order_id, amount):
        r = 0.0
        if oid in text:
            r += 0.5
        if f"{amt:,.0f}" in text or f"{amt:.2f}" in text or f"{int(amt)}" in text:
            r += 0.5
        rewards.append(r)
    return rewards


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype="float16", device_map="cuda")
    model = PeftModel.from_pretrained(model, START_ADAPTER, is_trainable=True)

    config = GRPOConfig(
        output_dir=OUT,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,          # G: rollouts scored per prompt
        max_completion_length=200,
        learning_rate=1e-5,
        max_steps=100,  # GRPO's generation sampler gives no dataloader length, so set steps directly
        logging_steps=5,
        fp16=True,
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[verifier_reward],
        args=config,
        train_dataset=_load_prompts(),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(OUT)
    print(f"GRPO adapter saved to {OUT}")


if __name__ == "__main__":
    main()
