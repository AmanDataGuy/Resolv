"""Builds the SFT + ORPO training pairs from the ingested exception pools.

Makes real Groq API calls — one "good" draft and one deliberately "bad" draft per
context — so a full run processes up to MAX_PER_TYPE contexts per type (hundreds of
calls) and spends real rate-limit budget even on Groq's free tier. A completed run
produced 190 SFT pairs and 190 ORPO pairs (data/resolv_sft.json, data/resolv_orpo.json),
which is what the fine-tuning stack trains on.

Pipeline: load the 4 real-data pools written by scripts/ingest_*.py, plus a small
hardcoded synthetic pool for customs_hold/supplier_failure (no public dataset exists
for these two types) -> for each context, ask Groq for a good draft and a
deliberately bad one -> score the good draft with a deterministic verifier (NOT an
LLM judge — see harness/guardrails.py's docstring for why that distinction is the
whole point of RLVR) -> bucket into SFT (good drafts scoring >= MIN_SFT_SCORE) and
ORPO chosen/rejected pairs (good vs bad, when the good one clears MIN_ORPO_CHOSEN_SCORE).

Resumable: every (context -> good draft, bad draft, score) result is cached to
data/build_results.json, so an interrupted run — a rate-limit stall, a killed
process — resumes without re-spending API calls on contexts already done.
"""
import json
import os
import time
from pathlib import Path

from litellm import completion
from tqdm import tqdm

# Groq is 30 req/min per key. Rotate across the 3 configured keys and back off on
# a rate-limit error — the cache means progress is never lost even if it stalls.
_GROQ_KEYS = [
    k for k in (os.environ.get("GROQ_API_KEY"), os.environ.get("GROQ_API_KEY_2"), os.environ.get("GROQ_API_KEY_3")) if k
]
_key_i = 0

DATA_DIR = Path(__file__).parent.parent / "data"
POOLS_DIR = DATA_DIR / "pools"
OUT_DIR = DATA_DIR / "datasets" / "legacy"   # the email-drafting era, superseded by complaint_cases
CACHE_FILE = DATA_DIR / "cache" / "build_results.json"

MAX_PER_TYPE = 50  # real run: ~300 contexts, ~600 Groq calls, resumable via build_results.json
MIN_SFT_SCORE = 65
MIN_ORPO_CHOSEN_SCORE = 80

GOOD_DRAFTER_PROMPT = """You are a senior supply chain operations manager.
Draft a professional, firm resolution email for this exception.
Include exact order/product identifiers and dollar amounts ONLY if given in the
context below. Do not invent any information not provided."""

BAD_DRAFTER_PROMPT = """You are a poorly-trained supply chain assistant.
Draft a flawed version of a supply chain notification email. Make at least
three of these mistakes: invent amounts not in the context, reference wrong
identifiers, use passive-aggressive tone ("I'm afraid I must unfortunately
inform you..."), omit critical dollar amounts, make promises not supported by
any contract."""

# No public dataset exists for these two exception types — these are hardcoded
# synthetic scenario templates, not LLM-generated contexts. Only the DRAFT
# generation below ever calls an LLM; this scenario list is fixed data.
SYNTHETIC_CONTEXTS = [
    {
        "exception_type": "customs_hold",
        "supplier_id": "acme-logistics",
        "order_ids": ["ORD-9101"],
        "revenue_at_risk_usd": 22000.0,
        "source": "synthetic",
    },
    {
        "exception_type": "customs_hold",
        "supplier_id": "northwind-parts",
        "order_ids": ["ORD-9102"],
        "revenue_at_risk_usd": 8000.0,
        "source": "synthetic",
    },
    {
        "exception_type": "supplier_failure",
        "supplier_id": "northwind-parts",
        "order_ids": ["ORD-9201"],
        "revenue_at_risk_usd": 30000.0,
        "source": "synthetic",
    },
    {
        "exception_type": "supplier_failure",
        "supplier_id": "acme-logistics",
        "order_ids": ["ORD-9202"],
        "revenue_at_risk_usd": 5000.0,
        "source": "synthetic",
    },
]


def _context_key(context: dict) -> str:
    ids = context.get("order_ids") or context.get("product_ids") or [context.get("supplier_id", "unknown")]
    return f"{context['exception_type']}:{context['source']}:{ids[0]}"


def _context_str(context: dict) -> str:
    parts = [f"Exception type: {context['exception_type']}"]
    if context.get("supplier_id"):
        parts.append(f"Supplier: {context['supplier_id']}")
    if context.get("order_ids"):
        parts.append(f"Orders: {', '.join(context['order_ids'])}")
    if context.get("product_ids"):
        parts.append(f"Products: {', '.join(str(p) for p in context['product_ids'])}")
    if context.get("revenue_at_risk_usd") is not None:
        parts.append(f"Revenue at risk: ${context['revenue_at_risk_usd']:.2f}")
    return " | ".join(parts)


def _score_draft(context: dict, draft_text: str) -> int:
    """Deterministic verifier, matching harness/guardrails.py's principle
    (exact-match checks, not an LLM's opinion) adapted for these heterogeneous
    ingestion-pool contexts rather than a full schemas.ResolutionDraft object.
    """
    score = 40  # baseline for a non-empty, on-topic draft
    order_ids = context.get("order_ids") or []
    if order_ids and all(oid in draft_text for oid in order_ids):
        score += 25
    elif not order_ids:
        score += 15  # nothing to check against (e.g. product-only contexts)

    revenue = context.get("revenue_at_risk_usd")
    if revenue is not None:
        if f"{revenue:,.0f}" in draft_text or f"{revenue:.2f}" in draft_text:
            score += 25
    else:
        score += 15

    if len(draft_text) > 40:
        score += 10

    return min(score, 100)


def _load_pool(filename: str) -> list[dict]:
    path = POOLS_DIR / filename
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _generate(prompt: str, context_str: str) -> str:
    global _key_i
    for _ in range(len(_GROQ_KEYS) * 3):
        try:
            response = completion(
                model="groq/llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": context_str}],
                api_key=_GROQ_KEYS[_key_i % len(_GROQ_KEYS)],
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" not in str(e).lower() and "429" not in str(e):
                raise
            _key_i += 1  # next key
            time.sleep(2)  # respect Groq's "try again in 2s"
    raise RuntimeError("Groq rate limit: all keys exhausted after retries — rerun later, cache resumes.")


def main() -> None:
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}

    all_contexts = (
        _load_pool("exceptions_pool_late_shipment.json")[:MAX_PER_TYPE]
        + _load_pool("exceptions_pool_price_dispute.json")[:MAX_PER_TYPE]
        + _load_pool("exceptions_pool_quality_rejection.json")[:MAX_PER_TYPE]
        + _load_pool("exceptions_pool_stockout.json")[:MAX_PER_TYPE]
        + SYNTHETIC_CONTEXTS
    )

    sft_pairs, orpo_pairs = [], []

    for context in tqdm(all_contexts, desc="Generating draft pairs"):
        key = _context_key(context)
        if key not in cache:
            context_str = _context_str(context)
            good = _generate(GOOD_DRAFTER_PROMPT, context_str)
            bad = _generate(BAD_DRAFTER_PROMPT, context_str)
            score = _score_draft(context, good)
            cache[key] = {"context_str": context_str, "good": good, "bad": bad, "score": score}
            CACHE_FILE.write_text(json.dumps(cache, indent=2))

        entry = cache[key]
        if entry["score"] >= MIN_SFT_SCORE:
            sft_pairs.append({"instruction": entry["context_str"], "input": "", "output": entry["good"]})
        if entry["score"] >= MIN_ORPO_CHOSEN_SCORE:
            orpo_pairs.append({"prompt": entry["context_str"], "chosen": entry["good"], "rejected": entry["bad"]})

    (OUT_DIR / "resolv_sft.json").write_text(json.dumps(sft_pairs, indent=2))
    (OUT_DIR / "resolv_orpo.json").write_text(json.dumps(orpo_pairs, indent=2))

    print(f"Processed {len(all_contexts)} contexts")
    print(f"SFT pairs: {len(sft_pairs)}")
    print(f"ORPO pairs: {len(orpo_pairs)}")


if __name__ == "__main__":
    main()
