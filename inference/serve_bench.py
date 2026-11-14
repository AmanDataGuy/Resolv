"""
# inference/serve_bench.py

**What:** extractor accuracy plus client-side TTFT/latency/throughput for a self-hosted model on
the 92-case held-out split, with a non-LLM regex arm as the bar every model must beat.
**Why:** every arm is scored by `evals.extractor.score()` on `evals.extractor.load_cases("test")`,
so results sit next to the hosted-API runs of `python -m evals.extractor`. Two prompt formats:
`line` sends the GRPO training prompt raw to /v1/completions (greedy, 40 tokens: the exact setup
the adapter was measured under in scripts/train_extractor.py); `json` sends the production
instruction as chat with the CustomerClaim JSON schema enforced (the SFT and zero-shot arms).
Each run is compared paired (McNemar) against the regex arm's rows when they exist.
**Used by:** the owner by hand (docs/inference.md); tests/test_serve_bench.py.

    python -m inference.serve_bench --backend regex --label regex
    python -m inference.serve_bench --model grpo --label vllm-grpo -c 1
    python -m inference.serve_bench --base-url http://localhost:8002/v1 --model local --label llamacpp-q4
    python -m inference.serve_bench --format json --model Qwen/Qwen2.5-1.5B-Instruct --label vllm-base-json
"""
import argparse
import asyncio
import json
import math
import re
import subprocess
import time
from pathlib import Path

from openai import AsyncOpenAI

from evals.extractor import _norm, compare, load_cases, score

OUT_DIR = Path(__file__).parent / "results" / "extractor"

# Must match scripts/train_extractor.py (that script runs standalone on Kaggle, so it can't be imported).
CLAIM_TYPES = ["late_delivery", "never_arrived", "order_canceled"]
INSTRUCTION = (
    "Extract the order number and the claim type from this customer message.\n"
    "Reply on ONE line exactly as: order=<ORD-#### or none> type=<one of: "
    + ", ".join(CLAIM_TYPES) + ">\n\nMessage:\n"
)


# First ORD-#### in the text, or "none".
def parse_order(text: str) -> str:
    m = re.search(r"ORD[-\s]?(\d{3,5})", text, re.IGNORECASE)
    return f"ORD-{m.group(1)}" if m else "none"


# First known claim type in the text, or "".
def parse_type(text: str) -> str:
    t = text.lower()
    return next((ct for ct in CLAIM_TYPES if ct in t), "")


# (order, claim) from a one-line answer.
def parse_line(text: str) -> tuple[str, str]:
    return _norm(parse_order(text)), parse_type(text)


# (order, claim) from a CustomerClaim JSON answer; raises ValueError on invalid JSON or schema.
def parse_json(text: str) -> tuple[str, str]:
    from resolv.schemas import CustomerClaim
    claim = CustomerClaim.model_validate_json(text)
    return _norm(claim.order_id), claim.claim_type or ""


# --- Regex arm: the non-LLM floor. Rules written against the train split only. ---------------
_WORDS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4",
          "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_FILLER = {"r", "d", "rd", "ord", "um", "umm", "ummm", "uh", "like", "is", "its", "number", "num", "numbr",
           "nmb", "was", "it", "are", "ar", "dee", "ordder"}
_DECOY = re.compile(r"(other|another|different|friend|second|last year|previous|earlier|"
                    r"whole other|not what|thats fine|was fine|not whats)", re.I)


# Up to four digits (numerals or spelled out) right after an order-number anchor.
def _digits_after(text: str, start: int) -> str:
    out = ""
    for tok in re.findall(r"[a-z]+|\d", text[start:start + 40]):
        if tok.isdigit():
            out += tok
        elif tok in _WORDS and not (tok == "o" and not out):  # the "o" of o-r-d is not a zero
            out += _WORDS[tok]
        elif tok in _FILLER:
            continue
        elif out:
            break
        if len(out) == 4:
            break
    return out


# Rule-based extraction in the same one-line format the GRPO adapter emits.
def regex_extract(msg: str) -> str:
    m = msg.lower()
    order = "none"
    for a in re.finditer(r"(?:o[\s\-.]*r[\s\-.]*d|0\s*0\s*r\s*d|order\s+(?:number|num|#)?)", m):
        d = _digits_after(m, a.end())
        if len(d) == 4 and not _DECOY.search(m[max(0, a.end() - 60):a.end()]):
            order = f"ORD-{d}"
            break
    if "cancel" in m:
        claim = "order_canceled"
    elif re.search(r"(never (arrived|came|got|showed|received)|still (havent|haven't|have not|hasnt|waiting|not)|"
                   r"hasn'?t (arrived|come|shown)|not (arrived|received)|where is|never got)", m) \
            and not re.search(r"\b(until|til|till)\b", m):
        claim = "never_arrived"
    else:
        claim = "late_delivery"
    return f"order={order} type={claim}"


# --- Server arm -------------------------------------------------------------------------------
# One streamed request in the chosen format; returns (text, ttft_s, output_tokens).
async def _complete(client: AsyncOpenAI, model: str, message: str, fmt: str) -> tuple[str, float | None, int]:
    started = time.perf_counter()
    ttft, text, chunks, usage_tokens = None, "", 0, None
    if fmt == "line":
        stream = await client.completions.create(
            model=model, prompt=INSTRUCTION + message, max_tokens=40, temperature=0.0, stream=True,
            stream_options={"include_usage": True})
    else:
        from resolv.agents.extractor import EXTRACTOR_INSTRUCTION
        from resolv.schemas import CustomerClaim
        stream = await client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=64, stream=True, stream_options={"include_usage": True},
            messages=[{"role": "system", "content": EXTRACTOR_INSTRUCTION}, {"role": "user", "content": message}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "CustomerClaim", "schema": CustomerClaim.model_json_schema()}},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})  # Qwen3: thinking off
    async for chunk in stream:
        if chunk.usage:
            usage_tokens = chunk.usage.completion_tokens
        piece = (chunk.choices[0].text if fmt == "line" else chunk.choices[0].delta.content) if chunk.choices else None
        if piece:
            if ttft is None:
                ttft = time.perf_counter() - started
            text += piece
            chunks += 1
    # ponytail: chunk count stands in for tokens when the server sends no usage (one token per chunk on vLLM and llama.cpp).
    return text, ttft, usage_tokens or chunks


# One scored row in evals.extractor's row shape plus serving fields.
def _row(case: dict, text: str, latency_s: float, ttft_s, tokens: int, error: str | None, fmt: str = "line") -> dict:
    truth_order, truth_claim = _norm(case["ground_truth"]["order_id"]), case["ground_truth"]["claim_type"]
    got_order, got_claim = "", ""
    if error is None:
        try:
            got_order, got_claim = parse_line(text) if fmt == "line" else parse_json(text)
        except ValueError as e:  # invalid JSON is a counted error, never a silent default
            error = f"invalid_output: {type(e).__name__}"
    return {
        "case_id": case["case_id"], "difficulty": case["difficulty"],
        "truth_order": truth_order, "truth_claim": truth_claim,
        "got_order": got_order, "got_claim": got_claim, "raw": text,
        "order_ok": got_order == truth_order, "claim_ok": got_claim == truth_claim,
        "both_ok": got_order == truth_order and got_claim == truth_claim,
        "hallucinated": (got_order not in ("none", "")) if truth_order == "none" else None,
        "latency_s": round(latency_s, 4), "ttft_s": None if ttft_s is None else round(ttft_s, 4),
        "output_tokens": tokens, "usd": 0.0,
        **({"error": error} if error else {}),
    }


# All cases against the server at a fixed concurrency; returns (rows, wall_s).
async def _run_server(cases: list[dict], base_url: str, model: str, concurrency: int, fmt: str) -> tuple[list[dict], float]:
    client = AsyncOpenAI(base_url=base_url, api_key="local")
    gate = asyncio.Semaphore(concurrency)

    # One case under the concurrency gate; failures become error rows.
    async def one(case: dict) -> dict:
        async with gate:
            started = time.perf_counter()
            try:
                text, ttft, tokens = await _complete(client, model, case["message"], fmt)
                return _row(case, text, time.perf_counter() - started, ttft, tokens, None, fmt)
            except Exception as e:
                return _row(case, "", time.perf_counter() - started, None, 0, f"{type(e).__name__}: {e}", fmt)

    # One untimed request first so model load / CUDA graph capture doesn't land in the numbers.
    await _complete(client, model, cases[0]["message"], fmt)
    started = time.perf_counter()
    rows = await asyncio.gather(*(one(c) for c in cases))
    return list(rows), time.perf_counter() - started


# Nearest-rank percentile (p in 0..1) of the non-None values, or None if there are none.
def pct(values: list, p: float):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return round(values[max(0, min(len(values) - 1, math.ceil(p * len(values)) - 1))], 4)


# Total GPU memory in use now; vLLM preallocates, so read it as footprint, not need.
def vram_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:
        return None


# CLI: run one arm, write card + rows, print the paired comparison against the regex arm.
def main() -> None:
    ap = argparse.ArgumentParser(description="Score a self-hosted extractor and time it.")
    ap.add_argument("--backend", choices=["server", "regex"], default="server")
    ap.add_argument("--format", choices=["line", "json"], default="line",
                    help="line = GRPO prompt over /v1/completions; json = production prompt, schema-constrained chat")
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="grpo", help="served model name (vLLM LoRA name, or anything for llama.cpp)")
    ap.add_argument("-c", "--concurrency", type=int, default=1)
    ap.add_argument("--label", required=True, help="names the output file, e.g. vllm-grpo-c8")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compare", default="regex", help="label of a previous run to compare against, paired")
    args = ap.parse_args()

    cases = load_cases("test")[: args.limit or None]
    if args.backend == "regex":
        started = time.perf_counter()
        rows = []
        for c in cases:
            t0 = time.perf_counter()
            rows.append(_row(c, regex_extract(c["message"]), time.perf_counter() - t0, None, 0, None))
        wall = time.perf_counter() - started
    else:
        rows, wall = asyncio.run(_run_server(cases, args.base_url, args.model, args.concurrency, args.format))

    ok = [r for r in rows if not r.get("error")]
    card = score(rows)
    card["serving"] = {
        "backend": args.backend, "format": args.format, "model": args.model,
        "base_url": args.base_url if args.backend == "server" else None,
        "concurrency": args.concurrency, "wall_s": round(wall, 3),
        "requests_per_s": round(len(ok) / wall, 3) if wall else None,
        "output_tokens_per_s": round(sum(r["output_tokens"] for r in ok) / wall, 1) if wall else None,
        "ttft_p50_s": pct([r["ttft_s"] for r in ok], 0.5), "ttft_p95_s": pct([r["ttft_s"] for r in ok], 0.95),
        "latency_p50_s": pct([r["latency_s"] for r in ok], 0.5),
        "latency_p95_s": pct([r["latency_s"] for r in ok], 0.95),
        "vram_used_mib": vram_mib() if args.backend == "server" else None,
    }

    bar = OUT_DIR / f"serve_{args.compare}.json"
    if args.label != args.compare and bar.exists():
        prev = json.loads(bar.read_text(encoding="utf-8"))
        card["vs_" + args.compare] = compare(rows, {"rows": prev["rows"], "score": prev["card"]})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"serve_{args.label}.json"
    out.write_text(json.dumps({"card": card, "rows": rows}, indent=2), encoding="utf-8")

    print(f"{args.label}: n={card['n']} both={card['both_acc']} ci95={card['both_ci95']} "
          f"halluc={card['hallucination_rate']} errors={card['errors']}")
    for k, v in card["serving"].items():
        print(f"  {k:<20} {v}")
    if "vs_" + args.compare in card:
        print(f"  vs {args.compare}: {card['vs_' + args.compare]}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
