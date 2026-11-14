"""
# resolv/config.py

**What:** every model choice, provider key, price and policy threshold, read once from the environment.
**Why:** switching provider, pointing a role at a self-hosted server or tuning a threshold is an `.env` change,
never a code edit. Each role (agent, extractor, simulator, judge) has its own model so one can move to a local
server while the others stay hosted. One key per provider: if a free tier's daily cap is hit, wait, use a paid
tier, or self-host (see docs/inference.md).
**Used by:** resolv/agents/*, resolv/api/*, evals/*, resolv/harness/policy.py, resolv/harness/tools.py, resolv/harness/validity.py.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# dev | test | eval | prod. prod turns on the fail-closed checks (resolv/api/auth.py) and pre-LLM redaction.
RESOLV_ENV = os.environ.get("RESOLV_ENV", "dev")

# Any OpenAI-compatible server (vLLM, llama.cpp). Port 8000 is left for the Resolv API.
SELFHOSTED_BASE_URL = os.environ.get("SELFHOSTED_BASE_URL", "http://localhost:8001/v1")
SELFHOSTED_MODEL = os.environ.get("SELFHOSTED_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# provider -> (default litellm model, env var holding its one key). Gemini is opt-in only: chosen solely by
# LLM_PROVIDER=gemini, so a key sitting in .env is never spent without an explicit choice.
_PROVIDERS = {
    "groq": ("groq/openai/gpt-oss-120b", "GROQ_API_KEY"),
    "openrouter": ("openrouter/nvidia/nemotron-3-super-120b-a12b:free", "OPENROUTER_API_KEY"),
    "gemini": ("gemini/gemini-3.6-flash", "GEMINI_API_KEY"),
    "selfhosted": (f"hosted_vllm/{SELFHOSTED_MODEL}", None),
}

# Explicit LLM_PROVIDER wins; otherwise OpenRouter when its key is set, else Groq.
LLM_PROVIDER = (
    os.environ.get("LLM_PROVIDER", "").lower()
    or ("openrouter" if os.environ.get("OPENROUTER_API_KEY") else "groq")
)
if LLM_PROVIDER not in _PROVIDERS:
    raise RuntimeError(f"LLM_PROVIDER={LLM_PROVIDER!r} is not one of {list(_PROVIDERS)}.")


# "selfhosted:<name>" is shorthand for litellm's "hosted_vllm/<name>"; anything else passes through.
def _resolve(model: str) -> str:
    return f"hosted_vllm/{model.split(':', 1)[1]}" if model.startswith("selfhosted:") else model


_DEFAULT_MODEL = _resolve(os.environ.get("LLM_MODEL") or _PROVIDERS[LLM_PROVIDER][0])

# One litellm model string per role; each env var overrides only its own role. The simulator and judge are
# separate so a model under test is never graded or pushed by itself.
AGENT_MODEL = _resolve(os.environ.get("AGENT_MODEL") or _DEFAULT_MODEL)
EXTRACTOR_MODEL = _resolve(os.environ.get("EXTRACTOR_MODEL") or _DEFAULT_MODEL)
SIMULATOR_MODEL = _resolve(os.environ.get("SIMULATOR_MODEL") or _DEFAULT_MODEL)
JUDGE_MODEL = _resolve(os.environ.get("JUDGE_MODEL") or _DEFAULT_MODEL)
ROLE_MODELS = {"agent": AGENT_MODEL, "extractor": EXTRACTOR_MODEL, "simulator": SIMULATOR_MODEL, "judge": JUDGE_MODEL}

# json = one structured-output completion (default); legacy_line = the GRPO adapter's one-line
# "order=... type=..." format on the completions endpoint; none = no extractor (ablation arm).
EXTRACTOR_BACKEND = os.environ.get("EXTRACTOR_BACKEND", "json")

# USD per million tokens (input, output). Free models and self-hosted tokens cost 0 here; GPU time for
# Check the provider pages before trusting a dollar figure.
PRICES_USD_PER_MTOK = {
    "groq/openai/gpt-oss-120b": (0.15, 0.60),
    "gemini/gemini-3.6-flash": (0.75, 3.75),
    "gemini/gemini-3.5-flash": (1.50, 9.00),
}


# Extra litellm kwargs a model needs: self-hosted servers get their base URL and a placeholder key.
def llm_kwargs(model: str) -> dict:
    if model.startswith("hosted_vllm/"):
        return {"api_base": SELFHOSTED_BASE_URL, "api_key": os.environ.get("SELFHOSTED_API_KEY", "local")}
    return {}


# True when price_usd knows this model's rate (rows can then say "unpriced" instead of a fake $0).
def is_priced(model: str) -> bool:
    return model in PRICES_USD_PER_MTOK or model.startswith("hosted_vllm/") or model.endswith(":free")


# Provider-aware dollar cost of one call; 0.0 for self-hosted, free and unknown models (check is_priced).
def price_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    usd_in, usd_out = PRICES_USD_PER_MTOK.get(model, (0.0, 0.0))
    return (prompt_tokens or 0) / 1e6 * usd_in + (completion_tokens or 0) / 1e6 * usd_out


# --- Policy thresholds: the only knobs resolv/harness/policy.py enforces. The agent can neither see nor change them. ---

# Above this a refund needs a human; at or below, the agent may issue it unattended.
AUTO_APPROVE_MAX_USD = float(os.environ.get("AUTO_APPROVE_MAX_USD", 200.0))
# Refund ceiling per claim type, as a fraction of what was paid. late_delivery is a goodwill credit: the goods arrived.
REFUND_CAP_FRACTION = {
    "never_arrived": 1.0,
    "order_canceled": 1.0,
    "late_delivery": 0.25,
}

# Claims on orders older than this are out of window. The demo clock is pinned in resolv/harness/policy.py (2016-2018 data).
CLAIM_WINDOW_DAYS = int(os.environ.get("CLAIM_WINDOW_DAYS", 90))

# data/raw (Kaggle downloads), data/db (the demo database the harness reads), data/datasets, data/cache.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_DIR = os.path.join(DATA_DIR, "db")
