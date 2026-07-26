"""Central configuration — model selection, provider keys, and business thresholds.

One place every other module reads from, so switching LLM provider or tuning an
escalation threshold is a config/.env change, never a code edit. Three concerns live
here:

  1. Provider + model resolution — MODEL is the litellm model string every call uses, and
     get_model() wraps it for ADK. The active provider is chosen once (see below), so no call
     site has a per-provider branch.
  2. Key rotation — up to three <PROVIDER>_API_KEY_* values. litellm resolves the provider's
     key from the environment fresh on every request, so rotating which value sits in
     os.environ[<env var>] moves the next call to a different key with no rebuild. rotate_key()
     is called after a rate-limit error in agents/runner_utils.py and returns False once every
     key is exhausted.
  3. Policy thresholds — the exact numbers harness/policy.py enforces, kept here so they're
     env-overridable and visible in one spot rather than buried in the rules.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini 2.0 Flash was shut down 2026-06-01 and 2.5 Flash retires 2026-10-16, so the
# live path targets 3.5 Flash (released 2026-05-19). Flash-Lite is the cheaper/faster
# tier — useful if a high-volume caller ever needs it.
GEMINI_MODEL_FAST = "gemini-3.5-flash"
GEMINI_MODEL_LITE = "gemini-3.1-flash-lite"

# --- Provider selection --------------------------------------------------------------------
# Two hosted providers reach the same open models through litellm; the harness is deliberately
# provider-agnostic, so which one runs a sweep is a config choice, not a code change. Each entry
# is (default litellm model, the env var litellm reads the API key from).
#
#   groq       — fastest, but free tier caps at 100k tokens/DAY/org. Fine for small runs; a full
#                200-run pass^k sweep needs millions of tokens and does not fit.
#   openrouter — one API over many providers. Free (:free) models are capped at ~50 req/DAY per
#                ACCOUNT (not per key — three keys on one account share it) and their upstream
#                endpoints are congested. ~$10 of credit lifts this to 1000 req/day AND unlocks
#                reliable paid model endpoints (the real unlock for a full sweep).
#
# Pick the best tool-calling model your budget allows via LLM_MODEL. Verified working free on
# OpenRouter: nvidia/nemotron-3-super-120b-a12b:free. Recommended once credited:
# meta-llama/llama-3.3-70b-instruct (continuity with the Groq runs) or deepseek/deepseek-chat.
_PROVIDERS = {
    "groq": ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "openrouter": ("openrouter/nvidia/nemotron-3-super-120b-a12b:free", "OPENROUTER_API_KEY"),
    # Gemini via litellm's Google AI Studio path (needs an AIza-prefixed key, not an AQ. Vertex
    # one). Pay-as-you-go, no minimum deposit — a full sweep is ~$1-3 of real usage on Flash.
    # OPT-IN ONLY: selected solely by LLM_PROVIDER=gemini, never auto-detected, so a Gemini key
    # sitting in .env is never spent without an explicit choice (the standing "ask first" rule).
    "gemini": ("gemini/gemini-3.5-flash", "GEMINI_API_KEY"),
}

# Explicit LLM_PROVIDER wins; otherwise prefer OpenRouter when its key is present (the user set
# it up deliberately), else Groq. Never a silent default to a provider whose key isn't there.
LLM_PROVIDER = (
    os.environ.get("LLM_PROVIDER", "").lower()
    or ("openrouter" if os.environ.get("OPENROUTER_API_KEY") else "groq")
)
if LLM_PROVIDER not in _PROVIDERS:
    raise RuntimeError(f"LLM_PROVIDER={LLM_PROVIDER!r} is not one of {list(_PROVIDERS)}.")

_DEFAULT_MODEL, _KEY_ENV = _PROVIDERS[LLM_PROVIDER]

# The one model string every litellm call uses. Override per-run with LLM_MODEL — e.g.
#   LLM_MODEL=openrouter/meta-llama/llama-3.3-70b-instruct python -m eval.runner
MODEL = os.environ.get("LLM_MODEL") or _DEFAULT_MODEL
GROQ_MODEL = _PROVIDERS["groq"][0]  # kept for callers/tests that name Groq explicitly

# Key rotation over <PROVIDER>_API_KEY, _2, _3. litellm reads os.environ[_KEY_ENV] fresh each
# request, so rotating the value there is enough to move the next call to a different key.
_KEYS = [
    v
    for v in (
        os.environ.get(_KEY_ENV),
        os.environ.get(f"{_KEY_ENV}_2"),
        os.environ.get(f"{_KEY_ENV}_3"),
    )
    if v
]
_key_index = 0
if _KEYS:
    os.environ[_KEY_ENV] = _KEYS[0]  # pin to the first key so rotation has a known start


def key_count() -> int:
    return len(_KEYS)


def rotate_key() -> bool:
    """Switch to the next configured key. Returns False once every key has been tried (caller
    stops retrying and raises, or waits and reset_key()s to start the cycle again).
    """
    global _key_index
    _key_index += 1
    if _key_index >= len(_KEYS):
        return False
    os.environ[_KEY_ENV] = _KEYS[_key_index]
    return True


def reset_key() -> None:
    """Rewind to the first key so rotate_key() can cycle again.

    Rate limits are windows, not one-shot quotas: exhausting every key means "all busy right
    now", not "all spent". A long sweep must be able to wait out the window and start the cycle
    again — without this, the first crowded minute would permanently burn every key.
    """
    global _key_index
    _key_index = 0
    if _KEYS:
        os.environ[_KEY_ENV] = _KEYS[0]


def get_model():
    """Returns the model to pass into an ADK Agent/LlmAgent's `model=` argument.

    GEMINI IS OPT-IN, never selected just because GEMINI_API_KEY is present — a key in .env is
    not consent to spend it. Set USE_GEMINI=1 to use it deliberately. Otherwise the active
    provider (LLM_PROVIDER / MODEL) is wrapped for ADK via LiteLlm. ADK speaks Gemini natively,
    so that branch returns a plain model-name string and litellm is never imported.
    """
    if os.environ.get("USE_GEMINI") == "1":
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("USE_GEMINI=1 but GEMINI_API_KEY is not set.")
        return GEMINI_MODEL_FAST
    if _KEYS:
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(model=MODEL)
    raise RuntimeError(
        f"No {_KEY_ENV} set. Set one, or set USE_GEMINI=1 to use Gemini deliberately."
    )


# --- Policy thresholds --------------------------------------------------------------------
# The exact numbers harness/policy.py enforces. Here, not inline in the rules, for two reasons:
# they're env-overridable per deployment (a company sets its own limits without a code change),
# and a reviewer can read every business limit the system has on one screen.
#
# These are the ONLY knobs, and the agent can neither see nor change them. It proposes a tool
# call; policy.py checks it against these. The agent does not decide what is allowed.

# Above this, a refund needs a human. At or below, the agent may issue it unattended.
AUTO_APPROVE_MAX_USD = float(os.environ.get("AUTO_APPROVE_MAX_USD", 200.0))

# Refund ceiling per claim type, as a fraction of what the customer actually paid.
#   never_arrived / order_canceled — they received nothing, so a full refund is defensible.
#   late_delivery                  — they DID receive the goods; the harm is the delay, so this
#                                    is a goodwill credit, not a refund of the item's price.
# Capping this in policy is what stops "so sorry, here's your money back" from being an option
# the model can talk itself into under pressure.
REFUND_CAP_FRACTION = {
    "never_arrived": 1.0,
    "order_canceled": 1.0,
    "late_delivery": 0.25,
}

# Claims on orders older than this are out of window — deny, regardless of merit.
# The dataset is real 2016-2018 Olist data, so the demo clock is pinned (harness/policy.py,
# NOW_ISO) rather than read from wall-clock, which would put every order out of window.
CLAIM_WINDOW_DAYS = int(os.environ.get("CLAIM_WINDOW_DAYS", 90))

# data/ is organised by job — each folder has exactly one:
#   raw/       untouched Kaggle downloads (~300 MB, gitignored)
#   db/        the demo database the harness verifies claims against — small, versioned, and
#              the only part the running app needs (orders.json)
#   datasets/  training + eval data
#   cache/     resumable generation caches (throwaway)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_DIR = os.path.join(DATA_DIR, "db")
