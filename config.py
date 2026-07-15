"""Central configuration — model selection, provider keys, and business thresholds.

One place every other module reads from, so switching LLM provider or tuning an
escalation threshold is a config/.env change, never a code edit. Three concerns live
here:

  1. Model resolution — get_model() returns what an ADK agent's `model=` expects.
     Groq (via ADK's LiteLlm wrapper) is preferred when a GROQ_API_KEY is set,
     falling back to Gemini otherwise. Provider choice is data, not a code branch
     at each call site.
  2. Groq key rotation — up to three GROQ_API_KEY_* values. litellm reads
     GROQ_API_KEY from the environment fresh on every request, so rotating which
     value sits in os.environ is enough to move the next call to a different key,
     with no agent rebuild. rotate_groq_key() is called after a rate-limit error
     in agents/runner_utils.py and returns False once all keys are exhausted.
  3. Escalation thresholds + never-auto-resolve types — the exact numbers
     harness/escalation.py enforces, kept here so they're env-overridable and
     visible in one spot rather than buried in the decision logic.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini 2.0 Flash was shut down 2026-06-01 and 2.5 Flash retires 2026-10-16, so the
# live path targets 3.5 Flash (released 2026-05-19). Flash-Lite is the cheaper/faster
# tier — useful if a high-volume caller ever needs it.
GEMINI_MODEL_FAST = "gemini-3.5-flash"
GEMINI_MODEL_LITE = "gemini-3.1-flash-lite"
GROQ_MODEL = "groq/llama-3.3-70b-versatile"

# Groq key rotation: litellm (the library ADK's LiteLlm wrapper calls under the
# hood) resolves GROQ_API_KEY from the environment fresh on every request, not
# once at construction time. That means rotating which value sits in
# os.environ["GROQ_API_KEY"] is enough to make the next call use a different key
# — no need to rebuild the agent. See agents/runner_utils.py for where this gets
# called after a rate-limit error.
_GROQ_KEYS = [
    v
    for v in (
        os.environ.get("GROQ_API_KEY"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
    )
    if v
]
_groq_key_index = 0


def groq_key_count() -> int:
    return len(_GROQ_KEYS)


def rotate_groq_key() -> bool:
    """Switches to the next configured Groq key. Returns False once every key
    has already been tried (caller should stop retrying and raise).
    """
    global _groq_key_index
    _groq_key_index += 1
    if _groq_key_index >= len(_GROQ_KEYS):
        return False
    os.environ["GROQ_API_KEY"] = _GROQ_KEYS[_groq_key_index]
    return True


def get_model():
    """Returns the model to pass into an ADK Agent/LlmAgent's `model=` argument.

    Gemini is preferred when GEMINI_API_KEY is set. ADK talks to Gemini **natively**
    (it's Google's own SDK), so this returns a plain model-name string and litellm is
    never imported — keeping the deployed path free of that dependency. litellm only
    ever existed to reach Groq, which ADK doesn't support natively.

    Groq (via ADK's LiteLlm wrapper) remains the fallback when no Gemini key is set.
    That split is deliberate: Gemini is quota-limited, so it serves the LOW-volume
    live path (2 calls per exception), while Groq's free tier absorbs the HIGH-volume
    offline work (the eval judge in scripts/eval_finetune.py, dataset generation).
    Provider choice stays a .env change, not a code change.
    """
    if os.environ.get("GEMINI_API_KEY"):
        return GEMINI_MODEL_FAST
    if _GROQ_KEYS:
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(model=GROQ_MODEL)
    return GEMINI_MODEL_FAST  # no key configured — fails loudly at call time


SEND_MODE = os.environ.get("SEND_MODE", "draft")


def drafter_backend() -> str:
    """Which drafter writes the email: 'groq' (default, the ADK LlmAgent) or
    'finetuned' (the local GRPO LoRA adapter, see agents/drafter_local.py). Read at
    call time so the Streamlit demo can flip it per request via the env var."""
    return os.environ.get("DRAFTER_BACKEND", "groq")

AUTO_RESOLVE_CONFIDENCE_THRESHOLD = float(os.environ.get("AUTO_RESOLVE_CONFIDENCE_THRESHOLD", 0.85))
AUTO_RESOLVE_MAX_RISK_USD = float(os.environ.get("AUTO_RESOLVE_MAX_RISK_USD", 10000.0))

# Exception types that must never auto-resolve, regardless of confidence/revenue.
# Enforced in harness/escalation.py, not just an LLM instruction.
NEVER_AUTO_RESOLVE_TYPES = {"quality_rejection", "supplier_failure"}

# data/ is organised by job — each folder has exactly one:
#   raw/       untouched Kaggle downloads (~300 MB, gitignored)
#   db/        the demo database the harness verifies claims against — small, versioned,
#              and the only part the running app needs (orders, suppliers, contracts)
#   pools/     raw CSV -> filtered exception contexts (intermediate)
#   datasets/  training + eval data (datasets/legacy = the email-drafting era)
#   cache/     resumable generation caches (throwaway)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_DIR = os.path.join(DATA_DIR, "db")
CONTRACTS_DIR = os.path.join(DB_DIR, "contracts")
