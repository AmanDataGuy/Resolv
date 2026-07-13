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

GEMINI_MODEL_FAST = "gemini-2.0-flash"
GEMINI_MODEL_SMART = "gemini-2.5-pro"
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

    Groq is preferred when GROQ_API_KEY is set: it's an OpenAI-compatible provider
    reached through ADK's LiteLlm wrapper (google.adk.models.lite_llm.LiteLlm),
    same pattern as the reference course's Module 5 openrouter_agent/ollama_agent
    examples. Falling back to a plain "gemini-2.0-flash" string only when no Groq
    key is configured means switching providers is a .env change, not a code change.
    """
    if _GROQ_KEYS:
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(model=GROQ_MODEL)
    return GEMINI_MODEL_FAST


SEND_MODE = os.environ.get("SEND_MODE", "draft")

AUTO_RESOLVE_CONFIDENCE_THRESHOLD = float(os.environ.get("AUTO_RESOLVE_CONFIDENCE_THRESHOLD", 0.85))
AUTO_RESOLVE_MAX_RISK_USD = float(os.environ.get("AUTO_RESOLVE_MAX_RISK_USD", 10000.0))

# Exception types that must never auto-resolve, regardless of confidence/revenue.
# Enforced in harness/escalation.py, not just an LLM instruction.
NEVER_AUTO_RESOLVE_TYPES = {"quality_rejection", "supplier_failure"}

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")

MOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CONTRACTS_DIR = os.path.join(MOCK_DATA_DIR, "contracts")
