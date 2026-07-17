"""Central configuration — model selection, provider keys, and business thresholds.

One place every other module reads from, so switching LLM provider or tuning an
escalation threshold is a config/.env change, never a code edit. Three concerns live
here:

  1. Model resolution — get_model() returns what an ADK agent's `model=` expects.
     Groq (via ADK's LiteLlm wrapper) is the default; Gemini is opt-in via USE_GEMINI=1
     and never selected just because a key is present. Provider choice is data, not a
     code branch at each call site. See get_model() for why the default is that way round.
  2. Groq key rotation — up to three GROQ_API_KEY_* values. litellm reads
     GROQ_API_KEY from the environment fresh on every request, so rotating which
     value sits in os.environ is enough to move the next call to a different key,
     with no agent rebuild. rotate_groq_key() is called after a rate-limit error
     in agents/runner_utils.py and returns False once all keys are exhausted.
  3. Policy thresholds — the exact numbers harness/policy.py enforces, kept here so
     they're env-overridable and visible in one spot rather than buried in the rules.
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
    has already been tried (caller should stop retrying and raise, or wait and
    reset_groq_key() to start the cycle again).
    """
    global _groq_key_index
    _groq_key_index += 1
    if _groq_key_index >= len(_GROQ_KEYS):
        return False
    os.environ["GROQ_API_KEY"] = _GROQ_KEYS[_groq_key_index]
    return True


def reset_groq_key() -> None:
    """Rewinds to the first key so rotate_groq_key() can cycle again.

    Exists because Groq's limit is tokens-PER-MINUTE, not a quota: exhausting every key means
    "all of them are busy right now", not "all of them are spent". A long eval sweep must be
    able to wait for the window to roll over and start the cycle again — without this, the first
    minute of rate limiting would permanently burn every key for the rest of the run.
    """
    global _groq_key_index
    _groq_key_index = 0
    if _GROQ_KEYS:
        os.environ["GROQ_API_KEY"] = _GROQ_KEYS[0]


def get_model():
    """Returns the model to pass into an ADK Agent/LlmAgent's `model=` argument.

    GROQ IS THE DEFAULT, AND GEMINI IS OPT-IN. This used to be the other way round — Gemini won
    whenever GEMINI_API_KEY was set. That's a bad default here for a blunt reason: a key sitting
    in .env is not consent to spend it. Every extractor call, every agent-loop step, and every
    eval run would have quietly billed Gemini just because the key existed, and nothing in the
    code would have said so.

    So the rule is inverted and made explicit: Groq unless someone deliberately sets
    USE_GEMINI=1. Preference is now a decision someone has to make, not a side effect of which
    keys happen to be configured.

    The volume argument also points this way. The eval runs each task n=5 times at temperature
    0.7 across ~40 tasks with several tool-calling steps each — thousands of calls per sweep.
    That belongs on Groq's free tier, not on a quota-limited paid key.

    ADK talks to Gemini natively (it's Google's own SDK), so that branch returns a plain
    model-name string and litellm is never imported. litellm only ever existed to reach Groq,
    which ADK has no native support for.
    """
    if os.environ.get("USE_GEMINI") == "1":
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("USE_GEMINI=1 but GEMINI_API_KEY is not set.")
        return GEMINI_MODEL_FAST
    if _GROQ_KEYS:
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(model=GROQ_MODEL)
    raise RuntimeError(
        "No GROQ_API_KEY set. Set one, or set USE_GEMINI=1 to use Gemini deliberately."
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
