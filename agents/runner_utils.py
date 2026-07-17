"""Small helper for invoking one ADK agent and getting its structured result back.

Real ADK usage (confirmed against github.com/mayank953/Youtube/tree/main/ADK) always
goes through a Runner + SessionService + a Content message, then reads the final
event's text. That's ~15 lines of boilerplate every time you call an agent. Since
Resolv calls six different agents back-to-back in a fixed pipeline (see
agents/orchestrator.py), this file wraps that boilerplate once.

Each call creates a *fresh* in-memory session. Agents in this pipeline are single-shot
(classify this one event, assess this one impact) — they don't need multi-turn
conversation memory between pipeline stages. The stages pass data to each other as
plain Python dicts, not via shared ADK session state.
"""
import json
import random
import re
import time
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import config

_session_service = InMemorySessionService()
_APP_NAME = "resolv"

MAX_LLM_ATTEMPTS = 8


def _is_rate_limit_error(error: Exception) -> bool:
    """Broad match, not a specific exception class: Gemini's SDK raises its own
    _ResourceExhaustedError, while Groq (through litellm) raises litellm.RateLimitError
    or a plain exception with "429"/"rate limit" in the message depending on version.
    Matching on the message is the only thing that reliably works across both.
    """
    message = str(error).lower()
    return "429" in message or "rate limit" in message or "resource_exhausted" in message


def _retry_after(error: Exception) -> float:
    """Seconds to wait, taken from Groq's own error text when it offers one.

    Groq says exactly how long to wait ("Please try again in 1.665s") and that beats any backoff
    curve we could invent — it knows when the token window rolls over and we don't. Blind
    exponential backoff either sleeps too long (wasting a sweep) or too short (burning attempts
    on a limit that hasn't lifted). The jittered fallback is only for when it doesn't say.
    """
    match = re.search(r"try again in ([\d.]+)(m?s)", str(error))
    if match:
        seconds = float(match.group(1))
        return seconds / 1000 if match.group(2) == "ms" else seconds
    return random.uniform(5, 15)


def complete(**kwargs):
    """litellm.completion() with Groq's tokens-per-minute limit survived rather than raised.

    THE LIMIT IS THE BINDING CONSTRAINT ON THIS PROJECT, not a rare edge case. The free tier
    allows 12,000 tokens/minute per org, and one tool-calling turn with three schemas and a
    conversation is ~3k. A 40-task x 5-repeat sweep is thousands of calls: without this, the
    first eval attempt recorded rate-limit errors for all 10 smoke runs and scored the agent 0.0
    across the board — a number about Groq's billing, not about the agent.

    Two lines of defence, in this order:
      1. ROTATE to the next key. Instant, and the keys sit in different orgs, so a fresh key has
         a fresh 12k window. Always try this first — it costs nothing.
      2. WAIT, then rewind and start the cycle again. Only once every key is busy. This is why
         config.reset_groq_key() exists: the limit is per-minute, so "all keys exhausted" means
         "all busy right now", never "all spent". Treating it as terminal would end the sweep on
         the first crowded minute.

    Retries only on rate limits. A malformed request retried eight times is eight identical
    failures and a slower error message.
    """
    from litellm import completion

    for attempt in range(MAX_LLM_ATTEMPTS):
        try:
            return completion(**kwargs)
        except Exception as error:
            if not _is_rate_limit_error(error) or attempt == MAX_LLM_ATTEMPTS - 1:
                raise
            if not config.rotate_groq_key():
                time.sleep(_retry_after(error))
                config.reset_groq_key()


async def run_agent_once(agent, prompt_text: str, output_key: str | None = None) -> dict | str:
    """Runs `agent` once with `prompt_text` as the user message and returns its result.

    If the agent was built with output_schema/output_key (see schemas.py), pass the
    same output_key here and this returns the parsed dict from session state.
    Otherwise it returns the final response as plain text.

    On a rate-limit error, rotates to the next GROQ_API_KEY_* and retries; once every key is
    busy it waits for the token window to roll over and cycles again, same as complete(). This
    used to give up after one pass through the keys, which is wrong for a per-minute limit —
    during an eval sweep that's a normal Tuesday, not a terminal condition.
    """
    attempts_left = MAX_LLM_ATTEMPTS

    while True:
        session_id = str(uuid.uuid4())
        session = await _session_service.create_session(
            app_name=_APP_NAME, user_id="system", session_id=session_id
        )

        message = types.Content(role="user", parts=[types.Part(text=prompt_text)])
        runner = Runner(agent=agent, app_name=_APP_NAME, session_service=_session_service)

        try:
            final_text = ""
            async for event in runner.run_async(
                user_id="system", session_id=session.id, new_message=message
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = event.content.parts[0].text or ""
            break
        except Exception as error:
            attempts_left -= 1
            if attempts_left <= 0 or not _is_rate_limit_error(error):
                raise
            if not config.rotate_groq_key():
                time.sleep(_retry_after(error))
                config.reset_groq_key()

    if output_key:
        updated_session = await _session_service.get_session(
            app_name=_APP_NAME, user_id="system", session_id=session.id
        )
        return updated_session.state.get(output_key, {})

    return final_text


def to_prompt(data: dict) -> str:
    """Serializes a dict to a JSON string agents can read as their input message."""
    return json.dumps(data, default=str)
