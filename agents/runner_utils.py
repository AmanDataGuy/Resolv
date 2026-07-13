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
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import config

_session_service = InMemorySessionService()
_APP_NAME = "resolv"


def _is_rate_limit_error(error: Exception) -> bool:
    """Broad match, not a specific exception class: Gemini's SDK raises its own
    _ResourceExhaustedError, while Groq (through litellm) raises litellm.RateLimitError
    or a plain exception with "429"/"rate limit" in the message depending on version.
    Matching on the message is the only thing that reliably works across both.
    """
    message = str(error).lower()
    return "429" in message or "rate limit" in message or "resource_exhausted" in message


async def run_agent_once(agent, prompt_text: str, output_key: str | None = None) -> dict | str:
    """Runs `agent` once with `prompt_text` as the user message and returns its result.

    If the agent was built with output_schema/output_key (see schemas.py), pass the
    same output_key here and this returns the parsed dict from session state.
    Otherwise it returns the final response as plain text.

    On a rate-limit error with Groq configured, rotates to the next GROQ_API_KEY_*
    (see config.rotate_groq_key) and retries — once per configured key, then gives up.
    """
    attempts_left = max(config.groq_key_count(), 1)

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
            if attempts_left > 0 and _is_rate_limit_error(error) and config.rotate_groq_key():
                continue
            raise

    if output_key:
        updated_session = await _session_service.get_session(
            app_name=_APP_NAME, user_id="system", session_id=session.id
        )
        return updated_session.state.get(output_key, {})

    return final_text


def to_prompt(data: dict) -> str:
    """Serializes a dict to a JSON string agents can read as their input message."""
    return json.dumps(data, default=str)
