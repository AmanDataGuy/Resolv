"""
# resolv/agents/llm.py

**What:** the single choke point for LLM calls: `complete()` (sync) and `acomplete()` (async) over litellm, plus
`run_agent_once()` for the one ADK agent (the extractor).
**Why:** rate limits, retries, a provider circuit breaker, token metering and the `chat` trace span live in one
place. Hosted providers wait out a 429 (honouring "try again in Ns"); a daily quota raises `QuotaExhausted`;
self-hosted servers fail fast with no sleep. The token meter is context-local (a ContextVar), so concurrent
requests are each charged their own tokens. `RESOLV_FAKE_LLM=1` (load tests only) returns a canned, tool-free
reply after `FAKE_LLM_LATENCY_S`, so a capacity run measures Resolv rather than the provider's queue.
**Used by:** resolv/agents/loop.py, resolv/agents/extractor.py, evals/simulator.py, evals/quality.py, evals/deepeval_model.py, evals/*.
"""
import asyncio
import os
import random
import re
import time
import uuid
from collections import deque
from contextvars import ContextVar

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from resolv import config
from resolv import observability

_session_service = InMemorySessionService()
_APP_NAME = "resolv"

MAX_LLM_ATTEMPTS = 15
CALL_TIMEOUT_S = 30  # per acomplete() attempt; one stuck provider call can't hold a request forever

# Circuit breaker: this many consecutive non-429 provider errors within the window opens it for COOLDOWN_S.
BREAKER_FAILURES = 5
BREAKER_WINDOW_S = 60
BREAKER_COOLDOWN_S = 30
FAKE_CLAIM_JSON = '{"order_id": null, "claim_type": null, "stated_amount_usd": null}'  # RESOLV_FAKE_LLM extractor


# A provider's daily quota is spent; waiting minutes won't help. The sweep records it as error_kind="quota".
class QuotaExhausted(RuntimeError):
    pass


# The circuit breaker is open: fail fast without calling the provider.
class CircuitOpen(RuntimeError):
    pass


# Process-wide totals (prompt, completion) for evals/cost.py; the per-context meter is below.
_TOTAL = [0, 0]
# Fallback meter for callers that never call new_meter() (old delta-style callers keep working).
_PROCESS_METER = {"prompt": 0, "completion": 0, "served": None}
_meter: ContextVar[dict] = ContextVar("resolv_meter", default=_PROCESS_METER)

# ponytail: breaker state is per process; shared state (e.g. Redis) if this ever runs multi-instance.
_failures: deque = deque(maxlen=BREAKER_FAILURES)
_open_until = 0.0
_tripped = False
_now = time.monotonic  # swapped for a fake clock in tests


# Start a fresh token meter for the current context (call at the top of each request or eval run).
def new_meter() -> None:
    _meter.set({"prompt": 0, "completion": 0, "served": None})


# (prompt, completion) tokens metered in the current context.
def tokens_split() -> tuple[int, int]:
    m = _meter.get()
    return m["prompt"], m["completion"]


# Total prompt+completion tokens spent by this whole process.
def tokens_used() -> int:
    return _TOTAL[0] + _TOTAL[1]


# The model name the provider actually served on the last call in this context (provenance, not the alias).
def served_model() -> str | None:
    return _meter.get()["served"]


# Add one call's usage to the context meter and the process total.
def _meter_add(prompt: int, completion: int, served: str | None) -> None:
    m = _meter.get()
    m["prompt"] += prompt
    m["completion"] += completion
    if served:
        m["served"] = served
    _TOTAL[0] += prompt
    _TOTAL[1] += completion


# Message-text match, because Gemini's SDK, litellm and OpenRouter each raise different 429 types.
def _is_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(s in message for s in ("429", "rate limit", "resource_exhausted", "too many requests"))


# A per-day quota (Groq says "tokens per day (TPD)"); verify: other providers' daily-quota wording.
def _is_daily_quota(error: Exception) -> bool:
    message = str(error)
    return "per day" in message.lower() or "TPD" in message or "RPD" in message


# A refused or unreachable connection (the self-hosted server isn't running).
def _is_connection_error(error: Exception) -> bool:
    message = str(error).lower()
    return type(error).__name__ == "APIConnectionError" or "connection refused" in message or "connect" in message


# Seconds to wait, from the provider's own "try again in 1.665s" text when it gives one, else jittered.
def _retry_after(error: Exception) -> float:
    match = re.search(r"try again in ([\d.]+)(m?s)", str(error))
    if match:
        seconds = float(match.group(1))
        return seconds / 1000 if match.group(2) == "ms" else seconds
    return random.uniform(5, 15)


# Raise CircuitOpen while the breaker is open; after the cooldown one trial call goes through (half-open).
def _breaker_check() -> None:
    if _now() < _open_until:
        raise CircuitOpen("LLM provider circuit is open; failing fast.")


# Record one call outcome; open the breaker on a burst of failures, or at once if a half-open trial fails.
def _breaker_record(ok: bool) -> None:
    global _open_until, _tripped
    if ok:
        _failures.clear()
        _tripped = False
        return
    now = _now()
    _failures.append(now)
    burst = len(_failures) == BREAKER_FAILURES and now - _failures[0] <= BREAKER_WINDOW_S
    if burst or _tripped:
        _open_until = now + BREAKER_COOLDOWN_S
        _tripped = True
        _failures.clear()


# Decide what to do with a failed attempt: return seconds to wait, or raise.
def _on_error(error: Exception, attempt: int, model: str) -> float:
    if model.startswith("hosted_vllm/"):
        _breaker_record(False)
        if _is_connection_error(error):
            raise RuntimeError(
                f"Could not reach {model}: is the server running at SELFHOSTED_BASE_URL={config.SELFHOSTED_BASE_URL}?"
            ) from error
        raise error  # self-hosted 429/503 are not retried with sleep: the queue is ours, waiting won't drain it
    if _is_rate_limit_error(error):
        if _is_daily_quota(error):
            raise QuotaExhausted(str(error)) from error
        if attempt == MAX_LLM_ATTEMPTS - 1:
            raise error
        return _retry_after(error)
    _breaker_record(False)
    raise error


# Merge the per-model kwargs (self-hosted base URL) and the opt-in gpt-oss reasoning effort.
def _prepare(kwargs: dict) -> dict:
    model = str(kwargs.get("model", ""))
    kwargs = {**config.llm_kwargs(model), **kwargs}
    effort = os.environ.get("REASONING_EFFORT")
    if effort and "gpt-oss" in model and "reasoning_effort" not in kwargs:
        # litellm doesn't list Groq for this param yet, so it is forced through (BerriAI/litellm#14163).
        kwargs = {**kwargs, "reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]}
    return kwargs


# (prompt, completion, cached) tokens of one response; cached is None when the provider doesn't report it.
def usage_of(resp) -> tuple[int, int, int | None]:
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    return getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0, cached


# Usage, served model and cost of one response, put on the chat span and the meter.
def _on_success(resp, span, model: str) -> None:
    _breaker_record(True)
    prompt, completion, cached = usage_of(resp)
    served = getattr(resp, "model", None)
    _meter_add(prompt, completion, served)
    finish = [getattr(c, "finish_reason", None) for c in (getattr(resp, "choices", None) or [])]
    observability.set_chat_response(span, served, prompt, completion, cached,
                                    config.price_usd(model, prompt, completion), finish)


# Load-test mode: RESOLV_FAKE_LLM=1 swaps every model call for a canned reply, so a capacity run measures Resolv.
def _fake_on() -> bool:
    return os.environ.get("RESOLV_FAKE_LLM") == "1"


# Seconds a fake call sleeps, standing in for provider latency.
def _fake_latency() -> float:
    return float(os.environ.get("FAKE_LLM_LATENCY_S", "0"))


# The canned response: an empty claim for a JSON-schema call, a short text reply otherwise; never a tool call.
def _fake_response(kwargs: dict):
    from types import SimpleNamespace

    import litellm

    if "prompt" in kwargs and "messages" not in kwargs:  # legacy_line extractor reads choices[0].text
        return SimpleNamespace(model="fake", usage=None, choices=[SimpleNamespace(text="none", finish_reason="stop")])
    text = (FAKE_CLAIM_JSON if "response_format" in kwargs
            else "Thanks for reaching out. I've noted your complaint and a colleague will follow up shortly.")
    return litellm.ModelResponse(model="fake", choices=[
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}])


# Sync litellm.completion with rate limits survived; kept for the sync eval callers (simulator, judge).
def complete(trace_role: str | None = None, **kwargs):
    from litellm import completion

    if _fake_on():
        time.sleep(_fake_latency())
        return _fake_response(kwargs)
    kwargs = _prepare(kwargs)
    model = str(kwargs.get("model", ""))
    for attempt in range(MAX_LLM_ATTEMPTS):
        _breaker_check()
        with observability.chat_span(model, trace_role, kwargs.get("temperature")) as span:
            observability.set_content(span, "gen_ai.input.messages", kwargs.get("messages"))
            try:
                resp = completion(**kwargs)
            except Exception as error:
                wait = _on_error(error, attempt, model)
                observability.mark_error(span, error)
            else:
                _on_success(resp, span, model)
                return resp
        time.sleep(wait)


# Async twin of complete() for the serving path: no blocking I/O, asyncio.sleep between retries.
# A `prompt=` kwarg (no messages) uses the text-completions endpoint (the legacy_line extractor).
async def acomplete(trace_role: str | None = None, **kwargs):
    import litellm

    if _fake_on():
        await asyncio.sleep(_fake_latency())
        return _fake_response(kwargs)
    kwargs = _prepare(kwargs)
    kwargs.setdefault("timeout", CALL_TIMEOUT_S)
    model = str(kwargs.get("model", ""))
    call = litellm.atext_completion if "prompt" in kwargs and "messages" not in kwargs else litellm.acompletion
    for attempt in range(MAX_LLM_ATTEMPTS):
        _breaker_check()
        with observability.chat_span(model, trace_role, kwargs.get("temperature")) as span:
            observability.set_content(span, "gen_ai.input.messages", kwargs.get("messages") or kwargs.get("prompt"))
            try:
                resp = await call(**kwargs)
            except Exception as error:
                wait = _on_error(error, attempt, model)
                observability.mark_error(span, error)
            else:
                _on_success(resp, span, model)
                return resp
        await asyncio.sleep(wait)


# Run one ADK agent on one message in a fresh session; return the output_key dict from state, else the final text.
async def run_agent_once(agent, prompt_text: str, output_key: str | None = None) -> dict | str:
    if _fake_on():
        await asyncio.sleep(_fake_latency())
        return {"order_id": None, "claim_type": None} if output_key else ""
    model = str(getattr(agent.model, "model", agent.model))
    for attempt in range(MAX_LLM_ATTEMPTS):
        session_id = str(uuid.uuid4())
        await _session_service.create_session(app_name=_APP_NAME, user_id="system", session_id=session_id)
        runner = Runner(agent=agent, app_name=_APP_NAME, session_service=_session_service)
        message = types.Content(role="user", parts=[types.Part(text=prompt_text)])
        try:
            final_text = ""
            async for event in runner.run_async(user_id="system", session_id=session_id, new_message=message):
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = event.content.parts[0].text or ""
                    usage = getattr(event, "usage_metadata", None)  # counted on the final chunk only
                    if usage is not None:
                        _meter_add(getattr(usage, "prompt_token_count", 0) or 0,
                                   getattr(usage, "candidates_token_count", 0) or 0,
                                   getattr(event, "model_version", None))
            if not output_key:
                return final_text
            session = await _session_service.get_session(app_name=_APP_NAME, user_id="system", session_id=session_id)
            return session.state.get(output_key, {})
        except Exception as error:
            wait = _on_error(error, attempt, model)
        finally:
            # Sessions are in memory; without this every request leaks one.
            await _session_service.delete_session(app_name=_APP_NAME, user_id="system", session_id=session_id)
        await asyncio.sleep(wait)
