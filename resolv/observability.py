"""
# resolv/observability.py

**What:** tracing for the serving path: OTel GenAI spans (`invoke_agent` > `chat` / `execute_tool`) exported to
Langfuse (Python SDK 4.x), plus Langfuse scores on each trace.
**Why:** our code only speaks the vendor-neutral OpenTelemetry API (schema: docs/observability.md);
Langfuse is just the exporter. With no keys nothing is installed, every span is a no-op and nothing touches
the network. Message content is off by default and always PII-masked when on. Observability must never break
the request it watches, so every Langfuse call is wrapped.
**Used by:** resolv/agents/loop.py, resolv/agents/llm.py, resolv/api/main.py, app.py.
"""
import hashlib
import json
import logging
import os
from contextlib import contextmanager
from functools import lru_cache

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from resolv import config
from resolv.harness import pii

logger = logging.getLogger(__name__)

# ponytail: one module-level tracer on the global provider; per-tenant providers only if ever needed.
tracer = trace.get_tracer("resolv")

# role -> (prompt name, prompt text), registered by the modules that own the prompts.
_PROMPTS: dict[str, tuple[str, str]] = {}
# prompt name -> Langfuse prompt version, filled by init() when keys are set.
_PROMPT_VERSIONS: dict[str, int] = {}


# First 12 hex chars of sha256(text): a stable short id for prompts and callers.
def short_hash(text: str | None) -> str | None:
    return hashlib.sha256(text.encode()).hexdigest()[:12] if text else None


# True when message content may be put on spans (RESOLV_TRACE_CONTENT=1); it is masked even then.
def content_enabled() -> bool:
    return os.environ.get("RESOLV_TRACE_CONTENT") == "1"


# The Langfuse client, or None when keys are missing, eval tracing is off, or the SDK fails. One per process.
@lru_cache(maxsize=1)
def _client():
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    if config.RESOLV_ENV == "eval" and os.environ.get("RESOLV_TRACE_EVAL", "0") != "1":
        return None  # sweeps are untraced unless asked for
    try:
        from langfuse import Langfuse

        # Constructing the client registers its span processor on the global TracerProvider, so the
        # tracer above starts exporting. Host comes from LANGFUSE_HOST.
        # verify: langfuse 4.x Langfuse(mask=, environment=, release=) kwargs and the mask(*, data) signature.
        return Langfuse(
            mask=lambda *, data, **_: pii.mask_any(data),
            environment=config.RESOLV_ENV,
            release=os.environ.get("GIT_SHA"),
        )
    except Exception:
        logger.warning("Langfuse client failed to start; tracing off", exc_info=True)
        return None


# Start tracing if keys are set; also push registered prompts as Langfuse prompt versions. Returns enabled().
def init() -> bool:
    client = _client()
    if client is None:
        return False
    for name, text in _PROMPTS.values():
        try:
            # verify: langfuse 4.x get_prompt(name, label=) / create_prompt(name=, prompt=, labels=, type=) and .version.
            try:
                current = client.get_prompt(name, label="production")
            except Exception:
                current = None
            if current is None or current.prompt != text:
                current = client.create_prompt(name=name, prompt=text, labels=["production"], type="text")
            _PROMPT_VERSIONS[name] = current.version
        except Exception:
            logger.warning("prompt sync failed for %s", name, exc_info=True)
    return True


# True only when a Langfuse client is live; reported by /health.
def enabled() -> bool:
    return _client() is not None


# Push buffered spans and scores; called on API shutdown and at the end of a demo run.
def flush() -> None:
    client = _client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass


# Remember which prompt a role's chat spans should link to (called at import by the prompt's owner).
def register_prompt(role: str, name: str, text: str) -> None:
    _PROMPTS[role] = (name, text)


# Put a masked content attribute on a span, only when content capture is on.
def set_content(span, key: str, value) -> None:
    if content_enabled():
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        span.set_attribute(key, pii.mask(text))


# The root span of one resolution. Not made current here: the caller attaches it around each await.
def start_agent_span(case_id: str, caller_id: str | None):
    attrs = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "resolv",
        "gen_ai.conversation.id": case_id,
        # verify: Langfuse OTel trace-level attribute names (session, user, environment, release, tags).
        "langfuse.session.id": case_id,
        "langfuse.environment": config.RESOLV_ENV,
        "langfuse.trace.tags": [f"prompt:{short_hash(_PROMPTS.get('agent', ('', ''))[1])}"],
    }
    caller = short_hash(caller_id)
    if caller:
        attrs["resolv.caller.hash"] = caller
        attrs["langfuse.user.id"] = caller
    if os.environ.get("GIT_SHA"):
        attrs["langfuse.release"] = os.environ["GIT_SHA"]
    return tracer.start_span("invoke_agent resolv", attributes=attrs)


# Hex trace id of a span, or None for a no-op span (tracing off).
def trace_id_of(span) -> str | None:
    ctx = span.get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


# One `chat <model>` CLIENT span around a single LLM call attempt.
@contextmanager
def chat_span(model: str, role: str | None, temperature=None):
    provider = "selfhosted" if model.startswith("hosted_vllm/") else model.split("/", 1)[0]
    attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": provider,
        "gen_ai.request.model": model,
        # verify: Langfuse maps langfuse.observation.* attributes onto a generation observation.
        "langfuse.observation.type": "generation",
        "langfuse.observation.model.name": model,
    }
    if temperature is not None:
        attrs["gen_ai.request.temperature"] = float(temperature)
    if role:
        attrs["resolv.role"] = role
        name, text = _PROMPTS.get(role, (None, None))
        if name:
            attrs["resolv.prompt.sha"] = short_hash(text)
            attrs["langfuse.observation.prompt.name"] = name
            if name in _PROMPT_VERSIONS:
                attrs["langfuse.observation.prompt.version"] = _PROMPT_VERSIONS[name]
    with tracer.start_as_current_span(f"chat {model}", kind=SpanKind.CLIENT, attributes=attrs) as span:
        yield span


# Response attributes on a chat span: served model, usage, cached tokens, cost, finish reasons.
def set_chat_response(span, served: str | None, prompt: int, completion: int, cached: int | None,
                      usd: float, finish_reasons: list[str]) -> None:
    span.set_attribute("gen_ai.usage.input_tokens", prompt)
    span.set_attribute("gen_ai.usage.output_tokens", completion)
    span.set_attribute("gen_ai.response.finish_reasons", [r for r in finish_reasons if r])
    span.set_attribute("resolv.usd", usd)
    span.set_attribute("langfuse.observation.usage_details", json.dumps({"input": prompt, "output": completion}))
    span.set_attribute("langfuse.observation.cost_details", json.dumps({"total": usd}))
    if served:
        span.set_attribute("gen_ai.response.model", served)
    if cached is not None:
        span.set_attribute("resolv.cached_input_tokens", cached)


# Mark a span failed without raising (used for retried errors that the caller swallows).
def mark_error(span, error: Exception) -> None:
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, type(error).__name__))


# One `execute_tool <name>` span around a tool call; the caller adds the policy verdict after.
@contextmanager
def tool_span(name: str, call_id: str | None):
    attrs = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name}
    if call_id:
        attrs["gen_ai.tool.call.id"] = call_id
    with tracer.start_as_current_span(f"execute_tool {name}", attributes=attrs) as span:
        yield span


# Copy an audit record's verdict onto the tool span.
def set_tool_record(span, record: dict | None) -> None:
    if not record:
        return
    span.set_attribute("resolv.policy.action", record.get("action") or "")
    span.set_attribute("resolv.policy.rule_id", record.get("rule_id") or "")
    span.set_attribute("resolv.audit.record_id", record.get("record_id") or "")
    span.set_attribute("resolv.order.id", record.get("order_id") or "")
    for key in ("seq", "mac"):
        if record.get(key) is not None:
            span.set_attribute(f"resolv.audit.{key}", record[key])


# Send the label-free online metrics (evals/monitor.record) as Langfuse scores on the request's trace.
def log_request(case_id: str, message: str, result: dict, metrics: dict) -> None:
    client = _client()
    trace_id = (result or {}).get("trace_id")
    if client is None or not trace_id:
        return
    try:
        # verify: langfuse 4.x create_score(name=, value=, trace_id=, data_type=) signature.
        for name in ("unauthorized", "reply_grounded", "looked_up_first"):
            if metrics.get(name) is not None:
                client.create_score(name=name, value=float(bool(metrics[name])), trace_id=trace_id,
                                    data_type="BOOLEAN")
        if result.get("steps") is not None:
            client.create_score(name="steps", value=float(result["steps"]), trace_id=trace_id, data_type="NUMERIC")
    except Exception:
        pass  # observability is never allowed to break the endpoint it observes
