"""The HTTP surface — one endpoint that runs a complaint end to end.

    uvicorn api.main:app --reload      then POST /resolve  {"message": "..."}

WHAT IT WIRES. This is the production path the demo and any real caller share: take a raw
customer message, run the agent loop (agents/loop.py), route the finished trail to a ticket
(harness/routing.py), send it (integrations/notify.py), and return everything that happened. The
endpoint itself holds no business logic — every decision was made and enforced downstream, and
this just orchestrates the four steps and reports them.

TEMPERATURE 0 HERE, unlike the eval. The eval samples at 0.7 because pass^k needs independent
attempts; a real customer wants the most reliable single answer, so the API defaults to
deterministic. Same code, different knob — which is the whole reason temperature is a parameter
on run_case rather than a constant.

OBSERVABILITY. Each /resolve is wrapped in an OpenTelemetry span carrying the outcome, team, and
step count — the fields you actually page on. OTel is OPTIONAL: if the packages aren't installed
the tracer degrades to a no-op and the endpoint still works. (ponytail: console exporter, no
collector — a real deployment points OTEL_EXPORTER_OTLP_ENDPOINT at one and this code doesn't
change.)
"""
import uuid
from contextlib import contextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from agents.loop import run_case
from harness import routing
from integrations import notify

# --- Optional OpenTelemetry ----------------------------------------------------------------
# Wired if present, no-op if not. A demo shouldn't hard-depend on a tracing stack, and a real
# deployment shouldn't have to strip one out — so it's detected, not required.
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    _provider = TracerProvider()
    _provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(_provider)
    _tracer = trace.get_tracer("resolv")
    _OTEL = True
except Exception:  # ImportError, or any SDK init failure — never let telemetry break serving
    _tracer = None
    _OTEL = False


@contextmanager
def _span(name: str, **attrs):
    """Start a span if OTel is available, else do nothing. Keeps the handler free of `if _OTEL`."""
    if not _tracer:
        yield None
        return
    with _tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, v)
        yield span


app = FastAPI(title="Resolv", description="Deduction-recovery support agent with a policy harness.")


class ComplaintIn(BaseModel):
    message: str
    case_id: str | None = None  # caller may supply one; otherwise we mint a fresh trail id


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "otel": _OTEL}


@app.post("/resolve")
async def resolve(body: ComplaintIn) -> dict:
    """Run one complaint: agent loop -> ticket -> notify. Returns reply, ticket, and the trail.

    The trail is returned in full on purpose — it's the evidence for what the agent did, and a
    caller that wants to trust the reply can check it against the record instead of the prose.
    """
    case_id = body.case_id or f"api-{uuid.uuid4().hex[:12]}"
    with _span("resolve", case_id=case_id) as span:
        result = await run_case(case_id, body.message, temperature=0.0)
        ticket = routing.route(case_id, result["trail"], result["claim"])
        sent = notify.send(ticket)
        if span:
            span.set_attribute("outcome", ticket.outcome)
            span.set_attribute("steps", result["steps"])
            if ticket.team:
                span.set_attribute("team", ticket.team)

    return {
        "case_id": case_id,
        "reply": result["reply"],
        "claim": result["claim"],
        "ticket": ticket.model_dump(),
        "delivered": sent,
        "trail": result["trail"],
    }


def _selfcheck() -> None:
    """No-network check: the app builds and /health answers. The /resolve path needs a live model,
    so it's exercised by the eval and the demo, not here.
    """
    from fastapi.testclient import TestClient

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    print(f"api selfcheck OK — /health responds, otel={_OTEL}")


if __name__ == "__main__":
    _selfcheck()
