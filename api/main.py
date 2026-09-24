"""The web API for Resolv — one HTTP endpoint that runs a customer complaint end to end.

Run it locally with:

    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs to try it in the browser, or POST to /resolve.

WHAT THIS FILE IS. This is the thin "front door" of the system. It has NO business logic of its
own — it just wires four steps together in order:

    1. run_case()       the agent reads the complaint and calls tools   (agents/loop.py)
    2. routing.route()  turn the finished case into a ticket + team      (harness/routing.py)
    3. notify.send()    "email" the customer and page the team           (integrations/notify.py)
    4. return           hand the whole thing back as JSON

Every real decision (was the refund allowed? how much? which team?) is made and enforced in those
modules, not here. Keeping the endpoint dumb is the point: there is nothing here to get wrong, so
the API can never approve something the harness wouldn't.
"""
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from agents.loop import run_case
from agents.runner_utils import tokens_split
from api import ratelimit
from eval import monitor, observability
from harness import audit, routing
from integrations import notify

# complete() (agents/runner_utils.py) already survives rate limits on its own -- this is for
# the OTHER kind of failure: a malformed model response, a transient provider hiccup, anything
# that raises past that. One retry, same case_id, trail cleared first -- a failed attempt's
# partial trail must not count toward rule 4 (harness/policy.py) or double up in the audit log.
MAX_RESOLVE_ATTEMPTS = 2

# Create the app. The title/description show up on the auto-generated /docs page.
app = FastAPI(
    title="Resolv",
    description="A customer-refund agent that can only act inside a policy harness.",
)


# The shape of the request body. FastAPI reads the incoming JSON and checks it against this
# automatically — if "message" is missing, the caller gets a clear 422 error instead of a crash.
class Complaint(BaseModel):
    message: str
    # Who is asking. A real deployment would attach this from an authenticated session rather
    # than trust a request field — there is still no auth layer here (see api/ratelimit.py's own
    # docstring) — but the harness-side half of that gap is closed: harness/policy.py's rule 2
    # denies any refund whose caller_id doesn't match the order's customer_id, and requiring the
    # field here is what makes that check reachable instead of permanently None.
    customer_id: str


@app.get("/health")
def health() -> dict:
    """A tiny endpoint to check the server is alive — handy for deploys and uptime checks.

    Reports whether Langfuse tracing is live, so a deploy can confirm observability came up without
    reading logs. `false` is normal — it just means no Langfuse keys are set; the local telemetry
    log records every request regardless.
    """
    return {"status": "ok", "observability": observability.enabled()}


@app.post("/resolve")
async def resolve(complaint: Complaint, request: Request) -> dict:
    """Resolve one complaint and return everything that happened.

    It's `async def` because run_case() is asynchronous (the agent awaits the language model).
    FastAPI handles the await for us, so a caller just POSTs and waits for the JSON back.
    """
    # Rate-limited BEFORE anything expensive happens -- this is the one thing standing between an
    # unauthenticated caller and unlimited real LLM calls on our bill (see api/ratelimit.py). Keyed
    # on the caller's IP; `request.client` is None under some test/proxy setups, hence the guard.
    client_id = request.client.host if request.client else "unknown"
    if not ratelimit.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again shortly.")

    # A unique id for this case. It names the audit trail this run reads and writes, so two
    # customers never share history. Generated here, never taken from the caller.
    case_id = f"api-{uuid.uuid4().hex[:8]}"

    # Step 1 — run the agent. temperature=0.0 makes it deterministic: the same complaint gives the
    # same answer. (The eval uses 0.7 to get varied samples; a live customer wants the single most
    # reliable answer.)
    # Wrapped in a clock + token counter so ONLINE EVAL can grade this real request: latency and
    # cost come from the delta across the call, the label-free quality signals from the trail.
    started = time.perf_counter()
    tokens_before = tokens_split()
    last_error: Exception | None = None
    result = None
    for attempt in range(MAX_RESOLVE_ATTEMPTS):
        try:
            result = await run_case(
                case_id, complaint.message, temperature=0.0, caller_id=complaint.customer_id
            )
            break
        except Exception as e:
            last_error = e
            if attempt < MAX_RESOLVE_ATTEMPTS - 1:
                audit.clear(case_id)  # a failed attempt's partial trail must not carry over
    if result is None:
        raise HTTPException(status_code=503, detail=f"Could not resolve this complaint: {last_error}")
    latency_s = time.perf_counter() - started
    prompt_tokens, completion_tokens = (a - b for a, b in zip(tokens_split(), tokens_before))

    # Online eval — record the label-free metrics for this live request (data/telemetry/live.jsonl)
    # and mirror them to Langfuse if it's configured. Both are best-effort observers of the request,
    # never gates on it: monitor.record only reads the finished result, and the Langfuse layer
    # swallows its own errors, so neither can change or block what the customer gets back.
    telemetry = monitor.record(case_id, complaint.message, result,
                               latency_s, prompt_tokens, completion_tokens)
    observability.log_request(case_id, complaint.message, result, telemetry)

    # Step 2 — read the finished audit trail and decide the ticket (refund / escalate / deny) and,
    # if escalated, which team. Pure and deterministic — no second call to the model.
    ticket = routing.route(case_id, result["trail"], result["claim"])

    # Step 3 — deliver it: write the customer email and, if escalated, page the team.
    # (Mocked to files under data/outbox/ — see integrations/notify.py.)
    notify.send(ticket)

    # Step 4 — return everything. `trail` is included on purpose: it's the evidence for what the
    # agent actually did, so a caller can trust the record instead of the prose reply.
    return {
        "case_id": case_id,
        "reply": result["reply"],       # what to say to the customer, in plain language
        "claim": result["claim"],       # what intake read the message as (order id + claim type)
        "ticket": ticket.model_dump(),  # outcome, team, ticket id, customer message
        "trail": result["trail"],       # every tool call and the policy verdict on each
    }


def _selfcheck() -> None:
    """One no-network check: the app builds and /health answers. The /resolve path needs a live
    model, so it's exercised by the demo and the eval, not here. Run: python -m api.main
    """
    from fastapi.testclient import TestClient

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "observability" in r.json(), "health should report whether Langfuse tracing is live"
    print("api selfcheck OK — /health responds")


if __name__ == "__main__":
    _selfcheck()
