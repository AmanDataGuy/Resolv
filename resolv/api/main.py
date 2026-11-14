"""
# resolv/api/main.py

**What:** the HTTP front door. POST /token exchanges an order's proof code for a session token; POST /resolve runs
one complaint end to end (agent -> routing -> notify) and returns the reply, ticket and audit trail.
**Why:** no business logic lives here, so the API can never approve something the harness wouldn't. It only adds
transport concerns: rate limits, token checks, input caps, a wall-clock timeout, generic errors, and online
telemetry. The lifespan also runs the refund reconciler and the outbox drain every 30 s, and the app serves the
approvals console at /console. Run: `uvicorn resolv.api.main:app` (add `--proxy-headers` behind a proxy, see
resolv/api/ratelimit.py).
**Used by:** uvicorn, tests/test_auth.py, tests/test_async_health.py.
"""
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from resolv.agents.loop import FALLBACK_REPLY, RESOLVE_TIMEOUT_S, run_case
from resolv.agents.llm import CircuitOpen, new_meter, tokens_split
from resolv.api import auth, escalations, ratelimit
from resolv import monitor, observability
from resolv.harness import audit, routing, tools
from resolv.integrations import notify

logger = logging.getLogger(__name__)

# One retry under a fresh case_id for failures past complete()'s own rate-limit handling; the failed trail is kept.
MAX_RESOLVE_ATTEMPTS = 2
RESOLVE_PER_MINUTE = 10
TOKEN_PER_MINUTE = 5  # with 10-char base32 proofs, 5 guesses/min makes brute force infeasible
UNAVAILABLE = "Could not resolve this complaint right now. Please try again."
BACKGROUND_EVERY_S = 30
WEB = Path(__file__).resolve().parents[2] / "web"


# Run one sync job (reconcile, outbox drain) every `every_s` seconds off the event loop until cancelled.
async def _every(job, every_s: float) -> None:
    while True:
        try:
            await asyncio.to_thread(job)
        except Exception:  # a failed pass is retried next tick; it must never kill the loop
            logger.exception("background job %s failed", job.__name__)
        await asyncio.sleep(every_s)


# Start tracing and the reconcile/outbox jobs on startup; cancel the jobs and flush spans/scores on shutdown.
@asynccontextmanager
async def lifespan(_app: FastAPI):
    observability.init()
    jobs = [asyncio.create_task(_every(tools.reconcile_once, BACKGROUND_EVERY_S)),
            asyncio.create_task(_every(notify.drain_outbox, BACKGROUND_EVERY_S))]
    yield
    for job in jobs:
        job.cancel()
    await asyncio.gather(*jobs, return_exceptions=True)
    observability.flush()


app = FastAPI(
    title="Resolv",
    description="A customer-refund agent that can only act inside a policy harness.",
    lifespan=lifespan,
)
app.include_router(escalations.router)
app.mount("/console", StaticFiles(directory=WEB / "console", html=True), name="console")


# /resolve body; 4000 chars is ~10x the dataset's median message.
class Complaint(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    token: str = Field(min_length=1, max_length=2048)


# /token body: the order id and the proof code printed on its receipt.
class TokenRequest(BaseModel):
    order_id: str = Field(pattern=r"^ORD-\d{3,6}$")
    proof: str = Field(min_length=1, max_length=32)


# Liveness, plus whether Langfuse tracing is on (false just means no keys are set).
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "observability": observability.enabled()}


# Exchange an order proof for a customer token. Unknown order and wrong proof give the same 401.
@app.post("/token")
def issue_token(req: TokenRequest, request: Request) -> dict:
    if not ratelimit.is_allowed(f"token:{ratelimit.client_ip(request)}", TOKEN_PER_MINUTE):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again shortly.")
    customer_id = auth.check_order_proof(req.order_id, req.proof)
    if customer_id is None:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    return {"token": auth.issue_token(customer_id)}


# Run one attempt under a wall-clock timeout; a timeout escalates the case instead of failing the request.
async def _attempt(case_id: str, message: str, customer_id: str) -> dict:
    try:
        async with asyncio.timeout(RESOLVE_TIMEOUT_S):
            return await run_case(case_id, message, temperature=0.0, caller_id=customer_id)
    except TimeoutError:
        tools.escalate_to_human(case_id, "unknown", "budget_exceeded:wall_clock", customer_id)
        return {"reply": FALLBACK_REPLY, "claim": {}, "steps": None, "trail": audit.read(case_id),
                "budget_exceeded": "wall_clock"}


# Resolve one complaint and return everything that happened. Rate limit and token are checked before any LLM call.
@app.post("/resolve")
async def resolve(complaint: Complaint, request: Request) -> dict:
    if not ratelimit.is_allowed(f"resolve:{ratelimit.client_ip(request)}", RESOLVE_PER_MINUTE):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again shortly.")
    customer_id = auth.verify_token(complaint.token)
    if customer_id is None:
        raise HTTPException(status_code=401, detail="Unauthorized.")

    new_meter()  # this request's tokens only, even with other requests in flight
    started = time.perf_counter()
    result = None
    for _ in range(MAX_RESOLVE_ATTEMPTS):
        case_id = f"api-{uuid.uuid4().hex[:8]}"  # fresh per attempt, never from the caller
        try:
            result = await _attempt(case_id, complaint.message, customer_id)
            break
        except CircuitOpen:
            raise HTTPException(status_code=503, detail=UNAVAILABLE)
        except Exception:
            logger.exception("resolve attempt failed for case %s", case_id)
    if result is None:
        raise HTTPException(status_code=503, detail=UNAVAILABLE)

    # Online eval: label-free metrics to data/telemetry and, when configured, as Langfuse scores. Never a gate.
    prompt_tokens, completion_tokens = tokens_split()
    telemetry = monitor.record(case_id, complaint.message, result, time.perf_counter() - started,
                               prompt_tokens, completion_tokens)
    observability.log_request(case_id, complaint.message, result, telemetry)

    ticket = routing.route(case_id, result["trail"], result["claim"])
    notify.send(ticket)
    return {
        "case_id": case_id,
        "reply": result["reply"],
        "claim": result["claim"],
        "ticket": ticket.model_dump(),
        "trail": result["trail"],
    }
