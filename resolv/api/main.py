"""
# resolv/api/main.py

**What:** the HTTP front door. POST /token exchanges an order's proof code for a session token; POST /resolve runs
one complaint end to end (agent -> routing -> notify) and returns the reply, ticket and audit trail. POST
/resolve/stream does the same as Server-Sent Events: one PII-masked event per agent step (see
resolv/agents/loop.py `public_view`), then a `result` event holding exactly the /resolve body. An optional
`Idempotency-Key` header on either stores the first response (24 h, ledger idempotency table) and replays it.
GET /metrics serves Prometheus metrics (resolv/metrics.py); it is open, see docs/RUNBOOK.md.
**Why:** no business logic lives here, so the API can never approve something the harness wouldn't. It only adds
transport concerns: rate limits, token checks, input caps, a wall-clock timeout, generic errors, and online
telemetry. The lifespan also runs the refund reconciler and the outbox drain every 30 s, and the app serves the
approvals console at /console. Run: `uvicorn resolv.api.main:app` (add `--proxy-headers` behind a proxy, see
resolv/api/ratelimit.py).
**Used by:** uvicorn, app.py (the stream), tests/test_auth.py, tests/test_async_health.py, tests/test_resolve_controls.py,
tests/test_stream_metrics.py.
"""
import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from resolv.agents.loop import FALLBACK_REPLY, RESOLVE_TIMEOUT_S, public_view, run_case
from resolv.agents.llm import CircuitOpen, new_meter, tokens_split
from resolv.api import auth, escalations, ratelimit
from resolv import metrics, monitor, observability
from resolv.harness import audit, ledger, routing, tools
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
            metrics.ERRORS.labels("background_job").inc()
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


# Prometheus scrape target. Open by design: no customer data in it; keep it off the public proxy (RUNBOOK).
@app.get("/metrics")
def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
# on_step, when given, receives each PII-masked step as it happens (the stream route).
async def _attempt(case_id: str, message: str, customer_id: str, on_step=None) -> dict:
    on_event = public_view(case_id, on_step) if on_step else None
    try:
        async with asyncio.timeout(RESOLVE_TIMEOUT_S):
            return await run_case(case_id, message, temperature=0.0, caller_id=customer_id, on_event=on_event)
    except TimeoutError:
        tools.escalate_to_human(case_id, "unknown", "budget_exceeded:wall_clock", customer_id)
        return {"reply": FALLBACK_REPLY, "claim": {}, "steps": None, "trail": audit.read(case_id),
                "budget_exceeded": "wall_clock"}


# Rate limit, then token: both checked before any LLM call. Returns the verified customer id.
def _gate(complaint: Complaint, request: Request) -> str:
    if not ratelimit.is_allowed(f"resolve:{ratelimit.client_ip(request)}", RESOLVE_PER_MINUTE):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again shortly.")
    customer_id = auth.verify_token(complaint.token)
    if customer_id is None:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    return customer_id


# Claim an Idempotency-Key for this request: the saved response on a replay, None when this request should run.
def _claim(key: str | None, customer_id: str, message: str) -> dict | None:
    if key is None:
        return None
    # The customer is part of the hash, so another customer's key is a conflict and never returns their response.
    request_sha = hashlib.sha256(json.dumps([customer_id, message]).encode()).hexdigest()
    state, saved = ledger.claim_key(key, "resolve", request_sha)
    if state == "conflict":
        raise HTTPException(422, "Idempotency-Key reused with a different request")
    if state == "in_flight":
        raise HTTPException(409, "A request with this Idempotency-Key is still running")
    return saved if state == "replay" else None


# Run the request and store its response under the key; any failure releases the key so the client can retry.
async def _resolve_keyed(message: str, customer_id: str, key: str | None, on_step=None) -> dict:
    try:
        response = await _resolve(message, customer_id, on_step)
    except BaseException:
        if key is not None:
            ledger.finish_key(key, None)
        raise
    if key is not None:
        ledger.finish_key(key, response)
    return response


# Resolve one complaint and return everything that happened.
# An optional Idempotency-Key makes a retried request return the first response instead of running a second case.
@app.post("/resolve")
async def resolve(complaint: Complaint, request: Request,
                  idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", min_length=1,
                                                       max_length=255)) -> dict:
    customer_id = _gate(complaint, request)
    saved = _claim(idempotency_key, customer_id, complaint.message)
    if saved is not None:
        return saved
    return await _resolve_keyed(complaint.message, customer_id, idempotency_key)


# One Server-Sent Event: the type as the event name, the whole event as JSON data.
def _sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


# /resolve as a live stream: same gate, same key handling, same body in the final `result` event.
# Errors after the stream has started arrive as an `error` event, since the status line is already sent.
@app.post("/resolve/stream")
async def resolve_stream(complaint: Complaint, request: Request,
                         idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", min_length=1,
                                                              max_length=255)) -> StreamingResponse:
    customer_id = _gate(complaint, request)
    saved = _claim(idempotency_key, customer_id, complaint.message)
    return StreamingResponse(_stream(complaint.message, customer_id, idempotency_key, saved),
                             media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# Run the case in a task that feeds a queue, and yield its steps as SSE frames as they arrive.
async def _stream(message: str, customer_id: str, key: str | None, saved: dict | None):
    if saved is not None:  # a replay: the steps already happened, only the stored result is sent
        yield _sse({"type": "result", "response": saved})
        return
    queue: asyncio.Queue = asyncio.Queue()

    # The run, with an end marker so the reader knows to stop waiting.
    async def run() -> dict:
        try:
            return await _resolve_keyed(message, customer_id, key, queue.put_nowait)
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(run())
    try:
        while (step := await queue.get()) is not None:
            yield _sse(step)
        try:
            yield _sse({"type": "result", "response": await task})
        except HTTPException as e:
            yield _sse({"type": "error", "status": e.status_code, "detail": e.detail})
        except Exception:
            logger.exception("stream resolve failed")
            yield _sse({"type": "error", "status": 500, "detail": UNAVAILABLE})
    finally:
        task.cancel()  # the client went away mid-run: stop it; _resolve_keyed releases the key


# Run the agent (with one retry), record telemetry, route and notify; the /resolve body for a verified customer.
async def _resolve(message: str, customer_id: str, on_step=None) -> dict:
    new_meter()  # this request's tokens only, even with other requests in flight
    started = time.perf_counter()
    result = None
    for _ in range(MAX_RESOLVE_ATTEMPTS):
        case_id = f"api-{uuid.uuid4().hex[:8]}"  # fresh per attempt, never from the caller
        try:
            result = await _attempt(case_id, message, customer_id, on_step)
            break
        except CircuitOpen:
            metrics.ERRORS.labels("circuit_open").inc()
            raise HTTPException(status_code=503, detail=UNAVAILABLE)
        except Exception:
            logger.exception("resolve attempt failed for case %s", case_id)
    if result is None:
        metrics.ERRORS.labels("resolve_failed").inc()
        raise HTTPException(status_code=503, detail=UNAVAILABLE)

    # Online eval: label-free metrics to data/telemetry and, when configured, as Langfuse scores. Never a gate.
    prompt_tokens, completion_tokens = tokens_split()
    telemetry = monitor.record(case_id, message, result, time.perf_counter() - started,
                               prompt_tokens, completion_tokens)
    observability.log_request(case_id, message, result, telemetry)

    ticket = routing.route(case_id, result["trail"], result["claim"])
    notify.send(ticket)
    metrics.RESOLVE_SECONDS.observe(time.perf_counter() - started)
    return {
        "case_id": case_id,
        "reply": result["reply"],
        "claim": result["claim"],
        "ticket": ticket.model_dump(),
        "trail": result["trail"],
    }
